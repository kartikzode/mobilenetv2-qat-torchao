"""mobilenetv2_qat_torchao.py: QAT of torchvision's MobileNetV2 on ImageNet with torchao (PT2E) and ExecuTorch's XNNPACK, then
lowering to ExecuTorch and evaluation, all in one run.

Takes the hyperparameters of torchvision's references/classification QAT recipe for MobileNetV2 (train_quantization.py),
minus distributed training, PTQ, checkpointing and resume. Steps:

  1. QAT on the GPU, starting from torchvision MobileNet_V2_Weights.IMAGENET1K_V1
  2. convert to int8 with convert_pt2e (on the CPU)
  3. lower the float model and the int8 model to ExecuTorch .pte files with the XNNPACK delegate (batch size 1)
  4. evaluate on the ImageNet validation set (or a random subset, --max-val-images), on the CPU:
       * the two .pte programs (real XNNPACK kernels)
       * the PyTorch float graph and the converted int8 graph (simulated in float), to see where accuracy is lost:
         converted graph -> .pte
  5. time the two .pte programs at batch 1

Everything is written to <out-dir>/run-<timestamp>/ : qat_state.pt (trained state, for lower_mobilenetv2.py),
mobilenetv2_fp32.pte, mobilenetv2_int8.pte.

Data: an ImageFolder layout, <data-dir>/train/<class>/*.JPEG and <data-dir>/val/<class>/*.JPEG (the layout the reference uses).

Differences from the reference recipe:
  * The reference ran 8 GPUs x batch 32 = 256 images per step. Here --batch-size defaults to 256 on one GPU.
  * --epochs defaults to 10, not 90: the README says training converges at about 10 epochs. LR step size stays 30,
    so below 30 epochs the learning rate is constant at 1e-4.
  * The quantizer is ExecuTorch's XNNPACKQuantizer (torch 2.15 no longer ships one), per-tensor weights unless --per-channel.

Latency here is measured on whatever CPU runs this script, so it says little about the ARM target.

Needs torch 2.15.0, torchao 0.18.0, executorch 1.6.0 (pip install executorch==1.6.0; it pins torch 2.15.0).

  python mobilenetv2_qat_torchao.py --data-dir /data/imagenet                       # 10 epochs, global batch 256
  python mobilenetv2_qat_torchao.py --data-dir /data/imagenet --epochs 90           # the reference budget
"""
import argparse, collections, copy, io, os, time

import numpy as np
import torch, torch.nn.functional as F, torchvision
from torch.export import Dim, export
from torchvision import transforms as T
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms.functional import InterpolationMode

import torchao.quantization.pt2e as pt2e
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

try:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime
except ImportError as e:
    raise SystemExit(f"ExecuTorch is needed for the quantizer, the lowering and the runtime ({e}). Install it with: pip install executorch==1.6.0")

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True, help="folder with train/ and val/ in ImageFolder layout")
p.add_argument("--out-dir", default="runs", help="a run-<timestamp> folder with the saved models is created inside")
p.add_argument("--workers", type=int, default=16)                                  # reference: 16
p.add_argument("--batch-size", type=int, default=256)                              # reference: 32 per GPU x 8 GPUs
p.add_argument("--eval-batch-size", type=int, default=128)                         # reference: 128
p.add_argument("--epochs", type=int, default=10)                                   # reference: 90 (converges at ~10)
p.add_argument("--lr", type=float, default=1e-4)                                   # reference: 0.0001
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--lr-step-size", type=int, default=30)
p.add_argument("--lr-gamma", type=float, default=0.1)
p.add_argument("--num-observer-update-epochs", type=int, default=4)
p.add_argument("--num-batch-norm-update-epochs", type=int, default=3)
p.add_argument("--per-channel", action="store_true", help="per-channel weight quantization (default is per-tensor)")
p.add_argument("--max-train-images", type=int, default=0, help="random subset of the training set (0 = all)")
p.add_argument("--max-val-images", type=int, default=0, help="evaluate on a seeded random subset of the validation set (0 = all 50000)")
p.add_argument("--latency-images", type=int, default=300, help="images used to time the .pte programs at batch 1")
p.add_argument("--print-freq", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
DYN = ({0: Dim("batch", min=1, max=4096)},)        # dynamic batch: train batch, eval batch and the last short batch all work
run_dir = os.path.join(args.out_dir, time.strftime("run-%Y%m%d-%H%M%S"))
os.makedirs(run_dir, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data (same transforms as the reference presets)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BILINEAR = InterpolationMode.BILINEAR
train_tf = T.Compose([T.RandomResizedCrop(224, interpolation=BILINEAR, antialias=True), T.RandomHorizontalFlip(0.5),
                      T.PILToTensor(), T.ConvertImageDtype(torch.float), T.Normalize(MEAN, STD)])
val_tf = T.Compose([T.Resize(256, interpolation=BILINEAR, antialias=True), T.CenterCrop(224),
                    T.PILToTensor(), T.ConvertImageDtype(torch.float), T.Normalize(MEAN, STD)])


def make_loader(split, tf, batch, train, limit):
    d = os.path.join(args.data_dir, split)
    if not os.path.isdir(d):
        raise SystemExit(f"{d} not found. Expected {args.data_dir}/train/<class>/*.JPEG and {args.data_dir}/val/<class>/*.JPEG")
    ds = torchvision.datasets.ImageFolder(d, tf)
    if limit and limit < len(ds):                                    # seeded random subset (ImageFolder is sorted by class)
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:limit].tolist())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=train, num_workers=args.workers, pin_memory=DEV.type == "cuda",
                                       persistent_workers=train and args.workers > 0)
    return loader, len(ds)


train_loader, n_train = make_loader("train", train_tf, args.batch_size, True, args.max_train_images)
val_loader, n_val = make_loader("val", val_tf, args.eval_batch_size, False, args.max_val_images)
log(f"device {DEV}, torch {torch.__version__}, train {n_train} images ({len(train_loader)} steps/epoch), val {n_val} images, saving to {run_dir}")


# ---------------------------------------------------------------- models
def float_model(pretrained):
    return torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)


def prepare_qat(model, example):
    """export (train mode) -> insert fake quantization for the XNNPACK int8 recipe. strict=True is required for torchao 0.18."""
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=args.per_channel, is_qat=True))
    return prepare_qat_pt2e(export(model.train(), (example,), dynamic_shapes=DYN, strict=True).module(), quantizer)



def fp32_size_mb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.getbuffer().nbytes / 1e6


def int8_size_mb_estimate(converted):
    """convert_pt2e keeps fp32 weights beside the int8 ones, so its state_dict is not the deployed size. Estimate the lowered
    model: int8 conv/linear weights plus a fp32 scale and a 32-bit bias per output channel (BatchNorm is folded away)."""
    return sum(v.numel() + v.shape[0] * 8 for v in converted.state_dict().values() if v.dtype == torch.int8 and v.ndim >= 2) / 1e6


# ---------------------------------------------------------------- 1. QAT (GPU)
log(f"QAT: float MobileNetV2 (ImageNet1K_V1) -> fake-quant graph, per-{'channel' if args.per_channel else 'tensor'} weights")
fp32 = float_model(True).eval()
qat_gpu = prepare_qat(copy.deepcopy(fp32).to(DEV), torch.randn(8, 3, 224, 224, device=DEV))
opt = torch.optim.SGD(qat_gpu.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

t0 = time.time()
for epoch in range(args.epochs):
    seen, loss_sum, correct, t_win = 0, 0.0, 0, time.time()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        out = qat_gpu(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        seen += len(y); loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item()
        if i % args.print_freq == 0:
            log(f"epoch {epoch} [{i}/{len(train_loader)}] loss {loss_sum / seen:.4f} train top-1 {100 * correct / seen:.2f} | {seen / (time.time() - t_win):.0f} img/s")
    sched.step()
    # same checks, in the same place, as the reference: after the epoch, once its index reaches the threshold
    if epoch >= args.num_observer_update_epochs:
        qat_gpu.apply(pt2e.disable_observer)                         # freeze activation/weight ranges
        if epoch == args.num_observer_update_epochs: log(f"  observers frozen after epoch {epoch}")
    if epoch >= args.num_batch_norm_update_epochs:
        pt2e.move_exported_model_to_eval(qat_gpu)                    # BatchNorm now uses its running statistics (dropout is off too)
        if epoch == args.num_batch_norm_update_epochs: log(f"  BatchNorm stats frozen after epoch {epoch}")
    log(f"epoch {epoch} done | loss {loss_sum / seen:.4f} | lr {opt.param_groups[0]['lr']:g} | {(time.time() - t0) / 60:.1f} min")
train_min = (time.time() - t0) / 60
del train_loader                                                     # shuts down the persistent worker processes before anything is timed

# ---------------------------------------------------------------- 2. GPU fake-quant graph -> CPU int8 graph
log("converting to int8 on CPU")
torch.set_grad_enabled(False)
try:                                                                 # re-export on CPU and load the trained state (weights, ranges)
    qat_cpu = prepare_qat(float_model(False), torch.randn(8, 3, 224, 224))
    qat_cpu.load_state_dict({k: v.cpu() for k, v in qat_gpu.state_dict().items()}, strict=True)
except Exception as e:
    log(f"  state-dict transfer failed ({type(e).__name__}: {str(e)[:120]}); moving the graph with .cpu() instead")
    qat_cpu = qat_gpu.cpu()
torch.save({"state_dict": qat_cpu.state_dict(), "per_channel": args.per_channel}, os.path.join(run_dir, "qat_state.pt"))
log(f"saved trained QAT state to {run_dir}/qat_state.pt")
qat_int8 = convert_pt2e(qat_cpu)
pt2e.move_exported_model_to_eval(qat_int8)                           # never call .eval() on an exported graph

# ---------------------------------------------------------------- 3. lower both models to ExecuTorch (XNNPACK delegate, batch 1)
EX1 = torch.randn(1, 3, 224, 224)


def lower(exported, name):
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()
    path = os.path.join(run_dir, name)
    with open(path, "wb") as f:
        f.write(program.buffer)
    nodes = list(edge.exported_program().graph_module.graph.nodes)
    delegates = sum(1 for n in nodes if "executorch_call_delegate" in str(n.target))
    outside = collections.Counter(str(n.target) for n in nodes if n.op == "call_function"
                                  and "executorch_call_delegate" not in str(n.target) and "getitem" not in str(n.target))
    log(f"  {name}: {os.path.getsize(path) / 1e6:.2f} MB | delegated subgraphs: {delegates} | ops left outside XNNPACK: {dict(outside) or 'none'}")
    return path


log("lowering to ExecuTorch")
pte_paths = {"float .pte (XNNPACK fp32)": lower(export(fp32, (EX1,)), "mobilenetv2_fp32.pte"),
             "QAT int8 .pte (XNNPACK int8)": lower(export(copy.deepcopy(qat_int8), (EX1,)), "mobilenetv2_int8.pte")}
runtime = Runtime.get()
pte = {}
for k, v in pte_paths.items():
    method = runtime.load_program(v).load_method("forward")         # typed Optional: None if the program has no "forward"
    if method is None:
        raise SystemExit(f"{v} has no 'forward' method")
    pte[k] = method
pte_sizes = {k: os.path.getsize(v) / 1e6 for k, v in pte_paths.items()}

# ---------------------------------------------------------------- 4. accuracy (CPU, one pass over the evaluated images)
log("evaluating on CPU")
models = {"float (PyTorch)": fp32, "QAT int8, simulated (PyTorch)": qat_int8}          # run batch-wise, for comparison with the .pte rows
sizes = {"float (PyTorch)": fp32_size_mb(fp32), "QAT int8, simulated (PyTorch)": int8_size_mb_estimate(qat_int8), **pte_sizes}
correct = {k: 0 for k in list(pte) + list(models)}
n_seen = 0
for i, (x, y) in enumerate(val_loader):
    for k, m in models.items():
        correct[k] += (m(x).argmax(1) == y).sum().item()
    for j in range(len(y)):                                                             # .pte programs are exported for batch 1
        xj = x[j:j + 1].contiguous()
        for k, m in pte.items():
            correct[k] += int(m.execute([xj])[0].argmax(1).item() == y[j].item())
    n_seen += len(y)
    if i % 50 == 0:
        log(f"  {n_seen}/{n_val} | top-1: " + " | ".join(f"{100 * c / n_seen:.2f}" for c in correct.values()))

# ---------------------------------------------------------------- 5. latency of the .pte programs at batch 1
step = max(1, n_val // args.latency_images)
lat_x = [val_loader.dataset[j][0].unsqueeze(0).contiguous() for j in range(0, n_val, step)][:args.latency_images]
for m in pte.values():                                               # warm-up: first calls are slower
    for _ in range(20): m.execute([lat_x[0]])
lat = {k: [] for k in pte}
for xj in lat_x:                                                     # alternate the models so machine noise hits both equally
    for k, m in pte.items():
        t = time.perf_counter(); m.execute([xj]); lat[k].append((time.perf_counter() - t) * 1e3)
base = np.median(lat["float .pte (XNNPACK fp32)"])

# ---------------------------------------------------------------- report
w = max(len(k) for k in correct)
line = "=" * (w + 46)
print("\n" + line)
print(f"MobileNetV2 | QAT {args.epochs} epochs, batch {args.batch_size}, lr {args.lr:g}, {train_min:.1f} min | CPU {os.cpu_count()} cores, {os.uname().machine}")
print(f"{'model':{w}s} {'size MB':>8s} {f'top-1 ({n_seen})':>14s} {'median ms':>10s} {'speedup':>8s}")
for k in correct:
    med = f"{np.median(lat[k]):.2f}" if k in lat else "-"
    spd = f"{base / np.median(lat[k]):.2f}x" if k in lat else "-"
    print(f"{k:{w}s} {sizes[k]:8.2f} {100 * correct[k] / n_seen:14.2f} {med:>10s} {spd:>8s}")
print(f"Float on all 50000 val images should be about 71.88 (torchvision's published figure); --max-val-images gives a random subset. Latency: batch 1,")
print(f"{len(lat_x)} images, execute() only, on this machine's CPU. The PyTorch int8 row is the converted graph run in float (simulated); the .pte rows use real XNNPACK kernels.")
print(f"models saved in {run_dir}")
print(line)
