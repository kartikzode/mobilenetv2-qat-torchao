"""bench_throughput.py: find out which part of the QAT pipeline limits img/s. Three independent measurements:

  1. raw file reads  : read the bytes of random training files with many threads, no decoding (storage speed)
  2. data loader only: the training DataLoader (decode + random crop + normalize), no model (CPU / loader speed)
  3. GPU step only   : the QAT graph on a random batch already on the GPU, no data loading (model speed)

Whichever number is closest to the ~250 img/s of the real run is the bottleneck.
"""
import argparse, os, random, time
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn.functional as F, torchvision
from torch.export import Dim, export
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--workers", type=int, default=48)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--read-files", type=int, default=3000)
p.add_argument("--loader-batches", type=int, default=40)
p.add_argument("--gpu-steps", type=int, default=30)
args = p.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"), T.Compose([
    T.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR, antialias=True), T.RandomHorizontalFlip(0.5),
    T.PILToTensor(), T.ConvertImageDtype(torch.float), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
idx = random.Random(1).sample(range(len(train)), min(len(train), 40000))

# 1. raw reads
paths = [train.samples[i][0] for i in idx[:args.read_files]]
def read(path):
    with open(path, "rb") as f: return len(f.read())
t = time.time()
with ThreadPoolExecutor(32) as ex: total = sum(ex.map(read, paths))
dt = time.time() - t
print(f"1. raw reads (32 threads, {len(paths)} random files): {len(paths) / dt:.0f} files/s, {total / dt / 1e6:.0f} MB/s", flush=True)

# 2. loader only (different random files from step 1, so reads are cold too)
sub = torch.utils.data.Subset(train, idx[args.read_files:])
loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
n, t = 0, time.time()
for i, (x, y) in enumerate(loader):
    if i == 3: t = time.time(); n = 0                     # skip worker start-up
    n += len(y)
    if i == 3 + args.loader_batches: break
print(f"2. data loader only ({args.workers} workers, batch {args.batch_size}): {n / (time.time() - t):.0f} img/s", flush=True)

# 3. GPU step only
q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=False, is_qat=True))
model = torchvision.models.mobilenet_v2(weights=None).to(DEV)
g = prepare_qat_pt2e(export(model.train(), (torch.randn(8, 3, 224, 224, device=DEV),), dynamic_shapes=({0: Dim("batch", min=1, max=4096)},), strict=True).module(), q)
opt = torch.optim.SGD(g.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
x = torch.randn(args.batch_size, 3, 224, 224, device=DEV); y = torch.randint(0, 1000, (args.batch_size,), device=DEV)
def step():
    loss = F.cross_entropy(g(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); return loss
for _ in range(5): step()
if DEV.type == "cuda": torch.cuda.synchronize()
t = time.time()
for _ in range(args.gpu_steps): step()
if DEV.type == "cuda": torch.cuda.synchronize()
dt = time.time() - t
print(f"3. GPU step only (batch {args.batch_size}, {DEV}): {args.batch_size * args.gpu_steps / dt:.0f} img/s, {dt / args.gpu_steps * 1e3:.0f} ms/step", flush=True)
