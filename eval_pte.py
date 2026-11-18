"""eval_pte.py: run the float and int8 ExecuTorch programs on ImageNet validation images and report size, top-1 and latency.

Both models see the same images, one at a time (batch 1), alternating per image so machine noise hits both equally.
Latency is the time of method.execute() only; image decoding and preprocessing are outside the timer and run in this
process (--workers 0 by default) so they never compete with the model for CPU cores.

  python eval_pte.py --data-dir /data/imagenet --pte-dir pte --max-val-images 5000
"""
import argparse, os, platform, time

import numpy as np
import torch, torchvision
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from executorch.runtime import Runtime

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True, help="folder with val/ in ImageFolder layout")
p.add_argument("--pte-dir", default="pte")
p.add_argument("--max-val-images", type=int, default=5000, help="seeded random subset (0 = all 50,000)")
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--workers", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

tf = T.Compose([T.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(224),
                T.PILToTensor(), T.ConvertImageDtype(torch.float), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
ds = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"), tf)
if args.max_val_images and args.max_val_images < len(ds):
    ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:args.max_val_images].tolist())
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=args.workers)

runtime = Runtime.get()
names = {"float (XNNPACK fp32)": "mobilenetv2_fp32.pte", "QAT int8 (XNNPACK int8)": "mobilenetv2_int8.pte"}
methods, sizes = {}, {}
for label, f in names.items():
    path = os.path.join(args.pte_dir, f)
    sizes[label] = os.path.getsize(path) / 1e6
    methods[label] = runtime.load_program(path).load_method("forward")

print(f"{platform.machine()} | {os.cpu_count()} cpus | {len(ds)} images | batch 1", flush=True)
x0 = ds[0][0].unsqueeze(0).contiguous()
for m in methods.values():                                   # warm-up: first calls are slower
    for _ in range(args.warmup):
        m.execute([x0])

correct = {k: 0 for k in methods}
lat = {k: [] for k in methods}
n = 0
for x, y in loader:
    for i in range(len(y)):
        xi = x[i:i + 1].contiguous()
        for label, m in methods.items():
            t = time.perf_counter()
            out = m.execute([xi])[0]
            lat[label].append((time.perf_counter() - t) * 1e3)
            correct[label] += int(out.argmax(1).item() == y[i].item())
        n += 1
    if (n // 32) % 20 == 0:
        print(f"  {n}/{len(ds)} | " + " | ".join(f"{100 * c / n:.2f}" for c in correct.values()), flush=True)

w = max(len(k) for k in methods)
base = np.median(lat["float (XNNPACK fp32)"])
print("\n" + "=" * (w + 62))
print(f"MobileNetV2 on ExecuTorch/XNNPACK | {platform.machine()}, {os.cpu_count()} cpus | ImageNet val subset of {n} images | batch 1")
print(f"{'model':{w}s} {'.pte MB':>8s} {'top-1 %':>8s} {'mean ms':>8s} {'median ms':>10s} {'p95 ms':>8s} {'speedup':>8s}")
for k in methods:
    a = np.array(lat[k])
    print(f"{k:{w}s} {sizes[k]:8.2f} {100 * correct[k] / n:8.2f} {a.mean():8.2f} {np.median(a):10.2f} {np.percentile(a, 95):8.2f} {base / np.median(a):7.2f}x")
print("speedup = float median latency / this model's median latency. Latency is execute() only.")
print("=" * (w + 62))
