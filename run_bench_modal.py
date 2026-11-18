import subprocess, modal

app = modal.App("mobilenetv2-qat-bench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "torchao==0.18.0", "executorch==1.6.0", "pillow", "numpy")
    .add_local_file("bench_throughput.py", "/root/bench_throughput.py")
)
vol = modal.Volume.from_name("imagenet-vol", version=2)
CPUS = 48


@app.function(image=image, gpu="H100", cpu=CPUS, memory=65536, timeout=3600, volumes={"/vol": vol})
def bench():
    subprocess.run(["python", "/root/bench_throughput.py", "--data-dir", "/vol/imagenet", "--workers", str(CPUS)], check=True)


@app.local_entrypoint()
def main():
    bench.remote()
