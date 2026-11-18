import subprocess, modal

app = modal.App("mobilenetv2-qat")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "torchao==0.18.0", "executorch==1.6.0", "pillow", "numpy")
    .add_local_file("mobilenetv2_qat_torchao.py", "/root/mobilenetv2_qat_torchao.py")
)
vol = modal.Volume.from_name("imagenet-vol", version=2)   # filled by prep_imagenet_modal.py: /vol/imagenet/{train,val}

# Data loading (JPEG decode + random crop on the CPU) is the likely bottleneck, so the speed-up is mostly CPU cores.
CPUS = 48


@app.function(image=image, gpu="H100", cpu=CPUS, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def run(epochs: int, workers: int, max_train_images: int, max_val_images: int):
    subprocess.run(
        ["python", "/root/mobilenetv2_qat_torchao.py",
         "--data-dir", "/vol/imagenet",
         "--epochs", str(epochs),
         "--workers", str(workers),
         "--max-train-images", str(max_train_images),
         "--max-val-images", str(max_val_images),
         "--out-dir", "/vol/runs"],       # saved models land in /vol/runs/run-<timestamp>/ ; download: modal volume get imagenet-vol runs ./runs
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def main(epochs: int = 10, workers: int = CPUS, max_train_images: int = 0, max_val_images: int = 0):
    run.remote(epochs, workers, max_train_images, max_val_images)
