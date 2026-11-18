# MobileNetV2 quantization-aware training with torchao and ExecuTorch (XNNPACK)

Quantization-aware training (QAT) of torchvision's MobileNetV2 on ImageNet-1k, using torchao's PT2E flow and ExecuTorch's
XNNPACK quantizer, then lowering the float and int8 models to ExecuTorch `.pte` files and comparing them on size, top-1 accuracy
and latency. Training runs on a Modal H100. The hyperparameters come from torchvision's
[`references/classification`](https://github.com/pytorch/vision/tree/main/references/classification) QAT recipe for MobileNetV2.

The work started as a code walkthrough for a talk on Jacob et al. (2017), *Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference* ([arXiv:1712.05877](https://arxiv.org/abs/1712.05877)). It is not a tutorial
on QAT; the paper's ideas are noted below where this code uses or differs from them.

## Results

Both runs fine-tune from `MobileNet_V2_Weights.IMAGENET1K_V1` on a **random 100k-image subset** of ImageNet-1k train (not the
full 1.28M), batch size 256, lr 1e-4, SGD momentum 0.9, weight decay 1e-4, per-tensor weights, one H100.

**Run 2 (latest): 15 epochs, 23.1 min of training. Evaluated on a seeded random 20,000-image subset of the validation set**
([full log](results/run2_15ep_100k_train_and_eval.log)). Latency is batch 1, median of 300 images, on a 64-core x86 Modal machine.

| model | size (MB) | top-1 (20k val) | median latency (ms) | speedup |
|---|---|---|---|---|
| float `.pte` (XNNPACK fp32) | 13.99 | 71.66 | 3.36 | 1.00x |
| QAT int8 `.pte` (XNNPACK int8) | 3.59 | 69.31 | 2.91 | 1.15x |
| float, PyTorch graph | 14.26 | 71.65 | - | - |
| QAT int8, converted graph run in float ("simulated") | 3.61 | 69.32 | - | - |

**Run 1 (earlier version of the script): 10 epochs, 26.7 min of training, evaluated on all 50,000 validation images in
PyTorch only (no `.pte`).** Float 71.87, trained fake-quant graph 70.29, converted int8 graph 68.63. The float number matches
torchvision's published 71.878, which validates the data pipeline, labels and transforms. The raw log of this run was not kept.

What the runs show:

- **Lowering to XNNPACK loses nothing.** Each `.pte` matches its PyTorch graph within 0.01 points (71.66 vs 71.65, 69.31 vs 69.32).
  The whole model is delegated to XNNPACK as a single subgraph with no ops left outside it.
- **int8 costs 2.35 points here** (71.66 to 69.31) in exchange for a 3.9x smaller file. torchvision's published quantized MobileNetV2
  (eager mode, qnnpack, 90 epochs) loses 0.22 points (71.878 to 71.658), so this run recovers much less.
- **The latency gain is small, 1.15x**, and was measured on a many-core x86 machine, which says little about an ARM CPU.
  Testing on an Oracle Cloud A1.Flex ARM VM is planned but **has not been done**.
- **Run 1 had an unexplained 1.66-point drop between the trained fake-quant graph (70.29) and the converted int8 graph (68.63).**
  Run 2 no longer evaluates the fake-quant graph, so that gap is not re-measured. It is the main open question.

Hypotheses for the gap to the reference, none tested: roughly 77x fewer training samples seen (15 epochs x 100k = 1.5M versus
90 epochs x 1.28M for the reference), per-tensor instead of per-channel weights (`--per-channel` exists but was not run), and
the conversion step above. Runs 1 and 2 differ in epochs and in evaluation set, so their numbers are not directly comparable.

Training throughput, from the logs: the first epoch is slow (about 270-320 img/s, 5.2 min in run 2) because files are read cold from the
Modal volume; later epochs run at about 1,250-1,300 img/s (about 1.3 min). The jump is consistent with the dataset subset being
cached in memory after the first pass. I inferred this from the numbers and did not profile it. The full 1.28M-image set does not fit
in the container's RAM cache, so full-data epochs would stay slow; `bench_throughput.py` was written to separate storage, data-loading and
GPU limits but has not been run on Modal.

## The paper and this code

Jacob et al. propose an affine quantization scheme, real = S (q - Z), so that inference can run with integer-only arithmetic,
together with a training procedure that simulates quantization in the forward pass so the network adapts to it.

| idea in the paper | in this repo |
|---|---|
| Insert fake-quantization ops where tensors would be quantized at inference; train with the rounding simulated in float | `prepare_qat_pt2e` with `XNNPACKQuantizer` (`is_qat=True`) inserts them into the exported model |
| Fold batch norm into the weights before quantizing them, and simulate that folding during training | the PT2E QAT flow fuses conv and batch norm; the folded weights are what get fake-quantized |
| Weight range min/max, restricted to [-127, 127] so a product never reaches -128 * -128 (lets two products accumulate in int16) | XNNPACK's symmetric int8 weights use the same [-127, 127] range |
| Activation ranges from exponential moving averages of the ranges seen during training | moving-average observers; frozen after epoch 4 (`pt2e.disable_observer`), following torchvision's recipe |
| Ranges are nudged so 0.0 is exactly representable, giving the scale S and zero point Z | activations use int8 per-tensor affine quantization with a zero point; weights are symmetric (zero point 0) |
| int32 accumulators, bias quantized to int32 with scale S_w x S_x, output rescaled by a fixed-point multiplier | done by the XNNPACK int8 kernels after lowering; this repo does not reimplement it |
| Disable activation quantization at the start of training (they delay it 50k to 2M steps; 500k in their ResNet/Inception protocols) | **not used**: fake-quant is on from step 0, since we fine-tune an already-trained float model |
| uint8 weights and activations, run with gemmlowp / TensorFlow Lite on Snapdragon cores | int8, ExecuTorch + XNNPACK, on x86 (Modal) now and ARM (planned) |
| Judge a quantization scheme by the latency-vs-accuracy trade-off measured on real hardware, not by accuracy alone | `.pte` latency is measured, but only on x86 so far |

Two schedule details come from torchvision's recipe, not the paper: batch norm statistics freeze after epoch 3, and observers
freeze after epoch 4. The recipe's step schedule (lr x0.1 every 30 epochs) resembles the paper's ResNet protocol, but with 10-15
epochs it never triggers here.

Why the converted graph is called "simulated": `convert_pt2e` produces explicit quantize/dequantize ops, but PyTorch still runs the math
in float. Real int8 arithmetic happens only after lowering to ExecuTorch, which is why the `.pte` rows are the ones that
carry latency numbers.

## Pipeline

`mobilenetv2_qat_torchao.py` runs everything in one go:

1. QAT on the GPU: export the model (train mode, dynamic batch) -> `prepare_qat_pt2e` -> train with SGD + StepLR.
2. Convert on the CPU with `convert_pt2e` and save the trained state as `qat_state.pt`.
3. Lower the float model and the int8 model to `.pte` with the XNNPACK delegate (batch size 1) and report any ops left outside XNNPACK.
4. Evaluate top-1 on the validation set (or a random subset with `--max-val-images`) for the two `.pte` models and the two PyTorch graphs.
5. Time both `.pte` models at batch 1, alternating between them, and report the median over `--latency-images` (default 300) images.

Output goes to `<out-dir>/run-<timestamp>/`: `qat_state.pt`, `mobilenetv2_fp32.pte`, `mobilenetv2_int8.pte`.

## Files

| file | purpose |
|---|---|
| `mobilenetv2_qat_torchao.py` | the training, conversion, lowering and evaluation script |
| `run_modal_mobilenetv2.py` | runs it on a Modal H100, reading ImageNet from a Modal volume and saving to `/vol/runs` |
| `prep_imagenet_modal.py` | one-time job that writes ImageNet-1k from the Hugging Face parquet shards to the volume as an ImageFolder tree (needs a Modal secret `huggingface-secret` with `HF_TOKEN`, and accepted dataset terms) |
| `lower_mobilenetv2.py` | lowers a saved `qat_state.pt` to `.pte` files on another machine, e.g. an ARM VM |
| `eval_pte.py` | evaluates and times `.pte` files on another machine (batch 1, defaults to 5,000 validation images) |
| `bench_throughput.py`, `run_bench_modal.py` | measure file-read, data-loader and GPU-step throughput separately (not yet run on Modal) |
| `results/` | log of run 2 |

## Running it

Requires torch 2.15.0, torchao 0.18.0 and executorch 1.6.0 (see `requirements.txt`; executorch 1.6.0 pins torch 2.15.0).
The data must be in ImageFolder layout: `<data-dir>/train/<class>/*.JPEG` and `<data-dir>/val/<class>/*.JPEG`.

```bash
# on Modal: prepare the data once, then train (the volume `imagenet-vol` is created as a v2 volume)
modal run prep_imagenet_modal.py
modal run --detach run_modal_mobilenetv2.py --epochs 15 --max-train-images 100000 --max-val-images 20000
modal volume get imagenet-vol runs ./runs

# anywhere with the requirements installed (a GPU is used for training if there is one)
python mobilenetv2_qat_torchao.py --data-dir /data/imagenet --epochs 15 --max-train-images 100000 --max-val-images 20000

# lower and evaluate on another machine, such as an ARM VM
python lower_mobilenetv2.py --qat-weights runs/run-<timestamp>/qat_state.pt --out-dir pte
python eval_pte.py --data-dir /data/imagenet --pte-dir pte
```

`--max-train-images 0` (the default) uses all 1.28M images. Epochs would then stay at the cold-read speed, which I estimate at
roughly 80 minutes each at about 260 img/s (not measured), so the Modal timeout in the runner is set to 24 hours.

## Caveats

- Training used a 100k-image subset, 10 to 15 epochs, one seed, and no repeated runs, so differences of a few tenths of a point between runs are not established.
- The 20k validation figures are for a random subset. The float model should score about 71.88 on the full 50,000 images.
- Latency was measured on a shared many-core x86 machine and includes the Python binding's overhead; treat it as a smoke test.
- Whether an x86-lowered `.pte` runs on ARM is untested. `lower_mobilenetv2.py` exists to re-lower on the target machine.
- `torch.ao.quantization` is removed in torch 2.15 and torchao 0.18 no longer ships an XNNPACK quantizer, which is why the quantizer is imported from ExecuTorch.
