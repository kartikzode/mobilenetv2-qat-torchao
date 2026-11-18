"""lower_mobilenetv2.py: lower MobileNetV2 (float and QAT int8) to ExecuTorch .pte files with the XNNPACK delegate.

  float  torchvision MobileNet_V2_Weights.IMAGENET1K_V1  -> mobilenetv2_fp32.pte  (XNNPACK fp32 kernels)
  int8   the QAT state saved by mobilenetv2_qat_torchao.py (qat_state.pt in its run folder) -> convert_pt2e -> mobilenetv2_int8.pte (XNNPACK int8 kernels)

Both use the same partitioner and runtime, so the comparison is XNNPACK against XNNPACK. Batch size is fixed at 1.
The script prints how many subgraphs were delegated to XNNPACK and which ops were left outside it: an int8 model whose
convolutions were not delegated would run, but slowly and without int8 kernels.

Needs torch 2.15.0, torchao 0.18.0, executorch 1.6.0. Run it on the machine that will run the .pte files (the Oracle VM).

  python lower_mobilenetv2.py --qat-weights runs/run-<timestamp>/qat_state.pt --out-dir pte
"""
import argparse, collections, os

import torch, torchvision
from torch.export import Dim, export
from torchvision.models import MobileNet_V2_Weights

import torchao.quantization.pt2e as pt2e
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.exir import to_edge_transform_and_lower

p = argparse.ArgumentParser()
p.add_argument("--qat-weights", required=True, help="qat_state.pt written by mobilenetv2_qat_torchao.py")
p.add_argument("--out-dir", default="pte")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
EX = torch.randn(1, 3, 224, 224)


def lower(exported, name):
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()
    path = os.path.join(args.out_dir, name)
    with open(path, "wb") as f:
        f.write(program.buffer)
    nodes = list(edge.exported_program().graph_module.graph.nodes)
    delegates = sum(1 for n in nodes if "executorch_call_delegate" in str(n.target))
    outside = collections.Counter(str(n.target) for n in nodes
                                  if n.op == "call_function" and "executorch_call_delegate" not in str(n.target) and "getitem" not in str(n.target))
    print(f"{name}: {os.path.getsize(path) / 1e6:.2f} MB | delegated subgraphs: {delegates} | ops left outside XNNPACK: {dict(outside) or 'none'}")


# ---- float
fp32 = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
lower(export(fp32, (EX,)), "mobilenetv2_fp32.pte")

# ---- int8: rebuild the QAT graph exactly as in training, load the trained state, convert, then export for ExecuTorch
ckpt = torch.load(args.qat_weights, map_location="cpu", weights_only=True)
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=ckpt["per_channel"], is_qat=True))
dyn = ({0: Dim("batch", min=1, max=4096)},)
qat = prepare_qat_pt2e(export(torchvision.models.mobilenet_v2(weights=None).train(), (torch.randn(8, 3, 224, 224),),
                              dynamic_shapes=dyn, strict=True).module(), quantizer)
qat.load_state_dict(ckpt["state_dict"], strict=True)
int8 = convert_pt2e(qat)
pt2e.move_exported_model_to_eval(int8)
lower(export(int8, (EX,)), "mobilenetv2_int8.pte")
