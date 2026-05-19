"""
Export PetFace-ID embedding model to ONNX.

If --checkpoint is omitted, exports the MegaDescriptor-based strong baseline.
If --checkpoint is provided, exports the fine-tuned embedding model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from petface_model import ModelConfig, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", default="hf-hub:BVRA/MegaDescriptor-B-224")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("models/petface_id.onnx"))
    parser.add_argument("--meta", type=Path, default=Path("models/petface_onnx_meta.json"))
    args = parser.parse_args()

    model, _ = load_model(ModelConfig(args.model_name, args.img_size, args.embed_dim, args.checkpoint), device=torch.device("cpu"))
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(args.out)))
    session = ort.InferenceSession(str(args.out))
    with torch.no_grad():
        pt = model(dummy).numpy()
    ot = session.run(["embedding"], {"image": dummy.numpy()})[0]
    max_diff = float(np.abs(pt - ot).max())
    meta = {
        "model_version": "PetFace-ID-2.0",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "img_size": args.img_size,
        "embed_dim": int(pt.shape[1]),
        "onnx": str(args.out),
        "max_diff": max_diff,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
