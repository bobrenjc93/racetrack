"""Convert 8-way FP8 shards to BF16, dequantizing all weights."""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

block_size = 128


def weight_dequant(weight, scale):
    shape = weight.shape
    bs = block_size
    out_f, in_f = shape
    pad_out = (bs - out_f % bs) % bs
    pad_in = (bs - in_f % bs) % bs
    w = weight.float()
    if pad_out > 0 or pad_in > 0:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    po, pi = w.shape
    w = w.view(po // bs, bs, pi // bs, bs).transpose(1, 2).contiguous()
    w = (w.view(-1, bs * bs) * scale.view(-1, 1).float())
    w = w.view(po // bs, pi // bs, bs, bs).transpose(1, 2).contiguous().view(po, pi)
    return w[:out_f, :in_f].to(torch.bfloat16)


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)

    for shard in sorted(Path(src).glob("model*-mp*.safetensors")):
        print(f"Converting {shard.name} ...")
        sd = load_file(str(shard), device="cpu")
        new_sd = {}
        scale_keys = {k for k in sd if k.endswith(".scale")}
        for key, tensor in sd.items():
            if key in scale_keys:
                continue
            scale_key = key.replace(".weight", ".scale")
            if tensor.dtype == torch.float8_e4m3fn and tensor.dim() == 2 and scale_key in sd:
                new_sd[key] = weight_dequant(tensor, sd[scale_key])
            else:
                new_sd[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        save_file(new_sd, str(Path(dst) / shard.name))
        print(f"  {len(new_sd)} tensors saved")

    for f in Path(src).glob("tokenizer*"):
        import shutil
        shutil.copy(f, Path(dst) / f.name)
        print(f"Copied {f.name}")


if __name__ == "__main__":
    main()
