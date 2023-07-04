import argparse
import torch
from typing import Dict
from safetensors.torch import save_file

def parse_args():
    parser = argparse.ArgumentParser(description='Convert an RWKV model checkpoint into safetensors')
    parser.add_argument('src_path', help='Path to PyTorch checkpoint file')
    parser.add_argument('dest_path', help='Path to safetensor checkpoint file, will be overwritten')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state_dict: Dict[str, torch.Tensor] = torch.load(args.src_path, map_location='cpu')

    safe_tensors = {}

    print("{")
    for key, tensor in state_dict.items():
        print(f"\"{key}\": \"{tensor.shape} {tensor.type()}\",")
        safe_tensors[key] = tensor.float().half()
    print("}")

    save_file(safe_tensors, args.dest_path)

if __name__ == "__main__":
    main()