"""Select and report the PyTorch device for training commands."""

import torch


def add_device_argument(parser):
    parser.add_argument("--device", default="auto",
                        help="auto selects CUDA when available; use cpu, cuda or cuda:N to override")


def resolve_device(parser, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (ValueError, RuntimeError) as exc:
        parser.error(f"Invalid training device: {exc}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Check your CUDA-enabled PyTorch installation and NVIDIA driver.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            parser.error(f"CUDA device {index} is unavailable.")
        device = torch.device("cuda", index)
        name = torch.cuda.get_device_name(index)
    else:
        name = device.type.upper()
    print(f"Training on {device} ({name})", flush=True)
    return str(device), name
