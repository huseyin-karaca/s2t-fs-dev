import os
import platform

import torch


def get_torch_device() -> torch.device:
    """Return the best available compute device: MPS > CUDA > CPU.

    Priority:
      1. Apple Metal (MPS) — Apple Silicon / AMD on macOS via Metal Performance Shaders
      2. CUDA             — NVIDIA GPU
      3. CPU              — fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_device(device: torch.device, seed: int) -> None:
    """Set the device-specific random seed for reproducibility."""
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def log_hardware_info(logger) -> None:
    """Log a hardware summary at experiment startup."""
    device = get_torch_device()
    cpu_count = os.cpu_count() or 1

    if device.type == "mps":
        device_label = "Apple Metal (MPS) ✓"
    elif device.type == "cuda":
        device_label = f"CUDA — {torch.cuda.get_device_name(0)}"
    else:
        device_label = "CPU (no GPU acceleration)"

    machine = platform.machine()
    system = platform.system()

    logger.bind(category="Hardware").info(
        f"Hardware — PyTorch device : {device_label}"
    )
    logger.bind(category="Hardware").info(
        f"Hardware — CPU cores      : {cpu_count}  ({machine} / {system})"
    )
    logger.bind(category="Hardware").info(
        f"Hardware — Tree models    : CPU, {cpu_count} threads (XGBoost / LightGBM)"
    )
