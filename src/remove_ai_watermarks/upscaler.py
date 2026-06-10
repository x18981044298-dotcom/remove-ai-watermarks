"""Optional pre-diffusion super-resolution for small inputs (Real-ESRGAN via spandrel).

Mirrors ``region_eraser``'s optional-backend pattern: ``is_available()`` guards the
``spandrel`` import, a lazy singleton (double-checked lock) holds the loaded model, and
the weights download on first use (cached by ``torch.hub``) -- they are never bundled.

The DEFAULT upscaler stays Lanczos (cv2, no deps); this is opt-in via the ``esrgan``
extra and feeds the ``--upscaler esrgan`` path. ``spandrel`` is a pure model-loader
(MIT) with NO basicsr dependency -- it pulls only torch/torchvision/safetensors/numpy/
einops -- so it sidesteps the basicsr / ``torchvision.transforms.functional_tensor``
breakage that the retired ``restore`` (GFPGAN) extra had to shim. Real-ESRGAN weights
are BSD-3-Clause.

CPU works but is slow on large inputs, so this is meant for the pre-diffusion upscale of
SMALL inputs (and the GPU worker). On a memory-constrained host it is a no-op (the extra
is absent), and the caller falls back to Lanczos.
"""

# torch/spandrel boundary: these libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Real-ESRGAN x2plus (BSD-3-Clause), official release. x2 is the right native factor for
# the pre-diffusion floor upscale (small inputs ~512 -> ~1024); spandrel infers the
# architecture and scale from the checkpoint, so swapping the URL is enough to change it.
_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
_MODEL_FILENAME = "RealESRGAN_x2plus.pth"

_model: Any = None  # lazy singleton (spandrel ImageModelDescriptor)
_model_device: str = "cpu"
_lock = threading.Lock()


def is_available() -> bool:
    """True if the ``esrgan`` extra (spandrel + torch) is importable."""
    from .optional_deps import module_available

    return module_available("spandrel", "torch")


def _model_cache_path() -> Path:
    """Path the weights are cached at (the torch.hub checkpoints dir)."""
    import torch

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _MODEL_FILENAME


def _get_model(device: str) -> Any:
    """Load the Real-ESRGAN model once (downloading the weights on first use)."""
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model
    with _lock:
        if _model is None:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader

            dst = _model_cache_path()
            if not dst.exists():
                logger.info("Downloading Real-ESRGAN weights to %s", dst)
                torch.hub.download_url_to_file(_MODEL_URL, str(dst), progress=False)
            model = ModelLoader().load_from_file(str(dst))
            if not isinstance(model, ImageModelDescriptor):
                raise RuntimeError(f"Unexpected spandrel model type: {type(model).__name__}")
            _model = model.eval()
        if _model_device != device:
            _model.to(device)
            _model_device = device
    return _model


def scale() -> int:
    """The model's native upscale factor (e.g. 2 for x2plus). Loads the model if needed."""
    return int(_get_model("cpu").scale)


def upscale(image: NDArray[Any], device: str | None = None) -> NDArray[Any]:
    """Upscale a BGR uint8 image by the model's native factor with Real-ESRGAN.

    Returns a BGR uint8 array. Falls back to CPU if the requested device errors (an
    MPS/CUDA OOM or unsupported-op on the small pre-diffusion input), mirroring the
    diffusion engine's MPS->CPU fallback.

    Raises:
        RuntimeError: if the ``esrgan`` extra is not installed (guard with
            ``is_available()`` first).
    """
    if not is_available():
        raise RuntimeError("Real-ESRGAN upscaler needs the 'esrgan' extra (spandrel). Install it or use Lanczos.")
    import cv2
    import numpy as np
    import torch

    target_device = (device or "cpu").lower()
    if target_device not in {"cpu", "mps", "cuda", "xpu"}:
        target_device = "cpu"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    def _run(dev: str) -> NDArray[Any]:
        model = _get_model(dev)
        with torch.no_grad():
            out = model(tensor.to(dev))
        arr = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        return cv2.cvtColor(arr.round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    try:
        return _run(target_device)
    except Exception as e:  # GPU OOM / unsupported op: fall back to CPU
        if target_device == "cpu":
            raise
        logger.warning("Real-ESRGAN on %s failed (%s); retrying on CPU", target_device, e)
        return _run("cpu")
