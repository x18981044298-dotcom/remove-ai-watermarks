"""Img2img pipeline execution with progress monitoring and MPS fallback.

Extracted from ``watermark_remover.py`` to keep the ``WatermarkRemover``
class focused on orchestration.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

from remove_ai_watermarks.noai.progress import is_mps_error, make_pipeline_progress

logger = logging.getLogger(__name__)


def run_img2img(
    pipeline: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
    extra_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """Execute img2img with live progress and return the generated image.

    ``extra_kwargs`` overlays additional pipeline arguments (e.g. the ControlNet
    ``control_image`` / ``controlnet_conditioning_scale`` and a non-empty prompt),
    so a ControlNet img2img pass reuses the same progress + fallback machinery.
    """
    effective_steps = max(1, int(num_inference_steps * strength))

    step_cb, first_step, done_ev, start_updater = make_pipeline_progress(
        effective_steps,
        device,
        set_progress,
    )
    start_updater()

    try:
        result = _call_pipeline(
            pipeline, image, strength, num_inference_steps, guidance_scale, generator, step_cb, extra_kwargs
        )
        done_ev.set()
        return result.images[0]
    except TypeError:
        first_step.set()
        result = _call_pipeline(
            pipeline, image, strength, num_inference_steps, guidance_scale, generator, None, extra_kwargs
        )
        done_ev.set()
        return result.images[0]
    finally:
        first_step.set()
        done_ev.set()


def run_img2img_with_mps_fallback(
    load_pipeline: Callable[[], Any],
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
    *,
    reload_on_cpu: Callable[[], Any],
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Image.Image, str]:
    """Run img2img; on MPS error, fall back to CPU.

    ``extra_kwargs`` overlays extra pipeline arguments (used by the ControlNet
    path). Returns ``(result_image, final_device)`` — device may change to
    ``"cpu"`` on fallback.
    """
    pipeline = load_pipeline()

    try:
        img = run_img2img(
            pipeline,
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            device,
            set_progress,
            extra_kwargs,
        )
        return img, device
    except RuntimeError as error:
        if device == "mps" and is_mps_error(error):
            logger.warning("MPS error detected: %s. Falling back to CPU.", error)
            set_progress("MPS error! Clearing cache and retrying on CPU...")
            try_empty_device_cache("mps")
            pipeline = reload_on_cpu()
            img = run_img2img(
                pipeline, image, strength, num_inference_steps, guidance_scale, None, "cpu", set_progress, extra_kwargs
            )
            return img, "cpu"
        raise


def _call_pipeline(
    pipeline: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    step_callback: Any,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "prompt": "",
        "image": image,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if step_callback is not None:
        kwargs["callback"] = step_callback
        kwargs["callback_steps"] = 1
    return pipeline(**kwargs)


def try_empty_device_cache(device: str) -> None:
    """Best-effort free of cached GPU/MPS/XPU memory for ``device``.

    ``torch.<device>.empty_cache()`` exists for cuda/mps/xpu but not cpu (the
    hasattr guard skips the cpu no-op). Never raises -- callers use it as cleanup
    (the MPS->CPU fallback here, and the batch loop in watermark_remover).
    """
    with contextlib.suppress(Exception):
        import torch

        backend = getattr(torch, device, None)
        if backend is not None and hasattr(backend, "empty_cache"):
            backend.empty_cache()  # type: ignore[attr-defined]
