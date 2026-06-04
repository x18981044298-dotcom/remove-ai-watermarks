"""Watermark removal using diffusion model regeneration attack.

Two pipelines:
1. ``default`` -- plain SDXL img2img. Partial-noise regeneration scrubs the
   invisible watermark; ``strength`` controls how much is regenerated.
2. ``controlnet`` -- SDXL img2img with a canny ControlNet. The watermark REMOVAL
   still comes from the img2img regeneration (``strength``); the ControlNet only
   PRESERVES structure (text/faces) by conditioning on the edge map. No original
   pixels are ever copied or frozen, so SynthID does not survive.
   ``controlnet_conditioning_scale`` is the preservation knob.
"""

# torch/diffusers/cv2 boundary: these libs ship no usable types for the tensor and
# array ops below; relax the unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from PIL import Image

from remove_ai_watermarks.noai.watermark_profiles import (
    CONTROLNET_CANNY_MODEL,
    DEFAULT_MODEL_ID,
    DEFAULT_STRENGTH,
    resolve_strength,
)

logger = logging.getLogger(__name__)

# Check for optional dependencies
_HAS_TORCH = False
_HAS_DIFFUSERS = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

try:
    from diffusers import AutoPipelineForImage2Image as AutoImg2ImgPipeline

    _HAS_DIFFUSERS = True
except ImportError:
    AutoImg2ImgPipeline = None  # type: ignore


def is_watermark_removal_available() -> bool:
    """Check if watermark removal dependencies are installed."""
    return _HAS_TORCH and _HAS_DIFFUSERS


# Drop-in fp16-safe replacement for the SDXL VAE. The stock SDXL VAE overflows
# to NaN in fp16 and decodes to an all-black image (issue #29: the raiw.cc black
# result on a CUDA fp16 backend). This community VAE is numerically rescaled to
# stay in fp16 range. SDXL-architecture only.
_SDXL_FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"


def _needs_fp16_vae_fix(model_id: str, default_model_id: str, is_fp16: bool) -> bool:
    """Whether the plain img2img pipeline must swap in the fp16-fixed SDXL VAE.

    Gated to the default SDXL checkpoint running in fp16: cpu/mps run fp32 (the
    stock VAE is fine there) and the differential pipeline upcasts the VAE on its
    own, so only this path on a fp16 GPU (CUDA/XPU) hits the NaN/black decode.
    A custom non-SDXL ``model_id`` keeps its own VAE (the fix is SDXL-specific).
    """
    return is_fp16 and model_id == default_model_id


_CUDA_FIX_ENV_KEY = "NOAI_CUDA_FIXED"


def _auto_install(packages: list[str], index_url: str | None = None) -> bool:
    """Attempt to install missing packages via pip. Returns True on success."""
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    if index_url:
        cmd.extend(["--index-url", index_url])
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is present via nvidia-smi."""
    import subprocess

    try:
        subprocess.check_call(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _detect_cuda_index_url() -> str:
    """Detect the appropriate PyTorch CUDA index URL from nvidia-smi output."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if "CUDA Version" in line:
                version_str = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                major, minor = version_str.split(".")[:2]
                cuda_tag = f"cu{major}{minor}"
                return f"https://download.pytorch.org/whl/{cuda_tag}"
    except Exception:  # noqa: S110
        pass
    return "https://download.pytorch.org/whl/cu121"


def _reinstall_torch_cuda_and_restart() -> None:
    """Reinstall torch with CUDA support showing live progress, then restart."""
    import re
    import subprocess

    from remove_ai_watermarks.noai.progress import run_with_progress

    index_url = _detect_cuda_index_url()
    progress_state: dict[str, str] = {"message": "NVIDIA GPU detected — installing CUDA-enabled PyTorch..."}

    pct_re = re.compile(r"(\d+)%")
    pkg_re = re.compile(r"(?:Collecting|Downloading|Installing)\s+(\S+)")

    def _run_pip() -> bool:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "torch",
            "--index-url",
            index_url,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            stripped = line.strip()
            if not stripped:
                continue
            pkg_m = pkg_re.search(stripped)
            pct_m = pct_re.search(stripped)
            if pct_m and pkg_m:
                progress_state["message"] = f"Downloading {pkg_m.group(1)} ({pct_m.group(1)}%)"
            elif pct_m:
                progress_state["message"] = f"Downloading CUDA packages ({pct_m.group(1)}%)"
            elif pkg_m:
                action = "Installing" if stripped.startswith("Installing") else "Downloading"
                progress_state["message"] = f"{action} {pkg_m.group(1)}"
            elif "Successfully installed" in stripped:
                progress_state["message"] = "CUDA-enabled PyTorch installed successfully"
        proc.wait()
        return proc.returncode == 0

    try:
        success = run_with_progress(_run_pip, progress_state)
    except Exception:
        success = False

    if not success:
        print(
            f"\n  Failed to install CUDA-enabled PyTorch.\n"
            f"  Install manually:\n"
            f"    pip install torch --index-url {index_url}\n",
            file=sys.stderr,
        )
        return

    os.environ[_CUDA_FIX_ENV_KEY] = "1"
    # Re-exec via ``-m`` rather than building a ``-c`` string from repr(sys.argv).
    # ``-m`` makes Python set argv[0] to the module path, so forward only the
    # actual args (sys.argv[1:]); passing the full argv would re-inject the
    # program name as a spurious first argument to Click.
    os.execv(sys.executable, [sys.executable, "-m", "remove_ai_watermarks.cli", *sys.argv[1:]])


def _ensure_watermark_deps() -> None:
    """Auto-install and re-import missing watermark removal dependencies."""
    global _HAS_TORCH, _HAS_DIFFUSERS, torch, AutoImg2ImgPipeline
    missing_pkgs: list[str] = []
    if not _HAS_TORCH:
        missing_pkgs.append("torch")
    if not _HAS_DIFFUSERS:
        missing_pkgs.extend(["diffusers", "transformers", "accelerate"])
    logger.info("Auto-installing missing dependencies: %s", missing_pkgs)
    if not _auto_install(missing_pkgs):
        raise ImportError(
            f"Failed to auto-install missing dependencies: {', '.join(missing_pkgs)}. "
            "Try manually: pip install --force-reinstall noai-watermark"
        )
    import torch as _torch

    torch = _torch
    _HAS_TORCH = True
    from diffusers import AutoPipelineForImage2Image

    AutoImg2ImgPipeline = AutoPipelineForImage2Image
    _HAS_DIFFUSERS = True


def get_device() -> str:
    """Get the best available device for inference."""
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():  # type: ignore
        try:
            t = torch.tensor([1.0], device="cuda")
            _ = t + t
            del t
            return "cuda"
        except (AssertionError, RuntimeError):
            pass
    # Intel GPU (Arc / Data Center) via the torch XPU backend. The torch.xpu
    # namespace exists in stock wheels, but is_available() is only True on an
    # XPU-enabled build (download.pytorch.org/whl/xpu), so this is inert on the
    # default CPU/CUDA install. Checked before the nvidia-smi path so an Intel
    # box never triggers the CUDA reinstaller.
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore
        try:
            t = torch.tensor([1.0], device="xpu")
            _ = t + t
            del t
            return "xpu"
        except (AssertionError, RuntimeError):
            pass
    if _has_nvidia_gpu() and not os.environ.get(_CUDA_FIX_ENV_KEY):
        _reinstall_torch_cuda_and_restart()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_seed_generator(device: str, seed: int) -> Any:
    """Build a seeded ``torch.Generator``, falling back to a CPU generator.

    Some backends have no device-side RNG (notably certain torch-xpu builds),
    so ``torch.Generator(device="xpu")`` can raise. A CPU generator is
    backend-agnostic and still seeds the pipeline reproducibly, so fall back to
    it rather than failing the run when ``--seed`` is used on such a device.
    """
    try:
        return torch.Generator(device=device).manual_seed(seed)  # type: ignore
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(seed)  # type: ignore


# Canny edge thresholds for the ControlNet control image (xinsir canny recipe:
# cv2.Canny(gray, 100, 200) -> a 3-channel edge map).
_CANNY_LOW = 100
_CANNY_HIGH = 200

# A neutral quality prompt: the goal is faithful regeneration, not creative edits.
_CONTROLNET_PROMPT = "best quality, high quality, sharp, detailed, photographic"
_CONTROLNET_NEGATIVE = "blurry, lowres, deformed, distorted text, garbled text, watermark, jpeg artifacts"


class WatermarkRemover:
    """Remove watermarks from images using diffusion model regeneration.

    Attributes:
        model_id: HuggingFace model ID for the diffusion model.
        device: Device to run inference on (cuda, xpu, mps, or cpu).
    """

    DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    DEFAULT_STRENGTH = DEFAULT_STRENGTH
    CONTROLNET_CANNY_MODEL = CONTROLNET_CANNY_MODEL

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: Any = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
        pipeline: str = "default",
        controlnet_conditioning_scale: float = 1.0,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        # The pipeline profile is threaded explicitly (not inferred from model_id):
        # both "default" and "controlnet" use the same SDXL base checkpoint.
        self.model_profile = pipeline
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        if not is_watermark_removal_available():
            _ensure_watermark_deps()
        self.device = (device or get_device()).lower()
        if self.device == "auto":
            self.device = get_device()
        if self.device not in {"cpu", "mps", "cuda", "xpu"}:
            raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, mps, cuda, xpu.")
        if torch_dtype is None:
            if self.device == "cpu" or self.device == "mps":
                self.torch_dtype = torch.float32  # type: ignore
            else:
                self.torch_dtype = torch.float16  # type: ignore
        else:
            self.torch_dtype = torch_dtype

        self._pipeline: AutoImg2ImgPipeline | None = None
        self._controlnet_pipeline: Any = None
        self._progress_callback = progress_callback
        self.hf_token: str | None = hf_token or os.environ.get("HF_TOKEN")

    def _set_progress(self, message: str) -> None:
        """Send a progress update through callback when available."""
        if self._progress_callback is None:
            return
        with contextlib.suppress(Exception):
            self._progress_callback(message)

    # ── Preload ──────────────────────────────────────────────────────

    def preload(self) -> None:
        """Eagerly load the pipeline so download progress bars are visible."""
        if self.model_profile == "controlnet":
            self._load_controlnet_pipeline()
        else:
            self._load_pipeline()

    # ── Pipeline loading ─────────────────────────────────────────────

    def _maybe_add_fp16_vae(self, load_kwargs: dict[str, Any]) -> None:
        """Swap in the fp16-fixed SDXL VAE for the default checkpoint on a fp16 GPU.

        The stock SDXL VAE overflows to NaN in fp16 and decodes to an all-black
        image (issue #29). Shared by both pipeline loaders; a no-op on fp32 (cpu/mps)
        or a non-SDXL checkpoint.
        """
        if _needs_fp16_vae_fix(self.model_id, self.DEFAULT_MODEL_ID, self.torch_dtype == torch.float16):
            from diffusers import AutoencoderKL

            self._set_progress("Loading fp16-fixed SDXL VAE (avoids black output)...")
            load_kwargs["vae"] = AutoencoderKL.from_pretrained(_SDXL_FP16_VAE_ID, torch_dtype=torch.float16)

    def _move_to_device_and_optimize(self, pipeline: Any) -> Any:
        """Move a freshly-loaded pipeline to ``self.device`` + enable memory opts.

        Shared by both loaders. On a CUDA move failure (missing CUDA torch build),
        trigger the torch-CUDA reinstall+restart. Returns the moved pipeline.
        """
        self._set_progress(f"Moving model to device: {self.device}")
        try:
            pipeline = pipeline.to(self.device)
        except (RuntimeError, AssertionError) as exc:
            if self.device == "cuda" and not os.environ.get(_CUDA_FIX_ENV_KEY):
                self._set_progress("CUDA failed. Reinstalling torch with CUDA support...")
                _reinstall_torch_cuda_and_restart()
            raise RuntimeError(
                f"Failed to move model to {self.device} ({exc}). "
                "Install CUDA-enabled PyTorch manually:\n"
                f"  pip install torch --index-url {_detect_cuda_index_url()}"
            ) from exc

        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            with contextlib.suppress(Exception):
                self._set_progress("Enabling memory optimizations...")
                pipeline.enable_xformers_memory_efficient_attention()

        # Mac Float32 memory slicing
        if self.device == "mps" and hasattr(pipeline, "enable_attention_slicing"):
            with contextlib.suppress(Exception):
                pipeline.enable_attention_slicing("max")

        return pipeline

    def _load_pipeline(self) -> AutoImg2ImgPipeline:
        """Load the plain SDXL img2img pipeline lazily."""
        if self._pipeline is None:
            logger.info("Loading model %s on %s...", self.model_id, self.device)
            self._set_progress(f"Loading model weights: {self.model_id}")

            load_kwargs: dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if self.hf_token:
                load_kwargs["token"] = self.hf_token
            self._maybe_add_fp16_vae(load_kwargs)

            pipeline = AutoImg2ImgPipeline.from_pretrained(self.model_id, **load_kwargs)  # type: ignore
            self._pipeline = self._move_to_device_and_optimize(pipeline)

            logger.info("Model loaded successfully")
            self._set_progress("Model initialized. Preparing input image...")

        return self._pipeline  # type: ignore

    def _load_controlnet_pipeline(self) -> Any:
        """Load the SDXL + canny-ControlNet img2img pipeline lazily.

        Mirrors ``_load_pipeline`` (same fp16-fix VAE, device move, attention
        slicing via the shared helpers) but loads the canny ControlNet on top of
        the SDXL base. The ControlNet only preserves structure via the edge map;
        removal still comes from the img2img regeneration (``strength``).
        """
        if self._controlnet_pipeline is None:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

            logger.info("Loading SDXL + ControlNet (%s) on %s...", CONTROLNET_CANNY_MODEL, self.device)
            self._set_progress(f"Loading ControlNet: {CONTROLNET_CANNY_MODEL}")
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_MODEL, torch_dtype=self.torch_dtype)

            load_kwargs: dict[str, Any] = {"controlnet": controlnet, "torch_dtype": self.torch_dtype}
            if self.hf_token:
                load_kwargs["token"] = self.hf_token
            self._maybe_add_fp16_vae(load_kwargs)

            self._set_progress(f"Loading model weights: {self.model_id}")
            pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(self.model_id, **load_kwargs)
            pipeline = self._move_to_device_and_optimize(pipeline)
            with contextlib.suppress(Exception):
                pipeline.set_progress_bar_config(disable=True)

            logger.info("ControlNet model loaded successfully")
            self._controlnet_pipeline = pipeline

        return self._controlnet_pipeline

    # ── Core removal ─────────────────────────────────────────────────

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float | None = None,
        seed: int | None = None,
        vendor: str | None = None,
    ) -> Path:
        """Remove watermark from an image using regeneration attack.

        Args:
            image_path: Path to the watermarked image.
            output_path: Path for the cleaned image. If None, modifies in place.
            strength: Denoising strength (0.0-1.0). None -> the vendor-adaptive
                default (see ``vendor``).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            vendor: SynthID vendor (``"openai"`` / ``"google"`` / None) used to pick the
                default strength when ``strength`` is None. Detect it from the ORIGINAL
                input with ``watermark_profiles.vendor_for_strength`` before processing
                strips the metadata; the caller passes it down so display and execution
                agree.

        Returns:
            Path to the cleaned image.

        Raises:
            FileNotFoundError: If input image doesn't exist.
            ValueError: If strength is not in valid range.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if output_path is None:
            output_path = image_path

        strength = resolve_strength(strength, vendor)

        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        if guidance_scale is None:
            guidance_scale = 7.5

        self._set_progress("Loading and preprocessing input image...")
        init_image = Image.open(image_path).convert("RGB")
        w, h = init_image.size
        self._set_progress(f"Image loaded: {w}x{h}px | Model: {self.model_id}")

        generator = None
        if seed is not None and _HAS_TORCH:
            self._set_progress(f"Setting reproducible seed: {seed}")
            generator = _make_seed_generator(self.device, seed)

        effective_steps = max(1, int(num_inference_steps * strength))
        self._set_progress(
            f"Config: strength={strength}, steps={num_inference_steps} "
            f"(~{effective_steps} effective), guidance={guidance_scale}, device={self.device}"
        )

        _total_start = time.monotonic()

        if self.model_profile == "controlnet":
            cleaned_image = self._run_controlnet(
                init_image,
                strength,
                num_inference_steps,
                guidance_scale,
                generator,
            )
        else:
            cleaned_image = self._run_img2img(
                init_image,
                strength,
                num_inference_steps,
                guidance_scale,
                generator,
            )

        self._set_progress(f"Regeneration complete · Output: {w}x{h}px {cleaned_image.mode}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower()
        if fmt in (".jpg", ".jpeg"):
            self._set_progress(f"Encoding as JPEG → {output_path.name}...")
        else:
            self._set_progress(f"Encoding as PNG → {output_path.name}...")
        cleaned_image.save(output_path)

        if output_path.exists():
            self._set_progress("Stripping AI metadata from output...")
            try:
                from remove_ai_watermarks.noai.cleaner import remove_ai_metadata

                remove_ai_metadata(output_path, output_path, keep_standard=True)
            except Exception:
                logger.debug("AI metadata stripping skipped", exc_info=True)

        total_time = time.monotonic() - _total_start

        size_str = ""
        try:
            file_size = output_path.stat().st_size
            if file_size < 1024 * 1024:
                size_str = f" ({file_size / 1024:.0f}KB)"
            else:
                size_str = f" ({file_size / (1024 * 1024):.1f}MB)"
        except OSError:
            pass

        logger.info("Cleaned image saved to %s", output_path)
        self._set_progress(f"✓ Saved {output_path.name}{size_str} · {w}x{h}px · {total_time:.0f}s total")

        return output_path

    # ── Img2img runner ───────────────────────────────────────────────

    def _run_img2img(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        """Execute the img2img pipeline with progress and MPS fallback."""
        from remove_ai_watermarks.noai.img2img_runner import run_img2img_with_mps_fallback

        result_image, final_device = run_img2img_with_mps_fallback(
            load_pipeline=self._load_pipeline,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            device=self.device,
            set_progress=self._set_progress,
            reload_on_cpu=self._reload_pipeline_on_cpu,
        )

        if final_device != self.device:
            self.device = final_device
            self.torch_dtype = torch.float32  # type: ignore[assignment]

        return result_image

    def _reload_pipeline_on_cpu(self) -> Any:
        """Reload pipeline on CPU after MPS failure."""
        self.device = "cpu"
        self.torch_dtype = torch.float32  # type: ignore[assignment]
        self._pipeline = None
        return self._load_pipeline()

    # ── ControlNet runner ────────────────────────────────────────────

    def _build_canny_control_image(self, init_image: Image.Image) -> Image.Image:
        """Build the canny ControlNet conditioning image (xinsir recipe).

        cv2.Canny on the RGB->gray array, stacked to 3 channels, wrapped as a PIL
        image. The edge map only PRESERVES structure; it never copies pixels.
        ``init_image`` is already RGB (``remove_watermark`` converts on load).
        """
        import cv2
        import numpy as np

        rgb = np.array(init_image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        return Image.fromarray(edges_rgb)

    def _run_controlnet(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        """Run the SDXL + canny-ControlNet img2img pass.

        Removal still comes from the img2img regeneration (``strength``); the canny
        ControlNet only PRESERVES text and face STRUCTURE via the edge map. No
        original pixels are copied/frozen, so SynthID does not survive (canny holds
        structure, not face identity). ``controlnet_conditioning_scale`` is the
        structure-preservation knob. Shares the img2img runner (live progress +
        MPS->CPU fallback) with ``_run_img2img``; the only delta is the extra
        ControlNet kwargs (canny control image + conditioning scale + a non-empty
        prompt) overlaid via ``extra_kwargs``.
        """
        from remove_ai_watermarks.noai.img2img_runner import run_img2img_with_mps_fallback

        extra_kwargs = {
            "prompt": _CONTROLNET_PROMPT,
            "negative_prompt": _CONTROLNET_NEGATIVE,
            "control_image": self._build_canny_control_image(init_image),
            "controlnet_conditioning_scale": float(self.controlnet_conditioning_scale),
        }
        result_image, final_device = run_img2img_with_mps_fallback(
            load_pipeline=self._load_controlnet_pipeline,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            device=self.device,
            set_progress=self._set_progress,
            reload_on_cpu=self._reload_controlnet_on_cpu,
            extra_kwargs=extra_kwargs,
        )

        if final_device != self.device:
            self.device = final_device
            self.torch_dtype = torch.float32  # type: ignore[assignment]

        return result_image

    def _reload_controlnet_on_cpu(self) -> Any:
        """Reload the controlnet pipeline on CPU after an MPS failure."""
        self.device = "cpu"
        self.torch_dtype = torch.float32  # type: ignore[assignment]
        self._controlnet_pipeline = None
        return self._load_controlnet_pipeline()

    # ── Batch ────────────────────────────────────────────────────────

    def remove_watermark_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        strength: float | None = None,
        num_inference_steps: int = 50,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> list[Path]:
        """Remove watermarks from all images in a directory."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_paths: list[Path] = []

        # Lazy import keeps this module torch-optional; frees device cache per image.
        from remove_ai_watermarks.noai.img2img_runner import try_empty_device_cache

        for ext in extensions:
            for image_path in input_dir.glob(f"*{ext}"):
                output_path = output_dir / image_path.name
                try:
                    result_path = self.remove_watermark(
                        image_path=image_path,
                        output_path=output_path,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                    )
                    cleaned_paths.append(result_path)
                except Exception as e:
                    logger.error("Failed to process %s: %s", image_path, e)
                try_empty_device_cache(self.device)

        return cleaned_paths


# ── Convenience function ─────────────────────────────────────────────


def remove_watermark(
    image_path: Path,
    output_path: Path | None = None,
    strength: float | None = None,
    model_id: str | None = None,
    device: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Convenience function to remove watermark from an image.

    ``strength=None`` lets the profile pick its vendor-adaptive SDXL default
    (0.10 OpenAI / 0.15 Google / 0.15 unknown, from the C2PA SynthID proxy on the
    input). Pass a value to override.
    """
    from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

    remover = WatermarkRemover(model_id=model_id, device=device, hf_token=hf_token)
    return remover.remove_watermark(
        image_path=image_path,
        output_path=output_path,
        strength=strength,
        vendor=vendor_for_strength(image_path),
    )
