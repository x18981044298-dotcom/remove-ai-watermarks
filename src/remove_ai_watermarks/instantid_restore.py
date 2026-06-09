"""SynthID-robust face identity restoration via InstantID.

**NON-COMMERCIAL.** InstantID's runtime depends on the InsightFace ``antelopev2``
ArcFace model pack, which InsightFace releases under a research-only license:

    "The training data containing the annotation (and the models trained with
     these data) are available for non-commercial research purposes only."
                                                  -- insightface upstream README

The InstantX maintainers themselves acknowledged on HuggingFace
(``InstantX/InstantID`` discussion #2) that "InstantID cannot be Apache 2.0 if it
is using Insight Face" and stated intent to retrain on commercial face encoders.
As of 2026-06-08 (deep-research synthesis in
``docs/synthid-robust-identity-research-2026-06-08.md``) that retrain has not
shipped. **A paid service (raiw.cc, any monetized SaaS) MUST NOT use this path.**

The default ``--restore-faces-method`` is ``instantid`` (this module). The
alternative ``photomaker`` is also non-commercial. There is no commercial-safe
ArcFace-grade identity-preservation stack for SDXL today.

Architecture (vs the earlier txt2img variant):
- The earlier (txt2img) integration generated each face from scratch in a fresh
  1024 scene with InstantID's standard pipeline. That produced studio-portrait
  faces with the wrong lighting / head angle for the surrounding scene; on
  group photos the per-face composites read as patchwork even after color
  matching and elliptical alphas.
- This (img2img on cleaned) integration feeds the CLEANED face crop as the
  img2img source. Diffusion sees the scene context (shoulders, hair edges,
  lighting, shadow direction) directly and harmonises the regenerated face
  with it. Identity still comes through the ArcFace embedding +
  landmark-ControlNet, which are semantic / pure-geometry and carry no
  watermark.

SynthID safety (load-bearing for raiw.cc):
- img2img source = CLEANED crop. Cleaned image is already oracle-verified
  SynthID-free at our controlnet strength; cropping is a subset operation that
  preserves that property.
- ArcFace embedding = from the ORIGINAL face crop (sharper identity, but the
  embedding is semantic 512-d, no pixel content).
- Landmark stick figure = pure colour-coded geometry rendered from kps; no
  source pixels.
- img2img diffusion adds noise to the cleaned source then denoises with
  ControlNet + IP-Adapter conditioning. Any residual high-frequency pattern
  in the cleaned crop is destroyed by that noise injection at the strengths we
  use.
- We must NEVER feed the original image as img2img source (would re-introduce
  SynthID outside the diffusion footprint at strength < 1). The code only ever
  reads pixels from ``cleaned_bgr`` into ``image=`` -- the original is used
  for the embedding + kps only.

Pipeline this module wires:
  1. Detect faces in the CLEANED image (YuNet via ``auto_config``).
  2. For each face: square-crop the SAME box from BOTH the original (for
     ArcFace + kps) and the cleaned image (for img2img source). Resize both
     to 1024x1024.
  3. Render the kps as a stick figure (the ControlNet conditioning image).
  4. Call the InstantID img2img pipeline
     (``StableDiffusionXLInstantIDImg2ImgPipeline``) with ``image`` = cleaned
     crop, ``control_image`` = landmark, ``image_embeds`` = ArcFace, and
     ``strength`` = ~0.55. The output 1024 is a face that fits the scene.
  5. Elliptical-alpha + colour-match composite into the cleaned image.

Requires the optional ``instantid`` extra: ``pip install
'remove-ai-watermarks[instantid]'``. Weights download on first use; the
upstream img2img pipeline file (not on PyPI) is cached from
``raw.githubusercontent.com`` on first run.
"""

# cv2/torch/diffusers boundary: relax unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks.photomaker_restore import _face_crop_square

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# InstantID checkpoint repo on HuggingFace. The IdentityNet ControlNet weights live
# under ``ControlNetModel/`` and the IP-Adapter file is ``ip-adapter.bin`` at the
# root. Both are Apache-2.0 (the InsightFace runtime dep is what makes the path
# non-commercial). Downloaded on first use.
_INSTANTID_REPO = "InstantX/InstantID"
_INSTANTID_CONTROLNET_SUBFOLDER = "ControlNetModel"
_INSTANTID_IP_ADAPTER = "ip-adapter.bin"

# Upstream InstantID img2img pipeline source. Not on PyPI, not on HF Hub at any path
# diffusers can auto-load -- the file lives in the InstantID GitHub repo. We download
# it once to a cache dir and pass it as ``custom_pipeline=<path>`` to diffusers.
_INSTANTID_IMG2IMG_URL = (
    "https://raw.githubusercontent.com/instantX-research/InstantID/"
    "main/pipeline_stable_diffusion_xl_instantid_img2img.py"
)

# SDXL base shared with the main pipeline (same checkpoint as `default`/`controlnet`).
_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Prompt format. InstantID is less sensitive to prompt than PhotoMaker because the
# ID branch is cross-attention; a neutral descriptive prompt is recommended by the
# upstream gradio demo.
_INSTANTID_PROMPT = "portrait photo of a person, natural skin, soft lighting, sharp focus, best quality"
_INSTANTID_NEGATIVE = (
    "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, "
    "cartoons, sketch), open mouth, blurry, watermark, deformed"
)

# Square size used to feed InstantID. SDXL is happiest at 1024 (a smaller value sends
# it into low-res mosaic mode -- caught visually on PhotoMaker, same root cause).
_INSTANTID_FACE_SIZE = 1024

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()
_face_analyser: Any | None = None
_face_analyser_lock = threading.Lock()


def is_available() -> bool:
    """True when the optional InstantID extra deps are importable."""
    return (
        importlib.util.find_spec("insightface") is not None
        and importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("huggingface_hub") is not None
    )


def _select_device() -> str:
    """Pick the InstantID pipeline device: CUDA when present, MPS on Apple, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug("instantid_restore: device probe failed (%s); using CPU", e)
    return "cpu"


def _fetch_img2img_pipeline_file() -> Path:
    """Cache the InstantID img2img pipeline source file locally on first use.

    The file lives in the InstantX GitHub repo (not on PyPI, not on HF Hub at any
    path diffusers can auto-load). We fetch the raw URL once into the package's
    HuggingFace cache so subsequent loads hit disk. Returns the path to feed to
    ``DiffusionPipeline.from_pretrained(custom_pipeline=...)``.
    """
    import os
    import urllib.request

    cache_root = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    cache_dir = cache_root / "remove_ai_watermarks" / "instantid"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "pipeline_stable_diffusion_xl_instantid_img2img.py"
    if not target.exists() or target.stat().st_size < 50_000:
        logger.info("instantid_restore: fetching img2img pipeline source from %s", _INSTANTID_IMG2IMG_URL)
        urllib.request.urlretrieve(_INSTANTID_IMG2IMG_URL, target)  # noqa: S310 (HTTPS pinned)
    return target


def _ensure_antelopev2(root: Path) -> Path:
    """Materialize the antelopev2 pack at ``<root>/models/antelopev2/`` if absent.

    InsightFace's built-in auto-download points at
    ``github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip``
    which has been broken since at least 2024 (verified upstream issue #2517,
    #2766; explicitly called out in InstantID's README: "manually download via
    this URL to models/antelopev2 as the default link is invalid"). Without the
    five expected ``.onnx`` files in place, ``FaceAnalysis.prepare()`` errors
    with ``assert 'detection' in self.models``.

    We side-step the broken default by fetching the five files from a HuggingFace
    mirror (``kidyu/antelopev2-for-InstantID-ComfyUI``) on first use. Returns the
    target directory containing the .onnx files.
    """
    from huggingface_hub import hf_hub_download

    target = root / "models" / "antelopev2"
    target.mkdir(parents=True, exist_ok=True)
    files = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]
    for fname in files:
        dest = target / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("instantid_restore: fetching antelopev2/%s from HF mirror", fname)
        path = hf_hub_download(repo_id="kidyu/antelopev2-for-InstantID-ComfyUI", filename=fname)
        # hf_hub_download caches under HF_HOME; symlink (or copy) into the
        # InsightFace-expected layout.
        if not dest.exists():
            try:
                dest.symlink_to(path)
            except OSError:
                import shutil

                shutil.copy(path, dest)
    return target


def _get_face_analyser() -> Any:
    """Return the InsightFace FaceAnalysis singleton (antelopev2, non-commercial).

    Pre-downloads the antelopev2 pack from a HuggingFace mirror (the InsightFace
    auto-download is broken). See the NON-COMMERCIAL notice at the top of the
    module.
    """
    global _face_analyser
    if _face_analyser is not None:
        return _face_analyser
    with _face_analyser_lock:
        if _face_analyser is None:
            import torch
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            # InstantID's upstream uses name='antelopev2' and root='./'. Materialise
            # the pack at the same place so FaceAnalysis finds it locally.
            root = Path.cwd()
            _ensure_antelopev2(root)
            fa = FaceAnalysis(name="antelopev2", root=str(root), providers=providers)
            fa.prepare(ctx_id=0, det_size=(640, 640))
            _face_analyser = fa
    return _face_analyser


def _get_pipeline() -> Any:
    """Return the lazily-built InstantID pipeline singleton (downloads weights on first use).

    Loads via diffusers' community-pipeline mechanism: the file
    ``pipeline_stable_diffusion_xl_instantid.py`` lives in
    ``diffusers/examples/community/`` and is selected by the slug
    ``pipeline_stable_diffusion_xl_instantid``.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            import torch
            from diffusers import ControlNetModel, DiffusionPipeline
            from huggingface_hub import hf_hub_download

            device = _select_device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("instantid_restore: loading SDXL+InstantID img2img on %s (%s)", device, dtype)

            # IdentityNet ControlNet weights.
            controlnet = ControlNetModel.from_pretrained(
                _INSTANTID_REPO,
                subfolder=_INSTANTID_CONTROLNET_SUBFOLDER,
                torch_dtype=dtype,
            )
            # Upstream InstantID img2img pipeline (StableDiffusionXLInstantIDImg2ImgPipeline).
            # Lets us feed the cleaned face crop as the diffusion source so the regenerated
            # face inherits scene lighting / shadows / head angle from the cleaned context
            # (vs the txt2img variant which generates a studio portrait from scratch).
            # Critical SynthID-safety property: the ``image`` arg MUST be the CLEANED crop,
            # never the original -- the original carries the watermark and img2img at
            # strength < 1 preserves some input pixel structure. The ArcFace embedding is
            # semantic (no pixel content), so taking it from the original is fine.
            pipe = DiffusionPipeline.from_pretrained(
                _SDXL_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                custom_pipeline=str(_fetch_img2img_pipeline_file()),
            )
            pipe.to(device)
            # IP-Adapter weights that wire the ArcFace embedding into cross-attention.
            ip_adapter_path = hf_hub_download(repo_id=_INSTANTID_REPO, filename=_INSTANTID_IP_ADAPTER)
            # IP-Adapter scale (the weight on the ArcFace cross-attention branch) is
            # set at load time, not at call time. 0.8 mirrors the upstream demo.
            pipe.load_ip_adapter_instantid(ip_adapter_path, scale=0.8)
            # Diffusers 0.38 vs InstantID upstream compat patch: InstantID's __call__
            # calls ``self.check_inputs(...)`` POSITIONALLY (signature from ~v0.29),
            # but diffusers 0.38 added two new params (``ip_adapter_image``,
            # ``ip_adapter_image_embeds``) BEFORE ``controlnet_conditioning_scale`` in
            # the parent's signature. That shifts every argument by two, so
            # ``control_guidance_end`` (which InstantID converts to ``[1.0]`` for the
            # single-controlnet case before this point) lands in the slot the parent
            # validates as ``controlnet_conditioning_scale`` and trips
            # ``TypeError("must be type float")``. Our inputs are programmatic and
            # already validated by our own callers, so neutralising the check is safe.
            pipe.check_inputs = lambda *_a, **_k: None
            _pipeline = pipe
    return _pipeline


def _draw_kps(image_size: tuple[int, int], kps: Any) -> Any:
    """Render the 5 facial keypoints as a colored stick figure.

    Mirrors upstream's ``draw_kps`` (in ``pipeline_stable_diffusion_xl_instantid.py``):
    the 5 keypoints (left eye, right eye, nose tip, left mouth corner, right mouth
    corner) get drawn as colored circles connected by colored lines, on a black
    background. The result is the ControlNet conditioning image -- pure landmark
    geometry, no pixels from the original face leak through this branch.

    ``image_size`` is ``(width, height)``; ``kps`` is a numpy array of shape (5, 2).
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Same color palette as upstream (blue/red/green/purple/yellow).
    stick_width = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    color_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    w, h = image_size
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

    kps_arr = np.array(kps)
    for i in range(len(limb_seq)):
        index = limb_seq[i]
        color = color_list[index[0]]
        x = kps_arr[index][:, 0]
        y = kps_arr[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stick_width),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for i, kp in enumerate(kps_arr):
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color_list[i], -1)

    return Image.fromarray(out_img.astype(np.uint8))


def restore_faces_instantid(
    original_bgr: NDArray[Any],
    cleaned_bgr: NDArray[Any],
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    controlnet_conditioning_scale: float = 0.8,
    img2img_strength: float = 0.55,
    seed: int | None = None,
    detect_faces_fn: Any | None = None,
) -> NDArray[Any]:
    """SynthID-robust face identity restoration via InstantID.

    Flow:
      1. Detect faces in ``cleaned_bgr`` (YuNet via ``auto_config`` by default;
         override via ``detect_faces_fn`` for tests).
      2. For each face: square-crop the SAME box from BOTH images (original ->
         ArcFace + kps; cleaned -> img2img source). Resize both to 1024.
      3. Render kps as a landmark stick figure (the ControlNet conditioning).
      4. Run InstantID img2img: ``image`` = cleaned crop, ``control_image`` =
         landmark, ``image_embeds`` = ArcFace embedding from the original.
      5. Elliptical-alpha + colour-match composite into the cleaned image.

    SynthID safety: ``image`` is the CLEANED crop (already oracle-clean); the
    original is read for the embedding and kps only (semantic / geometry, no
    pixel content). See the module docstring.

    ``detect_faces_fn`` returns a list of ``(x, y, w, h)`` boxes given a BGR image.
    """
    import cv2
    import numpy as np
    import torch

    if detect_faces_fn is None:
        from pathlib import Path

        from remove_ai_watermarks import auto_config as _ac

        def _default_detect(bgr: NDArray[Any]) -> list[tuple[int, int, int, int]]:
            h_d, w_d = bgr.shape[:2]
            model = Path(_ac.__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
            det = cv2.FaceDetectorYN.create(str(model), "", (w_d, h_d), _ac._FACE_SCORE, 0.3, 5000)
            det.setInputSize((w_d, h_d))
            _, faces = det.detect(bgr)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces if int(f[2]) > 0 and int(f[3]) > 0]

        detect_faces_fn = _default_detect

    boxes = detect_faces_fn(cleaned_bgr)
    if not boxes:
        logger.debug("instantid_restore: no faces detected; returning cleaned image unchanged")
        return cleaned_bgr

    pipeline = _get_pipeline()
    face_analyser = _get_face_analyser()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    h_c, w_c = cleaned_bgr.shape[:2]
    restored: list[tuple[NDArray[Any], tuple[int, int, int, int]]] = []
    for box in boxes:
        # Square crop with the SAME geometry from both the original (-> ArcFace
        # embedding + landmark kps -- semantic / pure-geometry, SynthID can't ride
        # either) AND the cleaned image (-> img2img source -- SynthID-safe because
        # the cleaned image is already oracle-verified clean and any residual
        # high-frequency pattern would be destroyed by the noise injection at our
        # strength setting). _face_crop_square gives a 2x-padded square box around
        # the face -- enough scene context so the img2img harmonises lighting and
        # head angle with the surroundings.
        original_crop_bgr, square_box = _face_crop_square(original_bgr, box)
        sx1, sy1, sx2, sy2 = square_box
        sx1c, sy1c = max(0, sx1), max(0, sy1)
        sx2c, sy2c = min(w_c, sx2), min(h_c, sy2)
        if original_crop_bgr.size == 0 or sx2c <= sx1c or sy2c <= sy1c:
            continue
        cleaned_crop_bgr = cleaned_bgr[sy1c:sy2c, sx1c:sx2c]
        if cleaned_crop_bgr.shape[:2] != original_crop_bgr.shape[:2]:
            # Edge effect at image border -- pad cleaned crop to match the original
            # crop dimensions so InsightFace / the pipeline see the same shape.
            cleaned_crop_bgr = cv2.resize(
                cleaned_crop_bgr,
                (original_crop_bgr.shape[1], original_crop_bgr.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Resize both crops to the SDXL working size.
        original_resized = cv2.resize(
            original_crop_bgr, (_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )
        cleaned_resized = cv2.resize(
            cleaned_crop_bgr, (_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )

        # ArcFace embedding + 5 kps from the ORIGINAL face (sharper identity).
        face_infos = face_analyser.get(original_resized)
        if not face_infos:
            logger.debug("instantid_restore: InsightFace did not find a face in the crop; skipping")
            continue
        face_info = sorted(
            face_infos,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        face_emb = face_info["embedding"]
        face_kps = face_info["kps"]

        # Render the landmark stick figure at the same size as the generation target.
        landmark_img = _draw_kps((_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), face_kps)

        # img2img call: source = CLEANED crop (SynthID-safe), control = landmark
        # geometry, identity = ArcFace embedding from original. Strength controls
        # how much of the cleaned input structure survives -- low enough (~0.55)
        # to keep the head angle / lighting / shoulders coherent with the rest of
        # the cleaned image, high enough that the face pixels are diffusion-fresh
        # and InstantID actually injects identity.
        from PIL import Image

        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_resized, cv2.COLOR_BGR2RGB))
        out = pipeline(
            prompt=_INSTANTID_PROMPT,
            negative_prompt=_INSTANTID_NEGATIVE,
            image=cleaned_pil,
            control_image=landmark_img,
            image_embeds=face_emb,
            strength=img2img_strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        gen_rgb = out.images[0]
        gen_bgr = cv2.cvtColor(np.array(gen_rgb), cv2.COLOR_RGB2BGR)

        # gen_bgr is at _INSTANTID_FACE_SIZE x _INSTANTID_FACE_SIZE. It represents
        # the 2x-padded square_box content as regenerated by img2img -- so the face
        # in it sits at the same RELATIVE position as in the cleaned input (img2img
        # preserves structure). Composite the whole square back into the square_box
        # location -- the cleaned-canvas elliptical alpha will keep the cleaned
        # background outside the face oval, and the img2img harmonisation handles
        # the seam INSIDE the oval (which is just face-on-face transition between
        # diffusion-output and cleaned).
        target_box = (sx1c, sy1c, sx2c, sy2c)
        gen_target = cv2.resize(gen_bgr, (sx2c - sx1c, sy2c - sy1c), interpolation=cv2.INTER_LANCZOS4)
        restored.append((gen_target, target_box))

    if not restored:
        return cleaned_bgr
    return _composite_faces_elliptical(cleaned_bgr, restored)


def _color_match(src_bgr: NDArray[Any], ref_bgr: NDArray[Any]) -> NDArray[Any]:
    """Shift ``src_bgr`` mean colour to ``ref_bgr`` mean colour, per channel.

    Each face is regenerated by InstantID with its own SDXL noise -- the white
    balance / mean tone drifts away from the surrounding scene (cool studio
    light vs warm bar lighting). A per-channel mean-shift brings the face crop
    into the same tonal range as the cleaned canvas where it lands. Contrast
    and saturation are preserved (we don't rescale variance).
    """
    import numpy as np

    src = src_bgr.astype(np.float32)
    ref = ref_bgr.astype(np.float32)
    if ref.size == 0:
        return src_bgr
    src_mean = src.mean(axis=(0, 1), keepdims=True)
    ref_mean = ref.mean(axis=(0, 1), keepdims=True)
    return np.clip(src - src_mean + ref_mean, 0, 255).astype(np.uint8)


def _composite_faces_elliptical(
    base_bgr: NDArray[Any],
    restored_crops: list[tuple[NDArray[Any], tuple[int, int, int, int]]],
    feather_div: int = 5,
) -> NDArray[Any]:
    """Composite face crops into ``base_bgr`` using an elliptical, feathered alpha.

    Two changes vs the simpler rectangular Gaussian feather:

    - **Inscribed face-shaped ellipse.** Axes are ``(0.32*bw, 0.42*bh)`` which
      fits comfortably inside the 2x padded bbox (the face naturally occupies
      the central ~50% of the bbox), covering the head silhouette without
      clipping the forehead or chin. The bbox corners (which carry
      regenerated-scene background pixels with a different tone per face) end
      up at alpha=0 so the cleaned-image background stays intact -- this is
      what eliminates multi-face patchwork on group photos.
    - **Soft feather.** ``min(bw, bh) // 5`` -- about twice as soft as the
      rectangular Gaussian, so the ellipse edge fades over a wider band into
      the cleaned canvas, hiding any residual seam.

    Additionally, before compositing, ``_color_match`` shifts the regenerated
    face's mean colour to match the cleaned canvas region it lands on -- this
    removes the warm/cool tone clash that group photos showed.
    """
    import cv2
    import numpy as np

    out = base_bgr.astype(np.float32)
    h_b, w_b = base_bgr.shape[:2]

    for crop, (x1, y1, x2, y2) in restored_crops:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_b, x2), min(h_b, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        resized = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
        # Tone match the regenerated face to the cleaned canvas it sits on.
        ref_region = base_bgr[y1:y2, x1:x2]
        resized = _color_match(resized, ref_region)

        alpha_crop = np.zeros((bh, bw), dtype=np.float32)
        center = (bw // 2, bh // 2)
        axes = (max(1, int(bw * 0.32)), max(1, int(bh * 0.42)))
        cv2.ellipse(alpha_crop, center, axes, 0, 0, 360, 1.0, -1)
        k = max(7, (min(bw, bh) // feather_div) | 1)
        alpha_crop = cv2.GaussianBlur(alpha_crop, (k, k), 0)

        alpha_full = np.zeros((h_b, w_b), dtype=np.float32)
        alpha_full[y1:y2, x1:x2] = alpha_crop
        full_restored = np.zeros_like(out)
        full_restored[y1:y2, x1:x2] = resized
        a = alpha_full[:, :, None]
        out = full_restored * a + out * (1.0 - a)

    return np.clip(out, 0, 255).astype(np.uint8)
