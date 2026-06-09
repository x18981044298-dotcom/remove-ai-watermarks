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

Architecture (vs PhotoMaker-V2):
- PhotoMaker-V2 conditions on a CLIP+ArcFace embedding and runs as txt2img with
  no spatial control. Identity drift on Asian male faces is documented upstream
  and was visually confirmed in our cert sweep.
- InstantID conditions on the ArcFace embedding via cross-attention (IP-Adapter
  style) AND uses a separate landmark ControlNet (5 facial keypoints) for weak
  pose control. The semantic identity branch and spatial landmark branch are
  decoupled, which gives stronger identity fidelity per the InstantID paper
  (arXiv:2401.07519) and our research report. Critically, NO original face
  pixels enter the diffusion -- only the ArcFace embedding (semantic) and the
  rendered landmark stick figure (geometry, content-free) -- so SynthID is not
  transported.

Pipeline this module wires:
  1. Detect faces in the CLEANED image (YuNet via ``auto_config``).
  2. For each face: take the SAME box from the ORIGINAL image, extract its
     ArcFace embedding + 5 keypoints via InsightFace ``FaceAnalysis(antelopev2)``.
  3. Render the keypoints as a stick figure (``draw_kps`` from upstream).
  4. Call the InstantID community pipeline
     (``StableDiffusionXLInstantIDPipeline``) with the ArcFace embedding as
     ``image_embeds=`` and the landmark image as ``image=`` (the ControlNet
     conditioning).
  5. Feather-composite the regenerated face into the cleaned image.

Requires the optional ``instantid`` extra: ``pip install
'remove-ai-watermarks[instantid]'``. Weights download on first use; never
bundled. The InstantID adapter weights (IdentityNet ControlNet +
``ip-adapter.bin``) are Apache-2.0; the runtime InsightFace ``antelopev2`` model
pack is non-commercial.

Multi-face: like PhotoMaker, this module loops over face boxes and composites
back. InstantID's strength is single-portrait; for group photos identity
fidelity per-face is preserved but the composite still uses the cleaned-image
geometry as the canvas.
"""

# cv2/torch/diffusers boundary: relax unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import importlib.util
import logging
import threading
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks.photomaker_restore import _composite_faces, _face_crop_square

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


def _get_face_analyser() -> Any:
    """Return the InsightFace FaceAnalysis singleton (antelopev2, non-commercial).

    Triggers InsightFace's auto-download of the antelopev2 pack on first
    instantiation. See the NON-COMMERCIAL notice at the top of the module.
    """
    global _face_analyser
    if _face_analyser is not None:
        return _face_analyser
    with _face_analyser_lock:
        if _face_analyser is None:
            import torch
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            # InstantID's upstream uses name='antelopev2' and root='./' (which puts
            # the auto-downloaded pack under ./models/antelopev2/). Use the same root
            # so the pack lands under the process cwd (Modal volume in prod).
            fa = FaceAnalysis(name="antelopev2", root="./", providers=providers)
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
            logger.info("instantid_restore: loading SDXL+InstantID on %s (%s)", device, dtype)

            # IdentityNet ControlNet weights.
            controlnet = ControlNetModel.from_pretrained(
                _INSTANTID_REPO,
                subfolder=_INSTANTID_CONTROLNET_SUBFOLDER,
                torch_dtype=dtype,
            )
            # SDXL base + InstantID community pipeline (txt2img w/ IdentityNet ControlNet
            # + IP-Adapter cross-attention conditioned on the ArcFace embedding).
            pipe = DiffusionPipeline.from_pretrained(
                _SDXL_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                custom_pipeline="pipeline_stable_diffusion_xl_instantid",
            )
            pipe.to(device)
            # IP-Adapter weights that wire the ArcFace embedding into cross-attention.
            ip_adapter_path = hf_hub_download(repo_id=_INSTANTID_REPO, filename=_INSTANTID_IP_ADAPTER)
            pipe.load_ip_adapter_instantid(ip_adapter_path)
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
    ip_adapter_scale: float = 0.8,
    controlnet_conditioning_scale: float = 0.8,
    seed: int | None = None,
    detect_faces_fn: Any | None = None,
) -> NDArray[Any]:
    """SynthID-robust face identity restoration via InstantID.

    Flow:
      1. Detect faces in ``cleaned_bgr`` (YuNet via ``auto_config`` by default;
         override via ``detect_faces_fn`` for tests).
      2. For each face: take the SAME box from ``original_bgr`` -> square crop ->
         InsightFace extracts ArcFace embedding + 5 keypoints -> ``_draw_kps``
         renders the landmark stick figure -> InstantID pipeline generates a
         fresh face conditioned on the embedding and the landmark control image.
      3. Feather-composite each regenerated face into ``cleaned_bgr``.

    Faces are read from ``original_bgr`` for the ArcFace embedding + landmarks, but
    the OUTPUT pixels are diffusion-fresh (ArcFace embedding is semantic; landmark
    image is pure geometry), so SynthID is not transported.

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

    restored: list[tuple[NDArray[Any], tuple[int, int, int, int]]] = []
    for box in boxes:
        id_crop_bgr, square_box = _face_crop_square(original_bgr, box)
        if id_crop_bgr.size == 0:
            continue

        # Resize the crop to the InstantID target so InsightFace + the pipeline both
        # work in the same coordinate space.
        crop_resized = cv2.resize(
            id_crop_bgr, (_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )

        # InsightFace expects BGR. It returns embedding + 5 keypoints per detected face.
        # Pick the largest face in the crop (sorted by bbox area).
        face_infos = face_analyser.get(crop_resized)
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

        out = pipeline(
            prompt=_INSTANTID_PROMPT,
            negative_prompt=_INSTANTID_NEGATIVE,
            image_embeds=face_emb,
            image=landmark_img,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        gen_rgb = out.images[0]
        gen_bgr = cv2.cvtColor(np.array(gen_rgb), cv2.COLOR_RGB2BGR)
        restored.append((gen_bgr, square_box))

    if not restored:
        return cleaned_bgr
    return _composite_faces(cleaned_bgr, restored)
