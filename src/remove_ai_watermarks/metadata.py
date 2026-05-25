"""AI metadata detection and removal.

Wraps the noai-watermark metadata handling for stripping AI-generation
metadata (EXIF, PNG text chunks, C2PA provenance) from images.

For metadata-only operations, the heavy ML dependencies are NOT required.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ── Known AI metadata keys ──────────────────────────────────────────

AI_METADATA_KEYS: frozenset[str] = frozenset(
    k.lower()
    for k in [
        "parameters",
        "prompt",
        "negative_prompt",
        "workflow",
        "comfyui",
        "sd-metadata",
        "invokeai_metadata",
        "generation_data",
        "ai_metadata",
        "dream",
        "sd:prompt",
        "sd:negative_prompt",
        "sd:seed",
        "sd:steps",
        "sd:sampler",
        "sd:cfg_scale",
        "sd:model_hash",
        "c2pa",
        "c2pa_chunk",
        "Software",
    ]
)

AI_KEYWORDS: tuple[str, ...] = (
    "stable_diffusion",
    "comfyui",
    "automatic1111",
    "invokeai",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "synthid",
    "google_ai",
    "openai",
    "c2pa",
)

# C2PA UUID used in ISOBMFF (AVIF, HEIF, MP4) ``uuid`` boxes.
# Reference: https://spec.c2pa.org/specifications/specifications/2.1/specs/C2PA_Specification.html
C2PA_UUID: bytes = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")

# IPTC ``digitalSourceType`` values (IPTC 2025.1) that flag AI provenance.
# Used by Instagram, Facebook, X (Twitter) to show "Made with AI" labels.
IPTC_AI_MARKERS: tuple[bytes, ...] = (
    b"trainedAlgorithmicMedia",
    b"compositeSynthetic",
    b"algorithmicMedia",
    b"compositeWithTrainedAlgorithmicMedia",
)

# China's mandatory AI-content labeling (TC260, the national cybersecurity
# standards committee). AI generators serving China embed an XMP block in the
# TC260 namespace -- ``<TC260:AIGC>{"Label":"1",...}``. Doubao (ByteDance) uses
# this; the same standard is mandatory for Jimeng, Kling, Qwen, Ernie, etc.,
# so the marker covers the whole China-AIGC-labeled ecosystem. Container-
# agnostic (XMP is text), so a raw-byte scan catches it in PNG/JPEG/etc.
AIGC_MARKERS: tuple[bytes, ...] = (
    b"tc260.org.cn/ns/AIGC",
    b"TC260:AIGC",
)

STANDARD_METADATA_KEYS: frozenset[str] = frozenset(
    [
        "Author",
        "Title",
        "Description",
        "Copyright",
        "Creation Time",
        "Software",
        "Comment",
        "Disclaimer",
        "Source",
        "Warning",
    ]
)


def _is_ai_key(key: str) -> bool:
    """Check if a metadata key is AI-related."""
    key_lower = key.lower()
    if key_lower in AI_METADATA_KEYS:
        return True
    return any(kw in key_lower for kw in AI_KEYWORDS)


def has_ai_metadata(image_path: Path) -> bool:
    """Check if an image contains AI-generation metadata.

    Args:
        image_path: Path to the image.

    Returns:
        True if AI metadata is detected.
    """
    from PIL import Image

    # PIL may not handle AVIF/HEIF/JPEG-XL without the optional plugins
    # (ultralytics also monkey-patches Image.open in a way that can raise
    # ModuleNotFoundError when pi_heif autoload fails), so any open failure
    # falls through to the binary scan.
    try:
        with Image.open(image_path) as img:
            for key in img.info:
                if _is_ai_key(key):
                    return True
    except Exception as exc:
        logger.debug("PIL could not open %s for metadata scan: %s", image_path, exc)

    # Check C2PA — via the official ``c2pa`` lib if available, otherwise via a
    # binary scan that also catches AVIF/HEIF/JPEG-XL containers (PIL doesn't
    # expose their metadata uniformly).
    try:
        from c2pa import has_c2pa_metadata

        if has_c2pa_metadata(image_path):
            return True
    except ImportError:
        pass

    # Binary scan covers C2PA (PNG caBX, JPEG APP11, AVIF/HEIF/JXL uuid boxes)
    # and IPTC AI markers in XMP. Read only the first 512KB to bound memory.
    with open(image_path, "rb") as f:
        data = f.read(512 * 1024)
    if b"c2pa" in data.lower() or b"C2PA" in data:
        return True
    if C2PA_UUID in data:
        return True
    if any(marker in data for marker in AIGC_MARKERS):
        return True
    return any(marker in data for marker in IPTC_AI_MARKERS)


def aigc_label(image_path: Path) -> dict[str, str] | None:
    """Parse a China TC260 ``<TC260:AIGC>`` AI-labeling block, if present.

    Returns the decoded JSON (e.g. ``{"Label": "1", "ContentProducer": ...}``)
    or None. The block is XMP text (HTML-entity encoded), so it is found by a
    container-agnostic raw-byte scan and works for PNG/JPEG/WebP alike.
    """
    import html
    import json
    import re

    with open(image_path, "rb") as f:
        data = f.read(1024 * 1024)
    match = re.search(rb"<TC260:AIGC>(.*?)</TC260:AIGC>", data, re.DOTALL)
    if not match:
        return None
    raw = html.unescape(match.group(1).decode("utf-8", "replace"))
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else None


def synthid_source(image_path: Path) -> str | None:
    """Return the vendor name(s) if the image carries a SynthID pixel watermark.

    This is a *metadata-based* proxy: Google (Imagen/Gemini) and OpenAI
    (ChatGPT/DALL-E/gpt-image) embed an invisible SynthID watermark alongside
    a C2PA manifest, so a C2PA manifest signed by one of them on AI-generated
    content implies SynthID in the pixels. Adobe Firefly / Microsoft Designer
    sign C2PA but do not use SynthID, so they return None.

    The verdict is reliable only while the C2PA manifest is intact -- absence
    is not proof, because C2PA can be stripped while the pixel watermark
    survives, and the pixel watermark itself is not locally detectable
    (proprietary decoder).

    Args:
        image_path: Path to the image (PNG, JPEG, WebP, or ISOBMFF container).

    Returns:
        Comma-joined vendor name(s) (e.g. ``"OpenAI"``) or None.
    """
    from remove_ai_watermarks.noai.c2pa import extract_c2pa_info, synthid_vendors_in

    # PNG: the caBX chunk parser gives a clean, structured issuer.
    vendors = extract_c2pa_info(image_path).get("synthid_vendors")
    if vendors:
        return ", ".join(vendors)

    # Non-PNG containers (JPEG APP11, WebP, AVIF/HEIF/JXL uuid box) keep the
    # C2PA manifest where the PNG parser can't reach it. Binary-scan for the
    # same signal: a C2PA manifest from a SynthID-using issuer on AI content.
    with open(image_path, "rb") as f:
        data = f.read(1024 * 1024)
    has_c2pa = b"c2pa" in data.lower() or C2PA_UUID in data
    # Matches both "trainedAlgorithmicMedia" and "compositeWithTrainedAlgorithmicMedia".
    ai_source = b"trainedAlgorithmicMedia" in data or b"TrainedAlgorithmicMedia" in data
    if not (has_c2pa and ai_source):
        return None
    matched = synthid_vendors_in(data)
    return ", ".join(matched) if matched else None


def exif_generator(image_path: Path) -> str | None:
    """Return an AI-generator name from the EXIF ``Software`` / XMP ``CreatorTool``
    field, if it matches a known generator (see ``AI_GENERATOR_TOKENS``), else None.

    Cross-format: EXIF is read via PIL + piexif for any container PIL can open
    (JPEG/WebP/AVIF/PNG); an XMP ``CreatorTool`` raw-byte scan additionally covers
    HEIF/JPEG-XL that PIL can't open without plugins. Only AI tokens match, so
    ordinary editors (plain "Adobe Photoshop", "GIMP") are not flagged.
    """
    import re

    from remove_ai_watermarks.noai.constants import AI_GENERATOR_TOKENS

    candidates: list[str] = []

    # EXIF Software / Artist / ImageDescription (0th IFD) via PIL exif bytes.
    try:
        import piexif
        from PIL import Image

        with Image.open(image_path) as img:
            exif_bytes = img.info.get("exif")
        if exif_bytes:
            tags = piexif.load(exif_bytes).get("0th", {})
            # Make catches camera-style tags AI tools reuse (Ideogram writes
            # Make="Ideogram AI"); real cameras put "Apple"/"Canon" there, which
            # carry no AI token, so this stays low-false-positive.
            for tag in (
                piexif.ImageIFD.Software,
                piexif.ImageIFD.Make,
                piexif.ImageIFD.Artist,
                piexif.ImageIFD.ImageDescription,
            ):
                value = tags.get(tag)
                if isinstance(value, bytes):
                    candidates.append(value.decode("latin1", "replace"))
    except Exception as exc:  # unopenable format / malformed EXIF
        logger.debug("EXIF generator read failed for %s: %s", image_path, exc)

    # XMP CreatorTool: text, container-agnostic (covers HEIF/JXL via raw scan).
    try:
        with open(image_path, "rb") as f:
            head = f.read(1024 * 1024)
        for match in re.finditer(rb"CreatorTool[>\"'=\s]{1,4}([^<\"']{1,80})", head):
            candidates.append(match.group(1).decode("latin1", "replace"))
    except Exception as exc:
        logger.debug("XMP CreatorTool scan failed for %s: %s", image_path, exc)

    for value in candidates:
        if any(token in value.lower() for token in AI_GENERATOR_TOKENS):
            return value.strip()
    return None


def get_ai_metadata(image_path: Path) -> dict[str, str]:
    """Extract AI-related metadata from an image.

    Args:
        image_path: Path to the image.

    Returns:
        Dictionary of AI metadata key-value pairs.
    """
    from PIL import Image

    from remove_ai_watermarks.noai.c2pa import extract_c2pa_info, synthid_verdict

    result: dict[str, str] = {}

    # PIL may not open AVIF/HEIF/JPEG-XL without optional plugins (and
    # ultralytics' Image.open patch can raise ModuleNotFoundError); fall through
    # to the C2PA/binary path on any open failure. See CLAUDE.md.
    try:
        with Image.open(image_path) as img:
            for key, value in img.info.items():
                if _is_ai_key(key):
                    if isinstance(value, bytes):
                        result[key] = f"<binary {len(value)} bytes>"
                    elif isinstance(value, str) and len(value) > 200:
                        result[key] = value[:200] + "…"
                    else:
                        result[key] = str(value)
    except Exception as exc:
        logger.debug("PIL could not open %s for AI-metadata scan: %s", image_path, exc)

    # C2PA manifest fields from the single canonical parser (noai/c2pa.py).
    c2pa = extract_c2pa_info(image_path)
    for key in (
        "c2pa_manifest",
        "claim_generator",
        "c2pa_spec",
        "issuer",
        "source_type",
        "actions",
        "synthid_watermark",
    ):
        if key in c2pa:
            result.setdefault(key, str(c2pa[key]))

    # Non-PNG containers (JPEG/WebP/AVIF): extract_c2pa_info is PNG-only, so
    # fall back to the format-agnostic source check for the SynthID verdict.
    if "synthid_watermark" not in result and (vendor := synthid_source(image_path)):
        result.setdefault("synthid_watermark", synthid_verdict(vendor))

    # China TC260 AI-content label (Doubao and other China-served generators).
    if aigc := aigc_label(image_path):
        producer = aigc.get("ContentProducer", "")
        result["aigc_label"] = f"China AIGC label (TC260){f'; producer {producer}' if producer else ''}"
    return result


def remove_ai_metadata(
    source_path: Path,
    output_path: Path | None = None,
    keep_standard: bool = True,
) -> Path:
    """Remove AI-generation metadata from an image.

    Strips EXIF AI tags, PNG text chunks, and C2PA provenance manifests
    while optionally preserving standard metadata (Author, Title, etc.).

    Args:
        source_path: Path to the source image.
        output_path: Output path (None = overwrite source).
        keep_standard: If True, preserve standard metadata fields.

    Returns:
        Path to the cleaned image.
    """
    import piexif
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    if output_path is None:
        output_path = source_path

    # AVIF/HEIF/JPEG-XL: strip C2PA boxes at the container level without
    # re-encoding. Avoids needing PIL plugins (pillow-heif / pillow-jxl) and
    # preserves pixel data bit-for-bit.
    if source_path.suffix.lower() in (".avif", ".heif", ".heic", ".jxl"):
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        data = source_path.read_bytes()
        cleaned, stripped = strip_c2pa_boxes(data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(cleaned)
        logger.info("Stripped %d C2PA box(es) → %s", stripped, output_path)
        return output_path

    # Read image and filter metadata
    with Image.open(source_path) as img:
        img = img.copy()
        fmt = output_path.suffix.lower()

        save_kwargs: dict = {}
        if fmt in (".jpg", ".jpeg"):
            save_kwargs["format"] = "JPEG"
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        else:
            save_kwargs["format"] = "PNG"

        # Collect non-AI metadata
        kept_meta: dict[str, str] = {}
        exif_data = None

        for key, value in img.info.items():
            if _is_ai_key(key):
                continue
            if key == "exif":
                with contextlib.suppress(Exception):
                    exif_data = piexif.load(value)
                continue
            if key in ("dpi", "gamma"):
                save_kwargs[key] = value
                continue
            if keep_standard and key in STANDARD_METADATA_KEYS:
                kept_meta[key] = str(value) if not isinstance(value, str) else value

        # Apply cleaned metadata
        if save_kwargs["format"] == "PNG" and kept_meta:
            pnginfo = PngInfo()
            for k, v in kept_meta.items():
                pnginfo.add_text(k, v)
            save_kwargs["pnginfo"] = pnginfo

        if exif_data and save_kwargs["format"] == "JPEG":
            with contextlib.suppress(Exception):
                save_kwargs["exif"] = piexif.dump(exif_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, **save_kwargs)

    logger.info("Stripped AI metadata → %s", output_path)
    return output_path
