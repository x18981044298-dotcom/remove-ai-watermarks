"""Image provenance: identify where an image was made and what watermarks it carries.

Aggregates every locally-readable signal into a single :class:`ProvenanceReport`:

- **C2PA Content Credentials** (issuer, claim generator, digital source type) ->
  the signing platform (OpenAI, Google, Adobe, Microsoft).
- **IPTC ``digitalSourceType``** "Made with AI" marker (Meta, X, others).
- **PNG text / EXIF generation parameters** (Stable Diffusion, ComfyUI, InvokeAI).
- **SynthID metadata proxy** -- a C2PA companion from a SynthID-using vendor
  (Google / OpenAI) implies the invisible pixel watermark.
- **Visible Gemini sparkle** (optional; needs cv2/numpy, no GPU).

Hard limit: a stripped image (re-encoded, screenshotted, social-media upload)
loses all metadata, and the SynthID *pixel* watermark is not locally decodable
(proprietary decoder). Absence of signals is therefore reported as ``Unknown``,
never as "clean". See CLAUDE.md "SynthID detection is metadata-only".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from remove_ai_watermarks.metadata import (
    AI_METADATA_KEYS,
    AIGC_MARKERS,
    C2PA_UUID,
    IPTC_AI_MARKERS,
    aigc_label,
    exif_generator,
    get_ai_metadata,
)
from remove_ai_watermarks.noai.c2pa import extract_c2pa_info
from remove_ai_watermarks.noai.constants import C2PA_AI_TOOLS, C2PA_ISSUERS

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# How much of a non-PNG container to binary-scan for the C2PA issuer.
_SCAN_BYTES = 1024 * 1024

# Visible-sparkle confidence above which the signal is trusted as provenance.
# Stricter than the removal default (0.25): on the corpus, Gemini-family
# sparkles score >= 0.56 while non-sparkle images top out at 0.49, so 0.5
# cleanly separates them and avoids false positives when sparkle is the only
# signal (e.g. an OpenAI image scored 0.37 -- below threshold, correctly dropped).
_SPARKLE_THRESHOLD = 0.5

# Issuer (C2PA signer) -> human-readable generating platform. Ordered: when a
# manifest names several issuers (Microsoft Designer signs as "OpenAI,
# Microsoft"), the first match wins so the product, not the backend, is named.
_ISSUER_PLATFORM: tuple[tuple[str, str], ...] = (
    # Microsoft signs both Designer and Bing Image Creator; Bing now runs its
    # own MAI-Image model (not DALL-E), so the label stays model-neutral.
    ("Microsoft", "Microsoft (Bing Image Creator / Designer)"),
    ("Adobe", "Adobe Firefly"),
    ("OpenAI", "OpenAI (ChatGPT / gpt-image / DALL-E / Sora)"),
    ("Google", "Google (Gemini / Imagen)"),
    ("Stability AI", "Stability AI (Stable Image / DreamStudio)"),
)

# PNG-text / EXIF keys that indicate a local diffusion pipeline (vs. a hosted
# platform's C2PA). Subset of AI_METADATA_KEYS; excludes the C2PA/Software keys.
_LOCAL_GEN_KEYS = frozenset(
    AI_METADATA_KEYS & {"parameters", "prompt", "negative_prompt", "workflow", "comfyui", "invokeai_metadata", "dream"}
)

_STRIP_CAVEAT = (
    "Absence of metadata is not proof the image is clean: C2PA, EXIF, and PNG "
    "text chunks are stripped by re-encoding, screenshots, or social-media upload."
)
_SYNTHID_CAVEAT = (
    "SynthID is a metadata proxy here; the pixel watermark is not locally "
    "verifiable (proprietary decoder). Confirm via the Gemini app or openai.com/verify."
)
_OPENAI_CAVEAT = (
    "OpenAI began pairing SynthID with C2PA around 2026-05; OpenAI images from "
    "before the rollout carry C2PA without SynthID, so the SynthID verdict is 'likely'."
)
_IPTC_ONLY_CAVEAT = "The IPTC 'Made with AI' tag flags AI provenance but does not identify the specific platform."
_INVISIBLE_WM_CAVEAT = (
    "The open invisible watermark is fragile: it does not survive JPEG re-encoding "
    "or resizing, so it confirms origin only on a pristine (un-re-encoded) file."
)


@dataclass
class Signal:
    """A single provenance signal that was found (or affirmatively absent)."""

    name: str
    detail: str
    confidence: str  # "high" | "medium"


@dataclass
class ProvenanceReport:
    """Aggregated provenance verdict for one image."""

    path: Path
    is_ai_generated: bool | None  # True / False is never asserted; None = unknown
    platform: str | None
    confidence: str  # "high" | "medium" | "none"
    watermarks: list[str] = field(default_factory=list[str])
    signals: list[Signal] = field(default_factory=list["Signal"])
    caveats: list[str] = field(default_factory=list[str])


def _issuers_in(data: bytes) -> list[str]:
    """C2PA issuer names whose signature byte appears in ``data`` (binary scan)."""
    return sorted({name for sig, name in C2PA_ISSUERS.items() if sig in data})


def _ai_tools_in(data: bytes) -> list[str]:
    """Known C2PA AI-tool / generator names appearing in ``data`` (binary scan).

    PNG has a structured claim_generator; for JPEG/WebP/AVIF/HEIF/JXL the
    generator lives in a JUMBF/EXIF/XMP blob the PNG parser can't reach, so a
    byte scan recovers the same attribution (e.g. "Imagen", "DALL-E").
    """
    return sorted({name for sig, name in C2PA_AI_TOOLS.items() if sig in data})


def _attribute_platform(issuers: list[str]) -> str | None:
    """Map a set of C2PA issuer names to a human-readable generating platform."""
    joined = " ".join(issuers)
    for needle, platform in _ISSUER_PLATFORM:
        if needle in joined:
            return platform
    if issuers:  # e.g. Truepic alone -- a signing authority, not a generator
        return f"C2PA signer: {', '.join(issuers)} (no known AI generator named)"
    return None


def _visible_sparkle(image_path: Path) -> float | None:
    """Visible Gemini-sparkle confidence in [0, 1], or None if unavailable.

    Optional: needs cv2/numpy (no GPU). The cv2 work lives in gemini_engine so
    this module stays dependency-light; returns None if cv2 or the engine
    assets are missing, or the image can't be read.
    """
    try:
        from remove_ai_watermarks.gemini_engine import detect_sparkle_confidence
    except Exception as exc:  # cv2/engine assets missing
        log.debug("visible-sparkle detector unavailable: %s", exc)
        return None
    return detect_sparkle_confidence(image_path)


def _invisible_watermark(image_path: Path) -> str | None:
    """Open invisible-watermark scheme name (SD/SDXL/FLUX) or None.

    Optional: needs the imwatermark decoder (extra ``detect``). Returns None if
    it is not installed or no known watermark decodes.
    """
    from remove_ai_watermarks.invisible_watermark import detect_invisible_watermark

    return detect_invisible_watermark(image_path)


def identify(image_path: Path, *, check_visible: bool = True, check_invisible: bool = True) -> ProvenanceReport:
    """Identify an image's origin platform and watermark inventory.

    Args:
        image_path: Path to the image (PNG, JPEG, WebP, or ISOBMFF container).
        check_visible: Also run the visible Gemini-sparkle detector (cv2). Set
            False for a pure-metadata, dependency-light scan.
        check_invisible: Also decode open invisible watermarks (SD/SDXL/FLUX) via
            the optional imwatermark library. No-op when it is not installed.

    Returns:
        A :class:`ProvenanceReport`. ``is_ai_generated`` is True when any AI
        signal is found and None (unknown) when none is -- it is never asserted
        False, because stripped metadata leaves no local proof of a clean origin.
    """
    info = extract_c2pa_info(image_path)  # PNG-structured; {} for other formats
    meta = get_ai_metadata(image_path)  # PNG text + EXIF + C2PA fields + synthid

    # First MB covers C2PA (PNG caBX, JPEG APP11, AVIF/HEIF/JXL uuid box) and
    # IPTC markers for the non-PNG path where extract_c2pa_info returns {}.
    with open(image_path, "rb") as f:
        head = f.read(_SCAN_BYTES)

    signals: list[Signal] = []
    watermarks: list[str] = []
    caveats: list[str] = []

    # ── C2PA Content Credentials ────────────────────────────────────
    has_c2pa = bool(info) or b"c2pa" in head.lower() or C2PA_UUID in head
    issuers = [info["issuer"]] if info.get("issuer") else _issuers_in(head)
    platform = _attribute_platform(issuers) if has_c2pa else None
    c2pa_is_ai = "trainedAlgorithmicMedia" in info.get("source_type", "") or any(
        m in head for m in (b"trainedAlgorithmicMedia", b"compositeWithTrainedAlgorithmicMedia")
    )
    # Generator: structured for PNG, binary-scanned for other containers.
    generator = info.get("claim_generator") or (", ".join(tools) if (tools := _ai_tools_in(head)) else None)
    if has_c2pa:
        detail = ", ".join(filter(None, [", ".join(issuers), generator, info.get("source_type")]))
        signals.append(Signal("c2pa", detail or "C2PA manifest present", "high"))
        watermarks.append(f"C2PA Content Credentials ({', '.join(issuers) or 'unknown signer'})")

    # ── SynthID metadata proxy ──────────────────────────────────────
    # get_ai_metadata already sets synthid_watermark for both PNG (caBX parser)
    # and non-PNG (its own synthid_source fallback), so no extra scan is needed.
    synthid = meta.get("synthid_watermark")
    if synthid:
        watermarks.append(f"SynthID pixel watermark ({synthid})")
        caveats.append(_SYNTHID_CAVEAT)
        if "OpenAI" in (" ".join(issuers) + synthid):
            caveats.append(_OPENAI_CAVEAT)

    # ── IPTC "Made with AI" (Meta etc.), only meaningful without C2PA ─
    iptc = any(m in head for m in IPTC_AI_MARKERS)
    if iptc and not has_c2pa:
        signals.append(Signal("iptc", "digitalSourceType (Made with AI)", "high"))
        watermarks.append("IPTC digitalSourceType (Made with AI)")
        caveats.append(_IPTC_ONLY_CAVEAT)
        if platform is None:
            platform = "Made-with-AI tag (e.g. Meta AI); platform not specified"

    # ── China TC260 AIGC label (Doubao and other China-served gens) ──
    aigc = any(m in head for m in AIGC_MARKERS)
    if aigc:
        producer = (aigc_label(image_path) or {}).get("ContentProducer", "")
        signals.append(Signal("aigc", f"TC260 AIGC label{f' (producer {producer})' if producer else ''}", "high"))
        watermarks.append("China AIGC label (TC260 standard)")
        if platform is None:
            platform = "China AIGC-labeled generator (TC260; e.g. Doubao)"

    # ── Local diffusion parameters (Stable Diffusion / ComfyUI) ──────
    local_keys = sorted(k for k in meta if k.lower() in _LOCAL_GEN_KEYS)
    if local_keys:
        signals.append(Signal("gen_params", f"embedded keys: {', '.join(local_keys)}", "high"))
        watermarks.append("Embedded generation parameters (Stable Diffusion / ComfyUI)")
        if platform is None:
            platform = "Stable Diffusion / local pipeline (Automatic1111, ComfyUI, InvokeAI)"

    # ── EXIF Software / XMP CreatorTool generator (cross-format) ─────
    # Catches a generator tag (incl. inside AVIF/HEIF/JXL) when there is no C2PA.
    if generator_tag := exif_generator(image_path):
        signals.append(Signal("exif_generator", f"EXIF/XMP generator: {generator_tag}", "high"))
        watermarks.append(f"Embedded generator tag: {generator_tag}")
        if platform is None:
            platform = f"{generator_tag} (EXIF/XMP generator tag)"

    # ── Open invisible watermark (SD / SDXL / FLUX, dwtDct) ──────────
    # Public decoder, no key -- a definitive embedded signal on pristine files.
    if check_invisible and (scheme := _invisible_watermark(image_path)) is not None:
        signals.append(Signal("invisible_watermark", scheme, "high"))
        watermarks.append(f"Open invisible watermark: {scheme}")
        caveats.append(_INVISIBLE_WM_CAVEAT)
        if platform is None:
            platform = f"{scheme} (open DWT-DCT watermark)"

    # ── Verdict so far (metadata + embedded watermark) ──────────────
    invisible_wm = any(s.name == "invisible_watermark" for s in signals)
    exif_gen = any(s.name == "exif_generator" for s in signals)
    ai_from_metadata = bool(
        (has_c2pa and (c2pa_is_ai or synthid)) or iptc or aigc or local_keys or invisible_wm or exif_gen
    )

    # ── Visible Gemini sparkle (fallback for stripped-metadata case) ─
    if check_visible and (conf := _visible_sparkle(image_path)) is not None and conf >= _SPARKLE_THRESHOLD:
        signals.append(Signal("visible_sparkle", f"NCC confidence {conf:.2f}", "medium"))
        watermarks.append(f"Visible Gemini sparkle (confidence {conf:.2f})")
        if platform is None:
            platform = "Google Gemini family (visible sparkle detected)"

    visible_only = any(s.name == "visible_sparkle" for s in signals) and not ai_from_metadata

    if ai_from_metadata:
        is_ai: bool | None = True
        confidence = "high"
    elif visible_only:
        is_ai = True
        confidence = "medium"
    else:
        is_ai = None
        confidence = "none"

    caveats.append(_STRIP_CAVEAT)
    # De-duplicate while preserving order.
    caveats = list(dict.fromkeys(caveats))

    return ProvenanceReport(
        path=image_path,
        is_ai_generated=is_ai,
        platform=platform,
        confidence=confidence,
        watermarks=watermarks,
        signals=signals,
        caveats=caveats,
    )
