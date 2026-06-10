"""AI metadata detection and removal.

Wraps the noai-watermark metadata handling for stripping AI-generation
metadata (EXIF, PNG text chunks, C2PA provenance) from images.

For metadata-only operations, the heavy ML dependencies are NOT required.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Smaller scan_head window for the cheap marker checks (has_ai_metadata,
# samsung_genai); the full-detail scans use scan_head's 1 MB default. Sharing
# one constant also keeps both call sites on the same memoized cache entry.
_QUICK_SCAN_BYTES = 512 * 1024

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


def c2pa_marker_in(data: bytes) -> bool:
    """True if ``data`` carries a real C2PA manifest marker, not just an
    incidental 4-byte ``c2pa`` substring.

    A bare ``c2pa`` byte match false-positives on compressed pixel data -- a
    recompressed PNG IDAT (or any large binary) can contain the bytes ``c2pa``
    by chance (verified 2026-05-29: 4 cleaned PNGs re-flagged this way after
    their manifest was correctly stripped). Every real manifest is JUMBF-wrapped
    (the ``jumb`` box FourCC accompanies the ``c2pa`` content type) or uses the
    standalone C2PA ``uuid`` box in ISOBMFF, so we require one of those: the
    joint ``jumb`` + ``c2pa`` match has negligible random-collision probability.
    """
    return C2PA_UUID in data or (b"jumb" in data and b"c2pa" in data.lower())


# IPTC ``digitalSourceType`` values (IPTC 2025.1) that flag AI provenance.
# Used by Instagram, Facebook, X (Twitter) to show "Made with AI" labels.
IPTC_AI_MARKERS: tuple[bytes, ...] = (
    b"trainedAlgorithmicMedia",
    b"compositeSynthetic",
    b"algorithmicMedia",
    b"compositeWithTrainedAlgorithmicMedia",
)

# IPTC Photo Metadata 2025.1 (published 2025-11-27) added explicit AI-disclosure
# XMP properties in the Iptc4xmpExt namespace. Their mere presence is an AI
# signal; ``AISystemUsed`` additionally carries the generator name. Property
# tokens verified against the IPTC 2025.1 specification.
IPTC_AI_FIELD_MARKERS: tuple[bytes, ...] = (
    b"AISystemUsed",
    b"AISystemVersionUsed",
    b"AIPromptInformation",
    b"AIPromptWriterName",
)

# ISOBMFF containers whose AI-provenance boxes ``remove_ai_metadata`` strips at
# the container level (image, video, audio -- all ISOBMFF). A content sniff
# (``ftyp``) is also accepted, so this is a fast-path hint, not the sole gate.
_ISOBMFF_EXTS: frozenset[str] = frozenset({".avif", ".heif", ".heic", ".jxl", ".mp4", ".mov", ".m4v", ".m4a"})

# Non-ISOBMFF audio/video the ISOBMFF box walker can't reach (EBML / framed /
# RIFF / Vorbis). remove_ai_metadata strips their container metadata losslessly
# via ffmpeg (`-c copy`), so it needs ffmpeg on PATH for these.
_FFMPEG_STRIP_EXTS: frozenset[str] = frozenset(
    {".webm", ".mkv", ".mka", ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac"}
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

# TC260 AIGC-label JSON fields (the standard's labeling object). Doubao writes
# the same object as a PNG ``tEXt`` chunk keyed ``AIGC`` (raw JSON, not XMP), so
# a JSON object carrying at least one of these is accepted as a valid TC260
# label even when the namespaced XMP element is absent.
_TC260_FIELDS: frozenset[str] = frozenset(
    {
        "Label",
        "ContentProducer",
        "ProduceID",
        "ContentPropagator",
        "PropagateID",
        "ReservedCode1",
        "ReservedCode2",
    }
)

# HuggingFace-hosted GPU jobs (Jobs / Spaces) stamp generated PNGs with this
# ``tEXt`` chunk key holding the job UUID. It marks the hosting job, not a
# specific model -- a medium-confidence AI signal (commonly diffusion output).
_HF_JOB_KEY: str = "hf-job-id"

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


# PNG ancillary chunks that can carry provenance metadata (XMP, EXIF, text).
# Never IDAT -- that is the compressed pixel stream.
_PNG_META_CHUNKS: frozenset[bytes] = frozenset({b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"iCCP"})


def _png_late_metadata(image_path: Path, window: int) -> bytes:
    """Payloads of PNG metadata chunks that start *beyond* the first ``window``
    bytes, found by seeking past the (large) ``IDAT`` pixel stream.

    A PNG encoder may append the XMP/EXIF packet after the image data, so a
    fixed first-``size`` read misses it (e.g. a TC260 AIGC label in an XMP
    ``iTXt`` chunk at ~2.7 MB). This is the PNG analogue of the ISOBMFF
    late-box scan in :func:`scan_head`. Returns only chunks past ``window`` so
    bytes already in the head are not duplicated; empty when there are none.
    """
    out = bytearray()
    try:
        with open(image_path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return b""
            f.seek(0, 2)
            file_size = f.tell()
            pos = 8
            while True:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                (length,) = struct.unpack(">I", header[:4])
                chunk_type = header[4:8]
                if chunk_type == b"IEND":
                    break
                data_start = pos + 8
                # Clamp the attacker-controlled 32-bit length to the bytes that
                # actually remain, so a malformed huge length can't allocate GBs.
                safe_length = max(0, min(length, file_size - data_start))
                if chunk_type in _PNG_META_CHUNKS and data_start >= window:
                    f.seek(data_start)
                    out += f.read(safe_length)
                # Advance by the CLAMPED length: a malformed/inflated `length` that
                # overshoots EOF must not push `pos` past the file and abort the scan
                # (which would silently skip a genuine AI-label chunk after it).
                pos = data_start + safe_length + 4  # data + CRC
    except OSError as exc:
        logger.debug("PNG late-metadata scan failed on %s: %s", image_path, exc)
        return b""
    return bytes(out)


def scan_head(image_path: Path, size: int = 1024 * 1024) -> bytes:
    """First ``size`` bytes of the file, plus the payloads of any provenance
    metadata found beyond that window: ISOBMFF ``uuid`` / ``jumb`` boxes (seeking
    past large boxes like ``mdat``) and PNG ``tEXt`` / ``iTXt`` / ``eXIf`` chunks
    (seeking past ``IDAT``).

    This is the shared input for every C2PA / AIGC / IPTC byte scan. The
    extensions catch a manifest or XMP packet placed AFTER the media data -- a
    non-faststart MP4 manifest, or a PNG XMP packet appended after the pixels --
    which a fixed first-MB read would miss. For other inputs, and for files that
    fit within ``size``, it is exactly ``f.read(size)`` -- behavior-neutral.

    The result is memoized per (path, size, mtime): one ``identify``/``get_ai_metadata``
    call fans out to ~8 byte-scan detectors that each call this on the same file, so
    the cache turns those repeated reads into one. The mtime key invalidates the entry
    when the file changes; the small ``maxsize`` bounds memory to a few MB.
    """
    try:
        mtime = image_path.stat().st_mtime_ns
    except OSError:
        # No stat (e.g. a pipe, or a race): read uncached rather than fail.
        return _scan_head_impl(image_path, size)
    return _scan_head_cached(str(image_path), size, mtime)


@functools.lru_cache(maxsize=8)
def _scan_head_cached(path_str: str, size: int, _mtime_ns: int) -> bytes:
    """Cache shim: ``_mtime_ns`` is part of the key only (invalidates on change)."""
    from pathlib import Path as _Path

    return _scan_head_impl(_Path(path_str), size)


def _scan_head_impl(image_path: Path, size: int) -> bytes:
    with open(image_path, "rb") as f:
        head = f.read(size)
    # Lazy import: isobmff imports this module's constants at top level.
    from remove_ai_watermarks.noai import isobmff

    if isobmff.is_isobmff(head):
        region = isobmff.scan_c2pa_region(image_path)
        if region:
            head += region
    elif head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) == size:
        # len(head) == size means the file is at least `size` bytes, so metadata
        # chunks may lie beyond the window; otherwise the whole PNG is in `head`.
        head += _png_late_metadata(image_path, size)
    return head


def has_ai_metadata(image_path: Path) -> bool:
    """Check if an image contains AI-generation metadata.

    Args:
        image_path: Path to the image.

    Returns:
        True if AI metadata is detected.
    """
    from PIL import Image

    # PIL may not handle AVIF/HEIF/JPEG-XL without the optional plugins, and a
    # third-party plugin autoload can raise a non-OSError (e.g. ModuleNotFoundError),
    # so any open failure falls through to the binary scan.
    try:
        with Image.open(image_path) as img:
            for key in img.info:
                if isinstance(key, str) and _is_ai_key(key):
                    return True
    except Exception as exc:
        logger.debug("PIL could not open %s for metadata scan: %s", image_path, exc)

    # Check C2PA — via the official ``c2pa`` lib if available, otherwise via a
    # binary scan that also catches AVIF/HEIF/JPEG-XL containers (PIL doesn't
    # expose their metadata uniformly).
    try:
        # optional official lib, not a declared dep -> falls back to the binary scan
        from c2pa import has_c2pa_metadata  # pyright: ignore[reportMissingImports, reportUnknownVariableType]

        if has_c2pa_metadata(image_path):
            return True
    except ImportError:
        pass

    # Binary scan covers C2PA (PNG caBX, JPEG APP11, AVIF/HEIF/JXL uuid boxes)
    # and IPTC AI markers in XMP. First 512KB (plus late ISOBMFF provenance boxes).
    data = scan_head(image_path, _QUICK_SCAN_BYTES)
    if c2pa_marker_in(data):
        return True
    if any(marker in data for marker in AIGC_MARKERS):
        return True
    if any(marker in data for marker in IPTC_AI_MARKERS):
        return True
    # IPTC 2025.1 AI-disclosure XMP properties (their presence flags AI content).
    if any(marker in data for marker in IPTC_AI_FIELD_MARKERS):
        return True
    # China TC260 AIGC label as a PNG text chunk (the byte scan above catches
    # only the XMP form; the raw-JSON tEXt chunk needs the PIL-based parse).
    if aigc_label(image_path):
        return True
    # HuggingFace-hosted job marker (hf-job-id PNG text chunk).
    if huggingface_job(image_path):
        return True
    # xAI / Grok: no C2PA/IPTC/XMP -- only the EXIF Signature + UUID-Artist pair.
    return xai_signature(image_path)


def aigc_label(image_path: Path) -> dict[str, str] | None:
    """Parse a China TC260 AI-labeling block, if present.

    Three serializations are recognized:

    - a PNG ``tEXt``/``iTXt`` chunk keyed ``AIGC`` carrying the raw JSON object
      (as written by Doubao / ByteDance), read via PIL;
    - an XMP ``<TC260:AIGC>{...}</TC260:AIGC>`` block (HTML-entity encoded text),
      found by a container-agnostic raw-byte scan (PNG/JPEG/WebP alike); and
    - a raw-JSON ``{"AIGC":{...}}`` block with no namespace, as embedded in JPEG
      EXIF (UserComment) by some China-served generators, brace-matched from the
      scan head.

    Returns the decoded JSON (e.g. ``{"Label": "1", "ContentProducer": ...}``)
    or None. The generic forms (the PNG-chunk key ``AIGC`` and the bare
    ``{"AIGC":...}`` object) are accepted only if they carry at least one known
    TC260 field (``_TC260_FIELDS``); the namespaced XMP element is unambiguous,
    so any JSON object is accepted.
    """
    import html
    import json
    from typing import cast

    def _parse(text: str, *, require_tc260_field: bool) -> dict[str, str] | None:
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        fields = {str(k): str(v) for k, v in cast("dict[object, object]", parsed).items()}
        if require_tc260_field and not (_TC260_FIELDS & fields.keys()):
            return None
        return fields

    # PNG tEXt chunk keyed "AIGC" with raw JSON (Doubao and other China gens).
    # The key is generic, so require a TC260 field to avoid a false positive.
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            value = img.info.get("AIGC")
    except Exception as exc:
        logger.debug("PIL could not open %s for AIGC chunk scan: %s", image_path, exc)
        value = None
    if isinstance(value, str) and (result := _parse(value, require_tc260_field=True)):
        return result

    # XMP TC260:AIGC, namespaced (unambiguous) in either serialization RDF allows:
    # an element  <TC260:AIGC>{...}</TC260:AIGC>  or an attribute  TC260:AIGC="{...}"
    # (the attribute form is what PicWish writes). Both are HTML-entity encoded.
    data = scan_head(image_path)
    match = re.search(
        rb'<TC260:AIGC>(.*?)</TC260:AIGC>|TC260:AIGC\s*=\s*"(.*?)"',
        data,
        re.DOTALL,
    )
    if match:
        body = match.group(1) if match.group(1) is not None else match.group(2)
        return _parse(html.unescape(body.decode("utf-8", "replace")), require_tc260_field=False)

    # Raw-JSON {"AIGC":{...}} block (no namespace), as written into JPEG EXIF
    # (UserComment) by some China-served generators -- the PNG-chunk and XMP
    # paths above both miss it. The bytes pre-check keeps the common (no-AIGC)
    # path off the full-buffer decode; raw_decode then brace-matches the inner
    # object (respecting nested braces / quoted strings) and `_parse` applies the
    # same dict coercion + TC260-field gate as the generic PNG-chunk path.
    if b'"AIGC"' in data:
        text = data.decode("latin-1")
        brace = text.find("{", text.find('"AIGC"') + len('"AIGC"'))
        if brace != -1:
            try:
                _, end = json.JSONDecoder().raw_decode(text, brace)
            except ValueError:
                return None
            return _parse(text[brace:end], require_tc260_field=True)
    return None


def huggingface_job(image_path: Path) -> str | None:
    """Return the HuggingFace job id if the image carries an ``hf-job-id`` PNG
    text chunk, else None.

    HuggingFace-hosted GPU jobs (Jobs / Spaces) stamp generated PNGs with an
    ``hf-job-id`` ``tEXt`` chunk holding the job's UUID. It identifies the
    *hosting job*, not a specific model, and is most commonly seen on diffusion-
    generation output -- a medium-confidence AI signal, not proof of AI pixels
    on its own.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            value = img.info.get(_HF_JOB_KEY)
    except Exception as exc:
        logger.debug("PIL could not open %s for hf-job-id scan: %s", image_path, exc)
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# Samsung Galaxy AI editing marker. Galaxy AI tools (Generative Edit, Sketch to
# Image, Portrait Studio, Drawing Assist, ...) record their re-edit data as a
# proprietary ``PhotoEditor_Re_Edit_Data`` JSON that carries a ``genAIType``
# field; a non-zero value flags that a generative-AI tool produced or altered
# the pixels. The field is undocumented by Samsung (verified 2026-05-29: absent
# from the C2PA spec and Samsung's public docs/forums), so detection is
# empirical -- on real Galaxy S23/S24/S25 files it co-occurs with the C2PA
# ``trainedAlgorithmicMedia`` source type (3/3 of the verified files that record
# that type), and on a Galaxy S24 sample it is the *only* AI marker (the C2PA
# source type was absent there). Medium confidence: it signals Galaxy AI editing
# without proving the whole image is AI-generated. Scoped to the Samsung editor
# container to avoid matching a stray ``genAIType`` token elsewhere.
_SAMSUNG_GENAI_RE = re.compile(rb'genAIType"\s*:\s*(-?\d+)')
_SAMSUNG_EDITOR_MARKER = b"PhotoEditor_Re_Edit_Data"


def samsung_genai(image_path: Path) -> int | None:
    """Return Samsung's non-zero ``genAIType`` value if the image carries the
    Galaxy AI editing marker, else None.

    See the module note above ``_SAMSUNG_GENAI_RE``: detection is empirical and
    gated on the ``PhotoEditor_Re_Edit_Data`` container so an incidental
    ``genAIType`` token cannot false-positive.
    """
    head = scan_head(image_path, _QUICK_SCAN_BYTES)
    if _SAMSUNG_EDITOR_MARKER not in head:
        return None
    m = _SAMSUNG_GENAI_RE.search(head)
    if m is None:
        return None
    return int(m.group(1)) or None


def iptc_ai_system(image_path: Path) -> str | None:
    """Return an IPTC 2025.1 AI-disclosure note if the file carries those XMP
    properties, else None.

    IPTC Photo Metadata 2025.1 added ``Iptc4xmpExt`` AI-disclosure properties
    (see ``IPTC_AI_FIELD_MARKERS``); their presence alone flags AI content, and
    ``AISystemUsed`` names the generator. Returns the ``AISystemUsed`` value when
    extractable, otherwise the literal ``"fields present"``. Container-agnostic
    raw-byte scan; handles both XMP element and attribute serializations.
    """
    data = scan_head(image_path)
    if not any(marker in data for marker in IPTC_AI_FIELD_MARKERS):
        return None
    match = re.search(rb"AISystemUsed[=:\s]*[\"'>]\s*([^<\"']{1,120})", data)
    if match and (value := match.group(1).decode("utf-8", "replace").strip()):
        return value
    return "fields present"


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
    data = scan_head(image_path)
    has_c2pa = c2pa_marker_in(data)
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
        head = scan_head(image_path)
        for match in re.finditer(rb"CreatorTool[>\"'=\s]{1,4}([^<\"']{1,80})", head):
            candidates.append(match.group(1).decode("latin1", "replace"))
    except Exception as exc:
        logger.debug("XMP CreatorTool scan failed for %s: %s", image_path, exc)

    for value in candidates:
        if any(token in value.lower() for token in AI_GENERATOR_TOKENS):
            return value.strip()
    return None


# xAI / Grok EXIF signature scheme. A 64+ char base64 blob after "Signature:"
# is far beyond any incidental description text, and the UUID Artist makes the
# pair xAI-specific -- both required keeps the false-positive rate near zero.
_XAI_SIGNATURE_RE = re.compile(r"Signature:\s*[A-Za-z0-9+/=]{64,}")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def _is_xai_signature_pair(description: str, artist: str) -> bool:
    """True if an EXIF (ImageDescription, Artist) pair is xAI/Grok's scheme."""
    return _XAI_SIGNATURE_RE.match(description) is not None and _UUID_RE.fullmatch(artist) is not None


def _exif_text(ifd: dict[int, Any], tag: int) -> str:
    """Decode a piexif 0th-IFD byte tag to a stripped string ('' if absent)."""
    value = ifd.get(tag)
    return value.decode("latin1", "replace").strip() if isinstance(value, bytes) else ""


def xai_signature(image_path: Path) -> bool:
    """Detect xAI / Grok's EXIF provenance signature scheme.

    Grok image downloads (Aurora model) carry no C2PA, XMP, SynthID, or IPTC --
    their only provenance signal is a private EXIF pair: ``ImageDescription`` =
    ``"Signature: <base64>"`` together with ``Artist`` = the image UUID. Verified
    stable across three independent generations (2026-05-26; see CLAUDE.md). The
    signature is xAI's and is not locally verifiable (no public key); detection
    keys on this distinctive, low-false-positive shape, not on the signature's
    validity. It survives only on the *original* JPEG download -- the web-UI
    image is a re-encoded WebP that drops EXIF.
    """
    try:
        import piexif
        from PIL import Image

        with Image.open(image_path) as img:
            exif_bytes = img.info.get("exif")
        if not exif_bytes:
            return False
        tags = piexif.load(exif_bytes).get("0th", {})
    except Exception as exc:  # unopenable format / malformed EXIF
        logger.debug("xAI-signature EXIF read failed for %s: %s", image_path, exc)
        return False

    return _is_xai_signature_pair(
        _exif_text(tags, piexif.ImageIFD.ImageDescription), _exif_text(tags, piexif.ImageIFD.Artist)
    )


def _scrub_ai_exif(exif_dict: dict[str, Any]) -> list[str]:
    """Delete AI-provenance tags from a piexif dict's ``0th`` IFD, in place.

    Removes (a) the xAI/Grok signature pair (``ImageDescription`` "Signature: ..."
    + UUID ``Artist``) and (b) any ``Software`` / ``Make`` / ``Artist`` /
    ``ImageDescription`` tag whose value carries an ``AI_GENERATOR_TOKENS`` token
    (Ideogram's ``Make``, Firefly's ``Software``, etc.). Mirrors the detection in
    ``xai_signature`` / ``exif_generator`` so removal scrubs exactly what
    ``identify`` flags, while leaving genuine camera/editor EXIF intact. Returns
    the names of the removed tags (for logging).
    """
    import piexif

    from remove_ai_watermarks.noai.constants import AI_GENERATOR_TOKENS

    ifd = exif_dict.get("0th")
    if not ifd:
        return []

    drop: dict[int, str] = {}

    # (a) xAI / Grok: the Signature blob and the UUID Artist go together.
    if _is_xai_signature_pair(
        _exif_text(ifd, piexif.ImageIFD.ImageDescription), _exif_text(ifd, piexif.ImageIFD.Artist)
    ):
        drop[piexif.ImageIFD.ImageDescription] = "ImageDescription"
        drop[piexif.ImageIFD.Artist] = "Artist"

    # (b) Known AI generator token in any of the text tags.
    for tag, name in (
        (piexif.ImageIFD.Software, "Software"),
        (piexif.ImageIFD.Make, "Make"),
        (piexif.ImageIFD.Artist, "Artist"),
        (piexif.ImageIFD.ImageDescription, "ImageDescription"),
    ):
        if any(token in _exif_text(ifd, tag).lower() for token in AI_GENERATOR_TOKENS):
            drop[tag] = name

    for tag in drop:
        ifd.pop(tag, None)
    return list(drop.values())


def get_ai_metadata(image_path: Path) -> dict[str, str]:
    """Extract AI-related metadata from an image.

    Args:
        image_path: Path to the image.

    Returns:
        Dictionary of AI metadata key-value pairs.
    """
    from PIL import Image

    from remove_ai_watermarks.noai.c2pa import extract_c2pa_info, soft_binding_vendors_in, synthid_verdict

    result: dict[str, str] = {}

    # PIL may not open AVIF/HEIF/JPEG-XL without optional plugins (and a
    # third-party plugin autoload can raise a non-OSError); fall through to the
    # C2PA/binary path on any open failure. See CLAUDE.md.
    try:
        with Image.open(image_path) as img:
            for key, value in img.info.items():
                if isinstance(key, str) and _is_ai_key(key):
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
        "soft_binding",
    ):
        if key in c2pa:
            result.setdefault(key, str(c2pa[key]))

    # Non-PNG containers (JPEG/WebP/AVIF/MP4): extract_c2pa_info is PNG-only, so
    # fall back to the format-agnostic source check for the SynthID verdict and
    # the soft-binding (forensic-watermark vendor) scan.
    if "synthid_watermark" not in result and (vendor := synthid_source(image_path)):
        result.setdefault("synthid_watermark", synthid_verdict(vendor))
    if "soft_binding" not in result:
        head = scan_head(image_path)
        if vendors := soft_binding_vendors_in(head):
            result["soft_binding"] = ", ".join(vendors)

    # China TC260 AI-content label (Doubao and other China-served generators).
    if aigc := aigc_label(image_path):
        producer = aigc.get("ContentProducer", "")
        result["aigc_label"] = f"China AIGC label (TC260){f'; producer {producer}' if producer else ''}"

    # xAI / Grok EXIF signature scheme (its only provenance signal).
    if xai_signature(image_path):
        result.setdefault("xai_signature", "xAI/Grok EXIF signature (Artist UUID + Signature blob)")

    # IPTC 2025.1 AI-disclosure XMP fields (Iptc4xmpExt:AISystemUsed etc.).
    if system := iptc_ai_system(image_path):
        result.setdefault("ai_system", f"IPTC 2025.1 AI disclosure ({system})")

    # HuggingFace-hosted job marker (hf-job-id PNG text chunk).
    if job := huggingface_job(image_path):
        result.setdefault("huggingface_job", f"HuggingFace-hosted job ({job})")
    # Samsung Galaxy AI editing marker (genAIType in PhotoEditor_Re_Edit_Data).
    if (genai := samsung_genai(image_path)) is not None:
        result.setdefault("samsung_genai", f"Samsung Galaxy AI editing marker (genAIType={genai})")
    return result


def _strip_with_ffmpeg(source_path: Path, output_path: Path) -> Path:
    """Strip container metadata from a non-ISOBMFF audio/video file via ffmpeg.

    Uses a lossless stream copy (``-c copy``), so codec data is untouched and only
    container-level tags/chapters are dropped -- the metadata strip for WebM /
    Matroska (EBML), MP3 (ID3), WAV / FLAC / OGG (RIFF / Vorbis comments) that the
    ISOBMFF box walker cannot reach. Requires ffmpeg on PATH (raises if absent).
    The output extension should match the source so ``-c copy`` can re-mux.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is required to strip metadata from {source_path.suffix} files but was not found on "
            "PATH; install ffmpeg (e.g. `brew install ffmpeg`) or re-encode the file with another tool"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to strip metadata from {source_path}: {result.stderr.strip()[:300]}")
    logger.info("Stripped container metadata via ffmpeg -> %s", output_path)
    return output_path


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

    # ISOBMFF containers (AVIF/HEIF/JPEG-XL images, MP4/MOV/M4V video, M4A audio):
    # strip C2PA + AI-label boxes at the container level without re-encoding.
    # Avoids needing PIL plugins (pillow-heif / pillow-jxl) and preserves the
    # codestream bit-for-bit. MP4/MOV/M4A are ISOBMFF too, so the same top-level
    # uuid/jumb box walker applies. Route by suffix OR by an ``ftyp`` content
    # sniff, so a correctly-shaped container is handled whatever its extension.
    from remove_ai_watermarks.noai.isobmff import blank_ai_xmp_packets, is_isobmff, strip_c2pa_boxes

    with open(source_path, "rb") as f:
        head = f.read(12)
    if source_path.suffix.lower() in _ISOBMFF_EXTS or is_isobmff(head):
        data = source_path.read_bytes()
        # Top-level uuid/jumb boxes (C2PA + AI-label XMP), then AI-label XMP that
        # lives inside a meta-box ``mime`` item (HEIF/AVIF) -- blanked in place so
        # box sizes and iloc offsets stay valid and the coded image is untouched.
        cleaned, stripped = strip_c2pa_boxes(data)
        cleaned, blanked = blank_ai_xmp_packets(cleaned)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(cleaned)
        logger.info(
            "Stripped %d AI-provenance box(es), blanked %d meta-box XMP packet(s) → %s",
            stripped,
            blanked,
            output_path,
        )
        return output_path

    # Non-ISOBMFF audio/video (WebM/Matroska EBML, MP3 ID3, WAV/FLAC/OGG): the
    # box walker can't reach these, so strip container metadata losslessly via
    # ffmpeg (-c copy -- codec data untouched, only tags/chapters dropped).
    if source_path.suffix.lower() in _FFMPEG_STRIP_EXTS:
        return _strip_with_ffmpeg(source_path, output_path)

    # Read image and filter metadata
    with Image.open(source_path) as img:
        img = img.copy()
        fmt = output_path.suffix.lower()

        save_kwargs: dict[str, Any] = {}
        if fmt in (".jpg", ".jpeg"):
            save_kwargs["format"] = "JPEG"
            # JPEG output is unavoidably lossy, so minimize the loss: high quality
            # and no chroma subsampling (4:4:4). Without these PIL defaults to
            # quality 75 + 4:2:0, which visibly degrades a re-saved image.
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        elif fmt == ".webp":
            # Preserve the WebP container losslessly instead of silently rewriting
            # it as PNG (which changes the format and bloats the file).
            save_kwargs["format"] = "WEBP"
            save_kwargs["lossless"] = True
            if img.mode == "P":  # WebP cannot encode palette mode
                img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        else:
            save_kwargs["format"] = "PNG"

        # Collect non-AI metadata
        kept_meta: dict[str, str] = {}
        exif_data = None

        for key, value in img.info.items():
            if not isinstance(key, str):
                continue
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
            # Scrub AI-provenance EXIF tags (xAI/Grok signature, generator tokens)
            # while keeping genuine camera/editor EXIF; PNG output drops EXIF entirely.
            if removed := _scrub_ai_exif(exif_data):
                logger.info("Scrubbed AI EXIF tag(s): %s", ", ".join(removed))
            with contextlib.suppress(Exception):
                save_kwargs["exif"] = piexif.dump(exif_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, **save_kwargs)

    logger.info("Stripped AI metadata → %s", output_path)
    return output_path
