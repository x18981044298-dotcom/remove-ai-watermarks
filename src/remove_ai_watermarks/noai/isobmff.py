"""Minimal ISOBMFF box walker for stripping C2PA from AVIF / HEIF / MP4 / JPEG-XL.

The ISO Base Media File Format wraps content in nested ``[size:4][type:4][...]``
boxes. C2PA stores its manifest in a top-level ``uuid`` box keyed by the
C2PA UUID; JPEG-XL uses a ``jumb`` box (JUMBF) instead. To strip provenance
without re-encoding the image, we walk the top-level box list, drop boxes that
carry C2PA, and emit the rest verbatim. The codestream (``mdat`` for ISOBMFF,
``jxlc`` / ``jxlp`` for JPEG-XL) is untouched, so pixel data is preserved
bit-for-bit.

This file intentionally avoids dependencies on format-specific libraries
(pillow-heif, pillow-jxl, pymp4) so it works on systems where they aren't
installed.

Reference: ISO/IEC 14496-12 (ISOBMFF) and C2PA 2.1 spec §11.
"""

from __future__ import annotations

import logging
import re
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from remove_ai_watermarks.metadata import (
    AIGC_MARKERS,
    C2PA_UUID,
    IPTC_AI_FIELD_MARKERS,
    IPTC_AI_MARKERS,
)

log = logging.getLogger(__name__)

# Top-level box types that may carry AI provenance. ``uuid`` boxes are checked
# against ``C2PA_UUID`` / AI-label markers before being stripped; ``jumb`` boxes
# are always stripped (JPEG-XL uses them exclusively for JUMBF).
C2PA_BOX_TYPES: frozenset[bytes] = frozenset({b"uuid", b"jumb"})

# AI-label byte markers (TC260 AIGC, IPTC "Made with AI", IPTC 2025.1 AI fields)
# whose presence inside an XMP ``uuid`` box means the box carries an AI label.
# Matching the payload rather than a fixed XMP UUID avoids the XMP-box UUID
# byte-order ambiguity and stays surgical: only AI-bearing XMP is dropped, plain
# XMP (copyright, camera info) is kept.
_AI_LABEL_MARKERS: tuple[bytes, ...] = AIGC_MARKERS + IPTC_AI_MARKERS + IPTC_AI_FIELD_MARKERS

# Adobe XMP packet delimiters (XMP spec part 3). In HEIF/AVIF the XMP packet
# sits inside a ``meta``-box ``mime`` item whose bytes live in ``mdat`` / ``idat``,
# out of reach of the top-level box stripper, so an AI-label packet there is
# blanked in place (see ``blank_ai_xmp_packets``).
_XMP_PACKET_RE = re.compile(rb"<\?xpacket begin=.*?<\?xpacket end=[^>]*?\?>", re.DOTALL)


def _iter_top_level_boxes(data: bytes) -> Iterator[tuple[int, int, bytes, int]]:
    """Yield ``(start, end, type, payload_offset)`` for each top-level box.

    Handles all three ISOBMFF box-size encodings:
    - ``size > 1``: 32-bit size field is the total box length.
    - ``size == 1``: 64-bit ``largesize`` follows after the type field.
    - ``size == 0``: box runs to end of file.
    """
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        size32 = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]
        if size32 == 1:
            if pos + 16 > n:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_off = pos + 16
        elif size32 == 0:
            size = n - pos
            payload_off = pos + 8
        else:
            size = size32
            payload_off = pos + 8
        if size < (payload_off - pos) or pos + size > n:
            return
        yield pos, pos + size, box_type, payload_off
        pos += size


def is_isobmff(data: bytes) -> bool:
    """Cheap sniff: ISOBMFF files start with an ``ftyp`` box."""
    return len(data) >= 8 and data[4:8] == b"ftyp"


def scan_c2pa_region(path: str | Path, *, max_total: int = 4 * 1024 * 1024) -> bytes:
    """Concatenated payloads of top-level ``uuid`` / ``jumb`` boxes in an ISOBMFF
    file, found by seeking past other boxes (``mdat`` etc.) by size.

    C2PA manifests and XMP packets (incl. AI labels) live in top-level ``uuid``
    boxes; JPEG-XL uses ``jumb``. In a streaming / non-faststart MP4 the manifest
    sits AFTER a multi-megabyte ``mdat``, so a fixed first-MB read misses it. This
    walks box headers (8-16 bytes each) and seeks past payloads it does not need,
    so it never loads ``mdat`` into memory and works on multi-GB files. Returns
    the relevant box payloads (capped at ``max_total``), or ``b""`` for a
    non-ISOBMFF file or on any read error.
    """
    collected = bytearray()
    try:
        with open(path, "rb") as f:
            sniff = f.read(8)
            if len(sniff) < 8 or sniff[4:8] != b"ftyp":
                return b""
            f.seek(0, 2)
            file_size = f.tell()
            pos = 0
            while pos + 8 <= file_size and len(collected) < max_total:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                size32 = struct.unpack(">I", header[:4])[0]
                box_type = header[4:8]
                payload_off = pos + 8
                if size32 == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    size = struct.unpack(">Q", ext)[0]
                    payload_off = pos + 16
                elif size32 == 0:
                    size = file_size - pos
                else:
                    size = size32
                if size < (payload_off - pos) or pos + size > file_size:
                    # Detection-only: a malformed box halts the walk, so a manifest
                    # placed after it is missed (best-effort scan; no resync).
                    break
                if box_type in C2PA_BOX_TYPES:
                    f.seek(payload_off)
                    to_read = min(pos + size - payload_off, max_total - len(collected))
                    if to_read > 0:
                        collected += f.read(to_read)
                pos += size
    except OSError:
        return b""
    return bytes(collected)


def strip_c2pa_boxes(data: bytes) -> tuple[bytes, int]:
    """Return ``(cleaned_bytes, stripped_count)`` with AI-provenance boxes removed.

    Walks top-level boxes and drops:
    - any ``uuid`` box whose UUID equals ``C2PA_UUID`` (a C2PA manifest);
    - any ``uuid`` box whose payload carries an AI-label marker (an XMP packet
      with a TC260 / IPTC / IPTC-2025.1 AI field -- caught by content, not by the
      XMP UUID, so it works regardless of the UUID's byte order, and leaves plain
      non-AI XMP intact);
    - any ``jumb`` box (JPEG-XL JUMBF container).

    All other boxes (incl. ``mdat`` / codestream) are emitted verbatim, so pixel
    and audio data is preserved bit-for-bit. Non-ISOBMFF input is returned
    unchanged. Despite the name this also covers MP4/MOV/M4A video and audio
    (all ISOBMFF). NOTE: this drops only top-level boxes. AI metadata stored as an
    *item inside the ``meta`` box* (typical for AVIF/HEIF) is handled separately and
    in place (same length, no offset rewrite): AI-label XMP by
    :func:`blank_ai_xmp_packets`, and AI-generator tokens in an ``Exif`` item by
    :func:`blank_ai_exif_tokens`.
    """
    if not is_isobmff(data):
        return data, 0

    out = bytearray()
    stripped = 0
    consumed = 0
    for start, end, box_type, payload_off in _iter_top_level_boxes(data):
        consumed = end
        if box_type == b"uuid":
            # uuid boxes carry the 16-byte UUID immediately after the type.
            is_c2pa = payload_off + 16 <= end and data[payload_off : payload_off + 16] == C2PA_UUID
            has_ai_label = any(marker in data[payload_off:end] for marker in _AI_LABEL_MARKERS)
            if is_c2pa or has_ai_label:
                stripped += 1
                continue
        elif box_type == b"jumb":
            stripped += 1
            continue
        out.extend(data[start:end])

    # Fail-safe: the walker returns early on a malformed box (bad size, or a box
    # that runs past EOF), so anything after it was never visited. Emitting `out`
    # would silently truncate the file from the bad box to EOF -- worse than not
    # stripping. If the walk did not consume the whole input, return it unchanged.
    if consumed != len(data):
        log.warning(
            "ISOBMFF box walk stopped at offset %d of %d (malformed box); "
            "returning input unchanged to avoid truncation",
            consumed,
            len(data),
        )
        return data, 0

    return bytes(out), stripped


def blank_ai_xmp_packets(data: bytes) -> tuple[bytes, int]:
    """Overwrite (with spaces, in place) any XMP packet carrying an AI-label
    marker; return ``(data, blanked_count)``.

    HEIF/AVIF store XMP as a ``meta``-box ``mime`` item whose bytes live in
    ``mdat`` / ``idat``, which ``strip_c2pa_boxes`` cannot remove without
    meta-box surgery (``iinf`` / ``iloc`` rewrite). Instead, the XMP packet is
    located by its ``<?xpacket begin ... end?>`` delimiters and, when it carries
    an AI-label marker (TC260 AIGC / IPTC / IPTC-2025.1), overwritten with spaces.
    Because the replacement is the **same length**, every box size and ``iloc``
    offset stays valid and the coded image data is untouched -- only the AI label
    content is destroyed. Packets without an AI marker (plain copyright / camera
    XMP) are left intact, mirroring the top-level XMP-``uuid`` content match.
    """
    blanked = 0

    def _scrub(match: re.Match[bytes]) -> bytes:
        nonlocal blanked
        packet = match.group()
        if any(marker in packet for marker in _AI_LABEL_MARKERS):
            blanked += 1
            return b" " * len(packet)
        return packet

    return _XMP_PACKET_RE.sub(_scrub, data), blanked


# EXIF TIFF byte-order headers: little-endian (II 0x2a 0x00) and big-endian
# (MM 0x00 0x2a). A HEIF/AVIF ``Exif`` meta-box item stores its TIFF block in
# ``mdat`` / ``idat``, so the block (and these headers) appear in the raw bytes.
_TIFF_HEADERS: tuple[bytes, ...] = (b"II\x2a\x00", b"MM\x00\x2a")
# How far past a TIFF header an EXIF block plausibly extends; bounds the slice we
# hand to piexif and search within (EXIF blocks are small kilobyte-scale).
_EXIF_WINDOW = 256 * 1024


def blank_ai_exif_tokens(data: bytes) -> tuple[bytes, int]:
    """Overwrite (with spaces, in place) any AI-generator token in an EXIF block
    stored as an ISOBMFF ``meta``-box ``Exif`` item; return ``(data, blanked_count)``.

    HEIF/AVIF can carry EXIF as a ``meta``-box ``Exif`` item whose TIFF bytes live
    in ``mdat`` / ``idat`` -- out of reach of the top-level box stripper, and (when
    no pillow-heif plugin is installed) of the PIL EXIF reader too, so an AI
    ``Software`` / ``Make`` / ``Artist`` / ``ImageDescription`` tag there survived
    ``remove_ai_metadata`` (a documented gap). This locates EXIF TIFF blocks by
    their byte-order header, **validates each with piexif** (so a coincidental
    II/MM run in pixel data is ignored -- it will not parse as a TIFF IFD), and
    overwrites any value carrying an ``AI_GENERATOR_TOKENS`` token with spaces of
    the SAME length. Because the replacement is same-length, every box size and
    ``iloc`` offset stays valid and the coded image is untouched -- only the AI tag
    content is destroyed; camera/editor EXIF without an AI token is left intact
    (mirrors ``metadata._scrub_ai_exif`` and ``blank_ai_xmp_packets``).
    """
    import piexif

    from remove_ai_watermarks.noai.constants import AI_GENERATOR_TOKENS

    ai_tags = (
        piexif.ImageIFD.Software,
        piexif.ImageIFD.Make,
        piexif.ImageIFD.Artist,
        piexif.ImageIFD.ImageDescription,
    )
    out = bytearray(data)
    blanked = 0
    for header in _TIFF_HEADERS:
        pos = data.find(header)
        while pos != -1:
            window = bytes(out[pos : pos + _EXIF_WINDOW])
            ifd: dict[int, Any] = {}
            try:
                ifd = piexif.load(window).get("0th", {})
            except Exception:
                ifd = {}
            for tag in ai_tags:
                value = ifd.get(tag)
                if not isinstance(value, bytes):
                    continue
                if any(token in value.decode("latin1", "replace").lower() for token in AI_GENERATOR_TOKENS):
                    # Blank the value bytes in place, within this EXIF block only.
                    vpos = out.find(value, pos, pos + _EXIF_WINDOW)
                    if vpos != -1:
                        out[vpos : vpos + len(value)] = b" " * len(value)
                        blanked += 1
            pos = data.find(header, pos + len(header))
    return bytes(out), blanked
