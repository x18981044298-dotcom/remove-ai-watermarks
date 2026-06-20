"""C2PA (Coalition for Content Provenance and Authenticity) metadata handling.

Reading goes through the official c2pa-python ``Reader`` first (any container it
supports), via ``extract_c2pa_info`` / ``read_manifest_store_json``. The
hand-rolled PNG ``caBX`` JUMBF-chunk tools below (``has_c2pa_metadata`` /
``extract_c2pa_chunk`` / ``inject_c2pa_chunk`` and the ``_extract_c2pa_info_png``
fallback) cover raw-chunk extraction, re-injection, and the cases the validator
rejects (synthetic/partial blobs, a broken/absent wheel). Known issuers:

- Google Imagen
- Adobe Firefly
- Microsoft Designer
- OpenAI (ChatGPT, GPT-4o, Sora, DALL-E)
- Truepic (signing authority)

The fallback parser uses byte-level scanning — it does not validate JUMBF/CBOR
structure but reliably identifies known signatures, issuers, tools, and actions.
The vendor / source-type / SynthID / soft-binding registry scan
(``_populate_registry_fields``) is shared by both the reader and fallback paths.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
import struct
from pathlib import Path
from typing import Any, cast

from remove_ai_watermarks.noai.constants import (
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    C2PA_SOFT_BINDINGS,
    PNG_SIGNATURE,
    SYNTHID_C2PA_ISSUERS,
)

log = logging.getLogger(__name__)

# Official C2PA reader (c2pa-python, a core dependency). It is the primary,
# spec-tracking manifest parser; the hand-rolled caBX/CBOR scanner below stays as
# a fallback for synthetic/partial blobs the validator rejects. The import is
# guarded so a partially-broken install degrades to the byte-scan rather than
# crashing the dependency-light identify path.
_C2paReader: Any = None
with contextlib.suppress(Exception):  # broken/absent wheel -> byte-scan fallback
    from c2pa import Reader as _C2paReader  # pyright: ignore[reportMissingTypeStubs]
_C2PA_READER_AVAILABLE = _C2paReader is not None


def reader_available() -> bool:
    """True when the official c2pa-python Reader imported successfully."""
    return _C2PA_READER_AVAILABLE


def read_manifest_store_json(image_path: Path) -> str | None:
    """Return the full C2PA manifest-store JSON for ``image_path``, or None.

    Uses the official c2pa-python ``Reader`` (any container it supports: PNG,
    JPEG, WebP, AVIF/HEIF, MP4, ...). Returns None when the reader is unavailable,
    the file carries no parseable manifest, or parsing fails. The JSON is the
    WHOLE store (every manifest plus ingredient manifests), matching the
    whole-chunk semantics of the legacy byte scan -- an AI-source marker in a
    parent/ingredient manifest (e.g. a ChatGPT edit of a Sora generation) is
    still seen.

    Memoized per (path, mtime): one identify/get_ai_metadata call invokes the
    structured parser ~3 times on the same file, so the cache turns the repeated
    crypto-validating reads into one.
    """
    if not _C2PA_READER_AVAILABLE:
        return None
    try:
        mtime = image_path.stat().st_mtime_ns
    except OSError:
        return _read_manifest_store_impl(str(image_path))
    return _read_manifest_store_cached(str(image_path), mtime)


@functools.lru_cache(maxsize=8)
def _read_manifest_store_cached(path_str: str, _mtime_ns: int) -> str | None:
    """Cache shim: ``_mtime_ns`` is part of the key only (invalidates on change)."""
    return _read_manifest_store_impl(path_str)


def _read_manifest_store_impl(path_str: str) -> str | None:
    # try_create returns None when there is no manifest; a default Reader does no
    # trust enforcement, so an untrusted signer still yields the manifest content
    # (we report what is in the file, we do not gate on certificate trust).
    try:
        reader = _C2paReader.try_create(path_str)
    except Exception as exc:  # malformed manifest, unsupported container, etc.
        log.debug("c2pa Reader could not parse %s: %s", path_str, exc)
        return None
    if reader is None:
        return None
    try:
        with reader:
            return reader.json()
    except Exception as exc:  # pragma: no cover - reader opened but json() failed
        log.debug("c2pa Reader.json() failed on %s: %s", path_str, exc)
        return None


def has_c2pa_metadata(image_path: Path) -> bool:
    """
    Check if an image contains C2PA metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        True if C2PA metadata is detected, False otherwise.
    """
    image_path = Path(image_path)

    if image_path.suffix.lower() != ".png":
        return False

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return False

            file_size = f.seek(0, 2)
            f.seek(8)

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]
                # Clamp the attacker-controlled 32-bit length to the bytes that
                # actually remain, so a malformed huge length can't allocate GBs.
                safe_length = max(0, min(length, file_size - f.tell()))

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(safe_length)
                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return True
                    # Also check if chunk_data itself contains C2PA-like patterns
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return True
                    f.read(4)
                else:
                    f.seek(safe_length + 4, 1)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return False


def _claim_generator_from_store(store: dict[str, Any]) -> str | None:
    """Structured claim-generator name from the active manifest of a store dict.

    Prefers the top-level ``claim_generator`` string (Firefly: "Adobe_Firefly"),
    falling back to the first ``claim_generator_info[].name`` (ChatGPT keys it
    only there). isprintable() guards against odd binary-ish values.
    """
    active = _active_manifest(store)
    generator: Any = active.get("claim_generator")
    if not (isinstance(generator, str) and generator):
        info_list: list[Any] = active.get("claim_generator_info") or []
        if info_list and isinstance(first := info_list[0], dict):
            generator = cast("dict[str, Any]", first).get("name")
    return generator if isinstance(generator, str) and generator and generator.isprintable() else None


def _active_manifest(store: dict[str, Any]) -> dict[str, Any]:
    """The active manifest dict from a manifest-store dict, or {} when absent."""
    manifests: Any = store.get("manifests")
    if not isinstance(manifests, dict):
        return {}
    active = cast("dict[str, Any]", manifests).get(store.get("active_manifest", ""))
    return cast("dict[str, Any]", active) if isinstance(active, dict) else {}


def _info_from_store_json(store_json: str) -> dict[str, Any]:
    """Build the C2PA info dict from a c2pa-python manifest-store JSON string."""
    store_bytes = store_json.encode("utf-8")
    c2pa_info: dict[str, Any] = {
        "has_c2pa": True,
        "type": "C2PA (Coalition for Content Provenance and Authenticity)",
        "c2pa_manifest": f"C2PA manifest store ({len(store_bytes)} bytes)",
    }
    # The whole-store JSON carries every vendor / source-type / SynthID /
    # soft-binding signature (across active + ingredient manifests), so the same
    # registry scan that runs on the raw caBX chunk applies unchanged here.
    _populate_registry_fields(store_bytes, c2pa_info)

    try:
        parsed: Any = json.loads(store_json)
    except (ValueError, TypeError):
        return c2pa_info
    if not isinstance(parsed, dict):
        return c2pa_info
    store = cast("dict[str, Any]", parsed)
    if generator := _claim_generator_from_store(store):
        c2pa_info["claim_generator"] = generator
    sig: Any = _active_manifest(store).get("signature_info")
    if isinstance(sig, dict) and (time := cast("dict[str, Any]", sig).get("time")):
        c2pa_info["timestamp"] = str(time)
    return c2pa_info


def extract_c2pa_info(image_path: Path) -> dict[str, Any]:
    """
    Extract C2PA metadata information from an image.

    Uses the official c2pa-python reader first (any supported container), falling
    back to the hand-rolled PNG caBX parser when the reader is unavailable or the
    file carries no parseable manifest (synthetic/partial blobs).

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary containing C2PA metadata info, or {} when none is found.
    """
    image_path = Path(image_path)

    if (store_json := read_manifest_store_json(image_path)) is not None:
        return _info_from_store_json(store_json)

    return _extract_c2pa_info_png(image_path)


def _extract_c2pa_info_png(image_path: Path) -> dict[str, Any]:
    """Fallback PNG caBX parser, used when the c2pa-python reader finds nothing."""
    c2pa_info: dict[str, Any] = {}

    if not has_c2pa_metadata(image_path):
        return c2pa_info

    c2pa_info["has_c2pa"] = True
    c2pa_info["type"] = "C2PA (Coalition for Content Provenance and Authenticity)"

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return c2pa_info

            file_size = f.seek(0, 2)
            f.seek(8)

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]
                # Clamp the attacker-controlled 32-bit length to the bytes that
                # actually remain, so a malformed huge length can't allocate GBs.
                safe_length = max(0, min(length, file_size - f.tell()))

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(safe_length)
                    _parse_c2pa_chunk(chunk_data, c2pa_info)
                    f.read(4)
                else:
                    f.seek(safe_length + 4, 1)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return c2pa_info


def cbor_text_after(payload: bytes, key: bytes) -> str | None:
    """Return the CBOR text-string immediately following ``key`` in ``payload``.

    Handles CBOR major-type 3 length prefixes: direct (0x60-0x77), 1-byte
    (0x78 NN), and 2-byte (0x79 NN NN). This reads the actual encoded value, so
    it avoids the byte-grabbing artifacts a loose regex produces (e.g. the
    leading length byte showing up as ``fGPT-4o``).
    """
    idx = payload.find(key)
    if idx < 0:
        return None
    p = idx + len(key)
    if p >= len(payload):
        return None
    head = payload[p]
    if 0x60 <= head <= 0x77:
        length, start = head - 0x60, p + 1
    elif head == 0x78 and p + 1 < len(payload):
        length, start = payload[p + 1], p + 2
    elif head == 0x79 and p + 2 < len(payload):
        length, start = (payload[p + 1] << 8) | payload[p + 2], p + 3
    else:
        return None
    raw_str = payload[start : start + length]
    try:
        return raw_str.decode("utf-8")
    except UnicodeDecodeError:
        return raw_str.decode("latin1", errors="replace")


def synthid_verdict(vendors: str) -> str:
    """Human-readable SynthID-source verdict, shared by all callers."""
    return f"likely present ({vendors} embeds SynthID with C2PA)"


def synthid_vendors_in(buffer: bytes) -> list[str]:
    """Return SynthID-using C2PA issuer names whose signature appears in ``buffer``.

    Shared by the PNG caBX parser and the format-agnostic binary scan so both
    apply the same SYNTHID_C2PA_ISSUERS rule against their respective bytes.
    """
    return sorted({name for sig, name in C2PA_ISSUERS.items() if sig in buffer and sig in SYNTHID_C2PA_ISSUERS})


def soft_binding_vendors_in(buffer: bytes) -> list[str]:
    """Return forensic-watermark vendor names whose C2PA soft-binding ``alg``
    identifier appears in ``buffer``.

    A ``c2pa.soft-binding`` assertion names the watermark scheme that stamped the
    pixels (Adobe TrustMark, Digimarc, Imatag, Steg.AI, ...). Shared by the PNG
    caBX parser and the format-agnostic binary scan so both apply the same
    C2PA_SOFT_BINDINGS rule against their respective bytes.
    """
    return sorted({name for sig, name in C2PA_SOFT_BINDINGS.items() if sig in buffer})


def _populate_registry_fields(buf: bytes, c2pa_info: dict[str, Any]) -> bool:
    """Populate the registry-driven C2PA fields by scanning ``buf``.

    Shared by the legacy caBX-chunk parser and the c2pa-python store-JSON path so
    both produce an identical dict shape. ``buf`` is the raw manifest bytes for
    the former and the manifest-store JSON (UTF-8) for the latter; the vendor /
    tool / action / source-type / SynthID / soft-binding signatures appear in
    both. Sets ``issuer``, ``ai_tool``, ``actions``, ``source_type``,
    ``synthid_vendors`` / ``synthid_watermark``, ``soft_binding_vendors`` /
    ``soft_binding`` when present and returns whether the source type is AI.
    """
    if issuers := [name for sig, name in C2PA_ISSUERS.items() if sig in buf]:
        c2pa_info["issuer"] = ", ".join(dict.fromkeys(issuers))

    if ai_tools := [name for sig, name in C2PA_AI_TOOLS.items() if sig in buf]:
        c2pa_info["ai_tool"] = ", ".join(dict.fromkeys(ai_tools))

    if actions := [name for sig, name in C2PA_ACTIONS.items() if sig in buf]:
        c2pa_info["actions"] = ", ".join(actions)

    # Digital source type (matched anywhere in the store, including ingredient
    # manifests -- a ChatGPT edit of a Sora generation carries the AI marker on
    # the parent, not the active manifest).
    # ``ai_source_kind`` is the structured generated-vs-enhanced split the caller
    # branches on (full-frame scrub vs region-targeted clean); ``source_type`` is the
    # human-readable form. The two byte strings are unambiguous:
    # "compositeWithTrainedAlgorithmicMedia" capitalizes the inner "Trained", so a
    # lowercase "trainedAlgorithmicMedia" match is standalone full generation, which
    # wins when both appear (an edit chain).
    ai_source = False
    if b"trainedAlgorithmicMedia" in buf:
        c2pa_info["source_type"] = "trainedAlgorithmicMedia (AI-generated)"
        c2pa_info["ai_source_kind"] = "generated"
        ai_source = True
    elif b"algorithmicMedia" in buf:
        c2pa_info["source_type"] = "algorithmicMedia"
    elif b"compositeWithTrainedAlgorithmicMedia" in buf:
        c2pa_info["source_type"] = "compositeWithTrainedAlgorithmicMedia (AI-enhanced)"
        c2pa_info["ai_source_kind"] = "enhanced"
        ai_source = True

    # SynthID pixel-watermark proxy: a C2PA manifest from a SynthID-using
    # vendor (Google/OpenAI) on AI-generated content implies an invisible
    # SynthID watermark in the pixels (see SYNTHID_C2PA_ISSUERS).
    synthid_vendors = synthid_vendors_in(buf)
    if synthid_vendors and ai_source:
        c2pa_info["synthid_vendors"] = synthid_vendors
        c2pa_info["synthid_watermark"] = synthid_verdict(", ".join(synthid_vendors))

    # Soft-binding: a forensic/third-party watermark vendor named in the
    # manifest (Adobe TrustMark, Digimarc, ...), independent of the issuer.
    soft_binding_vendors = soft_binding_vendors_in(buf)
    if soft_binding_vendors:
        c2pa_info["soft_binding_vendors"] = soft_binding_vendors
        c2pa_info["soft_binding"] = ", ".join(soft_binding_vendors)

    return ai_source


def _parse_c2pa_chunk(chunk_data: bytes, c2pa_info: dict[str, Any]) -> None:
    """Parse a raw caBX chunk payload and populate the info dictionary.

    The fallback path, used when the official c2pa-python reader is unavailable
    or rejects the file (synthetic/partial blobs, broken installs).
    """
    c2pa_info["c2pa_manifest"] = f"C2PA manifest ({len(chunk_data)} bytes)"

    _populate_registry_fields(chunk_data, c2pa_info)

    # Claim generator and spec version: read the CBOR text-string values
    # directly (regex byte-grabbing produced artifacts like ``fGPT-4o``).
    # Guard with isprintable(): on some manifests (e.g. Microsoft Designer) the
    # first ``name`` key precedes a binary field (a hash), not the generator
    # string, which would otherwise surface as control-char garbage.
    if (generator := cbor_text_after(chunk_data, b"name")) and generator.isprintable():
        c2pa_info["claim_generator"] = generator
    if (spec := cbor_text_after(chunk_data, b"specVersion")) and spec.isprintable():
        c2pa_info["c2pa_spec"] = spec

    # Find timestamps
    timestamp_matches = re.findall(rb"(\d{14}Z)", chunk_data)
    if timestamp_matches:
        c2pa_info["timestamp"] = timestamp_matches[0].decode("utf-8")
        if len(timestamp_matches) > 1:
            c2pa_info["timestamps"] = [t.decode("utf-8") for t in timestamp_matches[:3]]


def extract_c2pa_chunk(image_path: Path) -> bytes | None:
    """
    Extract the raw C2PA JUMBF chunk from a PNG file.

    Args:
        image_path: Path to the source PNG file.

    Returns:
        Raw bytes of the C2PA chunk or None.
    """
    if image_path.suffix.lower() != ".png":
        return None

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return None

            file_size = f.seek(0, 2)
            f.seek(8)

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]
                # Clamp the attacker-controlled 32-bit length to the bytes that
                # actually remain, so a malformed huge length can't allocate GBs.
                safe_length = max(0, min(length, file_size - f.tell()))

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(safe_length)
                    crc = f.read(4)

                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return chunk_header + chunk_data + crc

                    # Also check lowercase variants
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return chunk_header + chunk_data + crc
                else:
                    f.seek(safe_length + 4, 1)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return None


def inject_c2pa_chunk(target_path: Path, output_path: Path, c2pa_chunk: bytes) -> None:
    """
    Inject a C2PA JUMBF chunk into a PNG file.

    Args:
        target_path: Path to the target PNG file.
        output_path: Path where the output file will be saved.
        c2pa_chunk: Raw bytes of the C2PA chunk to inject.

    Raises:
        ValueError: If not PNG files.
    """
    if target_path.suffix.lower() != ".png" or output_path.suffix.lower() != ".png":
        raise ValueError("C2PA chunk injection is only supported for PNG files")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "rb") as f_in, open(output_path, "wb") as f_out:
        f_out.write(f_in.read(8))

        c2pa_injected = False
        while True:
            chunk_header = f_in.read(8)
            if len(chunk_header) < 8:
                break

            length = struct.unpack(">I", chunk_header[:4])[0]
            chunk_type = chunk_header[4:8]
            chunk_data = f_in.read(length)
            crc = f_in.read(4)

            if chunk_type == b"IDAT" and not c2pa_injected:
                f_out.write(c2pa_chunk)
                c2pa_injected = True

            if chunk_type == C2PA_CHUNK_TYPE:
                continue

            f_out.write(chunk_header)
            f_out.write(chunk_data)
            f_out.write(crc)

            if chunk_type == b"IEND":
                break
