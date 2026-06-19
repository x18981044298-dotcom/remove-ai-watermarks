"""Tests for vendored noai submodules: constants, extractor, cleaner, c2pa."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from remove_ai_watermarks.noai.c2pa import (
    _parse_c2pa_chunk,
    cbor_text_after,
    extract_c2pa_chunk,
    extract_c2pa_info,
    has_c2pa_metadata,
    inject_c2pa_chunk,
    synthid_verdict,
)
from remove_ai_watermarks.noai.cleaner import (
    has_ai_content,
)
from remove_ai_watermarks.noai.cleaner import (
    remove_ai_metadata as noai_remove_ai_metadata,
)
from remove_ai_watermarks.noai.constants import (
    AI_KEYWORDS,
    AI_METADATA_KEYS,
    C2PA_CHUNK_TYPE,
    PNG_SIGNATURE,
    SUPPORTED_FORMATS,
)
from remove_ai_watermarks.noai.extractor import (
    extract_ai_metadata,
    extract_metadata,
    get_ai_metadata_summary,
    has_ai_metadata,
)
from remove_ai_watermarks.noai.isobmff import (
    blank_ai_exif_tokens,
    is_isobmff,
    strip_c2pa_boxes,
)

# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    """Verify constant integrity."""

    def test_supported_formats_include_png(self):
        assert ".png" in SUPPORTED_FORMATS

    def test_supported_formats_include_jpg(self):
        assert ".jpg" in SUPPORTED_FORMATS

    def test_ai_metadata_keys_not_empty(self):
        assert len(AI_METADATA_KEYS) > 0

    def test_ai_keywords_not_empty(self):
        assert len(AI_KEYWORDS) > 0

    def test_png_signature_bytes(self):
        assert PNG_SIGNATURE == b"\x89PNG\r\n\x1a\n"

    def test_c2pa_chunk_type(self):
        assert C2PA_CHUNK_TYPE == b"caBX"


# ── Extractor ───────────────────────────────────────────────────────


class TestExtractor:
    """Tests for noai.extractor functions."""

    def test_extract_metadata_returns_dict(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert isinstance(meta, dict)

    def test_extract_metadata_gets_standard_keys(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert "Author" in meta

    def test_extract_ai_metadata_from_ai_image(self, tmp_png_with_ai_metadata):
        meta = extract_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta

    def test_extract_ai_metadata_from_clean_image(self, tmp_clean_png):
        meta = extract_ai_metadata(tmp_clean_png)
        assert len(meta) == 0

    def test_has_ai_metadata_detects(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_has_ai_metadata_clean(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_summary_with_ai(self, tmp_png_with_ai_metadata):
        summary = get_ai_metadata_summary(tmp_png_with_ai_metadata)
        assert "AI Image Metadata" in summary

    def test_summary_clean(self, tmp_clean_png):
        summary = get_ai_metadata_summary(tmp_clean_png)
        assert "No AI metadata" in summary


# ── Cleaner ─────────────────────────────────────────────────────────


class TestCleaner:
    """Tests for noai.cleaner functions."""

    def test_remove_ai_metadata(self, tmp_png_with_ai_metadata, tmp_path):
        output = tmp_path / "cleaned.png"
        noai_remove_ai_metadata(tmp_png_with_ai_metadata, output)
        assert output.exists()
        # Verify AI metadata removed
        meta = extract_ai_metadata(output)
        assert "parameters" not in meta

    def test_has_ai_content(self, tmp_png_with_ai_metadata):
        assert has_ai_content(tmp_png_with_ai_metadata)


# ── C2PA ────────────────────────────────────────────────────────────


class TestC2PA:
    """Tests for C2PA detection on regular (non-C2PA) images."""

    def test_no_c2pa_on_regular_png(self, tmp_clean_png):
        assert not has_c2pa_metadata(tmp_clean_png)

    def test_no_c2pa_on_jpeg(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)

    def test_extract_c2pa_none_on_regular(self, tmp_clean_png):
        assert extract_c2pa_chunk(tmp_clean_png) is None

    def test_extract_c2pa_info_empty(self, tmp_clean_png):
        info = extract_c2pa_info(tmp_clean_png)
        assert info == {}

    def test_c2pa_returns_false_for_non_png(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestC2PARealSamples:
    """Parser behavior on real committed C2PA images."""

    def test_detects_c2pa_in_openai_png(self):
        assert has_c2pa_metadata(SAMPLES_DIR / "chatgpt-1.png")

    def test_extract_info_openai_fields(self):
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        assert info["has_c2pa"] is True
        assert "OpenAI" in info["issuer"]
        assert "c2pa_manifest" in info  # "C2PA manifest (N bytes)"
        assert "trainedAlgorithmicMedia" in info["source_type"]
        # CBOR-clean claim generator, no regex artifacts (e.g. "fGPT-4o").
        assert info["claim_generator"]
        assert not info["claim_generator"].startswith("f")
        assert "synthid_watermark" in info

    def test_extract_info_adobe_has_no_synthid(self):
        info = extract_c2pa_info(SAMPLES_DIR / "firefly-1.png")
        assert "Adobe" in info["issuer"]
        assert "synthid_watermark" not in info

    def test_extract_chunk_returns_bytes(self):
        chunk = extract_c2pa_chunk(SAMPLES_DIR / "chatgpt-1.png")
        assert chunk is not None
        assert chunk[4:8] == b"caBX"  # chunk type in the 8-byte header

    def test_inject_round_trip(self, tmp_clean_png, tmp_path):
        """Extract a real C2PA chunk, inject into a clean PNG, re-detect."""
        chunk = extract_c2pa_chunk(SAMPLES_DIR / "chatgpt-1.png")
        out = tmp_path / "injected.png"
        inject_c2pa_chunk(tmp_clean_png, out, chunk)
        assert has_c2pa_metadata(out)
        assert "OpenAI" in extract_c2pa_info(out)["issuer"]

    def test_extract_info_flux_jpeg_via_reader(self):
        """Real committed JPEG-with-C2PA fixture: the non-PNG reader path works."""
        info = extract_c2pa_info(SAMPLES_DIR / "flux-1.jpg")
        assert info["has_c2pa"] is True
        assert info["c2pa_manifest"].startswith("C2PA manifest store")  # reader, not chunk
        assert "Black Forest Labs" in info["issuer"]
        assert "trainedAlgorithmicMedia" in info["source_type"]

    def test_extract_info_uses_reader_store(self):
        """The c2pa-python reader path: structured (not heuristic) extraction."""
        from remove_ai_watermarks.noai import c2pa

        assert c2pa.reader_available()
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        # The store-JSON label proves the reader path served this, not the
        # caBX-chunk fallback ("C2PA manifest (...)").
        assert info["c2pa_manifest"].startswith("C2PA manifest store")
        # Structured claim generator is exact, not a CBOR-scanned best-effort.
        assert info["claim_generator"] == "ChatGPT"

    def test_fallback_to_png_parser_when_reader_unavailable(self, monkeypatch):
        """With the reader disabled, the hand-rolled PNG parser still works."""
        from remove_ai_watermarks.noai import c2pa

        monkeypatch.setattr(c2pa, "_C2PA_READER_AVAILABLE", False)
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        assert info["c2pa_manifest"].startswith("C2PA manifest (")  # chunk path
        assert "OpenAI" in info["issuer"]
        assert "trainedAlgorithmicMedia" in info["source_type"]
        assert "synthid_watermark" in info


class TestC2PAInjectValidation:
    def test_inject_rejects_non_png(self, tmp_path):
        with pytest.raises(ValueError, match="only supported for PNG"):
            inject_c2pa_chunk(tmp_path / "in.jpg", tmp_path / "out.png", b"")


# ── CBOR text extraction (parser internals) ─────────────────────────


class TestCborTextAfter:
    """cbor_text_after handles the three CBOR text-string length prefixes."""

    def test_direct_length(self):
        # major-type 3, direct length (0x60 + len). "abc" -> 0x63.
        payload = b"name" + bytes([0x63]) + b"abc"
        assert cbor_text_after(payload, b"name") == "abc"

    def test_one_byte_length(self):
        s = b"x" * 30
        payload = b"name" + bytes([0x78, 30]) + s
        assert cbor_text_after(payload, b"name") == "x" * 30

    def test_two_byte_length(self):
        s = b"y" * 300
        payload = b"name" + bytes([0x79]) + struct.pack(">H", 300) + s
        assert cbor_text_after(payload, b"name") == "y" * 300

    def test_key_not_found_returns_none(self):
        assert cbor_text_after(b"nothing here", b"name") is None

    def test_key_at_end_returns_none(self):
        assert cbor_text_after(b"prefixname", b"name") is None

    def test_invalid_head_returns_none(self):
        # 0x00 is not a text-string head.
        assert cbor_text_after(b"name" + bytes([0x00]) + b"abc", b"name") is None

    def test_latin1_fallback_on_invalid_utf8(self):
        payload = b"name" + bytes([0x61]) + b"\xff"  # len 1, invalid utf-8
        assert cbor_text_after(payload, b"name") is not None


class TestSynthIDVerdict:
    def test_format(self):
        assert synthid_verdict("OpenAI") == "likely present (OpenAI embeds SynthID with C2PA)"

    def test_multiple_vendors(self):
        assert "Google LLC, OpenAI" in synthid_verdict("Google LLC, OpenAI")


class TestParseChunkGuards:
    """_parse_c2pa_chunk rejects non-printable claim_generator garbage.

    On some manifests (observed: Microsoft Designer) the first ``name`` key
    precedes a binary hash field, not the generator string. The clean issuer +
    SynthID verdict must still come through.
    """

    def test_clean_generator_kept(self):
        # "name" + CBOR text-string (head 0x69 = 0x60+9) "gpt-image"
        chunk = b"...name" + bytes([0x69]) + b"gpt-image" + b"OpenAI trainedAlgorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert info["claim_generator"] == "gpt-image"
        assert "OpenAI" in info["issuer"]
        assert "synthid_watermark" in info  # OpenAI + trainedAlgorithmicMedia

    def test_nonprintable_generator_dropped(self):
        # "name" + CBOR string (head 0x64 = len 4) with a control byte -> garbage
        chunk = b"...name" + bytes([0x64]) + b"\x81abc" + b"OpenAI trainedAlgorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert "claim_generator" not in info  # control-char garbage rejected
        assert "OpenAI" in info["issuer"]  # issuer byte-search still robust


class TestC2PADigitalSourceType:
    """The three IPTC digitalSourceType variants drive the AI verdict.

    Only *trained* and *composite-with-trained* mean AI-generated (and so imply
    a SynthID proxy for a SynthID vendor); plain ``algorithmicMedia`` is
    procedural (not trained) and must NOT be flagged as AI.
    """

    def test_plain_algorithmic_media_not_flagged_ai(self):
        chunk = b"...name" + bytes([0x69]) + b"some-tool" + b" OpenAI algorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert info["source_type"] == "algorithmicMedia"
        assert "synthid_watermark" not in info  # procedural, not AI-generated

    def test_composite_with_trained_is_ai_and_synthid(self):
        chunk = b"...name" + bytes([0x69]) + b"some-tool" + b" OpenAI compositeWithTrainedAlgorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert "compositeWithTrainedAlgorithmicMedia" in info["source_type"]
        assert "synthid_watermark" in info  # AI-enhanced + OpenAI issuer


# ── ISOBMFF (AVIF / HEIF / JPEG-XL container stripping) ──────────────

FTYP = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"  # 24-byte ftyp box


class TestISOBMFF:
    def test_is_isobmff_true(self):
        assert is_isobmff(FTYP)

    def test_is_isobmff_false_for_png(self):
        assert not is_isobmff(b"\x89PNG\r\n\x1a\n\x00\x00")

    def test_is_isobmff_false_for_short(self):
        assert not is_isobmff(b"abc")

    def test_strips_jpegxl_jumb_box(self):
        """JPEG-XL stores JUMBF in a ``jumb`` box, always stripped."""
        jumb = struct.pack(">I", 8 + 5) + b"jumb" + b"hello"
        cleaned, stripped = strip_c2pa_boxes(FTYP + jumb)
        assert stripped == 1
        assert cleaned == FTYP

    def test_keeps_non_c2pa_box_with_64bit_size(self):
        """size==1 means a 64-bit largesize follows; non-C2PA box is kept."""
        payload = b"\x00" * 8
        box = b"\x00\x00\x00\x01" + b"free" + struct.pack(">Q", 16 + len(payload)) + payload
        cleaned, stripped = strip_c2pa_boxes(FTYP + box)
        assert stripped == 0
        assert cleaned == FTYP + box

    def test_malformed_box_does_not_crash(self):
        # A box claiming size 4 (< 8-byte header) must terminate iteration safely.
        cleaned, stripped = strip_c2pa_boxes(FTYP + b"\x00\x00\x00\x04XXXX")
        assert stripped == 0
        assert cleaned.startswith(FTYP)

    def test_size_zero_box_runs_to_eof(self):
        # size32==0 means the box extends to EOF; a non-C2PA box round-trips.
        box = struct.pack(">I", 0) + b"free" + b"\x00\x00\x00\x00"
        cleaned, stripped = strip_c2pa_boxes(FTYP + box)
        assert stripped == 0
        assert cleaned == FTYP + box

    def test_truncated_largesize_terminates_safely(self):
        # size32==1 promises a 64-bit largesize, but the box ends after 8 bytes;
        # iteration must stop rather than read the missing largesize past EOF.
        # The walk halts before EOF, so the fail-safe returns the input unchanged
        # (emitting only FTYP would silently truncate the file).
        data = FTYP + b"\x00\x00\x00\x01uuid"
        cleaned, stripped = strip_c2pa_boxes(data)
        assert stripped == 0
        assert cleaned == data

    @staticmethod
    def _avif_with_exif(exif_0th: dict) -> bytes:
        """A fake AVIF (ftyp + mdat) whose mdat carries an EXIF TIFF block, as a
        HEIF/AVIF ``Exif`` meta-box item stores it (bytes in mdat)."""
        import piexif

        blob = piexif.dump({"0th": exif_0th})
        mdat = struct.pack(">I", 8 + len(blob)) + b"mdat" + blob
        return FTYP + mdat

    def test_blank_ai_token_in_exif_item(self):
        import piexif

        data = self._avif_with_exif({piexif.ImageIFD.Software: b"DALL-E", piexif.ImageIFD.Make: b"Canon"})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked == 1
        assert len(out) == len(data)  # same length -> box sizes / iloc stay valid
        assert b"DALL-E" not in out  # AI token destroyed
        assert b"Canon" in out  # camera tag preserved
        # The TIFF structure still parses, with the AI value blanked and Make kept.
        blob = out[out.index(b"Exif\x00\x00") + 6 :]
        ifd = piexif.load(blob)["0th"]
        assert ifd[piexif.ImageIFD.Software].strip() == b""
        assert ifd[piexif.ImageIFD.Make] == b"Canon"

    def test_blank_leaves_clean_exif_untouched(self):
        import piexif

        data = self._avif_with_exif({piexif.ImageIFD.Software: b"Adobe Photoshop", piexif.ImageIFD.Make: b"NIKON"})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked == 0
        assert out == data  # no AI token -> byte-for-byte unchanged

    def test_blank_no_exif_is_noop(self):
        out, blanked = blank_ai_exif_tokens(FTYP + b"\x00\x00\x00\x0cmdat" + b"pixels!!")
        assert blanked == 0
        assert out == FTYP + b"\x00\x00\x00\x0cmdat" + b"pixels!!"


class TestC2PAInvalidSignature:
    """A .png file that is not actually PNG-signed must read as clean, not crash."""

    def test_has_c2pa_false_for_non_png_bytes(self, tmp_path: Path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"\xff\xd8\xff\xe0 not a png at all, just garbage bytes")
        assert has_c2pa_metadata(fake) is False

    def test_extract_chunk_none_for_non_png_bytes(self, tmp_path: Path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"\xff\xd8\xff\xe0 not a png at all, just garbage bytes")
        assert extract_c2pa_chunk(fake) is None
