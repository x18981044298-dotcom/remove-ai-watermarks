"""Tests for AI metadata detection and removal."""

from __future__ import annotations

from pathlib import Path

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.metadata import (
    _is_ai_key,
    exif_generator,
    get_ai_metadata,
    has_ai_metadata,
    remove_ai_metadata,
    synthid_source,
)

# Real, committed C2PA sample images used to ground the SynthID-source tests.
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# ── Key detection ───────────────────────────────────────────────────


class TestIsAiKey:
    """Tests for _is_ai_key helper."""

    def test_exact_match_lowercase(self):
        assert _is_ai_key("parameters")

    def test_exact_match_mixed_case(self):
        assert _is_ai_key("Parameters")

    def test_keyword_substring(self):
        assert _is_ai_key("stable_diffusion_model_v2")

    def test_c2pa_detected(self):
        assert _is_ai_key("c2pa_chunk")

    def test_standard_key_not_flagged(self):
        assert not _is_ai_key("Author")

    def test_innocuous_key_not_flagged(self):
        assert not _is_ai_key("Title")

    def test_dpi_not_flagged(self):
        assert not _is_ai_key("dpi")


# ── has_ai_metadata / get_ai_metadata ───────────────────────────────


class TestHasAiMetadata:
    """Tests for detecting AI metadata in images."""

    def test_detects_ai_metadata(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_clean_image_no_ai(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_detects_c2pa_uuid_in_isobmff_container(self, tmp_path: Path):
        """C2PA in AVIF/HEIF/MP4 lives in a ``uuid`` box identified by a fixed UUID.

        Real AVIF/HEIF fixtures aren't shipped, so simulate the container by
        prepending an ISOBMFF-shaped ftyp box and the C2PA UUID bytes.
        """
        from remove_ai_watermarks.metadata import C2PA_UUID

        path = tmp_path / "fake.avif"
        # ftyp box: size(4) + 'ftyp' + 'avif' + minor_version(4) + 'avif'
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"jumb-payload"
        path.write_bytes(ftyp + uuid_box + b"\x00" * 64)
        assert has_ai_metadata(path)

    def test_strip_c2pa_boxes_removes_uuid_box(self, tmp_path: Path):
        """ISOBMFF strip should drop the C2PA uuid box and keep everything else."""
        from remove_ai_watermarks.metadata import C2PA_UUID
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload (8 bytes -> total 32)
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        cleaned, stripped = strip_c2pa_boxes(ftyp + uuid_box + mdat)
        assert stripped == 1
        assert cleaned == ftyp + mdat

    def test_strip_c2pa_boxes_passthrough_for_non_isobmff(self):
        """Non-ISOBMFF input must be returned unchanged."""
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
        cleaned, stripped = strip_c2pa_boxes(data)
        assert stripped == 0
        assert cleaned == data

    def test_remove_ai_metadata_strips_c2pa_in_avif(self, tmp_path: Path):
        """End-to-end: ``remove_ai_metadata`` on a fake .avif drops the C2PA box."""
        from remove_ai_watermarks.metadata import C2PA_UUID, remove_ai_metadata

        src = tmp_path / "in.avif"
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        src.write_bytes(ftyp + uuid_box + mdat)

        out = tmp_path / "out.avif"
        result = remove_ai_metadata(src, out)
        assert result == out
        assert out.read_bytes() == ftyp + mdat
        # And after stripping, detection must no longer flag the cleaned file.
        from remove_ai_watermarks.metadata import has_ai_metadata

        assert not has_ai_metadata(out)

    def test_detects_iptc_trained_algorithmic_media_marker(self, tmp_path: Path):
        """Some pipelines embed only the IPTC AI marker in XMP, no C2PA manifest."""
        path = tmp_path / "fake.jpg"
        # Minimal JPEG-ish bytes containing the IPTC AI marker in an XMP-like blob.
        xmp = (
            b"<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>"
            b"trainedAlgorithmicMedia"
            b"</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>"
        )
        path.write_bytes(b"\xff\xd8\xff\xe1" + xmp + b"\xff\xd9")
        assert has_ai_metadata(path)


class TestGetAiMetadata:
    """Tests for extracting AI metadata."""

    def test_extracts_parameters_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta
        assert "Euler" in meta["parameters"]

    def test_extracts_prompt_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "prompt" in meta

    def test_does_not_extract_author(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "Author" not in meta

    def test_clean_image_empty_dict(self, tmp_clean_png):
        meta = get_ai_metadata(tmp_clean_png)
        assert meta == {}

    def test_long_value_is_truncated(self, tmp_path: Path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "x" * 300)
        path = tmp_path / "long.png"
        img.save(path, pnginfo=pnginfo)
        meta = get_ai_metadata(path)
        assert meta["parameters"].endswith("…")
        assert len(meta["parameters"]) <= 205

    def test_unopenable_file_does_not_raise(self, tmp_path: Path):
        # PIL can't open HEIC without pillow-heif; get_ai_metadata must fall
        # through to the binary scan, not propagate UnidentifiedImageError.
        path = tmp_path / "iphone.heic"
        path.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
        assert get_ai_metadata(path) == {}


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestGetAiMetadataRealSample:
    """get_ai_metadata surfaces the consolidated C2PA fields on real images."""

    def test_openai_sample_fields(self):
        meta = get_ai_metadata(SAMPLES_DIR / "chatgpt-1.png")
        assert "claim_generator" in meta
        assert "OpenAI" in meta["issuer"]
        assert "OpenAI" in meta["synthid_watermark"]
        assert "trainedAlgorithmicMedia" in meta["source_type"]


@pytest.mark.parametrize(
    "marker",
    [
        b"trainedAlgorithmicMedia",
        b"compositeSynthetic",
        b"algorithmicMedia",
        b"compositeWithTrainedAlgorithmicMedia",
    ],
)
def test_has_ai_metadata_detects_each_iptc_marker(tmp_path: Path, marker: bytes):
    """Each IPTC digitalSourceType AI marker in XMP triggers detection."""
    path = tmp_path / "iptc.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe1<x:xmpmeta>" + marker + b"</x:xmpmeta>\xff\xd9")
    assert has_ai_metadata(path)


# ── SynthID-source detection (metadata proxy) ────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestSynthIDSource:
    """SynthID detection via the C2PA companion manifest.

    Google (Imagen/Gemini) and OpenAI (ChatGPT/DALL-E/gpt-image) pair an
    invisible SynthID pixel watermark with a C2PA manifest. Adobe Firefly and
    Microsoft Designer sign C2PA Content Credentials but do NOT use SynthID,
    so the discriminating signal is the C2PA *issuer*, not the mere presence
    of a manifest. These tests run against real, committed sample images.
    """

    def test_openai_chatgpt_is_synthid_source(self):
        assert synthid_source(SAMPLES_DIR / "chatgpt-1.png") == "OpenAI"

    def test_openai_verdict_in_get_ai_metadata(self):
        meta = get_ai_metadata(SAMPLES_DIR / "chatgpt-1.png")
        assert "synthid_watermark" in meta
        assert "OpenAI" in meta["synthid_watermark"]

    def test_adobe_firefly_is_not_synthid_source(self):
        # Adobe signs C2PA (trainedAlgorithmicMedia) but embeds no SynthID.
        assert synthid_source(SAMPLES_DIR / "firefly-1.png") is None
        assert "synthid_watermark" not in get_ai_metadata(SAMPLES_DIR / "firefly-1.png")

    def test_non_ai_image_is_not_synthid_source(self):
        assert synthid_source(SAMPLES_DIR / "not-ai-1.jpeg") is None


class TestSynthIDSourceNonPng:
    """SynthID-source detection must work beyond PNG.

    ChatGPT/Gemini images saved as JPEG/WebP/AVIF carry their C2PA manifest in
    a non-PNG container (JPEG APP11, ISOBMFF uuid box), so the PNG caBX parser
    misses them. These use synthetic byte blobs (real fixtures aren't shipped).
    """

    def _c2pa_jpeg(self, tmp_path: Path, name: str, issuer: bytes, marker: bytes = b"trainedAlgorithmicMedia") -> Path:
        path = tmp_path / name
        # Minimal JPEG shell with an embedded C2PA-ish blob.
        blob = b"jumbc2pa" + issuer + b"..." + marker
        path.write_bytes(b"\xff\xd8\xff\xe1" + blob + b"\xff\xd9")
        return path

    def test_openai_c2pa_in_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, "chatgpt.jpg", b"OpenAI")
        assert synthid_source(path) == "OpenAI"

    def test_google_c2pa_in_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, "gemini.jpg", b"Google")
        assert synthid_source(path) == "Google LLC"

    def test_adobe_c2pa_in_jpeg_is_none(self, tmp_path: Path):
        # Adobe signs C2PA but embeds no SynthID.
        path = self._c2pa_jpeg(tmp_path, "firefly.jpg", b"Adobe")
        assert synthid_source(path) is None

    def test_openai_without_ai_marker_is_none(self, tmp_path: Path):
        # Issuer present but no AI digital-source marker -> not a SynthID source.
        path = self._c2pa_jpeg(tmp_path, "edited.jpg", b"OpenAI", marker=b"")
        assert synthid_source(path) is None


# ── remove_ai_metadata ──────────────────────────────────────────────


class TestRemoveAiMetadata:
    """Tests for stripping AI metadata."""

    def test_removes_ai_keys(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output)

        with Image.open(output) as img:
            assert "parameters" not in img.info
            assert "prompt" not in img.info

    def test_keeps_standard_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=True)

        with Image.open(output) as img:
            assert "Author" in img.info
            assert img.info["Author"] == "Test Author"

    def test_remove_all_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=False)
        with Image.open(output) as img:
            assert "Author" not in img.info
            assert "parameters" not in img.info

    def test_overwrite_in_place(self, tmp_path):
        """When output_path is None, should overwrite source."""
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test data")
        path = tmp_path / "inplace.png"
        img.save(path, pnginfo=pnginfo)

        result = remove_ai_metadata(path)
        assert result == path

        with Image.open(path) as cleaned:
            assert "parameters" not in cleaned.info

    def test_jpeg_output(self, tmp_path):
        """Test metadata removal for JPEG format."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test")
        png_path = tmp_path / "source.png"
        img.save(png_path, pnginfo=pnginfo)

        jpg_path = tmp_path / "output.jpg"
        result = remove_ai_metadata(png_path, jpg_path)
        assert result == jpg_path
        assert jpg_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", "test")
        path = tmp_path / "source.png"
        img.save(path, pnginfo=pnginfo)

        output = tmp_path / "sub" / "dir" / "cleaned.png"
        remove_ai_metadata(path, output)
        assert output.exists()

    def test_returns_path(self, tmp_clean_png):
        output = tmp_clean_png.parent / "out.png"
        result = remove_ai_metadata(tmp_clean_png, output)
        assert isinstance(result, Path)
        assert result == output


def _img_with_software(tmp_path: Path, fmt: str, software: str) -> Path:
    """Write a tiny image carrying an EXIF Software tag."""
    exif = piexif.dump({"0th": {piexif.ImageIFD.Software: software.encode()}, "Exif": {}, "GPS": {}, "1st": {}})
    path = tmp_path / f"img.{fmt}"
    Image.new("RGB", (64, 64), (100, 90, 80)).save(path, exif=exif)
    return path


class TestExifGenerator:
    """exif_generator extracts AI-tool names from EXIF/XMP across formats."""

    def test_avif_software_ai_tool_detected(self, tmp_path: Path):
        path = _img_with_software(tmp_path, "avif", "Adobe Firefly")
        assert exif_generator(path) == "Adobe Firefly"

    def test_jpeg_software_ai_tool_detected(self, tmp_path: Path):
        path = _img_with_software(tmp_path, "jpg", "ComfyUI v1.2")
        result = exif_generator(path)
        assert result is not None
        assert "ComfyUI" in result

    def test_plain_editor_not_flagged(self, tmp_path: Path):
        # An ordinary editor tag carries no AI token and must not be flagged.
        path = _img_with_software(tmp_path, "jpg", "Adobe Photoshop 25.0")
        assert exif_generator(path) is None

    def test_make_tag_ai_tool_detected(self, tmp_path: Path):
        # Ideogram tags its output with EXIF Make="Ideogram AI" (verified on a
        # real download), so the Make tag must be read too.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Ideogram AI"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "ideogram.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) == "Ideogram AI"

    def test_camera_make_not_flagged(self, tmp_path: Path):
        # A real camera Make ("Apple") carries no AI token -> not flagged.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Apple"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "iphone.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) is None

    def test_xmp_creatortool_scan_covers_unopenable(self, tmp_path: Path):
        # PIL can't open this fake HEIF; the raw XMP CreatorTool scan still works.
        path = tmp_path / "fake.heic"
        path.write_bytes(
            b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"
            b"<x:xmpmeta><xmp:CreatorTool>Midjourney v7</xmp:CreatorTool></x:xmpmeta>"
        )
        result = exif_generator(path)
        assert result is not None
        assert "Midjourney" in result

    def test_clean_image_is_none(self, tmp_clean_png: Path):
        assert exif_generator(tmp_clean_png) is None


class TestAIGCLabel:
    """China TC260 AIGC labeling (Doubao and other China-served generators)."""

    def _aigc_png(self, tmp_path: Path, label: str = "1", producer: str = "TESTPRODUCER001") -> Path:
        from remove_ai_watermarks.metadata import aigc_label  # noqa: F401  (import-time guard)

        p = tmp_path / "doubao.png"
        Image.new("RGB", (32, 32)).save(p)
        # XMP is HTML-entity encoded in real files; aigc_label must unescape it.
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" '
            'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"><TC260:AIGC>'
            f"{{&quot;Label&quot;:&quot;{label}&quot;,&quot;ContentProducer&quot;:&quot;{producer}&quot;}}"
            "</TC260:AIGC></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        with open(p, "ab") as f:
            f.write(xmp.encode())
        return p

    def test_parses_label_and_producer(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_png(tmp_path))
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"] == "TESTPRODUCER001"

    def test_none_when_absent(self, tmp_clean_png):
        from remove_ai_watermarks.metadata import aigc_label

        assert aigc_label(tmp_clean_png) is None

    def test_has_ai_metadata_detects_aigc(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_png(tmp_path))

    def test_get_ai_metadata_surfaces_aigc(self, tmp_path: Path):
        meta = get_ai_metadata(self._aigc_png(tmp_path))
        assert "aigc_label" in meta
        assert "TC260" in meta["aigc_label"]
