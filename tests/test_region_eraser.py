"""Tests for the universal region eraser."""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks.region_eraser import boxes_to_mask, erase, lama_available


class TestBoxesToMask:
    def test_mask_set_inside_box(self):
        mask = boxes_to_mask((100, 100), [(10, 20, 30, 40)], dilate=0)
        assert mask[25, 15] == 255  # inside
        assert mask[0, 0] == 0  # outside
        assert mask.shape == (100, 100)

    def test_multiple_boxes(self):
        mask = boxes_to_mask((100, 100), [(0, 0, 10, 10), (90, 90, 10, 10)], dilate=0)
        assert mask[5, 5] == 255
        assert mask[95, 95] == 255
        assert mask[50, 50] == 0

    def test_dilate_grows_mask(self):
        m0 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=0)
        m5 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=5)
        assert m5.sum() > m0.sum()

    def test_box_clipped_to_bounds(self):
        # box partly outside the image must not raise and stays in-bounds
        mask = boxes_to_mask((50, 50), [(40, 40, 100, 100)], dilate=0)
        assert mask[45, 45] == 255


class TestEraseCv2:
    def _image_with_logo(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        img = np.full((200, 200, 3), 120, np.uint8)  # flat gray background
        box = (140, 160, 50, 30)
        x, y, w, h = box
        img[y : y + h, x : x + w] = (255, 255, 255)  # bright "logo"
        return img, box

    def test_erase_changes_region(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2")
        x, y, w, h = box
        # on a flat background the logo region should be repainted near gray
        region = out[y : y + h, x : x + w]
        assert abs(float(region.mean()) - 120) < 20
        assert not np.array_equal(out, img)

    def test_pixels_outside_box_untouched(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2", dilate=0)
        # a far corner must be identical
        assert np.array_equal(img[:50, :50], out[:50, :50])

    def test_no_boxes_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, boxes=[], backend="cv2")
        assert np.array_equal(img, out)

    def test_empty_mask_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, mask=np.zeros((100, 100), np.uint8), backend="cv2")
        assert np.array_equal(img, out)


class TestNonBgrInputs:
    """cv2.inpaint rejects 4-channel BGRA and 2D-only entry points must work."""

    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="cv2")
        assert out.shape == gray.shape

    def test_bgra_preserves_alpha_and_does_not_raise(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200  # opaque-ish alpha plane
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="cv2", dilate=0)
        assert out.shape == bgra.shape
        # alpha plane is carried through unchanged
        assert np.array_equal(out[..., 3], bgra[..., 3])


class TestLamaBackend:
    def test_lama_raises_when_unavailable(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        if lama_available():
            pytest.skip("onnxruntime installed; cannot test the unavailable path")
        with pytest.raises(RuntimeError, match="onnxruntime"):
            erase(img, boxes=[(10, 10, 20, 20)], backend="lama")


class TestLamaChannelHandling:
    """erase_lama must accept grayscale (2D) and BGRA (4-channel) like erase_cv2.

    The real ONNX model is never loaded -- the session is faked to an identity
    inpaint, so this exercises only the channel promote/split wrapper (the fix for
    LaMa crashing on grayscale and dropping alpha on BGRA).
    """

    @pytest.fixture
    def _fake_lama(self, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import region_eraser

        class _In:
            def __init__(self, name: str, shape: list[int]):
                self.name = name
                self.shape = shape

        class _FakeSession:
            def get_inputs(self):
                return [_In("image", [1, 3, 512, 512]), _In("mask", [1, 1, 512, 512])]

            def run(self, _outputs, feeds):
                # Identity inpaint: echo the image tensor (1,3,size,size) back.
                return [feeds["image"]]

        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        monkeypatch.setattr(region_eraser, "_get_lama_session", lambda: _FakeSession())

    @pytest.mark.usefixtures("_fake_lama")
    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="lama")
        assert out.ndim == 2
        assert out.shape == gray.shape

    @pytest.mark.usefixtures("_fake_lama")
    def test_bgra_preserves_alpha(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200  # opaque-ish alpha plane
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="lama")
        assert out.shape == bgra.shape
        assert np.array_equal(out[..., 3], bgra[..., 3])  # alpha carried through unchanged
