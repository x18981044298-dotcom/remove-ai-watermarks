import numpy as np

from remove_ai_watermarks.humanizer import apply_analog_humanizer, unsharp_mask


def test_humanizer_does_not_modify_original_if_disabled():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50, 50] = [100, 150, 200]
    org_img = img.copy()

    # grain=0, shift=0 means disabled — result should match original.
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=0)
    assert np.array_equal(result, org_img)


def test_chromatic_shift():
    # Only green channel is centered, red/blue should shift.
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[2, 2] = [255, 255, 255]  # B, G, R

    # shift=1
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=1)

    # G (index 1) stays at [2,2]
    assert result[2, 2, 1] == 255
    # B (index 0) shifted right (+1 axis 1) -> [2, 3]
    assert result[2, 3, 0] == 255
    # R (index 2) shifted left (-1 axis 1) -> [2, 1]
    assert result[2, 1, 2] == 255


def test_grain_intensity():
    # Gray image
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Add strong noise
    result = apply_analog_humanizer(img, grain_intensity=10.0, chromatic_shift=0)

    # Image should no longer be purely 128
    unique_vals = np.unique(result)
    assert len(unique_vals) > 5

    # Mean should roughly be 128
    assert 126 < np.mean(result) < 130


def test_invalid_shape():
    # Missing color channel
    img = np.zeros((100, 100), dtype=np.uint8)
    img[0, 0] = 50
    result = apply_analog_humanizer(img)
    assert np.array_equal(img, result)


def test_chromatic_shift_does_not_wrap_opposite_edge():
    # On a horizontal gradient (dark left, bright right), a circular np.roll
    # would wrap the bright right edge into the R channel's left border and the
    # dark left edge into the B channel's right border, producing a colored
    # fringe. After the fix the border columns must replicate their own edge.
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    gray = np.broadcast_to(ramp, (32, 64))
    img = np.stack([gray, gray, gray], axis=2).copy()  # B, G, R

    shift = 3
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=shift)

    # B (index 0) rolled right -> its left border must stay dark (near 0),
    # NOT wrap the bright right edge.
    assert result[:, :shift, 0].max() < 60
    # R (index 2) rolled left -> its right border must stay bright (near 255),
    # NOT wrap the dark left edge.
    assert result[:, -shift:, 2].min() > 195


def test_unsharp_disabled_returns_unchanged_copy():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    img[10, 10] = [100, 150, 200]
    result = unsharp_mask(img, amount=0.0)
    assert np.array_equal(result, img)
    assert result is not img  # a fresh copy, never the same array


def test_unsharp_overshoots_at_an_edge():
    # A vertical step (left 100, right 150). Unsharp masking overshoots at the
    # boundary, pushing pixels above the bright level and below the dark level.
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    img[:, 10:] = 150
    result = unsharp_mask(img, amount=1.0, sigma=1.5)
    assert int(result.max()) > 150  # bright-side overshoot
    assert int(result.min()) < 100  # dark-side undershoot


def test_unsharp_preserves_shape_and_dtype():
    img = np.full((15, 25, 3), 120, dtype=np.uint8)
    result = unsharp_mask(img, amount=0.6)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_unsharp_flat_image_is_a_noop():
    # No edges -> blur equals the image -> unsharp cancels to the original.
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    result = unsharp_mask(img, amount=0.8, sigma=1.0)
    assert np.array_equal(result, img)
