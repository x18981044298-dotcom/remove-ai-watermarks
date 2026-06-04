"""Post-processing filters for the cleaned output.

``apply_analog_humanizer`` injects film grain and chromatic aberration to defeat
digital AI-perfection classifiers (ported from NeuralBleach); ``unsharp_mask``
counters the soft, over-smoothed look that diffusion + face-restoration leave
behind (itself a common "this is AI" tell).
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
import cv2
import numpy as np
from numpy.typing import NDArray


def apply_analog_humanizer(image: NDArray, grain_intensity: float = 4.0, chromatic_shift: int = 1) -> NDArray:
    """
    Apply Analog Humanizer (film grain and chromatic aberration) to an image.
    This simulates analog film imperfections to defeat digital AI perfection classifiers.

    Ported from NeuralBleach.

    Args:
        image: BGR image as numpy array (uint8).
        grain_intensity: Standard deviation of the Gaussian noise (film grain).
        chromatic_shift: Number of pixels to shift the red/blue color channels.

    Returns:
        Humanized BGR image.
    """
    # Ensure image is BGR
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image.copy()

    # Split channels (OpenCV uses BGR)
    # B = 0, G = 1, R = 2
    b, g, r = cv2.split(image)

    # 1. Chromatic Aberration
    # Shift R channel left, B channel right. np.roll is circular, so it wraps
    # the opposite edge into a thin colored fringe at the L/R borders; replicate
    # the original edge columns there to keep the intended offset interior-only.
    if chromatic_shift > 0:
        r = np.roll(r, -chromatic_shift, axis=1)
        r[:, -chromatic_shift:] = r[:, -chromatic_shift - 1 : -chromatic_shift]
        b = np.roll(b, chromatic_shift, axis=1)
        b[:, :chromatic_shift] = b[:, chromatic_shift : chromatic_shift + 1]

    merged = cv2.merge((b, g, r))

    # 2. Film Grain (Gaussian Noise)
    if grain_intensity > 0:
        img_f = merged.astype(np.float32)
        noise = np.random.normal(0, grain_intensity, img_f.shape).astype(np.float32)
        humanized = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    else:
        humanized = merged

    return humanized


def unsharp_mask(image: NDArray, amount: float = 0.5, sigma: float = 1.0) -> NDArray:
    """Sharpen via unsharp masking: ``out = image + amount * (image - blur(image))``.

    Counters the soft, over-smoothed look of the diffusion + GFPGAN passes, which
    reads as an AI tell. ``amount`` 0 = no-op (returns an unchanged copy); ~0.5-0.8
    is a safe range -- higher risks bright edge halos that are their own artifact.
    ``sigma`` is the Gaussian radius of the unsharp kernel.

    Args:
        image: BGR image as numpy array (uint8).
        amount: Sharpening strength (0 = off).
        sigma: Gaussian blur sigma for the unsharp kernel.

    Returns:
        Sharpened BGR image (uint8).
    """
    if amount <= 0.0:
        return image.copy()
    img_f = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img_f, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
