"""Doubao visible watermark removal engine.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Like the Gemini sparkle and the Jimeng wordmark, it is a fixed overlay, so removal
starts from **reverse-alpha blending** against a captured alpha map
(``remove_watermark_reverse_alpha``): ``original = (wm - a*logo)/(1-a)``. The alpha
map is rebuilt by ``scripts/visible_alpha_solve.py`` from black/gray Doubao captures
(the careful gray-self solve; logo is pure white) and bundled as
``assets/doubao_alpha.png``. The mark re-rasterizes a few px off per image, so
removal ALWAYS NCC-aligns the template to the actual mark and then clears the
residual edges with a deliberately THIN inpaint over the glyph footprint (an
earlier under-estimated alpha + fixed-no-inpaint left a readable outline that the
detector did not flag -- see the reverse-alpha section below).

Detection (``detect``) is shape-consistent: it matches that same alpha glyph
silhouette against the corner via normalized correlation, so it keys on the actual
"豆包AI生成" shape rather than coverage/structure heuristics.

``locate`` (geometry box, scales with image WIDTH) and ``extract_mask`` (the
candidate glyph mask the detector correlates) mirror the Jimeng engine.
Arbitrary-region inpainting still lives in ``region_eraser`` / the ``erase``
command. Fast, offline, no GPU.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Geometry as a fraction of image WIDTH. The Doubao mark scales with width and
# is anchored bottom-right. The box must be GENEROUSLY wider than the mark and
# reach close to the corner -- the mark is re-rasterized a few px off per image,
# and the NCC alignment search only registers within this box, so a tight box
# (the old 0.185 / margin 0.012) let a corner-ward shift fall partly outside it
# and the alignment missed. The glyph mask tightens the actual removal.
WM_WIDTH_FRAC = 0.22
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.004

# Glyph appearance: the label is a low-saturation light gray, rendered brighter
# than the surrounding content (the common case: a generated photo/illustration).
# We detect it as a local bright feature (white top-hat: brighter than a blurred
# local background) intersected with the grayish + minimum-brightness tests.
# This is polarity-correct for bright-on-darker backgrounds and, crucially,
# leaves white-paper documents untouched (there the mark is not brighter than
# its surroundings, so nothing is masked rather than damaging the document text).
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Detection is reverse-alpha-consistent: the mark is recognized by matching the
# bundled alpha-template glyph silhouette (assets/doubao_alpha.png -- the exact
# shape we invert) against the extracted candidate mask via zero-mean normalized
# correlation (cv2 TM_CCOEFF_NORMED). It keys on the actual "豆包AI生成" glyph
# SHAPE, not on coverage/structure heuristics, so a merely-textured corner does
# not fire (the old coverage detector false-positived on ~28% of images; #23).
# Corpus-tuned: real marks score median ~0.61, arbitrary corners <=0.17 (p99);
# threshold 0.4 -> false positives 7/1243 (0.6%). A small coverage floor skips
# the template match on a near-empty candidate box.
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.4

# ── Reverse-alpha (recovery + thin residual inpaint) ─────────────────
# The Doubao mark is a fixed semi-transparent white overlay, so given its alpha
# map the original pixels are recovered by inverting the blend: (wm - a*logo)/(1-a).
# The alpha map is rebuilt by scripts/visible_alpha_solve.py from the black/gray
# Doubao captures (data/doubao_capture/): the CAREFUL solve -- a = (I - B)/(255 - B)
# on the gray capture with B a per-channel cubic background fit, mean over channels,
# full halo extent, unblurred. The earlier build (a coarser solve) under-estimated
# the alpha and left a clearly READABLE "豆包AI生成" outline on real samples
# (issue #13 follow-up: the detector was fooled by the outline -- conf 0.0 -- so the
# test passed while the result was visibly bad; suspect the captured alpha map, not
# the method). The mark is re-rasterized and a few px off per image, so removal
# does NOT trust fixed geometry: it ALWAYS tries fixed AND `_aligned_alpha_map`'s
# TM_CCOEFF_NORMED scale+position search and keeps the lower-residual placement,
# then a deliberately THIN residual inpaint clears the leftover edges without
# smearing the recovered texture. Geometry below is emitted by the solver -- keep in
# sync when the asset is rebuilt.
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_LOGO_BGR: tuple[float, float, float] = (255.0, 255.0, 255.0)
_ALPHA_WIDTH_FRAC = 0.1636  # asset width / image width -- the alignment scale seed
_ALPHA_HEIGHT_FRAC = 0.0405
# Margins (of image WIDTH) of the captured mark -- the geometry record / where to
# seed; alignment refines the actual position, so these are not load-bearing.
_ALPHA_MARGIN_RIGHT_FRAC = 0.0132
_ALPHA_MARGIN_BOTTOM_FRAC = 0.0166
# Alignment scale search (np.linspace args) around the width-scaled glyph size.
_ALPHA_ALIGN_SEARCH = (0.88, 1.12, 25)
# Residual inpaint over the glyph footprint -- thin (NS, small radius) so it clears
# the leftover edges without the smear a wide full-footprint pass caused.
_RESIDUAL_ALPHA_FLOOR = 0.05
_RESIDUAL_DILATE = 5
_RESIDUAL_INPAINT_RADIUS = 2
_alpha_template_cache: NDArray[Any] | None = None


def _alpha_template() -> NDArray[Any] | None:
    """Lazily load the bundled Doubao alpha template (float [0,1]), or None."""
    global _alpha_template_cache
    if _alpha_template_cache is None:
        from pathlib import Path

        from remove_ai_watermarks import image_io

        path = Path(__file__).parent / "assets" / "doubao_alpha.png"
        img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _alpha_template_cache = img.astype(np.float32) / 255.0
    return _alpha_template_cache


@dataclass(frozen=True)
class DoubaoLocation:
    """Located watermark box (bottom-right), in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class DoubaoDetection:
    """Result of visible Doubao watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


_silhouette_cache: NDArray[Any] | None = None


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the bundled alpha map,
    used as the detection template. None if the alpha asset is missing."""
    global _silhouette_cache
    if _silhouette_cache is None:
        at = _alpha_template()
        if at is None:
            return None
        _silhouette_cache = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache


def _template_match_score(box_mask: NDArray[Any], image_width: int) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``.

    TM_CCOEFF_NORMED keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual "豆包AI生成" shape does.
    """
    sil = _glyph_silhouette()
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(8, int(_ALPHA_WIDTH_FRAC * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(_ALPHA_HEIGHT_FRAC * image_width)))
    if gw < 8 or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class DoubaoEngine:
    """Remove the visible Doubao "豆包AI生成" watermark (locate -> mask -> inpaint)."""

    def __init__(
        self,
        *,
        width_frac: float = WM_WIDTH_FRAC,
        height_frac: float = WM_HEIGHT_FRAC,
        margin_right_frac: float = MARGIN_RIGHT_FRAC,
        margin_bottom_frac: float = MARGIN_BOTTOM_FRAC,
    ) -> None:
        self.width_frac = width_frac
        self.height_frac = height_frac
        self.margin_right_frac = margin_right_frac
        self.margin_bottom_frac = margin_bottom_frac

    # ── Locate ────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> DoubaoLocation:
        """Anchor the watermark box in the bottom-right corner by geometry."""
        h, w = image.shape[:2]
        wm_w = max(40, int(w * self.width_frac))
        wm_h = max(16, int(w * self.height_frac))
        margin_r = max(4, int(w * self.margin_right_frac))
        margin_b = max(4, int(w * self.margin_bottom_frac))
        x = max(0, w - margin_r - wm_w)
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return DoubaoLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ──────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: DoubaoLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = watermark glyph) for the box.

        Polarity-aware: the mark is a light, low-saturation gray. On a dark
        background it is the bright region; on a light background it is the
        off-white gray below paper-white. Both cases are captured by the logo
        luminance band intersected with the grayish constraint, plus a
        brighter-than-local-background test on dark backgrounds.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault the native code on some platforms (observed: a Windows
        # access violation via the always-align removal's residual `detect`). Skip
        # the cv2 pipeline and return an empty mask there.
        if bh < 16 or bw < 16:
            return np.zeros((h, w), np.uint8)
        # Normalize the ROI to 3-channel BGR: a 2D grayscale or 4-channel BGRA
        # input would otherwise break the axis=2 channel reductions below.
        roi = image[y : y + bh, x : x + bw]
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        roi = roi.astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        # Local background model: a strong Gaussian blur (sigma ~ box height)
        # approximates the content under the glyphs. The white top-hat
        # (luma - local_bg) lights up bright thin strokes regardless of the
        # absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > TOPHAT_DELTA) & (luma > LOGO_MIN_LUMA)
        glyph = cand.astype(np.uint8) * 255
        # Connect glyph parts, then drop isolated specks (5x5 open clears the
        # scattered grayish pixels that random/textured corners produce).
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> DoubaoDetection:
        """Detect the visible Doubao mark by matching the alpha-template glyph
        silhouette against the corner candidate (TM_CCOEFF_NORMED).

        Keys on the "豆包AI生成" SHAPE, not coverage, so a textured corner does
        not fire. ``confidence`` is the correlation score; ``detected`` is it
        clearing ``DETECT_NCC_THRESHOLD``.
        """
        det = DoubaoDetection()
        if image is None or image.size == 0:
            return det
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        box = mask[y : y + bh, x : x + bw]
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        if coverage >= DETECT_MIN_COVERAGE:
            score = _template_match_score(box, image.shape[1])
            det.confidence = score
            det.detected = score >= DETECT_NCC_THRESHOLD
            logger.debug("Doubao detect: coverage=%.3f ncc=%.2f detected=%s", coverage, score, det.detected)
        return det

    # ── Reverse-alpha (exact recovery) ────────────────────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable. Sub-pixel NCC alignment
        (see ``_aligned_alpha_map``) places it on the actual mark at ANY
        resolution, so there is no width gate -- the caller still gates on
        ``detect`` so a clean corner is never touched."""
        return image is not None and image.size > 0 and _alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry -- pixel-exact at
        the captured width (used there instead of integer-pixel NCC alignment)."""
        at = _alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        # Glyph box scales with WIDTH; on a wide/short image the height-from-width
        # box can exceed the image height. Clamp both dims so the slice assignment
        # below cannot overflow (a degenerate 2048x1 input otherwise raised
        # ValueError on the broadcast). Normal images are unaffected.
        gw = min(w, max(1, int(_ALPHA_WIDTH_FRAC * w)))
        gh = min(h, max(1, int(_ALPHA_HEIGHT_FRAC * w)))
        ax = max(0, w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw)
        ay = max(0, h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Build a full-image alpha map with the captured template registered to
        the actual mark via a TM_CCOEFF_NORMED scale + position search -- so the
        single capture works off the captured width (a pure width-scale ghosts).
        Returns ``(alpha_map, glyph_bbox)`` or None."""
        at = _alpha_template()
        sil = _glyph_silhouette()
        if at is None or sil is None:
            return None
        h, w = image.shape[:2]
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        box_mask = self.extract_mask(image, loc)[by : by + bh, bx : bx + bw]
        expected = _ALPHA_WIDTH_FRAC * w
        best: tuple[float, int, int, int, int] | None = None
        for scale in np.linspace(*_ALPHA_ALIGN_SEARCH):
            gw, gh = int(expected * scale), int(_ALPHA_HEIGHT_FRAC * w * scale)
            if gw < 8 or gh < 4 or gw >= bw or gh >= bh:
                continue
            t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
            _, score, _, top_left = cv2.minMaxLoc(cv2.matchTemplate(box_mask, t, cv2.TM_CCOEFF_NORMED))
            if best is None or score > best[0]:
                best = (score, gw, gh, top_left[0], top_left[1])
        if best is None:
            return None
        _, gw, gh, ox, oy = best
        ax, ay = bx + ox, by + oy
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _apply_reverse_alpha(self, image: NDArray[Any], amap: NDArray[Any]) -> NDArray[Any]:
        """Invert the alpha blend with ``amap``: ``original = (wm - a*logo)/(1-a)``."""
        a3 = np.clip(amap, 0.0, 1.0)[:, :, None]
        logo = np.array(_ALPHA_LOGO_BGR, np.float32)
        return np.clip((image.astype(np.float32) - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255).astype(np.uint8)

    def remove_watermark_reverse_alpha(self, image: NDArray[Any], *, residual_inpaint: bool = True) -> NDArray[Any]:
        """Recover the original pixels by inverting the alpha blend
        ``original = (wm - a*logo)/(1-a)``, then clear the residual edges with a
        thin inpaint over the glyph footprint.

        Placement: fixed geometry AND the NCC-aligned placement are always tried and
        the one leaving the least residual mark (lowest re-``detect`` confidence) is
        kept -- the mark is re-rasterized and a few px off per image, so fixed
        geometry alone leaves a visible outline (it did on the doubao-1.png sample).
        A single capture cannot pixel-cancel the mark on every image, so a
        deliberately THIN residual inpaint (``_RESIDUAL_*``) follows: reverse-alpha
        has already recovered the true background under the mark, so the inpaint only
        finishes the leftover edges instead of smearing the whole footprint.
        Call only when :meth:`reverse_alpha_available` and the mark is detected.
        """
        # Normalize to 3-channel BGR so a 2D grayscale or 4-channel BGRA input
        # does not break the reverse-alpha math (which assumes a 3-channel logo).
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        maps = [c for c in (self._fixed_alpha_map(image), self._aligned_alpha_map(image)) if c is not None]
        if not maps:
            return image.copy()
        best_out: NDArray[Any] | None = None
        best_amap: NDArray[Any] | None = None
        best_residual = float("inf")
        for amap, _region in maps:
            out = self._apply_reverse_alpha(image, amap)
            residual = self.detect(out).confidence
            if residual < best_residual:
                best_residual, best_out, best_amap = residual, out, amap
        if best_out is None or best_amap is None:  # pragma: no cover - maps is non-empty
            return image.copy()
        if residual_inpaint:
            kernel = np.ones((_RESIDUAL_DILATE, _RESIDUAL_DILATE), np.uint8)
            rm = cv2.dilate((best_amap > _RESIDUAL_ALPHA_FLOOR).astype(np.uint8) * 255, kernel)
            best_out = cv2.inpaint(best_out, rm, _RESIDUAL_INPAINT_RADIUS, cv2.INPAINT_NS)
        return best_out


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
