"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Reverse-alpha based.** A known mark is a fixed semi-transparent overlay, so it
is removed by inverting the alpha blend against a captured alpha map
(``original = (wm - a*logo)/(1-a)``) -- recovering the true pixels rather than
inpainting a guess. Gemini and Doubao recover exactly with no inpaint at native on
bright/flat backgrounds (Gemini falls back to inpainting the sparkle footprint when
reverse-alpha would over-subtract on a dark background -- issue #30, see gemini_engine);
Jimeng adds a thin residual inpaint over the glyph footprint to clear the outline
its per-image render variation leaves behind (still seeded by the reverse-alpha
recovery, not a blind inpaint). Detection is consistent with that: each mark is
recognized by matching its known shape/template (the thing we invert), not by
heuristics. A mark is therefore listed here only once a real alpha map has been
captured for it; everything else (arbitrary logos/objects) is the user-directed
``erase --region`` tool, not this catalog.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# cv2 method for the Gemini reverse-alpha edge-residual cleanup (not a standalone
# remover): "ns" / "telea".
InpaintMethod = Literal["telea", "ns"]
Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and remove it."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    recovery: str  # removal strategy (all reverse-alpha today)
    _detect: Callable[[NDArray[Any]], MarkDetection]
    _remove: Callable[..., tuple[NDArray[Any], Region | None]]

    def detect(self, image: NDArray[Any]) -> MarkDetection:
        return self._detect(image)

    def remove(
        self,
        image: NDArray[Any],
        *,
        inpaint_method: InpaintMethod = "ns",
        inpaint: bool = True,
        inpaint_strength: float = 0.85,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by reverse-alpha; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox (for residual-inpaint positioning),
        or None if nothing was removed. NB: the CLI does NOT use ``region`` to
        clear alpha on save -- that zeroing caused the issue-#30 white box.

        ``inpaint`` / ``inpaint_strength`` / ``inpaint_method`` tune the Gemini
        reverse-alpha edge-residual cleanup only. ``force`` removes at the mark's
        usual location even without a positive detection (the ``--no-detect`` path).
        """
        return self._remove(image, inpaint_method, inpaint, inpaint_strength, force)


# Single source of truth for the Gemini-sparkle "trust this as a real mark"
# confidence, shared by BOTH the removal arbitration here (`best_auto_mark` /
# `_gemini_detect`) and the provenance detector in `identify` (which imports it
# as its sparkle threshold). Defining it once removes the detect-vs-remove
# threshold drift the retained-corpus mining surfaced (2026-06-20): identify
# would report a sparkle while removal declined it, or vice versa, whenever the
# two independently-maintained 0.5 constants fell out of step. Now they cannot.
#
# Value 0.5 is corpus-validated: the gemini engine's own `detected` flag uses a
# looser internal threshold (0.35) and weakly fires (~0.36-0.42) on unrelated
# bottom-right text -- a real Doubao mark scores ~0.40-0.42 as a gemini match,
# and its core-ring brightness margin is HIGHER than a genuine faint sparkle's,
# so neither confidence nor the brightness gate separates them in the [0.35, 0.5)
# band. Lowering this gate to recover faint sparkles was evaluated against that
# band (2026-06-20) and REJECTED: it cannot be done without re-admitting the
# Doubao-text / content false positives, trading a rare miss for false-positive
# removals on clean images. The band below the gate is therefore intentionally
# left to the higher-strength / metadata paths. 0.5 gives 0 false positives on
# the corpus.
GEMINI_SPARKLE_TRUST_CONF = 0.5
_GEMINI_AUTO_MIN_CONF = GEMINI_SPARKLE_TRUST_CONF

# ── Engine adapters (lazy singletons; engines are cv2-only, no model load) ──

_engines: dict[str, Any] = {}


def _engine(key: str) -> Any:
    if key not in _engines:
        if key == "gemini":
            from remove_ai_watermarks.gemini_engine import GeminiEngine

            _engines[key] = GeminiEngine()
        elif key == "doubao":
            from remove_ai_watermarks.doubao_engine import DoubaoEngine

            _engines[key] = DoubaoEngine()
        elif key == "jimeng":
            from remove_ai_watermarks.jimeng_engine import JimengEngine

            _engines[key] = JimengEngine()
        elif key == "samsung":
            from remove_ai_watermarks.samsung_engine import SamsungEngine

            _engines[key] = SamsungEngine()
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def _gemini_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image)
    detected = bool(d.detected) and d.confidence >= _GEMINI_AUTO_MIN_CONF
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_remove(
    image: NDArray[Any], inpaint_method: InpaintMethod, inpaint: bool, strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
    engine = _engine("gemini")
    det = engine.detect_watermark(image)
    if not det.detected:
        if not force:
            return image.copy(), None
        # Forced (--no-detect): remove at the default sparkle slot for the size.
        from remove_ai_watermarks.gemini_engine import get_watermark_config

        h, w = image.shape[:2]
        cfg = get_watermark_config(w, h)
        px, py = cfg.get_position(w, h)
        region = (px, py, cfg.logo_size, cfg.logo_size)
        result = engine.remove_watermark_custom(image, region)
        if inpaint:
            result = engine.inpaint_residual(result, region, strength=strength, method=inpaint_method)
        return result, region
    result = engine.remove_watermark(image)
    # Reverse-alpha leaves a faint residual at the sparkle edge; the engine's
    # own residual inpaint cleans that seam (part of its reverse-alpha pipeline).
    if inpaint:
        result = engine.inpaint_residual(result, det.region, strength=strength, method=inpaint_method)
    return result, det.region


# The three text-mark engines (Doubao/Jimeng/Samsung) share the TextMarkEngine
# interface, so one parameterized adapter pair drives all of them -- a new
# reverse-alpha text mark is one `_text_mark(...)` row below, not another copy-paste
# of these bodies. Removal is reverse-alpha only: applied when the mark is detected
# (or forced) and the alpha asset loads, otherwise skipped (no hallucination on a
# clean corner).
def _text_mark_detect(key: str, label: str, location: str) -> Callable[[NDArray[Any]], MarkDetection]:
    def detect(image: NDArray[Any]) -> MarkDetection:
        d = _engine(key).detect(image)
        return MarkDetection(key, label, location, d.detected, d.confidence, d.region)

    return detect


def _text_mark_remove(key: str) -> Callable[..., tuple[NDArray[Any], Region | None]]:
    def remove(
        image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool
    ) -> tuple[NDArray[Any], Region | None]:
        engine = _engine(key)
        det = engine.detect(image)
        if (det.detected or force) and engine.reverse_alpha_available(image):
            return engine.remove_watermark_reverse_alpha(image), (det.region if det.detected else None)
        return image.copy(), None

    return remove


def _text_mark(key: str, label: str, location: str) -> KnownMark:
    """A reverse-alpha text-mark registry row (Doubao/Jimeng/Samsung)."""
    return KnownMark(
        key, label, location, True, "reverse-alpha", _text_mark_detect(key, label, location), _text_mark_remove(key)
    )


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, "reverse-alpha", _gemini_detect, _gemini_remove),
    _text_mark("doubao", "Doubao 豆包AI生成 text", "bottom-right"),
    _text_mark("jimeng", "Jimeng 即梦AI wordmark", "bottom-right"),
    _text_mark("samsung", "Samsung Galaxy AI text", "bottom-left"),
)


def known_marks() -> tuple[KnownMark, ...]:
    """All registered known visible watermarks."""
    return _REGISTRY


def mark_keys() -> list[str]:
    """Keys of all registered marks (for CLI choices)."""
    return [m.key for m in _REGISTRY]


def get_mark(key: str) -> KnownMark:
    """Look up a known mark by key (raises KeyError if unknown)."""
    for m in _REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def detect_marks(image: NDArray[Any], *, include_explicit: bool = True) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    Returns one MarkDetection per scanned mark (``detected`` flags which fired).
    ``include_explicit=False`` scans only the ``in_auto`` marks -- the set used
    by ``--mark auto``.
    """
    return [m.detect(image) for m in _REGISTRY if include_explicit or m.in_auto]


def best_auto_mark(image: NDArray[Any]) -> MarkDetection | None:
    """The highest-confidence detected ``in_auto`` mark, or None if none fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    return max(fired, key=lambda d: d.confidence) if fired else None
