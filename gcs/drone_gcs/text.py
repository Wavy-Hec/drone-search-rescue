"""Anti-aliased TrueType text on OpenCV (BGR) images, with a cache. Falls back to Hershey fonts if PIL or the
fonts are missing."""

from __future__ import annotations

import collections
import os

import numpy as np

_FONT_CANDIDATES = {
    "sans": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "medium": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "bold": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": [
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
    "mono_bold": [
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ],
}


class TextRenderer:
    """``put(img, text, x, y, size, color, kind, anchor)``; anchor = horizontal (l|m|r) + vertical (t|m|b)."""

    def __init__(self, cache_size: int = 4000, use_pil: bool = True):
        self._fonts: dict = {}
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._cache_size = cache_size
        self._pil = None
        if use_pil:
            try:
                from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

                self._pil = (Image, ImageDraw, ImageFont)
            except ImportError:
                self._pil = None

    @property
    def truetype(self) -> bool:
        """TrueType fonts are in use (else OpenCV's Hershey fonts)."""
        return self._pil is not None and self._font("sans", 14) is not None

    def _font(self, kind: str, size: int):
        key = (kind, size)
        if key not in self._fonts:
            font = None
            if self._pil is not None:
                for path in _FONT_CANDIDATES.get(kind, _FONT_CANDIDATES["sans"]):
                    if os.path.exists(path):
                        try:
                            font = self._pil[2].truetype(path, size)
                            break
                        except OSError:
                            continue
            self._fonts[key] = font
        return self._fonts[key]

    def _sprite(self, text: str, size: int, kind: str) -> np.ndarray:
        """Alpha mask (h, w) float32 for ``text``; height = ascent + descent of the font (stable baselines)."""
        key = (text, size, kind)
        a = self._cache.get(key)
        if a is not None:
            self._cache.move_to_end(key)
            return a
        font = self._font(kind, size)
        if font is not None:
            Image, ImageDraw, _ = self._pil
            ascent, descent = font.getmetrics()
            w = max(1, int(np.ceil(font.getlength(text))) + 2)
            h = ascent + descent
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((1, 0), text, fill=255, font=font)
            a = np.asarray(im, dtype=np.float32) / 255.0
        else:
            import cv2  # noqa: PLC0415

            scale = size / 30.0
            thick = 2 if kind in ("bold", "mono_bold", "medium") else 1
            (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            im = np.zeros((h + base + 2, w + 2), np.uint8)
            cv2.putText(im, text, (1, h + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
            a = im.astype(np.float32) / 255.0
        self._cache[key] = a
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return a

    def size(self, text: str, size: int = 14, kind: str = "sans") -> tuple[int, int]:
        """Size ``(w, h)`` [px] that :meth:`put` would draw ``text`` at."""
        a = self._sprite(text, size, kind)
        return a.shape[1], a.shape[0]

    def put(
        self,
        img: np.ndarray,
        text: str,
        x: float,
        y: float,
        size: int = 14,
        color: tuple[int, int, int] = (255, 255, 255),
        kind: str = "sans",
        anchor: str = "lt",
        bg: tuple[int, int, int] | None = None,
        pad: int = 3,
        bg_alpha: float = 1.0,
    ) -> tuple[int, int, int, int]:
        """Draw ``text``; returns its box ``(x0, y0, x1, y1)``. ``bg`` draws a filled box behind it."""
        if not text:
            return (int(x), int(y), int(x), int(y))
        a = self._sprite(str(text), int(size), kind)
        h, w = a.shape
        x0 = int(round(x - (w / 2 if anchor[0] == "m" else w if anchor[0] == "r" else 0)))
        y0 = int(round(y - (h / 2 if anchor[1] == "m" else h if anchor[1] == "b" else 0)))
        H, W = img.shape[:2]
        if bg is not None:
            bx0, by0, bx1, by1 = max(0, x0 - pad), max(0, y0 - pad + 1), min(W, x0 + w + pad), min(H, y0 + h + pad - 1)
            if bx1 > bx0 and by1 > by0:
                if bg_alpha >= 1.0:
                    img[by0:by1, bx0:bx1] = bg
                else:
                    roi = img[by0:by1, bx0:bx1].astype(np.float32)
                    img[by0:by1, bx0:bx1] = (roi * (1 - bg_alpha) + np.array(bg, np.float32) * bg_alpha).astype(
                        np.uint8
                    )
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return (x0, y0, x0 + w, y0 + h)
        sub = a[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0, None]
        roi = img[cy0:cy1, cx0:cx1].astype(np.float32)
        img[cy0:cy1, cx0:cx1] = (roi + (np.array(color, np.float32) - roi) * sub).astype(np.uint8)
        return (x0, y0, x0 + w, y0 + h)
