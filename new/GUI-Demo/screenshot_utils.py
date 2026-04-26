"""Screenshot helpers for the AndroidWorld Doubao GUI runner."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IMAGE_RESIZE_SCALE = 0.5
DEFAULT_IMAGE_JPEG_QUALITY = 85


def save_state_screenshot(state: Any, path: str | Path, *, quality: int = 92) -> Path:
    """Save an AndroidWorld state screenshot as JPEG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(_to_uint8_rgb(state.pixels))
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
    return output_path


def image_path_to_data_url(
    image_path: str | Path,
    *,
    resize_scale: float = DEFAULT_IMAGE_RESIZE_SCALE,
    quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
) -> str:
    """Encode an image as a compressed data URL for Doubao GUI messages."""
    path = Path(image_path)
    with Image.open(path) as image:
        if resize_scale and resize_scale != 1.0:
            width, height = image.size
            target_size = (
                max(1, int(width * resize_scale)),
                max(1, int(height * resize_scale)),
            )
            image = image.resize(target_size)
        if image.mode != "RGB":
            image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def image_size(image_path: str | Path) -> tuple[int, int]:
    """Return image size as (width, height)."""
    with Image.open(image_path) as image:
        return image.size


def _to_uint8_rgb(pixels: Any) -> np.ndarray:
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array
