import numpy as np
from PIL import Image


def to_float_rgb(image):
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("Expected a grayscale or RGB image.")
    a = a[..., :3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if a.max() > 1.5:
            a /= 255.0
    return np.nan_to_num(np.clip(a, 0, 1))


def load_image(path):
    return to_float_rgb(Image.open(path).convert("RGB"))


def save_image(path, image):
    a = np.clip(to_float_rgb(image) * 255 + 0.5, 0, 255).astype("uint8")
    Image.fromarray(a, "RGB").save(path)


def validate_output(image, shape=None):
    a = to_float_rgb(image)
    if shape and a.shape[:2] != shape[:2]:
        raise ValueError("Output dimensions do not match input.")
    return a
