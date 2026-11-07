"""Fetch a small public photo set used for training / evaluation (no registration, ~15 MB).

Sources: OpenCV sample data + opencv_extra test data (raw.githubusercontent.com),
scikit-image and scikit-learn bundled images (exported locally).
Images are *clean* ground truth: the sensor simulator creates the degraded RAW from them.
For better results drop more photos (DIV2K, MIT-Adobe FiveK, your own...) into data/public/.

    python download_data.py
"""
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from config import ROOT

OUT = ROOT / "data" / "public"
CV = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
EX = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/"
FILES = {
    # name: url
    "cv_lena.jpg": CV + "lena.jpg", "cv_baboon.jpg": CV + "baboon.jpg", "cv_fruits.jpg": CV + "fruits.jpg",
    "cv_butterfly.jpg": CV + "butterfly.jpg", "cv_building.jpg": CV + "building.jpg", "cv_home.jpg": CV + "home.jpg",
    "cv_messi.jpg": CV + "messi5.jpg", "cv_apple.jpg": CV + "apple.jpg", "cv_orange.jpg": CV + "orange.jpg",
    "cv_starry.jpg": CV + "starry_night.jpg", "cv_aloe.jpg": CV + "aloeL.jpg", "cv_stuff.jpg": CV + "stuff.jpg",
    "ex_airplane.png": EX + "cv/shared/airplane.png", "ex_teddy.png": EX + "cv/stereomatching/datasets/teddy/im2.png",
    "ex_cones.png": EX + "cv/stereomatching/datasets/cones/im2.png", "ex_venus.png": EX + "cv/stereomatching/datasets/venus/im2.png",
    **{f"ex_boat{i}.jpg": EX + f"stitching/boat{i}.jpg" for i in range(1, 7)},
    "ex_a2.png": EX + "stitching/a2.png",
    "ex_s1.jpg": EX + "stitching/s1.jpg", "ex_s2.jpg": EX + "stitching/s2.jpg",
}


def export_bundled():
    from skimage import data as skd
    for n in ["astronaut", "coffee", "chelsea", "rocket"]:
        a = getattr(skd, n)()
        Image.fromarray(a[..., :3]).save(OUT / f"sk_{n}.png")
    try:
        from sklearn.datasets import load_sample_images
        for n, a in zip(["china", "flower"], load_sample_images().images):
            Image.fromarray(a).save(OUT / f"sl_{n}.png")
    except Exception as e:
        print("sklearn images skipped:", e)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export_bundled()
    for name, url in FILES.items():
        dst = OUT / name
        if dst.exists():
            continue
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"skip {name}: {e}")
    print(f"{len(list(OUT.glob('*')))} images in {OUT}")


if __name__ == "__main__":
    main()
