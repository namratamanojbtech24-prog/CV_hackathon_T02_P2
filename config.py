"""Global configuration: paths, sensor model constants and checkpoint names."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for _p in ["data/input", "data/reference", "data/output", "data/sample",
           "models/checkpoints", "results"]:
    (ROOT / _p).mkdir(parents=True, exist_ok=True)

CKPT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINTS = {
    "learned": CKPT_DIR / "learned_e2e.pt",          # end-to-end RAW -> sRGB U-Net
    "hybrid": CKPT_DIR / "hybrid_teacher.pt",        # classical front/back-end + learned residual
    "hybrid_lite": CKPT_DIR / "hybrid_student.pt",   # distilled student of the hybrid model
    "hybrid_int8": CKPT_DIR / "hybrid_student_int8.pt",
}
RESULTS = ROOT / "results"

# ---- Simulated 12-bit sensor ------------------------------------------------
BAYER_PATTERN = "RGGB"
BIT_DEPTH = 12
BLACK_DN = 256                      # black-level pedestal in digital numbers
WHITE_DN = 2 ** BIT_DEPTH - 1

# Camera-RGB -> linear sRGB matrix of the simulated sensor (rows sum to 1 so that
# neutral stays neutral after white balance). A real ISP would obtain this from
# sensor calibration (or the DNG ColorMatrix tag); the classical ISP is given the
# calibrated matrix, exactly like a phone ISP would be.
SENSOR_CCM = np.array([[1.72, -0.52, -0.20],
                       [-0.31, 1.58, -0.27],
                       [0.04, -0.47, 1.43]], dtype=np.float32)

# Back-compat alias used by older modules
COLOR_MATRIX = SENSOR_CCM
