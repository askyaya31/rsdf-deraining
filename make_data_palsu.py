import os, sys
import numpy as np
from skimage import data, io, transform, img_as_ubyte
import cv2

rng = np.random.default_rng(0)
out = sys.argv[1] if len(sys.argv) > 1 else "data_palsu"
os.makedirs(f"{out}/input", exist_ok=True); os.makedirs(f"{out}/target", exist_ok=True)

srcs = []
for fn in ["camera", "astronaut", "coins", "moon", "page", "chelsea", "coffee", "rocket"]:
    try:
        im = getattr(data, fn)()
        if im.ndim == 3: im = im[..., :3].mean(-1) / 255.0
        else: im = im / 255.0
        srcs.append(im.astype(np.float64))
    except Exception as e:
        print("lewati", fn, e)

def rain_layer(h, w, angle_deg, length=21, dens=0.0025):
    layer = np.zeros((h, w), np.float32)
    pts = rng.random((h, w)) < dens
    layer[pts] = rng.uniform(0.5, 1.0, pts.sum())
    k = np.zeros((length, length), np.float32)
    c = length // 2
    t = np.radians(angle_deg)
    cv2.line(k, (int(c - c * np.cos(t)), int(c - c * np.sin(t))),
             (int(c + c * np.cos(t)), int(c + c * np.sin(t))), 1.0, 1)
    k /= k.sum()
    return cv2.filter2D(layer, -1, k * length)

n = 0
for im in srcs:
    h, w = im.shape
    s = 256
    if h <= s or w <= s:
        continue
    for _ in range(6):
        y0, x0 = rng.integers(0, h - s), rng.integers(0, w - s)
        clean = im[y0:y0 + s, x0:x0 + s]
        ang = rng.uniform(70, 110)
        rainy = np.clip(clean + 0.6 * rain_layer(s, s, ang), 0, 1)
        io.imsave(f"{out}/target/{n:03d}.png", img_as_ubyte(clean), check_contrast=False)
        io.imsave(f"{out}/input/{n:03d}.png", img_as_ubyte(rainy), check_contrast=False)
        n += 1
print("selesai:", n, "pasang")
