import argparse, glob, os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, uniform_filter1d
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

EXT = (".png", ".jpg", ".jpeg", ".bmp")
PAD = 32

def read_gray(path):
    im = io.imread(path)
    if im.ndim == 3:
        im = rgb2gray(im[..., :3])
    else:
        im = im.astype(np.float64) / (255.0 if im.dtype == np.uint8 else 1.0)
    return np.clip(im, 0, 1)


def load_pairs(root, n):
    def lst(d):
        return sorted(f for f in glob.glob(os.path.join(root, d, "*")) if f.lower().endswith(EXT))
    fi, ft = lst("input"), lst("target")
    assert len(fi) == len(ft) > 0, "folder input/target kosong atau jumlahnya beda"
    out = []
    for a, b in list(zip(fi, ft))[:n]:
        x, y = read_gray(a), read_gray(b)
        assert x.shape == y.shape, f"ukuran beda: {a}"
        out.append((os.path.basename(a), x, y))
    return out


def metrics(ref, out):
    out = np.clip(out, 0, 1)
    return psnr(ref, out, data_range=1.0), ssim(ref, out, data_range=1.0)

def grid(shape):
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    return u - w // 2, v - h // 2 

def freq_filter(img, make_mask):
    p = np.pad(img, PAD, mode="reflect")
    F = np.fft.fftshift(np.fft.fft2(p))
    M = make_mask(p.shape)
    out = np.fft.ifft2(np.fft.ifftshift(F * M)).real
    return out[PAD:-PAD, PAD:-PAD]

def mask_lowpass(rfrac):
    def f(shape):
        x, y = grid(shape)
        R = rfrac * min(shape) / 2
        return (np.hypot(x, y) <= R).astype(float)
    return f


def filt_A(img, rfrac, pre=True, sc=0.05, ss=5):
    x = img.astype(np.float32)
    if pre:
        x = cv2.GaussianBlur(x, (7, 7), 2)
        x = cv2.bilateralFilter(x, d=9, sigmaColor=sc, sigmaSpace=ss)
    return freq_filter(x.astype(np.float64), mask_lowpass(rfrac))

def ridge_angle(img, rlo=0.03, rhi=0.45, nang=180, smooth=31, allowed=None):
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    mag = np.log1p(np.abs(F))
    cy, cx, m = h // 2, w // 2, min(h, w)
    rr = np.arange(rlo * m, rhi * m)
    ang = np.arange(nang) * (180.0 / nang)
    prof = np.empty(nang)
    for i, a in enumerate(ang):
        t = np.radians(a)
        xs, ys = cx + rr * np.cos(t), cy - rr * np.sin(t)
        prof[i] = map_coordinates(mag, [ys, xs], order=1, mode="nearest").mean()
    prof = prof - uniform_filter1d(prof, smooth, mode="wrap")
    p = prof.copy()
    if allowed is not None:
        p[~allowed] = -np.inf
    i = int(np.argmax(p))
    conf = (prof[i] - np.median(prof)) / (prof.std() + 1e-9)
    return ang[i], ang, prof, conf


def allowed_mask(angles, margin=15, nang=180):
    ang = np.arange(nang) * (180.0 / nang)
    m = np.zeros(nang, bool)
    for a in angles:
        d = np.abs(ang - a)
        m |= np.minimum(d, 180 - d) <= margin
    return m


def ang_err(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def mask_notch(phi, sig, r0frac):
    def f(shape):
        x, y = grid(shape)
        th = np.degrees(np.arctan2(-y, x)) % 180
        d = np.abs(th - phi % 180)
        d = np.minimum(d, 180 - d)
        r = np.hypot(x, y)
        R0 = r0frac * min(shape) / 2
        wedge = np.exp(-d**2 / (2 * sig**2))
        keep = np.exp(-r**2 / (2 * R0**2))
        return 1 - wedge * (1 - keep) 
    return f


def gate_alpha(conf, th, ramp):
    if ramp == 0:
        return 1.0 if conf >= th else 0.0
    return float(np.clip((conf - th) / ramp, 0.0, 1.0))


def filt_B(img, phi, sig, r0frac, conf=None, th=None, ramp=0):
    out = freq_filter(img, mask_notch(phi, sig, r0frac))
    if conf is None or th is None:
        return out
    a = gate_alpha(conf, th, ramp)
    return img + a * (out - img)

def tune(pairs, cands, apply):
    sc = [np.mean([psnr(y, np.clip(apply(n, x, kw), 0, 1), data_range=1.0) for n, x, y in pairs])
          for kw in cands]
    i = int(np.argmax(sc))
    return cands[i], sc[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_tune", type=int, default=20)
    ap.add_argument("--out", default="hasil")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pairs = load_pairs(a.root, a.n)
    tune_set, test_set = pairs[:a.n_tune], pairs[a.n_tune:]
    print(f"pasang: {len(pairs)} (tuning {len(tune_set)}, evaluasi {len(test_set)})")
    phi_orc = {n: ridge_angle(x - y)[0] for n, x, y in pairs}
    allowed = allowed_mask([phi_orc[n] for n, _, _ in tune_set], margin=15)
    print("sudut spektrum yang diizinkan (derajat):", np.arange(180)[allowed].min(), "...", np.arange(180)[allowed].max(),
          f"({allowed.sum()} dari 180)")
    phi_auto, conf = {}, {}
    for n, x, y in pairs:
        a_, _, _, c_ = ridge_angle(x, allowed=allowed)
        phi_auto[n], conf[n] = a_, c_

    candA = [dict(rfrac=r, pre=p) for r in (0.5, 0.7, 0.85, 1.0) for p in (False, True)]
    bestA, bestB = {}, None
    for p in (False, True):
        cs = [c for c in candA if c["pre"] == p]
        bestA[p] = tune(tune_set, cs, lambda n, x, kw: filt_A(x, **kw))
    SIGS, R0S = (3, 5, 8, 12, 20), (0.05, 0.1, 0.2, 0.3)
    ctune = np.array([conf[n] for n, _, _ in tune_set])
    THS = tuple(float(v) for v in np.quantile(ctune, (0, 0.2, 0.4, 0.5, 0.6, 0.8)))
    print("keyakinan (tuning): min %.2f, median %.2f, maks %.2f" % (ctune.min(), np.median(ctune), ctune.max()))
    RAMPS = (0, 1.0, 3.0)
    candB = [dict(sig=s, r0frac=r, th=t, ramp=rp) for s in SIGS for r in R0S for t in THS for rp in RAMPS]
    bestB = tune(tune_set, candB, lambda n, x, kw: filt_B(x, phi_auto[n], conf=conf[n], **kw))
    kA_lp, kA_full, kB = bestA[False][0], bestA[True][0], bestB[0]
    use_full = bestA[True][1] >= bestA[False][1]
    kA = kA_full if use_full else kA_lp
    print("parameter terpilih (dari tuning):")
    print("  A tanpa pra-proses:", kA_lp, " | A penuh:", kA_full, " | A dipakai pembanding:", "penuh" if use_full else "tanpa pra-proses")
    print("  B:", kB)
    if kB["sig"] in (SIGS[0], SIGS[-1]) or kB["r0frac"] in (R0S[0], R0S[-1]):
        print("  PERINGATAN: parameter B terbaik ada di tepi grid; perlebar rentang SIGS/R0S lalu ulangi.")

    rows = []
    for n, x, y in test_set:
        r = {"file": n}
        r["psnr_none"], r["ssim_none"] = metrics(y, x)
        r["psnr_Alp"], r["ssim_Alp"] = metrics(y, filt_A(x, **kA_lp))
        r["psnr_Afull"], r["ssim_Afull"] = metrics(y, filt_A(x, **kA_full))
        r["psnr_B"], r["ssim_B"] = metrics(y, filt_B(x, phi_auto[n], conf=conf[n], **kB))
        kB0 = {k: v for k, v in kB.items() if k not in ("th", "ramp")}
        r["psnr_Borc"], r["ssim_Borc"] = metrics(y, filt_B(x, phi_orc[n], **kB0))
        r["conf"] = conf[n]
        r["alpha"] = gate_alpha(conf[n], kB["th"], kB["ramp"])
        r["gated"] = int(r["alpha"] < 1.0)
        r["ang_auto"], r["ang_orc"] = phi_auto[n], phi_orc[n]
        r["ang_err"] = ang_err(phi_auto[n], phi_orc[n])
        rows.append(r)
    df = pd.DataFrame(rows)
    df["psnr_A"] = df["psnr_Afull"] if use_full else df["psnr_Alp"]
    df["ssim_A"] = df["ssim_Afull"] if use_full else df["ssim_Alp"]
    df.to_csv(os.path.join(a.out, "hasil_per_pasang.csv"), index=False)

    cols = ["psnr_none", "psnr_Alp", "psnr_Afull", "psnr_B", "psnr_Borc",
            "ssim_none", "ssim_Alp", "ssim_Afull", "ssim_B", "ssim_Borc"]
    print("\nRingkasan evaluasi (rata-rata +- simpangan baku):")
    for c in cols:
        print(f"  {c:12s} {df[c].mean():.4f} +- {df[c].std():.4f}")
    print(f"Citra yang tidak difilter (gerbang keyakinan): {df['gated'].mean() * 100:.0f}% pasang")
    print(f"\nGalat sudut estimator vs oracle: rata-rata {df['ang_err'].mean():.1f} derajat, "
          f"median {df['ang_err'].median():.1f}")

    win_A = (df["psnr_B"] > df["psnr_A"]).mean() * 100
    win_none = (df["psnr_B"] > df["psnr_none"]).mean() * 100
    diff = (df["psnr_B"] - df["psnr_A"]).mean()
    print(f"\nB > A pada {win_A:.0f}% pasang, selisih rata-rata {diff:+.2f} dB")
    print(f"B > tanpa proses pada {win_none:.0f}% pasang")
    ge_none = (df["psnr_B"] >= df["psnr_none"] - 1e-9).mean() * 100
    gain = (df["psnr_B"] - df["psnr_none"]).mean()
    print(f"B >= tanpa proses pada {ge_none:.0f}% pasang (seri = tidak difilter); selisih rata-rata {gain:+.2f} dB")
    print(f"B >= tanpa proses menurut SSIM pada {(df['ssim_B'] >= df['ssim_none'] - 1e-9).mean() * 100:.0f}% pasang")
    ok_main = (ge_none >= 75) and (gain >= 1.0) and (win_A >= 70)
    print("KRITERIA TOPIK UTAMA (>=75% pasang, >=+1 dB, B>A >=70%):", "TERPENUHI" if ok_main else "BELUM")
    print(f"B > A menurut SSIM pada {(df['ssim_B'] > df['ssim_A']).mean() * 100:.0f}% pasang "
          "(PSNR bisa memihak hasil yang kabur; cek SSIM dan contoh.png)")
    if win_A >= 70 and diff > 0.5:
        print("KEPUTUSAN: LULUS (B jelas lebih baik dari A)")
    elif win_A < 50:
        print("KEPUTUSAN: GAGAL (B tidak lebih baik dari A) -> pertimbangkan segmentasi retakan")
    else:
        print("KEPUTUSAN: ABU-ABU -> coba perbaiki parameter B selama 1 jam, lalu ulangi")

    k = min(3, len(test_set))
    fig, ax = plt.subplots(k, 4, figsize=(12, 3 * k), squeeze=False)
    for i in range(k):
        n, x, y = test_set[i]
        outs = [x, filt_A(x, **kA), filt_B(x, phi_auto[n], conf=conf[n], **kB), y]
        for j, (im, t) in enumerate(zip(outs, ["berhujan", "A", "B", "bersih"])):
            ax[i, j].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_title(t); ax[i, j].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "contoh.png"), dpi=120); plt.close()

    n, x, y = test_set[0]
    _, ang, prof, _ = ridge_angle(x, allowed=allowed)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))))
    ax[0].imshow(F, cmap="gray"); ax[0].set_title("log spektrum (berhujan)")
    ax[1].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((x - y) - (x - y).mean())))), cmap="gray")
    ax[1].set_title("log spektrum lapisan hujan")
    ax[2].plot(ang, prof); ax[2].set_title(f"profil sudut; puncak {phi_auto[n]:.0f} derajat")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "spektrum.png"), dpi=120); plt.close()
    print(f"\nTersimpan di folder '{a.out}': hasil_per_pasang.csv, contoh.png, spektrum.png")


if __name__ == "__main__":
    main()
