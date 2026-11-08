import argparse, os, time
import numpy as np, pandas as pd, cv2
from scipy.ndimage import median_filter
from scipy.stats import spearmanr
import uji_deraining_v5 as U
import deraining_ablasi as D

def filt_median(x, k):
    return median_filter(x, size=k, mode="reflect")


def filt_bilateral(x, sc, ss):
    return cv2.bilateralFilter(x.astype(np.float32), 9, sc, ss).astype(np.float64)


def _box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def filt_guided(x, r, eps):
    I = x.astype(np.float32)
    mI = _box(I, r)
    var = _box(I * I, r) - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return (_box(a, r) * I + _box(b, r)).astype(np.float64)


NEW = {
    "med": (filt_median, [dict(k=k) for k in (3, 5, 7)]),
    "bil": (filt_bilateral, [dict(sc=c, ss=s) for c in (0.05, 0.1, 0.2) for s in (3, 5, 9)]),
    "gf": (filt_guided, [dict(r=r, eps=e) for r in (2, 4, 8) for e in (1e-3, 1e-2, 1e-1)]),
}


def tune_new(tune_set):
    return {k: U.tune(tune_set, cands, lambda n, x, kw, f=f: f(x, **kw))[0] for k, (f, cands) in NEW.items()}

def tune_all_fast(pairs, n_tune, P):
    ts = pairs[:n_tune]
    K = {}
    K["A_lp"] = U.tune(ts, [dict(rfrac=r, pre=False) for r in (0.5, 0.7, 0.85, 1.0)], lambda n, x, kw: U.filt_A(x, **kw))[0]
    K["A_full"] = U.tune(ts, [dict(rfrac=r, pre=True) for r in (0.5, 0.7, 0.85, 1.0)], lambda n, x, kw: U.filt_A(x, **kw))[0]
    gridB = [dict(sig=s, r0frac=r) for s in D.SIGS for r in D.R0S]
    K["B_free_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["af"][n], **kw))[0]
    K["B_range_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["ar"][n], **kw))[0]
    K["B_oracle_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["orc"][n], **kw))[0]
    ct = np.array([P["cr"][n] for n, _, _ in ts])
    THS = tuple(float(v) for v in np.quantile(ct, (0, 0.2, 0.4, 0.5, 0.6, 0.8)))
    base = K["B_range_nogate"]
    candG = [dict(sig=base["sig"], r0frac=base["r0frac"], th=t, ramp=rp) for t in THS for rp in D.RAMPS]
    K["B_range_gate"] = U.tune(ts, candG, lambda n, x, kw: U.filt_B(x, P["ar"][n], conf=P["cr"][n], **kw))[0]
    return K

def evaluate_all(test_set, K, P, KN):
    df = D.evaluate(test_set, K, P)
    rows = []
    for n, x, y in test_set:
        r = {"file": n, "rain_dens": float(np.std(x - y))}  
        for k, (f, _) in NEW.items():
            r["psnr_" + k], r["ssim_" + k] = U.metrics(y, f(x, **KN[k]))
        rows.append(r)
    return df.merge(pd.DataFrame(rows), on="file")


SHOW = ["none", "A_lp", "med", "bil", "gf", "B_range_gate", "B_oracle_nogate"]


def summarize_all(dfs, title):
    print(f"\n=== {title}")
    print(f"{'varian':16s} {'PSNR (rata2 +- std antar seed)':>32s} {'SSIM':>18s}")
    for v in SHOW:
        ps = [d["psnr_" + v].mean() for d in dfs]
        ss = [d["ssim_" + v].mean() for d in dfs]
        print(f"{v:16s} {np.mean(ps):10.3f} +- {np.std(ps):5.3f}   {np.mean(ss):10.4f} +- {np.std(ss):.4f}")

    allr = pd.concat(dfs)
    per = allr.groupby("file").mean(numeric_only=True)
    rng = np.random.default_rng(0)
    print(f"\nSelisih PSNR B_range_gate dikurangi pembanding, 95% CI bootstrap atas {len(per)} citra unik:")
    for v in ["none", "A_lp", "med", "bil", "gf"]:
        d = (per["psnr_B_range_gate"] - per["psnr_" + v]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f"  vs {v:5s}: {d.mean():+.3f} dB  [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")

    per["gain_B"] = per["psnr_B_range_gate"] - per["psnr_none"]
    print("\nDiagnosis kegagalan: korelasi Spearman gain B (vs tanpa proses) dengan:")
    for c in ("conf", "ang_err_range", "rain_dens"):
        rho, p = spearmanr(per[c], per["gain_B"])
        print(f"  {c:14s} rho={rho:+.2f} (p={p:.3g}, n={len(per)})")
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_tune", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="tambah")
    ap.add_argument("--full_tune", action="store_true", help="pakai tuning penuh D.tune_all (lambat) alih-alih dua tahap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dfs, t0 = [], time.time()
    for sd in a.seeds:
        pairs = D.shuffled_pairs(a.root, a.n, sd)
        P = D.prepare(pairs, a.n_tune)
        K = (D.tune_all if a.full_tune else tune_all_fast)(pairs, a.n_tune, P)
        KN = tune_new(pairs[:a.n_tune])
        df = evaluate_all(pairs[a.n_tune:], K, P, KN)
        df["seed"] = sd
        df.to_csv(f"{a.out}/tambah_seed{sd}.csv", index=False)
        dfs.append(df)
        print(f"seed {sd} selesai ({(time.time() - t0) / 60:.1f} menit)")
        print("   B_range_gate:", K["B_range_gate"], "| pembanding:", KN)
    per = summarize_all(dfs, f"{a.root} ({len(a.seeds)} seed)")
    per.to_csv(f"{a.out}/per_citra_rata2_seed.csv")


if __name__ == "__main__":
    main()
