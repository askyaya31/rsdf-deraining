import argparse, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uji_deraining_v5 as U

SIGS, R0S, RAMPS = (3, 5, 8, 12, 20), (0.05, 0.1, 0.2, 0.3), (0, 1.0, 3.0)

def shuffled_pairs(root, n, seed):
    pairs = U.load_pairs(root, 10 ** 6)
    idx = np.random.default_rng(seed).permutation(len(pairs))
    return [pairs[i] for i in idx][:n]

def prepare(pairs, n_tune, allowed=None):
    tune_set = pairs[:n_tune]
    orc = {n: U.ridge_angle(x - y)[0] for n, x, y in pairs}
    if allowed is None:
        allowed = U.allowed_mask([orc[n] for n, _, _ in tune_set], margin=15)
    P = dict(orc=orc, allowed=allowed, ar={}, cr={}, af={}, cf={})
    for n, x, y in pairs:
        a, _, _, c = U.ridge_angle(x, allowed=allowed); P["ar"][n], P["cr"][n] = a, c
        a, _, _, c = U.ridge_angle(x); P["af"][n], P["cf"][n] = a, c
    return P

def tune_all(pairs, n_tune, P):
    ts = pairs[:n_tune]; K = {}
    K["A_lp"] = U.tune(ts, [dict(rfrac=r, pre=False) for r in (0.5, 0.7, 0.85, 1.0)], lambda n, x, kw: U.filt_A(x, **kw))[0]
    K["A_full"] = U.tune(ts, [dict(rfrac=r, pre=True) for r in (0.5, 0.7, 0.85, 1.0)], lambda n, x, kw: U.filt_A(x, **kw))[0]
    gridB = [dict(sig=s, r0frac=r) for s in SIGS for r in R0S]
    K["B_free_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["af"][n], **kw))[0]
    K["B_range_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["ar"][n], **kw))[0]
    K["B_oracle_nogate"] = U.tune(ts, gridB, lambda n, x, kw: U.filt_B(x, P["orc"][n], **kw))[0]
    ct = np.array([P["cr"][n] for n, _, _ in ts])
    THS = tuple(float(v) for v in np.quantile(ct, (0, 0.2, 0.4, 0.5, 0.6, 0.8)))
    candG = [dict(sig=s, r0frac=r, th=t, ramp=rp) for s in SIGS for r in R0S for t in THS for rp in RAMPS]
    K["B_range_gate"] = U.tune(ts, candG, lambda n, x, kw: U.filt_B(x, P["ar"][n], conf=P["cr"][n], **kw))[0]
    return K


def apply_variant(v, x, n, K, P):
    k = K.get(v)
    if v == "none": return x
    if v in ("A_lp", "A_full"): return U.filt_A(x, **k)
    if v == "B_free_nogate": return U.filt_B(x, P["af"][n], **k)
    if v == "B_range_nogate": return U.filt_B(x, P["ar"][n], **k)
    if v == "B_oracle_nogate": return U.filt_B(x, P["orc"][n], **k)
    if v == "B_range_gate": return U.filt_B(x, P["ar"][n], conf=P["cr"][n], **k)
    raise ValueError(v)

VARIANTS = ["none", "A_lp", "A_full", "B_free_nogate", "B_range_nogate", "B_range_gate", "B_oracle_nogate"]

def evaluate(test_set, K, P, variants=VARIANTS):
    rows = []
    for n, x, y in test_set:
        r = {"file": n}
        for v in variants:
            r["psnr_" + v], r["ssim_" + v] = U.metrics(y, apply_variant(v, x, n, K, P))
        r["ang_err_range"] = U.ang_err(P["ar"][n], P["orc"][n]); r["ang_err_free"] = U.ang_err(P["af"][n], P["orc"][n])
        r["conf"] = P["cr"][n]
        rows.append(r)
    return pd.DataFrame(rows)

def summarize(dfs, title):
    print(f"\n=== {title}")
    print(f"{'varian':16s} {'PSNR (rata2+-std antar seed)':>30s} {'SSIM':>16s}  menang/seri/kalah vs none (gabungan)")
    allr = pd.concat(dfs)
    for v in VARIANTS:
        ps = [d["psnr_" + v].mean() for d in dfs]; ss = [d["ssim_" + v].mean() for d in dfs]
        diff = allr["psnr_" + v] - allr["psnr_none"]
        w, l = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        print(f"{v:16s} {np.mean(ps):9.3f} +- {np.std(ps):5.3f}  {np.mean(ss):9.4f} +- {np.std(ss):.4f}   {w}/{len(diff) - w - l}/{l}")
    print("\nSudut: galat rata-rata terhadap oracle (derajat): dengan rentang %.2f | tanpa rentang %.2f" %
          (allr.ang_err_range.mean(), allr.ang_err_free.mean()))
    for a, b, lab in [("B_range_nogate", "B_free_nogate", "efek RENTANG sudut"), ("B_range_gate", "B_range_nogate", "efek GERBANG keyakinan"),
                      ("B_range_gate", "A_lp", "B penuh vs A (low-pass)")]:
        per = [(d["psnr_" + a] - d["psnr_" + b]).mean() for d in dfs]
        print(f"  {lab:28s}: selisih PSNR rata-rata {np.mean(per):+.3f} dB (std antar seed {np.std(per):.3f})")


def param_sens(test_set, kgate, P, out_png):
    M = np.zeros((len(SIGS), len(R0S)))
    base = np.mean([U.metrics(y, x)[0] for n, x, y in test_set])
    for i, s in enumerate(SIGS):
        for j, r in enumerate(R0S):
            M[i, j] = np.mean([U.metrics(y, U.filt_B(x, P["ar"][n], sig=s, r0frac=r, conf=P["cr"][n], th=kgate["th"], ramp=kgate["ramp"]))[0]
                               for n, x, y in test_set]) - base
    df = pd.DataFrame(M, index=[f"sig={s}" for s in SIGS], columns=[f"r0={r}" for r in R0S])
    print("\nSensitivitas parameter B: selisih PSNR rata-rata terhadap 'tanpa proses' (dB), th dan ramp dikunci")
    print(df.round(2).to_string())
    fig, ax = plt.subplots(figsize=(5, 3.6)); im = ax.imshow(M, cmap="viridis"); plt.colorbar(im, label="ΔPSNR vs tanpa proses (dB)")
    ax.set_xticks(range(len(R0S))); ax.set_xticklabels(R0S); ax.set_yticks(range(len(SIGS))); ax.set_yticklabels(SIGS)
    ax.set_xlabel("r0frac (jari-jari pusat)"); ax.set_ylabel("sig (lebar baji, derajat)")
    for i in range(len(SIGS)):
        for j in range(len(R0S)): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--n_tune", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--out", default="abl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dfs = []; t0 = time.time()
    for sd in a.seeds:
        pairs = shuffled_pairs(a.root, a.n, sd); P = prepare(pairs, a.n_tune); K = tune_all(pairs, a.n_tune, P)
        test = pairs[a.n_tune:]; df = evaluate(test, K, P); df["seed"] = sd; df.to_csv(f"{a.out}/ablasi_seed{sd}.csv", index=False); dfs.append(df)
        print(f"seed {sd} selesai ({(time.time() - t0) / 60:.1f} menit). Parameter terpilih:")
        for k, v in K.items(): print("   ", k, v)
        if sd == a.seeds[0]: param_sens(test, K["B_range_gate"], P, f"{a.out}/sensitivitas_parameter.png")
    summarize(dfs, f"Ablasi pada {a.root} ({len(a.seeds)} seed, {a.n - a.n_tune} pasang evaluasi per seed)")


if __name__ == "__main__":
    main()
