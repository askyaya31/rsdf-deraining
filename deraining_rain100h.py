import argparse, os, time
import numpy as np, pandas as pd
import deraining_ablasi as D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootL", required=True); ap.add_argument("--rootH", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--n_tune", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--out", default="abl_H")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); frozen, retuned = [], []; t0 = time.time()
    for sd in a.seeds:
        pL = D.shuffled_pairs(a.rootL, a.n, sd); PL = D.prepare(pL, a.n_tune); KL = D.tune_all(pL, a.n_tune, PL)
        pH = D.shuffled_pairs(a.rootH, a.n, sd); testH = pH[a.n_tune:]
        PHf = D.prepare(pH, a.n_tune, allowed=PL["allowed"])
        dfF = D.evaluate(testH, KL, PHf); dfF["seed"] = sd; frozen.append(dfF)
        PH = D.prepare(pH, a.n_tune); KH = D.tune_all(pH, a.n_tune, PH)
        dfR = D.evaluate(testH, KH, PH); dfR["seed"] = sd; retuned.append(dfR)
        out_of_range = np.mean([not PL["allowed"][int(round(PH["orc"][n])) % 180] for n, _, _ in testH]) * 100
        print(f"seed {sd} selesai ({(time.time() - t0) / 60:.1f} menit). Sudut oracle Rain100H di luar rentang Rain100L: {out_of_range:.0f}% citra uji")
        print("   parameter B (dituning di H):", KH["B_range_gate"])
        dfF.to_csv(f"{a.out}/H_beku_seed{sd}.csv", index=False); dfR.to_csv(f"{a.out}/H_tuning_ulang_seed{sd}.csv", index=False)
    D.summarize(frozen, "Rain100H: parameter dan rentang sudut DIBEKUKAN dari Rain100L")
    D.summarize(retuned, "Rain100H: SEMUA dituning ulang pada Rain100H")

if __name__ == "__main__":
    main()
