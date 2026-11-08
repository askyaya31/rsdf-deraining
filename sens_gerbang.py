import sys
import numpy as np
import pandas as pd
df = pd.read_csv(f"{sys.argv[1]}/hasil_per_pasang.csv")
print("conf: min %.2f  median %.2f  maks %.2f" % (df.conf.min(), df.conf.median(), df.conf.max()))
print("th    %difilter  gain_rata2(dB)  menang seri kalah")
for th in [0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    use = df.conf >= th
    p = np.where(use, df.psnr_Bng, df.psnr_none)
    g = p - df.psnr_none
    w, l = int((g > 1e-9).sum()), int((g < -1e-9).sum())
    print(f"{th:4.1f}  {100 * (~use).mean():7.0f}%  {g.mean():12.2f}   {w:5d} {len(df) - w - l:4d} {l:5d}")
