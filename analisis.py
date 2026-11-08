import pandas as pd
df = pd.read_csv("hasil_rain100l_v4/hasil_per_pasang.csv")
print("porsi sudut gagal (>10 derajat):", (df.ang_err > 10).mean())
ok, bad = df[df.ang_err <= 10], df[df.ang_err > 10]
for nm, d in [("sudut OK", ok), ("sudut gagal", bad)]:
    print(nm, len(d), "none %.2f  B %.2f  Borc %.2f" % (d.psnr_none.mean(), d.psnr_B.mean(), d.psnr_Borc.mean()))
print("B > none pada sudut OK:", (ok.psnr_B > ok.psnr_none).mean())
print("Borc > none (semua):", (df.psnr_Borc > df.psnr_none).mean())
print(df.ang_orc.round().value_counts().head(10))