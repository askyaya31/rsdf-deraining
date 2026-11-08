import pandas as pd
df = pd.read_csv("hasil_rain100l_v3/hasil_per_pasang.csv")
df["gain_B"] = df.psnr_B - df.psnr_none
df["gain_orc"] = df.psnr_Borc - df.psnr_none
print(df[["psnr_none", "gain_B", "gain_orc", "ang_err", "conf"]].corr().round(2))
q = df.psnr_none > df.psnr_none.median()
print("hujan ringan (PSNR none tinggi): gain B %.2f | oracle %.2f" % (df[q].gain_B.mean(), df[q].gain_orc.mean()))
print("hujan berat (PSNR none rendah):  gain B %.2f | oracle %.2f" % (df[~q].gain_B.mean(), df[~q].gain_orc.mean()))
print("B > none menurut SSIM:", (df.ssim_B > df.ssim_none).mean().round(2))