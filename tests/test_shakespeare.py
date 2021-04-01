"""Test shakespeare.


corr500x510 = fetch_sent_corr(lines_en[:500], lines_zh[:510])  # 3.13m
corr1000x1100 = fetch_sent_corr(lines_en[:1000], lines_zh[:1100])  # 4m

joblib.dump(corr1000x1100, "corr500x510.lzma")  # 74s

argmax = np.array(corr500x510).argmax(axis=0)
max = np.array(corr500x510).max(axis=0)
df = pd.DataFrame({"lang2": range(argmax.shape[0]), "argmax": argmax, "max": max})

sns.relplot(x="lang2", y="argmax", hue="max", data=df)
sns.relplot(x="lang2", y="argmax", size="max", data=df)

sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df)

testarr = df.to_numpy().copy()

testarr = np.array(corr500x510)
lowess_test = lowess_pairs(testarr)

max = np.array(corr500x510).max(axis=0)
argmax = np.array(corr500x510).argmax(axis=0)

df = pd.DataFrame({"lang2": range(argmax.shape[0]), "argmax": argmax, "max": max})
sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df)

df_lowess = pd.DataFrame(lowess_test)
df_lowess.columns = ['lang2', 'argmax', 'max']
sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df_lowess)

# ---
from tinybee.find_pairs import find_pairs as savgol_pairs

testarr = np.array(corr500x510)
savgol_test = savgol_pairs(testarr)
df_savgo = pd.DataFrame(savgol_test)
df_savgo.columns = ['lang2', 'argmax', 'max']
sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df_savgo)

arr1 = testarr.copy()

"""
from pathlib import Path
import joblib

from tinybee.fetch_sent_corr import fetch_sent_corr

file_en = r"C:\Users\mike\Documents\Tencent Files\41947782\FileRecv\shakespeare_en.txt"
file_zh = r"C:\Users\mike\Documents\Tencent Files\41947782\FileRecv\shakespeare_zh.txt"

lines_en = Path(file_en).read_text("utf8").splitlines()
lines_zh = Path(file_zh).read_text("utf8").splitlines()
