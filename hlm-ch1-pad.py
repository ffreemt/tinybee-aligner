# hlm-ch1
# pypi st-bumblebee-aligner\data\hlm-ch1-en.txt
# pypi st-bumblebee-aligner\data\hlm-ch1-zh.txt

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from lozero import logger

if 'get_ipython' in globals():
    # %matplotlib  # get_ipython().run_line_magic('matplotlib', '')
    import matplotlib
    matplotlib.use('tkagg')

cdir = 'C:\\dl\\Dropbox\\mat-dir\\myapps\\pypi-projects\\tinybee-aligner'
os.chdir(cdir)

sys.path.insert(0, "../fetch-embed")
# from tinybee.embed_text import embed_text
from fetch_embed.embed_text import embed_text

# need to use PYTHONPATH environ?
# need to do
# !pip install msgpack textblob nltk
# !pip install rank_bm25

sys.path.insert(0, "../light-scores")
sys.path.insert(0, "../light-aligner")

from light_scores import light_scores
from light_aligner.bingmdx_tr import bingmdx_tr

from tinybee.cmat2tset import cmat2tset
from tinybee.find_pairs import find_pairs
from tinybee.plot_tset import plot_tset
from tinybee.cos_matrix2 import cos_matrix2
from tinybee.optics_trace import optics_trace

# --- w4w
# paras_en = Path("data/wu_ch3_en.txt").read_text("utf8").splitlines()
# paras_zh = Path("data/wu_ch3_zh.txt").read_text("utf8").splitlines()

file_en = r"..\st-bumblebee-aligner\data\hlm-ch1-en.txt"
file_zh = r"..\st-bumblebee-aligner\data\hlm-ch1-zh.txt"
paras_en = Path(file_en).read_text("utf8").splitlines()
paras_zh = Path(file_zh).read_text("utf8").splitlines()

# --- w4w
# paras_w4w = [bingmdx_tr(elm) for elm in paras_en]
paras_w4w = []
for elm in tqdm(paras_en):
    paras_w4w.append(bingmdx_tr(elm))

lmat_w4w = light_scores([" ".join([*elm]) for elm in paras_w4w], [" ".join([*elm]) for elm in paras_zh])

fig = plt.figure()
ax = fig.subplots()
sns.heatmap(lmat_w4w, cmap="gist_earth_r")
ax.invert_yaxis()
plt.ion()
plt.show()

ltset_w4w = cmat2tset(lmat_w4w)
plot_tset(ltset_w4w)
 
labels, n_cl, n_out = optics_trace(ltset_w4w)
labels2, n_cl2, n_out2 = optics_trace(ltset_w4w[:, :2])
