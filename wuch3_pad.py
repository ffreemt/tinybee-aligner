import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# need to start in pypi-projects\\light-scores?

if 'get_ipython' in globals():
    %matplotlib  # matplotlib.use("tkagg")

cdir = 'C:\\dl\\Dropbox\\mat-dir\\myapps\\pypi-projects\\tinybee-aligner'
os.chdir(cdir)

sys.path.insert(0, "../fetch-embed")
# from tinybee.embed_text import embed_text
from fetch_embed.embed_text import embed_text
from tinybee.cmat2tset import cmat2tset
from tinybee.find_pairs import find_pairs
from tinybee.plot_tset import plot_tset

from tinybee.cos_matrix2 import cos_matrix2
from tinybee.cmat2tset import cmat2tset

paras_en = Path("data/wu_ch3_en.txt").read_text("utf8").splitlines()
paras_zh = Path("data/wu_ch3_zh.txt").read_text("utf8").splitlines()

%time embed_en = embed_text(paras_en)
# Wall time: 1min 17s

%time embed_zh = embed_text(paras_zh)
# Wall time: 1min 20s

cmat = cos_matrix2(embed_en, embed_zh)

src_len, tgt_len = cmat.shape

sns.heatmap(cmat, cmap="gist_earth_r")

pset = find_pairs(cmat)

%time embed_la_zh = embed_text(paras_zh, bsize=8, endpoint='http://ttw.hopto.org/embed_la')
# bsize=8, 5min 14s

%time embed_la_en = embed_text(paras_en, bsize=8, endpoint='http://ttw.hopto.org/embed_la')
# bsize=8, 4.5 min
# 40s/batch, eta = 40 * (len_//8 + bool(len_ % 8)); eta

cmat_la = cos_matrix2(embed_la_en, embed_la_zh)

sns.heatmap(cmat_la, cmap='gist_earth_r')

ymax_la, xmax_la = cmat_la.shape
pset_la = find_pairs(cmat_la)
plot_tset(pset_la)

plot_tset(pset_la, xmax_la, ymax_la)

# --- w4w
import sys

sys.path.insert(0, "../light-aligner/")
sys.path.insert(0, "../light-scores/")
from light_scores import light_scores
from light_aligner.bingmdx_tr import bingmdx_tr

file_en = 'data/wu_ch3_en.txt'
file_tr = 'data/wu_ch3_zh-tr.txt'

paras_en = Path("data/wu_ch3_en.txt").read_text("utf8").splitlines()
paras_tr = Path(file_tr).read_text('utf8').splitlines()

lmat = light_scores(paras_en, paras_tr, norm=1)

fig = plt.figure()
ax = fig.subplots()
sns.heatmap(lmat, cmap="gist_earth_r", ax=ax)
ax.invert_yaxis()

# --- w4w
paras_w4w = [bingmdx_tr(elm) for elm in paras_en]

# this does not work
# lmat_w4w = light_scores(paras_w4w, paras_zh)
# need special treatment for chinese

lmat_w4w = light_scores([" ".join([*elm]) for elm in paras_w4w], [" ".join([*elm]) for elm in paras_zh])

fig = plt.figure()
ax = fig.subplots()
sns.heatmap(lmat_w4w, cmap="gist_earth_r")
ax.invert_yaxis()
# plt.ion()
# plt.show()

ltset_w4w = cmat2tset(lmat_w4w)
plot_tset(ltset_w4w)

opt_w4w = OPTICS(min_samples=2, xi=0.01).fit(ltset_w4w)
opt_w4w.labels_
src_len, tgt_len = lmat_w4w.shape  # tgt_len = len(ltset_w4w)

opt_w4w2 = OPTICS(min_samples=2, xi=0.01).fit(ltset_w4w[:,:2])

# @ snippets dbscan-optics-clustering-sklearn.txt
# n_clusters1 = set([elm for elm in opt1.labels_ if elm > -1]).__len__()
# clustered = sum([1 for elm in opt1.labels_ if elm > -1])
# fig = plt.figure(figsize=(10, 7))
# ax12 = fig.subplots()
# Xk = tset[opt1.labels_ > -1]
# ax12.plot(Xk[:, 0], Xk[:, 1], "g.", alpha=0.8)
# outliers
# ax12.plot(tset[opt1.labels_ == -1, 0], tset[opt1.labels_ == -1, 1], 'bo', alpha=0.9)

# --- opt_w4w ---
n_clusters = set([elm for elm in opt_w4w.labels_ if elm > -1]).__len__()
clustered = sum([1 for elm in opt_w4w.labels_ if elm > -1])
print("n_slusters", n_clusters)
print("clustered", clustered)

min_samples = 2  # default 5
xi = 0.00000001  # default 0.05
opt_w4w = OPTICS(min_samples=min_samples, xi=xi).fit(ltset_w4w)
figw4w = plt.figure(figsize=(10, 7))
axw4w = figw4w.subplots()
Xk = ltset_w4w[opt_w4w.labels_ > -1]
axw4w.plot(Xk[:, 0], Xk[:, 1], "g.", alpha=0.8)
axw4w.plot(ltset_w4w[opt_w4w.labels_ == -1, 0], ltset_w4w[opt_w4w.labels_ == -1, 1], 'bo', alpha=0.9)
axw4w.set_xlim(xmin=0)
axw4w.set_ylim(ymin=0)
axw4w.set_title(f"w4w -- min_samples: {min_samples}, xi: {xi}")
 
# --- opt_w4w2 ---
min_samples = 2  # default 5
xi = 0.00000001  # default 0.05
opt_w4w2 = OPTICS(min_samples=min_samples, xi=xi).fit(ltset_w4w[:,:2])
figw4w2a = plt.figure(figsize=(10, 7))
axw4w2 = figw4w2a.subplots()
Xk = ltset_w4w[:,:2][opt_w4w2.labels_ > -1]
axw4w2.plot(Xk[:, 0], Xk[:, 1], "g.", alpha=0.8)
# outliers
axw4w2.plot(ltset_w4w[:,:2][opt_w4w2.labels_ == -1, 0], ltset_w4w[:,:2][opt_w4w2.labels_ == -1, 1], 'bo', alpha=0.9)
axw4w2.set_xlim(xmin=0)
axw4w2.set_ylim(ymin=0)
axw4w2.set_title(f"min_samples: {min_samples}, xi: {xi}")

# === c_m ===  snippets-mat\matplotlib-pyplot-ax-plot-formt-memo.txt
colors = "bgrcmykw"
colors = "bgrcmyk"
colors = "bgrcmy"  # k (black) for outliers
markers = ".,ov^<>1234sp*hH+xDd|_"
style = ['-', '--', '-.', ':']

c_m = [c + m for m in markers for c in colors]
c_m_s = [c + m + s for s in [''] + style for m in markers for c in colors]
# In [953]: c_m[:10]  # len(c_m) 132
# Out[953]: ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'b,', 'g,', 'r,', 'c,']

"""optics_trace.py in tinybee-aligner/tinybee."""
from typing import List, Union



def optics_trace(tset: List, min_samples: Union[float, int] = 5, xi: float = 0.05):
    """Find and plot optics trace for a given list of triple or tuple.

    Args:
        tset:
        min_samples: refer to sklearn.cluster.OPTICS
        xi: refer to sklearn.cluster.OPTICS

    Returns:
        OPTICS
    """
    ltset_w4w = tset

    opt_w4w = OPTICS(min_samples=min_samples, xi=xi).fit(ltset_w4w)
    figw4w = plt.figure(figsize=(10, 7))
    axw4w = figw4w.subplots()

    n_clusters = set([elm for elm in opt_w4w.labels_ if elm > -1]).__len__()
    clustered = sum([1 for elm in opt_w4w.labels_ if elm > -1])

    # Xk = ltset_w4w[opt_w4w.labels_ > -1]
    # axw4w.plot(Xk[:, 0], Xk[:, 1], "g.", alpha=0.8)
    for klass, color_mark in zip(range(n_clusters), c_m):
        Xk = ltset_w4w[opt_w4w.labels_ == klass]
        axw4w.plot(Xk[:, 0], Xk[:, 1], color_mark, alpha=0.8)

    # outliers
    axw4w.plot(ltset_w4w[opt_w4w.labels_ == -1, 0], ltset_w4w[opt_w4w.labels_ == -1, 1], 'ko', alpha=0.9)
    axw4w.set_xlim(xmin=0)
    axw4w.set_ylim(ymin=0)
    axw4w.set_title(f"w4w -- min_samples: {min_samples}, xi: {xi}")

    return opt_w4w
