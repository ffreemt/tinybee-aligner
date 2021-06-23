# adopted from hlm-ch1-pad.py
# poetry add git+https
# poetry update

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import OPTICS
from sklearn.preprocessing import normalize

from tqdm import tqdm
from lozero import logger

if 'get_ipython' in globals():
    # %matplotlib  # get_ipython().run_line_magic('matplotlib', '')
    import matplotlib
    matplotlib.use('tkagg')

cdir = 'C:\\dl\\Dropbox\\mat-dir\\myapps\\pypi-projects\\tinybee-aligner'
os.chdir(cdir)

from batch_tr import batch

sys.path.insert(0, "../fetch-embed")
# from tinybee.embed_text import embed_text
from fetch_embed.embed_text import embed_text

sys.path.insert(0, "../light-scores")
from light_scores import light_scores

# sys.path.insert(0, "../light-aligner")
# from light_aligner.bingmdx_tr import bingmdx_tr

from tinybee.cmat2tset import cmat2tset
from tinybee.find_pairs import find_pairs
from tinybee.plot_tset import plot_tset
from tinybee.cos_matrix2 import cos_matrix2
from tinybee.optics_trace import optics_trace

file_en = r"..\st-bumblebee-aligner\data\hlm-ch1-en.txt"
file_zh = r"..\st-bumblebee-aligner\data\hlm-ch1-zh.txt"

# rid of '\ufeff'
paras_en = Path(file_en).read_text("utf8").replace('\ufeff', '').splitlines()
paras_zh = Path(file_zh).read_text("utf8").replace('\ufeff', '').splitlines()

%time paras_tr = batch_tr(paras_en, 'zh')

paras_tr1 = [re.sub(r'([一-龟])', r'\1 ', elm).strip() for elm in paras_tr]
paras_zh1 = [re.sub(r'([一-龟])', r'\1 ', elm).strip() for elm in paras_zh]

# cmat = light_scores([" ".join([*elm]) for elm in paras_w4w], [" ".join([*elm]) for elm in paras_zh])

cmat0 = normalize(cmat, axis=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.subplots()
sns.heatmap(cmat0, cmap="gist_earth_r", ax=ax)
ax.invert_yaxis()

labels = OPTICS().fit(tset0).labels_

clustered = tset0[labels > -1]

# discard 3rd colmn 0
clu = clustered[clustered[:, 2]>0]

outliers = tset0[labels == -1]

pset = gen_row_align(clu, src_len, tgt_len)

pset0 = [[int(elm0), int(elm1), elm2] for elm0, elm1, elm2 in pset]

aset = gen_aset(pset0, src_len, tgt_len)

aset0 = tuple((elm0, elm1, elm2) for elm0, elm1, elm2 in aset if isinstance(elm2, float))
clu0 = tuple((elm0, elm1, elm2) for elm0, elm1, elm2 in clu)

set(aset0) - set(clu0)  # {}
set(clu0) - set(aset0)  #
# {(6.0, 3.0, 0.31950011846610543),
# (7.0, 4.0, 0.2851313475213671),
# (8.0, 6.0, 0.21791096959255946),
# (12.0, 49.0, 0.14378956860759795),
# (34.0, 114.0, 0.2821991336723522),
# (35.0, 102.0, 0.2381121994978362),
# (47.0, 118.0, 0.2126666652358159),
# (51.0, 98.0, 0.19299812802257518)}
