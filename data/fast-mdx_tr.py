# """%matplotlib tk  # matplotlib.use("TkAgg")
"""
slow rank_bm25.BM25Okapi(corpus).get_scores(query)
use texthero tfidf?
 
"""
# 1. translate
from pathlib import Path
import re
import msgpack
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from rank_bm25 import BM25Okapi
import simplemma
from simplemma import text_lemmatizer

from tinybee.plot_tset import plot_tset
from tinybee.cmat2tset import cmat2tset

langdata = simplemma.load_data('en')

DIR_PATH = Path(r"../light-aligner/light_aligner")
DICT_FILE = Path(DIR_PATH, "msbing_c_e.msgpk")
HWD_FILE = Path(DIR_PATH, "msbing_c_e_hw.msgpk")
EHWD_FILE = Path(DIR_PATH, "msbing_c_e_ehw.msgpk")
MDX_DICT = msgpack.load(open(DICT_FILE, "rb"))

import joblib
# paras_en = joblib.load('data/shakespearetxt_en.lzma')[:10000]
# paras_zh = joblib.load('data/shakespearetxt_zh.lzma')[:8000]

file_en = "data/hlm-ch1-en.txt"
file_zh = "data/hlm-ch1-zh.txt"

file_en = "data/wu_ch3_en.txt"
file_zh = "data/wu_ch3_zh.txt"

paras_en = Path(file_en).read_text("utf8").replace('\ufeff', "").lower().splitlines()
paras_en = [elm.strip() for elm in paras_en if elm.strip()]

# %%time
paras_zh = Path(file_zh).read_text("utf8").replace('\ufeff', "").splitlines()
paras_zh = [elm.strip() for elm in paras_zh if elm.strip()]

# paras_en: List[str] =
# clean and filter
paras_en_cl = [" ".join(filter(lambda x: len(x) > 2, line.split())) for line in map(lambda x: re.sub("[^a-zA-Z]", " ", x), paras_en)]  # .5s

# simplemma
paras_en_le = [*map(lambda x: text_lemmatizer(x, langdata), paras_en_cl)]  # 3s, add lower()?

# MDX_DICT, text_util
paras_tr = ["".join("".join(MDX_DICT.get(word, {}).values()) for word in line) for line in paras_en_le]  # 4.7 s

# insert spaces in chinese text
import re
paras_tr1 = [re.sub(r"(?<=[a-zA-Z]) (?=[a-zA-Z])", "", elm.replace("", " ")).split() for elm in paras_tr]  # 3.25 s
paras_zh1 = [re.sub(r"(?<=[a-zA-Z]) (?=[a-zA-Z])", "", elm.replace("", " ")).split() for elm in paras_zh]  # 200ms

"""
# dedup
paras_zh2 =

2. correlation (bm25)
    bm25 = rank_bm25.BM25Okapi(corpus)
        [bm25.get_scores(elm) for elm in tokenized_target_lang]

bm25 = rank_bm25.BM25Okapi(paras_tr1)  # 3.64 s
corr_mat = np.array([bm25.get_scores(sent) for sent in paras_zh1]).T
# 2.1 s/10,  1min 55s/100   2000s/10000?
# cmata 2min 35s/500
# """

from sklearn.preprocessing import normalize
from sklearn.cluster import OPTICS

bm25_ = BM25Okapi(paras_zh1)
corr_mat_ = np.array([bm25_.get_scores(sent) for sent in paras_tr1]).T
corr_mat_ = normalize(corr_mat_, axis=0)
sns.heatmap(corr_mat_, cmap="gist_earth_r")
tset_ = cmat2tset(corr_mat_)
plot_tset(tset_)

print(f"{sum(OPTICS().fit(tset_).labels_ > -1)/45*100:.1f}%")
print(f"{sum(OPTICS(min_samples=2).fit(tset_).labels_ > -1)/45*100:.1f}%")

# MDX_DICT.get(word, {})

# bm25 = BM25Okapi(tokenized_corpus).get_scores(tokenized_query_list)  #

# corr_mat = np.array(corr_mat).T

import simplemma
from simplemma import text_lemmatizer
langdata = simplemma.load_data('en')

%time shakespearetxt_le10000 = [*map(lambda x: text_lemmatizer(x, langdata), shakespearetxt_en[:10000])]
# 7.7 s/10000 lines vs 200s using light_scores.normalize (baed on lemmatize_sentence.py)

simple_tokenizer('地方位置地点场所安置安放认出使人处于某位置'.replace("", " "))

# """

import sys
from pathlib import Path
import re
import msgpack

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import OPTICS

from tinybee.plot_tset import plot_tset
from tinybee.cmat2tset import cmat2tset
from tinybee.gen_row_align import gen_row_align
from tinybee.interpolate_pset import interpolate_pset
from tinybee.gen_aset import gen_aset

sys.path.insert(0,  r"../light-scores/")
from light_scores import light_scores

DIR_PATH = Path(r"../light-aligner/light_aligner")
DICT_FILE = Path(DIR_PATH, "msbing_c_e.msgpk")
HWD_FILE = Path(DIR_PATH, "msbing_c_e_hw.msgpk")
EHWD_FILE = Path(DIR_PATH, "msbing_c_e_ehw.msgpk")
MDX_DICT = msgpack.load(open(DICT_FILE, "rb"))

file_en = "data/hlm-ch1-en.txt"
file_zh = "data/hlm-ch1-zh.txt"
file_en = "data/wu_ch3_en.txt"
file_zh = "data/wu_ch3_zh.txt"

paras_en = Path(file_en).read_text("utf8").replace('\ufeff', "").lower().splitlines()
paras_en = [elm.strip() for elm in paras_en if elm.strip()]

# %%time
paras_zh = Path(file_zh).read_text("utf8").replace('\ufeff', "").splitlines()
paras_zh = [elm.strip() for elm in paras_zh if elm.strip()]

_ = """ #
import joblib
# lines
shakespearetxt_en = joblib.load('data/shakespearetxt_en.lzma')
shakespearetxt_zh = joblib.load('data/shakespearetxt_zh.lzma')
shakespearetxt_en = [elm.lower() for elm in shakespearetxt_en]  # 179918 lines,
shakespearetxt_zh = [elm.lower() for elm in shakespearetxt_zh]

paras_en = shakespearetxt_en
paras_zh = shakespearetxt_zh

%time _ = [*map(clean, paras_en)]  # 13s
# """

def clean(text):
    return re.sub("[^a-zA-Z\n]", " ", text)

def clean_and_filter(text, minlen=3):
    if minlen < 1:
        minlen = 1
    flag = " " + (minlen + 2) * 'x' + " "

    text = re.sub("[^a-zA-Z\n]", " ", text)
    text = text.replace("\n", f"{flag}")
    _ = minlen - 1
    _ = " ".join(filter(lambda x: len(x) > _, text.split()))

    if _.startswith(flag.strip()):
        _ = "\n" + _.lstrip(flag.strip())

    if _.endswith(flag.strip()):
        _ = _.rstrip(flag.strip()) + "\n"

    return _.replace(flag, "\n")

# %time _c = clean("\n".join(shakespearetxt_en))
# 7.18 s

# %time _ = clean_and_filter("\n".join(shakespearetxt_en))
# 6s
# %time _1 = [*map(clean_and_filter, _c.splitlines())]
# 5.06s 6.45s
# %time _2 = clean_and_filter("\n".join(shakespearetxt_en))
# 8.1s
# %time _3 = [*map(clean_and_filter, shakespearetxt_en)]
# 18.8 24s

# clean and filter: ~7-9 s
# %time _c = clean("\n".join(shakespearetxt_en))  # re.sub("[^a-zA-Z\n]", " ", "\n".join(shakespearetxt_en))
# 4.26 s
# %time _d = [*map(lambda x: re.sub("[^a-zA-Z]", " ", x), shakespearetxt_en)]
# 5.63 s
# %time paran_en_cf = [" ".join(filter(lambda x: len(x) > 2, line.split())) for line in _c.splitlines()]
# Wall time: 2.52 s

# clea and filter  # 11-12.1 s
%time paras_en_cf = [" ".join(filter(lambda x: len(x) > 2, line.split())) for line in map(lambda x: re.sub("[^a-zA-Z]", " ", x), shakespearetxt_en)]

# ---
# re_esc = re.escape('。\s?εσ\s?。')
# re.findall('。εσ。', r)
re.split('εσ', r)

# 1.45s/1000lines 54.8s/10000lines  180000
# %%time
paras_tr = []
for line in paras_en:
    # line_tr = "".join(map(str, filter(None, [MDX_DICT.get(word) for word in line.split()])))
    line_tr = "".join(map(str, (MDX_DICT.get(word, "") for word in line.split()])))  # 1.1s
    # _ = """
    # _ = [MDX_DICT.get(word, "") for word in line.split()]
    # _ = [" ".join(d.values()) if d else "" for d in _]
    # line_tr = "".join(" ".join(d.values()) if d else "" for d in [MDX_DICT.get(word, "") for word in line.split()])  #  s
    # """
    paras_tr.append(line_tr)  # 1.25 s 1.44 s

# comprehension supposed to be faster then for loop

%time paras_tr1 = ["".join(map(str, [MDX_DICT.get(word, "") for word in line.split()])) for line in paras_en]
# comprehension shakespearetxt_en[:10000] 1.26s
# comprehension shakespearetxt_en 25.3 s 21.2 s 24 s
# for loop shakespearetxt_en[:10000] 2.42 s
# for loop shakespearetxt_en 28 s 33s

%time paras_tr2 = ["".join(map(str, (MDX_DICT.get(word, "") for word in line.split()))) for line in paras_en]

%timeit paras_tr2 = ["".join(map(str, (MDX_DICT.get(word, {}) for word in line.split()))) for line in paras_en]  # 21s 31.5 s

# translate w4w
%timeit paras_tr3 = ["".join("".join(MDX_DICT.get(word, {}).values()) for word in line.split()) for line in paras_en]  # 16s 22.3 s

# translate clean and filtered
%timeit paras_tr_cf = ["".join("".join(MDX_DICT.get(word, {}).values()) for word in line.split()) for line in paras_en_cf]
# 10.2 s 9s

# also dedup and insert spaces
%time paras_tr4 = [" ".join(" ".join(set("".join(MDX_DICT.get(word, {}).values()))) for word in line.split()) for line in paras_en]  #  1min 30s

# remove all non-chinese
# paras_tr = [re.sub(r"[^一-龟]", "", elm) for elm in paras_tr]
paras_tr = [" ".join(re.findall(r"[一-龟]", elm)) for elm in paras_tr]  # 1min 5s
paras_tr = [" ".join(re.sub(r"[^一-龟]", "", elm)) for elm in paras_tr]  #

# insert space
# vzh = ["".join(re.sub(r"([一-龟])", r" \1 ", elm)) for elm in paras_zh]  #  1min 15s
vzh = [elm.replace("", " ") for elm in paras_zh]  #

# vtr = ["".join(re.sub(r"([一-龟])", r" \1 ", elm)) for elm in paras_tr]  #  23.4 s
v_tr = paras_tr

# %time cmat = light_scores(vtr, vzh)
# %time cmat = light_scores(vtr, vzh[:5000])  #
# %time cmat = light_scores(vtr[:2000], vzh[:1000])  #
# %time cmat = light_scores(vtr[:200], vzh[:100])  #  6.05 s
%time cmat = light_scores(vtr[:500], vzh[:400])  #  29s
cmat0 = normalize(cmat, axis=0)

# sns.heatmap(cmat, cmap="gist_earth_r")
sns.heatmap(cmat0, cmap="gist_earth_r")

tset0 = cmat2tset(cmat0)
plot_tset(tset0)

labels = OPTICS().fit(tset0).labels_

# clustered and outliers
tset_c = tset0[labels > -1]
tset_o = tset0[labels == -1]

plot_tset(tset_c)
plot_tset(tset_o)

src_len, tgt_len = cmat.shape
pset = gen_row_align(tset_c, src_len, tgt_len)
pset = [[int(elm0), int(elm1), float(elm2)] for elm0, elm1, elm2 in pset]

plot_tset(pset)

iset = interpolate_pset(pset, tgt_len)
plot_tset(iset)

aset = gen_aset(pset, src_len, tgt_len)
aset0 = [[elm0, elm1, elm2] for elm0, elm1, elm2 in aset if elm2 not in [""]]

