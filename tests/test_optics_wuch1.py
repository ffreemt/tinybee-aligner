r"""Test sklearn.cluster.OPTICS."""
import pickle
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS

sns.set()
sns.set_theme(style="darkgrid")

wuch1 = pickle.load(open("tests/nll_matrix_wuch1.pkl", "rb"))


def test_optics_wuch1():
    """Test sklearn.cluster.OPTICS wuch1."""
    df = pd.DataFrame({'x': range(wuch1.shape[1]), 'y': wuch1.argmax(axis=0), 'cos': wuch1.max(0)})

    # import matplotlib.pyplot as plt
    # plt.ion()
    # sns.heatmap(wuch1, cmap="gist_earth_r")
    # plt.gca().invert_yaxis()
    # df.plot.scatter('x', 'y', c='cos', cmap='gist_earth_r')

    assert sum(OPTICS(min_samples=3).fit(df).labels_ > -1) > 20  # 23

    assert sum(OPTICS(min_samples=3, max_eps=3).fit(df).labels_ > -1) >= 10  # 10

    assert sum(OPTICS(min_samples=2, max_eps=3).fit(df).labels_ > -1) >= 19


def test_dbscan_wuch1():
    """Test sklearn.cluster.DBSCAN wuch1."""

    assert sum(DBSCAN(eps=3, min_samples=3).fit(df).labels_ > -1) >= 15
    # df.plot.scatter('x', 'y', c=DBSCAN(eps=4.3, min_samples=3).fit(df).labels_ > -1, cmap='gist_earth_r')

    df1 = df[DBSCAN(eps=4.3, min_samples=3).fit(df).labels_ > -1]

    assert sum(DBSCAN(eps=3, min_samples=3).fit(df1).labels_ > -1) >= 15
