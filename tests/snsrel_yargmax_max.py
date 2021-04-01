"""Plot argmax max."""
#


def plot_argmax(yargmax, ymax=None):
    try:
        len_ = yargmax.shape[0]
    except Exception:
        len_ = len(yargmax)

    if ymax:
        df = pd.DataFrame({"lang2": range(len_), "argmax": yargmax, "max": ymax})
        sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df)
    else:
        df = pd.DataFrame({"lang2": range(len_), "argmax": yargmax})
        sns.relplot(x="lang2", y="argmax", data=df)


def plot_tset(res):
    shape = np.array(res).shape
    if len(shape) != 2:
        logger.error("shape length not equal to 2: %s", shape)
        return

    if shape[1] == 2:
        df_res = pd.DataFrame(res, columns=["lang2", "argmax"])
        sns.relplot(x="lang2", y="argmax", data=df_res)
        return
        
    if shape[1] == 3:
        df_res = pd.DataFrame(res, columns=["lang2", "argmax", "max"])
        sns.relplot(x="lang2", y="argmax", size="max", hue="max", data=df_res)
        return