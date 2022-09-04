"""Test tk_plot."""
# pylint: disbale=invalid-name

import sys
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import matplotlib
import pytest

from tinybee.tk_plot import tk_plot

t = np.arange(0, 3, 0.01)
# ax.plot(t, 2 * np.sin(2 * np.pi * t))


@pytest.mark.skipif(sys.platform not in ("win32",), reason="github or linux vm nonlocal, no gui")
def test_tk_plot():
    """Test tk_plot."""
    root = tk.Tk()
    root.wm_title("Fig in Tk")

    win = ttk.Frame(root)
    win.pack()

    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)

    canvas = tk_plot(fig, win)

    ax1 = fig.add_subplot(211)
    ax1.plot(t, 2 * np.sin(2 * np.pi * t))

    fig.tight_layout()

    canvas.draw()
    # canvas.cla()  # to clear the canvas

    assert 1

    # close root win in 2 secs
    root.after(12000, root.destroy)

    root.mainloop()
