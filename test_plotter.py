import visdom
import numpy as np

vis = visdom.Visdom()
win = "loss_plot"

vis.line(
    Y = np.array([np.random.random(), np.random.random()]).reshape(1, 2),
    X = np.array([0]),
    win = win
)

for k in range(10):
    vis.line(
        Y = np.array([np.random.random(), np.random.random()]).reshape(1, 2),
        X = np.array([k+1]),
        win = win,
        update="append"
    )
