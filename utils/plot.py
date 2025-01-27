import wandb
from collections.abc import Iterable

""" Plotting utilities for wandb: allows to plot several lines on the same plot. """

def plot_losses(xs, ys, title, keys, xname, vega_spec):

    if not isinstance(xs[0], Iterable) or isinstance(xs[0], (str, bytes)):
        xs = [xs for _ in range(len(ys))]
    assert len(xs) == len(ys), "Number of x-lines and y-lines must match"
    
    if keys is not None:
        assert len(keys) == len(
            ys), "Number of keys and y-lines must match"

    data = [
        [x, f"key_{i}" if keys is None else keys[i], y]
        for i, (xx, yy) in enumerate(zip(xs, ys))
        for x, y in zip(xx, yy)
    ]
    table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])
    custom_line = wandb.plot_table(
        vega_spec,
        table,
        {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
        {"title": title, "xname": xname or "x"}
    )

    return custom_line