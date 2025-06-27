"""Plotting utilities for Amazon deforestation data visualization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_active_pixels(
    array_3d,
    frame_idx=0,
    zoom=None,
    cmap_name="inferno",
    vmax=1,
):
    """Plot active pixels from a 3D array.

    Args:
        array_3d: 3D numpy array containing the data to plot
        frame_idx: Index of the frame to plot (default: 0)
        zoom: Optional zoom parameters as [ymin, ymax, xmin, xmax]
        cmap_name: Name of the colormap to use (default: "inferno")
        vmax: Maximum value for color mapping (default: 1)
    """
    frame = array_3d[frame_idx]

    if zoom is not None:
        ymin, ymax, xmin, xmax = [int(z) for z in zoom]
        frame = frame[ymin:ymax, xmin:xmax]

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    img = np.ma.masked_where(frame == 0, frame)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap, vmin=1e-6, vmax=vmax, interpolation="nearest")
    plt.axis("off")
    plt.title(
        f"Frame {frame_idx}" + (f" (zoom {zoom})" if zoom else ""),
        color="white",
    )
    plt.show()
