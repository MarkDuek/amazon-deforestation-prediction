import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_active_pixels(
    array_3d,
    frame_idx=0,
    zoom=None,
    cmap_name="inferno",
    vmax=1,
):
    frame = array_3d[frame_idx]

    if zoom is not None:
        ymin, ymax, xmin, xmax = [int(z) for z in zoom]
        frame = frame[ymin:ymax, xmin:xmax]

    cmap = mpl.cm.get_cmap(cmap_name).copy()
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
