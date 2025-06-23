import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_active_pixels(
        array_3d,
        frame_idx=0,
        zoom=None,  # tuple: (ymin, ymax, xmin, xmax)
        cmap_name='inferno',
        vmax=1
):
    """
    Plota pixels ativos de um frame, fundo preto, zoom opcional.

    Parâmetros:
    -----------
    array_3d : ndarray
        Array 3D de shape (frames, height, width)
    frame_idx : int
        Índice do frame a ser plotado
    zoom : tuple ou None
        (ymin, ymax, xmin, xmax) para cropar área do frame. Se None, plota tudo.
    cmap_name : str
        Colormap matplotlib
    vmax : float
        Valor máximo para escala de cor
    """
    frame = array_3d[frame_idx]

    # Aplicar zoom/crop se solicitado
    if zoom is not None:
        ymin, ymax, xmin, xmax = [int(z) for z in zoom]
        frame = frame[ymin:ymax, xmin:xmax]

    # Preparar colormap: zeros ficam pretos
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_under('black')
    cmap.set_bad('black')

    # Para garantir fundo preto: vmin > 0 e mascarar zeros
    img = np.ma.masked_where(frame == 0, frame)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap, vmin=1e-6, vmax=vmax, interpolation='nearest')
    plt.axis('off')
    plt.title(f'Frame {frame_idx}' + (f' (zoom {zoom})' if zoom else ''), color='white')
    plt.show()
