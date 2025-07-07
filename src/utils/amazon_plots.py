"""Plotting utilities for Amazon deforestation data visualization."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


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


def plot_prediction_patch(
    prediction: np.ndarray,
    binary_prediction: np.ndarray,
    target: np.ndarray,
    patch_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot a patch showing prediction, binary prediction, and target.

    Args:
        prediction: Raw prediction values (probabilities) from model
        binary_prediction: Binary predictions (0 or 1) after thresholding
        target: Ground truth target values
        patch_idx: Optional patch index for title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    # Handle batch dimension if present
    if prediction.ndim == 4:
        prediction = prediction[0]
        binary_prediction = binary_prediction[0]
        target = target[0]
    
    # Handle channel dimension if present (squeeze first channel)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)
        binary_prediction = binary_prediction.squeeze(0)
        target = target.squeeze(0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot raw prediction
    im1 = axes[0].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Raw Prediction\n(Probabilities)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot binary prediction
    im2 = axes[1].imshow(binary_prediction, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1].set_title('Binary Prediction\n(Thresholded)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot target
    im3 = axes[2].imshow(target, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth\n(Target)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add main title
    if patch_idx is not None:
        fig.suptitle(f'Patch {patch_idx} - Prediction Comparison', 
                     fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Prediction Comparison', 
                     fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    if not show:
        plt.close(fig)


def plot_prediction_from_h5(
    h5_file_path: str,
    patch_idx: int,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Load and plot predictions from h5 file for a specific patch.

    Args:
        h5_file_path: Path to the h5 file containing predictions
        patch_idx: Index of the patch to plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    with h5py.File(h5_file_path, 'r') as f:
        # Load data for specific patch - convert h5py datasets to numpy arrays
        prediction = f['predictions'][patch_idx][:]
        binary_prediction = f['binary_predictions'][patch_idx][:]
        target = f['targets'][patch_idx][:]
        
        # Get threshold from metadata
        threshold = f.attrs.get('threshold', 0.5)
        
        print(f"Loading patch {patch_idx}")
        print(f"Threshold used: {threshold}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction range: [{prediction.min():.4f}, "
              f"{prediction.max():.4f}]")
        print(f"Binary prediction unique values: "
              f"{np.unique(binary_prediction)}")
        print(f"Target unique values: {np.unique(target)}")
        
        # Plot the patch
        plot_prediction_patch(
            prediction=prediction,
            binary_prediction=binary_prediction,
            target=target,
            patch_idx=patch_idx,
            figsize=figsize,
            save_path=save_path,
            show=show,
        )
