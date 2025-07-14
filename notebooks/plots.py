"""Script to load predictions and plot patches sorted by deforestation area."""

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_predictions_from_h5(h5_file_path: str):
    """Load predictions from HDF5 file.
    
    Args:
        h5_file_path: Path to the predictions HDF5 file
        
    Returns:
        tuple: (predictions, binary_predictions, targets, metadata)
    """
    if not Path(h5_file_path).exists():
        raise FileNotFoundError(
            f"Predictions file not found at {h5_file_path}"
        )
    
    with h5py.File(h5_file_path, 'r') as f:
        predictions = f['predictions'][:]
        binary_predictions = f['binary_predictions'][:]
        targets = f['targets'][:]
        
        # Load metadata
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
            
        print(f"Loaded predictions from {h5_file_path}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Binary predictions shape: {binary_predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Metadata: {metadata}")
        
        return predictions, binary_predictions, targets, metadata


def calculate_deforestation_area(targets):
    """Calculate deforestation area for each target patch.
    
    Args:
        targets: Array of target patches with shape (N, 1, 1, H, W)
        
    Returns:
        np.ndarray: Array of deforestation areas for each patch
    """
    # Handle different possible shapes
    if targets.ndim == 5:  # (N, 1, 1, H, W)
        # Sum over spatial dimensions (H, W) for each patch
        deforestation_areas = targets.sum(axis=(2, 3, 4))  # Shape: (N, 1)
        deforestation_areas = deforestation_areas.flatten()  # Shape: (N,)
    elif targets.ndim == 4:  # (N, 1, H, W)
        deforestation_areas = targets.sum(axis=(2, 3))  # Shape: (N, 1)
        deforestation_areas = deforestation_areas.flatten()  # Shape: (N,)
    elif targets.ndim == 3:  # (N, H, W)
        deforestation_areas = targets.sum(axis=(1, 2))  # Shape: (N,)
    else:
        raise ValueError(f"Unexpected targets shape: {targets.shape}")
    
    return deforestation_areas


def sort_by_deforestation_area(predictions, binary_predictions, targets):
    """Sort patches by deforestation area in descending order.
    
    Args:
        predictions: Raw predictions array
        binary_predictions: Binary predictions array
        targets: Targets array
        
    Returns:
        tuple: (sorted_predictions, sorted_binary_predictions, 
                sorted_targets, sorted_indices)
    """
    # Calculate deforestation areas
    deforestation_areas = calculate_deforestation_area(targets)
    
    # Get indices that would sort by deforestation area in descending order
    sorted_indices = np.argsort(deforestation_areas)[::-1]
    
    # Sort all arrays using the indices
    sorted_predictions = predictions[sorted_indices]
    sorted_binary_predictions = binary_predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    print("Sorted patches by deforestation area (descending):")
    print(
        f"Top 10 deforestation areas: "
        f"{deforestation_areas[sorted_indices[:10]]}"
    )
    
    return (sorted_predictions, sorted_binary_predictions,
            sorted_targets, sorted_indices)


def plot_top_patches(predictions, binary_predictions, targets,
                     sorted_indices, n_patches=4):
    """Plot the top n patches with highest deforestation area.
    
    Args:
        predictions: Sorted predictions array
        binary_predictions: Sorted binary predictions array
        targets: Sorted targets array
        sorted_indices: Original indices of sorted patches
        n_patches: Number of patches to plot
    """
    # Set color scale maximum
    vmax = 1.0
    
    # Calculate deforestation areas for display
    deforestation_areas = calculate_deforestation_area(targets)
    
    # Create a figure with subplots for each patch
    fig, axes = plt.subplots(n_patches, 3, figsize=(15, 5*n_patches))
    fig.suptitle('Top 4 Patches with Highest Deforestation Area',
                 fontsize=16, fontweight='bold')
    
    for i in range(n_patches):
        patch_idx = sorted_indices[i]
        deforestation_area = deforestation_areas[i]
        
        # Get the data for this patch
        prediction = predictions[i]
        binary_prediction = binary_predictions[i]
        target = targets[i]
        
        # Handle dimensions - squeeze out single dimensions
        if prediction.ndim == 4:  # (1, 1, H, W)
            prediction = prediction.squeeze()
            binary_prediction = binary_prediction.squeeze()
            target = target.squeeze()
        elif prediction.ndim == 3:  # (1, H, W)
            prediction = prediction.squeeze(0)
            binary_prediction = binary_prediction.squeeze(0)
            target = target.squeeze(0)
        
        # Calculate some statistics
        pred_area = np.sum(binary_prediction)
        target_area = np.sum(target)
        accuracy = np.mean(binary_prediction == target)
        
        # Plot raw prediction
        row = i
        im1 = axes[row, 0].imshow(prediction, cmap='coolwarm', vmin=0, vmax=vmax)
        axes[row, 0].set_title(f'Raw Prediction\n(Patch {patch_idx})')
        axes[row, 0].axis('off')
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.046, pad=0.04)
        
        # Plot binary prediction
        im2 = axes[row, 1].imshow(binary_prediction, cmap='coolwarm',
                                  vmin=0, vmax=vmax)
        axes[row, 1].set_title(
            f'Binary Prediction\n(Pred Area: {pred_area:.0f})'
        )
        axes[row, 1].axis('off')
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # Plot target
        im3 = axes[row, 2].imshow(target, cmap='coolwarm', vmin=0, vmax=vmax)
        axes[row, 2].set_title(
            f'Ground Truth\n(Target Area: {target_area:.0f})'
        )
        axes[row, 2].axis('off')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)
        
        # # Add patch info as text
        # patch_info = (f"Patch {patch_idx}: Deforestation Area = "
        #               f"{deforestation_area:.0f}, Accuracy = {accuracy:.3f}")
        # fig.text(0.02, 0.95 - i*0.23, patch_info, fontsize=10,
        #          fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def main():
    """Main function to load predictions and plot sorted patches."""
    # Path to predictions file
    predictions_path = "../results/predictions.h5"
    
    try:
        # Load predictions
        predictions, binary_predictions, targets, metadata = (
            load_predictions_from_h5(predictions_path)
        )
        
        # Debug: Check data statistics
        print("\n=== DATA ANALYSIS ===")
        print(f"Total number of patches: {len(targets)}")
        print(f"Target data type: {targets.dtype}")
        print(f"Target shape: {targets.shape}")
        print(f"Target min: {targets.min()}, max: {targets.max()}")
        print(f"Target unique values: {np.unique(targets)}")
        
        # Calculate deforestation areas for all patches
        all_deforestation_areas = calculate_deforestation_area(targets)
        print(f"\nDeforestation area statistics:")
        print(f"  Min area: {all_deforestation_areas.min()}")
        print(f"  Max area: {all_deforestation_areas.max()}")
        print(f"  Mean area: {all_deforestation_areas.mean():.2f}")
        print(f"  Median area: {np.median(all_deforestation_areas):.2f}")
        
        # Check distribution of deforestation areas
        zero_patches = np.sum(all_deforestation_areas == 0)
        full_patches = np.sum(all_deforestation_areas == targets[0].size)
        print(f"  Patches with 0 deforestation: {zero_patches}")
        print(f"  Patches with full deforestation: {full_patches}")
        print(f"  Patches with partial deforestation: {len(targets) - zero_patches - full_patches}")
        
        # Show top 10 deforestation areas
        sorted_areas = np.sort(all_deforestation_areas)[::-1]
        print(f"\nTop 10 highest deforestation areas: {sorted_areas[:10]}")
        print(f"Bottom 10 lowest deforestation areas: {sorted_areas[-10:]}")
        
        # Sort by deforestation area
        (sorted_predictions, sorted_binary_predictions,
         sorted_targets, sorted_indices) = sort_by_deforestation_area(
            predictions, binary_predictions, targets
        )
        
        # Debug: Check individual patch statistics
        print(f"\n=== TOP 4 PATCHES ANALYSIS ===")
        for i in range(4):
            patch_idx = sorted_indices[i]
            target_patch = sorted_targets[i]
            if target_patch.ndim > 2:
                target_patch = target_patch.squeeze()
            
            print(f"Patch {patch_idx}:")
            print(f"  Target shape: {target_patch.shape}")
            print(f"  Target unique values: {np.unique(target_patch)}")
            print(f"  Target mean: {target_patch.mean():.4f}")
            print(f"  Non-zero pixels: {np.sum(target_patch > 0)}")
            print(f"  Total pixels: {target_patch.size}")
        
        # Plot top 4 patches
        plot_top_patches(
            sorted_predictions, sorted_binary_predictions, sorted_targets,
            sorted_indices, n_patches=4
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first by "
              "running: python main.py")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

