import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from Utilities import KLD, MSE, JSD
from KRISPU import KRISPU
from scipy.spatial import ConvexHull
from matplotlib.path import Path

if __name__ == "__main__":
    X_coords,Y_coords,Z_coords = np.loadtxt('example_data2.csv', unpack=True, dtype=float,delimiter=' ', skiprows=1)    

    Points = np.column_stack((X_coords, Y_coords))

    boundry_points = 3

    #kwargs for the model see pykrige documentation for more details
    model_kwargs = {'variogram_model': 'gaussian', 
                    'verbose': True, 
                    'enable_plotting': False, 
                    'exact_values':False, 
                    'nlags':3,
                    'anisotropy_scaling': 0.000305, #scaling applied in y direction 
                    }
    
    #if X and Y are diffrent units use anisotropy scaling value calculated below
    x_range = np.max(X_coords) - np.min(X_coords)
    y_range = np.max(Y_coords) - np.min(Y_coords)
    anisotropy_scaling = x_range / y_range
    print(f"Anisotropy scaling factor (y direction): {anisotropy_scaling}")



    #as a sanity check plot what it looks like scaled just use interpolation
    ok = UniversalKriging(X_coords, Y_coords, Z_coords, variogram_model=model_kwargs['variogram_model'],
                        verbose=model_kwargs['verbose'], 
                        enable_plotting=True, 
                        exact_values=model_kwargs['exact_values'], 
                        nlags=model_kwargs['nlags'],
                        anisotropy_scaling=model_kwargs.get('anisotropy_scaling', 1))
    # Define grid for visualization
    gridy = np.linspace(1000, 5000, 1000)
    gridx = np.linspace(0, 2, 1000)

    z_map_ok, _ = ok.execute('grid', gridx, gridy)
    # Create a mask for the convex hull
    hull = ConvexHull(Points)
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    hull_path = Path(Points[hull.vertices])
    inside_hull = hull_path.contains_points(grid_points).reshape(grid_x.shape)
    # Apply convex hull mask to predictions
    z_map_ok_masked = z_map_ok.copy()
    z_map_ok_masked[~inside_hull] = np.nan
    # Create a plot for Ordinary Kriging predictions
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z_map_ok_masked, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
                     origin='lower', cmap='viridis', aspect='auto')
    ax.scatter(Points[:, 0], Points[:, 1], c='red', label='Data Points', s=100, edgecolor='black', linewidth=2)
    ax.plot(Points[hull.vertices, 0], Points[hull.vertices, 1], 'k--', alpha=0.5, label='Convex Hull')
    ax.plot(np.append(Points[hull.vertices, 0], Points[hull.vertices[0], 0]),
                np.append(Points[hull.vertices, 1], Points[hull.vertices[0], 1]), 'k--', alpha=0.5)
    ax.set_xlabel('A parameter')
    ax.set_ylabel('B parameter')
    ax.set_title(f'Ordinary Kriging Predictions Variogram: {model_kwargs.get("variogram_model", "hole-effect")}')
    ax.legend()
    plt.colorbar(im, ax=ax, label='C parameter')
    plt.tight_layout()
    plt.savefig('ordinary_kriging_predictions.png')
    plt.show()


    # Initialize KRISPU with the data and model class any pykrige model class can be used
    krispu = KRISPU(Points, Z_coords, model_class=UniversalKriging, model_kwargs=model_kwargs,n_boundary_points=boundry_points)

    krispu.get_stats()


    #fit the model and predict
    z_map = krispu.fit(gridx, gridy)
    # Evaluate uncertainty can be done using different metrics, see utilities.py 
    krispu.evaluate(KLD)
    # Interpolate uncertainty 
    uncertainty = krispu.generate_uncertainty_map(gridx, gridy, method='cubic')

    krispu.print_stats()

    # Test point selection methods
    print("\n=== Point Selection Methods ===")
    
    # Method 1: Maximum uncertainty
    max_point = krispu.pick_next_point(method='max')
    print(f"Next point using 'max' method: {max_point}")
    
    # Method 2: Weighted centroid with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    centroid_points = []
    
    for threshold in thresholds:
        try:
            centroid_point = krispu.pick_next_point(method='weighted_centroid', threshold=threshold)
            centroid_points.append((threshold, centroid_point))
            print(f"Next point using 'weighted_centroid' method (threshold={threshold}): {centroid_point}")
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
            centroid_points.append((threshold, None))

    # Create convex hull mask to limit predictions to data region
    hull = ConvexHull(Points)
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    hull_path = Path(Points[hull.vertices])
    inside_hull = hull_path.contains_points(grid_points).reshape(grid_x.shape)
    
    # Apply convex hull mask to predictions and uncertainty
    z_map_masked = z_map.copy()
    z_map_masked[~inside_hull] = np.nan
    
    uncertainty_masked = uncertainty.copy()
    uncertainty_masked[~inside_hull] = np.nan
    uncertainty_masked[uncertainty_masked == 0] = np.nan  # Also mask zero uncertainty

    # Create main comparison plot (2x1: predictions and uncertainty)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Masked prediction with data points
    im1 = axes[0].imshow(z_map_masked, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), 
                        origin='lower', cmap='viridis', aspect='auto')
    
    # Separate boundary and interior points for visualization
    coords, uncertainties = krispu.uncertainty_points
    boundary_coords = coords[:krispu.n_boundary_points]
    interior_coords = coords[krispu.n_boundary_points:]
    boundary_z = Z_coords[:krispu.n_boundary_points]
    interior_z = Z_coords[krispu.n_boundary_points:]
    
    # Plot boundary points differently
    if krispu.n_boundary_points > 0:
        axes[0].scatter(boundary_coords[:, 0], boundary_coords[:, 1], c='red', 
                       label=f'Boundary Points', s=100, 
                       edgecolor='black', linewidth=2, marker='s')

    # Plot interior points
    if len(interior_coords) > 0:
        axes[0].scatter(interior_coords[:, 0], interior_coords[:, 1], c='white', 
                       label=f'Interior Points', s=100, 
                       edgecolor='black', linewidth=2, marker='o')

    axes[0].plot(Points[hull.vertices, 0], Points[hull.vertices, 1], 'k--', alpha=0.5, label='Convex Hull')
    axes[0].plot(np.append(Points[hull.vertices, 0], Points[hull.vertices[0], 0]), 
                np.append(Points[hull.vertices, 1], Points[hull.vertices[0], 1]), 'k--', alpha=0.5)
    axes[0].set_xlabel('A parameter')
    axes[0].set_ylabel('B parameter')
    axes[0].set_title(f'A) Kriging Predictions Variogram: {model_kwargs.get("variogram_model", "gaussian")}')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label='C parameter')

    # Plot 2: Uncertainty map with max point
    im2 = axes[1].imshow(uncertainty_masked, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), 
                        origin='lower', cmap='magma', aspect='auto')
    
    # Plot boundary and interior points separately
    if krispu.n_boundary_points > 0:
        axes[1].scatter(boundary_coords[:, 0], boundary_coords[:, 1], c='red', s=100, 
                       label=f'Boundary Points', edgecolor='black', linewidth=2, marker='s')
    
    if len(interior_coords) > 0:
        axes[1].scatter(interior_coords[:, 0], interior_coords[:, 1], c='white', s=100, 
                       label=f'Interior Points', edgecolor='black', linewidth=2, marker='o')
    
    axes[1].scatter(max_point[0], max_point[1], c='cyan', label='Next Test Point (Max Method)', 
                   s=150, edgecolor='black', linewidth=2, marker='^')
    axes[1].plot(Points[hull.vertices, 0], Points[hull.vertices, 1], 'k--', alpha=0.5, label='Convex Hull')
    axes[1].plot(np.append(Points[hull.vertices, 0], Points[hull.vertices[0], 0]), 
                np.append(Points[hull.vertices, 1], Points[hull.vertices[0], 1]), 'k--', alpha=0.5)
    axes[1].set_xlabel('A parameter')
    axes[1].set_ylabel('B parameter')
    axes[1].set_title('B) Uncertainty Map')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='Uncertainty')

    plt.tight_layout()
    plt.savefig('krispu_prediction_uncertainty_masked.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Create separate plots for each weighted centroid threshold
    for threshold, centroid_point in centroid_points:
        if centroid_point is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Create binary mask
            binary_mask = (uncertainty >= threshold).astype(np.uint8)
            binary_mask_masked = binary_mask.copy().astype(float)
            binary_mask_masked[~inside_hull] = np.nan
            
            # Use contrasting colors: blue for low uncertainty (0), red for high uncertainty (1)
            colors = ['lightblue', 'darkred']  # Low uncertainty (below threshold), High uncertainty (above threshold)
            cmap = ListedColormap(colors)
            
            im = ax.imshow(binary_mask_masked, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), 
                          origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=1)
            
            # Create custom legend for the fill colors
            legend_elements = [
                Patch(facecolor='lightblue', label=f'Low Uncertainty (< {threshold})'),
                Patch(facecolor='darkred', label=f'High Uncertainty (≥ {threshold})')
            ]
            
            ax.scatter(X_coords, Y_coords, c='white', s=60, alpha=0.9, label='Data Points', edgecolor='black', linewidth=1, marker='o')
            ax.scatter(centroid_point[0], centroid_point[1], c='lime', 
                      label=f'Next Test Point (Weighted Centroid)', 
                      s=150, edgecolor='black', linewidth=2, marker='^')
            ax.plot(Points[hull.vertices, 0], Points[hull.vertices, 1], 'k-', alpha=0.8, linewidth=2, label='Convex Hull')
            ax.plot(np.append(Points[hull.vertices, 0], Points[hull.vertices[0], 0]), 
                   np.append(Points[hull.vertices, 1], Points[hull.vertices[0], 1]), 'k-', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('A parameter')
            ax.set_ylabel('B parameter')
            ax.set_title(f'Uncertainty-Based Test Point Selection (threshold={threshold})')
            
            # Combine custom legend elements with regular legend
            handles, labels = ax.get_legend_handles_labels()
            all_handles = legend_elements + handles
            ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
            
            plt.tight_layout()
            plt.savefig(f'krispu_weighted_centroid_threshold_{threshold}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

    # Create a detailed comparison plot showing all methods on uncertainty map
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    im = ax.imshow(uncertainty_masked, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), 
                   origin='lower', cmap='magma', aspect='auto')
    
    # Plot boundary and interior points separately
    coords, uncertainties = krispu.uncertainty_points
    boundary_coords = coords[:krispu.n_boundary_points]
    interior_coords = coords[krispu.n_boundary_points:]
    
    if krispu.n_boundary_points > 0:
        ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], c='red', s=100, alpha=0.9, 
                  label=f'Boundary Points', 
                  edgecolor='white', linewidth=2, marker='s')
    
    if len(interior_coords) > 0:
        ax.scatter(interior_coords[:, 0], interior_coords[:, 1], c='white', s=100, alpha=0.8, 
                  label=f'Interior Points', 
                  edgecolor='black', linewidth=2, marker='o')
    
    # Plot max uncertainty point
    ax.scatter(max_point[0], max_point[1], c='cyan', label='Next Test Point (Max Method)', 
               s=150, edgecolor='black', linewidth=2, marker='^')
    
    # Plot weighted centroid points
    colors = ['lime', 'orange', 'magenta']
    
    for i, (threshold, centroid_point) in enumerate(centroid_points):
        if centroid_point is not None:
            ax.scatter(centroid_point[0], centroid_point[1], c=colors[i], 
                      label=f'Next Test Point (Centroid t={threshold})', 
                      s=120, edgecolor='black', linewidth=2, marker='v')
    
    # Add convex hull
    ax.plot(Points[hull.vertices, 0], Points[hull.vertices, 1], 'k--', alpha=0.5, label='Convex Hull')
    ax.plot(np.append(Points[hull.vertices, 0], Points[hull.vertices[0], 0]), 
           np.append(Points[hull.vertices, 1], Points[hull.vertices[0], 1]), 'k--', alpha=0.5)
    
    ax.set_xlabel('A parameter')
    ax.set_ylabel('B parameter')
    ax.set_title('Comparison of Test Point Selection Methods')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Uncertainty')
    
    plt.tight_layout()
    plt.savefig('krispu_all_methods_comparison_masked.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"\n=== Summary ===")
    print(f"Maximum uncertainty point: {max_point}")
    for threshold, centroid_point in centroid_points:
        if centroid_point is not None:
            print(f"Weighted centroid (threshold={threshold}): {centroid_point}")
        else:
            print(f"Weighted centroid (threshold={threshold}): No suitable region found")