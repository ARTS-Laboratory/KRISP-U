import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
from Utilities import KLD, MSE, JSD
from KRISPU import KRISPU

### plotting code
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')

PREDICTION_VMIN = 9.0
PREDICTION_VMAX = 19.0
###

if __name__ == "__main__":
    X_coords, Y_coords, Z = np.loadtxt('example_data.csv', unpack=True, dtype=float, delimiter='\t', skiprows=1)
    X = np.column_stack((X_coords, Y_coords))

    model_kwargs = {'variogram_model': 'spherical', 'verbose': False, 'enable_plotting': False}

    # Define grid
    gridx = np.linspace(4, 11, 200)
    gridy = np.linspace(4, 11, 200)

    sum_uncertainty_ls = []

    n_iterations = 10  # Number of iterations to add new points
    for iteration in range(n_iterations):
        krispu = KRISPU(X, Z, model_class=UniversalKriging, model_kwargs=model_kwargs)
        z = krispu.fit(gridx, gridy)
        sum_uncertainty = krispu.evaluate(KLD)
        uncertainty = krispu.generate_uncertainty_map(gridx, gridy)
        sum_uncertainty_ls.append(sum_uncertainty)

        z = np.asarray(z, dtype=float)
        z[np.where(uncertainty == 0)] = np.nan

        # Find coordinates of highest uncertainty
        max_uncertainty_index = np.unravel_index(np.argmax(uncertainty), uncertainty.shape)
        max_uncertainty_coords = (gridx[max_uncertainty_index[1]], gridy[max_uncertainty_index[0]])

        #check if the coordinates are already in X
        if np.any(np.all(X == max_uncertainty_coords, axis=1)):
            print(f"Coordinates {max_uncertainty_coords} already exist in X. Skipping addition.")
            #make a new point by adding a small random perturbation
            max_uncertainty_coords = (max_uncertainty_coords[0] + np.random.uniform(-0.1, 0.1), 
                                      max_uncertainty_coords[1] + np.random.uniform(-0.1, 0.1))

        # Plot prediction with the same plotting sequence as uncertainty;
        # only the plotted field and colormap differ.
        im = plt.imshow(
            z,
            extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
            origin='lower',
            cmap='viridis',
            vmin=PREDICTION_VMIN,
            vmax=PREDICTION_VMAX,
        )
        plt.scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        plt.scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
        colorbar = plt.colorbar(im, label='# to failure (output)')
        colorbar.ax.tick_params(labelsize=18)
        colorbar.set_label('# to failure (output)', fontsize=20)
        plt.xlabel('Peak')
        plt.ylabel('width')
        plt.legend()
        plt.title(f'Prediction Iteration {iteration+1}')
        plt.savefig(f'figures/krispu_prediction_iter{iteration+1}.png', dpi=300)
        #plt.show()
        plt.close()

        # Plot uncertainty with the same plotting sequence and a different cmap
        uncertainty[np.where(uncertainty == 0)] = np.nan  # Set zero uncertainty to NaN for better visualization
        im = plt.imshow(
            uncertainty,
            extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
            origin='lower',
            cmap='magma',
            vmin=0.0,
            vmax=1.0,
        )
        plt.scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        plt.scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
        colorbar = plt.colorbar(im, label='Uncertainty')
        colorbar.ax.tick_params(labelsize=18)
        colorbar.set_label('Uncertainty', fontsize=20)
        plt.xlabel('Peak')
        plt.ylabel('width')
        plt.legend()
        plt.title(f'Uncertainty Iteration {iteration+1}')
        plt.savefig(f'figures/krispu_uncertainty_iter{iteration+1}.png', dpi=300)
        #plt.show()
        plt.close()

        # Plot the two panels directly in one figure for the joint animation.
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

        prediction_image = axes[0].imshow(
            z,
            extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
            origin='lower',
            cmap='viridis',
            vmin=PREDICTION_VMIN,
            vmax=PREDICTION_VMAX,
        )
        axes[0].scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        axes[0].scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
        prediction_colorbar = fig.colorbar(prediction_image, ax=axes[0], label='# to failure (output)')
        prediction_colorbar.ax.tick_params(labelsize=18)
        prediction_colorbar.set_label('# to failure (output)', fontsize=20)
        axes[0].set_xlabel('Peak')
        axes[0].set_ylabel('width')
        axes[0].legend()
        axes[0].set_title(f'Prediction Iteration {iteration+1}')

        uncertainty_image = axes[1].imshow(
            uncertainty,
            extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
            origin='lower',
            cmap='magma',
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        axes[1].scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
        uncertainty_colorbar = fig.colorbar(uncertainty_image, ax=axes[1], label='Uncertainty')
        uncertainty_colorbar.ax.tick_params(labelsize=18)
        uncertainty_colorbar.set_label('Uncertainty', fontsize=20)
        axes[1].set_xlabel('Peak')
        axes[1].set_ylabel('width')
        axes[1].legend()
        axes[1].set_title(f'Uncertainty Iteration {iteration+1}')

        fig.tight_layout()
        fig.savefig(f'figures/krispu_prediction_uncertainty_iter{iteration+1}.png', dpi=300)
        plt.close(fig)

        # Add new data point at max uncertainty
        # For demonstration, use the predicted value at that location as the new Z value
        new_X = np.array([[max_uncertainty_coords[0], max_uncertainty_coords[1]]])
        new_Z = np.array([z[max_uncertainty_index[0], max_uncertainty_index[1]]]) + np.random.normal(0, 0.1)  # Adding some noise to the new Z value
        X = np.vstack([X, new_X])
        Z = np.append(Z, new_Z)

    # Plot sum of uncertainties over iterations
    print(f"Sum of uncertainties over iterations: {sum_uncertainty_ls}")
    sum_uncertainty_ls = np.array(sum_uncertainty_ls)

    points_added = np.arange(1, len(sum_uncertainty_ls) + 1)
    plt.figure(figsize=(10, 4.5))
    plt.plot(points_added, sum_uncertainty_ls, marker='o')
    plt.xlabel('points added')
    plt.ylabel('total uncertainty')
    plt.xticks([1, 3, 5, 7, 9])
    plt.yticks(np.arange(0.0, 0.101, 0.02))
    plt.grid()
    plt.tight_layout()
    plt.savefig('sum_uncertainties.png', dpi=300)
    plt.show()
    plt.close()
