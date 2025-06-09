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

        # Plot prediction
        im = plt.imshow(z, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='viridis')
        plt.colorbar(im, label='# to failure (output)')
        plt.scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        plt.xlabel('Peak')
        plt.ylabel('width')
        plt.legend()
        plt.title(f'Prediction Iteration {iteration+1}')
        plt.savefig(f'figures\krispu_prediction_iter{iteration+1}.png', dpi=300)
        #plt.show()
        plt.close()

        # Find coordinates of highest uncertainty
        max_uncertainty_index = np.unravel_index(np.argmax(uncertainty), uncertainty.shape)
        max_uncertainty_coords = (gridx[max_uncertainty_index[1]], gridy[max_uncertainty_index[0]])

        uncertainty[np.where(uncertainty == 0)] = np.nan  # Set zero uncertainty to NaN for better visualization

        #check if the coordinates are already in X
        if np.any(np.all(X == max_uncertainty_coords, axis=1)):
            print(f"Coordinates {max_uncertainty_coords} already exist in X. Skipping addition.")
            #make a new point by adding a small random perturbation
            max_uncertainty_coords = (max_uncertainty_coords[0] + np.random.uniform(-0.1, 0.1), 
                                      max_uncertainty_coords[1] + np.random.uniform(-0.1, 0.1))
        # Plot uncertainty
        im = plt.imshow(uncertainty, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='magma')
        plt.scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        plt.scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
        plt.colorbar(im, label='Uncertainty')
        plt.xlabel('Peak')
        plt.ylabel('width')
        plt.legend()
        plt.title(f'Uncertainty Iteration {iteration+1}')
        plt.savefig(f'figures\krispu_uncertainty_iter{iteration+1}.png', dpi=300)
        #plt.show()
        plt.close()

        # Add new data point at max uncertainty
        # For demonstration, use the predicted value at that location as the new Z value
        new_X = np.array([[max_uncertainty_coords[0], max_uncertainty_coords[1]]])
        new_Z = np.array([z[max_uncertainty_index[0], max_uncertainty_index[1]]]) + np.random.normal(0, 0.1)  # Adding some noise to the new Z value
        X = np.vstack([X, new_X])
        Z = np.append(Z, new_Z)

    # Plot sum of uncertainties over iterations
    print(f"Sum of uncertainties over iterations: {sum_uncertainty_ls}")
    sum_uncertainty_ls = np.array(sum_uncertainty_ls)

    plt.figure(figsize=(10, 4.5))
    plt.plot(sum_uncertainty_ls, marker='o')
    plt.xlabel('points added')
    plt.ylabel('total uncertainty')
    plt.grid()
    plt.savefig('sum_uncertainties.png', dpi=300)
    plt.tight_layout()
    plt.show()
    plt.close()