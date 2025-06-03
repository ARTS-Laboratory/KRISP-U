import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from Utilities import KLD, MSE, JSD
from KRISPU import KRISPU


if __name__ == "__main__":
    X_coords, Y_coords, Z = np.loadtxt('example_data.csv', unpack=True, dtype=float, delimiter='\t', skiprows=1)
    X = np.column_stack((X_coords, Y_coords))

    model_kwargs = {'variogram_model': 'linear', 'verbose': False, 'enable_plotting': False}

    # Define grid
    gridx = np.linspace(4, 11, 200)
    gridy = np.linspace(4, 11, 200)

    sum_uncertainty_ls = []

    n_iterations = 10  # Number of iterations to add new points
    for iteration in range(n_iterations):
        krispu = KRISPU(X, Z, model_class=OrdinaryKriging, model_kwargs=model_kwargs)
        z = krispu.fit(gridx, gridy)
        sum_uncertainty = krispu.evaluate(KLD)
        uncertainty = krispu.interpolate_uncertainty(gridx, gridy)
        sum_uncertainty_ls.append(sum_uncertainty)
        # Plot prediction
        im = plt.imshow(z, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='viridis')
        plt.colorbar(im, label='# to failure (output)')
        plt.scatter(X[:, 0], X[:, 1], c='red', label='Data Points', s=50)
        plt.xlabel('Peak')
        plt.ylabel('width')
        plt.legend()
        plt.title(f'Prediction Iteration {iteration+1}')
        plt.savefig(f'krispu_prediction_iter{iteration+1}.png', dpi=300)
        #plt.show()
        plt.close()

        # Find coordinates of highest uncertainty
        max_uncertainty_index = np.unravel_index(np.argmax(uncertainty), uncertainty.shape)
        max_uncertainty_coords = (gridx[max_uncertainty_index[1]], gridy[max_uncertainty_index[0]])
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
        plt.savefig(f'krispu_uncertainty_iter{iteration+1}.png', dpi=300)
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

    plt.plot(sum_uncertainty_ls, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Uncertainties')
    plt.title('Sum of Uncertainties Over Iterations')
    plt.grid()
    plt.savefig('sum_uncertainties.png', dpi=300)
    plt.show()
    plt.close()