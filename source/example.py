import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from Utilities import KLD, MSE, JSD
from KRISPU import KRISPU

if __name__ == "__main__":
    X_coords,Y_coords,Z = np.loadtxt('example_data.csv', unpack=True, dtype=float,delimiter='\t', skiprows=1)    

    X = np.column_stack((X_coords, Y_coords))

    model_kwargs = {'variogram_model': 'linear', 'verbose': False, 'enable_plotting': False}
    krispu = KRISPU(X, Z, model_class=OrdinaryKriging, model_kwargs=model_kwargs)

    # Evaluate uncertainty
    krispu.evaluate(KLD)

    # Define grid
    gridx = np.linspace(4, 11, 200)
    gridy = np.linspace(4, 11, 200)

    # Fit model and interpolate uncertainty
    z = krispu.fit(gridx, gridy)
    uncertainty = krispu.interpolate_uncertainty(gridx, gridy)

    #plotting
    im = plt.imshow(z, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='viridis')
    plt.colorbar(im, label='# to failure (output)')
    plt.scatter(X_coords, Y_coords, c='red', label='Data Points', s=50)
    plt.xlabel('Peak')
    plt.ylabel('width')
    plt.legend()
    plt.savefig('krispu_prediction.png', dpi=300)
    plt.show()
    plt.close()

    #coordinates of highest uncertainty
    max_uncertainty_index = np.unravel_index(np.argmax(uncertainty), uncertainty.shape)
    max_uncertainty_coords = (gridx[max_uncertainty_index[1]], gridy[max_uncertainty_index[0]])
    print(f"Coordinates of highest uncertainty: {max_uncertainty_coords}")

    # Plot both prediction and uncertainty
    im = plt.imshow(uncertainty, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='magma')
    plt.scatter(X_coords, Y_coords, c='red', label='Data Points', s=50)
    plt.scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
    plt.colorbar(im, label='Uncertainty')
    plt.xlabel('Peak')
    plt.ylabel('width')
    plt.savefig('krispu_uncertainty.png', dpi=300)
    plt.show()
    plt.close()