import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
from Utilities import KLD, MSE, JSD
from KRISPU import KRISPU

if __name__ == "__main__":
    X_coords,Y_coords,Z_coords = np.loadtxt('example_data.csv', unpack=True, dtype=float,delimiter='\t', skiprows=1)    

    Points = np.column_stack((X_coords, Y_coords))

    #kwargs for the model see pykrige documentation for more details
    model_kwargs = {'variogram_model': 'spherical', 
                    'verbose': False, 
                    'enable_plotting': False, 
                    'exact_values':True, 
                    'nlags':6
                    }

    # Initialize KRISPU with the data and model class any pykrige model class can be used
    krispu = KRISPU(Points, Z_coords, model_class=UniversalKriging, model_kwargs=model_kwargs)

    krispu.get_stats()

    # Define grid
    gridx = np.linspace(4, 11, 200)
    gridy = np.linspace(4, 11, 200)

    #fit the model and predict
    z_map = krispu.fit(gridx, gridy)
    # Evaluate uncertainty can be done using different metrics, see utilities.py 
    krispu.evaluate(KLD)
    # Interpolate uncertainty 
    uncertainty = krispu.generate_uncertainty_map(gridx, gridy, method='cubic')

    krispu.print_stats()

    #plotting
    im = plt.imshow(z_map, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='viridis')
    plt.colorbar(im, label='C parameter')
    plt.scatter(X_coords, Y_coords, c=Z_coords, label='Data Points', s=50, edgecolor='black')
    plt.xlabel('A parameter')
    plt.ylabel('B parameter')
    plt.legend()
    plt.savefig('krispu_prediction.png', dpi=300)
    plt.show()
    plt.close()

    #coordinates of highest uncertainty
    max_uncertainty_index = np.unravel_index(np.argmax(uncertainty), uncertainty.shape)
    max_uncertainty_coords = (gridx[max_uncertainty_index[1]], gridy[max_uncertainty_index[0]])

    uncertainty[np.where(uncertainty == 0)] = np.nan  # Set zero uncertainty to NaN for better visualization

    # Plot both prediction and uncertainty
    im = plt.imshow(uncertainty, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()), origin='lower', cmap='magma')
    plt.scatter(X_coords, Y_coords, c='red', label='Data Points', s=50)
    plt.scatter(max_uncertainty_coords[0], max_uncertainty_coords[1], c='blue', label='Max Uncertainty', s=100, edgecolor='black')
    plt.colorbar(im, label='Uncertainty')
    plt.xlabel('A parameter')
    plt.ylabel('B parameter')
    plt.legend(loc='upper right')
    plt.savefig('krispu_uncertainty.png', dpi=300)
    plt.show()
    plt.close()