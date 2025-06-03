import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from Utilities import KLD,MSE,JSD

class KRISPU:
    """
    Kriging with Iterative Spatial Prediction of Uncertainty (KRISPU)
    """

    def __init__(self, X, y, model_class, model_kwargs=None, splitter=LeaveOneOut()):
        self.X = X
        self.y = y
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.splitter = splitter
        self.uncertainty_points = None
        self.fitted_model = None
        self.variance = None
        self.gridx = None
        self.gridy = None
        self.uncertainty_grid = None
        
        #check that X is 2D and that y is 1D and has the same length as X 
        if self.X.ndim != 2 or self.X.shape[1] != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, 2).")
        if self.y.ndim != 1 or len(self.y) != self.X.shape[0]:
            raise ValueError("y must be a 1D array with the same length as X's first dimension.")
        if not isinstance(self.splitter, (LeaveOneOut, KFold)):
            raise ValueError("splitter must be an instance of LeaveOneOut or KFold.")
        if not isinstance(self.model_class, type) or not hasattr(self.model_class, 'execute'):
            raise ValueError("model_class an existing pykrige model class with an 'execute' method.")
        if not isinstance(self.model_kwargs, dict):
            raise ValueError("model_kwargs must be a dictionary of parameters for the model class.")
        if 'variogram_model' not in self.model_kwargs:
            raise ValueError("model_kwargs must include 'variogram_model'.")
        if len(np.unique(self.X, axis=0)) != self.X.shape[0]:
            raise ValueError("X must not contain identical points.")
        if not np.issubdtype(self.X.dtype, np.floating):
            raise ValueError("X must be of floating point type.")
        if not np.issubdtype(self.y.dtype, np.floating):
            raise ValueError("y must be of floating point type.")
        
        
    def fit(self, gridx, gridy):
        """
        Fits the kriging model to the entire dataset and predicts over a grid.
        """
        self.gridx = gridx
        self.gridy = gridy
        model = self.model_class(
            self.X[:, 0], self.X[:, 1], self.y, **self.model_kwargs
        )
        z, ss = model.execute("grid", gridx, gridy)
        self.fitted_model = z
        self.variance = ss

        return z

    def evaluate(self, metric=None):
        """
        Evaluates the model using cross-validation by predicting all coordinates in each fold.

        Computes per-point uncertainty as KLD between the distribution of predictions
        across folds and the ground truth.

        Returns:
            uncertainties (np.ndarray): Per-point uncertainty values.
            mean_uncertainty (float): Mean uncertainty across all points.
        """
        all_fold_preds = []  # Each entry is a prediction of all points from one fold
        y_true_all = self.y.copy()

        for train_index, _ in self.splitter.split(self.X):
            X_train, y_train = self.X[train_index], self.y[train_index]

            model = self.model_class(
                X_train[:, 0], X_train[:, 1], y_train, **self.model_kwargs
            )

            # Predict all coordinates
            z, _ = model.execute("points", self.X[:, 0], self.X[:, 1])
            all_fold_preds.append(z)

        # Convert to 2D array: shape (n_folds, n_points)
        preds_matrix = np.vstack(all_fold_preds).T  # shape: (n_points, n_folds)

        # Compute KLD for each point using its distribution of predictions
        uncertainties = []
        for i in range(preds_matrix.shape[0]):
            preds_i = preds_matrix[i, :]  # predictions for point i across folds
            true_i = np.full_like(preds_i, fill_value=y_true_all[i])
            kld_i = KLD(true_i, preds_i)
            uncertainties.append(kld_i)

        uncertainties = np.array(uncertainties)
        mean_uncertainty = np.mean(uncertainties)
        self.uncertainty_points = (self.X, uncertainties)

        print(f"Mean uncertainty: {mean_uncertainty:.4f}")
        return uncertainties, mean_uncertainty

    def interpolate_uncertainty(self, gridx, gridy):
        """
        Interpolates the predicted uncertainties over a spatial grid.
        """
        if self.uncertainty_points is None:
            raise ValueError("Call evaluate() before interpolating uncertainties.")

        coords, uncertainties = self.uncertainty_points
        x = coords[:, 0]
        y = coords[:, 1]
        z = uncertainties

        kriging = OrdinaryKriging(
            x, y, z, variogram_model='gaussian', verbose=False, enable_plotting=False
        )
        z_grid, _ = kriging.execute("grid", gridx, gridy)
        z_grid = z_grid / np.max(z_grid)
        self.uncertainty_grid = z_grid  # Normalize uncertainty values
        return z_grid

