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
        Evaluates the model using cross-validation by removing one point at a time.

        For each fold, removes one point, fits the model to the rest, predicts over the grid,
        computes the metric (e.g., KLD) between the full-data prediction and the leave-one-out prediction
        over the entire field, and assigns that uncertainty to the removed point.

        Returns:
            uncertainties (np.ndarray): Uncertainty value for each original data point (shape: n_samples,).
            mean_uncertainty (float): Mean uncertainty across all points.
        """
        if metric is None:
            raise ValueError("A metric function must be provided for evaluation.")
        if not callable(metric):
            raise ValueError("The metric must be a callable function.")
        if self.gridx is None or self.gridy is None:
            raise ValueError("Grid coordinates (gridx, gridy) must be defined before evaluation, use fit() method first.")

        n_samples = self.X.shape[0]
        uncertainties = np.zeros(n_samples)

        # Fit model on all data to get "ground truth" grid
        model_full = self.model_class(
            self.X[:, 0], self.X[:, 1], self.y, **self.model_kwargs
        )
        z_true, _ = model_full.execute("grid", self.gridx, self.gridy)
        z_true_flat = z_true.ravel()

        for idx, (train_index, test_index) in enumerate(self.splitter.split(self.X)):
            X_train, y_train = self.X[train_index], self.y[train_index]
            model = self.model_class(
                X_train[:, 0], X_train[:, 1], y_train, **self.model_kwargs
            )
            z_pred, _ = model.execute("grid", self.gridx, self.gridy)
            z_pred_flat = z_pred.ravel()

            # Compute uncertainty over the whole field
            uncertainty = metric(z_true_flat, z_pred_flat)
            # Assign this uncertainty to the removed point
            uncertainties[test_index[0]] = uncertainty

        sum_uncertainty = np.sum(uncertainties)
        self.uncertainty_points = (self.X, uncertainties)

        print(f"sum uncertainty: {sum_uncertainty:.4f}")
        return sum_uncertainty

    def interpolate_uncertainty(self, gridx, gridy, method='linear'):
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
            x, y, z, variogram_model=method, verbose=False, enable_plotting=False
        )
        z_grid, _ = kriging.execute("grid", gridx, gridy)
        z_grid = z_grid / np.max(z_grid)
        self.uncertainty_grid = z_grid  # Normalize uncertainty values
        return z_grid

