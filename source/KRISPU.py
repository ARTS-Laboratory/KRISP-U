import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from Utilities import KLD,MSE,JSD
from scipy.interpolate import griddata

class KRISPU:
    """
    Kriging with Iterative Spatial Prediction of Uncertainty (KRISPU)

    This class implements a kriging model that iteratively predicts uncertainty
    in spatial data using cross-validation. It allows for fitting a kriging model,
    evaluating uncertainty using a specified metric, and interpolating the uncertainty
    over a spatial grid.

    Attributes:
        X (np.ndarray): 2D array of spatial coordinates (shape: n_samples, 2).
        y (np.ndarray): 1D array of target values (shape: n_samples,).
        model_class (type): A pykrige model class (e.g., OrdinaryKriging, UniversalKriging).
        model_kwargs (dict): Parameters for the kriging model.
        splitter (object): A cross-validation splitter (e.g., LeaveOneOut, KFold).
        uncertainty_points (tuple): Coordinates and uncertainties for each point.
        fitted_model (np.ndarray): The fitted kriging model over the grid.
        variance (np.ndarray): Variance of the predictions.
        gridx (np.ndarray): x-coordinates of the grid.
        gridy (np.ndarray): y-coordinates of the grid.
        uncertainty_grid (np.ndarray): Interpolated uncertainty values over the grid.
    Methods:
        fit(gridx, gridy): Fits the kriging model to the dataset and predicts over a grid.
        evaluate(metric): Evaluates the model using cross-validation and computes uncertainties.
        interpolate_uncertainty(gridx, gridy, method='cubic'): Interpolates uncertainties over a spatial grid.
        print_stats(): Prints statistics of the fitted model.
        get_stats(): Analyzes the variogram of the fitted model.
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

    def generate_uncertainty_map(self, gridx, gridy, method='cubic'):
        """
        Interpolates the predicted uncertainties over a spatial grid using interpolation.
        """
        if self.uncertainty_points is None:
            raise ValueError("Call evaluate() before interpolating uncertainties.")


        coords, uncertainties = self.uncertainty_points
        x = coords[:, 0]
        y = coords[:, 1]
        z = uncertainties

        grid_x, grid_y = np.meshgrid(gridx, gridy)
        z_grid = griddata(
            (x, y), z, (grid_x, grid_y), method=method,fill_value=0)
        if np.nanmax(z_grid) > 0:
            z_grid = z_grid / np.nanmax(z_grid)
        self.uncertainty_grid = z_grid  # Normalize uncertainty values
        return z_grid
    def print_stats(self):
        """
        Prints the statistics of the fitted model.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        print(f"Fitted model: {self.model_class.__name__}")
        print(f"Model parameters: {self.model_kwargs}")
        print(f"Grid shape: {self.fitted_model.shape}")
        print(f"Variance shape: {self.variance.shape if self.variance is not None else 'Not computed'}")
        #calculate MSE, R2 
        #preidct value at the original points
        if self.X is not None and self.y is not None:
            model = self.model_class(
                self.X[:, 0], self.X[:, 1], self.y, **self.model_kwargs
            )
            y_pred, _ = model.execute("points", self.X[:, 0], self.X[:, 1])
            mse = np.mean((self.y - y_pred) ** 2)
            r2 = 1 - (np.sum((self.y - y_pred) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2))
            print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
        else:
            print("No original data points available for MSE and R2 calculation.")
    def get_stats(self):
        """
        Analyzes the variogram of the fitted model.
        """
        model = self.model_class(
            self.X[:, 0], self.X[:, 1], self.y, **self.model_kwargs
        )
        model.print_statistics()