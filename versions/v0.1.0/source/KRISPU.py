import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from pykrige.ok import OrdinaryKriging
from pykrige.rk import RegressionKriging
from Utilities import KLD,MSE,JSD
from scipy.interpolate import griddata
import warnings
import cv2

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
        n_boundary_points (int): Number of points at the beginning of X to treat as boundary points
                                that should never be removed during cross-validation.
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

    def __init__(self, X, y, model_class, model_kwargs=None, splitter=LeaveOneOut(), n_boundary_points=0):
        self.X = X
        self.y = y
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.splitter = splitter
        self.n_boundary_points = n_boundary_points
        self.uncertainty_points = None
        self.fitted_model = None
        self.variance = None
        self.gridx = None
        self.gridy = None
        self.uncertainty_grid = None
        self.regression_component = None  # Store regression component for RegressionKriging
        self.kriging_component = None     # Store kriging component for RegressionKriging
        
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
        if not isinstance(self.n_boundary_points, int) or self.n_boundary_points < 0:
            raise ValueError("n_boundary_points must be a non-negative integer.")
        if self.n_boundary_points >= self.X.shape[0]:
            raise ValueError("n_boundary_points must be less than the total number of data points.")
        
        
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
        
        The first n_boundary_points are excluded from removal to maintain model stability.

        Returns:
            sum_uncertainty (float): Sum of uncertainty across all evaluated points.
        """
        if metric is None:
            raise ValueError("A metric function must be provided for evaluation.")
        if not callable(metric):
            raise ValueError("The metric must be a callable function.")
        if self.gridx is None or self.gridy is None:
            raise ValueError("Grid coordinates (gridx, gridy) must be defined before evaluation, use fit() method first.")

        n_samples = self.X.shape[0]
        uncertainties = np.zeros(n_samples)

        # Create subset of data excluding boundary points for cross-validation
        if self.n_boundary_points > 0:
            X_eval = self.X[self.n_boundary_points:]  # Points to evaluate (exclude boundary points)
            y_eval = self.y[self.n_boundary_points:]
            X_boundary = self.X[:self.n_boundary_points]  # Boundary points (always kept)
            y_boundary = self.y[:self.n_boundary_points]
            print(f"Excluding first {self.n_boundary_points} boundary points from cross-validation")
            print(f"Evaluating {len(X_eval)} interior points")
        else:
            X_eval = self.X
            y_eval = self.y
            X_boundary = np.empty((0, 2))
            y_boundary = np.empty(0)
            print(f"Evaluating all {n_samples} points")

        # Fit model on all data to get "ground truth" grid
        model_full = self.model_class(
            self.X[:, 0], self.X[:, 1], self.y, **self.model_kwargs
        )

        z_true, _ = model_full.execute("grid", self.gridx, self.gridy)
        z_true_flat = z_true.ravel()

        # Perform cross-validation only on non-boundary points
        for idx in range(len(X_eval)):
            # Create training set: all boundary points + all non-boundary points except current one
            train_indices = np.concatenate([
                np.arange(len(X_boundary)),  # All boundary points
                np.arange(len(X_boundary), len(X_boundary) + len(X_eval))[np.arange(len(X_eval)) != idx]  # Other non-boundary points
            ])
            
            # Combine boundary points with remaining evaluation points
            if len(X_boundary) > 0:
                X_train = np.vstack([X_boundary, X_eval[np.arange(len(X_eval)) != idx]])
                y_train = np.concatenate([y_boundary, y_eval[np.arange(len(X_eval)) != idx]])
            else:
                X_train = X_eval[np.arange(len(X_eval)) != idx]
                y_train = y_eval[np.arange(len(X_eval)) != idx]
            
            model = self.model_class(
                X_train[:, 0], X_train[:, 1], y_train, **self.model_kwargs
            )
            z_pred, _ = model.execute("grid", self.gridx, self.gridy)
            z_pred_flat = z_pred.ravel()

            # Compute uncertainty over the whole field
            uncertainty = metric(z_true_flat, z_pred_flat)
            # Assign this uncertainty to the removed point (accounting for boundary offset)
            uncertainties[self.n_boundary_points + idx] = uncertainty

        # Set boundary points uncertainty to zero (they were never removed)
        if self.n_boundary_points > 0:
            uncertainties[:self.n_boundary_points] = 0.0

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
        #if nans are present, replace them with 0
        if np.isnan(z).any():
            z = np.nan_to_num(z, nan=0.0)
            warnings.warn("NaN values found in uncertainties, may be issue with kriging model")
        grid_x, grid_y = np.meshgrid(gridx, gridy)
        z_grid = griddata(
            (x, y), z, (grid_x, grid_y), method=method,fill_value=0)
        if np.nanmax(z_grid) > 0:
            z_grid = z_grid / np.nanmax(z_grid)
        self.uncertainty_grid = z_grid  # Normalize uncertainty values
        return z_grid
    def pick_next_point(self, method='max',threshold=None):
        """
        Picks the next point to sample based on the uncertainty map.
        
        Parameters:
            method (str): Method to pick the next point ('max', 'random', etc.).
            kwargs (dict): Additional parameters for the method.
        
        Returns:
            tuple: Coordinates of the next point to sample.
        """
        if self.uncertainty_grid is None:
            raise ValueError("Uncertainty grid is not computed. Call evaluate() and generate_uncertainty_map() first.")
        
        if method == 'max':
            if threshold is not None:
                raise ValueError("Threshold is not applicable for 'max' method.")
            max_index = np.unravel_index(np.argmax(self.uncertainty_grid), self.uncertainty_grid.shape)
            return (self.gridx[max_index[1]], self.gridy[max_index[0]])
        elif method == 'weighted_centroid':
            if threshold is None:
                raise ValueError("Threshold must be provided for 'weighted_centroid' method.")

            # Create binary mask based on threshold
            binary_mask = (self.uncertainty_grid >= threshold).astype(np.uint8)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary_mask)
            
            if num_labels <= 1:  # Only background found
                # Fall back to max uncertainty point if no regions above threshold
                max_index = np.unravel_index(np.argmax(self.uncertainty_grid), self.uncertainty_grid.shape)
                return (self.gridx[max_index[1]], self.gridy[max_index[0]])
            
            # Find the largest connected component (excluding background label 0)
            largest_area = 0
            largest_label = 1
            
            for label in range(1, num_labels):
                area = np.sum(labels == label)
                if area > largest_area:
                    largest_area = area
                    largest_label = label
            
            # Get coordinates of the largest uncertain region
            region_mask = (labels == largest_label)
            region_coords = np.where(region_mask)
            
            # Get uncertainty values for this region
            region_uncertainties = self.uncertainty_grid[region_coords]
            
            # Calculate weighted centroid
            total_weight = np.sum(region_uncertainties)
            if total_weight == 0:
                # If weights are zero, use geometric centroid
                centroid_y = np.mean(region_coords[0])
                centroid_x = np.mean(region_coords[1])
            else:
                # Weighted centroid calculation
                centroid_y = np.sum(region_coords[0] * region_uncertainties) / total_weight
                centroid_x = np.sum(region_coords[1] * region_uncertainties) / total_weight
            
            # Convert grid indices to actual coordinates
            # Ensure indices are within bounds
            centroid_x = int(np.clip(centroid_x, 0, len(self.gridx) - 1))
            centroid_y = int(np.clip(centroid_y, 0, len(self.gridy) - 1))
            
            return (self.gridx[centroid_x], self.gridy[centroid_y])

        else:
            raise ValueError(f"Unknown method: {method}")
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