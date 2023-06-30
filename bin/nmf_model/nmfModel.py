"""This module contains models that we'll use for non-negative matrix
factorization (NMF). Additionally, we can expand this model to perform deep
non-negative matrix factorization.

The standard NMF model is based off of the scikit-learn implementation, but
explicitly supports missing values. We've specifically implemented the
"multiplicative update" (MU) optimization strategy for NMF. Additionally, we
implemented this in PyTorch, so a GPU can be used for optimization if
available.

To use this module in your code, you'll need to add tell Python to search this
directory for modules. This can be simply done by adding the following to your
script:

    import sys
    sys.path.append("../../../bin") # The path to this "bin" directory
    from models import NMFImputer

See the docstrings below for details about how to use each model.

"""
import warnings
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class NMFImputer:
    """Impute with Non-Negative Matrix Factorization

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. The default loss function is Frobenius norm (essentially
    equivalent to MSE), although KL-divergence can also be used. While MSE is
    great for problems with Gaussian noise, using KL-divergence can be
    beneficial for problems with Poisson noise.

    Parameters
    ----------
    n_factors : int, default=None
        The number of latent factors.
    loss : {"frobenius", "kullback-leibler"}, default="frobenius"
        The loss function to be minimized, measuring the distance between X and
        the dot product WH.
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    seed : int, default=None
        The seed used for initialization.
    device : str, default="cpu"
        The device to use for computation. Using a GPU may be faster.
    """

    def __init__(
        self,
        n_factors=None,
        loss="frobenius",
        tol=1e-4,
        max_iter=200,
        seed=None,
        device="cpu",
    ):
        """Initialize an NMF model"""
        # check loss:
        if loss not in {"frobenius", "kullback-leibler"}:
            raise ValueError("Unrecognized loss.")

        self.n_factors = n_factors
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.device = device
        self.H = None
        self.W = None
        self.mask = None
        self.error = None
        self.train_mse = None
        self.val_mse = None
        self._fitted = False
        self._to_np = False

    def transform(self):
        """Return the NMF approximation of X

        Returns
        -------
        X_hat : numpy.ndarray or torch.Tensor
            The NMF approximation of X
        """

        if not self._fitted:
            raise RuntimeError("The model must be fit first.")

        X_hat = torch.mm(self.W, self.H)

        if self._to_np:
            X_hat = X_hat.detach().to("cpu").numpy()

        return X_hat

    def fit(self, X, Y=None):
        """Learn the NMF model for the data X.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            The ORIGINAL data matrix to be decomposed
        Y : numpy.ndarray or torch.Tensor, optional
            The validation set.

        Returns
        -------
        self
        """

        # Set the random seed:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Remember whether to convert X back to a numpy array or not.
        self._to_np = isinstance(X, np.ndarray)

        # Check that X is non-negative
        if (X < 0).any():
            raise ValueError("X must be non-negative.")

        X = torch.Tensor(X).to(device=self.device).float()
        if Y is not None:
            Y = torch.Tensor(Y).to(device=self.device).float()

        eps = torch.finfo(float).eps
        self.mask = X.isnan()

        X_mean = X[~self.mask].mean()

        n_features, n_samples = X.shape
        if self.n_factors is None:
            self.n_factors = n_samples

        n_factors = self.n_factors

        # Initialize W and H. For now these are just random:
        avg = torch.sqrt(X_mean / self.n_factors)
        H = torch.abs(avg * torch.randn(n_factors, n_samples)).to(self.device)
        W = torch.abs(avg * torch.randn(n_features, n_factors)).to(self.device)

        self.error = []
        self.train_mse = []
        if Y is not None:
            self.val_mse = []

        self.error.append(self._calc_loss(X, W, H, eps).item())
        self.train_mse.append(mse(X, W@H).item())
        if Y is not None:
            self.val_mse.append(mse(Y, W@H).item())

        for n_iter in range(1, self.max_iter + 1):
            W *= self._update_w(X, W, H, eps)
            H *= self._update_h(X, W, H, eps)

            if self.tol > 0 and n_iter % 10 == 0:
                self.error.append(self._calc_loss(X, W, H, eps).item())
                self.train_mse.append(mse(X, W@H).item())
                if Y is not None:
                    self.val_mse.append(mse(Y, W@H).item())

                if (self.error[-2] - self.error[-1]) / self.error[0] < self.tol:
                    break

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter
            )

        self.W = W
        self.H = H
        self._fitted = True
        return self

    def _update_w(self, X, W, H, eps):
        """Update W using a multiplicative update

        Returns
        -------
        delta_W : How much to change W.
        """
        if self.loss == "frobenius":
            numerator = _masked_mm(X, H.T, A_mask=self.mask)
            denominator = torch.mm(W, torch.mm(H, H.T))

        else:  # KL-divergence
            WH = torch.mm(W, H)
            WH[WH == 0] = eps
            torch.div(X, WH, out=WH)

            numerator = _masked_mm(WH, H.T, A_mask=self.mask)
            H_sum = torch.sum(H, axis=1)
            denominator = H_sum[None, :]

        return numerator / denominator

    def fit_transform(self, X, Y=None):
        """Learn the NMF model and return the approximation for data X.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            The ORIGINAL data matrix to be decomposed
        Y : numpy.ndarray or torch.Tensor
            The CORRUPTED data matrix to be decomposed
        """
        self.fit(X, Y)
        return self.transform()

    def _update_h(self, X, W, H, eps):
        """Update H using a multiplicative update

        Returns
        -------
        delta_H : The update step for H.
        """
        if self.loss == "frobenius":
            numerator = _masked_mm(W.T, X, B_mask=self.mask)
            denominator = torch.mm(W.T, torch.mm(W, H))

        else:  # KL-divergence
            WH = torch.mm(W, H)
            WH[WH == 0] = eps
            torch.div(X, WH, out=WH)

            numerator = _masked_mm(W.T, WH, B_mask=self.mask)
            W_sum = torch.sum(W, axis=0)
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, None]

        return numerator / denominator

    def _calc_loss(self, X, W, H, eps):
        """ Calculate the loss (Frobenius) """

        if self.loss == "frobenius":
            # whats up with this 'reshape' nonsense?
            diff = (X - torch.mm(W, H)).reshape(-1)[~self.mask.reshape(-1)]
            return torch.sqrt(torch.sum(diff ** 2))

        WH = torch.mm(W, H)
        WH_data = WH.reshape(-1)
        X_data = X.reshape(-1)

        # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
        indices = X_data > eps
        WH_data = WH_data[indices]
        X_data = X_data[indices]

        # used to avoid division by zero
        WH_data[WH_data == 0] = eps

        # KL-divergence
        sum_WH = WH_data.sum()
        res = torch.dot(X_data, torch.log(X_data / WH_data))
        return torch.sqrt(2 * (res + sum_WH - X_data.sum()))

    def calc_mse(self, x, y):
        """Calcuate Mean Squared Error (MSE) btwn two arrays
            Replacing with Will's MSE method, bc it works on arrays
        	that contain NaNs

        Parameters
        ----------
        x : the first array (1D)
        y : the second array (1D)

        Returns
        -------
        MSE of two same-size matrices.
        """
        x_bool = np.isnan(x)
        y_bool = np.isnan(y)
    
        x = x[~x_bool & ~y_bool]
        y = y[~x_bool & ~y_bool]
    
        mse = np.linalg.norm(x.ravel() - y.ravel())**2
        return mse / len(y.ravel())


def mse(X, Y):
    """Calculate the MSE with missing values."""
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    missing = torch.isnan(X) | torch.isnan(Y)
    mse = torch.linalg.norm(X[~missing] - Y[~missing])**2
    return mse / torch.sum(~missing)


# Utility functions -------------------------------------------------------
def _masked_mm(A, B, A_mask=None, B_mask=None):
    """Matrix Multiplication on a masked array."""
    if A_mask is not None:
        A[A_mask] = 0

    if B_mask is not None:
        B[B_mask] = 0

    return torch.mm(A, B)
