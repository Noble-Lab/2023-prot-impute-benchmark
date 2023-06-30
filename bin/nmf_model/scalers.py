"""Methods to rescale our input matrices"""
import torch
import numpy as np

class MinMaxScaler():
    """ 
    Min/Max scaling of the matrix
    """
    def __init__(self):
        """ Init the MinMaxScaler """
        self.min = None
        self.max = None

    def fit(self, X):
        """
        Learn the scaling parameters for the matrix

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        self
        """
        self.min = np.nanmin(X)
        self.max = np.nanmax(X)

        return self

    def transform(self, X):
        """
        Transform the input matrix

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        Y : torch.Tensor 
            The rescaled matrix
        """
        Y = (X - self.min) / (self.max - self.min)
        return Y

    def inverse_transform(self, X):
        """
        Inverse transform the input matrix

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        Y : torch.Tensor 
            The (un)scaled matrix
        """

        Y = X * (self.max - self.min) + self.min
        return Y

    def fit_transform(self, X):
        """
        Fit the scaling parameters then transform the
        input matrix

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        torch.Tensor 
            The rescaled matrix
        """
        return self.fit(X).transform(X)


class StandardScaler():
    """Scale a matrix according to the standard deviation of the entire matrix.

    Optionally, the values can be centered as well.

    Parameters
    ----------
    center : bool, optional
        Center the values? This sets the mean of the matrix equal to 0.
    """
    def __init__(self, center=False):
        """Initialize a StandardScaler"""
        self.mean = None
        self.std = None
        self.center = center

    def fit(self, X):
        """Learn the scaling parameters from a matrix.

        Parameters
        ----------
        X : torch.Tensor
            The matrix to learn from.

        Returns
        -------
        self
        """
        nonmissing = X[~torch.isnan(X)]
        self.mean = torch.mean(nonmissing)
        self.std = torch.std(nonmissing)
        return self

    def transform(self, X):
        """Rescale the matrix.

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        torch.Tensor
            The rescaled matrix
        """
        X /= self.std
        if self.center:
           X -= self.mean
        return X


    def inverse_transform(self, X):
        """Reverse the rescaling of the matrix.

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        torch.Tensor
            The matrix on the original scale.
        """
        if self.center:
           X += self.mean
        return X * self.std


    def fit_transform(self, X):
        """Rescale the matrix.

        Parameters
        ----------
        X : torch.Tensor
            The matrix to rescale

        Returns
        -------
        torch.Tensor
            The rescaled matrix
        """
        return self.fit(X).transform(X)
