"""
BASE_CLEAN
10.5.22

The base PyTorch classes. This version has been cleaned up a bit. 
Getting rid of some of the code that I haven't used in a while. 
The only loss function option here is MSE. 

This module specifically contains two classes:
    1. BaseImputer - An abstract base class that should work for most of our
       imputation models.
    2. FactorizationDataset - A PyTorch Dataset class for matrix factorization
       problems.
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ranksums
import sys
import os

sys.path.append('../')
import util_functions

from .scalers import StandardScaler

torch.use_deterministic_algorithms(True)

class BaseImputer(torch.nn.Module):
    """
    A base class for the boilerplate code.
    This class does most of the heavy lifting!

    Parameters
    ----------
    n_rows : int
        The number of rows in the input matrix.
    n_cols : int
        The number of columns in the input matrix.
    n_row_factors : int
        The number of latent factors used to represent the rows.
    n_col_factors : int
        The number of latent factors used to represent the columns.
    train_batch_size : int
        The number of training examples in each mini-batch. ``None``
        will use the full dataset.
    eval_batch_size : int
        The number of examples in each mini-batch during evaluation. ``None``
        will use the full dataset.
    n_epochs : int, optional
        The number of epochs to train the model.
    patience : int
        The number of epochs to wait for early stopping.
    stopping_tol : float
        The early stopping tolerance. The loss must decrease by at least this
        amount between epochs to avoid early stopping.
    loss_func : string, optional 
        The loss function to use for training. Must be "MSE"
    optimizer : torch.optim.Optimizer, optional
        The uninitialized optimizer to use. ``None`` is
        ``torch.optim.Adam``.
    optimizer_kwargs : Dict
        Keyword arguments for the optimizer.
    non_negative : bool
        Should the latent factors be non-negative?
    """
    def __init__(
        self,
        n_rows,
        n_cols,
        n_row_factors,
        n_col_factors,
        train_batch_size,
        eval_batch_size,
        n_epochs,
        patience,
        stopping_tol,
        loss_func,
        optimizer,
        optimizer_kwargs,
        non_negative,
        rand_seed,
    ):
        super().__init__()

        # Readable
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._n_row_factors = n_row_factors
        self._n_col_factors = n_col_factors
        self._non_negative = non_negative

        # Safely writable
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.stopping_tol = stopping_tol
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        # Latent factors
        self.row_factors = torch.nn.Embedding(n_rows, n_row_factors)
        self.col_factors = torch.nn.Embedding(n_cols, n_col_factors)

        if self._non_negative:
            with torch.no_grad():
                self.row_factors.weight.abs_()
                self.col_factors.weight.abs_()

        # For training
        self._history = []

        # The scaler
        self._scaler = StandardScaler(center=False)

        # For writing the model state to disk
        self.MODELPATH = "./OPT_MODEL_INTERNAL.pt"

        # Need to remove the previously saved model before training a new one
            # FIXME: is there a better way to do this? 
        try:
            os.remove(self.MODELPATH)
        except FileNotFoundError:
            pass

        # Check the loss function. MSE is the only option here
        if loss_func != "MSE":
            raise ValueError("Unrecognized loss function")

        # Set the loss function
        self.loss_func = torch.nn.MSELoss(reduction="mean")
        self.loss = "MSE"

        # Set the random seed
        if rand_seed:
            self.rand_seed = rand_seed
        else:
            self.rand_seed = torch.seed()

        torch.manual_seed(self.rand_seed)


    def forward(self, locs):
        """
        The forward pass.

        Parameters
        ----------
        locs : torch.tensor of shape (batch_size, 2)
            The indices for the rows and columns of the matrix.
        """
        raise NotImplementedError

    def fit(self, X, X_val=None):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like
            The matrix to factorize.
        X_val : array-like, optional
            A validation set to evaluate. This should be the same shape as
            ``X``, with ``np.nan`` in all locations other than the validation
            set values.

        Returns
        -------
        self
        """
        X = _check_tensor(X)
        X = self._scaler.fit_transform(X)

        # Setting zeros to nans, for both matrices
        X[X==0.0] = np.nan
        X_val[X_val==0.0] = np.nan

        train_loader = FactorizationDataLoader(
                                    X,
                                    self.train_batch_size,
                                    shuffle=True,
        )

        eval_loader = FactorizationDataLoader(
                                    X, 
                                    self.eval_batch_size, 
                                    shuffle=True,
        )

        if X_val is not None:
            X_val = _check_tensor(X_val).type_as(self.row_factors.weight)
            X_val = self._scaler.transform(X_val.clone())
            validation_loader = FactorizationDataLoader(
                X_val,
                self.eval_batch_size,
                shuffle=True,
            )

        opt = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        best_loss = np.inf
        stopping_counter = 0

        # Evaluate the model
        loss = self._evaluate(eval_loader, 0, "Train MSE")
        if X_val is not None:
            loss = self._evaluate(
                validation_loader, 0, "Validation MSE")

        # The main training loop -- for each epoch
        for epoch in tqdm(range(1, self.n_epochs + 1), unit="epoch"):

            # Train a mini-batch
            for locs, target in train_loader:
                target = target.type_as(self.row_factors.weight)
                opt.zero_grad()
                pred = self(locs)

                train_loss = self.loss_func(pred, target)
                train_loss.backward()
                opt.step()

                # FIXME: Should turn this off for the NN model 
                    # Something funky happens when I turn this on??!
                if self._non_negative:
                   self._constrain_factors()

            # Evaluate the model -- at the end of each epoch
            loss = self._evaluate(eval_loader, epoch, "Train MSE")
            if X_val is not None:
                loss = self._evaluate(
                    validation_loader, epoch, "Validation MSE")

            # Checkpoint, if the curr validation loss is lower than the 
                # lowest yet recorded validation loss
            if loss < best_loss:
                torch.save(self, self.MODELPATH)
                best_loss = loss

            # Evaluate early stopping -- has validation loss plateaued? 
            if self.stopping_tol > 0 and epoch > 20:
                tol = np.abs((best_loss - loss) / best_loss)
                loss_ratio = loss / best_loss

                if tol < self.stopping_tol:
                    stopping_counter += 1
                else:
                    stopping_counter = 0

                if stopping_counter == self.patience:
                    print("early stopping triggered: standard criteria")
                    break

            # Evaluate early stopping -- is validation loss going back up?
            if X_val is not None and self.stopping_tol > 0 and epoch > 20:
                window2 = np.array(self.history["Validation MSE"][-5:])
                window1 = np.array(self.history["Validation MSE"][-13:-8])

                wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

                if wilcoxon_p < 0.05:
                    print("early stopping triggered: wilcoxon criteria")
                    break

        return self

    def _evaluate(self, loader, epoch, name):
        """
        Evaluate model progress.

        Parameters
        ----------
        loader : torch.DataLoader
            The data loader to use.
        epoch : int
            The current epoch.
        name : str
            The name to use in the history; {"Train MSE", "Validation MSE"}
        """
        self.eval()

        locs = torch.cat(loader.locs, axis=1)

        with torch.no_grad():
            res = [(self(l), t) for l, t in loader]

            pred, target = list(zip(*res))
            pred = torch.cat(pred)
            target = torch.cat(target)

            if self.loss == "MSE":
                loss = self.loss_func(pred, target)
            else:
                loss = self.loss_func(pred, target, locs.T)

            try:
                self._history[epoch][name] = loss.item()
            except IndexError:
                self._history.append({"epoch": epoch, name: loss.item()})

        return loss.item()

    def _constrain_factors(self):
        """
        Constrain latent factors to be non-negative.
        This might not be essential for the NN model? 
        """
        with torch.no_grad():
            self.row_factors.weight.clamp_(0)
            self.col_factors.weight.clamp_(0)

    def transform(self, X):
        """
        Impute missing values using the learned model.

        Parameters
        ----------
        X : array-like,
            The matrix to factorize.

        Returns
        -------
        numpy.ndarray,
            X, with the missing values imputed.
        """
        # revert back to the checkpointed model state
        self = torch.load(self.MODELPATH)
        self.eval()

        # sanity check -- works!
        # print("final epoch: ", list(self.history["epoch"])[-1])

        X = _check_tensor(X).float()
        X = self._scaler.transform(X)

        loader = FactorizationDataLoader(
            X,
            missing=True,
            batch_size=self.eval_batch_size,
            shuffle=True,
        )

        for locs, _ in loader:
            X[tuple(locs.T)] = self(locs)

        X = self._scaler.inverse_transform(X)
        return X.detach().cpu().numpy()

    def train_set_transform(self, X):
        """
        Impute just the training set values

        Parameters
        ----------
        X : array-like,
            The matrix to factorize.

        Returns
        -------
        numpy.ndarray,
            X, with the missing values imputed.
        """
        # revert back to the checkpointed model state
        self = torch.load(self.MODELPATH)
        self.eval()

        X = _check_tensor(X).float()
        X = self._scaler.transform(X)

        loader = FactorizationDataLoader(
            X,
            missing=False,
            batch_size=self.eval_batch_size,
            shuffle=True,
        )

        for locs, _ in loader:
            X[tuple(locs.T)] = self(locs)

        X = self._scaler.inverse_transform(X)
        return X.detach().cpu().numpy()

    def fit_transform(self, X, X_val=None):
        """
        Fit the model, then impute missing values.

        Parameters
        ----------
        X : array-like,
            The matrix to factorize.
        X_val : array-like, optional
            A validation set to evaluate. This should be the same shape as
            ``X``, with ``np.nan`` in all locations other than the validation
            set values.

        Returns
        -------
        numpy.ndarray,
            X, with the missing values imputed.
        """
        return self.fit(X, X_val).transform(X)

    @property
    def history(self):
        """ Training & evaluation loss across every training iter """
        return pd.DataFrame(self._history)

    @property
    def n_rows(self):
        """ The expected number of rows. """
        return self._n_rows

    @property
    def n_cols(self):
        """ The expected number of columns. """
        return self._n_cols

    @property
    def n_row_factors(self):
        """ The number of latent factors representing the rows """
        return self._n_row_factors

    @property
    def n_col_factors(self):
        """ The number of latent factors representing the columns """
        return self._n_col_factor

    @property
    def optimizer(self):
        """ The PyTorch optimizer to use. """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        """ Set the optimizer """
        if opt is None:
            opt = torch.optim.Adam

        self._optimizer = opt

    @property
    def optimizer_kwargs(self):
        """ The keyword arguments for the optimizer. """
        return self._optimizer_kwargs

    @optimizer_kwargs.setter
    def optimizer_kwargs(self, kwargs):
        """ Set the optimizer keyword arguments """
        if kwargs is None:
            kwargs = {}

        self._optimizer_kwargs = kwargs


class FactorizationDataLoader:
    """
    A PyTorch dataset for matrix factorization

    Parameters
    ----------
    X : numpy.ndarray,
        The matrix to factorize.
    missing : bool, optional
        If ``True`` the missing elements are returned.
    shuffle : bool, optional
        Shuffle the order of returned examples.
    batch_size  : int, optional
        The batch size. `None` uses the full dataset.
    """
    def __init__(self, X, batch_size=None, missing=False, shuffle=False):
        """ Initialize a FactorizationDataset """

        # FIXME: this should be a parameter
        g = torch.Generator()
        g.manual_seed(42)

        self.batch_size = batch_size if batch_size is not None else X.numel()

        selected = torch.isnan(X)
        if not missing:
            selected = ~selected

        self.locs = torch.nonzero(selected).T
        if shuffle:
            self.locs = self.locs[:, torch.randperm(self.locs.shape[1], generator=g)]

        # standard split -- returns a single batch
        # self.locs = torch.split(self.locs, self.batch_size)

        # mini-batch split
        self.locs = torch.tensor_split(self.locs, self.batch_size, dim=1)
        self.data = X

    def __iter__(self):
        """ Return an iterable of samples """
        for loc in self.locs:
            yield loc.T, self.data[tuple(loc)]

    def __len__(self):
        """ The number of data points """
        return len(self.locs)


def _get_batch_size(batch_size, dataset_length):
    """
    Get the batch size.

    Parameters
    ----------
    batch_size : int or None,
        If None, return the dataset_length
    dataset_length : int,
        The length of the dataset.

    Returns
    -------
    int
        The batch size to use.
    """
    if batch_size is None:
        return dataset_length

    return batch_size


def _check_tensor(array, n_dim=2, label="X"):
    """
    Check that an array is the correct shape and a tensor.

    This function will also coerce :py:ref:`pandas.DataFrame` and
    :py:ref:`numpy.ndarray` objects to :py:ref:`torch.Tensor` objects.

    Parameters
    ----------
    array : array-like,
        The input array.
    n_dim : int, optional
        The expected number of dimensions.
    label : str, optional
        The label to use in the error message, if needed.

    Returns
    -------
    array : torch.Tensor,
        The array as a :py:ref:`torch.Tensor`
   """
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)

    if len(array.shape) != n_dim:
        raise ValueError(f"{label} must have {n_dim} dimensions.")

    return array

