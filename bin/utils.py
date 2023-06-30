"""
UTILS

This module contains commonly used utility and plotting functions. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys

# plotting templates
sns.set(context="talk", style="ticks") 
pal = sns.color_palette()

def plt_obs_vs_imputed(recon_mat, orig_mat, log_scale=False):
    """ 
    Generate an observed vs imputed peptide abundance plot for a 
    given model. Can be for the training or validation set. 

    Parameters
    ----------
    recon_mat : np.ndarray, 
        The reconstructed matrix
    orig_mat : np.ndarray, 
        The original matrix. Can be either the validation or the
        training matrix
    log_scale : bool, 
        Log scale the plot axes? 

    Returns
    -------
    None
    """
    # get index of nan values in original set
    orig_nans = np.isnan(orig_mat)

    # get non-nan values in both matrices, for the {valid, train} 
    # set only
    orig_set = orig_mat[~orig_nans]
    recon_set = recon_mat[~orig_nans]

    # get Pearson's correlation coefficient
    corr_mat = np.corrcoef(x=orig_set, y=recon_set)
    pearson_r = np.around(corr_mat[0][1], 2)

    # initialize the figure
    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x=orig_set, y=recon_set, alpha=0.5)

    ax.set_xlabel('Observed Abundance')
    ax.set_ylabel('Imputed Abundance')

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # add the correlation coefficient
    ax.text(0.95, 0.05, "R: %s"%(pearson_r),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=20)

    set_min = min(np.min(orig_set), np.min(recon_set))
    set_max = max(np.max(orig_set), np.max(recon_set))

    if set_min < 1:
        set_min = 1

    ax.set_xlim(left=set_min, right=set_max)
    ax.set_ylim(bottom=set_min, top=set_max)

    # add diagonal line
    x = np.linspace(set_min, set_max, 100)
    y = x
    plt.plot(x, y, '-r', label='y=x', alpha=0.6)

    plt.minorticks_off()

    plt.show()
    plt.close()
    return

def mcar_partition(
			matrix, 
			val_frac=0.1, 
			test_frac=0.1, 
			min_present=5, 
			random_state=42
):
    """
    Use MCAR procedure to split a data matrix into training, 
    validation, test sets. Note that the fractions of data in the 
    validation and tests sets is only approximate due to the need 
    to drop rows with too much missing data.
    
    Parameters
    ----------
    matrix : array-like,
        The data matrix to split.
    val_frac : float, optional
        The fraction of data to assign to the validation set.
    test_frac : float, optional
        The fraction of data to assign to the test set.
    min_present : int, optional
        The minimum number of non-missing values required in each 
        row of the training set.
    random_state : int or numpy.random.Generator
        The random state for reproducibility.
    
    Returns
    -------
    train_set, val_set, test_set : numpy.ndarray,
        The training set, validation and test sets. In the case
        of validation and test, the non-valid/test set values
        are NaNs
    """
    rng = np.random.default_rng(random_state)
    if val_frac + test_frac > 1:
        raise ValueError(
        	"'val_frac' and 'test_frac' cannot sum to more than 1.")

    # Prepare the matrix:
    matrix = np.array(matrix).astype(float)
    matrix[matrix == 0] = np.nan
    num_present = np.sum(~np.isnan(matrix), axis=1)
    discard = num_present < min_present
    num_discard = discard.sum()

    matrix = np.delete(matrix, discard, axis=0)

    # Assign splits:
    indices = np.vstack(np.nonzero(~np.isnan(matrix)))
    rng.shuffle(indices, axis=1)

    n_val = int(indices.shape[1] * val_frac)
    n_test = int(indices.shape[1] * test_frac)
    n_train = indices.shape[1] - n_val - n_test

    train_idx = tuple(indices[:, :n_train])
    val_idx = tuple(indices[:, n_train:(n_train + n_val)])
    test_idx = tuple(indices[:, -n_test:])

    train_set = np.full(matrix.shape, np.nan)
    val_set = np.full(matrix.shape, np.nan)
    test_set = np.full(matrix.shape, np.nan)

    train_set[train_idx] = matrix[train_idx]
    val_set[val_idx] = matrix[val_idx]
    test_set[test_idx] = matrix[test_idx]

    # Remove Proteins with too many missing values:
    num_present = np.sum(~np.isnan(train_set), axis=1)
    discard = num_present < min_present
    num_discard = discard.sum()

    train_set = np.delete(train_set, discard, axis=0)
    val_set = np.delete(val_set, discard, axis=0)
    test_set = np.delete(test_set, discard, axis=0)

    return train_set, val_set, test_set

def mnar_partition_thresholds_matrix(
								mat, 
								q_anchor=0.2, 
								t_std=0.1, 
								brnl_prob=0.5, 
								min_pres=4,
								rand_state=42,
):
    """
    For a given peptide/protein quants matrix, constructs an 
    equally sized thresholds matrix that is filled with Gaussian 
    selected values, anchored at a given percentile of the 
    peptide/protein quants distribution, with a given standard 
    deviation. For each peptide quants matrix element X_ij, if 
    the corresponding thresholds matrix element T_ij is less, 
    pass. Else, conduct a single Bernoulli trial with specified 
    success probability. If success, X_ij is selected for the mask. 
    Else, pass. 

    Parameters
    ----------
    mat : np.ndarray, 
        The unpartitioned peptide/protein quants matrix
    q_anchor : float, 
        The percentile of the abundance values on which to 
        anchor the thresholds matrix
    t_std : float, 
        How many standard deviations of the quants matrix to use 
        when constructing the thresholds matrix? 
    brnl_prob : float, 
        The probability of success for the Bernoulli draw
    min_pres : int, 
        The minimum number of present values for each row in the
        training and validation sets. 
    rand_state : int, 
        The integer for seeding numpy's random number generator

    Returns
    -------
    train_mat, val_mat, test_mat : np.ndarray, 
        The training, validation & test matrices, respectively
    """
    rng = np.random.default_rng(rand_state)

    # get the specified quantile from the original matrix
    q_thresh = np.nanquantile(mat, q_anchor)
    # get the standard deviation from the original matrix
    quants_std = np.nanstd(mat)

    thresh_mat = rng.normal(
                       loc=q_thresh, 
                       scale=(quants_std * t_std), 
                       size=mat.shape,
    )
    # no longer strictly Gaussian
    thresh_mat = np.abs(thresh_mat)

    # define the training mask
    zeros = np.zeros(shape=mat.shape)
    mask = zeros > 1

    # loop through every entry in the matrix
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            quants_mat_curr = mat[i,j]
            thresh_mat_curr = thresh_mat[i,j]

            if thresh_mat_curr > quants_mat_curr: 
                # for n=1 trials, I believe binomial == bernoulli
                success = rng.binomial(1, brnl_prob)
            else: 
                success = 0

            mask[i,j] = success

    # get the indices corresponding to `True` in the mask
    indices = np.vstack(np.nonzero(mask))
    # shuffle the indices
    rng.shuffle(indices, axis=1)
    # get n=half the number of `True`s in the mask
    n_idx = np.int32(np.floor(indices.shape[1] / 2))
    # divide the indices into two disjoint sets
    val_idx = tuple(indices[:,:n_idx])
    test_idx = tuple(indices[:,n_idx:])
    # init val and test sets
    val_mat = np.full(mat.shape, np.nan)
    test_mat = np.full(mat.shape, np.nan)
    # assign values to val and test sets
    val_mat[val_idx] = mat[val_idx]
    test_mat[test_idx] = mat[test_idx]    
    # define the training set
    train_mat = mat.copy()
    train_mat[mask] = np.nan

    # remove peptides with fewer than min_present present values
    num_present = np.sum(~np.isnan(train_mat), axis=1)
    discard = num_present < min_pres

    train_mat = np.delete(train_mat, discard, axis=0)
    val_mat = np.delete(val_mat, discard, axis=0)
    test_mat = np.delete(test_mat, discard, axis=0)

    return train_mat, val_mat, test_mat

def mse_func(x_mat, y_mat):
    """
    Calculate the MSE for two matricies with missing values. Each
    matrix can contain MVs, in the form of np.nans
    
    Parameters
    ----------
    x_mat : np.ndarray, 
        The first matrix 
    y_mat : np.ndarray, T
        he second matrix
    
    Returns
    -------
    float, the mean squared error between values present 
            across both matrices
    """
    x_rav = x_mat.ravel()
    y_rav = y_mat.ravel()
    missing = np.isnan(x_rav) | np.isnan(y_rav)
    mse = np.sum((x_rav[~missing] - y_rav[~missing])**2)

    return mse / np.sum(~missing)

def plot_loss_curves_utils(model):
    """ 
    Generate model loss vs training epoch plot. For both training
    and validation sets. A basic sanity check method. Note that 
    the scale of the y axis will reflect the scaled values. 

    Parameters
    ----------
    model : {NNImputer, NMFImputer, TransformerFactorizationImputer,
             TransformerFactorizationNNImputer}
        The imputation model
    """
    plt.figure()
    plt.plot(list(model.history.epoch[1:]), 
        list(model.history["Train"][1:]), 
        label="Training loss")
    plt.plot(list(model.history.epoch[1:]), 
        list(model.history["Validation"][1:]), 
        label="Validation loss")

    plt.ylim(ymin=0)

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.show()
    plt.close()

    return
