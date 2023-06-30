"""
IMPUTE-FOR-DISTRIBUTIONS-FIG
3.6.23

Impute the four datasets used in the mean-x-variance figure with
our five imputation methods. Does imputation change the distribution
of the means and variances? 
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import torch
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

# suppressing this CUDA initialization warning I always get
    # this could be dangerous
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# import my modules
sys.path.append('../../bin/')
sys.path.append('../../bin/nmf_model/')
from models.linear import GradNMFImputer
import utils

# for missForest:
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, r

# plotting templates
sns.set(context="talk", style="ticks") 
pal = sns.color_palette()

#####################################################################
###  CONFIGS  #######################################################
#####################################################################
# MCAR partitioning params
val_frac = 0.2
test_frac = 0.0
min_present = 4

# MNAR partition params
q_anchor=0.3   
t_std=0.4
brnl_prob=0.5

# NMF model params
n_factors = 4              # 4 is default
tolerance = 0.0001         # 0.0001 is default
max_epochs = 1000          # 1000 is default
learning_rate = 0.01       # 0.01 is default
batch_size = 64            # 64 is default
loss_func = "MSE"

# kNN params
k_neighbors = 4

# missForest impute params
n_trees = 100              # 100 is default, according to mf manual
max_iters_mf = 10          # 10 is default, according to mf manual
r_seed = 36                # the random seed for rpy2
n_cores_mf = 1

# the random number generator
rng = np.random.default_rng(seed=18)

# the random state for the partition
split_rand_state = 18

# The peptide quants matrices
full_path = \
    "/net/noble/vol2/home/lincolnh/code/"\
    "2021_ljharris_ms-impute/data/peptides-data/"

# the (PXDs)
pxds = ["PXD014525", "PXD034525", "PXD016079", "PXD006109"]

#####################################################################
###  HELPER FUNCTIONS  ##############################################
#####################################################################
def mse_func(x_mat, y_mat):    
    """
    Get the mean square error (MSE) between two numpy
    arrays. The input arrays can have missing values.
    
    Parameters
    ----------
    x_mat, y_mat : np.ndarray, 
        The arrays, x and y, to calculate the MSE 
        between
        
    Returns
    ----------
    The MSE
    """
    x_rav = x_mat.ravel()
    y_rav = y_mat.ravel()
    missing = np.isnan(x_rav) | np.isnan(y_rav)
    mse = np.sum((x_rav[~missing] - y_rav[~missing])**2)

    if (np.sum(~missing) == 0):
        print("Warning: Computing MSE from all missing values.")
        return 0
    
    return mse / np.sum(~missing)

def nmf_impute(train_mat, val_mat):
    """
    Impute a peptide quants matrix with my NMF impute
    method. Transforms the validation set.
    
    Parameters
    ----------
    train_mat, val_mat : np.ndarray, 
        The training and validation matrices, respectively
    
    Returns
    ----------
    recon : np.ndarray, 
        The NMF reconstructed matrix
    """
    print(" ")
    print("working on NMF")

    # get the optimal number of training batches for NMF
    if len(~np.isnan(train_mat)) > 100:
        n_batches = int(np.floor(len(~np.isnan(train_mat)) / batch_size))
        # setting the minimum n_batches to 100
        n_batches = max(n_batches, 100) 
    else: 
        n_batches = 1

    # init model 
    nmf_model = GradNMFImputer(
                    n_rows = train_mat.shape[0], 
                    n_cols = train_mat.shape[1], 
                    n_factors=n_factors, 
                    stopping_tol=tolerance,
                    train_batch_size=n_batches, 
                    eval_batch_size=n_batches,
                    n_epochs=max_epochs, 
                    loss_func=loss_func,
                    optimizer=torch.optim.Adam,
                    optimizer_kwargs={"lr": learning_rate},
                    non_negative=True,
                    rand_seed=rng.random(),
    )
    # fit and transform
    recon = nmf_model.fit_transform(train_mat, val_mat)
    
    return recon

def kNN_impute(train_mat):
    """
    Impute a peptide quants matrix with sklearn's
    kNN impute method
    
    Parameters
    ----------
    train_mat : np.ndarray,
        The training matrix
    
    Returns
    ----------
    k_recon : np.ndarray, 
        The kNN imputed matrix
    """
    print(" ")
    print("working on kNN")

    knn_model = KNNImputer(n_neighbors=k_neighbors)
    k_recon = knn_model.fit_transform(train_mat)
    
    return k_recon


def sample_min_impute(train_mat):
    """
    Impute a peptide quants matrix with sample
    (column) min impute. 
    
    Parameters
    ----------
    train_mat : np.ndarray,
        The training matrix
    
    Returns
    ----------
    sm_recon : np.ndarray,
        The sample min reconstructed matrix
    """
    col_min = np.nanmin(train_mat, axis=0)
    nan_idx = np.where(np.isnan(train_mat))
    sm_recon = train_mat.copy()
    # nan_idx[1] -> take index of column
    sm_recon[nan_idx] = np.take(col_min, nan_idx[1])
    
    return sm_recon

def gaussian_sample_impute(train_mat):
    """
    Impute the peptide quants matrix with my 
    custom implementation of Gaussian sample 
    impute.
    
    Parameters
    ----------
    train_mat : np.ndarray, 
        The training matrix. 
    
    Returns
    ----------
    std_recon : np.ndarray,
        The Gaussian random sample reconstructed matrix. 
    """
    # get the column mins
    col_min = np.nanmin(train_mat, axis=0)

    # get the mean and std of the entire training matrix
    train_mean = np.nanmean(train_mat)
    train_sd = np.nanstd(train_mat)

    # get the indicies of the MVs 
    nan_idx = np.where(np.isnan(train_mat))
    std_recon = train_mat.copy()

    # how many total MVs? 
    n_mv = len(nan_idx[0])

    # fill in the MVs with random draws 
    std_recon[nan_idx] = rng.normal(
                                loc=np.mean(col_min), 
                                scale=np.std(col_min), 
                                size=n_mv
    )

    # don't want negative values
    std_recon = np.abs(std_recon)
    
    return std_recon

def missForest_impute(train_mat):
    """
    Impute a peptide quants matrix with missForest
    
    Parameters
    ----------
    train_mat : np.ndarray, 
        The training matrix
        
    Returns
    ----------
    m_recon : np.ndarray,
        The missForest reconstructed matrix
    """
    print(" ")
    print("working on missForest")

    set_seed = r('set.seed')
    set_seed(r_seed)

    base = importr("base")
    doParallel = importr("doParallel")
    rngtools = importr("rngtools")
    missForest = importr("missForest")

    # activate automatic conversion of NumPy arrays
    numpy2ri.activate()

    # set up parallelization
    doParallel.registerDoParallel(cores=n_cores_mf)

    # run missForest
    m_recon, err = missForest.missForest(
                                train_mat, 
                                maxiter=max_iters_mf,
                                ntree=n_trees,
                                parallelize="no", 
                                verbose=True,
    )
    m_recon = np.array(m_recon)
    
    return m_recon

#####################################################################
###  THE MAIN LOOP 
### 
###  For every peptide quants dataset, impute with all five of our
###  imputation methods, and record the reconstruction MSE. 
#####################################################################
# init the reconstruction errors df
cols = [
    "pxd", "NMF MSE", "KNN MSE", "Sample min MSE",
    "Gaussian sample MSE", "missForest MSE",
]
recon_err_mcar = pd.DataFrame(columns=cols)

for pxd in pxds: 
    print(" ")
    print("working on: ", pxd)

    # pre-process the peptide quants df
    quants_raw = pd.read_csv(full_path + pxd + "_peptides.csv")

    # convert 0s to NaNs
    quants_raw[quants_raw == 0] = np.nan
    quants = np.array(quants_raw)
    
    # MCAR partition 
    train, val, test = util_functions.split(
                                    quants, 
                                    val_frac=val_frac,
                                    test_frac=test_frac, 
                                    min_present=min_present,
                                    random_state=split_rand_state,
    )
    # MNAR partition 
    # train, val = util_functions.MNAR_partition_thresholds_matrix(
    #                                     quants, 
    #                                     q_anchor=q_anchor, 
    #                                     t_std=t_std, 
    #                                     brnl_prob=brnl_prob, 
    #                                     min_pres=min_present,
    #                                     rand_state=split_rand_state,
    # )
    
    # take a look at the MV fractions in the three sets
    orig_mv_frac = np.count_nonzero(np.isnan(quants)) / quants.size
    train_mv_frac = np.count_nonzero(np.isnan(train)) / train.size
    val_mv_frac = np.count_nonzero(np.isnan(val)) / val.size

    print("mv frac original: ", np.around(orig_mv_frac, decimals=3))
    print("mv frac train: ", np.around(train_mv_frac, decimals=3))
    print("mv frac validation: ", np.around(val_mv_frac, decimals=3))
    
    # impute the training set with each method
    nmf_recon = nmf_impute(train, val)
    knn_recon = kNN_impute(train)
    smin_recon = sample_min_impute(train)
    gsample_recon = gaussian_sample_impute(train)
    mf_recon = missForest_impute(train)

    # convert to pandas (for the sake of writing)
    nmf_recon_pd = pd.DataFrame(nmf_recon)
    knn_recon_pd = pd.DataFrame(knn_recon)
    smin_recon_pd = pd.DataFrame(smin_recon)
    gsample_recon_pd = pd.DataFrame(gsample_recon)
    mf_recon_pd = pd.DataFrame(mf_recon)

    # save the imputed matrices
    nmf_recon_pd.to_csv(
        "output/" + pxd + "-nmf-imputed.csv", index=None)
    knn_recon_pd.to_csv(
       "output/" + pxd + "-knn-imputed.csv", index=None)
    smin_recon_pd.to_csv(
        "output/" + pxd + "-smin-imputed.csv", index=None)
    gsample_recon_pd.to_csv(
        "output/" + pxd + "-gsample-imputed.csv", index=None)
    mf_recon_pd.to_csv(
       "output/" + pxd + "-mf-imputed.csv", index=None)

print(" ")
print("done with impute!")
