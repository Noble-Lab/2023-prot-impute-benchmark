"""
IMPUTE-W-VARIOUS-MODELS
1.11.23

This script generates optimized reconstructions of the yeast 
calibration curves peptide quants matrix with various imputation 
methods: NMF, kNN, missForest, sample min and Gaussian random sample.
You can configure it to do full hyperparam searches if you want. 
It then makes the recon matrices compatiable with the 
`calculate-loq.py` script that Lindsay wrote. This will tell you
which of the reconstructed peptides are "quantifiable", meaning 
they match the expected linear range. You can then use the 
`rescue-experiment-comparisons` python notebook to determine which 
imputation method does the best job of reconstructing the expected 
peptide abundance ratios. 

1.11.23: updating this to add in Gaussian random sample impute, 
missForest impute and sample min impute. Note that this is basically
the same script as `DE-test-simple.py`, only with some data wrangling
at the end to make the imputed matrices compatiable with
`calculate-loq.py`. The other difference is that the input dataset is
the yeast calibration curves instead of the SMTG dataset. 
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# For missForest:
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

# import my modules
sys.path.append('../../bin/')
sys.path.append('../../bin/nmf_model/')
from models.linear import GradNMFImputer
import utils

# suppressing this CUDA initialization warning I always get
    # this could be dangerous
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# plotting templates
sns.set(context="talk", style="ticks") 
pal = sns.color_palette()

#####################################################################
#### CONFIGS 	
#####################################################################
# partitioning params
val_frac = 0.27
test_frac = 0.0
# setting this to 0 ensures that no peptides will be filtered out
min_present = 0     # during partitioning
q_anchor=0.2  # these three for MNAR partition 
t_std=0.2 
brnl_prob=0.3

# NMF model params
n_factors = [1,2,4,8,16,32] # 1,2,4,8,16,32 is default
tolerance = 0.0001          # 0.0001 is default
max_epochs = 1000           # 1000 is default
learning_rate = 0.01        # 0.01 is default
batch_size = 64             # 64 is default
loss_func = "MSE"

# kNN impute params
k_neighbors = [1,2,4,8,16,32] # 1,2,4,8,16,32 is default

# missForest impute params
n_trees = [200]             # [200] is default
max_iters_mf = 12   

MODELPATH = "./best-model.pt"

# the random number generator
rng = np.random.default_rng(seed=18)

# the random state for the partition
split_rand_state = 18

# path to the yeast calibration curve quants data
quants_path = "../../../../data/maccoss-data/yeast-cal-curves.csv"

#####################################################################
#### HELPER FUNCTIONS
#####################################################################
def fit_missForest(train_mat, n_trees, max_iters):
    """
    Use missForest with the default parameters.

    Parameters
    ----------
    train_mat : np.ndarray, the training matrix with missing values.
    n_trees : int, the number of trees to construct forest with
    max_iters : int, the maximum number of missForest training
                    iterations

    Returns
    -------
    np.ndarray, the imputed matrix
    """
    base = importr("base")
    doParallel = importr("doParallel")
    rngtools = importr("rngtools")
    missForest = importr("missForest")

    # Activate automatic conversion of NumPy arrays:
    numpy2ri.activate()

    # Setup parallelization:
    doParallel.registerDoParallel(cores=16)

    # Run missForest:
    imputed, err = missForest.missForest(
                                train_mat, 
                                maxiter=max_iters,
                                ntree=n_trees,
                                parallelize="forests", 
                                verbose=True,
    )
    return np.array(imputed)

#####################################################################
### THE MAIN BODY
###
### All of the normal pre-imputation workflow stuff. Read-in and
### pre-process the yeast calibration curves peptide quants dataset, 
### partition with either strategy, print out some sanity check 
### stats. 
#####################################################################

# read in the yeast calibration curves dataset
quants_orig = pd.read_csv(quants_path)

# retain the original peptide IDs
pids_orig = quants_orig["Peptide"]

# remove extraneous columns
to_remove = ["Peptide", "Protein", "numFragments"]
quants = quants_orig.drop(to_remove, axis=1)

# convert to numpy
quants = np.array(quants)

# convert zeros to NaNs
quants[quants == 0.0] = np.nan

# because this is EncyclopeDIA data we're going to convert anything <1 to NaN 
quants[quants < 1.0] = np.nan

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

# what is the MV fraction of the partitions? 
orig_mv_frac = np.count_nonzero(np.isnan(quants)) / quants.size
train_mv_frac = np.count_nonzero(np.isnan(train)) / train.size
val_mv_frac = np.count_nonzero(np.isnan(val)) / val.size

print("original MV frac: ", np.around(orig_mv_frac, decimals=2))
print("train MV frac: ", np.around(train_mv_frac, decimals=2))
print("val MV frac: ", np.around(val_mv_frac, decimals=2))

# get the optimal n_batches for training and evaluation
if len(~np.isnan(train)) > 100:
    n_batches = int(np.floor(len(~np.isnan(train)) / batch_size))
    # setting the minimum n_batches to 100
    n_batches = max(n_batches, 100) 
else: 
    n_batches = 1

#####################################################################
### NMF IMPUTE
### 
### Standard NMF impute + hyperparam search. For every factor in 
### n_factors, run NMF imputation, and record the best loss. Keep 
### track of the best lost, optimal number of factors and the best 
### NMF reconstruction. 
#####################################################################
# print("working on NMF")

# best_loss_nmf = np.inf 
# best_factors_nmf = 0

# for factors in n_factors:
#     # init model 
#     nmf_model = GradNMFImputer(
#                     n_rows = train.shape[0], 
#                     n_cols = train.shape[1], 
#                     n_factors=factors, 
#                     stopping_tol=tolerance,
#                     train_batch_size=n_batches, 
#                     eval_batch_size=n_batches,
#                     n_epochs=max_epochs, 
#                     loss_func=loss_func,
#                     optimizer=torch.optim.Adam,
#                     optimizer_kwargs={"lr": learning_rate},
#                     non_negative=True,
#                     rand_seed=rng.random(),
#     )
#     # fit and transform
#     nmf_recon = nmf_model.fit_transform(train, val)
#     nmf_full_recon = nmf_model.train_set_transform(train)

#     # get the reconstruction loss, for valid and train sets
#     val_loss_curr = util_functions.mse_func_np(nmf_recon, val)
#     train_loss_curr = util_functions.mse_func_np(nmf_full_recon, train)

#     if val_loss_curr < best_loss_nmf:
#         best_loss_nmf = val_loss_curr
#         best_factors_nmf = factors
#         best_recon_nmf = nmf_recon
#         best_full_recon_nmf = nmf_full_recon
#         best_model_nmf = nmf_model 

# # Get plots for the best performing NMF model 
# intermediate_plots.plot_train_loss(
#             model=best_model_nmf, 
#             PXD="SMTG", 
#             n_row_factors=best_model_nmf.n_row_factors,
#             n_col_factors=best_model_nmf._n_col_factors, 
#             model_type="NMF", 
#             eval_loss="MSE",
#             tail=None,
# )                   
# intermediate_plots.real_v_imputed_basic(
#             recon_mat=best_recon_nmf, 
#             val_mat=val, 
#             PXD="SMTG",
#             row_factors=best_model_nmf.n_row_factors,
#             col_factors=best_model_nmf._n_col_factors,
#             model_type="NMF",
#             log_transform=False,
#             tail="valid",
# )
# intermediate_plots.real_v_imputed_basic(
#             recon_mat=best_full_recon_nmf,
#             val_mat=train,
#             PXD="SMTG",
#             row_factors=best_model_nmf.n_row_factors,
#             col_factors=best_model_nmf._n_col_factors,
#             model_type="NMF", 
#             log_transform=False,
#             tail="train",
# )

#####################################################################
### KNN IMPUTE
###
### Impute with the sk-learn kNN method. Doing full hyperparameter
### search here too.  
#####################################################################
# print(" ")
# print("working on kNN")

# best_knn_loss = np.inf
# opt_neighbors = 0

# for k in k_neighbors:
#     knn_model = KNNImputer(n_neighbors=k)
#     knn_recon = knn_model.fit_transform(train)
#     knn_loss = util_functions.mse_func_np(knn_recon, val)

#     if knn_loss < best_knn_loss:
#         best_knn_loss = knn_loss 
#         best_knn_recon = knn_recon
#         opt_neighbors = k

# intermediate_plots.real_v_imputed_basic(
#             recon_mat=best_knn_recon,
#             val_mat=val,
#             PXD="SMTG",
#             row_factors=opt_neighbors,
#             col_factors=opt_neighbors,
#             model_type="kNN", 
#             log_transform=False,
#             tail="valid",
# )    

#####################################################################
### MISSFOREST IMPUTE
###
### We also need to benchmark relative to missForest. Also doing a 
### hyperparam search here. This is gonna be painfully slow. 
####################################################################
print(" ")
print("working on missForest")

best_mf_loss = np.inf 
opt_trees = 0

for n_tree in n_trees:
    mf_recon = fit_missForest(train, n_tree, max_iters_mf)
    mf_loss = util_functions.mse_func_np(mf_recon, val)

    if mf_loss < best_mf_loss:
        best_mf_loss = mf_loss 
        best_mf_recon = mf_recon 
        opt_trees = n_tree

intermediate_plots.real_v_imputed_basic(
            recon_mat=best_mf_recon, 
            val_mat=val, 
            PXD="SMTG",
            row_factors=opt_trees,
            col_factors=opt_trees,
            model_type="missForest",
            log_transform=False,
            tail="valid",
)

# And write to csv. As a way of checkpointing, just because this step
    # takes so damn long.
best_mf_recon_pd = pd.DataFrame(best_mf_recon)
best_mf_recon_pd.to_csv("data/smtg-missForest-recon.csv", index=None)

#####################################################################
### MIN IMPUTE
###
### Here we're taking the sample min. But note that this could also
### be done as the peptide min, or the absolute min of the matrix. 
#####################################################################
# print(" ")
# print("working on min impute")

# col_min = np.nanmin(train, axis=0)
# nan_idx = np.where(np.isnan(train))
# min_recon = train.copy()
# # nan_idx[1] -> take index of column
# min_recon[nan_idx] = np.take(col_min, nan_idx[1]) 

# # get MSE
# min_loss = util_functions.mse_func_np(min_recon, val)

#####################################################################
### GAUSSIAN RANDOM SAMPLE IMPUTE
###
### Here we're gonna center the normal distribution about the mean 
### of the column mins. We'll use the standard deviation of the 
### column mins as well. NOTE this is not quite what Perseus does, 
### but its proven to be a lot nicer for our data, 
### ie. returns smaller MSE.
#####################################################################
# print(" ")
# print("working on random sample impute")

# # get the column mins
# col_min = np.nanmin(train, axis=0)

# # get the mean and std of the entire training matrix
# train_mean = np.nanmean(train)
# train_sd = np.nanstd(train)

# # get the indicies of the MVs 
# nan_idx = np.where(np.isnan(train))
# std_recon = train.copy()

# # how many total MVs? 
# n_mv = len(nan_idx[0])

# # fill in the MVs with random draws 
# std_recon[nan_idx] = rng.normal(
#                             loc=np.mean(col_min), 
#                             scale=np.std(col_min), 
#                             size=n_mv
# )

# # don't want negative values
# std_recon = np.abs(std_recon)

# # get MSE
# std_loss = util_functions.mse_func_np(std_recon, val)

#####################################################################
### POST-PROCESS AND WRITE TO CSV
###
### Make sure these are compatiable with the calculate-loq.py script.
#####################################################################

print(" ")
print("writing to csvs")

# convert sub 1.0 values to NaNs
# best_recon_nmf[best_recon_nmf < 1.0] = np.nan
# best_knn_recon[best_knn_recon < 1.0] = np.nan
best_mf_recon[best_mf_recon < 1.0] = np.nan
# min_recon[min_recon < 1.0] = np.nan
# std_recon[std_recon < 1.0] = np.nan

# Add the column labels and peptide IDs back
	# so that we can run calculate-loq.py on these

# for the NMF imputed matrix
# NMF_recon = pd.DataFrame(best_recon_nmf)
# NMF_recon.columns = quants_orig.columns[3:]
# NMF_recon["Peptide"] = quants_orig["Peptide"]
# NMF_recon["Protein"] = quants_orig["Protein"]
# NMF_recon["numFragments"] = quants_orig["numFragments"]

# for the kNN imputed matrix
# KNN_recon = pd.DataFrame(best_knn_recon)
# KNN_recon.columns = quants_orig.columns[3:]
# KNN_recon["Peptide"] = quants_orig["Peptide"]
# KNN_recon["Protein"] = quants_orig["Protein"]
# KNN_recon["numFragments"] = quants_orig["numFragments"]

# for the missForest imputed matrix
mf_recon = pd.DataFrame(best_mf_recon)
mf_recon.columns = quants_orig.columns[3:]
mf_recon["Peptide"] = quants_orig["Peptide"]
mf_recon["Protein"] = quants_orig["Protein"]
mf_recon["numFragments"] = quants_orig["numFragments"]

# for the sample min imputed matrix
# min_recon = pd.DataFrame(min_recon)
# min_recon.columns = quants_orig.columns[3:]
# min_recon["Peptide"] = quants_orig["Peptide"]
# min_recon["Protein"] = quants_orig["Protein"]
# min_recon["numFragments"] = quants_orig["numFragments"]

# for the random sample imputed matrix
# std_recon = pd.DataFrame(std_recon)
# std_recon.columns = quants_orig.columns[3:]
# std_recon["Peptide"] = quants_orig["Peptide"]
# std_recon["Protein"] = quants_orig["Protein"]
# std_recon["numFragments"] = quants_orig["numFragments"]

# for the original matrix
    # this already has <1 values converted to NaNs
# quants_orig_nan = pd.DataFrame(quants)
# quants_orig_nan.columns = quants_orig.columns[3:]
# quants_orig_nan["Peptide"] = quants_orig["Peptide"]
# quants_orig_nan["Protein"] = quants_orig["Protein"]
# quants_orig_nan["numFragments"] = quants_orig["numFragments"]

# write these to csvs
# NMF_recon.to_csv("data/cal-curves-NMF-recon-MCAR.csv", index=None)
# KNN_recon.to_csv("data/cal-curves-KNN-recon-MCAR.csv", index=None)
mf_recon.to_csv("data/cal-curves-mf-recon-MCAR.csv", index=None)
# min_recon.to_csv("data/cal-curves-min-recon-MCAR.csv", index=None)
# std_recon.to_csv("data/cal-curves-std-recon-MCAR.csv", index=None)
# quants_orig_nan.to_csv("data/cal-curves-orig-MCAR.csv", index=None)

# and lets also get subsets, for sanity checking
# NMF_recon_sub = NMF_recon[:300]
# KNN_recon_sub = KNN_recon[:300]
mf_recon_sub = mf_recon[:300]
# min_recon_sub = min_recon[:300]
# std_recon_sub = std_recon[:300]
# quants_orig_nan_sub = quants_orig_nan[:300]

# and write to csv
# NMF_recon_sub.to_csv("data/cc-NMF-recon-MCAR-tester.csv", index=None)
# KNN_recon_sub.to_csv("data/cc-KNN-recon-MCAR-tester.csv", index=None)
mf_recon_sub.to_csv("data/cc-mf-recon-MCAR-tester.csv", index=None)
# min_recon_sub.to_csv("data/cc-min-recon-MCAR-tester.csv", index=None)
# std_recon_sub.to_csv("data/cc-std-recon-MCAR-tester.csv", index=None)
# quants_orig_nan_sub.to_csv("data/cc-orig-MCAR-tester.csv", index=None)

print(" ")
print("done!")
print(" ")

