"""
DE-TEST-RUNNER-Petralia2020
6.16.23

Here I'm running the differential expression test on a different 
dataset--Petralia2020--for the sake of manuscript revisions. We're
not running the NN model here, rather just NMF. But we are still 
doing full hyperparam searches for NMF, kNN and missForest. 

This will simulate either MNAR or MCAR missingness.

This one includes both missForest and the no impute condition. No
hyperparam searching for missForest. Adding code so that the 
Precision Recall stats will be output to a dataframe after each 
run. This way we won't need to rerun this entire script just to
make plots. 
"""
import pandas as pd
import numpy as np
import sys
from scipy import stats
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_recall_curve

# for missForest:
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, r

# suppressing this CUDA initialization warning I always get
    # this could be dangerous
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# import my modules
sys.path.append('../../../bin/')
from models.linear import GradNMFImputer
import util_functions
import intermediate_plots

# plotting templates
sns.set(context="talk", style="ticks") 
sns.set_palette("tab10")

#####################################################################
#### CONFIGS 	
#####################################################################
# the datasets
cond1_path = "data/Petralia2020-low-grad-glioma-quants.csv"
cond2_path = "data/Petralia2020-ependymoma-quants.csv"

# original data filtering params
min_pres_orig = 16 # when selecting peptides to include in the DE 
                # analysis, what is the minimum number of present 
                # observations? 

# partitioning params
val_frac = 0.2
test_frac = 0.0
# setting this to 0 ensures that no peptides will be filtered out
min_present = 0     # during partitioning
q_anchor=0.01  # these three for MNAR partition 
t_std=0.2
brnl_prob=0.45

# NMF model params
n_factors = [1]   # [1,2,4,8,16,32] is default
tolerance = 0.0001            # 0.0001 is default
max_epochs = 512             # 1000 is default
learning_rate = 0.01          # 0.01 is default
batch_size = 128               # 64 is default
loss_func = "MSE"

# kNN impute params
k_neighbors = [1] # [1,2,4,8,16,32] is default

# missForest impute params
n_trees = [100]               # [100] is default
max_iters_mf = 10             # 10 is default
r_seed = 36 # the random seed for rpy2
n_cores_mf = 1

# for determining the ground truth DE peptides
    # also the imputed DE peptides
alpha = 0.01
correction = "BH"

# the random number generator
rng = np.random.default_rng(seed=18)

# the random state for the partition
split_rand_state = 18

#####################################################################
#### HELPER FUNCTIONS
#####################################################################
def get_uncorrected_pvalues(mat1, mat2):
    """
    Get the raw (uncorrected) p-values evaluating the
    strength of the null hypothesis between every peptide
    for two experimental conditions
    
    Parameters
    ----------
    mat1, mat2 : np.ndarray, 
        The two quants matrices. Must be the same shape.
    Returns
    ----------
    pvals : np.array, 
        The p-values
    """
    pvals = []
    for i in range(0, mat1.shape[0]):
        res = stats.ttest_ind(mat1[i], mat2[i], nan_policy="omit")
        t_stat = res[0]
        p_val = res[1]
        pvals.append(p_val)

    pvals = np.array(pvals)
    return pvals

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
    # set the R random seed
    set_seed = r('set.seed')
    set_seed(r_seed)

    base = importr("base")
    doParallel = importr("doParallel")
    rngtools = importr("rngtools")
    missForest = importr("missForest")

    # Activate automatic conversion of NumPy arrays:
    numpy2ri.activate()

    # Setup parallelization:
    doParallel.registerDoParallel(cores=n_cores_mf)

    # Run missForest:
    imputed, err = missForest.missForest(
                                train_mat, 
                                maxiter=max_iters,
                                ntree=n_trees,
                                parallelize="no", 
                                verbose=True,
    )
    return np.array(imputed)

#####################################################################
### THE MAIN BODY
###
### Read in and process the peptide quants matrix, perform the 
### initial DE peptides test to establish a ground truth. 
#####################################################################
# read in 
cond1_df = pd.read_csv(cond1_path)
cond2_df = pd.read_csv(cond2_path)

# # get the list of peptide IDs
#     # these will be the same for the two datasets
peptide_ids = np.array(cond1_df["PeptideSequence"])

# now remove the peptide ID columns
to_remove = ["PeptideSequence"]
cond1_df = cond1_df.drop(to_remove, axis=1)
cond2_df = cond2_df.drop(to_remove, axis=1)

# convert to numpy arrays
cond1_quants = np.array(cond1_df)
cond2_quants = np.array(cond2_df)

# subset down to just peptides with a low missinngess fraction
    # the missingness threshold is defined in the configs section
num_present_c1 = np.sum(~np.isnan(cond1_quants), axis=1)
discard = num_present_c1 < min_pres_orig

cond1_quants = np.delete(cond1_quants, discard, axis=0)
cond2_quants = np.delete(cond2_quants, discard, axis=0)
peptide_ids = np.delete(peptide_ids, discard, axis=0)

print("condition 1 quants: ", cond1_quants.shape)
print("condition 2 quants: ", cond2_quants.shape)

# get the ground truth DE peptides
    # what if I don't correct the p-values? 
gt_pvals = get_uncorrected_pvalues(cond1_quants, cond2_quants)
    
reject_null_gt = np.float32(gt_pvals) < alpha
DE_peptides_gt = list(peptide_ids[reject_null_gt])

#####################################################################
### THE GENERAL IMPUTATION WORKFLOW
###
### Create a combined tumor + non-tumor quants matrix, partition
### (MCAR or MNAR), get n_batches
#####################################################################
comb_quants = np.concatenate([cond1_quants, cond2_quants], axis=1)

# this is for eventually separating the reconstructed matrices 
    # into normal and tumor matrices
cols_cutoff = cond1_quants.shape[1]

# MCAR partition 
# train, val, test = util_functions.split(
#                                     comb_quants, 
#                                     val_frac=val_frac,
#                                     test_frac=test_frac, 
#                                     min_present=min_present,
#                                     random_state=split_rand_state,
# )
# print(" ")
# print("MCAR Partition")

# MNAR partition 
train, val = util_functions.MNAR_partition_thresholds_matrix(
                                    comb_quants, 
                                    q_anchor=q_anchor, 
                                    t_std=t_std, 
                                    brnl_prob=brnl_prob, 
                                    min_pres=min_present,
                                    rand_state=split_rand_state,
)
print(" ")
print("MNAR Partition")

# get the missingness fractions
orig_mv_frac = np.count_nonzero(np.isnan(comb_quants)) / comb_quants.size
train_mv_frac = np.count_nonzero(np.isnan(train)) / train.size
val_mv_frac = np.count_nonzero(np.isnan(val)) / val.size

print(" ")
print("original mv frac: ", np.around(orig_mv_frac, decimals=2))
print("training mv frac: ", np.around(train_mv_frac, decimals=2))
print("validation mv frac: ", np.around(val_mv_frac, decimals=2))

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
print("working on NMF")

best_loss_nmf = np.inf 
best_factors_nmf = 0

for factors in n_factors:
	# init model 
	nmf_model = GradNMFImputer(
	                n_rows = train.shape[0], 
	                n_cols = train.shape[1], 
	                n_factors=factors, 
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
	nmf_recon = nmf_model.fit_transform(train, val)
	nmf_full_recon = nmf_model.train_set_transform(train)

	# get the reconstruction loss, for valid and train sets
	val_loss_curr = util_functions.mse_func_np(nmf_recon, val)
	train_loss_curr = util_functions.mse_func_np(nmf_full_recon, train)

	if val_loss_curr < best_loss_nmf:
		best_loss_nmf = val_loss_curr
		best_factors_nmf = factors
		best_recon_nmf = nmf_recon
		best_full_recon_nmf = nmf_full_recon
		best_model_nmf = nmf_model 

# Get plots for the best performing NMF model 
intermediate_plots.plot_train_loss(
            model=best_model_nmf, 
            PXD="SMTG-tester", 
            n_row_factors=best_model_nmf.n_row_factors,
            n_col_factors=best_model_nmf._n_col_factors, 
            model_type="NMF", 
            eval_loss="MSE",
            tail=None,
)                   
intermediate_plots.real_v_imputed_basic(
            recon_mat=best_recon_nmf, 
            val_mat=val, 
            PXD="SMTG-tester",
            row_factors=best_model_nmf.n_row_factors,
            col_factors=best_model_nmf._n_col_factors,
            model_type="NMF",
            log_transform=False,
            tail="valid",
)
intermediate_plots.real_v_imputed_basic(
            recon_mat=best_full_recon_nmf,
            val_mat=train,
            PXD="SMTG-tester",
            row_factors=best_model_nmf.n_row_factors,
            col_factors=best_model_nmf._n_col_factors,
            model_type="NMF", 
            log_transform=False,
            tail="train",
)
#checkpoint
best_recon_nmf_pd = pd.DataFrame(best_recon_nmf)
#best_recon_nmf_pd.to_csv("out/nmf-recon-mcar.csv", index=None)

#####################################################################
### KNN IMPUTE
###
### Impute with the sk-learn kNN method. Doing full hyperparameter
### search here too.  
#####################################################################
print(" ")
print("working on kNN")

best_knn_loss = np.inf
opt_neighbors = 0

for k in k_neighbors:
    knn_model = KNNImputer(n_neighbors=k)
    knn_recon = knn_model.fit_transform(train)
    knn_loss = util_functions.mse_func_np(knn_recon, val)

    if knn_loss < best_knn_loss:
        best_knn_loss = knn_loss 
        best_knn_recon = knn_recon
        opt_neighbors = k

intermediate_plots.real_v_imputed_basic(
            recon_mat=best_knn_recon,
            val_mat=val,
            PXD="SMTG-tester",
            row_factors=opt_neighbors,
            col_factors=opt_neighbors,
            model_type="kNN", 
            log_transform=False,
            tail="valid",
)   
#checkpoint
best_recon_knn_pd = pd.DataFrame(best_knn_recon)
#best_recon_knn_pd.to_csv("out/knn-recon-mcar.csv", index=None) 

#####################################################################
### MISSFOREST IMPUTE
###
### We also need to benchmark relative to missForest. Also doing a 
### hyperparam search here. This is gonna be painfully slow. 
#####################################################################
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
            PXD="SMTG-tester",
            row_factors=opt_trees,
            col_factors=opt_trees,
            model_type="missForest",
            log_transform=False,
            tail="valid",
)

# checkpoint
best_mf_recon_pd = pd.DataFrame(best_mf_recon)
best_mf_recon_pd.to_csv("out/missForest-recon-mnar.csv", index=None)

#####################################################################
### MIN IMPUTE
###
### Here we're taking the sample min. But note that this could also
### be done as the peptide min, or the absolute min of the matrix. 
#####################################################################
print(" ")
print("working on min impute")

col_min = np.nanmin(train, axis=0)
nan_idx = np.where(np.isnan(train))
smin_recon = train.copy()
# nan_idx[1] -> take index of column
smin_recon[nan_idx] = np.take(col_min, nan_idx[1]) 

# checkpoint
smin_recon_pd = pd.DataFrame(smin_recon)
#smin_recon_pd.to_csv("out/sample-min-recon-mcar.csv", index=None)

#####################################################################
### GAUSSIAN RANDOM SAMPLE IMPUTE
###
### Here we're gonna center the normal distribution about the mean 
### of the column mins. We'll use the standard deviation of the 
### column mins as well. NOTE this is not quite what Perseus does, 
### but its proven to be a lot nicer for our data, 
### ie. returns smaller MSE.
#####################################################################
print(" ")
print("working on random sample impute")

# get the column mins
col_min = np.nanmin(train, axis=0)

# get the mean and std of the entire training matrix
train_mean = np.nanmean(train)
train_sd = np.nanstd(train)

# get the indicies of the MVs 
nan_idx = np.where(np.isnan(train))
gsample_recon = train.copy()

# how many total MVs? 
n_mv = len(nan_idx[0])

# fill in the MVs with random draws 
gsample_recon[nan_idx] = rng.normal(
                            loc=np.mean(col_min), 
                            scale=np.std(col_min), 
                            size=n_mv
)
# don't want negative values
gsample_recon = np.abs(gsample_recon)

# checkpoint
gsample_recon_pd = pd.DataFrame(gsample_recon)
#gsample_recon_pd.to_csv("out/gsample-recon-mcar.csv", index=None)

#####################################################################
### PRECISION-RECALL CURVES
###
### Generate Precision-Recall curves for the optimized reconstruction
### for each imputation method. The task is reconstructing ground truth
### DE peptides between tumor and non-tumor matrices. Save to png. 
#####################################################################
print(" ")
print("generating precision-recall plots")

# for the NMF model 
cond1_recon_nmf = best_recon_nmf[:,0:cols_cutoff]
cond2_recon_nmf = best_recon_nmf[:,cols_cutoff:]

# for the kNN model 
cond1_recon_knn = best_knn_recon[:,0:cols_cutoff]
cond2_recon_knn = best_knn_recon[:,cols_cutoff:]

# for missForest
# cond1_recon_mf = best_mf_recon[:,0:cols_cutoff]
# cond2_recon_mf = best_mf_recon[:,cols_cutoff:]

# for min impute
cond1_recon_smin = smin_recon[:,0:cols_cutoff]
cond2_recon_smin = smin_recon[:,cols_cutoff:]

# for random sample impute
cond1_recon_gsample = gsample_recon[:,0:cols_cutoff]
cond2_recon_gsample = gsample_recon[:,cols_cutoff:]

# for no impute
no_impute_mat = train.copy()
noimp_pd = pd.DataFrame(no_impute_mat)
#noimp_pd.to_csv("out/noimpute-recon-mcar.csv", index=False)

cond1_noimp = no_impute_mat[:,0:cols_cutoff]
cond2_noimp = no_impute_mat[:,cols_cutoff:]

# again, not correcting the p-values here
pvals_nmf = \
    get_uncorrected_pvalues(cond1_recon_nmf, cond2_recon_nmf)
pvals_knn = \
    get_uncorrected_pvalues(cond1_recon_knn, cond2_recon_knn)
pvals_gsample = \
    get_uncorrected_pvalues(cond1_recon_gsample, cond2_recon_gsample)
pvals_smin = \
    get_uncorrected_pvalues(cond1_recon_smin, cond2_recon_smin)
pvals_noimp = \
    get_uncorrected_pvalues(cond1_noimp, cond2_noimp)

pvals_noimp[np.isnan(pvals_noimp)] = 1.0

# the ground truth labels. True means DE peptide, False means not
gt_labels = reject_null_gt.astype(float)

# get the DE probabilities, for each model 
prob_true_nmf = 1 - pvals_nmf
prob_true_knn = 1 - pvals_knn
#prob_true_mf = 1 - pvals_mf
prob_true_smin = 1 - pvals_smin
prob_true_gsample = 1 - pvals_gsample
prob_true_noimp = 1 - pvals_noimp

# call the sklearn PR plotting function 
pr_nmf, recall_nmf, t = precision_recall_curve(
                        y_true=gt_labels, probas_pred=prob_true_nmf)
pr_knn, recall_knn, t = precision_recall_curve(
                        y_true=gt_labels, probas_pred=prob_true_knn)
# pr_mf, recall_mf, t = precision_recall_curve(
#                         y_true=gt_labels, probas_pred=prob_true_mf)
pr_smin, recall_smin, t = precision_recall_curve(
                        y_true=gt_labels, probas_pred=prob_true_smin)
pr_gsample, recall_gsample, t = precision_recall_curve(
                        y_true=gt_labels, probas_pred=prob_true_gsample)
pr_noimp, recall_noimp, t = precision_recall_curve(
                        y_true=gt_labels, probas_pred=prob_true_noimp)

# get the AUCs
nmf_auc = np.around(auc(recall_nmf, pr_nmf), 2)
knn_auc = np.around(auc(recall_knn, pr_knn), 2)
#mf_auc = np.around(auc(recall_mf, pr_mf), 2)
smin_auc = np.around(auc(recall_smin, pr_smin), 2)
gsample_auc = np.around(auc(recall_gsample, pr_gsample), 2)
noimp_auc = np.around(auc(recall_noimp, pr_noimp), 2)

# plot
plt.figure()
plt.plot(
    recall_nmf, pr_nmf, label="NMF (" + str(nmf_auc) + ")")
plt.plot(
  recall_knn, pr_knn, label="kNN (" + str(knn_auc) + ")")
# plt.plot(
#     recall_mf, pr_mf, 
#     label="missForest (" + str(mf_auc) + ")")
plt.plot(
    recall_smin, pr_smin, 
    label="Sample min (" + str(smin_auc) + ")")
plt.plot(
    recall_gsample, pr_gsample, 
    label="Gaussian sample (" + str(gsample_auc) + ")")
plt.plot(
    recall_noimp, pr_noimp, 
    label="No impute (" + str(noimp_auc) + ")")

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("MCAR", pad=20, size=24)

plt.savefig(
    "out/DE-experiment-Petralia-MCAR.png",
    dpi=250, 
    bbox_inches="tight",
)
print(" ")
print("done!")
