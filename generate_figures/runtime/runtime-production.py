"""
RUNTIME-PRODUCTION.PY
1.26.23

A scaled up version of my runtime-sandbox jupyer notebook. Want to 
run our five imputation methods on a number (20?) of peptide quants 
datasets and record runtime for each method. 

I think it probably makes the most sense to represent dataset size 
in terms of number of observations in the training set. This prevents
linear vs log conversion issues. 

I also think it makes sense to just do a standard MCAR partition, with
the same validation set fraction, for all of these datasets. This is
probably the most straightforward thing to do. 
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
####  CONFIGS  ######################################################
#####################################################################
# MCAR partitioning params
val_frac = 0.3
test_frac = 0.0
min_present = 1

# MNAR partition params
q_anchor=0.3   
t_std=0.35
brnl_prob=0.4

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

#####################################################################
####  SOME FUNCTIONS  ###############################################
#####################################################################
def nmf_impute(train_mat, val_mat):
	"""
	Runs NMF on a provided training dataset, and then imputes. 
	Records the elapsed time.

	Parameters
	----------
	train_mat, val_mat : np.ndarray, 
		The training and validation sets.

	Returns
	----------
	nmf_sec_elapsed : float, 
		The number of seconds it took to run NMF on the
		provided training dataset
	"""
	# get the optimal number of training batches for NMF
	if len(~np.isnan(train_mat)) > 100:
	    n_batches = int(np.floor(len(~np.isnan(train_mat)) / batch_size))
	    # setting the minimum n_batches to 100
	    n_batches = max(n_batches, 100) 
	else: 
	    n_batches = 1

	# get the start time, measured since the Unix epoch
	nmf_start_sec = time.time()

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
	nmf_recon = nmf_model.fit_transform(train_mat, val_mat)

	# get the elapsed time
	nmf_end_sec = time.time()
	nmf_sec_elapsed = nmf_end_sec - nmf_start_sec

	return nmf_sec_elapsed

def knn_impute(train_mat):
	"""
	Runs kNN impute on a provided training dataset.
	Records the elapsed time.

	Parameters
	----------
	train_mat : np.ndarray, 
		The training set.

	Returns
	----------
	knn_sec_elapsed : float, 
		The number of seconds it took to run kNN impute
		on the provided training dataset.
	"""
	# get the start time, measured since the Unix epoch
	knn_start_sec = time.time()

	knn_model = KNNImputer(n_neighbors=k_neighbors)
	knn_recon = knn_model.fit_transform(train_mat)

	# get the elapsed time
	knn_end_sec = time.time()
	knn_sec_elapsed = knn_end_sec - knn_start_sec

	return knn_sec_elapsed

def sample_min_impute(train_mat):
	"""
	Runs sample min impute on a provided training dataset.
	Records the elapsed time.

	Parameters
	----------
	train_mat : np.ndarray, 
		The training set.

	Returns
	----------
	smin_sec_elapsed : float, 
		The number of seconds it took to run sample min
		impute on the provided training dataset.
	"""
	# get the start time, measured since the Unix epoch
	min_start_sec = time.time()

	col_min = np.nanmin(train_mat, axis=0)
	nan_idx = np.where(np.isnan(train_mat))
	min_recon = train_mat.copy()
	# nan_idx[1] -> take index of column
	min_recon[nan_idx] = np.take(col_min, nan_idx[1])

	# get the elapsed time
	min_end_sec = time.time()
	smin_sec_elapsed = min_end_sec - min_start_sec

	return smin_sec_elapsed

def gaussian_sample_impute(train_mat):
	"""
	Runs Gaussian random sample impute on a provided 
	training dataset. Records the elapsed time.

	Parameters
	----------
	train_mat : np.ndarray, 
		The training set.

	Returns
	----------
	gsample_sec_elapsed : float, 
		The number of seconds it took to run Gaussian
		random sample impute on the provided training dataset.
	"""
	# get the start time, measured since the Unix epoch
	std_start_sec = time.time()

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

	# get the elapsed time
	std_end_sec = time.time()
	gsample_sec_elapsed = std_end_sec - std_start_sec

	return gsample_sec_elapsed

def missForest_impute(train_mat):
	"""
	Runs missForest impute on a provided 
	training dataset. Records the elapsed time.

	Parameters
	----------
	train_mat : np.ndarray, 
		The training set.

	Returns
	----------
	mf_sec_elapsed : float, 
		The number of seconds it took to run missForest
		impute on the provided training dataset.
	"""
	# get the start time, measured since the Unix epoch
	mf_start_sec = time.time()

	set_seed = r('set.seed')
	set_seed(r_seed)

	base = importr("base")
	doParallel = importr("doParallel")
	rngtools = importr("rngtools")
	missForest = importr("missForest")

	# activate automatic conversion of NumPy arrays
	numpy2ri.activate()

	# set up parallelization
	    # not totally sure how to set this. Maybe allow it to 
	    #   occupy all of the cores in my machine? 
	doParallel.registerDoParallel(cores=n_cores_mf)

	# run missForest
	mf_recon, err = missForest.missForest(
	                            train_mat, 
	                            maxiter=max_iters_mf,
	                            ntree=n_trees,
	                            parallelize="no",
	                            #parallelize="forests", 
	                            verbose=True,
	)
	mf_recon = np.array(mf_recon)

	# get the elapsed time
	mf_end_sec = time.time()
	mf_sec_elapsed = mf_end_sec - mf_start_sec

	return mf_sec_elapsed

#####################################################################
####  THE MAIN LOOP
#### 
#### For each peptide quants dataset, read in, pre-process, partition
#### (standard MCAR), then impute with various methods and record 
#### the run times.
#####################################################################
full_path = \
	"/net/noble/vol2/home/lincolnh/code/"\
	"2021_ljharris_ms-impute/data/peptides-data/"
#pxds = ["PXD013792", "PXD014156", "PXD011961"]
pxds = ["PXD013792", "PXD014156", "PXD011961",
		"PXD001010", "PXD014525", "PXD015939",
		"PXD006109", "PXD006348",
		"PXD019254", "PXD010612", "PXD010709",
		"Pino2020", "Thomas2020", "PXD016079",
]

# init the runtime dataframe
runtime_df = pd.DataFrame(
    columns=[
        "dataset", 
        "n observations", 
        "NMF sec", 
        "kNN sec", 
        "min sec", 
        "std sec", 
        "mf sec"]
)

for pxd in pxds:
	print(" ")
	print("working on: ", pxd)

	# pre-process the peptide quants df
	quants_raw = pd.read_csv(full_path + pxd + "_peptides.csv")

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

	# sanity checking -- make sure the MV fractions in each 
		# partition are roughly the same
	orig_mv_frac = np.count_nonzero(np.isnan(quants)) / quants.size
	train_mv_frac = np.count_nonzero(np.isnan(train)) / train.size
	val_mv_frac = np.count_nonzero(np.isnan(val)) / val.size

	print("mv frac original: ", np.around(orig_mv_frac, decimals=3))
	print("mv frac train: ", np.around(train_mv_frac, decimals=3))
	print("mv frac validation: ", np.around(val_mv_frac, decimals=3))

	# record the number of observations in the training set
	n_obs = np.count_nonzero(~np.isnan(train))

	# impute with various methods, get runtimes
	nmf_rtime = nmf_impute(train, val)
	knn_rtime = knn_impute(train)
	smin_rtime = sample_min_impute(train)
	gsample_rtime = gaussian_sample_impute(train)
	mf_rtime = missForest_impute(train)

	# append current results to the runtime dataframe
	toadd = {
	    "dataset" : pxd, 
	    "n observations" : n_obs,
	    "NMF sec" : nmf_rtime,
	    "kNN sec" : knn_rtime,
	    "min sec" : smin_rtime,
	    "std sec" : gsample_rtime,
	    "mf sec" : mf_rtime,
	}

	runtime_df = runtime_df.append(toadd, ignore_index=True)

print(" ")
print("done!")

# save to csv
runtime_df.to_csv("runtimes-mf.csv", index=False)

