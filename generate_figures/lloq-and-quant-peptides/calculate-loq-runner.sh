#####################################################################
### CALCULATE-LOQ-RUNNER.SH
### 11.09.22
###
### Run the `calculate-loq.py` script for teach of the full output 
### files from `impute-w-various-models.py`. Rename output files 
### something more palletable. Would like to do this from within 
### the python script but doing so would require re-engineering
### `calculate-loq.py`, which I didn't write. 
###
### Run with $ ./calculate-loq-runner.sh 
###       or $ source calculate-loq-runner.sh
###
### Note that this will take a really long time to run (several days)
### as the `calculate-loq.py` script is not parallelized.
#####################################################################

# the original (unimputed) matrix
python calculate-loq.py data/cal-curves-orig-MCAR.csv \
	 filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-orig-MCAR.csv
echo "done with original matrix. "
echo " "

# the NMF imputed matrix
python calculate-loq.py data/cal-curves-NMF-recon-MCAR.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-NMF-recon-MCAR.csv
echo "done with NMF reconstructed matrix. "
echo " "

# the kNN imputed matrix 
python calculate-loq.py data/cal-curves-KNN-recon-MCAR.csv\
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-KNN-recon-MCAR.csv
echo "done with kNN reconstructed matrix. "
echo " "

# the missForest imputed matrix 
python calculate-loq.py data/cal-curves-mf-recon-MCAR.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-mf-recon-MCAR.csv
echo "done with missForest reconstructed matrix. "
echo " "

# the sample min imputed matrix 
python calculate-loq.py data/cal-curves-min-recon-MCAR.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-min-recon-MCAR.csv
echo "done with sample min reconstructed matrix. "
echo " "

# the Gaussian random sample imputed matrix 
python calculate-loq.py data/cal-curves-std-recon-MCAR.csv\
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-std-recon-MCAR.csv
echo "done with Gaussian random sample reconstructed matrix. "
echo " "
