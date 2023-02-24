#####################################################################
### CALCULATE-LOQ-RUNNER-TEST.SH
### 1.19.23
###
### Run the `calculate-loq.py` script for each of the test output 
### files from `impute-w-various-models.py`. Rename output files 
### something more palletable. Would like to do this from within 
### the python script but doing so would require re-engineering
### `calculate-loq.py`, which I didn't write. 
###
### Run with $ ./calculate-loq-runner.sh 
###       or $ source calculate-loq-runner.sh
###
### Note that this will take a really long time to run, as the 
### `calculate-loq.py` script is not parallelized. This is
### set up to run `calculate-loq.py` on the small (300 peptide) 
### "tester" recon matrices.
#####################################################################

# the original (unimputed) matrix
python calculate-loq.py data/cc-orig-MCAR-tester.csv \
	 filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-orig-MCAR-test.csv
echo "done with orig matrix. "
echo " "

# the NMF imputed matrix
python calculate-loq.py data/cc-NMF-recon-MCAR-tester.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-NMF-recon-MCAR-test.csv
echo "done with NMF reconstructed matrix. "
echo " "

# the kNN imputed matrix 
python calculate-loq.py data/cc-KNN-recon-MCAR-tester.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-KNN-recon-MCAR-test.csv
echo "done with kNN reconstructed matrix. "
echo " "

# the missForest imputed matrix 
python calculate-loq.py data/cc-mf-recon-MCAR-tester.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-mf-recon-MCAR-test.csv
echo "done with missForest reconstructed matrix. "
echo " "

# the sample min imputed matrix 
python calculate-loq.py data/cc-min-recon-MCAR-tester.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-min-recon-MCAR-test.csv
echo "done with sample min imputed matrix. "
echo " "

# the Gaussian random sample imputed matrix 
python calculate-loq.py data/cc-std-recon-MCAR-tester.csv \
	filename2samplegroup_map.csv --plot n
mv figuresofmerit.csv out/fom-std-recon-MCAR-test.csv
echo "done with Gaussian random sample reconstructed matrix. "
echo " "
