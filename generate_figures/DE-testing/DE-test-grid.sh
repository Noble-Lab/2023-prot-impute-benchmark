# DE-TEST-GRID
# 1.13.23
#
# Launches the DE-test-simply.py script on grid. 
# 
# launch with:
#       >$ qsub DE-test-grid.sh
# (from grid-head)
#
#!/bin/bash
#$ -S "/bin/bash"
#$ -cwd
#$ -N DE-test-grid
#$ -o ./nobackup/
#$ -e ./nobackup/
#$ -l m_mem_free=32G
#$ -R y
#$ -l h_rt=48:0:0

conda activate ms-impute-trim

python DE-test-simple.py

echo "Job ${JOB_ID} ended" >&2
