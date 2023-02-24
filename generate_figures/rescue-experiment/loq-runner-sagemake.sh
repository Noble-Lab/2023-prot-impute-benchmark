# LOQ-RUNNER-SAGEMAKE
# 1.19.23
# This script launches a distributed-memory parallelism 
# snakemake job on the GS cluster. 
#
# launch with:
#       >$ qsub loq-runner-sagemake.sh
# (from grid-head)
#
#!/bin/bash
#$ -S "/bin/bash"
#$ -cwd
#$ -N loq-runner-sage
#$ -o ./nobackup/
#$ -e ./nobackup/
#$ -l m_mem_free=4G
#$ -R y
#$ -l h_rt=54:0:0

SNAKE_FILE="Snakefile"

# configure cluster jobs
NUM_JOBS=5        # upper limit on concurrent jobs to submit
LATENCY_WAIT=180   # wait for as-yet-incomplete files
RESTART_ATTEMPTS=2 # number of times to restart any failed job

# helper function
function run_pipeline() {

        # run pipeline
        echo -e "\n~~~~~~~~\n\nLOQ-RUNNER PIPELINE\n\n~~~~~~~~~\n"

        # activate conda env
        conda activate ms-impute-trim

        # unlock working directory
        snakemake --unlock --cores 1

        # run pipeline
        snakemake --snakefile  $SNAKE_FILE \
                --use-conda \
                --cores all \
                --jobs $NUM_JOBS \
                --keep-going \
                --rerun-incomplete \
                --latency-wait $LATENCY_WAIT \
                --restart-times $RESTART_ATTEMPTS \
                --cluster "qsub -cwd -l mfree={resources.mfree} \
                -l h_rt={resources.h_rt} -pe serial {resources.cpus} \
                -R y -e ./nobackup/ -o ./nobackup/" 
}

# run pipeline
run_pipeline

echo "Job ${JOB_ID} ended" >&2

