#!/bin/bash
#SBATCH --job-name=ACAEtest
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:teslaK80:1        # Request for 1 GPU card
#SBATCH --mem-per-cpu=32G   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o joboutput_%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joberror_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hungyi_wu@g.harvard.edu  #Email to which notifications will be sent

export FP="jobresult_$SLURM_JOB_ID"
mkdir $FP
module load gcc/6.2.0 python/3.6.0 cuda/10.0
source /home/hw233/virtualenv/py3/bin/activate
python test_train_MNIST.py
mv "joboutput_$SLURM_JOB_ID.out" "$FP/job.out"
mv "joberror_$SLURM_JOB_ID.err" "$FP/job.err"
