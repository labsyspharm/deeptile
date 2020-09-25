#!/bin/bash
#SBATCH --job-name=prepvae
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p short            # Partition to submit to
#SBATCH --mem 16G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%a.txt    # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e error_%a.txt     # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hungyi_wu@g.harvard.edu  #Email to which notifications will be sent

module load gcc/6.2.0 python/3.7.4
source /home/hw233/virtualenv/py374/bin/activate
python prep_vae.py $SLURM_ARRAY_TASK_ID
