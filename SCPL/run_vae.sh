#!/bin/bash
#SBATCH --job-name=runVAE
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-01:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:teslaK80:1
#SBATCH --mem 8G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output.txt       # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e error.txt        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hungyi_wu@g.harvard.edu  #Email to which notifications will be sent

module load gcc/6.2.0 python/3.7.4 cuda/10.0
source /home/hw233/virtualenv/gpu_py374_cuda10/bin/activate
python run_vae.py
