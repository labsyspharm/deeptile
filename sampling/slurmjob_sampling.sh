#!/bin/bash
#SBATCH --job-name=CVAEsamp
#SBATCH -n 1               # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu            # Partition to submit to
#SBATCH --gres=gpu:1 # Request for 1 GPU card
#SBATCH --mem-per-cpu=32G    # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o joboutput_%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joberrors_%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hungyi_wu@g.harvard.edu  #Email to which notifications will be sent

module load gcc/6.2.0 python/3.6.0 cuda/10.0
source /home/hw233/virtualenv/py3/bin/activate
python run_tile_sampling.py
