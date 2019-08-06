#!/bin/bash
#SBATCH --job-name=compare
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p short   # Partition to submit to
#SBATCH --mem-per-cpu=8G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o joboutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joberrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hungyi_wu@g.harvard.edu  #Email to which notifications will be sent

module load gcc/6.2.0
module load python/3.6.0
source /home/hw233/virtualenv/py3/bin/activate
python compare_speed.py
