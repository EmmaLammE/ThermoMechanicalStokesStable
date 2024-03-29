#!/bin/bash
#SBATCH --job-name=dull
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gpus 1
#SBATCH -o ./sbatch_output_logs/output_"%j".out
#SBATCH -e ./sbatch_output_logs/error_"%j".err
# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM module load python/3.6.1
julia -t 10 ./../Stokes2D.jl