#!/bin/bash
#SBATCH --job-name=n3a50w20
#SBATCH --time=5-00:00:00
#SBATCH --mem=40GB
#SBATCH -p serc
#SBATCH -c 20
#SBATCH --ntasks=1
#SBATCH --gpus 2
#SBATCH -o ./../sbatch_output_logs/out_n3_amp50_num20.%j.out
#SBATCH -e ./../sbatch_output_logs/err_n3_amp50_num20.%j.err

# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM module load python/3.6.1
mkdir sbatch_output_logs
julia -t 20 ./../Stokes2D_sin_n3_amp50_num20.jl