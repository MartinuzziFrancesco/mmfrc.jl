#!/bin/bash

# Set max time to your time estimation for your program be as precise as possible
# for optimal cluster utilization


#SBATCH --time=24:00:00
#SBATCH --partition=medium

#medium 2-00:00:00
#short 02:00:00
#long 14-00:00:00

#
# There are a lot of ways to specify hardware ressources we recommend this one
#


#SBATCH --ntasks=28  # number of processor cores (i.e. tasks)

#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=400M

# If you so choose Slurm will notify you when certain events happen for your job
# for a full list of possible options look furhter down
#SBATCH --mail-type END

# Feel free to give your job a good name to better identify it later
# the same name expansions as for the ourput and error path apply here as well
# see below for additional information

#SBATCH --array=1-1

#SBATCH --job-name=parc


#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

#
# Prepare your environment
#

# causes jobs to fail if one command fails - makes failed jobs easier to find with tools like sacct

set -e
#Set time for log folder and create log folder
time=$(date +"_%Y-%m-%d_%H-%M-%S")

# Set variables you need

srun julia --project=.  -t $SLURM_CPUS_PER_TASK parc_runs.jl

# copy results to an accessable location
# only copy things you really need

#Move log files

# Clean up after yourself

# exit gracefully
exit 0
