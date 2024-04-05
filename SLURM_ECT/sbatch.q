#!/bin/bash
#SBATCH --array=1-116%10
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ECT/runs/run-%a.out
#SBATCH --error=logs/ECT/errors/errors-%a.err

module load gcc
module load python
python my_script.py $SLURM_ARRAY_TASK_ID 

