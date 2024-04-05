#!/bin/bash
#SBATCH --array=1-6786%20
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ECT/runs2/run-%a.out
#SBATCH --error=logs/ECT/errors2/errors-%a.err

module load gcc
module load python
python my_script_2.py $SLURM_ARRAY_TASK_ID 

