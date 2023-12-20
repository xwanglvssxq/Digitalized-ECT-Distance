#!/bin/bash
#SBATCH --array=1-6671%15
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/MC/runs2/run-%a.out
#SBATCH --error=logs/MC/errors2/errors-%a.err

#load conda and activate the desired environment
# (the directory here is just an example, won't work on EPFL systems
# (and these 2 lines may not even be needed)
module load gcc
module load python
#conda activate /data/mukherjeelab/envs/conda
# fire off python
python my_script_2.py $SLURM_ARRAY_TASK_ID 

