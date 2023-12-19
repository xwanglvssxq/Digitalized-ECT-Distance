import sys
import csv
import os
import ect_slurm_tools
import numpy as np
# Read in the $SLURM_ARRAY_TASK_ID (Note python indexes start from 0, the arrays were from 1
array_id=int(sys.argv[1])

# Read in the CSV
with open('MC.csv', newline='') as csvfile:
    params = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    
#Extract the values for A,B,C corresponding to the $SLURM_ARRAY_TASK_ID 
meshid,mesh_name=params[array_id]
infile=os.path.join('teeth',mesh_name)
outfile=os.path.join('outputs',meshid+'.npy')
ECT=ect_slurm_tools.compute_ECT(infile, outfile)
