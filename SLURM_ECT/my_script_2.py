import sys
import csv
import ect_slurm_tools.py
import numpy as np
import os
# Read in the $SLURM_ARRAY_TASK_ID (Note python indexes start from 0, the arrays were from 1
array_id=int(sys.argv[1])

# Read in the CSV
with open('CSV_combos.csv', newline='') as csvfile:
    params = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    
#Extract the values for A,B,C corresponding to the $SLURM_ARRAY_TASK_ID 
id1,id2=params[array_id]
ECT1=np.load(os.path.join('outputs',id1+'.npy'))
ECT2=np.load(os.path.join('outputs',id2+'.npy'))

path1=os.path.join('results',str(id1))
path2=os.path.join('results',str(id2))

if(not(os.path.exists(path1))):
    os.mkdir(path1)
    tmpfilename=os.path.join(path1,str(id1)+'.npy')
    tmp=np.array([0])
    np.save(tmpfilename,tmp)
if(not(os.path.exists(path2))):
    os.mkdir(path2)
    tmpfilename=os.path.join(path2,str(id2)+'.npy')
    tmp=np.array([0])
    np.save(tmpfilename,tmp)

d=ect_slurm_tools.compute_ECT_distance_p(ECT1, ECT2)
crosspath1=os.path.join(path2,str(id1)+'.npy')
crosspath2=os.path.join(path1,str(id2)+'.npy')
np.save(crosspath1,d)
np.save(crosspath2,d)
