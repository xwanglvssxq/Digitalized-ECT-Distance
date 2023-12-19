from numpy import genfromtxt
from shape_reader import ShapeReader
import numpy as np
import MC_ECT_distance as mc

def compute_ECT(infile,outfile):
    s1=ShapeReader.shape_from_file(infile)
    ECT=compute_ECT_MC(shape = s1)
    np.save(outfile, ECT)

def compute_ECT_MC(shape,p=100):
    directions = genfromtxt('directions326.csv', delimiter=',')
    n=directions.shape[0]
    heights=np.linspace(-1,1,p)
    
    V = shape.V
    T = shape.T
    sorted_T = []
    for t in T:
        t = np.sort(t)
        sorted_T.append(t)
    F_idx = sorted_T
    E, F = mc.get_EF(V, F_idx)
    
    ECT=np.zeros([n,p])
    for i in range(n):
        direction=directions[i,:]
        for j in range(p):
            height=heights[j]
            ECT[i,j]=mc.EC(V, E, F, direction, heights[j]*direction)
    return(ECT)
