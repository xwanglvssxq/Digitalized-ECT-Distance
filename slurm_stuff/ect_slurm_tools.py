from numpy import genfromtxt
import numpy as np
import ????

def compute_ECT(infile,outfile):
    s1=Shapereader(infile)
    ECT=compute_ECT_MC(s1)
    np.save(outfile,ECT)

def compute_ECT_MC(shape,p=100)
    directions = genfromtxt('directions326.csv', delimiter=',')
    n=directions.shape[0]
    heights=np.linspace(-1,1,p)
    ECT=np.zeros([n,p])
    for i in range(n):
        direction=directions[i,:]
        for j in range(p):
            height=heights[j]
            ECT[i,j]=EC(heights[j]*n # TODO: Fix
    return(ECT)
