from shape import Shape
from numpy import array, ndarray, concatenate, empty, full
from os.path import isfile, splitext
import numpy as np
class ShapeWriter(object):
    @staticmethod
    def subshape(shape, direction, threshold):
        '''
        Method for restricting a shape to specicifc half-space
        Note: triangles renamed nicely!
        '''
        heights=np.matmul(direction,shape.V.T)
        inds=heights<threshold
        inds1=np.where(inds)[0]
        cutoff=np.sum(inds)
        sinds=np.argsort(heights)
        tmpV=shape.V[sinds[0:cutoff],:]
        T=shape.T.astype(int)
        tmpset=set(inds1)
        lista=[len(set.intersection(set(triangle),tmpset))==3 for triangle in T]
        kikka=np.where(lista)[0]
        triangles=np.zeros([kikka.shape[0],3])
        for i in range(len(kikka)):
            triangle=T[kikka[i],:]
            t0=np.where(sinds==triangle[0])[0][0]
            t1=np.where(sinds==triangle[1])[0][0]
            t2=np.where(sinds==triangle[2])[0][0]
            triangles[i,:]=[t0,t1,t2]
        return(tmpV,triangles)
        
    @staticmethod
    def write_off_file(v,t,filename):
        rimpsu=str(v.shape[0]) +str(' ') +str(t.shape[0]) + str(' 0 \n')
        with open(filename, 'a') as the_file:
            the_file.write('OFF \n')
            the_file.write(rimpsu)
            for vertex in v:
                the_file.write(str(vertex[0]) + str(' ') + str(vertex[1]) + str(' ') + str(vertex[2])+ str('\n'))
            for triangle in t:
                triangle=triangle.astype(int)
                the_file.write(str('3 ') + str(triangle[0]) + str(' ') + str(triangle[1]) + str(' ') + str(triangle[2])+ str('\n'))
        return True
