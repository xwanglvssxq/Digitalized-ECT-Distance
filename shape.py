import numpy as np
import math
#import ect_utils
from itertools import combinations
from scipy.spatial import Delaunay
class Shape:
    # params: self, Vertices, Edges, Triangles
    def __init__(self, vertices, triangles,name=None):
        self.V=vertices
        self.T=triangles
        self.links={}
        self.polygon_angles={}
        self.polygon_triangles={}
        self.polygon_midpoints={}
        self.vertex_faces={}
        self.vertex_edges={}
        self.detecting_polygons={}
        self.polygon_gains={}
        
    def center_n_scale(self):
        '''
        Not needed for now
        '''
        pass
       
    def compute_links(self):
        '''
        Links for each vertex, 1st step of the algorithm
        '''
        # Go through the triangles and get all neighbors for all vertices
        for i in range(len(self.V)):
            a=np.where(self.T[:,0]==i)
            b=np.where(self.T[:,1]==i)
            c=np.where(self.T[:,2]==i)
            A=np.delete(self.T[a,:],0,axis=2)
            B=np.delete(self.T[b,:],1,axis=2)
            C=np.delete(self.T[c,:],2,axis=2)
            neighbor_array=np.concatenate((A,B,C),1)
            self.links[i]=np.unique(neighbor_array)
            
    def compute_triple_points(self):
        '''
        Second step of the algorithm.
        Finds the triple points in the link of each vertex  
        '''
        for key in self.links:
            # Step 1: Compute the local stratification:
            p0=self.V[key,:]
            # Initialize the putative strata: the angles
            n=len(self.links[key])
            trick=np.array(list(combinations(self.links[key],2)))
            coords=np.empty((trick.shape[0],3))
            # For each vertex p0=(x0,y0), go through the neighbors p=(x,y)
            for i in range(trick.shape[0]):
                neighbor1=trick[i,0].astype(int)
                neighbor2=trick[i,1].astype(int)
                p1=self.V[neighbor1,:]
                p2=self.V[neighbor2,:]
                v1=p1-p0
                v2=p2-p0
                res=np.cross(v1,v2)
                press=res/(sum(res**2))
                coords[i,:]=press
            self.polygon_angles[key]=np.concatenate((coords,-coords))

    def compute_delaunay_triangles(self):
        '''
        The third step. Finds the Delaunay triangulation of the 
        TODO: Fix the implementation to support vertices with less than 2 neighbors. If the link only contains 2 neighbors we can't get the delaunay triangulation
        Assuming X is triangulated (so no extra edges), that case is exactly the case where x_0 is the lowest of them all.
        So this is a tomatowedge shape
        '''
        for key in self.polygon_angles:
            test=self.polygon_angles[key]
            t1=np.array([0,0,0])
            t1=t1.reshape(1,-1)
            t2=np.concatenate([test,t1])
            D2=Delaunay(t2)
            telek=D2.simplices
            D1=telek[:,1:4]
            D1=D1-1 # Stupid index thingy
            midpoints=np.zeros(D1.shape)
            for i in range(D1.shape[0]):
                tmp=np.mean(test[D1[i]],1)
                midpoints[i]=tmp/(sum(tmp**2))**(0.5)
            self.polygon_triangles[key]=D1
            self.polygon_midpoints[key]=midpoints

    def compute_gains(self):
        '''
        Fourth step: evaluate the ECT gains in each triangle of the Delaunay triangulation.
        '''
        for key in self.polygon_midpoints:
            directions=self.polygon_midpoints[key]
            gains=np.zeros([directions.shape[0],1])
            for i in range(len(directions)):
                direction=directions[i]
                gains[i]=self.evaluate_local_ECT(direction, key)           
            self.polygon_gains[key]=gains
        return(True)

    def evaluate_local_ECT(self,direction, vertex):
        '''
        Helper function for finding the spherical triangles that matter.
        This is the combinatorial algorithm! TODO: Find if the geometric one is faster
        Also, currently this uses all of the vertices. Would it be sufficient to just use the link?
        '''
        dir_vector=direction
        heights=np.matmul(dir_vector,self.V.T)
        order=np.argsort(heights)
        cutpoint=np.where(order==vertex)[0][0]
        subset1=order[:cutpoint]
        subset2=order[:(cutpoint+1)]
        faces=self.vertex_faces[vertex]
        edges=self.vertex_edges[vertex]
        # Change in ECT: 1 (p0) -edges + triangles
        chi=1+sum([set(subset2).issuperset(row) for row in faces.tolist()]) \
        -sum([set(subset1).issuperset(row) for row in faces.tolist()]) \
        -sum([set(subset2).issuperset(row) for row in edges.tolist()]) \
        +sum([set(subset1).issuperset(row) for row in edges.tolist()])
        return(chi)
        
    def compute_transform(self):
        pass

    def prepare(self):
        '''
        This helper function speedens up the combinatorial ECT evaluation algorithm
        '''
        for i in range(len(self.V)):
            a=self.T[np.where(self.T[:,0]==i),:]
            b=self.T[np.where(self.T[:,1]==i),:]
            c=self.T[np.where(self.T[:,2]==i),:]
            faces=np.concatenate((a,b,c),1)[0]
        # Initialize edges: At most 3 * faces
            edges=np.zeros((3*faces.shape[0],2))
            for j in range(faces.shape[0]):
                v0=min(faces[j,(0,1)])
                v1=max(faces[j,(0,1)])
                v2=min(faces[j,(0,2)])
                v3=max(faces[j,(0,2)])
                v4=min(faces[j,(1,2)])
                v5=max(faces[j,(1,2)])
                edges[3*j,:]=[v0,v1]
                edges[3*j+1,:]=[v2,v3]
                edges[3*j+2,:]=[v4,v5]
            edges=np.unique(edges,axis=0)
            verts=np.unique(edges,0)
            self.vertex_faces[i]=faces
            self.vertex_edges[i]=edges
