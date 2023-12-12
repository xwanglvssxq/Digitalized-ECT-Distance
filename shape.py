import numpy as np
import math
#import ect_utils
from itertools import combinations
from scipy.spatial import Delaunay
import itertools
import ect_tools
from scipy.sparse.csgraph import minimum_spanning_tree
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
        # Nov 14 add
        self.clean_polygon_triangles={}
        self.clean_polygon_gains={}
        # November 9 2023: Add helper dictionary for edges
        # The format of this will be as follows:
        # For vertex i,  dictionary listing all the triple points.
        # Note: these are in the same order as the self.polygon_angles
        self.sphere_point_names={}
        
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
            pointnames=np.empty((trick.shape[0],2)) # Contains the names for the point
            # For each vertex p0=(x0,y0), go through the neighbors p=(x,y)
            for i in range(trick.shape[0]):
                neighbor1=trick[i,0].astype(int)
                neighbor2=trick[i,1].astype(int)
                p1=self.V[neighbor1,:]
                p2=self.V[neighbor2,:]
                v1=p1-p0
                v2=p2-p0
                res=np.cross(v1,v2)
                press=res/(sum(res**2))**(1/2)
                coords[i,:]=press
                pointnames[i,:]=[neighbor1,neighbor2]
            self.polygon_angles[key]=np.concatenate((coords,-coords))
            self.sphere_point_names[key]=np.concatenate((pointnames,pointnames))

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
            #print(D2.simplices)
            telek=D2.simplices
            print(telek)
            # post processing here
            D1=np.zeros(telek[:,1:4].shape,dtype=int)
            #print(D2.simplices)
            #import pdb; pdb.set_trace()
            for i in range(telek.shape[0]):
                row=telek[i,:]
                index=np.where(row==np.max(row)) #the old way
                newrow=np.delete(row,index) #the old way
                #newrow=np.setdiff1d(row,np.max(row))
                #import pdb; pdb.set_trace()
                #if index[0][0]%2==1:
                #    newrow=[newrow[1],newrow[0],newrow[2]]
                D1[i,:]=newrow
            #print(D1)#2.simplices)
            #D1=D1-1 # Stupid index thingy
            midpoints=np.zeros(D1.shape)
            for i in range(D1.shape[0]):
                tmp=np.mean(test[D1[i]],0)
                midpoints[i]=tmp/(sum(tmp**2))**(0.5)
            #D1=np.unique(D1,axis=0)
            self.polygon_triangles[key]=D1
            #print(self.polygon_triangles[key])
            self.polygon_midpoints[key]=midpoints

    def compute_TP_DT_vol2(self): # New Nov 23
        for key in self.links:
            edges,verts=self.compute_triangluation(key)
            self.polygon_angles[key]=verts
            tmp,_=self.construct_fan_triangulation(edges)
            tmp_T=self.construct_triangles_from_fans(tmp)
            midpoints=np.zeros(tmp_T.shape)
            for i in range(tmp_T.shape[0]):
                #import pdb; pdb.set_trace()
                tmp=np.mean(verts[tmp_T[i].astype(int)],0)
                midpoints[i]=tmp/(sum(tmp**2))**(0.5)
            self.polygon_triangles[key]=tmp_T.astype(int)
            self.polygon_midpoints[key]=midpoints

    def compute_TP_DT_vol3(self): # New Nov 23
        for key in self.links:
            edges,verts=self.compute_triangluation(key)
            self.polygon_angles[key]=verts
            # Ugly fix but it will work: TODO: Make this right, I bet this is behind the weird behavior in tetrahedron as well
            triangles=self.vertex_faces[key]
            if triangles.shape[0]==1:
                tmp_T=np.stack([[0,2,3],[1,2,3]])
            else:
                tmp,I=self.construct_fan_triangulation(edges)
                tmp_T=self.construct_triangulation(I,edges)
            midpoints=np.zeros(tmp_T.shape)
            for i in range(tmp_T.shape[0]):
                #import pdb; pdb.set_trace()
                tmp=np.mean(verts[tmp_T[i].astype(int)],0)
                midpoints[i]=tmp/(sum(tmp**2))**(0.5)
            self.polygon_triangles[key]=tmp_T.astype(int)
            self.polygon_midpoints[key]=midpoints

    def construct_fan_triangulation(self,edges):  # New Nov 23
        '''
        Given edges describing polygons (collection of cycles), returns a fan triangulation
        so that each vertex in a cycle is connected to the root of that cycle
        '''
        tmpmatrix=np.zeros([len(edges),len(edges)])
        for i in range(len(edges)): # Construct the adjacency matrix
            tmpmatrix[i,:]=[len(set.intersection(set(edge),set(edges[i])))==1 for edge in edges]
        sum_matrix=np.zeros([len(edges),len(edges)])
        for i in range(len(edges)): # Find H_0 with Perron-Frobenius Theorem
            sum_matrix=sum_matrix+np.linalg.matrix_power(tmpmatrix,(i+1))
        Imatrix=(sum_matrix>0).astype(int)
        dist=np.unique(Imatrix,axis=0)
        for i in range(dist.shape[0]):
            #print('paska')
            tmp=Imatrix==dist[i] # Here: This needs to be fixed
            ind=np.mean(tmp,1).astype(int)
            inds=np.where(ind==1)
            ind1=inds[0][0]
            #print(inds)
            #print(len(inds[0]))
            for j in range(1,len(inds[0])):
                #print(tmpmatrix)
                #print(ind1)
                #print(inds)
                tmpmatrix[ind1,inds[0][j]]=1
                tmpmatrix[inds[0][j],ind1]=1
        return(tmpmatrix,Imatrix)

    def orient_polygons(self):
        for key in self.polygon_triangles:
            triangles=self.polygon_triangles[key]
            n=triangles.shape[0]
            dm=np.zeros([n,n])
            for i in range(n):
                triangle=triangles[i,:]
                dm[i,:]=np.array([len(set.intersection(set(triangle),set(triangles[i,:])))==2 \
                                  for i in range(triangles.shape[0])]).astype(int)
            mst=minimum_spanning_tree(dm)
            temppi=ect_tools.transitive_closure(mst)
            self.polygon_triangles[key]=ect_tools.orient_mst(temppi,mst,triangles)

    def unique_list(a_list: list) -> list:
        """
        A helper file for topological regularizer
        Given a list a_list,
        returns the unique elements in that.
        Used for computing the connections
        when building the linear surrogate function
        Args:
             a_list:
                 a list
        Returns
            uniquelist:
                a list of unique entries of the list a_list
        """
        uniquelist = []
        used = set()
        for item in a_list:
            tmp = repr(item)
            if tmp not in used:
                used.add(tmp)
                uniquelist.append(item)
        return uniquelist

    def construct_triangulation(self,I,e):
        dist=np.unique(I,axis=0)
        triangles=[]
        for i in range(dist.shape[0]):
            row=dist[i]
            #print(i)
            #print(row)
            inds=np.where(np.mean(I==row,1)==1)[0]
            tmpedges=[e[ind] for ind in inds]
            trick=np.unique(tmpedges)
            main=trick[0]
            inds2=np.where([len(set.intersection(set([main]),set(edge)))==0 for edge in tmpedges])[0]
            for j in range(len(inds2)):
                triangles.append([main,*tmpedges[inds2[j]]])
        return(np.array(triangles))

    def construct_triangles_from_fans(self,matrix):  # New Nov 23
        '''
        Given a fan triangulation matrix, returns the triangles
        '''
        inds=[sum(row)==3 for row in matrix]
        triangles=np.zeros([sum(inds),3])
        for i in range(triangles.shape[0]):
            row=matrix[np.where(inds)[0][i],:]
            #print(row)
            #print(np.where(row))
            triangles[i,:]=np.where(row)[0]
        return(triangles)
    
    def compute_face_normal(self,triangle):  # New Nov 23
        x1=triangle[0,:]
        x2=triangle[1,:]
        x3=triangle[2,:]
        tmp=np.cross(x2-x1,x3-x1)
        return(tmp/sum(tmp**2)**(0.5))
        
    def dualize_link(self,key):  # New Nov 23
        '''
        Returns the dual graph
        '''
        edges=[]
        triangles=self.vertex_faces[key]
        for i in range(triangles.shape[0]):
            t1=triangles[i,:]
            for j in range(i):
                t2=triangles[j,:]
                if(len(set.intersection(set(t1),set(t2)))==2):
                    edges.append([i,j])
        return(np.array(edges))
        
    def compute_triangluation(self,key):
        triangles=self.vertex_faces[key]
        if triangles.shape[0]==1:
            index=np.where(triangles[0]==key)[0][0]
            triangle=self.V[triangles.astype(int),:][0]
            stack,E=ect_tools.triangulate_a_triangle(triangle,index)
            return(E,stack)
        tmp_verts=np.zeros(triangles.shape)
        for i in range(triangles.shape[0]):
            triangle=triangles[i,:]
            n1=self.compute_face_normal(self.V[triangle.astype(int),:])
            #print(n1)
            tmp_verts[i,:]=n1
        verts=np.concatenate([tmp_verts,-tmp_verts])
        tmp_edges=self.dualize_link(key)
        #print(verts)
        edges=[]
        for i in range(tmp_edges.shape[0]):
            ind0=tmp_edges[i,0]
            ind1=tmp_edges[i,1]
            ind2=ind0+triangles.shape[0]
            ind3=ind1+triangles.shape[0]
            #print(verts[ind0,:])
            trick0=2*(np.dot(verts[ind0,:],self.V[key,:])>0)-1
            trick1=2*(np.dot(verts[ind1,:],self.V[key,:])>0)-1
            trick2=2*(np.dot(verts[ind2,:],self.V[key,:])>0)-1
            trick3=2*(np.dot(verts[ind3,:],self.V[key,:])>0)-1
            if(trick0*trick1>0):
                edges.append([ind0,ind1])
            if(trick0*trick3>0):
                edges.append([ind0,ind3])
            if(trick1*trick2>0):
                edges.append([ind2,ind1])
            if(trick2*trick3>0):
                edges.append([ind2,ind3])
            #trick0=np.dot(verts[ind0,:],self.V[key,:])>0
            #if(np.dot(verts[ind0,:],verts[ind1,:])>0):
            #    edges.append([ind0,ind1])
            #if(np.dot(verts[ind0,:],verts[ind3,:])>0): 
            #    edges.append([ind0,ind3])
            #if(np.dot(verts[ind2,:],verts[ind1,:])>0):
            #    edges.append([ind2,ind1])        
            #if(np.dot(verts[ind2,:],verts[ind3,:])>0):
            #    edges.append([ind2,ind3])
        return(edges,verts)    


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
        
    def evaluate_local_ECT2(self,direction, vertex):
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
    def clean_triangles(self):
        for key in self.polygon_triangles:
            tmp=np.where(self.polygon_gains[key]!=0)[0]
            self.clean_polygon_triangles[key]=self.polygon_triangles[key][tmp,:]
            self.clean_polygon_gains[key]=self.polygon_gains[key][tmp]
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
            ind1=edges[:,0]==i
            ind2=edges[:,1]==i
            edges=edges[np.where(np.any([ind1,ind2],0)),:][0]
            verts=np.unique(edges,0)
            self.vertex_faces[i]=faces
            self.vertex_edges[i]=edges
