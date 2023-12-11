from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import numpy as np
def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))
def normalize(vector):
    return vector/np.sum(vector**2)**0.5
def triangulate_a_triangle(triangle,index=0):
    '''
    triangle: the triangle in question, (p0,p1,p2)
    index: which entry of the triangle is the basis (0,1 or 2)

    Returns: triangulation of the directions where index is lower than the rest
    This region looks like a tomato wedge: the ends are the main_dir, and the midpoints of the edges are
    t1 and t2. These are obtained by rotating the ends along the axes defined by the cuts
    t1: Point where p0=p1 and p0 has negative sign
    t2: Point where p0=p2 and p0 has negative sign
    '''
    p0=triangle[index]
    p1=np.delete(triangle,index,0)[0]
    p2=np.delete(triangle,index,0)[1]
    main_dir=normalize(np.cross(p1-p0,p2-p0)) # The triple points
    # Next we rotate the triple point about p0=p1, p0=p2
    theta=np.pi/2 # Rotate half a circle
    axis1=normalize(p1-p0)
    axis2=normalize(p2-p0)
    #print(main_dir)
    t1=np.dot(M(axis1,theta),main_dir)
    t2=np.dot(M(axis2,theta),main_dir)

    if(np.dot(t1,triangle)[index]>0): # From the rotated antipodal points, pick the smaller one
        t1=-t1 
    if(np.dot(t2,triangle)[index]>0):
        t2=-t2

    stack=np.stack([main_dir,-main_dir,t1,t2])
    Edges=np.stack([[0, 2], [2, 1], [1, 3], [3, 0]]) # The edges are between (0,1) and (2,3)
    # (Need add (2,3)? This would make sense)
    return(stack,Edges)
