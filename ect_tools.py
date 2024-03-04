from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import numpy as np
from collections import Counter
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

    if(np.dot(t1,p0-p2)>0):
        t1=-t1
    if(np.dot(t2,p0-p1)>0):
        t2=-t2    
    #if(np.dot(t1,triangle)[index]>0): # From the rotated antipodal points, pick the smaller one
    #    print('testisuure')
    #    print(np.dot(t1,triangle)[index])
    #    t1=-t1 
    #if(np.dot(t2,triangle)[index]>0):
    #    t2=-t2
    stack=np.stack([main_dir,-main_dir,t1,t2])
    triangles=np.stack([[0,2,3],[1,2,3]])
    #Edges=np.stack([[0, 2], [2, 1], [1, 3], [3, 0]]) # The edges are between (0,1) and (2,3)
    # (Need add (2,3)? This would make sense)
    return(stack,triangles)

def triangulate_a_triangle_old(triangle,index=0):
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


def align_triangles(base,other):
    edge=np.array([list(set.intersection(set(base),set(other)))])[0]
    #print(edge)
    tmp1=np.where(base==edge[0])[0][0]
    tmp2=np.where(base==edge[1])[0][0]
    tmp3=np.where(other==edge[0])[0][0]
    tmp4=np.where(other==edge[1])[0][0]
    oddoneout=list(set.difference(set([0,1,2]),set([tmp3,tmp4])))[0]
    #print(tmp3)
    #print(tmp4)
    replacement=np.zeros(3)
    if((tmp2-tmp1)%2==(tmp4-tmp3)%2): # Flip the triangle if same parity
        replacement[oddoneout]=other[oddoneout]
        replacement[tmp3]=other[tmp4]
        replacement[tmp4]=other[tmp3]
        #print(replacement)
    else:
        replacement[oddoneout]=other[oddoneout]
        replacement[tmp4]=other[tmp4]
        replacement[tmp3]=other[tmp3]
    replacement=replacement.astype(int)
    return(replacement)


# Helper functions for orienting polygon triangulations
def orient_mst(t,mst,triangles):
    '''
    Given a transitive closure and mst,
    Orients the triangles
    '''
    nbs=np.stack([mst.nonzero()[0],mst.nonzero()[1]]).T
    for i in range(len(list(t))):
        root=np.array(list(t[i]))[0]
        targetset=t[i]
        seen=set([root])
        unseen=set.difference(targetset,seen)
        seenlist=list(seen)
        while (len(unseen)>0):
            new_nbs=nbs[np.where([len(set.intersection(set(nb),seen))==1 for nb in nbs])]
            #print(new_nbs)
            newly_seen=nbs[np.where([len(set.intersection(set(nb),seen))==1 for nb in nbs])[0]]
            seenones=[set(new) for new in newly_seen]
            for j in range(len(seenones)):
                edge=newly_seen[j,:]
                if len(set.intersection(set([edge[0]]),seen))==1:
                    base=edge[0]
                    other=edge[1]
                else:
                    base=edge[1]
                    other=edge[0]
                #print(triangles[other,:])
                #print(triangles[base,:])
                triangles[other,:]=align_triangles(triangles[base,:],triangles[other,:])
                seen=set.union(seen,set(seenones[j]))
                unseen=set.difference(targetset,seen)
    return(triangles)

def test_triangle(triangle,triangle_coords):
    '''
    Reorients triangle
    '''
    p0=triangle_coords[0,:]
    p1=triangle_coords[1,:]
    p2=triangle_coords[2,:]
    replacement=np.zeros(3)
    #print(np.dot(p0,np.cross(p1-p0,p2-p0)))
    #print(triangle)
    if (np.dot(p0,np.cross(p1-p0,p2-p0))<0):
        replacement[0]=triangle[0]
        replacement[1]=triangle[2]
        replacement[2]=triangle[1]
    else:
        replacement[0]=triangle[0]
        replacement[1]=triangle[1]
        replacement[2]=triangle[2]
    #print(replacement)
    return(replacement.astype(int))

def transitive_closure(mst):
    '''
    Given a minimal spanning forest
    computes the transitive closure of the nbd relations
    '''
    nbs=np.stack([mst.nonzero()[0],mst.nonzero()[1]]).T
    #print(nbs)
    cap=nbs.shape[0]**2
    grand_sum=0
    sets=[set(nb) for nb in nbs]
    while grand_sum<cap:
        for i in range(len(sets)):
            total=sum(len(set1) for set1 in sets)
            element=sets[i]
            new_element=element
            #print(element)
            inds=np.where([len(set.intersection(element,element2))>0 for element2 in sets])
            tmp_nbs=nbs[inds]
            set2=[set(nb) for nb in tmp_nbs]
            for j in range(len(set2)):
                new_element=set.union(new_element,set2[j])
            sets[i]=new_element
        new_total=sum(len(set1) for set1 in sets)
        if new_total==total:
            #print(sets)
            return(np.unique(sets))

# The following 3 functions are helper function for
# going from triangles to polygons
def find_integer_pairs_with_row_count(S):
    result = {}

    for row in S:
        pairs = [(num1, num2) for i, num1 in enumerate(row) for num2 in row[i+1:]]
        for pair in pairs:
            result[pair] = result.get(pair, 0) + 1

    return result

def filter_pairs_with_row_count(pairs_with_row_count):
    filtered_pairs = {pair: count for pair, count in pairs_with_row_count.items() if count == 1}
    return filtered_pairs

def find_all_cycles(edges):
    '''
    Given edges of a cycle, spits out the polygon
    '''
    n=len(edges)
    cycles=list()
    used_edges=list()
    unused_edges=list(range(n))
    while(len(unused_edges)>0):
        edge_ind=unused_edges[0]
        edge=edges[edge_ind]
        used_edges.append(edge_ind)
        unused_edges.remove(edge_ind)
        cycle=list()
        x0=edge[0]
        xnext=edge[1]
        cycle.append(x0)
        cycle.append(xnext)
        while xnext!=x0:
            availableones=set([*np.where([len(set.intersection(set(edge),set([xnext])))==1 for edge in edges])[0]])
            goodone=set.difference(availableones,set(used_edges))
            used_edges.append([*goodone][0])
            unused_edges.remove([*goodone][0])
            nextedge=edges[[*goodone][0]]
            nextcand=nextedge[0]
            if(nextcand==xnext):
                nextcand=nextedge[1]
                xnext=nextedge[1]
            else:
                xnext=nextedge[0]
            cycle.append(nextcand)
        cycles.append(cycle)
    return(cycles)

def polygon_wrapper(S):
    pairs=find_integer_pairs_with_row_count(S)
    inds=filter_pairs_with_row_count(pairs)
    cycles=find_all_cycles([*inds])
    return(cycles)
