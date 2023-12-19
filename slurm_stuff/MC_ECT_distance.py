import numpy as np
from scipy.stats import special_ortho_group

def gen_S2points(num_pts):
    # random SO(3) rotation
    rotation_matrix = special_ortho_group.rvs(3, size=num_pts)
    initial_point = np.array([0, 0, 1])
    points = np.dot(rotation_matrix, initial_point)
    return points

def in_halfspace(points, halfspace_level, normal):
    # For V
    if np.array(points).shape[0] == 1:
        if np.dot(normal, points[0] - halfspace_level) > 0: return False
        return True
    # For E,F
    for pt in points:
        if np.dot(normal, pt - halfspace_level) > 0:
            return False
    return True

def EC(V, E, F, halfspace_normal, halfspace_level):
    V_in = [v for v in V if in_halfspace([v], halfspace_level, halfspace_normal)]
    E_in = [e for e in E if in_halfspace(e, halfspace_level, halfspace_normal)]
    F_in = [f for f in F if in_halfspace(f, halfspace_level, halfspace_normal)]
    V_in = np.array(V_in)
    E_in = np.array(E_in)
    F_in = np.array(F_in)
    EC = V_in.shape[0] - E_in.shape[0] + F_in.shape[0]
    return EC

def get_EF(V, F_idx):
    E_idx = []
    E = []
    F = []
    for f in F_idx: 
        tmp = np.array([np.delete(f, 0), np.delete(f, 1), np.delete(f, 2)])
        tmp = tmp.astype(int)
        E_idx.extend(tmp)
        f = f.astype(int)
        F.append(V[f])
        
    # clean repetitions
    E_idx = set(map(tuple, E_idx))
    E_idx = list(map(list, E_idx))
    
    for e in E_idx: E.append(V[e])

    return E, F

def ECT_distance_MC(s1, s2, num_pts = 1000):
    V1 = s1.V
    V2 = s2.V
    T1 = s1.T
    T2 = s2.T
    sorted_T1 = []
    sorted_T2 = []
    for t in T1:
        t = np.sort(t)
        sorted_T1.append(t)
    for t in T2:
        t = np.sort(t)
        sorted_T2.append(t)
    F1_idx = sorted_T1
    F2_idx = sorted_T2
    E1, F1 = get_EF(V1, F1_idx)
    E2, F2 = get_EF(V2, F2_idx)
    
    Diff = 0
    halfspace_normals = gen_S2points(num_pts)
    for n in halfspace_normals:
        scalar = np.random.uniform(-1, 1, 1)
        halfspace_level = scalar * n
        EC1 = EC(V1, E1, F1, n, halfspace_level)
        EC2 = EC(V2, E2, F2, n, halfspace_level)
        Diff += (EC1-EC2)**2
    return Diff/num_pts
    E1, F1 = get_EF(V1, F1_idx)
    E2, F2 = get_EF(V2, F2_idx)
    
    Diff = 0
    halfspace_normals = gen_S2points(num_pts)
    for n in halfspace_normals:
        scalar = np.random.uniform(-1, 1, 1)
        halfspace_level = scalar * n
        EC1 = EC(V1, E1, F1, n, halfspace_level)
        EC2 = EC(V2, E2, F2, n, halfspace_level)
        Diff += (EC1-EC2)**2
    return Diff/num_pts
