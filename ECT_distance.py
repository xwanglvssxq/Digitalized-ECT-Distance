import numpy as np
from spherical_integral import int_tri
import math
tol = 0.000001 # e-06 is enough, too small causes problems
# depends on the data precision


def rotate_x_axis(point, degrees = 15):
    radians = math.radians(degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(radians), -math.sin(radians)],
        [0, math.sin(radians), math.cos(radians)]
    ])
    point_array = np.array(point)
    rotated_point = rotation_matrix.dot(point_array)

    return rotated_point

def rotate_y_axis(point, degrees = 15):
    radians = math.radians(degrees)
    rotation_matrix = np.array([
        [math.cos(radians), 0, -math.sin(radians)],
        [0, 1, 0],
        [math.sin(radians),0, math.cos(radians)]
    ])
    point_array = np.array(point)
    rotated_point = rotation_matrix.dot(point_array)

    return rotated_point

def rotate_axis(vector, axis, angle):
# Any given axis
    axis = axis / np.linalg.norm(axis)
    angle = angle/360 * 2 * np.pi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * np.outer(axis, axis)
    
    return np.dot(rotation_matrix, vector)

def len_arc(p1, p2):
    # length of the great arc(the shorter one) on a unit sphere
    # p1, p2: 3d-np.array
    
    # Normalization is very important here! Make everything coherent with tol.
    p1 = p1/np.linalg.norm(p1)
    p2 = p2/np.linalg.norm(p2)
    if np.abs(np.dot(p1, p2)) <= 1:
        return np.arccos(np.dot(p1, p2))
    elif np.abs(np.dot(p1, p2)) > 1:
        return 0
    else:
        print('wrong inner product in len_arc')
        return None
    
def arc_diff(p, p1, p2):
    arc_diff = len_arc(p, p1) + len_arc(p, p2) - len_arc(p1, p2)
    return arc_diff
    
def great_cir(p1, p2):
    # return (pi, pj) s.t. piv=pjv on this arc
    # NOT active points
    pi = np.cross(p1, p2)
    # must normalize since comparison in arc_arc
    pi = pi/np.linalg.norm(pi)
    pj = -pi
    return pi, pj

def pt_in_arc(p, p1, p2):
    arc_diff = len_arc(p, p1) + len_arc(p, p2) - len_arc(p1, p2)
    if np.abs(arc_diff) < tol: 
        return True
    else: 
        return False
    
def arc_overlap(p11, p12, p21, p22):
    # check if two arcs belong to a same great circle
    pi, pj = great_cir(p11, p12)
    pk, pl = great_cir(p21, p22)
    if (np.abs(pi-pk) <= tol).all() or (np.abs(pi+pk) <= tol).all():
        return True
    return False
    
def arc_arc(p11, p12, p21, p22):
    pi, pj = great_cir(p11, p12)
    pk, pl = great_cir(p21, p22)
    
    # Same great circle
    if arc_overlap(p11, p12, p21, p22):
        return None
    
    # Necessary condition of no intersection
    if np.dot(pi-pj, p21) * np.dot(pi-pj, p22) > 0 or np.dot(pk-pl, p11) * np.dot(pk-pl, p12) > 0:
        return None
    else:
        indexes = [[1,2,0],[0,2,1],[0,1,2]]
        for idx in indexes:
            
            A = np.array([[pi[idx[0]]-pj[idx[0]], pi[idx[1]]-pj[idx[1]]], [pk[idx[0]]-pl[idx[0]], pk[idx[1]]-pl[idx[1]]]])
            b = np.array([pj[idx[2]]-pi[idx[2]], pl[idx[2]]-pk[idx[2]]])
            # Assume first v = [1, a, b] or [a, 1, b] or [a, b, 1], then normalize
            # Solve Ax = b, i.e. {p_iv-p_jv=0, p_kv-p_lv=0}
            if np.linalg.det(A) == 0:
                continue
            else:
                v_raw = np.linalg.solve(A, b)
                v_raw = np.insert(v_raw, idx[2],1)
                break
        
        v = v_raw/np.sqrt(np.dot(v_raw,v_raw))
        # exclude the antipodal point
        # if data points aren't precise enough, problem may arise in the following 'if'
 
        if pt_in_arc(v, p11, p12) and pt_in_arc(v, p21, p22):
            return v
        elif pt_in_arc(-v, p11, p12) and pt_in_arc(-v, p21, p22):
            return -v
        else:
            return None
        
def pt_in_sphtri(p, p1, p2, p3):
    if sph_area(p, p1, p2) == None or sph_area(p, p2, p3) == None or sph_area(p, p3, p1) == None:
        return False
    area_diff = sph_area(p, p1, p2) + sph_area(p, p2, p3) + sph_area(p, p3, p1) - sph_area(p1, p2, p3)
    if np.abs(area_diff) < tol: 
        return True
    else: 
        return False
    
def sph_angle(p1, p2, p3):
    # Check coinciding point/points on the same arc to avoid /0
    # then v1_raw, v2_raw can't be 0 in this case
    # Have checked in sph_area
    v1_raw = np.cross(np.cross(p2, p1),p2)
    v2_raw = np.cross(np.cross(p2, p3),p2)
    
    # Special case of antipodal points
    if np.linalg.norm(v1_raw) == 0 or np.linalg.norm(v2_raw) == 0:
        return None
    
    v1 = v1_raw/np.linalg.norm(v1_raw)
    v2 = v2_raw/np.linalg.norm(v2_raw)
    
    inprod = np.dot(v1, v2)
    if inprod > 1: inprod = 1
    if inprod < -1: inprod = -1
    return np.arccos(inprod)

def sph_area(p1, p2, p3):
    p1 = p1/np.linalg.norm(p1)
    p2 = p2/np.linalg.norm(p2)
    p3 = p3/np.linalg.norm(p3)
    if (np.abs(p1-p2) < tol).all(): return 0
    if (np.abs(p2-p3) < tol).all(): return 0
    if (np.abs(p1-p3) < tol).all(): return 0
    if sph_angle(p1, p2, p3) == None or sph_angle(p2, p3, p1) == None or sph_angle(p3, p1, p2) == None:
        return None
    return sph_angle(p1, p2, p3) + sph_angle(p2, p3, p1) + sph_angle(p3, p1, p2) - np.pi

def poly2tri(polygon):
    num_triangles = len(polygon) - 2 
    triangles = []
    if num_triangles <= 0:
        return np.array(triangles)
    for i in range(num_triangles):
        triangles.append([polygon[0], polygon[i+1], polygon[i+2]])
    return np.array(triangles)

def del_repetition(T_int):
    T_int = np.array(T_int)
    # check up to demicals=3
    T_check = np.round(T_int, decimals=3)
    _, T_indices = np.unique(T_check, axis=0, return_index=True)
    T_int = T_int[np.sort(T_indices)]
    return T_int

def T_intersect(Ti, Tj):
    Ti = np.array(Ti)
    Tj = np.array(Tj)
    
    # identical
    if np.array_equal(Ti, Tj):
        return Ti
    # Tj in Ti
    if pt_in_sphtri(Tj[0], Ti[0], Ti[1], Ti[2]) and \
    pt_in_sphtri(Tj[1], Ti[0], Ti[1], Ti[2]) and \
    pt_in_sphtri(Tj[2], Ti[0], Ti[1], Ti[2]):
        return Tj
    # Ti in Tj
    if pt_in_sphtri(Ti[0], Tj[0], Tj[1], Tj[2]) and \
    pt_in_sphtri(Ti[1], Tj[0], Tj[1], Tj[2]) and \
    pt_in_sphtri(Ti[2], Tj[0], Tj[1], Tj[2]):
        return Ti
    
    idxes = np.array([[0,1], [1,2], [2,0]])
    T_int = []
    
    p_addition = Tj[np.array([pt_in_sphtri(Tj[v], Ti[0], Ti[1], Ti[2]) for v in range(3)])]
    # len(p_addition) = 0,1,2
    if len(p_addition) == 2:
        # 0-->2 to 2-->0
        if pt_in_sphtri(Tj[0], Ti[0], Ti[1], Ti[2]) and pt_in_sphtri(Tj[2], Ti[0], Ti[1], Ti[2]):
            p_addition[0], p_addition[1] = p_addition[1], p_addition[0]
    addition = False
    
    # compute three p_ints
    for idx_i in idxes:
        p_ints = []
        for idx_j in idxes:
            p_int = arc_arc(Ti[idx_i[0]], Ti[idx_i[1]], Tj[idx_j[0]], Tj[idx_j[1]])
            # if two arcs overlap, p_int is also none
            if p_int is not None:
                p_ints.append(p_int)
        # Now consider when arcs overlap
        for idx_j in idxes:
            if arc_overlap(Ti[idx_i[0]], Ti[idx_i[1]], Tj[idx_j[0]], Tj[idx_j[1]]):
                p_ints = []

        #reverse the order since first meet p_ints[1]
        if len(p_ints) == 2:
            if len_arc(Ti[idx_i[0]], p_ints[1]) < len_arc(Ti[idx_i[0]], p_ints[0]):
                p_ints[0], p_ints[1] = p_ints[1], p_ints[0]
            
        if pt_in_sphtri(Ti[idx_i[0]], Tj[0], Tj[1], Tj[2]):
            # state == 'in'
            # Think about this, it's subtle
            T_int.extend([Ti[idx_i[0]]] + p_ints) # [nparray] + [nparray] = [..., ...]
        else:
            # state == 'out'
            if addition is False:
                # find a proper timing to insert p_addition(when Ti can go into Tj two times)
                # len(p_addition) == 2 won't meet this problem
                if len(p_addition) == 1 and len(p_ints) != 0:
                    if not any(pt_in_arc(p_ints[0], p_addition[0], pt) for pt in Tj[:]):
                        pass
                    else:
                        T_int.extend(p_addition)
                        addition = True
                else:
                    T_int.extend(p_addition)
                    addition = True
            T_int.extend(p_ints)  
    
    #delete repetitions
    T_int = del_repetition(T_int)
    if len(T_int[:]) > 2:
        return T_int 
    return None
def arc_cir(p1, p2, pk, pl):
    #NOTICE: pk, pl here are not antipodal!
    
    # check whether arc p1->p2 intersects piv=pjv
    if np.dot(pk-pl, p1) * np.dot(pk-pl, p2) > 0:
        return None
    # overlap and hence no division, use tol to avoid extreme small det(A)
    elif np.abs(np.dot(pk-pl, p1)) < tol and np.abs(np.dot(pk-pl, p2)) < tol:
        return None
    else:
        pi, pj = great_cir(p1, p2)
        indexes = [[1,2,0],[0,2,1],[0,1,2]]
        for idx in indexes:
            A = np.array([[pi[idx[0]]-pj[idx[0]], pi[idx[1]]-pj[idx[1]]], [pk[idx[0]]-pl[idx[0]], pk[idx[1]]-pl[idx[1]]]])
            b = np.array([pj[idx[2]]-pi[idx[2]], pl[idx[2]]-pk[idx[2]]])
            # Assume first v = [1, a, b] or [a, 1, b] or [a, b, 1], then normalize
            # Solve Ax = b, i.e. {p_iv-p_jv=0, p_kv-p_lv=0}
            if np.linalg.det(A) == 0:
                continue
            else:
                v_raw = np.linalg.solve(A, b)
                v_raw = np.insert(v_raw, idx[2],1)
                break
        v = v_raw/np.sqrt(np.dot(v_raw,v_raw))

        if pt_in_arc(v, p1, p2): 
            p_int = v
        elif pt_in_arc(-v, p1, p2): 
            p_int = -v
        else: 
            print('error in arc_cir: ',arc_diff(v, p1, p2), arc_diff(-v, p1, p2))
            if arc_diff(v, p1, p2) < arc_diff(-v, p1, p2):
                p_int = v
            else:
                p_int = -v

        return p_int
    
def T_div(Ti, Tj, pi, pj):
    T_int = T_intersect(Ti, Tj)
    T_divi = []
    T_divj = []
    
    # nothing
    if T_int is None:
        return np.array(T_divi), np.array(T_divj)
    
    # identical
    if (np.abs(pi-pj) < tol).all():
        return np.array(T_int), np.array([])
    
    for idx_pt, pt in enumerate(T_int):
        if np.dot(pt, pi-pj) == 0:
            T_divi.append(pt)
            T_divj.append(pt)
        else:
            if np.dot(pt, pi-pj) < 0:
                T_divj.append(pt)
            if np.dot(pt, pi-pj) > 0:
                T_divi.append(pt)
            # n-1 ----> 0
            if idx_pt == len(T_int) - 1:
                idx_pt = -1
            p_int = arc_cir(pt, T_int[idx_pt+1], pi, pj)
            if p_int is not None:
                T_divi.append(p_int)
                T_divj.append(p_int)
    T_divi = del_repetition(T_divi)
    T_divj = del_repetition(T_divj)
    
    if len(T_divi[:]) <= 2:
        T_divi = np.array([])
    if len(T_divj[:]) <= 2:
        T_divj = np.array([])
        
    return T_divi, T_divj

def ECT_distance(ECT1, ECT2):
    '''
    Formation of 'ECT1':
    ECT1 = sum alpha_i * f_i
    should be stored as array of triples: (alpha_i, p_i, T_i), i.e. active point
    
    Example:
    ECT1 = [[alpha_0, p_0, T_0], [alpha_1, p_1, T_1], ..., [alpha_n, p_n, T_n]]
    '''
    ECT = ECT1 + ECT2
    integral = 0
    
    for idx1 in range(len(ECT)):
        for idx2 in range(len(ECT)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            T_divi, T_divj = T_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][1], ECT[idx2][1])
            if len(T_divi[:]) > 2:
                triangles_i = poly2tri(T_divi)
                for Ti in triangles_i:
                    integral_i += int_tri(Ti[0], Ti[1], Ti[2], ECT[idx1][1])
                    integral_s += sph_area(Ti[0], Ti[1], Ti[2])
            if len(T_divj[:]) > 2:
                triangles_j = poly2tri(T_divj)
                for Tj in triangles_j:
                    integral_j += int_tri(Tj[0], Tj[1], Tj[2], ECT[idx2][1])
                    integral_s += sph_area(Tj[0], Tj[1], Tj[2])
            integral += ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)
            
    return integral

if __name__ == "__main__":
    p0 = np.array([0.577, 0.577, 0.577])
    p0 = p0/np.linalg.norm(p0)
    p1 = np.array([1,0,0])
    p2 = np.array([0,1,0])
    p3 = np.array([0,0,1])
    p4 = np.array([0.182, -0.69, -0.69])
    p4 = p4/np.linalg.norm(p4)
    p5 = np.array([-0.69, -0.69, 0.182])
    p5 = p5/np.linalg.norm(p5)
    p6 = np.array([-0.69, 0.182, -0.69])
    p6 = p6/np.linalg.norm(p6)
    T0 = [p4, p5, p6]
    T1 = [p0, p6, p5]
    T2 = [p0, p5, p4]
    T3 = [p0, p4, p6]
    ECT1 = [[1, p0, T0], [1, p1, T1], [1, p2, T2], [1, p3, T3]]
    
    D_ECT = []
    # Choose a direction
    v = np.array([1,2,3])

    for deg in np.arange(0, 365, 5):
        pp0 = rotate_axis(p0,v, deg)
        pp1 = rotate_axis(p1, v,deg)
        pp2 = rotate_axis(p2, v,deg)
        pp3 = rotate_axis(p3, v,deg)
        pp4 = rotate_axis(p4, v,deg)
        pp5 = rotate_axis(p5, v,deg)
        pp6 = rotate_axis(p6, v,deg)

        '''
        pp0 = rotate_y_axis(p0, deg)
        pp1 = rotate_y_axis(p1, deg)
        pp2 = rotate_y_axis(p2, deg)
        pp3 = rotate_y_axis(p3, deg)
        pp4 = rotate_y_axis(p4, deg)
        pp5 = rotate_y_axis(p5, deg)
        pp6 = rotate_y_axis(p6, deg)
        '''

        TT0 = [pp4, pp5, pp6]
        TT1 = [pp0, pp6, pp5]
        TT2 = [pp0, pp5, pp4]
        TT3 = [pp0, pp4, pp6]
        ECT2 = [[-1, pp0, TT0], [-1, pp1, TT1], [-1, pp2, TT2], [-1, pp3, TT3]]
        d = ECT_distance(ECT1, ECT2)
        print(f'deg={deg}, d_ECT=', d)
        D_ECT.append(d)
        
    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, 365, 5), D_ECT)

    plt.title("Tetrahedron")
    plt.xlabel("degrees of rotation around randomly chosen axis")
    plt.ylabel("ECT distance")

    plt.show()
