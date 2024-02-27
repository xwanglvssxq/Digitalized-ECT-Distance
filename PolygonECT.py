'''
ECT_distance receive *oriented*(maybe not necessary) *ECT2 reversed jump* digitization.
'''

import numpy as np
import math
from scipy.stats import special_ortho_group
from scipy import integrate

tol = 0.000001 # e-06 is enough, too small tol causes problems

def detect_rot(P, pi):
    tol = 0.000001
    N = len(P)
    
    # Detect poles
    for i in range(N):
        if np.abs(P[i][2]) > 1-tol: return False
    if np.abs(pi[2]) > 1-tol: return False

    # Detect phi=0 arc, i.e.
    for i in range(N):
        if P[i][0] > 0 and np.abs(P[i][1]) == 0: return False
    if pi[0] > 0 and np.abs(pi[1]) == 0: return False

    # Detect meridian
    for i in range(N):
        if np.abs(P[i][0] * P[(i+1)%N][1] - P[(i+1)%N][0] * P[i][1]) <= tol: return False

    # Detect equator
    for i in range(N):
        if P[i][2] == 0 and P[(i+1)%N][2] == 0: return False

    # Detect crossing arc
    phis = []
    for i in range(N):
        phi = math.atan2(P[i][1], P[i][0])
        if phi < 0: phi += 2 * np.pi
        phis.append(phi)
    for i in range(N):
        if np.abs(phis[i]-phis[(i+1)%N]) > np.pi: return False
    return True

def roted_sphere(P, pi):
    truth_value = detect_rot(P, pi)
    while truth_value == False:
        rotation_matrix = special_ortho_group.rvs(3)
        for i in range(len(P)):
            P[i] = np.dot(rotation_matrix, P[i])
        pi = np.dot(rotation_matrix, pi)
        truth_value = detect_rot(P, pi)
    return P, pi

def cartesian_to_spherical(x, y, z):
    tau = math.asin(z)
    phi = math.atan2(y, x)
    if math.atan2(y, x) < 0:
        phi += 2 * np.pi
    return phi, tau

def solve_great_circle(phi_1, tau_1, phi_2, tau_2):
    # compute phi_0, a
    r1 = np.cos(phi_1) * np.tan(tau_2) - np.cos(phi_2) * np.tan(tau_1)
    r2 = -np.sin(phi_1) * np.tan(tau_2) + np.sin(phi_2) * np.tan(tau_1)

    #if r2 = 0:
    if np.abs(r2) < tol:
        phi_0 = np.pi/2
        # phi_0 can be -pi/2 with 'a' becoming '-a', we get the same equation                                
    else: 
        phi_0 = np.arctan(r1/r2)

    # cos = 0, use another point
    if np.abs(np.abs((phi_1 - phi_0)%np.pi) - np.pi/2) < tol:       
        a = np.tan(tau_2)/np.cos(phi_2 - phi_0)
    else:
        a = np.tan(tau_1)/np.cos(phi_1 - phi_0)
    
    return phi_0, a


def int_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i):
    phi_0, a = solve_great_circle(phi_1, tau_1, phi_2, tau_2)
    
    def f1(phi):
        return (1 - np.power(a,2) * np.power(np.cos(phi-phi_0),2))/(1 + np.power(a,2) * np.power(np.cos(phi-phi_0),2))
    def f2(phi):
        return (2 * a * np.cos(phi-phi_0) * np.cos(phi-phi_i))/(1 + np.power(a,2) * np.power(np.cos(phi-phi_0),2))
    def f3(phi):
        return np.arctan(a * np.cos(phi-phi_0)) * np.cos(phi-phi_i)
        
    I1, error1 = integrate.quad(f1, phi_1, phi_2)
    I2, error2 = integrate.quad(f2, phi_1, phi_2)
    I3, error3 = integrate.quad(f3, phi_1, phi_2)

    integral = 0.25 * np.sin(tau_i) * I1 - 0.25 * np.cos(tau_i) * I2 -0.5 * np.cos(tau_i) * I3
    return integral

def int_polygon(P, pi):
    I = 0
    N = len(P)
    P, pi = roted_sphere(P, pi)
    
    phi_i, tau_i = cartesian_to_spherical(pi[0], pi[1], pi[2])
    phis = []
    taus = []
    
    for i in range(N):
        phi, tau = cartesian_to_spherical(P[i][0], P[i][1], P[i][2])
        phis.append(phi)
        taus.append(tau)
    
    for i in range(N):
        I += int_arc(phis[i], taus[i], phis[(i+1)%N], taus[(i+1)%N], phi_i, tau_i)

    return I

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

def pt_in_arc(p, p1, p2):
    arc_diff = len_arc(p, p1) + len_arc(p, p2) - len_arc(p1, p2)
    if np.abs(arc_diff) < tol: 
        return True
    else: 
        return False

def great_cir(p1, p2):
    # return (pi, pj) s.t. piv=pjv on this arc
    # NOT active points
    pi = np.cross(p1, p2)
    pi = pi/np.linalg.norm(pi)
    pj = -pi
    return pi, pj
    
def arc_overlap(p11, p12, p21, p22):
    pi, pj = great_cir(p11, p12)
    pk, pl = great_cir(p21, p22)
    if (np.abs(pi-pk) <= tol).all() or (np.abs(pi+pk) <= tol).all():
        return True
    return False

def arc_arc(p11, p12, p21, p22):
    pi, pj = great_cir(p11, p12)
    pk, pl = great_cir(p21, p22)
    
    if arc_overlap(p11, p12, p21, p22):
        return None
    
    # good necessary condition of no intersection
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

def polygon_area(P):
    N = len(P)
    S = 0
    for i in range(N):
        P[i] = P[i]/np.linalg.norm(P[i])
    for i in range(N):
        if (np.abs(P[i]-P[(i+1)%N]) < tol).all(): return 0
        if sph_angle(P[i], P[(i+1)%N], P[(i+2)%N]) == None: return None
        S += sph_angle(P[i], P[(i+1)%N], P[(i+2)%N])
    return S - (N-2) * np.pi

def del_repetition(P_int):
    P_int = np.array(P_int)
    # check up to demicals=3
    P_check = np.round(P_int, decimals=3)
    _, P_indices = np.unique(P_check, axis=0, return_index=True)
    P_int = P_int[np.sort(P_indices)]
    return P_int

def P_intersect(P1, P2):
    '''
    Pi = [[ai], [pi], B = [oriented boundary: [v0, v1, v2, ..., vn]]
    '''
    pts = []
    N1 = len(P1)
    N2 = len(P2)
    P1_in = np.zeros(N1, dtype=int)
    P2_in = np.zeros(N2, dtype=int)
    
    
    for i in range(N1):
        for j in range(N2):
            
            #check if P2[j] inside P1(or converse)
            if P1_in[i] == 0 and j > 0 and j < N2-1: 
                if pt_in_sphtri(P1[i], P2[0], P2[j], P2[j+1]):
                    pts.append(P1[i])
                    P1_in[i] = 1
            if P2_in[j] == 0 and i > 0 and i < N1-1:
                if pt_in_sphtri(P2[j], P1[0], P1[i], P1[i+1]): 
                    pts.append(P2[j])
                    P2_in[j] = 1
            
            #Check if (P1[i], P1[i+1]) intersects with (P2[j], P2[j+1])
            #No need for colinear case
            p_int = arc_arc(P1[i], P1[(i+1)%N1], P2[j], P2[(j+1)%N2])
            if p_int is not None: pts.append(p_int) 

    #clean data
    pts = del_repetition(pts)
    if len(pts) <= 2: return None
    
    n = pts[0]/np.linalg.norm(pts[0])
    proj_pts = [pts[i] - np.dot(pts[i], n) * n for i in range(len(pts))]
    # rearrange angles from neg to pos, get index
    angles = [np.sign(np.dot(n , np.cross(proj_pts[1], proj_pts[i+1]))) * np.degrees( np.arccos(np.clip(np.dot(proj_pts[1]/np.linalg.norm(proj_pts[1]) , proj_pts[i+1]/np.linalg.norm(proj_pts[i+1]) ), -1, 1)) ) for i in range(len(pts)-1)]
    angles = np.array(angles)
    sorted_indices = angles.argsort()
    sorted_pts = pts[sorted_indices+1]
    sorted_pts = np.vstack((pts[0], sorted_pts))
    return sorted_pts

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
    
def P_div(Pi, Pj, pi, pj):
    P_int = P_intersect(Pi, Pj)
    P_divi = []
    P_divj = []
    
    # nothing
    if P_int is None:
        return np.array(P_divi), np.array(P_divj)
    
    # identical
    if (np.abs(pi-pj) < tol).all():
        return np.array(P_int), np.array([])
    
    for idx_pt, pt in enumerate(P_int):
        if np.dot(pt, pi-pj) == 0:
            P_divi.append(pt)
            P_divj.append(pt)
        else:
            if np.dot(pt, pi-pj) < 0:
                P_divj.append(pt)
            if np.dot(pt, pi-pj) > 0:
                P_divi.append(pt)
            # n-1 ----> 0
            if idx_pt == len(P_int) - 1:
                idx_pt = -1
            p_int = arc_cir(pt, P_int[idx_pt+1], pi, pj)
            if p_int is not None:
                P_divi.append(p_int)
                P_divj.append(p_int)
    P_divi = del_repetition(P_divi)
    P_divj = del_repetition(P_divj)
    
    if len(P_divi) <= 2:
        P_divi = np.array([])
    if len(P_divj) <= 2:
        P_divj = np.array([])
        
    return P_divi, P_divj


def ECT_distance(ECT1, ECT2):
    '''
    Formation of 'ECT1':
    ECT1 = sum alpha_i * f_i
    should be stored as array of triples: (alpha_i, p_i, P_i), i.e. active point
    
    Example:
    ECT1 = [[alpha_0, p_0, P_0], [alpha_1, p_1, P_1], ..., [alpha_n, p_n, P_n]]
    '''
    ECT = ECT1 + ECT2
    integral = 0
    
    for idx1 in range(len(ECT)):
        for idx2 in range(len(ECT)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            P_divi, P_divj = P_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][1], ECT[idx2][1])
            if len(P_divi) > 2:
                integral_i = int_polygon(P_divi, ECT[idx1][1])
                integral_s += polygon_area(P_divi)
            if len(P_divj) > 2:
                integral_j = int_polygon(P_divj, ECT[idx2][1])
                integral_s += polygon_area(P_divj)
                    
            integral += ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)
    return integral

###### Auxiliary functions ######
def rotate_axis(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    angle = angle/360 * 2 * np.pi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * np.outer(axis, axis)
    
    return np.dot(rotation_matrix, vector)

def return_ECT(s1,s2):
    tmp=[]
    for key in s1.clean_polygon_gains:
        TMP=s1.clean_polygon_gains[key]
        for j in range(TMP.shape[0]):
            megatmp=[]
            megatmp.append(TMP[j])
            megatmp.append(s1.V[key,:])
            megatmp.append(s1.polygon_angles[key][s1.polygon_triangles[key][j,:],:])
            tmp.append(megatmp)
    tmp2=[]
    for key in s2.clean_polygon_gains:
        TMP=-s2.clean_polygon_gains[key]
        for j in range(TMP.shape[0]):
            megatmp=[]
            megatmp.append(TMP[j])
            megatmp.append(s2.V[key,:])
            megatmp.append(s2.polygon_angles[key][s2.polygon_triangles[key][j,:],:])
            tmp2.append(megatmp)
    ECT1=tmp
    ECT2=tmp2
    return ECT1, ECT2

from shape_reader import ShapeReader

def com_ECT(degrees, v):
    s1=ShapeReader.shape_from_file('meshes/octahedron.off')
    s2=ShapeReader.shape_from_file('meshes/octahedron.off')
    for i in range(s1.V.shape[0]):
        s1.V[i,:] = s1.V[i,:]/(np.sum(s1.V[i,:]**2)**(0.5))
    for i in range(s2.V.shape[0]):
        s2.V[i,:] = s2.V[i,:]/(np.sum(s2.V[i,:]**2)**(0.5))
        s2.V[i,:] = rotate_axis(s2.V[i,:], v, degrees)
        #v=v/sum(v**2)**(0.5)
    s1.prepare()
    s1.compute_links()
    #s1.compute_TP_DT_vol4()
    s1.compute_TP_DT_vol3()
    s1.compute_gains()
    s1.clean_triangles()
    s2.prepare()
    s2.compute_links()
    #s2.compute_TP_DT_vol4()
    s2.compute_TP_DT_vol3()
    s2.compute_gains()
    s2.clean_triangles()
    '''
    ECT1 = return_ECT(s1)
    ECT2 = return_ECT(s2)
    '''
    ECT1, ECT2 = return_ECT(s1, s2)
    
    #Change orientation!
    for t in range(len(ECT1)):
        vec_1 = ECT1[t][2][1] - ECT1[t][2][0]
        vec_2 = ECT1[t][2][2] - ECT1[t][2][0]
        # assume the orientation to be T[0]-->T[1]-->T[2]
        orientation = np.dot(ECT1[t][2][0], np.cross(vec_1, vec_2))    
        if orientation < 0:
            tmp1 = ECT1[t][2][2].copy()
            ECT1[t][2][2] = ECT1[t][2][1]
            ECT1[t][2][1] = tmp1
    for t in range(len(ECT2)):
        vec_3 = ECT2[t][2][1] - ECT2[t][2][0]
        vec_4 = ECT2[t][2][2] - ECT2[t][2][0]
        # assume the orientation to be T[0]-->T[1]-->T[2]
        orientation = np.dot(ECT2[t][2][0], np.cross(vec_3, vec_4))    
        if orientation < 0:
            tmp2 = ECT2[t][2][2].copy()
            ECT2[t][2][2] = ECT2[t][2][1]
            ECT2[t][2][1] = tmp2
    
    return ECT_distance(ECT1, ECT2)
