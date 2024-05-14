# Remember to check:
# 1. pts on spherical polygon should be normalied at least and only once (done in orientation).
# 2. polygons in ECT should be in correct orientation (done in orientation).
# 3. meshes should be supported in a unit ball.

# 1. 4.28: non-stopping then detect_rot modified.
# 2. 4.29, 5.9: modify tol in pt_in_arc (need to consider total len).
# 3. 4.29: modify colinear case in [P_intersect,angles] (removed)
# 4. 4.30: modify pt_in_sphpoly(applied), oppo_hemisphere tol(removed): from tol to 0.
# 5. 5.6: pt_in_sph, opposemi: normalize n (can mitigate the problem of tol).
# 6. 5.6: add error report: convex_checker(improved n0 in 5.9), polygon area.
# 7. 5.8: delete normalize in pt_in_arc since has been normalized in arc_arc and arc_cir.
# 8. 5.8: tighten tol in arc_arc to e-10(removed), tighten demical in del_repitition to 6(removed).
# 9. 5.9: modify ECTd, ECTs; no use ECTp and return_ECT
# 10. 5.9: modify arc_arc: normalization of p1, p2 avoid influence of arc length.
# 11. 5.9: modify P_int: delete N<=1 case and add len(pts)==2 discussion.
# 12. 5.9 rewrite del_rep, set d=5 in arc_cir.(removed)
# 13. 5.9 switch to del_rep2, avoid extrmely closed pts.
# 14. 5.10 about#4: pt_in_sphpoly tol to 0; oppo still tol.

from scipy.stats import special_ortho_group
import numpy as np
import math
from scipy import integrate
from shape_reader import ShapeReader

tol = 0.000001 

def detect_rot(P, pi):
    # Here p1, p2, p3, pi are the Cartesian coordinates
    # P sperical polygon
    # pi not necessarily on sphere
    N = len(P)

    # Detect poles
    for i in range(N):
        if np.abs(P[i][2]) > 1-tol: return False
    
    if pi[0] == 0 and pi[1] == 0: return False

    # Detect phi=0 arc
    for i in range(N):
        if P[i][0] > 0 and np.abs(P[i][1]) < tol: return False
    
    if pi[0] > 0 and np.abs(pi[1]) < tol: return False

    # Detect meridian
    for i in range(N):
        #if np.abs(np.abs(P[i][0]/P[i][1])-np.abs(P[(i+1)%N][0]/P[(i+1)%N][1])) < tol: return False
        if np.abs(P[i][0]*P[(i+1)%N][1]-P[i][1]*P[(i+1)%N][0]) < tol * np.sum((P[i]-P[(i+1)%N])**2): return False

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
    # Apr. 5: pi is inside the unit ball,
    # but NOT necessarily on the unit sphere
    I = 0
    N = len(P)
    P, pi = roted_sphere(P, pi)
    r = fast_norm(pi)
    
    phi_i, tau_i = cartesian_to_spherical(pi[0]/r, pi[1]/r, pi[2]/r)
    phis = []
    taus = []
    
    for i in range(N):
        phi, tau = cartesian_to_spherical(P[i][0], P[i][1], P[i][2])
        phis.append(phi)
        taus.append(tau)
    
    for i in range(N):
        I += int_arc(phis[i], taus[i], phis[(i+1)%N], taus[(i+1)%N], phi_i, tau_i)
    
    I = r * I

    return I

def fast_cross(a, b):
    c = np.empty(3)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

def fast_dot(a, b):
    d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return d

def fast_norm(a):
    n = fast_dot(a, a) ** 0.5
    return n

def fast_compare(a,b):
    if np.abs(a[0]-b[0]) < tol and np.abs(a[1]-b[1]) < tol and np.abs(a[2]-b[2]) < tol:
        return True
    return False

def fast_clip(r):
    if r > 1: r = 1
    elif r < -1: r = -1
    return r

def len_arc(p1, p2):
    # length of the great arc(the shorter one) on a unit sphere
    # need to include pi
    # p1, p2: 3d-np.array
    
    # Normalization is very important here! Make everything coherent with tol.
    # Normalized in arc_arc and initialization.
    
    pdot = fast_dot(p1, p2)
    pdot = fast_clip(pdot)
    length = np.arccos(pdot)
    return length

def pt_in_arc(p, p1, p2):
    '''
    p = p/fast_norm(p)
    p1 = p1/fast_norm(p1)
    p2 = p2/fast_norm(p2)
    '''
    #arc_diff = len_arc(p, p1) + len_arc(p, p2) - len_arc(p1, p2)
    l1 = len_arc(p, p1)
    l2 = len_arc(p, p2)
    l12 = len_arc(p1, p2)
    arc_diff = l1 + l2 -l12
    #if np.abs(arc_diff) < 50 * tol: 
    #if np.abs(arc_diff) < tol: 
    #print(arc_diff, 100*tol*(l12**0.5), l12)
    if np.abs(arc_diff) < 10*tol*(l12**0.5):
        return True
    else: 
        return False

def arc_arc(p11, p12, p21, p22):
    p1 = fast_cross(p11, p12)
    p2 = fast_cross(p21, p22)
    
    p1 = p1/fast_norm(p1)
    p2 = p2/fast_norm(p2)
    

    #VERY GOOD necessary condition of no intersection
    #If fast_dot(p1, p21) * fast_dot(p1, p22) > tol or fast_dot(p2, p11) * fast_dot(p2, p12) > tol:
    #why '>0': Now we only need to consider different triangles, avoiding colinear problems.
    #No. In d11 and d22, it's still possible to be colinear.(POSSIBLE BUGS)
    if fast_dot(p1, p21) * fast_dot(p1, p22) > 0 or fast_dot(p2, p11) * fast_dot(p2, p12) > 0:
        return None
    v = fast_cross(p1, p2)
    nv = fast_norm(v)
    if nv < tol: 
        #In d11 and d22, it's still possible to be colinear as different vertices share the same edges.
        #It's experimentally rare(never seen) but possible: two arcs are so closed to each other that nv < tol.(POSSIBLE BUGS)
        #if nv != 0: print('nauty case in arc_arc')
        return None
    #if nv == 0: return None
    #if nv < 0.01 * tol: return None
    v = v/nv
    # exclude the antipodal point
    # if data points aren't precise enough, problem may arise in the following 'if'
    if pt_in_arc(v, p11, p12) and pt_in_arc(v, p21, p22):
        return v
    elif pt_in_arc(-v, p11, p12) and pt_in_arc(-v, p21, p22):
        return -v
    else:
        return None

def sph_angle(p1, p2, p3):
    # Check coinciding point/points on the same arc to avoid /0
    # then v1_raw, v2_raw can't be 0 in this case
    # Have checked in sph_area
    v1_raw = fast_cross(fast_cross(p2, p1),p2)
    v2_raw = fast_cross(fast_cross(p2, p3),p2)
    
    # Special case of antipodal points
    if fast_norm(v1_raw) == 0 or fast_norm(v2_raw) == 0:
        return None
    
    v1 = v1_raw/fast_norm(v1_raw)
    v2 = v2_raw/fast_norm(v2_raw)
    
    inprod = fast_dot(v1, v2)
    inprod = fast_clip(inprod)
    return np.arccos(inprod)

def polygon_area(P):
    #used in ECT_distance
    N = len(P)
    S = 0
    for i in range(N):
        P[i] = P[i]/fast_norm(P[i])
    for i in range(N):
        if (np.abs(P[i]-P[(i+1)%N]) < tol).all(): return 0
        if sph_angle(P[i], P[(i+1)%N], P[(i+2)%N]) == None: return None
        S += sph_angle(P[i], P[(i+1)%N], P[(i+2)%N])
    Area = S - (N-2) * np.pi
    if Area <= -tol: print('error in polygon_area')
    return S - (N-2) * np.pi

def pt_in_sphpoly(pt, P):
    #ECT is orientized in orientation(return_ECT)
    N = len(P)

    for i in range(N):
        n = fast_cross(P[i], P[(i+1)%N])
        n = n/fast_norm(n)
        n_value = fast_dot(n, pt)
        if n_value < 0: return False
        #if n_value < -tol: return False
    return True
'''
def del_repetition(P_int, d):
    P_int = np.array(P_int)
    unique_dict = {}
    for index, vec in enumerate(P_int):
        key = tuple(np.round(vec, decimals = d))
        #key = tuple(np.round(vec, decimals = 3))
        #key = tuple(np.round(vec, decimals = 6))
        #key = tuple(np.round(vec, decimals = 5))
        if key not in unique_dict:
            unique_dict[key] = index
    unique_indices = list(unique_dict.values())
    unique_vectors = P_int[unique_indices]
    return unique_vectors
'''

def del_repetition2(P_int, th):
    P_int = np.array(P_int)
    P_unique = []
    N = len(P_int)
    repetition = 0
    for i in range(N):
        for j in range(N-i-1):
            if len_arc(P_int[i], P_int[j+i+1]) < th: 
                repetition = 1
                break
        if repetition == 1: repetition = 0
        else: P_unique.append(P_int[i])
    return np.array(P_unique)


def oppo_hemisphere(n, P):
    # n is a normal vector for an edge, if P[i]*n<0 for any i, no intersections beween P and the polygon for n
    N = len(P)
    for i in range(N):
        #if fast_dot(n, P[i]) > 0: return False
        if fast_dot(n, P[i]) > tol: return False
    return True

def necessary_nointersect(P1, P2):
    #GOOD(maybe BEST) necessary condition for no intersection
    #ECT is orientized in orientation(return_ECT)
    N1 = len(P1)

    for i in range(N1):
        n = fast_cross(P1[i], P1[(i+1)%N1])
        n = n/fast_norm(n)
        if oppo_hemisphere(n, P2): return True
    return False

def P_intersect(P1, P2):
    '''
    Pi = [v0, v1, v2, ..., vn]
    Better to be oriented
    '''
    if necessary_nointersect(P1, P2): return None
    
    raw_pts = []
    N1 = len(P1)
    N2 = len(P2)
    
    for i in range(N1):
        for j in range(N2):
            #Check if (P1[i], P1[i+1]) intersects with (P2[j], P2[j+1])
            #No need for colinear case, which is considered by pt_in_sphpoly(POSSIBLE BUGS).
            p_int = arc_arc(P1[i], P1[(i+1)%N1], P2[j], P2[(j+1)%N2])
            if p_int is not None: raw_pts.append(p_int) 
    #pts = del_repetition(pts)
    #pts = pts.tolist()
    '''
    #There are at least 2 intersections in nontrivial cases
    if N3 > 1:
        for i in range(N1):
            if pt_in_sphpoly(P1[i], P2): pts.append(P1[i])
        for j in range(N2):
            if pt_in_sphpoly(P2[j], P1): pts.append(P2[j])
            
    #check if P2[j] inside P1(or converse)     
    elif pt_in_sphpoly(P1[0], P2):
        pts.append(P1[0])
        for i in range(N1):
            if i > 0 and pt_in_sphpoly(P1[i], P2): pts.append(P1[i])
    elif pt_in_sphpoly(P2[0], P1):
        pts.append(P2[0])
        for j in range(N2):
            if j > 0 and pt_in_sphpoly(P2[j], P1): pts.append(P2[j])
    else: return None
    '''
    for i in range(N1):
        if pt_in_sphpoly(P1[i], P2): raw_pts.append(P1[i])
    for j in range(N2):
        if pt_in_sphpoly(P2[j], P1): raw_pts.append(P2[j])
    
    #clean data
    N3 = len(raw_pts)
    pts = del_repetition2(raw_pts, 0.0005)
    
    #special case when N4=2 & N3>2, one long arc leads to non-trivial polygon area.
    #POSSIBLE BUGS since I don't have many chances to check this special case.
    N4 = len(pts)
    if N4 < 2: return None
    elif N4 == 2 and N3 == 3: 
        lmax = len_arc(pts[0], pts[1])
        if lmax >= 0.05:
            n = fast_cross(raw_pts[0], raw_pts[1])
            n = n/fast_norm(n)
            #print(fast_dot(n, raw_pts[2]))
            h = fast_dot(n, raw_pts[2])
            if np.abs(h) < tol: return None
            elif h < 0: return np.array(raw_pts[::-1])
            elif h > 0: return np.array(raw_pts)
    elif N4 == 2 and N3 > 3: 
        lmax = len_arc(pts[0], pts[1])
        if lmax >= 0.05:
            #print(raw_pts, pts)
            n01 = fast_cross(pts[0], pts[1])
            n01 = n01/fast_norm(n01)
            gp0 = []
            vgp0 = []
            gp1 = []
            vgp1 = []
            for pt in raw_pts:
                if len_arc(pt, pts[0]) < len_arc(pt, pts[1]): 
                    gp0.append(pt)
                    vgp0.append(fast_dot(pt, n01))
                else: 
                    gp1.append(pt)
                    vgp1.append(fast_dot(pt, n01))
            if len(vgp0) == 0 or len(vgp1) == 0: print('error in the special case of p_intersect')
            vgp0 = np.array(vgp0)
            p0max = np.argmax(vgp0)
            p0min = np.argmin(vgp0)
            h0 = np.abs(p0max) + np.abs(p0min)
            vgp1 = np.array(vgp1)
            p1max = np.argmax(vgp1)
            p1min = np.argmin(vgp1)
            h1 = np.abs(p1max) + np.abs(p1min)
            #print(len_arc(gp1[p1max], gp1[p1min]), vgp1[p1max], vgp1[p1min])
            sorted_pts = []
            if p0max == p0min:
                if h1 < tol: return None
                sorted_pts.append(gp1[p1max])
                sorted_pts.append(gp1[p1min])
                sorted_pts.append(gp0[p0max])
                np1 = fast_cross(gp1[p1max], gp1[p1min])
                np1 = np1/fast_norm(np1)
                if fast_dot(np1, gp0[p0max]) > 0:
                    return np.array(sorted_pts)
                else: return np.array(sorted_pts[::-1])
            elif p1max == p1min:
                if h0 < tol: return None
                sorted_pts.append(gp0[p0max])
                sorted_pts.append(gp0[p0min])
                sorted_pts.append(gp1[p1max])
                np0 = fast_cross(gp0[p0max], gp0[p0min])
                np0 = np0/fast_norm(np0)
                if fast_dot(np0, gp1[p1max]) > 0:
                    return np.array(sorted_pts)
                else: return np.array(sorted_pts[::-1])
            else:
                if h0 < tol and h1 < tol: return None
                if h0 < tol:
                    sorted_pts.append(gp1[p1max])
                    sorted_pts.append(gp1[p1min])
                    sorted_pts.append(gp0[p0max])
                    np1 = fast_cross(gp1[p1max], gp1[p1min])
                    np1 = np1/fast_norm(np1)
                    if np.abs(fast_dot(np1, gp0[p0max]))<tol: print('error2 in the special case')
                    if fast_dot(np1, gp0[p0max]) > 0:
                        return np.array(sorted_pts)
                    else: return np.array(sorted_pts[::-1])
                if h1 < tol:
                    sorted_pts.append(gp0[p0max])
                    sorted_pts.append(gp0[p0min])
                    sorted_pts.append(gp1[p1max])
                    np0 = fast_cross(gp0[p0max], gp0[p0min])
                    np0 = np0/fast_norm(np0)
                    if np.abs(fast_dot(np0, gp1[p1max]))<tol: print('error2 in the special case')
                    if fast_dot(np0, gp1[p1max]) > 0:
                        return np.array(sorted_pts)
                    else: return np.array(sorted_pts[::-1])
                
                np0 = fast_cross(gp0[p0max], gp0[p0min])
                np0 = np0/fast_norm(np0)
                np1 = fast_cross(gp1[p1max], gp1[p1min])
                np1 = np1/fast_norm(np1)
                if fast_dot(np0, gp1[p1max]) > 0:
                    sorted_pts.append(gp0[p0max])
                    sorted_pts.append(gp0[p0min])
                else: 
                    sorted_pts.append(gp0[p0min])
                    sorted_pts.append(gp0[p0max])
                if fast_dot(np1, gp0[p0max]) > 0:
                    sorted_pts.append(gp1[p1max])
                    sorted_pts.append(gp1[p1min])
                else: 
                    sorted_pts.append(gp1[p1min])
                    sorted_pts.append(gp1[p1max])
            #print('noodles polygon', N3, lmax, sorted_pts)
            return np.array(sorted_pts)            
        
    
    #find a normal vector
    #WARNING: This method fails when pts are too closed.(POSSIBLE BUGS)
    n = pts[0]/fast_norm(pts[0])
    proj_pts = [pts[i] - fast_dot(pts[i], n) * n for i in range(len(pts))]
    # rearrange angles from neg to pos, get index
    angles = [np.sign(fast_dot(n , fast_cross(proj_pts[1], proj_pts[i+1]))) * np.arccos(fast_clip(fast_dot(proj_pts[1]/fast_norm(proj_pts[1]) , proj_pts[i+1]/fast_norm(proj_pts[i+1]) )))  for i in range(len(pts)-1)]
    angles = np.array(angles)
    #print(angles)
    sorted_indices = angles.argsort()
    sorted_pts = pts[sorted_indices+1]
    sorted_pts = np.vstack((pts[0], sorted_pts))
    return sorted_pts

def great_cir(p1, p2):
    # return (pi, pj) s.t. piv=pjv on this arc
    # NOT active points
    pi = fast_cross(p1, p2)
    # must normalize since comparison in arc_cir
    pi = pi/fast_norm(pi)
    pj = -pi
    return pi, pj

def arc_cir(p1, p2, pk, pl):
    #NOTICE: pk, pl here are NOT antipodal! and NOT normalized!
    # check whether arc p1->p2 intersects piv=pjv
    if fast_dot(pk-pl, p1) * fast_dot(pk-pl, p2) > 0:
        return None
    # overlap and hence no division, use tol to avoid extreme small det(A)
    elif np.abs(fast_dot(pk-pl, p1)) < tol and np.abs(fast_dot(pk-pl, p2)) < tol:
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
        v = v_raw/fast_norm(v_raw)

        if pt_in_arc(v, p1, p2): 
            p_int = v
        elif pt_in_arc(-v, p1, p2): 
            p_int = -v
        else: 
            print('error in arc_cir: ')

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
        if fast_dot(pt, pi-pj) == 0:
            P_divi.append(pt)
            P_divj.append(pt)
        else:
            if fast_dot(pt, pi-pj) < 0:
                P_divj.append(pt)
            if fast_dot(pt, pi-pj) > 0:
                P_divi.append(pt)
            # n-1 ----> 0
            if idx_pt == len(P_int) - 1:
                idx_pt = -1
            p_int = arc_cir(pt, P_int[idx_pt+1], pi, pj)
            if p_int is not None:
                P_divi.append(p_int)
                P_divj.append(p_int)
    P_divi = del_repetition2(P_divi, 0.0001)
    P_divj = del_repetition2(P_divj, 0.0001)
    
    if len(P_divi) <= 2:
        P_divi = np.array([])
    if len(P_divj) <= 2:
        P_divj = np.array([])
        
    return P_divi, P_divj


def ECT_distance(ECT1, ECT2):
    #original version, to accelarate computation, we use ECT_distance_s and ECT_distance_d
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
            P_divi, P_divj = P_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][1], ECT[idx2][1])
            if len(P_divi) > 2:
                integral_i = int_polygon(P_divi, ECT[idx1][1])
                integral_s += polygon_area(P_divi)
            if len(P_divj) > 2:
                integral_j = int_polygon(P_divj, ECT[idx2][1])
                integral_s += polygon_area(P_divj)
                    
            integral += ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)
    return integral

# integrate ECTs and ECTd, final result should be: ECTs1 + ECTs2 - 2 * ECTd12

def ECT_distance_s(ECT):
    '''
    d_ECT = \int\sum(ECT1-ECT2)^2 
          = \int\sum (ECT1[i]*ECT1[j] + ECT2[i]*ECT2[j] - 2 * ECT1[i] * ECT2[j])
    ECT_s computes the first two terms
    ECT_d computes the last term
    '''
    integral = 0
    
    for idx1 in range(len(ECT)):
        #print(idx1)
        #if idx1 != 208: continue
        for idx2 in range(len(ECT)):
            #if idx2 != 569: continue
            #print(idx1, idx2)
            '''
            # Same active point, no intersections for a single ECT
            if fast_compare(ECT[idx1][1], ECT[idx2][1]) and idx1 != idx2:
                continue
            '''
            integral_i = 0
            integral_j = 0
            integral_s = 0
            if ECT[idx1][3] == ECT[idx2][3] and idx1 == idx2:
                P = np.array(ECT[idx1][2])
                integral_i = int_polygon(P, ECT[idx1][1])
                integral_s = polygon_area(P)
                #convex_checker(P)
            elif ECT[idx1][3] == ECT[idx2][3] and idx1 != idx2: continue
            else:
                P_divi, P_divj = P_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][1], ECT[idx2][1])
                if len(P_divi) > 2:
                    integral_i = int_polygon(P_divi, ECT[idx1][1])
                    integral_s += polygon_area(P_divi)
                    #convex_checker(P_divi)
                if len(P_divj) > 2:
                    integral_j = int_polygon(P_divj, ECT[idx2][1])
                    integral_s += polygon_area(P_divj)
                    #convex_checker(P_divj)
                    
            integral += ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)
            
    return integral

def ECT_distance_d(ECT1, ECT2):
    integral = 0
    
    for idx1 in range(len(ECT1)):
        #print(idx1)
        for idx2 in range(len(ECT2)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            P_divi, P_divj = P_div(ECT1[idx1][2], ECT2[idx2][2], ECT1[idx1][1], ECT2[idx2][1])
            if len(P_divi) > 2:
                integral_i = int_polygon(P_divi, ECT1[idx1][1])
                integral_s += polygon_area(P_divi)
                #convex_checker(P_divi)
            if len(P_divj) > 2:
                integral_j = int_polygon(P_divj, ECT2[idx2][1])
                integral_s += polygon_area(P_divj)
                #convex_checker(P_divj)
            #integral += -1 * ECT1[idx1][0] * ECT2[idx2][0] * (integral_s - integral_i - integral_j)
            integral += ECT1[idx1][0] * ECT2[idx2][0] * (integral_s - integral_i - integral_j)

    return integral


def ECT_distance_partial(ECT1, ECT2):
    #integrate ECTs and ECTd, final result should be: ECTp11 + ECTp22 - 2 * ECTp12
    integral = 0
    
    for idx1 in range(len(ECT1)):
        for idx2 in range(len(ECT2)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            P_divi, P_divj = P_div(ECT1[idx1][2], ECT2[idx2][2], ECT1[idx1][1], ECT2[idx2][1])
            if len(P_divi) > 2:
                integral_i = int_polygon(P_divi, ECT1[idx1][1])
                integral_s += polygon_area(P_divi)
                #convex_checker(P_divi)
            if len(P_divj) > 2:
                integral_j = int_polygon(P_divj, ECT2[idx2][1])
                integral_s += polygon_area(P_divj)
                #convex_checker(P_divj)
            integral += ECT1[idx1][0] * ECT2[idx2][0] * (integral_s - integral_i - integral_j)
            #print(idx1,idx2,ECT1[idx1][0] * ECT2[idx2][0] * (integral_s - integral_i - integral_j),integral)

    return integral


# ##### Auxiliary functions ######

def orientation(ECT):
    #Change orientation and normalized sphpoly
    N = len(ECT)
    for t in range(N):
        n = fast_cross(ECT[t][2][0], ECT[t][2][1])
        # assume the orientation to be P[0]-->P[1]--...-->P[n-1]
        if fast_dot(n, ECT[t][2][2]) < 0:
            #print(t)
            tmp = ECT[t][2].copy()
            reversed_ECT = tmp[::-1]
            ECT[t][2] = reversed_ECT
        M = len(ECT[t][2])
        for i in range(M):
            tmp1 = ECT[t][2][i].copy()
            normed_ECT = tmp1/fast_norm(tmp1)
            ECT[t][2][i] = normed_ECT
    return ECT

def return_ECT(s1):
    tmp=[]
    for key in s1.clean_polygon_gains:
        #TMP: gain for each vertices
        TMP=s1.clean_polygon_gains[key]
        for j in range(TMP.shape[0]):
            megatmp=[]
            megatmp.append(TMP[j])
            megatmp.append(s1.V[key,:])
            poly = s1.polygon_angles[key][s1.clean_polygons[key][j]]
            if len(poly) < 3: continue
            #poly = np.delete(poly, -1, axis = 0)
            megatmp.append(poly)
            megatmp.append(key)
            tmp.append(megatmp)
    ECT1_raw = tmp
    ECT1 = orientation(ECT1_raw)
    return ECT1   

'''
def return_ECT(s1):
    tmp=[]
    for key in s1.clean_polygon_gains:
        #tmp=[]
        TMP=s1.clean_polygon_gains[key]
        for j in range(TMP.shape[0]):
            megatmp=[]
            megatmp.append(TMP[j])
            megatmp.append(s1.V[key,:])
            poly = s1.polygon_angles[key][s1.clean_polygons[key][j]]
            if len(poly) < 3: continue
            #poly = np.delete(poly, -1, axis = 0)
            megatmp.append(poly)            
            tmp.append(megatmp)
    ECT1_raw = tmp
    ECT1 = orientation(ECT1_raw)
    return ECT1

def com_ECT(degrees = 0, v = [0,0,1], file1 = 'octahedron.off', file2 = 'octahedron.off'):
    #s1=ShapeReader.shape_from_file('H16_sas_aligned.off')
    #s2=ShapeReader.shape_from_file('P30_sas_aligned.off')
    s1=ShapeReader.shape_from_file(file1)
    s2=ShapeReader.shape_from_file(file2)
    #s2=ShapeReader.shape_from_file('small_teeth.off')
    
    #we need this
    #############################################################
    s1.V = s1.V-np.mean(s1.V,0)
    scales = [sum(tmp**2)**(0.5) for tmp in s1.V]
    s1.V = s1.V/max(scales)
    s2.V = s2.V-np.mean(s2.V,0)
    scales2 = [sum(tmp**2)**(0.5) for tmp in s2.V]
    s2.V = s2.V/max(scales2)
    for i in range(s2.V.shape[0]):
        s2.V[i,:] = rotate_axis(s2.V[i,:], v, degrees)
    ############################################################# 
    
    s1.prepare()
    s1.compute_links()
    s1.compute_polygons()
    s1.compute_gains2()
    s1.clean_gains2()
    s2.prepare()
    s2.compute_links()
    s2.compute_polygons()
    s2.compute_gains2()
    s2.clean_gains2()

    ECT1 = return_ECT(s1)
    ECT2 = return_ECT(s2)

    return ECT_distance_s(ECT1) + ECT_distance_s(ECT2) + 2 * ECT_distance_d(ECT1,ECT2)
'''


def rotate_axis(vector, axis, angle):
    """
    angle: in degree
    """
    axis = axis / np.linalg.norm(axis)
    angle = angle/360 * 2 * np.pi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * np.outer(axis, axis)
    
    return np.dot(rotation_matrix, vector)

def convex_checker(P):
    #check if orientations are good
    t = 0.0000001
    N = len(P)
    n0 = fast_cross(P[0], P[1])
    norm_n0 = fast_norm(n0)
    if norm_n0 < t: print('strange arc in convex_checker')
    n0 = n0/norm_n0
    ori = 1
    ori2 = fast_dot(n0, P[2])
    if ori2 < -t: print('wrong orientation in convex_checker, ori2:', ori2)
    '''
    #Since some of the polygons have opposite orientations
    if np.dot(n0, P[2]) < 0:
        ori = -1
    '''
    for i in range(N):
        n = fast_cross(P[i], P[(i+1)%N])
        n = n/fast_norm(n)
        for j in range(N):
            value = fast_dot(n, P[j])
            if value < -t: 
                print('wrong orientation in convex_checker', value)
