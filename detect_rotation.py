from scipy.stats import special_ortho_group
import numpy as np
import math

def detect_rot(p1, p2, p3, pi):
    # Here p1, p2, p3, pi are the Cartesian coordinates
    # Set tolerence
    tol = 0.0001

    # Detect poles
    if np.abs(p1[2]) > 1-tol: return False
    if np.abs(p2[2]) > 1-tol: return False
    if np.abs(p3[2]) > 1-tol: return False
    if np.abs(pi[2]) > 1-tol: return False

    # Detect phi=0 arc, i.e.
    if p1[0] > 0 and np.abs(p1[1]) == 0: return False
    if p2[0] > 0 and np.abs(p2[1]) == 0: return False
    if p3[0] > 0 and np.abs(p3[1]) == 0: return False
    if pi[0] > 0 and np.abs(pi[1]) == 0: return False

    # Detect meridian
    if np.abs(p1[0] * p2[1] - p2[0] * p1[1]) == 0: return False
    if np.abs(p2[0] * p3[1] - p3[0] * p2[1]) == 0: return False
    if np.abs(p3[0] * p1[1] - p1[0] * p3[1]) == 0: return False

    # Detect equator
    count_zeros = np.count_nonzero([p1[2], p2[2], p3[2]] == 0)
    if np.size(count_zeros) > 1: return False

    # Detect crossing arc
    phi1 = math.atan2(p1[1], p1[0])
    if phi1 < 0:
        phi1 += 2 * np.pi
    phi2 = math.atan2(p2[1], p2[0])
    if phi2 < 0:
        phi2 += 2 * np.pi
    phi3 = math.atan2(p3[1], p3[0])
    if phi3 < 0:
        phi3 += 2 * np.pi
    if np.abs(phi1-phi2) > np.pi or np.abs(phi2-phi3) > np.pi or np.abs(phi1-phi3) > np.pi: return False

    return True

def roted_sphere(p1, p2, p3, pi):
    truth_value = detect_rot(p1, p2, p3, pi)
    while truth_value == False:
        #print('rotate')
        rotation_matrix = special_ortho_group.rvs(3)
        p1 = np.dot(rotation_matrix, p1)
        p2 = np.dot(rotation_matrix, p2)
        p3 = np.dot(rotation_matrix, p3)
        pi = np.dot(rotation_matrix, pi)
        truth_value = detect_rot(p1, p2, p3, pi)
    return p1, p2, p3, pi

if __name__ == "__main__":
    p1 = np.array([0.182,-0.69,-0.69])
    p2 = np.array([-0.69,0.182,-0.69])
    p3 = np.array([-0.69,-0.69,0.182])
    p4 = np.array([0.577,0.577,0.577])
    p5 = np.array([1,0,0])
    p6 = np.array([0,1,0])
    p7 = np.array([0,0,1])

    pp1, pp2, pp3, ppi = roted_sphere(p1, p3, p4, p6)
    print(pp1, pp2, pp3, ppi)



        