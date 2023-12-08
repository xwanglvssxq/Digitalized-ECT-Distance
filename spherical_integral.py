import numpy as np
import math
from scipy import integrate
from detect_rotation import roted_sphere

def cartesian_to_spherical(x, y, z):
    tau = math.asin(z)
    phi = math.atan2(y, x)
    if math.atan2(y, x) < 0:
        phi += 2 * np.pi
    return phi, tau

def solve_great_circle(phi_1, tau_1, phi_2, tau_2):
    tol = 0.000001

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

def int_tri(p1, p2, p3, pi):
    p1, p2, p3, pi = roted_sphere(p1, p2, p3, pi)
    phi_1, tau_1 = cartesian_to_spherical(p1[0], p1[1], p1[2])
    phi_2, tau_2 = cartesian_to_spherical(p2[0], p2[1], p2[2])
    phi_3, tau_3 = cartesian_to_spherical(p3[0], p3[1], p3[2])
    phi_i, tau_i = cartesian_to_spherical(pi[0], pi[1], pi[2])

    I12 = int_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i)
    I23 = int_arc(phi_2, tau_2, phi_3, tau_3, phi_i, tau_i)
    I31 = int_arc(phi_3, tau_3, phi_1, tau_1, phi_i, tau_i)

    return I12 + I23 + I31

if __name__ == "__main__":
    p1 = np.array([0.182,-0.69,-0.69])
    p2 = np.array([-0.69,0.182,-0.69])
    p3 = np.array([-0.69,-0.69,0.182])
    p4 = np.array([0.577,0.577,0.577])
    p5 = np.array([1,0,0])
    p6 = np.array([0,1,0])
    p7 = np.array([0,0,1])

    #print(int_tri(p1, p3, p2, p4)) 
    #correct -0.9210862748964177

    #print(int_tri(p1, p4, p3, p6))
    #correct -1.959946366700603

    #print(int_tri(p1, p2, p4, p7))
    #correct -1.9582264989128386

    #print(int_tri(p2, p3, p4, p5))
    #correct -1.9640991162297654