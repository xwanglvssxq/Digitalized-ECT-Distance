import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere2car(phi, tau):
    x = np.cos(tau) * np.cos(phi)
    y = np.cos(tau) * np.sin(phi)
    z = np.sin(tau)
    return x, y, z

def solve_great_circle(phi_1, tau_1, phi_2, tau_2):
    '''
    This parateterization DOES NOT work for: equator, all meridians
    One way to simplify is to estimate equator and meridians by a very closed great circle
    '''
    
    # ONLY useful when plotting, NOTHING to do with 'a' and 'phi_0'
    step = 300    

    # when longtitude and equator: if tau_1 = (+ or -)pi/2, phi_ + eps; else phi_1 + eps
    # WARNING: need to SAME hemisphere!!! ####################################
    eps = 0.03    

    tol = 0.01
    
    # coincide
    if phi_1 == phi_2 and tau_1 == tau_2:              
        phi = phi_1 * np.ones(step)
        tau = tau_1 * np.ones(step)
        x, y, z = sphere2car(phi, tau)
        print("two poingts coincide")
        return [x, y, z], 'case_1'

    # antipodal
    elif np.abs(phi_1-phi_2)/np.pi == 1 and np.abs(tau_1-tau_2)/np.pi == 1:
        print("undertermined great circle")            
        return [0, 0, 0], 'case_0'   

    # ONLY one pole, assume p = (0, pi/2)
    elif np.abs(tau_1) == np.pi/2:                     
        if tau_1 > 0: 
            tau_1 -= eps/2                                 # eps here!!
            phi_1 += eps/2
        else:
            tau_1 += eps/2
            phi_1 -= eps/2
        
    elif np.abs(np.abs(tau_2) - np.pi/2) < tol:
        print('here, longtitude!')
        if tau_2 > 0: 
            tau_2 -= eps/2                                 # eps here!!
            phi_2 += eps/2
        else:
            tau_2 += eps/2
            phi_2 -= eps/2
    
    # longtitude: phi_1 = phi_2
    elif np.abs(phi_1 - phi_2) < tol:                               
        phi_1 += eps                                        # eps here!!!
    
    # longtitude: |phi_1-phi_2| = pi
    elif np.abs(np.abs(phi_1-phi_2)/np.pi - 1) < tol: 
        phi_1 += eps                                        # eps here!!!

    elif tau_1 == 0 and tau_2 == 0:
        phi = np.linspace(phi_1, phi_2, step)
        tau = np.zeros(step)
        x, y, z = sphere2car(phi, tau)                 # equator
        return [x, y, z], 'case_3'
                

    # compute phi_0, a
    r1 = np.cos(phi_1) * np.tan(tau_2) - np.cos(phi_2) * np.tan(tau_1)
    r2 = -np.sin(phi_1) * np.tan(tau_2) + np.sin(phi_2) * np.tan(tau_1)

    #if r2 = 0:
    if np.abs(r2) <= 0.01:                             # POSSIBLE MISTAKE HERE
        phi_0 = np.pi/2
        # phi_0 can be -pi/2 with 'a' becomes '-a', then we get the same equation                                
    else: 
        phi_0 = np.arctan(r1/r2)

    # if np.abs(phi_1 - phi_0) == np.pi/2:
    #print((phi_2 - phi_0)/np.pi)
    if np.abs(np.abs((phi_1 - phi_0)%np.pi) - np.pi/2) < tol:       # cos = 0, use another point, it can't be antipodal
        #print('here')                                               ##########POSSIBLE MISTAKE HERE
        a = np.tan(tau_2)/np.cos(phi_2 - phi_0)
    else:
        #print('hereeeeeee')
        #print(np.cos(phi_1 - phi_0))
        a = np.tan(tau_1)/np.cos(phi_1 - phi_0)
    print('a=',a, 'phi_0=', phi_0)

    phi = np.linspace(phi_1, phi_2, step)
    tau = np.arctan(a * np.cos(phi-phi_0))
    x, y, z = sphere2car(phi, tau)
    return [x, y, z, a, phi_0], 'case_main'

if __name__ == "__main__":
    '''
    #very strange
    phi_2 = 0
    tau_2 = -0.785
    phi_1 = 5.498
    tau_1 = 0
    step = 300
    #-0.0 0.7856110196153104
    # HAVE corrected
    '''
    

    '''
    phi_1 = 0
    tau_1 = -0.785
    phi_2 = 5.498
    tau_2 = 0
    step = 300
    #-1.4133887149765751 0.7856110196153104
    '''

    
    # this is a longtitude!
    phi_2 = 3.927
    tau_2 = 0.183
    phi_1 = np.pi/4
    tau_1 = np.arctan(1/(2**(1/2)))
    step = 500
    # eps = 0.03, a = 29.73
    

    '''
    # test on another longtitue
    phi_1 = 0
    tau_1 = -np.pi/2
    phi_2 = np.pi/4
    tau_2 = np.arctan(1/(2**(1/2)))
    step = 300
    # eps = 0.03, a = 94.99
    '''

    '''
    phi_1 = 2.884
    tau_1 = -0.761
    phi_2 = 3.927
    tau_2 = 0.183
    step = 300
    #chech on (-0.69, 0.183): a, phi_0 correct
    '''

    '''
    phi_1 = 4.97
    tau_1 = -0.761
    phi_2 = np.pi/4
    tau_2 = np.arctan(1/(2**(1/2)))
    step = 300
    '''

    
    phi_2 = 4.97
    tau_2 = -0.761
    phi_1 = 2.884
    tau_1 = -0.761
    step = 300
    #chech on (-0.69, 0.183): a, phi_0 correct
    
    
    

    coordinate, case = solve_great_circle(phi_1, tau_1, phi_2, tau_2)
    if case == 'case_main':
        x, y, z, a, phi_0 = coordinate
        print(a, phi_0)
    else:
        x, y, z = coordinate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # range of axis
    ax.set_xlim([-1, 1])  
    ax.set_ylim([-1, 1])  
    ax.set_zlim([-1, 1])  

    ax.set_title('Graph of tan(tau) = cos(phi)')
    plt.grid(True)
    plt.show()
