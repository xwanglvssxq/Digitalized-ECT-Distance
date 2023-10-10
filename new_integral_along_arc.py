import numpy as np
from scipy import integrate
from new_solve_great_circle import solve_great_circle

def int_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i):
    coordinate, case = solve_great_circle(phi_1, tau_1, phi_2, tau_2)
    if case == 'case_1':
        return 0
    
    if case == 'case_3':
        if np.abs(phi_2 - phi_1) < np.pi/2:
            return np.sin(tau_i) * (phi_2 - phi_1)
        elif phi_1 < phi_2:
            return np.sin(tau_i) * (0 - phi_1 + phi_2 - 2 * np.pi)
        else: return np.sin(tau_i) * (2* np.pi - phi_1 + phi_2 - 0)


    if case == 'case_main':
        x, y, z, a, phi_0 = coordinate
        def f1(phi):
            return (1 - np.power(a,2) * np.power(np.cos(phi-phi_0),2))/(1 + np.power(a,2) * np.power(np.cos(phi-phi_0),2))
        def f2(phi):
            return (2 * a * np.cos(phi-phi_0) * np.cos(phi-phi_i))/(1 + np.power(a,2) * np.power(np.cos(phi-phi_0),2))
        def f3(phi):
            return np.arctan(a * np.cos(phi-phi_0)) * np.cos(phi-phi_i)
        
        # MAYBE wrong: need to compute small arc
        #print(phi_2-phi_1)
        if np.abs(phi_2 - phi_1) < np.pi:
            #print('here!!!')
            I1, error1 = integrate.quad(f1, phi_1, phi_2)
            I2, error2 = integrate.quad(f2, phi_1, phi_2)
            I3, error3 = integrate.quad(f3, phi_1, phi_2)
        elif phi_1 < phi_2:
            #print('here~~~')
            #print(phi_1, phi_2)
            Int11, error11 = integrate.quad(f1, phi_1, 0)
            Int12, error12 = integrate.quad(f1, 2 * np.pi, phi_2)
            Int21, error21 = integrate.quad(f2, phi_1, 0)
            Int22, error22 = integrate.quad(f2, 2 * np.pi, phi_2)
            Int31, error31 = integrate.quad(f3, phi_1, 0)
            Int32, error32 = integrate.quad(f3, 2 * np.pi, phi_2)
            I1 = Int11 + Int12
            I2 = Int21 + Int22
            I3 = Int31 + Int32
        elif phi_1 > phi_2:
            #print('here!!')
            Int11, error11 = integrate.quad(f1, phi_1, 2 * np.pi)
            Int12, error12 = integrate.quad(f1, 0, phi_2)
            Int21, error21 = integrate.quad(f2, phi_1, 2 * np.pi)
            Int22, error22 = integrate.quad(f2, 0, phi_2)
            Int31, error31 = integrate.quad(f3, phi_1, 2 * np.pi)
            Int32, error32 = integrate.quad(f3, 0, phi_2)
            I1 = Int11 + Int12
            I2 = Int21 + Int22
            I3 = Int31 + Int32
        

        integral = 0.25 * np.sin(tau_i) * I1 - 0.25 * np.cos(tau_i) * I2 -0.5 * np.cos(tau_i) * I3
        #print(integral, 0.25 * np.sin(tau_i) * I1)
        return integral



if __name__ == "__main__":
    step = 300
    
    # p_ij are different from p_j
    phi_i1 = 0
    tau_i1 = 0
    phi_i2 = np.pi/2
    tau_i2 = 0
    phi_i3 = 0
    tau_i3 = np.pi/2
    phi_i4 = np.pi/4
    tau_i4 = np.arctan(1/(2**(1/2)))
    
    phi_1 = 4.97
    tau_1 = -0.761
    phi_2 = 2.884
    tau_2 = -0.761
    phi_3 = 3.927
    tau_3 = 0.183
    phi_4 = np.pi/4
    tau_4 = np.arctan(1/(2**(1/2)))

    
    I41 = int_arc(phi_1, tau_1, phi_3, tau_3, phi_i4, tau_i4)
    I42 = int_arc(phi_2, tau_2, phi_1, tau_1, phi_i4, tau_i4)
    I43 = int_arc(phi_3, tau_3, phi_2, tau_2, phi_i4, tau_i4)
    #print(I41, I42, I43, I41+I42+I43)
    #0.11122934170730314 -1.1433179798740425 0.11123276224602342 -0.9208558759207159
    

    
    I21 = int_arc(phi_3, tau_3, phi_1, tau_1, phi_i2, tau_i2)
    # normal type
    I22 = int_arc(phi_1, tau_1, phi_4, tau_4, phi_i2, tau_i2)
    # phi_1 to 2*pi, 0 to phi_2
    I23 = int_arc(phi_4 + 0.05, tau_4, phi_3, tau_3, phi_i2, tau_i2)
    #print(I23)
    #print(I21,I22,I23,I21+I22+I23)
    # [a phi_0 correct] 
    # -0.33845278785651933 -0.5079213087031045 -1.089159324386236 -1.93553342094586
    # +0.05: -1.09 ;     -0.05: +1.09
    # +0.05 is better, exchange coordinate sign also change, avoid "0.01" threshold, 
    # all [case_main-here] checked, correct
    # the -1.94 can be trusted
    
    

    '''
    I31 = int_arc(phi_4, tau_4, phi_1, tau_1, phi_i3, tau_i3)
    I32 = int_arc(phi_1, tau_1, phi_2, tau_2, phi_i3, tau_i3)
    I33 = int_arc(phi_2, tau_2, phi_4, tau_4, phi_i3, tau_i3)
    #print(I32)
    print(I31,I32,I33,I31+I32+I33)
    # -0.3077988765215084 0.20552553357088305 -0.3078005587607291 -0.41007390171135444 (too small?)
    # only I1 term due to pi
    # continuous around pi
    # here~~~ checked, negative to exchange case, I think it's ok
    '''
    
    

    
    I11 = int_arc(phi_2, tau_2, phi_3, tau_3, phi_i1, tau_i1)
    I12 = int_arc(phi_3, tau_3, phi_4+0.05, tau_4, phi_i1, tau_i1)
    I13 = int_arc(phi_4, tau_4, phi_2, tau_2, phi_i1, tau_i1)
    #print(I12)
    #print(I11, I12, I13, I11+I12+I13)
    # I12 = I21(exchange 4,3 I mean) WRONG ##### I21 = I11; I22 = I13?????? make sense: (same: tau_i = 0)
    # phenomenon same as I2x
    # +0.05: a= -17.85059291970957  +  here!!!  +  I12=-1.1280676063526145
    # phi_4+0.05: -0.33845249637351105 -1.1280676063526145 -0.5079309700646228 -1.9744510727907483
    # this -1.97 can also be trusted
    
    

    '''
    # TEST_1
    I31 = int_arc(phi_i1, tau_i1, phi_i2, tau_i2-0.011, phi_i3, tau_i3)
    I32 = int_arc(phi_i2, tau_i2-0.011, phi_i4, tau_i3-0.011, phi_i3, tau_i3)
    I33 = int_arc(phi_i4, tau_i3-0.011, phi_i1, tau_i1, phi_i3, tau_i3)
    #print(I32)
    print(I31,I32,I33,I31+I32+I33)
    # a= -0.011000443688141185
    # a= 128.57068472937138
    # a= 128.559683815087
    # 0.3926515655889622 0.19022850986307785 0.19027076521270375 0.7731508406647438 == pi/4 
    # so WHY???
    '''

    '''
    # TEST_2
    I31 = int_arc(phi_i1, tau_i1, phi_i2, tau_i2-0.011, phi_i3, tau_i3)
    I32 = int_arc(phi_i2, tau_i2-0.011, phi_i4, tau_i4, phi_i3, tau_i3)
    I33 = int_arc(phi_i4, tau_i4, phi_i1, tau_i1, phi_i3, tau_i3)
    #print(I32)
    print(I31,I32,I33,I31+I32+I33)
    #a= -0.011000443688141185
    #a= 1.0110602884590711
    #a= 0.9999999999999997
    #0.3926515655889622 -0.14228553176395772 -0.14140588857865793 0.10896014524634651 = 0.092(by hand)
    # still correct, and it's an estimate of the positive part of my desired integral, very small make sense.
    '''

    '''
    # TEST_3(minus to get the correct orientation)
    I31 = -int_arc(phi_i1, tau_i1, phi_i2, tau_i2-0.011, phi_i3, tau_i3)
    I32 = -int_arc(phi_i2, tau_i2-0.011, phi_i4, -tau_i3 + 0.02, phi_i3, tau_i3)
    I33 = -int_arc(phi_i4, -tau_i3 + 0.02, phi_i1, tau_i1, phi_i3, tau_i3)
    #print(I32)
    print(I31,I32,I33,I31+I32+I33)
    #-0.3926515655889622 -0.1854180394958804 -0.18534196330374603 -0.7634115683885887 == -pi/4(by hands)
    # HERE COMES the problem!! This is just part of the negative term, has become large enough!!
    '''

    
    # TEST_4
    I31d = int_arc(phi_i2, tau_i2, phi_i1, tau_i1-0.011, phi_i3, tau_i3)
    I32d = int_arc(phi_i1, tau_i1-0.011, phi_2, tau_2, phi_i3, tau_i3)
    I33d = int_arc(phi_2, tau_2, phi_i2, tau_i2, phi_i3, tau_i3)
    #print(I32d)
    #print(I31d,I32d,I33d,I31d+I32d+I33d)
    # -0.9756472610170595
    Id = I31d+I32d+I33d

    I31a = int_arc(phi_i1, tau_i1, phi_1, tau_1, phi_i3, tau_i3)
    I32a = int_arc(phi_1, tau_1, np.pi+0.011, -0.93, phi_i3, tau_i3)
    I33a = int_arc(np.pi+0.011, -0.93, phi_i1, tau_i1, phi_i3, tau_i3)
    #print(I31a,I32a,I33a,I31a+I32a+I33a)
    # -0.6831352449660608
    Ia = I31a+I32a+I33a

    I31b = int_arc(phi_i1, tau_i1, np.pi - 0.011, -0.93, phi_i3, tau_i3)
    I32b = int_arc(np.pi - 0.011, -0.93, phi_2, tau_2, phi_i3, tau_i3)
    I33b = int_arc(phi_2, tau_2, phi_i1, tau_i1, phi_i3, tau_i3)
    #print(I31b,I32b,I33b,I31b+I32b+I33b)
    # -0.30345228299162985
    Ib = I31b+I32b+I33b

    I3 = Ia+Ib+Id+0.1089

    #print(I3)
    # -1.8672734706235854

    
    I = 4 * np.pi - I11 - I12 - I13 - I21 - I22 - I23 - I3 - I41 - I42 - I43
    print(I/(8 * np.pi))
    
    

    '''
    TEST_WRONG!
    I31c = int_arc(phi_i1, tau_i1, phi_1, tau_1, phi_i3, tau_i3, step)
    I32c = int_arc(phi_1, tau_1, phi_2, tau_2, phi_i3, tau_i3, step)
    I33c = int_arc(phi_2, tau_2, phi_i1, tau_i1, phi_i3, tau_i3, step)
    print(I31c,I32c,I33c,I31c+I32c+I33c)
    # 0.4581538613311002 SO WRONG!!!!!!
    '''

    
    
    

    
    

    '''
    # test: \dot p_i must == 0, good sample to test, no! wrong!
    phi_1 = 0
    tau_1 = -0.785
    phi_2 = 5.498
    tau_2 = 0
    phi_4 = np.pi/4
    tau_4 = np.arctan(1/(2**(1/2)))
    I42 = int_arc(phi_2, tau_2, phi_1, tau_1, phi_4, tau_4, step)
    print("result=", I42)
    #new test: now the result is correct, p1top2 = -p2top1 now
    '''
    
    
    '''
    #test 2, around a vertical pt, seems correct
    eps = 0.05
    phi_1 = 0 + eps
    tau_1 = -0.785
    phi_2 = 0 - eps
    tau_2 = -0.785
    phi_3 = 0
    tau_3 = -0.785 - eps

    phi_i = np.pi/4
    tau_i = np.arctan(1/(2**(1/2)))

    I41 = int_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i, step)
    I42 = int_arc(phi_2, tau_2, phi_3, tau_3, phi_i, tau_i, step)
    I43 = int_arc(phi_3, tau_3, phi_1, tau_1, phi_i, tau_i, step)
    print(I41, I42, I43, I41+I42+I43)
    #-0.08163265095839287 0.040280391589358075 0.041317737950963496 -3.4521418071295495e-05(very small)
    '''

    '''
    phi_1 = 4.97
    tau_1 = -0.761
    phi_2 = 2.884
    tau_2 = -0.761
    phi_3 = 3.927
    tau_3 = 0.183
    
    phi_i = np.pi/4
    tau_i = np.arctan(1/(2**(1/2)))

    I41 = int_arc(phi_1, tau_1, phi_3, tau_3, phi_i, tau_i, step)
    I42 = int_arc(phi_2, tau_2, phi_1, tau_1, phi_i, tau_i, step)
    I43 = int_arc(phi_3, tau_3, phi_2, tau_2, phi_i, tau_i, step)
    print(I41, I42, I43, I41+I42+I43)
    # arc correct, a,phi_0 correct, [case_main--phi2-phi1<pi for all three]
    # 0.11122934170730314 -1.1433179798740425 0.11123276224602342 -0.9208558759207159
    # maybe I have found the correct result
    '''