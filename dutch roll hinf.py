# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:21:30 2020

@author: kemper
Dutch roll Hinf and H2 control example
"""
import numpy as np
import matplotlib.pyplot as plt
from control import tf, ss, hinfsyn, h2syn, step_response, augw, minreal

def triv_sigma(g, w):
    """triv_sigma(g,w) -> s
    g - LTI object, order n
    w - frequencies, length m
    s - (m,n) array of singular values of g(1j*w)"""
    m, p, _ = g.freqresp(w)
    sjw = (m*np.exp(1j*p*np.pi/180)).transpose(2, 0, 1)
    sv = np.linalg.svd(sjw, compute_uv=False)
    return sv
w = np.logspace(-3, 3, 101)
def weighting(wb, m, a):
    """weighting(wb,m,a) -> wf
    wb - design frequency (where |wf| is approximately 1)
    m - high frequency gain of 1/wf; should be > 1
    a - low frequency gain of 1/wf; should be < 1
    wf - SISO LTI object
    """
    s = tf([1, 0], [1])
    return (s/m + wb) / (s + wb*a)

# Lateral stability derivatives and aircraft parameters
Y_b = 43.72      # ft/s^2
Y_r = 0.0
N_b = 4.395      # 1/s^2
N_r = -0.744     # 1/s
Y_dr = 12.17     # ft/s^2
N_dr = -4.495    # 1/s^2
N_da = -0.217     # 1/s^2
u_0 = 176.0        # ft/s

# Aircraft model 
A = [[Y_b/u_0, -(1.0-Y_r/u_0)],[N_b, N_r]]  # x1 = beta, x2 = r
B = [[0.0,Y_dr/u_0],[N_da,N_dr]]         #u1 = delta_a, u2 = delta_r
C = [[1.0,0.0],[0.0,1.0]]                               # y1 = beta, y2 = r
D = [[0.0,0.0],[0.0,0.0]]
G = ss(A,B,C,D)

#weighting function
w1 = weighting(2, 2, 0.005)
w1 = ss(w1) #needs to be converted for append
wp = w1.append(w1)
wu = ss([], [], [], np.eye(2)) #unwieghted actuator authority size of 2

#Augmented plant need to make by hand for hinf syn
I = ss([], [], [], np.eye(2))
P = augw(G,wp,wu) #generalized plant P
P = minreal(P)
wps = tf(wp)
Gs= tf(G)
Is= tf(I)

# correct generalized plant
p11 = wps
p12 = wps*Gs
p21 = -Is
p22 = -Gs


K2 = h2syn(P, 2, 2)
# H INF CONTROLLER
K, CL, gam, rcond = hinfsyn(P,2,2) #generalized plant incorrect so doing mixsyn
print(gam)

#open loop plot
openloop = triv_sigma(G*K, w)
openloop2 = triv_sigma(G*K2, w)
plt.figure(1)
plt.semilogx(w, 20*np.log10(openloop[:, 0]), label=r'G*K')
plt.semilogx(w, 20*np.log10(openloop2[:, 0]), label=r'G*K2')
plt.ylim([-50, 50])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.title('open loop')
plt.grid(1)
plt.legend()

# closed loop tf
I = ss([], [], [], np.eye(2))
s1 = I.feedback(G*K)
s1 = minreal(s1)
s2 = I.feedback(G*K2)
s2 = minreal(s2)
invwp = I.feedback(wp)
t1 = (G*K).feedback(I)
t1 = minreal(t1)
t2 = (G*K2).feedback(I)
t2 = minreal(t2)

# frequency response
sv1 = triv_sigma(s1, w)
sv2 = triv_sigma(s2, w)
iwp = triv_sigma(invwp, w)

plt.figure(2)
plt.semilogx(w, 20*np.log10(sv1[:, 0]), label=r'$\sigma_1(S_1)$')
plt.semilogx(w, 20*np.log10(sv2[:, 0]), label=r'$\sigma_1(S_2)$')
plt.semilogx(w, 20*np.log10(iwp[:, 0]), label=r'invWp')
plt.ylim([-60, 10])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.legend()
plt.title('Singular values of S')

#step responses
time = np.linspace(0, 10, 301)
_,y1 = step_response(t1, time,input=0)
_,y2 = step_response(t1, time,input=1)

plt.figure(3)
plt.subplot(1, 2, 1)
plt.plot(time, y1[0], label='in 1 $y_1(t))$')
plt.plot(time, y2[1], label='in 2 $y_2(t))$')
plt.xlabel('time [s]')
plt.ylabel('response [1]')
plt.legend()
plt.title('step response hinf')

_,y1 = step_response(t2, time,input=0)
_,y2 = step_response(t2, time,input=1)
plt.subplot(1, 2, 2)
plt.plot(time, y1[0], label='in 1 $y_1(t))$')
plt.plot(time, y2[1], label='in 2 $y_2(t))$')
plt.xlabel('time [s]')
plt.ylabel('response [1]')
plt.legend()
plt.title('step response h2')

#disturbance step responses
time = np.linspace(0, 10, 301)
_,y1 = step_response(s1, time,input=0)
_,y2 = step_response(s1, time,input=1)

plt.figure(4)
plt.subplot(1, 2, 1)
plt.plot(time, y1[0], label='dist 1 $y_1(t))$')
plt.plot(time, y2[1], label='dist 2 $y_2(t))$')
plt.xlabel('time [s]')
plt.ylabel('response [1]')
plt.legend()
plt.title('step response hinf')

_,y1 = step_response(s2, time,input=0)
_,y2 = step_response(s2, time,input=1)
plt.subplot(1, 2, 2)
plt.plot(time, y1[0], label='dist 1 $y_1(t))$')
plt.plot(time, y2[1], label='dist 2 $y_2(t))$')
plt.xlabel('time [s]')
plt.ylabel('response [1]')
plt.legend()
plt.title('step response h2')