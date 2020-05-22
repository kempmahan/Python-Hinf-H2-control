# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:31:33 2020

@author: kemper
 Pitch rate Mixed syn example
"""

import numpy as np
import matplotlib.pyplot as plt
import control as con 

#%% Hinf mixed sen
# equivalent to Matlab's sigma function, so use a trivial stand-in.
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

# Longitudinal stability and control derivatives 
Za = -140.22
Zq = -1.456
Ma = -2.01
Mq = -0.35
Zde = -25.9
Mde = -5.0

A = [[Za, Zq],[Ma, Mq]]
B = [[Zde],[Mde]]
C = [0.0, 1.0]
D = 0.0
G = con.StateSpace(A,B,C,D)

# Weighting functions
w0 = 8.0                        # Desired closed-loop bandwidth
A = 1/100                      # Desired disturbance attenuation inside bandwidth
M = 1.5                        # Desired bound on hinfnorm(S) & hinfnorm(T)

w1 = con.TransferFunction([1/M, w0],[1.0, w0*A])  #sensitivity weight
w2 = con.TransferFunction([1.0, 0.1],[1,100])   # K*S weight
# w3 = []  #empty control weight on T
k,cl,info = con.mixsyn(G,w1,w2,w3=None)
print(info[0]) #gamma
L = G*k
Ltf = con.ss2tf(L)
Ltf = con.minreal(Ltf)
S = 1.0/(1.0+Ltf)
T = 1-S 
plt.figure(1)

sw1 = triv_sigma(1.0/w1, w)
sigS = triv_sigma(S, w)
plt.semilogx(w, 20*np.log10(sw1[:, 0]), label=r'$\sigma_1(1/W1)$')
plt.semilogx(w, 20*np.log10(sigS[:, 0]), label=r'$\sigma_1(S)$')
plt.ylim([-60, 10])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.legend()
plt.title('Singular values of S')
plt.grid(1)

plt.figure(2)
sw2 = triv_sigma(1/w2, w)
ks = triv_sigma(k*S, w)
plt.semilogx(w, 20*np.log10(sw2[:, 0]), label=r'$\sigma_1(1/W2)$')
plt.semilogx(w, 20*np.log10(ks[:, 0]), label=r'$\sigma_1(K*S)$')
plt.ylim([-30, 60])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.legend()
plt.title('Singular values of K*S')
plt.grid(1)

plt.figure(3)
t,y = con.step_response(T)
plt.plot(t,y)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Step Response')

# h = con.feedback(L)
# t,y = con.step_response(h)
# r = 1/ w1
