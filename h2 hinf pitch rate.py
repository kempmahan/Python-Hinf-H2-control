

import numpy as np
import matplotlib.pyplot as plt
import control as con  

#%% Hinf and H2
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

# Controller synthesis
wb = 8                         # Desired closed-loop bandwidth
A = 1/100                      # Desired disturbance attenuation inside bandwidth
M = 1.5                        # Desired bound on hinfnorm(S) & hinfnorm(T)
w1 = con.TransferFunction([1/M, wb],[1.0, wb*A])  #tracking/disturbance weight
w2 = con.TransferFunction([1.0, 0.1],[1,100])     #K*S (actuator authority) weight

# Augmented plant
P = con.augw(G,w1,w2) #generalized plant P
P = con.minreal(P)

# H 2 CONTROLLER
K2 = con.h2syn(P, 1, 1)

L = G*K2
Ltf = con.ss2tf(L)
So2 = 1.0/(1.0+Ltf)
So2 = con.minreal(So2)
To2 = G*K2*So2 
To2 = con.minreal(To2)

# H INF CONTROLLER
K, CL, gam, rcond = con.hinfsyn(P,1,1)
print(gam)

L = G*K
Ltf = con.ss2tf(L)
So = 1.0/(1.0+Ltf)
So = con.minreal(So)
To = G*K*So 
To = con.minreal(To)

plt.figure(1)

sw1 = triv_sigma(1.0/w1, w)
sigS = triv_sigma(So, w)
sigS2 = triv_sigma(So2, w)
plt.semilogx(w, 20*np.log10(sw1[:, 0]), label=r'$\sigma_1(1/W1)$')
plt.semilogx(w, 20*np.log10(sigS[:, 0]), label=r'$\sigma_1(So,Hinf)$')
plt.semilogx(w, 20*np.log10(sigS2[:, 0]), label=r'$\sigma_1(So,H2)$')
plt.ylim([-50, 10])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.title('Singular values of S0')
plt.grid(1)
plt.legend()

plt.figure(2)
sw2 = triv_sigma(1/w2, w)
ks = triv_sigma(K*So, w)
ks2 = triv_sigma(K2*So2, w)
plt.semilogx(w, 20*np.log10(sw2[:, 0]), label=r'$\sigma_1(1/W2)$')
plt.semilogx(w, 20*np.log10(ks[:, 0]), label=r'$\sigma_1(K*S, hinf)$')
plt.semilogx(w, 20*np.log10(ks2[:, 0]), label=r'$\sigma_1(K*S, h2)$')
plt.ylim([-50, 60])
plt.ylabel('magnitude [dB]')
plt.xlim([1e-3, 1e3])
plt.xlabel('freq [rad/s]')
plt.legend()
plt.title('Singular values of K*S')
plt.grid(1)

plt.figure(3)
t,y = con.step_response(To)
plt.plot(t,y, label=r'Hinf')
t,y = con.step_response(To2)
plt.plot(t,y, label=r'H2')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Step Response')
plt.legend()

