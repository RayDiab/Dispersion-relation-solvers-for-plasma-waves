# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 01:17:58 2021

@author: Raymond Diab

This code solves the IBW dispersion relation in the limit of cold electrons.
For reference, see (eq. 13) in "M. Ono et al., The Physics of Fluids 26, 298 (1983); doi: 10.1063/1.863972"
"""

#Import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from numpy import sqrt
from scipy.optimize import root

#Constants
c =2.99792e8 # [m/s] speed of light
echar = 1.602e-19 # elementary charge [C]
kB = 1.38064e-23 #Boltzmann constant [J/K]
me = 9.11e-31 #[kg] electron mass
mp = 1.67e-27 #[kg] proton mass
eps0 = 8.854e-12 #[F/m] vacuum permittivity



#Ions
A = 2 #[AU]
Ti = 50 #[eV]
Zi = 1
qi = Zi*echar # [C] charge
mi = A*mp #[kg] mass

ne = 2e18 #[m^-3] density


#Parameters
Bt = 0.24 #23 #[T] magnetic field

wci=Zi*echar*Bt/(mp*A)       #[Hz], ion cyclotron frequency
vthi = sqrt(2*Ti*echar/(A*mp))
L_Di = sqrt(eps0*Ti/(Zi*echar)/ne) #Ion Debye length
rci = vthi/wci
wpe=sqrt(ne*(echar**2)/(me*eps0))       #[Hz], electron plasma frequency
wpi=sqrt(ne*(Zi*echar)**2/(A*mp*eps0))   #[Hz], ion plasma frequency

#wave parameters
kpar = 8 #[m^-1] parallel wavenumber
w0 = 2*np.pi*3e7 #[Hz] wave frequency
m = int(w0//wci) # harmonic number

def func2(y,kperp,m):
    w0 = y[0]+1j*y[1]
    k = np.sqrt(kpar**2+kperp**2)
    lambdi = kperp**2 * Ti*echar/mi/wci**2
    A = wpi**2*np.exp(-lambdi)/lambdi *sum(iv(n,lambdi)*2*n**2/(n**2 *wci**2 - w0**2) for n in range(m,150))
    B = -wpi**2 * np.exp(-lambdi)/lambdi * sum(iv(n,lambdi)*2*n**2/(w0**2-n**2*wci**2) for n in range(1,m))
    C = 1 - wpe**2/w0**2 * kpar**2/k**2
    OUT = A + B + C
    return np.array([np.real(OUT),np.imag(OUT)])

plt.figure(1,(12,12))
kprci = np.linspace(0.01,37,100)
kperp = kprci/rci

## Choose the range of hamonics of the IBW that you want to solve
for i in range(7,12):
    print("Doing", i)
    wsol = np.zeros(len(kprci))
    for j in range(len(kprci)):
        ysol = root(func2,[(i + 0.99)*wci,0],args = (kperp[j],i),method = 'lm').x
        wsol[j] = ysol[0]
        if wsol[j]<i*wci or wsol[j]>(i+1)*wci: wsol[j] = np.nan
    plt.plot(kprci,wsol/wci,marker = 'x')
    plt.plot(kprci,np.zeros(len(kprci))+i,'k--')

plt.title("IBW dispersion relation",fontsize = 20)
plt.ylabel(r'$\omega / \omega_{ci}$',fontsize = 15)
plt.xlabel(r'$k_{\perp}r_{ci}$',fontsize = 15)


