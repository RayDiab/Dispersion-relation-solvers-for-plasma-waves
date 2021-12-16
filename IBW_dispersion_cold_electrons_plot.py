# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 01:17:58 2021

@author: Raymond Diab


This code plots the dispersion function for the IBWs in the limit of cold electrons.
This is the function that is equal to 0 in the dispersion relation.
For reference, see (eq. 13) in "M. Ono et al., The Physics of Fluids 26, 298 (1983); doi: 10.1063/1.863972"
"""

#Import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from numpy import sqrt

#Constants
c =2.99792e8 # [m/s] speed of light
echar = 1.602e-19 # elementary charge [C]
kB = 1.38064e-23 #Boltzmann constant [J/K]
me = 9.11e-31 #[kg] electron mass
mp = 1.67e-27 #[kg] proton mass
eps0 = 8.854e-12 #[F/m] vacuum permittivity



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
L_Di = sqrt(eps0*Ti/(Zi*echar)/ne) # Ion Debye length
rci = vthi/wci                     # Larmor radius
wpe=sqrt(ne*(echar**2)/(me*eps0))       #[Hz], electron plasma frequency
wpi=sqrt(ne*(Zi*echar)**2/(A*mp*eps0))   #[Hz], ion plasma frequency

#wave parameters
kpar = 8 #[m^-1] parallel wavenumber
w0 = 2*np.pi*3e7 #[Hz] wave frequency
m = int(w0//wci) # harmonic number

def func1(kperp):
    
    k = np.sqrt(kpar**2+kperp**2)
    lambdi = kperp**2 * Ti*echar/mi/wci**2
    A = wpi**2*np.exp(-lambdi)/lambdi *sum(iv(n,lambdi)*2*n**2/(n**2 *wci**2 - w0**2) for n in range(m,150))
    B = -wpi**2 * np.exp(-lambdi)/lambdi * sum(iv(n,lambdi)*2*n**2/(w0**2-n**2*wci**2) for n in range(1,m))
    C = 1 - wpe**2/w0**2 * kpar**2/k**2
    return A + B + C

kprci = np.linspace(1,35,1000)
kperp = kprci/rci
plt.figure(1,figsize = (10,10))
plt.plot(kprci, func1(kperp))
    
plt.title(r'Dispersion function vs. $k_{\perp}r_{ci}$', fontsize = 22)
plt.ylabel(r'Dispersion function',fontsize = 20)
plt.xlabel(r'$k_{\perp}r_{ci}$',fontsize = 20)

