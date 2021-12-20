# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 01:17:58 2021

@author: Raymond Diab
"""

#Import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
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
qi = Zi*echar
mi = A*mp

#Electrons
ne = 2e18 #[m^-3] Electron density
Te = 50 #[eV] Electron temperature

#Parameters
Bt = 0.24 #[T] magnetic field

wce=echar*Bt/me             #[Hz], electron cyclotron frequency
wci=Zi*echar*Bt/(mp*A)       #[Hz], ion cyclotron frequency
vthe = sqrt(2*Te*echar/me) #[m/s] #multiply with e?
vthi = sqrt(2*Ti*echar/(A*mp))
L_De = sqrt(eps0*Te/echar/ne) #Electron Debye length
L_Di = sqrt(eps0*Ti/(Zi*echar)/ne) #Ion Debye length
rci = vthi/wci
rce= vthe/wce
wpe=sqrt(ne*(echar**2)/(me*eps0))       #[Hz], electron plasma frequency
wpi=sqrt(ne*(Zi*echar)**2/(A*mp*eps0))   #[Hz], ion plasma frequency

kpar = wci/0.7/vthe

def func1(y,sqrtbi):
    w = y[0]+1j*y[1]
    bi = 0.5*sqrtbi**2
    be = 0.5*(sqrtbi*rce/rci)**2
    kperp = np.sqrt(bi*2/rci**2)
    betae = -sum(np.exp(-be)*iv(n,be)*(1+w/kpar/vthe*1j*np.sqrt(np.pi)*wofz((w+n*wce)/kpar/vthe)) for n in range(-50,51))
    betai = -sum(np.exp(-bi)*iv(n,bi)*(1+w/kpar/vthi*1j*np.sqrt(np.pi)*wofz((w+n*wci)/kpar/vthi)) for n in range(-50,51))
    OUT = 1+kpar**2/kperp**2 - wpe**2/wce**2 * betae/be - wpi**2/wci**2 * betai/bi
    return np.array([np.real(OUT),np.imag(OUT)])


sqrtbi = np.linspace(0.01,37,100)
plt.figure(1,(12,12))
for i in range(7,12):
    print("Doing", i)
    wsol = np.zeros(len(sqrtbi))
    for j in range(len(sqrtbi)):
        if i<=6:
            if sqrtbi[j]<2: ysol0 = [(i+0.99999)*wci,0]
            else: ysol0 = [(i+0.9)*wci,0]
        else:
            if sqrtbi[j]<4: ysol0 = [(i+0.9999)*wci,0]
            else: ysol0 = [(i+0.9)*wci,0]
            
        ysol = root(func1,ysol0,args = (sqrtbi[j]),method = 'lm').x
        #print(ysol)
        wsol[j] = ysol[0]
        if wsol[j]<i*wci or wsol[j]>(i+1)*wci: wsol[j] = np.nan
    plt.plot(sqrtbi,wsol/wci,marker = 'x')
    plt.plot(sqrtbi,np.zeros(len(sqrtbi))+i,'k--')
    
plt.title("IBW dispersion relation (full Stix)", fontsize = 22)
plt.ylabel(r'$\omega / \omega_{ci}$',fontsize = 20)
plt.xlabel(r'$k_{\perp}r_{ci}$',fontsize = 20)
plt.legend(fontsize = 20)



