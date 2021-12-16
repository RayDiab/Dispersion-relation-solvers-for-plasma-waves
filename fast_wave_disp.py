#Author: Raymond Diab
# Solves the Fast Wave (FW) dispersion relation

import numpy as np


eps0 = 8.854e-12 #[F/m] vacuum permittivity
freq = 3e7
w = freq * 2 * np.pi           # omega

Zi = 1

q = 1.60217662e-19
qe = -q
qi = Zi*q
me = 9.10938356e-31
mi = 1.660539040e-27  # unified mass unit
Bnorm = 0.23


ne =2e18
ni = ne
nim = 0

#T  = max((90*npsifit(r,z)[0]+10, 10))
T = 50

Te = T*q     # in MKSA
Ti = T*q     # in MKSA
Tim = T*q     # in MKSA

vTe = np.sqrt(2*Te/me)
vTi = np.sqrt(2*Ti/mi)


wpe = np.sqrt(ne * q**2/(me*eps0))
wpi = np.sqrt(ni * q**2/(mi*eps0))

wce = qe * Bnorm/me
wci = qi * Bnorm/mi


P = (1 - wpe**2/w**2
     - wpi**2/w**2)

S = (1-wpe**2/(w**2-wce**2)-wpi**2/(w**2-wci**2))
D = (wce*wpe**2/(w*(w**2-wce**2)) + wci*wpi**2/(w*(w**2-wci**2)))
R = 1+wpe**2/wce/(w+wce)+wpi**2/wci/(w+wci)
L = 1-wpe**2/wce/(w-wce)-wpi**2/wci/(w-wci)


k0 = w/3e8 
kpar = 5 #[m^-1] parallel wavenumber
npar = kpar/k0
A = S
B = (npar**2 - S)*(S+P) + D**2
C = P*((npar**2- S**2)-D**2)

# Dispersion relation
kperp = np.sqrt(k0**2*((R - npar**2)*(L-npar**2))/(S-npar**2))


nperp = kperp/k0
npes = kperp**2 / k0**2
npas = npar**2

print("kpar = ", kpar, "m^-1")
print("kperp = ", kperp, "m^-1")
if kpar != 0: print("lambda_par = ", 2*np.pi/kpar, "m")
print("lambda_perp = ", 2*np.pi/kperp, "m")