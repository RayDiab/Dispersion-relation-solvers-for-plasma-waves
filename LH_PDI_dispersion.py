### Solves the PDI dispersion relation in the LH regime. 
### This is a Python adaptation of a FORTRAN code by Takase and Porkolab.

#Import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.special import iv
from numpy import exp, cos, sin, pi
from cmath import sqrt

import ast


iroot=1;          # Root finding solver 1
# [0.0625 0.0641 100 0.001 0.0001]
########################################################################## input
f0=4.6;          # GHz
Lz=44.2;          # cm
Ly=77.6;          # cm
# power will be asked later....
n0para=2.;       # c*k_{0||}/(w_0)
npara=10.;          # c*k_{||}/(w_0)
theta_pump_d=0.0; #(deg), Angle between the x-axis and the perpendicular wavevector of pump
theta_ls_d=90.0;  #(deg), Angle between the x-axis and the perpendicular wavevector of the lower sideband

gas=2;      #(AU),      gas, 2 for deuterium
Bt=1.83;     #(T),      magnetic field
ne=1e19;    #(m**{-3}), electron density
Te=36.7;      #(eV),     electron temperature
tau=1;      #(),       Te/Ti
Ln=999;     #(cm),     (ne)/|d(ne)/dx|
n_neut=0.01;   # neutral density
n_wg=0.1e18;  #(m**{-3}) density right at the waveguide
####################################################################### end of input
c=2.99792458e8;             # (m/s), speed of light
mp=1.672621581e-27;        # (kg),  proton mass
me=9.109381887e-31;        # (kg),  electron mass
eps0= 8.854187817e-12;       # (F/m), vacuum permittivity
echar=1.602176565e-19;      # (C),   elementary charge


##############################################
theta=abs(theta_pump_d-theta_ls_d)/180.0*pi; #  angle between pump and lower sideband
theta_pump=theta_pump_d*pi/180.0;
theta_ls=theta_ls_d*pi/180.0;
#############################################

fpe=np.sqrt(ne*echar**2/(me*eps0))/(2*pi)/1e9;       #(GHz), electron plasma frequency
fce=echar*Bt/me/(2*pi)/1e9;                      #(GHz), electron cyclotron frequency
fpi=np.sqrt(ne*echar**2/(gas*mp*eps0))/(2*pi)/1e9;   #(GHz), ion plasma frequency
fci=echar*Bt/(mp*gas)/(2*pi)/1e9;                #(GHz), ion cyclotron frequency


asq=mp/me*gas;        # mi/me
a=sqrt(asq);          # sqrt(mi/me)
bsq=(fpe/fce)**2;      #(wpe/wce)**2
b=sqrt(bsq);          #
bsq_wg=(n_wg/ne)*bsq; #(wpe/wce)**2 at the waveguide
x=fpi/f0;             # wpi/w0
xsq=x**2;              #
x_wg=sqrt(n_wg/ne)*x; # wpi/w0 at the waveguide
wLH=x/sqrt(1+bsq);    # wlh/w0

cei=1.33e5*(ne/1e20)/(Te/1e3)**(1.5);   # s**{-1}, electron ion collision frequency, Freidberg p.197, Eq(9.51)
cei=cei/(f0*1e9);                      # normalized colllisional frequency
icoll=complex(0.0,cei);                # make it imaginary
lambda_de=sqrt(eps0*Te/echar/ne);      #(m) electron debye length
dg=lambda_de/(Ln/100);                 # 
loglambda=np.log(9*4*pi/3*ne*lambda_de**3);#log(lambda) Friedberg p.194, Eq(9.36) loglambda=log(4.9e7*(Te/1e3)**(1.5)/(ne/1e20)**(0.5));
########################################################################

rhoz=(n0para/c*2*pi*(f0*1e9))*(lambda_de); #k_{0||}*lambda_{de}
rho=rhoz*npara/n0para;                     #k_{||}*lambda_{de}
rhom=rho-rhoz;                             #k_{-||}*lambda_{de}
rhop=rho+rhoz;                             #k_{+||}*lambda_{de}  


##########################################################################################


######################################################################
def predict_klambda(asq,bsq,x,rho0,rho,thetaz,thetam):
    #thetaz: the angle between x-axis and kp0
    #thetam: the angle between x-axis and kp-
    
    theta=abs(thetam-thetaz)/180.0*pi; #[rad]
    o0oLH=(1.0+bsq)/x**2-1;
    
    rhom2=(rho-rho0)**2;
    em2=rhom2*asq/o0oLH;
    kpm=sqrt(em2-rhom2);
    
    rho02=rho0**2;
    e02=rho02*asq/o0oLH;
    kx0=sqrt(e02-rho02);
    
    kp2=kpm**2+kx0**2+2.0*kpm*kx0*cos(theta);
    e=sqrt(kp2+rho*rho);
    
    kp_low_limit=abs(kpm-kx0);
    e_low_limit=abs(kp_low_limit**2+rho*rho);
    
    kp_upper_limit=abs(kpm+kx0);
    e_upper_limit=abs(kp_upper_limit**2+rho*rho);
    
    
    #disp([num2str(e_low_limit),' < k*lambda_{De} < ',num2str(e_upper_limit),',  k*lambda_{De}=',num2str(e)])
    print(['Expected k*lambda_{De}=',str(e_low_limit),' ', str(e),' ',str(e_upper_limit)])
    
    e_predict=e;
    return [e_predict]

##########################################################################
e_predict=predict_klambda(asq,bsq,x,rhoz,rho,theta_pump_d,theta_ls_d);
########################################################################
def calc_phi(theta,kp,kp0,kp2):
    ctheta=cos(theta);
    stheta=sin(theta);
    kp0kpm=kp0*ctheta;
    kpkpm=sqrt(kp**2-(kp0*stheta)**2); #from projecting the expression of kpm on the direction perp to kpm.
    kpm1=abs((+kpkpm-kp0kpm)-kp2); 
    kpm2=abs((-kpkpm-kp0kpm)-kp2);
    if kpm1 < kpm2:
        kpm=kpkpm-kp0kpm;
        iskpm=1;
        kpmmm=kpm1/kpm;
    else:
        kpm=-kpkpm-kp0kpm;
        iskpm=2;
        kpmmm=kpm2/kpm;
    
    cphi=np.real((kp**2+kp0**2-kpm**2)/(2.0*kp*kp0));
    
    if (cphi > 1.0):
        phi=0.0;
    elif (cphi < -1.0):
        phi=pi;
    else:
        phi=np.arccos(cphi);
    kpp=sqrt(kp**2+kp0**2+2.0*kp*kp0*cphi);
    return [phi,kpm,kpp,iskpm,kpmmm]
###################################################################################################
def calc_eta_epar(f0,Ly,Lz,power,bsq_wg,x_wg,n0para,asq,bsq,x,Bt,Te,gas):
    cfac=1/sqrt(1. -(15./(f0*Ly))**2);
    E0x_wg_vac=sqrt(power*(376.7/90)*4*cfac/(Lz*Ly));
    o0LH_wg=(1.0+bsq_wg)/(x_wg**2)-1.0;
    E0x_wg2=(E0x_wg_vac)**2 *n0para/(2.0*x_wg**2)*(o0LH_wg**(-1.5))*sqrt(asq/(1.0-o0LH_wg/asq))/cfac;
    E0para_wg=sqrt(E0x_wg2*o0LH_wg/asq); # one resonance cone
    oLH2o02=x**2/(1.0+bsq);
    o0LH=1.0/oLH2o02-1.0;
    #o0pe=1.0/(x*sqrt(asq));
    ratio_zz=(1.0+bsq_wg-x_wg**2)/(1.0+bsq-x**2);
    ratio_xx=(asq*x_wg**2-1.0)/(asq*x**2-1.0);
    ratio_k0xk0p=sqrt(asq/o0LH-1.0);
    
    E0para=E0para_wg*(abs(ratio_zz*ratio_xx))**0.25;
    E0x=E0para*ratio_k0xk0p;
    eta=(E0x/Bt)/sqrt(Te/gas*1.602/(1.67*9.0));
    print(abs(E0x))
    epar=48.0/6.283*E0para/f0/sqrt(Te*1.6*9.11);
    return eta,epar,E0para,E0x 
###################################################################################
def calc_pdr(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi):    
    zeta=(y+icoll)/T2;
    pdfunc=1j*sqrt(pi)*wofz(zeta);
    zeta1=(y+icoll-dge)/T2;

    chie=(1.0+T1*zeta1*pdfunc)/((1.0+T11*pdfunc)*esq);

    zetam=(y-1.0+icoll)/T2m;
    zetam1=(y-1.0+icoll-dge)/T2m;
    pdfuncm=1j*sqrt(pi)*wofz(zetam);
    chiem=(1.0+T1m*zetam1*pdfuncm)/((1+T11m*pdfuncm)*emsq);

    zetap=(y+1.0+icoll)/T2p;
    zetap1=(y+1.0+icoll-dge)/T2p;
    pdfuncp=1j*sqrt(pi)*wofz(zetap);
    chiep=(1.0+T1p*zetap1*pdfuncp)/((1+T11p*pdfuncp)*epsq);

    #zeta_i=y/T10;
    zeta_i1=(y+dgi)/T10;
    #zeta_im=(y-1.0)/T10m;
    zeta_im1=(y-1.0+dgi)/T10m;
    #zeta_ip=(y+1.0)/T10p;
    zeta_ip1=(y+1.0+dgi)/T10p;
#     n21=201;
#     n1=101;
#     
    n21=101;
    n1=51;
    chisum=complex(0,0);
    chimsum=complex(0,0);
    chipsum=complex(0,0);
    for index in range(1,n21): 
        n=index-n1;
        zetani=(y-T12*n)/T10;
        zetanim=(y-1.0-T12*n)/T10m;
        zetanip=(y+1.0-T12*n)/T10p;
        chisum=chisum+iv(abs(n),bi)*exp(-bi)*1j*sqrt(pi)*wofz(zetani);
        chimsum=chimsum+iv(abs(n),bim)*exp(-bim)*1j*sqrt(pi)*wofz(zetanim);
        chipsum=chipsum+iv(abs(n),bip)*exp(-bip)*1j*sqrt(pi)*wofz(zetanip);   
        
        #temp_zetani(index)=zetani;
        #temp_temp(index)=besseli(abs(n),bi)*exp(-bi)*i*sqrt(pi)*wofz(zetani);
        
        #temp_zetanim(index)=zetanim;
        #temp_temp2(index)=besseli(abs(n),bim)*exp(-bim)*i*sqrt(pi)*wofz(zetanim);

    chii=tau*(1.0+zeta_i1*chisum)/esq;
    chiim=tau*(1.0+zeta_im1*chimsum)/emsq;
    chiip=tau*(1.0+zeta_ip1*chipsum)/epsq;
    eps=1.0+chii+chie;
    epsm=1.0+chiim+chiem;
    epsp=1.0+chiip+chiep;
    print("CHIE",chie,"CHIEM",chiem,"CHIEP",chiep,"CHII",chii,"CHIIM",chiim,"CHIIP",chiip,"eps",eps,"epsm",epsm,"epsp",epsp)
    
    
    f1=eps+0.25*chie*(1.0+chii)*(munega**2/epsm+muposi**2/epsp);
    return [f1,chii,chie,eps,chiim,chiem,epsm,chiip,chiep,epsp]
###############################################################################
def predict_nextval(k,y,ylast1,ylast2,ylast3):
    if(k==1):
        ylast3=0;
        ylast2=0;
        ylast1=y;
        yout=y;    
    elif(k==2):
        ylast3=0;
        ylast2=ylast1;
        ylast1=y;
        yout=2*ylast1-ylast2;
    else:
        ylast3=ylast2;
        ylast2=ylast1;
        ylast1=y;
        yout=3*ylast1-3*ylast2+ylast3;
    return [yout,ylast1,ylast2,ylast3]
################################################################################
def find_root1(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi):
    # nits = nbr of iterations
    y0=y;
    y1=y0;
    y=1.0001*y1;
    fy1=calc_pdr(y1,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi)[0];
    print("FY",fy1)
    fy=calc_pdr(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi)[0];
    #print(fy)
    nits=0;
    iflag_in=1;
    while (iflag_in==1):
        dy=-fy*(y-y1)/(fy-fy1);
        if(abs(dy/y)<1e-5):
            #print(fy)
            break
        nits=nits+1;
        if(nits > 50): 
            print('end case on excessive iterations')
            iflag_in=0;
            break
        y1=y;
        fy1=fy;
        y=y+dy;
        fy=calc_pdr(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi)[0];
        print("Fy",fy)
        print("YYY",y)
    return [ y, iflag_in,nits ]

######################################################################################
    

iflag_out=1;
iflag_power=1;
kplot=1; # index number for plotting purposes
ylast1=0;
ylast2=0;
ylast3=0;
while iflag_power==1:
    power=float(input('power(kW): '));
    if power==0: break
    eta,epar,E0para,E0x = calc_eta_epar(f0,Ly,Lz,power,bsq_wg,x_wg,n0para,asq,bsq,x,Bt,Te,gas);    

    while iflag_out==1:
        input_array=input('[emin emax n_number Re(w) Im(w)]: ');
        input_array = ast.literal_eval(input_array)
        if input_array[1]==0:
            break
        emin=input_array[0];
        emax=input_array[1];
        n_number=input_array[2];
        ystart=input_array[3]+1j*input_array[4];
        y=ystart;
        de=(emax-emin)/n_number;
        n_numtot=n_number+1;
        #yset = np.zeros(n_numtot,dtype = 'complex')
        klambdaset = []
        yset = []
        epsset = []
        epspset = []
        epsmset = []
        chieset = []
        chiiset = []
        chiimset = []
        chiemset = []
        #klambdaset = np.zeros(n_numtot,dtype = 'complex')
        for k in range(1,n_numtot):          
            e=emin+de*(k-1);      #k*lambda_{de}
            esq=e*e;              
            
            cost=rho/e;           # (k||/k)*lambda_{de}
            costm=rhom/e;         # (k||**{-}/k)*lambda_{de}
            costp=rhop/e;         # (k||**{+}/k)*lambda_{de}

            e0sq=rhoz**2*asq/((1.0+bsq)/xsq-1.0);   # k0*lambda_de, based on the linear disperion relation of LH waves
            emsq=rhom**2*asq/((1.0+bsq)/xsq-1.0);   # k**{-}*lambda_de, based on both the linear dispersion of LH waves and the value of k||**{-}*lambda_de

            kp=sqrt(esq-rho**2);     # kperp*lambda_de
            kp0=sqrt(e0sq-rhoz**2);  # kperp0*lambda_de
            kp2=sqrt(emsq-rhom**2);  # kperp-*lambda_de
            [phi,kpm,kpp,iskpm,kpmmm]=calc_phi(theta,kp,kp0,kp2); # find phi (i.e., the angle between the kperp0 and kperp)
                               
            nperp=kp/lambda_de*c/(2*pi*f0);
            nperp0=kp0/lambda_de*c/(2*pi*f0);
            nperpm=kpm/lambda_de*c/(2*pi*f0);
            
            ep=sqrt(kpp**2+rhop**2);   # k**{+}*lambda_de, based on the kperp**{+} that is found from the matching condition
            em=sqrt(kpm**2+rhom**2);   # k**{-}*lambda_de, based on the kperp**{-} that is found from the matching condition
            epsq=ep**2;       
            emsq=em**2;
            eovem=abs(e/em);         # (k/k**{-})*lambda_de

            be=esq*bsq*(1-cost*cost); # b_e
            bi=be*(asq/tau);          # b_i
            bem=kpm**2*bsq;            # lower sideband b_e
            bim=bem*asq/tau;          # lower sideband b_i
            bep=kpp**2*bsq;            # upper sideband b_e
            bip=bep*asq/tau;          # upper sideband b_i
             
            kx=kp*cos(phi+theta_pump); # ion mode perpendicuar wavevector(kp) component projected on the x-axis
            ky=kp*sin(phi+theta_pump); # ion mode perpendicuar wavevector(kp) component projected on the y-axis
            kx0=kp0*cos(theta_pump);   # pump perpendicuar wavevector(kp0) component on the x-axis
            ky0=kp0*sin(theta_pump);   # pump perpendicuar wavevector(kp0) component on the y-axis
            kxm=kpm*cos(theta_ls);     # lower sideband perpendicuar wavevector(kpm) component on the x-axis
            kym=kpm*sin(theta_ls);     # lower sideband perpendicuar wavevector(kpm) component on the y-axis
            kxp=kx+kx0;                # upper sideband perpendicuar wavevector(kpp) component on the x-axis
            kyp=ky+ky0;                # upper sideband perpendicuar wavevector(kpp) component on the y-axis

            mueta_ls=abs(eta*kym*x);                         
            mueta_us=abs(eta*kyp*x);
            muepar_ls=abs(epar*rhom*x*a);
            muepar_us=abs(epar*rhop*x*a);
            munega=abs(e/em)*sqrt(mueta_ls**2+muepar_ls**2);
            muposi=abs(e/ep)*sqrt(mueta_us**2+muepar_us**2);
            mu1=mueta_ls/muepar_ls;
            print("MUSSS",muposi,munega)
            T1=iv(0,be)*exp(-be);
            T1m=iv(0,bem)*exp(-bem);
            T1p=iv(0,bep)*exp(-bep);

            T2=sqrt(2)*e*x*a*cost;    # k_{||} v_{te} / w_0
            T2m=sqrt(2)*e*x*a*costm;  # lower sideband
            T2p=sqrt(2)*e*x*a*costp;  # upper sideband
            sr=T2/abs(T2);
            srm=T2m/abs(T2m);
            srp=T2p/abs(T2p);

            T10=sqrt(2.0/tau)*e*x*cost;   # k_{||} v_{ti} / w_0
            T10m=sqrt(2.0/tau)*e*x*costm; # lower sideband
            T10p=sqrt(2.0/tau)*e*x*costp; # upper sideband

            T11=icoll*T1/T2;
            T11m=icoll*T1m/T2m;
            T11p=icoll*T1p/T2p;

            T12=x/(a*b);
            sint=sqrt(1-cost**2);
            dge=e*sint*asq*bsq*T12*dg;
            dgi=dge/tau;
            
            if iroot==1: #Newton's method
                [y,iflag_in,nits]=find_root1(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi);
            if k == 1:
                print("be",be,"bi",bi,"bem",bem,"bim",bim,"e",e)
                xtab = np.linspace(0,5*fci,100)/f0
                A=calc_pdr(xtab,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi)
                for i in range(len(A)):
                    
                    plt.figure(20+i,figsize = (10,10))
                    plt.plot(xtab*f0/fci,A[i])
                    plt.plot(xtab*f0/fci,np.imag(A[i]))
                    plt.plot(xtab*f0/fci,np.zeros(len(xtab)))
                    plt.title(str(i))
                
            if iflag_in==1:
               [fchk,chii,chie,eps,chiim,chiem,epsm,chiip,chiep,epsp]=calc_pdr(y,esq,emsq,epsq,T1,T1m,T1p,T2,T2m,T2p,T10,T10m,T10p,T11,T11m,T11p,T12,icoll,dge,dgi,bi,bim,bip,tau,munega,muposi);
               logf=np.log10(abs(fchk));
               #yset[kplot]=y;
               #print(np.real(y),np.imag(y))
               klambdaset.append(e);  
               yset.append(y);
               #klambdaset[kplot] = e
               epsset.append(eps);
               epsmset.append(epsm);
               epspset.append(epsp);
               chiiset.append(chii);
               chieset.append(chie);
               chiimset.append(chiim);
               chiemset.append(chiem);
               kplot=kplot+1;
               print(kplot)
            elif iflag_in==0:
                break
            [y,ylast1,ylast2,ylast3]=predict_nextval(k,y,ylast1,ylast2,ylast3); #for next initial value   
            if k == n_numtot:
                break
        break
    break
            
print('The end')

plt.figure(1,figsize = (10,10))
plt.plot(np.real(yset),np.imag(yset))
plt.figure(2,figsize = (10,10))
plt.plot(klambdaset,np.real(yset))
plt.figure(3,figsize = (10,10))
plt.plot(klambdaset,np.imag(yset))
plt.figure(4,figsize = (10,10))
plt.plot(klambdaset,np.real(epsset))
plt.figure(5,figsize = (10,10))
plt.plot(klambdaset,np.imag(chiemset))
plt.figure(6,figsize = (10,10))
plt.plot(klambdaset,np.real(epsmset))

#TF=isempty(yset);
#if TF==0 :         
#    figure(1)
#    subplot(3,3,1);
#    ytemp=yset;
#    plot(real(ytemp),imag(ytemp),'x');xlabel('\omega_R');ylabel('\gamma');
#    subplot(3,3,2);
#    [AX,H1,H2]=plotyy(klambdaset,real(eps_set), klambdaset,imag(eps_set)); xlabel('k \lambda_{De}');ylabel(AX(1),'Re(\epsilon)');ylabel(AX(2),'Im(\epsilon)');
#    subplot(3,3,3)
#    [AX,H1,H2]=plotyy(klambdaset,real(ytemp),klambdaset,imag(ytemp));xlabel('k \lambda_{De}');ylabel(AX(1),'\omega_R');ylabel(AX(2),'\gamma');
#    subplot(3,3,4);
#    [AX,H1,H2]=plotyy(klambdaset,real(epsm_set), klambdaset,imag(epsm_set));xlabel('k \lambda_{De}');ylabel(AX(1),'Re(\epsilon^{-})');ylabel(AX(2),'Im(\epsilon^{-})');
#    subplot(3,3,5);    
#    [AX,H1,H2]=plotyy(klambdaset,real(epsp_set), klambdaset,imag(epsp_set));xlabel('k \lambda_{De}');ylabel(AX(1),'Re(\epsilon^{+})');ylabel(AX(2),'Im(\epsilon^{+})');
#    subplot(3,3,6);    
#    [AX,H1,H2]=plotyy(klambdaset,real(chiim_set), klambdaset,imag(chiim_set));xlabel('k \lambda_{De}');ylabel(AX(1),'Re(\chi_i^{-})');ylabel(AX(2),'Im(\chi_i^{-})');
#    subplot(3,3,7);    
#    [AX,H1,H2]=plotyy(klambdaset,real(chiem_set), klambdaset,imag(chiem_set));xlabel('k \lambda_{De}');ylabel(AX(1),'Re(\chi_e^{-})');ylabel(AX(2),'Im(\chi_e^{-})');
#    subplot(3,3,8);
#    plotyy(klambdaset,real(ytemp)*f0/fci,klambdaset,imag(ytemp)*f0/fci);xlabel('klambda');
#




#fclose(fid);
























