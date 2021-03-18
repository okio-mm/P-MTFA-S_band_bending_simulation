# This code is used to calculate the band bending profile and Quantum Well states in a semiconductor or insulator given the material parameters and some intial conditions.
# The calculation of the band bending profile in one dimension is performed using the Modified Thomas Fermi Approximation (MTFA) and solving the Poisson equation (One can find the equations in [King et al. PRB 77, 125305 (2008)] ).
# The solution (V(z)) to this second order differential equation is found by solving the boundary value problem, and it requires to know or guess the potential at the surface V0.
# The code works on the assumption that the electron and hole bands are parabolic.
# Quantum Well states' energies and wavefunctions are calculated by solving the Schroedinger equation within the V(z) previosly calculated. 
#%%
import numpy as np
import scipy as sp
import scipy.integrate as spint
import matplotlib.pyplot as plt 
import scipy.optimize as spopt
import scipy.interpolate as interp
import scipy.misc as spmisc
 
#%%
## define the constants used
class Const:
   kb = 8.61735E-5                  # Boltzmann Constant
   hbar = 1.054571817E-34           # Reduced Planck constant in different units
   hb = 6.58212E-16
   hbc = 0.1973269804E-6
   q0 = -1.60218E-19                # Elemental charge
   q0V = 1
   m0 = 9.10938E-31                 # Electroon mass
   m0c = 0.51099895E6   
   eps0 = 8.85419E-12               # vacuum permittivity

## define the parameters of the calculation
nofepoints = 201        # number of points in energy
emax = 6                # maximum energy cutoff for the calculation
nofzpoints = 201        # number of points in the z (distance from surface) direction
maxdepth = 150E-9       # maximum depth z in meters
tolerance = 0.01        # tolerance

materialname = "CBS"    # chose the material from the materials_database dictionary
v0 = -0.544            # Potential at the surface in V(eV) - Boundary condition.
T=20                    # Temperature

## Initialization of energy and depth vectors
eaxes = np.linspace(-emax,emax,nofepoints)
zaxes = np.linspace(0,maxdepth,nofzpoints) 
  
#%%
## Enter here the material specific parameters in a new entry of the dictionary.
## nd, na = concentration of acceptor and donor atoms in m^(-3)
## eps_r = static relative dielectric constant
## ms_e, ms_h1, ms_h2 = electron and hole effective masses
## ei = ionization energy of the acceptor/donor states
## eg = band gap of the material
## cb = energy reference of the conduction band (if 0 th vb will be at -eg)
## isdegenerate = true if semiconductor is degenerate, false otherwise        

materials_database={
    'CBS' :{'nd' : 0e16, #0E16,
            'na' : 5e24, #1.7E16,
            'eps_r' : 113,
            'ms_e' : 0.13,
            'ms_h1' : -0.8,
            'ms_h2' : -0.13,
            'ei' : 0.004,
            'eg' : 0.25,
            'cb' : 0,
            'isdegenerate' : False
           },
    'GaAs' :{'nd' : 5E22,
            'na' : 0e22,
            'eps_r' : 12.91,
            'ms_e' : 0.069,
            'ms_h1' : -0.5,
            'ms_h2' : -0.068,
            'ei' : 0.004,
            'eg' : 1.453,
            'cb' : 0,
            'isdegenerate' : False,
            'ef' : -0.05607667
           }
    }


#%%
## Instantiate a Material class, the constructor requires the material name to fetch information from the materials_database.

class Material:

    # Method to calculate the density of states of a band with effective mass 'ms' at a certain energy 'e'
    def calculatedensityofstates(self,e,ms):
        if ms>=0:
            e_ref = self.cb        
        elif ms<0:
            e_ref = self.vb
        prefactor = (1 / (2 * np.pi**2)) * (2 * np.abs(ms) * Const.m0c / Const.hbc**2)**(3/2) # standard density of states for a parabolic band
        dos = prefactor*(((e-e_ref)*np.sign(ms))**0.5)
        return dos
   
    # Methoid to calculate the Tomas Fermi factor that modulates the carrier density taking into account the interference between electron waves scattering at the surface 
    def calculateTF(self,e,z,ms,ef,v):
        lf=1
        if self.isdegenerate:
            lf = Const.hbc/(2*ms*Const.m0c*(ef+v))**0.5
        elif not self.isdegenerate:
            lf = Const.hbc/(2*np.abs(ms)*Const.m0c*Const.kb*T)**0.5
        tf=1- np.sinc((2*z/lf)*((np.abs(e)/(Const.kb*T))**0.5)*((1+np.abs(e)/self.eg)**0.5))
        return tf
  
    # Method to calculate the fermi Dirac distribution at energy 'e' with Fermi level at 'ef'-'v' where 'v' is the potential (the band bending is computed as a bending of ef by v)
    def calculatefermidirac(self,e,ef,v):
        fd = 1/(1+np.exp((e-ef+v)/(Const.kb*T)))
        return fd

    # Methods to calculate the carrier density as \int(dos*f_{FD}*f{TF})
    # returns carrier density at certain energy and depth 'z'
    def carrierdensityofenergy(self,e,z,ms,ef,v):
        if ms>=0:
            e_ref = self.cb
            cofe = self.calculatedensityofstates(e,ms)*self.calculatefermidirac(e,ef,v)*self.calculateTF(e,z,ms,ef,v)
        elif ms<0:
            e_ref = self.vb
            cofe = self.calculatedensityofstates(e,ms)*(1-self.calculatefermidirac(e,ef,v))*self.calculateTF(e,z,ms,ef,v)
        return cofe
   
    # returns carrier density at z (integrated in energy) 
    def calculatecofx(self,z,ms,ef,v):      
        if ms>=0:
            e_ref = self.cb
            #cofx = spint.quad(self.carrierdensityofenergy,e_ref,np.inf,args=(z,ms,ef,v), limit=1000)   # more accurate, very slow
            cofx = spint.fixed_quad(self.carrierdensityofenergy,e_ref,emax,args=(z,ms,ef,v),n=200)      # fast integration
        elif ms<0:
            e_ref = self.vb
            #cofx = spint.quad(self.carrierdensityofenergy,-np.inf,e_ref,args=(z,ms,ef,v),limit=1000)
            cofx = spint.fixed_quad(self.carrierdensityofenergy,-emax,e_ref,args=(z,ms,ef,v),n=200)        
        return cofx[0]
   
    # Method to calculate the argument of the Poisson equation (-e/epsilon)x(Nd-Na+p(z)-n(z))  
    def calculatePoissonofef(self,ef):
        nofx = self.calculatecofx(1,self.ms_e,ef,0)                                                  # calculate the density of states at infinity (1 m) where no band bending exists
        pofx = self.calculatecofx(1,self.ms_h1,ef,0)+self.calculatecofx(1,self.ms_h2,ef,0)
        p_argument = -Const.q0/(Const.eps0 * self.eps_r)*(self.nd-self.na+pofx-nofx) 
        return p_argument 
   
    # Method to find the fermi level position in the material given the level of doping (acceptors-donors) in the system
    def findef(self):
        tmp_isdeg=self.isdegenerate
        self.isdegenerate=False
        ef = spopt.bisect(self.calculatePoissonofef,-emax,emax)
        self.isdegenerate = tmp_isdeg 
        return ef

    # Constructor: it takes the attributes from the database and fills the object properties
    def __init__(self,matrialname):
        for key, value in materials_database[materialname].items():
            setattr(self,key,value)  
        self.vb=self.cb-self.eg
        self.ef=self.findef()
    
    # Method called for re-initializing the Material object
    def reinitialize(self):
        self.__init__(materialname)



#%%
# Functions used to prepare for the numerical implementation of the solver for the Poisson equation.
# The Poisson equation d2V(z)/dz2 = -e/eps_r(Nd-nA+p(z)-n(z)) is a second order differential equation to be solved for V(z) 

# calculate the carrier densities and return the poisson argument. x is depth, y is energy
def poisson(x,y):
    nofx = mat.calculatecofx(x,mat.ms_e,mat.ef,y)
    pofx = mat.calculatecofx(x,mat.ms_h1,mat.ef,y)+mat.calculatecofx(x,mat.ms_h2,mat.ef,y)
    p_argument = -Const.q0/(Const.eps0 * mat.eps_r)*(mat.nd-mat.na+pofx-nofx) 
    return p_argument

# creation of the arrays to pass to the solver
def fun(x,y):
    poiss = np.array([poisson(x[i],y[0][i]) for i in range(x.size)])
    fofx = np.vstack((y[1],poiss))
    return fofx

# bc defines the boundary conditions of the system v(0) and v(\infty). v0 is defined in the initial parameters, change there.
def bc(ya,yb):
    yleft=v0
    yright=0
    residuals = np.array([ya[0] - yleft, yb[0] - yright])
    return residuals

#%%

# Solve the Poisson equation using damped Newton method as a boundary problem 
def solve_v():
    x = np.linspace(0,maxdepth,5)
    y = np.zeros((2,x.size))

    print(f'ef = {mat.ef}')
    res = spint.solve_bvp(fun,bc,x,y,tol=tolerance,verbose=2)   # Solve the boundary problem second order differential equation with the solve_bvp solver
    
    x_plot = np.linspace(0,maxdepth,nofzpoints)                 # x and y for a plot of the potential
    y_plot = res.sol(x_plot)[0]
   
    plt.plot(x_plot,y_plot)                                     # Plotting the V(z)
    plt.show()
   
    return y_plot,x_plot

#%%
## Some utility functions:

# return n(z) and p(z) give the potential v
def get_cofx(v):
    nofx,pofx = np.empty(nofzpoints) , np.empty(nofzpoints)
    for i in range(nofzpoints):
        nofx[i] = mat.calculatecofx(zaxes[i],mat.ms_e,mat.ef,v[i])
        pofx[i] = mat.calculatecofx(zaxes[i],mat.ms_h1,mat.ef,v[i])+mat.calculatecofx(zaxes[i],mat.ms_h2,mat.ef,v[i])
    return nofx,pofx

# returns the right side of thePoisson equation (charge) given the potential v
def get_poissarg(v):
    efi = np.diff(v,n=1)/(maxdepth/(nofepoints-1))      # electric field
    poiss = np.diff(v,n=2)/(maxdepth/(nofepoints-1))    # "effective charge"
    return efi,poiss

# A bit of sanity check: you can input the z-axes, n(z), p(z), Na, Nd, eps_r, the method calculates the integrals and outputs "effective charge", Electric Field and Potential V(z)
def get_v(zaxes,nofx,pofx,na,nd,eps_r):
    nd_ar = np.array([nd]*len(zaxes))
    na_ar = np.array([na]*len(zaxes))
    p_argument = -Const.q0/(Const.eps0 * eps_r)*(nd_ar-na_ar+np.array(pofx)-np.array(nofx)) 
    p_argument = np.insert(p_argument,0,0)
    p_argument = np.insert(p_argument,0,0)
    zaxes2 = zaxes.copy()
    zaxes2 = np.insert(zaxes2,0,-1*(maxdepth/(nofepoints-1)))
    zaxes2 = np.insert(zaxes2,0,-2*(maxdepth/(nofepoints-1)))
    e_field = spint.cumtrapz(p_argument,zaxes2)
    e_field -= np.amin(e_field)
    v_out = spint.cumtrapz(e_field,zaxes2[1:])
    v_out -= np.amax(v_out)
    return p_argument,e_field,v_out

def gv():
    return get_v(zaxes,nofx,pofx,mat.na,mat.nd,mat.eps_r)

#%%

## ######### Numerov solution to Shroedinger equation to find the Quantum Well States ############ ##

def fofx(en,vv,ms):
    fx = -(2*ms*Const.m0/Const.hbar**2)*(en-vv)*np.abs(Const.q0)
    return fx
def phi2psi(phiv,d,fx):
    psiv = phiv/(1-d**2*(fx/12))
    return psiv
def psi2phi(psiv,d,fx):
    phiv = psiv*(1-d**2*(fx/12))
    return phiv

# Find the numerov solution of the Schroedinger equation at a certain energy
def solveNumerov(en,v,zaxes,ms):
    psi = np.empty(len(zaxes))
    phi = np.empty(len(zaxes))
    f =  np.empty(len(zaxes))
    d = zaxes[1:]-zaxes[:-1]
    
    psi[0],psi[1] = 0,0.1
    f[0],f[1] = fofx(en,v[0],ms) , fofx(en,v[1],ms) 
    phi[0],phi[1] = psi2phi(psi[0],d[0],f[0]) , psi2phi(psi[1],d[1],f[1]) 

    for i in range(2,len(zaxes)):
        phi[i] = 2*phi[i-1]-phi[i-2]+d[i-1]**2*f[i-1]*psi[i-1]
        f[i] = fofx(en,v[i],ms)
        psi[i] = phi2psi(phi[i],d[i-1],f[i])
    return psi

def get_Numerov_bc(en,v,zaxes,ms):
        sol = solveNumerov(en,v,zaxes,ms)
        return sol[-1]

# Master function, Returns the wavefunctions and eigenvalues given the potential 'v', the effective mass of the conduction/valence band and the maximum energy
# It consists in solving the Schroedinger equation at "all" energies from the potential minimum to the max_en(somewhere close to CB). 
# The real solutions are found at those specific energy values where the wavefunction does not diverge. Numerically speaking this energy value is found around a change in the sign of the wave function at \infty
# When the code detects a sign change after an energy step (step size determined by inputed n), it finds the zero of the function numerically around the energy value using a Newton algorithm.
def numerical1Dshroed(v,zaxes,ms,max_en,n=1000):
    e0 = np.min(v)
    e1 = max_en
    d_e = (e1-e0)/(n-1)  
    bc1=bc2=0
    wavefunctions = np.empty((0,len(zaxes)))
    eigenvalues = np.empty(0)
    for i in range(n):
        en =e0+d_e*i
        psi = solveNumerov(en,v,zaxes,ms)
        bc1 = psi[-1]
        if np.sign(bc1)!=np.sign(bc2) and i!=0 :
            ex = spopt.newton(get_Numerov_bc,en,args=(v,zaxes,ms), maxiter =2000, rtol = 1e-8)      # precision given by tolerance and maxiter-ations
            #ex = (en+(e0+d_e*(i-1)))/2                                                             # midpoint approximation for faster but less precise calculation (precision here = energy step)
            eigenvalues = np.insert(eigenvalues,eigenvalues.size,ex)
            psi = solveNumerov(ex,v,zaxes,ms)
            if eigenvalues.size > wavefunctions.shape[0]:
                wavefunctions = np.insert(wavefunctions,wavefunctions.shape[0],psi,axis=0)
        bc2=bc1
    return wavefunctions, eigenvalues

# Because of the numerical nature of the calculation, the wavefunction might still diverge approching \infty. This function will flatten the tail of the wave function for plotting purposes.  
# The divergence must be deep inside the bulk of the material, if the divergence happens close to the potential well, you need to re-evaluate the precision of the calculation (increase n?)
def correctwfs(wfs,evs,v):
    zvs,zvs2 = evs.copy(),evs.copy()
    #wfs2 = wfs.copy()
    for i in range(evs.size):
        try:
            zvs[i] = np.where(np.isclose(v,evs[i],rtol=0.1)==True)[0][0]
            zvs2[i] = np.where(np.isclose(wfs[i][int(zvs[i]):],0,atol=1e-5)==True)[0][0]+zvs[i]
            wfs[i][int(zvs2[i]):]=0    
        except:  
            print("one wave function could not be corrected")  
    return wfs,zvs2

#%%
## If the Spin Orbit Coupling is relevant here you can calculate the Rashba-Spin-Orbit Coupling (RSOC) and the band splitting

# calculate the RSOC strength, it requires a 'deltapar' that is material specific.
def alphaf(v,z,qw_en,deltapar,ms):
    vofz_func = interp.interp1d(z,v)
    zofv_func = interp.interp1d(v,z)
    z_qw = zofv_func(qw_en)
    efield_z = spmisc.derivative(vofz_func,z_qw,dx=0.01e-9)
    alpha = Const.hbc**2/(2*Const.m0c*ms)*(deltapar/mat.eg)*((2*mat.eg+deltapar)/(mat.eg+deltapar)*(3*mat.eg+2*deltapar))*1*efield_z
    return alpha

def deltae(alpha,k):
    dele = 2*alpha*k
    return dele

def deltak(alpha,ms):
    delk=2*alpha*ms*Const.m0c/Const.hbc**2
    return delk*1e-10

# calculate the splitting in energy and momentum. Splitting in momentum depends on the effective mass. Splitting in energy depends on momemtuum 
def deltas(v,z,qw_en,deltapar,ms,k=.05e10):
    alpha = alphaf(v,z,qw_en,deltapar,ms)
    dele=deltae(alpha,k)
    delk=deltak(alpha,ms)
    return dele,delk

## ########################################################## ##

#%%
if __name__ == "__main__":
    
    mat = Material(materialname)                        # initialize the material object and its parameters
    v,x=solve_v()                                       # find v(z) and plot it
    nofx,pofx = get_cofx(v)                             # find the carrier densities             
    #pois,efield,vout = gv()                            # sanity check - inverting the poisson equation using the calculated charges densities, you should get the same result 
    efi,poiss = get_poissarg(v)                         # get electric field and charge
    wfs, evs = numerical1Dshroed(v,zaxes,mat.ms_e,-0.01)        # calculate wavefunctions and eigen-values of the Quantum Well states living in the band-bending potential well 
    correctwfs(wfs,evs,v)                               # get rid of numerical divergences for plotting reasons (first check the precision of your result)
   
    print("Number of calculated QWSs in defined range ' " , wfs.shape[0])
    print("QWSs energies = ", evs)
    for i in range(evs.size):                           # plot the wave functions
        plt.plot(zaxes,wfs[i])
    plt.axis([0,6e-8,-1,1])
    plt.show()

