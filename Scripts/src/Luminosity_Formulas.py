import numpy as np

from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo 

# Cosmology

def d_L(z):
    
    omega_m = 0.3
    c = 3 * (10**5) 
    H_0 = 73 
        
    def f(x):
        return 1 / np.sqrt(omega_m * (1 + x)**3 + 1 - omega_m)

    integral, _ = quad(f, 0, z)
    return (1 + z) * (c / H_0) * integral 

def dVdz_dOmega(z):
    return cosmo.differential_comoving_volume(z).value

# Single energy Flux and Luminosity Calculation

def Monochromatic_k_Correction(nuLnu, z, E):
    return nuLnu(E*(1+z)) / nuLnu(E)

def K_Corrected_luminosity(flux, nuLnu, z, E):
    Lum_Distance = d_L(z)
    k_corr = Monochromatic_k_Correction(nuLnu, z, E)
    L = 4 * np.pi * (Lum_Distance * 3.085677581e24)**2 * flux * k_corr
    return L

# Band energy Flux and Luminosity Calculations

def integral_energy_flux(nuFnu, E1, E2):

    def integrand(logE):
        E = np.exp(logE)
        return E**2 * nuFnu(E)

    return quad(integrand, np.log(E1), np.log(E2))[0] * 1.602176634e-6

def K_Corrected_Band_luminosity(nuFnu, z, E1, E2):
    
    Lum_Distance = d_L(z)
    
    obs_flux = integral_energy_flux(nuFnu, E1, E2)
    rest_flux = integral_energy_flux(nuFnu, E1*(1+z), E2*(1+z))
    
    k_corr = rest_flux / obs_flux
    L = 4 * np.pi * ((Lum_Distance * 3.085677581e24)**2) * obs_flux * k_corr
    
    return L


