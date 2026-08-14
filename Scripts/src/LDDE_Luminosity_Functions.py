import numpy as np

from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import interp1d

from src.Luminosity_Formulas import K_Corrected_luminosity, dVdz_dOmega, d_L, Monochromatic_k_Correction
from src.SED_Models import nuLnu_SBPL, nuLnu_LP

# LDDE Luminosity Function for Swift-BAT FSRQs from https://arxiv.org/pdf/2005.02648.
# Parameters are taken from Table 1. 
# A in units of Mpc^-3
# L_c in units of erg s^-1

def LDDE1(L,z, A=10**-13.02,L_c=10**51, gam1=5, gam2=0.8, z_c=1.36, p1=3.58, p2=-7.7, alpha=.42):
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    term3 = ((((1+z)/(1+(z_c*(L/10**47.5)**alpha)))**p1) + (((1+z)/(1+(z_c*(L/10**47.5)**alpha)))**p2))**-1
                        
    return term1 * term2 * term3

# LDDE Luminosity Function for Fermi-LAT FSRQs from https://arxiv.org/pdf/2510.05515.
# Parameters are taken from Table 4.
# A in units of Mpc^-3 erg^-1 s 
# L_c in units of erg s^-1
                
def LDDE2(L,z, A=18878.4*10**-13, L_c=1.09*10**48, gam1=0.29, gam2=1.63, z_c=2.05, p1=3.5, p2=-9, alpha=0.22, mu=2.42, beta=0.025, sigma=0.182):
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**(-p1)) + (((1+z)/(1+(z_c*(L/10**48)**alpha)))**(-p2)))**-1
    
    mu_eff = mu + beta*(np.log10(L) - 46)
    
    gamma_min = 1
    gamma_max = 4
    
    norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
    norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
    
    gamma_integral = sigma * np.sqrt(2*np.pi) * 0.5 * (erf(norm_max) - erf(norm_min))
    
    return term1 * term2 * term3 * gamma_integral

# LDDE Luminosity Function for Swift-BAT FSRQs from https://iopscience.iop.org/article/10.3847/1538-4357/ac937f/pdf.
# Parameters are listed in the final row of Table 3.
# A in units of Mpc^-3
# L_c in units of erg s^-1

def LDDE3(L,z, A=7.7e-5,L_c=1.55e44, gam1=-3.48, gam2=1.55, z_c=3.51, p1=4.54, p2=-9.43, alpha=2356.08e-8):
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    term3 = ((((1+z)/(1+(z_c*(L/10**48.6)**alpha)))**(-p1)) +(((1+z)/(1+(z_c*(L/10**48.6)**alpha)))**(-p2)))**-1
    
    return term1 * term2 * term3

# LDDE Luminosity Function for Fermi-LAT BL Lacs from: https://arxiv.org/pdf/1310.0006.
# Parameters are listed in the row labeled LDDE_1 in Table 3.
# A in units of Mpc^-3 erg^-1 s 
# L_c in units of erg s^-1

def LDDE_BLL1(L,z, A=9.2*1e2*10**-13, L_c=2.43*10**48, gam1=1.12, gam2=3.71, z_c=1.67, p1=4.5, p2=-12.88, alpha=4.46e-2, mu=2.12, beta=6.04e-2, sigma=0.26,tau=0):
    
    gamma_min = 1.45
    gamma_max = 2.8
    
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    p1 = p1 + tau*(np.log10(L) - 46)
    term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**p1) + (((1+z)/(1+(z_c*(L/10**48)**alpha)))**p2))**-1
    
    mu_eff = mu + beta*(np.log10(L) - 46)
    norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
    norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
    
    gamma_integral = sigma * np.sqrt(2*np.pi) * 0.5 * (erf(norm_max) - erf(norm_min))
    
    return term1 * term2 * term3 * gamma_integral

def LDDE_BLL2(L,z, A=3.39*1e4*10**-13, L_c=0.28*10**48, gam1=0.27, gam2=1.86, z_c=1.34, p1=2.24, p2=-7.37, alpha=4.53e-2, mu=2.1, beta=6.46e-2, sigma=0.26,tau=4.92):
    
    gamma_min = 1.45
    gamma_max = 2.8
    
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    p1 = p1 + tau*(np.log10(L) - 46)
    term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**p1) + (((1+z)/(1+(z_c*(L/10**48)**alpha)))**p2))**-1
    
    mu_eff = mu + beta*(np.log10(L) - 46)
    norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
    norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
    
    gamma_integral = sigma * np.sqrt(2*np.pi) * 0.5 * (erf(norm_max) - erf(norm_min))
    
    return term1 * term2 * term3 * gamma_integral

# Integrand for cumulative redshift distribution N(>z) calculation.

def dN_dz(z, F_lim, E, LDDE, log_L_max, i, Lum_bin, nuLnu):
    log_L_min = np.log10(K_Corrected_luminosity(F_lim, nuLnu, z, E))
    
    if LDDE == LDDE_BLL1:
        if nuLnu.func == nuLnu_SBPL:
            log_L_min = max(Lum_bin[0], log_L_min)
    
        elif nuLnu.func == nuLnu_LP:
            log_L_min = max(Lum_bin[0], log_L_min)
    else:
        if i != 0:
            log_L_min = max(Lum_bin[0], log_L_min)

    if log_L_min >= log_L_max:
        return 0.0

    def integrand(log_L):
        L = 10**log_L
        return L * np.log(10) * LDDE(L, z)

    lum_integral = quad(integrand, log_L_min, log_L_max)[0]
    return lum_integral * dVdz_dOmega(z)

# Integrand for cumulative source counts N(>S) calculation.

def N_S_integrand(nuLnu, LDDE, Energy, z_min, z_max, F_lim, Lum_bin, log_L_max, i):
    
    z_grid = np.linspace(z_min, z_max, 300)
    
    dL_grid = np.array([d_L(zz) for zz in z_grid])   
    dL_interp = interp1d(z_grid, dL_grid, kind='cubic', fill_value='extrapolate')
    
    kcorr_grid = np.array([Monochromatic_k_Correction(nuLnu, zz, Energy) for zz in z_grid])
    kcorr_interp = interp1d(z_grid, kcorr_grid, kind='cubic', fill_value='extrapolate')
    
    def integrand_LDDE(log_L, z):
        L = 10**log_L
        return L * np.log(10) * LDDE(L, z) * dVdz_dOmega(z) 
    
    def log_L_min_of_z(z):
        d_L_z = dL_interp(z)
        log_L_min = np.log10(4*np.pi*(d_L_z*3.085677581e24)**2 * F_lim * kcorr_interp(z))
        if LDDE == LDDE_BLL1:
            if nuLnu.func == nuLnu_SBPL:
                log_L_min = max(Lum_bin[0], log_L_min)

            elif nuLnu.func == nuLnu_LP:
                log_L_min = max(Lum_bin[0], log_L_min)
        else:
            if i != 0:
                log_L_min = max(Lum_bin[0], log_L_min)
                
        return log_L_min
    
    def log_L_max_of_z(z):
            return log_L_max
        
    def inner_integral(z):
        lo = log_L_min_of_z(z)
        hi = log_L_max_of_z(z)
        if lo >= hi:
            return 0.0
        return quad(integrand_LDDE, lo, hi, args=(z,), epsabs=1e-6, epsrel=1e-4)[0]
    
    return inner_integral