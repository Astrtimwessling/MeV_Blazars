import yaml
import matplotlib.pyplot as plt
import time
import os
import argparse
import numpy as np
import math
import astropy.units as u

from astropy.table import Table
from scipy.integrate import quad, dblquad, tplquad
from scipy.optimize import brentq
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import FlatLambdaCDM
from scipy.special import erf


# Inputs: F_lim, Int_Flux, Model, Class, 

start = time.time()

omega_m = 0.3
c = 3 * (10**5) # km/s
H_0 = 70 # km/s

def Luminosity_distance(z):
    def f(x):
        return 1 / np.sqrt(omega_m * (1 + x)**3 + 1 - omega_m)

    integral, _ = quad(f, 0, z)
    return (1 + z) * (c / H_0) * integral 

def luminosity(d_L, flux):
    L = 4 * math.pi * (d_L * 3.085677581e24)**2 * flux 


def int_flux(dnde, E1, E2):
    return quad(lambda E: E * dnde(E), E1, E2)[0] * 1.602176634e-6

def LDDE(A, L, L_c, gam1, gam2, z_c, z, p1, p2, alpha):
    term1 = (A / (np.log(10) * L))
    term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
    term3 = ((((1+z)/(1+(z_c*(L/10**47.5)**alpha)))**p1) +((1+z)/(1+(z_c*(L/10*47.5)**alpha)**p2)))**-1
    return term1 * term2 * term3

def dVdz_dOmega(z):
    return cosmo.differential_comoving_volume(z).value

def k_correction(nuLnu, z, E1, E2):
        integrand = lambda E: nuLnu(E) / E

        numerator = quad(integrand, E1*(1+z), E2*(1+z))[0]
        denominator = quad(integrand, E1, E2)[0]

        return numerator / denominator 

config_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Scripts/Configs/'

parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
out_dir = config['Out_dir']
    
Catalog = Table.read(config['Catalog_dir'] + config['Catalog_Name'])
LP_FSRQ_Mask = (Catalog[config['columns']['SED_Type']] == 'LP') & (Catalog[config['columns']['Class']] == 'FSRQ')
FSRQ_Catalog = Catalog[LP_FSRQ_Mask]

SBPL_BLL_Mask = (Catalog[config['columns']['SED_Type']] == 'SBPL') &  (Catalog[config['columns']['Class']] == 'BLL')
BLL_Catalog = Catalog[SBPL_BLL_Mask]

LP_BLL_Mask = (Catalog[config['columns']['SED_Type']] == 'LP') & (Catalog[config['columns']['Class']] == 'BLL')
BLL_LP_Catalog = Catalog[LP_BLL_Mask]

F_lim = list(np.logspace(-13, -9, 30)) + [1e-12, 1e-11, 1e-10]
F_lim.sort()
print(F_lim)
logL_bins = np.arange(43, 49, 0.5)
GRAMS_Flux_Bins = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
GRAMS_Reduced_Flux_Bins = [0.2,0.5,1,5,10,50,100]

BLL_cmap = plt.get_cmap('Reds')
BLL_colors = [BLL_cmap(x) for x in np.linspace(0.4, 0.9, 3)]

print("Calculating counts for HSP type BLLs...")
N_per_sensitivity_per_band_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Reduced_Flux_Bins)-1)] for _ in range(len(F_lim))]

for i in range(len(BLL_Catalog)):
    Lum_bin = BLL_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 =  BLL_Catalog[config['columns']['L0']].tolist()[i]
    Eb =  BLL_Catalog[config['columns']['Break_Energy']].tolist()[i]
    Index1 =  BLL_Catalog[config['columns']['Index1']].tolist()[i]
    Index2 =  BLL_Catalog[config['columns']['Index2']].tolist()[i]
    z = BLL_Catalog[config['columns']['Redshift']].tolist()[i]
    
    def nuLnu_SBPL(E, L0=L0, Eb=Eb, alpha1=Index1, alpha2=Index2, s=-1):
        term = (E / Eb)
        return L0 * term**(-alpha1) * (1 + term**((alpha2 - alpha1)/s))**(-s)

    E_Space = np.logspace(-2, 6, 100)
    dLde_Space = [nuLnu_SBPL(E) for E in E_Space]
    
    plt.plot(E_Space, dLde_Space, label=f'{Lum_bin[0]} <= log(Lum) < {Lum_bin[1]}', color=BLL_colors[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'$E^2 \frac{dL}{dE}$ (erg s$^{-1}$)')
    if i == len(BLL_Catalog)-1:
        print("Saving Averaged BLL SBPL SED Plots...")
        plt.savefig(out_dir + 'Binned_Lum_SED_HSP_BLL.png',bbox_inches='tight',dpi=300)
        plt.show()
        plt.close()
        
    for f in range(len(GRAMS_Reduced_Flux_Bins)-1):
        k_correction_band = k_correction(nuLnu_SBPL, z, GRAMS_Reduced_Flux_Bins[f], GRAMS_Reduced_Flux_Bins[f+1])
        
        def LDDE_BLL1(L,z, A=9.2*1e2*10**-13, L_c=2.43*10**48, gam1=1.12, gam2=3.71, z_c=1.67, p1=4.5, p2=-12.88, alpha=4.46e-2, mu=2.12, beta=6.04e-2, sigma=0.26,tau=0):
                term1 = (A / (np.log(10) * L))
                term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
                p1 = p1 + tau*(np.log10(L) - 46)
                term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**p1) +((1+z)/(1+(z_c*(L/10**48)**alpha))**p2))**-1
                
                mu_eff = mu + beta*(np.log10(L) - 46)
                norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
                norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
                
                gamma_integral = 0.5 * (erf(norm_max) - erf(norm_min))
                
                return term1 * term2 * term3 * gamma_integral
            
        d_L_bin = Luminosity_distance(z)
        
        for s in range(len(F_lim)):
                F_lim_band = F_lim[s]
                k_corr_lum_band = 4*math.pi* (d_L_bin * 3.085677581e24)**2 * F_lim_band * k_correction_band
                
                def integrand_LDDE_BLL1(z, log_L):
                    L = 10**log_L
                    return L * LDDE_BLL1(L, z) * dVdz_dOmega(z) 
                
                log_L_min = np.log10(k_corr_lum_band)
                log_L_max = 50
                gamma_min = 1.45
                gamma_max = 2.8
                
                N_LDDE_BLL1 = dblquad(integrand_LDDE_BLL1, log_L_min, log_L_max, lambda log_L: 0, lambda log_L: 6, epsabs=1e-10, epsrel=1e-6)[0]
                N_total_LDDE_BLL1 = N_LDDE_BLL1 * 4 * np.pi
            
                N_per_sensitivity_per_band_LDDE_BLL1[s][f] += N_total_LDDE_BLL1

print("Calculating counts for LP type BLLs...")               
N_per_sensitivity_per_band_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Reduced_Flux_Bins)-1)] for _ in range(len(F_lim))]

for i in range(len(BLL_LP_Catalog)):
    Lum_bin = BLL_LP_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = BLL_LP_Catalog[config['columns']['L0']].tolist()[i]
    z = BLL_LP_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = BLL_LP_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = BLL_LP_Catalog[config['columns']['LP_Beta']].tolist()[i]
    
    def nuLnu_LP(E, L0=L0, alpha=Alpha, beta=Beta, E0=1.0):
        return L0 * (E / E0) ** (2 - alpha - beta * np.log(E / E0))
    
    E_Space = np.logspace(-2, 6, 100)
    dnde_Space = [nuLnu_LP(E) for E in E_Space]
    
    plt.plot(E_Space, dnde_Space, label=f'{Lum_bin[0]} <= log(Lum) < {Lum_bin[1]}',color=BLL_colors[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'$E^2 \frac{dL}{dE}$ (erg s$^{-1}$)')
    if i == len(BLL_LP_Catalog)-1:
        print("Saving Averaged BLL LP SED Plots...")
        plt.savefig(out_dir + 'Binned_Lum_SED_BLL_LP.png',bbox_inches='tight',dpi=300)
        plt.show()
        plt.close()
        
    for f in range(len(GRAMS_Reduced_Flux_Bins)-1):
        K_correction_band = k_correction(nuLnu_LP, z, GRAMS_Reduced_Flux_Bins[f], GRAMS_Reduced_Flux_Bins[f+1])
        
        def LDDE_BLL1(L,z, A=9.2*(10**-13)*1e2, L_c=2.43*10**48, gam1=1.12, gam2=3.71, z_c=1.67, p1=4.5, p2=-12.88, alpha=4.46e-2, mu=2.12, beta=6.04e-2, sigma=0.26,tau=0):
                term1 = (A / (np.log(10) * L))
                term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
                p1 = p1 + tau*(np.log10(L) - 46)
                term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**p1) +((1+z)/(1+(z_c*(L/10**48)**alpha))**p2))**-1
                
                mu_eff = mu + beta*(np.log10(L) - 46)
                norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
                norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
                
                gamma_integral = 0.5 * (erf(norm_max) - erf(norm_min))
                
                return term1 * term2 * term3 * gamma_integral
            
        for s in range(len(F_lim)):
                F_lim_band = F_lim[s]
                k_corr_lum_band = 4*math.pi* (d_L_bin * 3.085677581e24)**2 * F_lim_band * K_correction_band
                
                def integrand_LDDE_BLL1(z, log_L):
                    L = 10**log_L
                    return L * LDDE_BLL1(L, z) * dVdz_dOmega(z) 
                
                log_L_min = np.log10(k_corr_lum_band)
                log_L_max = 52
                gamma_min = 1.45
                gamma_max = 2.8
                
                N_LDDE_BLL1 = dblquad(integrand_LDDE_BLL1, log_L_min, log_L_max, lambda log_L: 0.03, lambda log_L: 6, epsabs=1e-10, epsrel=1e-6)[0]
                N_total_LDDE_BLL1 = N_LDDE_BLL1 * 4 * np.pi
                #print(N_total_LDDE_BLL1)
                N_per_sensitivity_per_band_LDDE_BLL1_LP[s][f] += N_total_LDDE_BLL1
    
#print(FSRQ_Lum_bins)
print("Calculating counts for FSRQs...")
N_per_sensitivity_per_band_LDDE1 = [[0 for _ in range(len(GRAMS_Reduced_Flux_Bins)-1)] for _ in range(len(F_lim))]
N_per_sensitivity_per_band_LDDE2 = [[0 for _ in range(len(GRAMS_Reduced_Flux_Bins)-1)] for _ in range(len(F_lim))]

FSRQ_cmap = plt.get_cmap('Blues')
FSRQ_colors = [FSRQ_cmap(x) for x in np.linspace(0.4, 0.9, 5)]

for i in range(len(FSRQ_Catalog)):
    Lum_bin = FSRQ_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = FSRQ_Catalog[config['columns']['L0']].tolist()[i]
    z = FSRQ_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = FSRQ_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = FSRQ_Catalog[config['columns']['LP_Beta']].tolist()[i]
    
    def nuLnu_LP(E, L0=L0, alpha=Alpha, beta=Beta, E0=1.0):
        return L0 * (E / E0) ** (2 - alpha - beta * np.log(E / E0))
    
    E_Space = np.logspace(-2, 6, 100)
    dLde_Space = [nuLnu_LP(E) for E in E_Space]
    
    plt.plot(E_Space, dLde_Space, label=f'{Lum_bin[0]} <= log(Lum) < {Lum_bin[1]}', color=FSRQ_colors[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'$E^2 \frac{dL}{dE}$ (erg s$^{-1}$)')
    if i == len(FSRQ_Catalog)-1:
        print("Saving Averaged FSRQ SED Plots...")
        plt.savefig(out_dir + 'Binned_Lum_SED_FSRQ.png',bbox_inches='tight',dpi=300)
        plt.show()
        plt.close()
    
    # Loop over energy bands
    for f in range(len(GRAMS_Reduced_Flux_Bins)-1):
        K_correction_band = k_correction(nuLnu_LP, z, GRAMS_Reduced_Flux_Bins[f], GRAMS_Reduced_Flux_Bins[f+1])
        print(K_correction_band)
        
        def LDDE1(L,z, A=10**-13.02,L_c=10**51, gam1=5, gam2=0.8, z_c=1.36, p1=3.68, p2=-7.7, alpha=.42):
            term1 = (A / (np.log(10) * L))
            term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
            term3 = ((((1+z)/(1+(z_c*(L/10**47.5)**alpha)))**p1) +((1+z)/(1+(z_c*(L/10**47.5)**alpha))**p2))**-1
            
            return term1 * term2 * term3
        
        def LDDE2(L,z, A=18878.4*10**-13, L_c=1.09*10**48, gam1=0.29, gam2=1.63, z_c=2.05, p1=3.5, p2=-9, alpha=0.22, mu=2.42, beta=0.026, sigma=0.182):
            term1 = (A / (np.log(10) * L))
            term2 = (((L/L_c)**gam1) + (L/L_c)**gam2)**(-1)
            term3 = ((((1+z)/(1+(z_c*(L/10**48)**alpha)))**p1) +((1+z)/(1+(z_c*(L/10**48)**alpha))**p2))**-1
            
            mu_eff = mu + beta*(np.log10(L) - 46)
            norm_min = (gamma_min - mu_eff) / (sigma * np.sqrt(2))
            norm_max = (gamma_max - mu_eff) / (sigma * np.sqrt(2))
            
            gamma_integral = 0.5 * (erf(norm_max) - erf(norm_min))
            
            return term1 * term2 * term3 * gamma_integral
        
        for s in range(len(F_lim)):
            F_lim_band = F_lim[s]
            k_corr_lum_band = 4*math.pi* (d_L_bin * 3.085677581e24)**2 * F_lim_band * K_correction_band
            log_L_min = np.log10(k_corr_lum_band)
            
            log_L_max1 = 50
            log_L_max2 = np.log10(7.3e48)
            gamma_min = 1
            gamma_max = 4
            
            def integrand_LDDE1(z, log_L):
                L = 10**log_L
                return L * LDDE1(L, z) * dVdz_dOmega(z)  # L is the Jacobian for dlogL
            
            def integrand_LDDE2(z, log_L):
                L = 10**log_L
                return L * LDDE2(L, z) * dVdz_dOmega(z)
            
            # Now integrate over log(L) instead of L

            N_LDDE1 = dblquad(integrand_LDDE1, log_L_min, log_L_max1, lambda log_L: 0, lambda log_L: 6, epsabs=1e-10, epsrel=1e-6)[0]
            N_total_LDDE1 = N_LDDE1 * 4 * np.pi
            #print(N_total_LDDE1)
            N_per_sensitivity_per_band_LDDE1[s][f] += N_total_LDDE1
            
            N_LDDE2 = dblquad(integrand_LDDE2, log_L_min, log_L_max2, lambda log_L: 0.0001, lambda log_L: 5, epsabs=1e-10, epsrel=1e-6)[0]
            N_total_LDDE2 = N_LDDE2 * 4 * np.pi
            #print(N_total_LDDE2)
            N_per_sensitivity_per_band_LDDE2[s][f] += N_total_LDDE2

N_S_bins_LDDE1 = list(zip(*N_per_sensitivity_per_band_LDDE1))
N_S_bins_LDDE2 = list(zip(*N_per_sensitivity_per_band_LDDE2))
N_S_bins_LDDE_BLL1 = list(zip(*N_per_sensitivity_per_band_LDDE_BLL1))
N_S_bins_LDDE_BLL1_LP = list(zip(*N_per_sensitivity_per_band_LDDE_BLL1_LP))

N_S_bins_LDDE_BLL1_all = []
for s in range(len(N_S_bins_LDDE1)):
    N_S_bins_LDDE_BLL1_all.append([N_S_bins_LDDE_BLL1[s][f] + N_S_bins_LDDE_BLL1_LP[s][f] for f in range(len(N_S_bins_LDDE_BLL1[s]))])
    plt.figure()
    plt.plot(F_lim, N_S_bins_LDDE1[s], linestyle='--', color='blue', label='Toda, K. (2020) BAT FSRQ LDDE')
    plt.plot(F_lim, N_S_bins_LDDE2[s], linestyle='--', color='purple', label='Rajguru, G. (2025) LAT FSRQ LDDE')
    plt.plot(F_lim, N_S_bins_LDDE_BLL1_all[s], linestyle='--', color='red', label='Ajello, M. (2014) LAT BLL LDDE')
    plt.xlabel(f'Sensitivity ({GRAMS_Reduced_Flux_Bins[s]}-{GRAMS_Reduced_Flux_Bins[s+1]} MeV)' r'(erg cm$^{-2}$ s$^{-1}$)')
    plt.ylabel(r'N(>S) [4$\pi$ str]')
    plt.xscale('log')
    plt.yscale('log')
    #plt.title(f'N(>S) {GRAMS_Reduced_Flux_Bins}')
    plt.legend()
    plt.savefig(out_dir + f'N_S_{GRAMS_Reduced_Flux_Bins[s]}-{GRAMS_Reduced_Flux_Bins[s+1]}_MeV.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
LF_Data_Table = Table([F_lim, N_S_bins_LDDE1[0], N_S_bins_LDDE1[1], N_S_bins_LDDE1[2], N_S_bins_LDDE1[3], N_S_bins_LDDE1[4], N_S_bins_LDDE1[5], N_S_bins_LDDE2[0], N_S_bins_LDDE2[1], N_S_bins_LDDE2[2], N_S_bins_LDDE2[3], N_S_bins_LDDE2[4], N_S_bins_LDDE2[5], N_S_bins_LDDE_BLL1_all[0], N_S_bins_LDDE_BLL1_all[1], N_S_bins_LDDE_BLL1_all[2], N_S_bins_LDDE_BLL1_all[3], N_S_bins_LDDE_BLL1_all[4], N_S_bins_LDDE_BLL1_all[5]], names=['Sensitivity', 'Toda_0_2_0_5', 'Toda_0_5_1_0','Toda_1_0_5_0', 'Toda_5_0_10_0', 'Toda_10_0_50_0', 'Toda_50_0_100_0', 'Rajguru_0_2_0_5', 'Rajguru_0_5_1_0', 'Rajguru_1_0_5_0', 'Rajguru_5_0_10_0', 'Rajguru_10_0_50_0', 'Rajguru_50_0_100_0', 'Ajello_BLL_0_2_0_5', 'Ajello_BLL_0_5_1_0', 'Ajello_BLL_1_0_5_0', 'Ajello_BLL_5_0_10_0', 'Ajello_BLL_10_0_50_0', 'Ajello_BLL_50_0_100_0'])
LF_Data_Table.write(config['Catalog_dir'] + 'Luminosity_Function_Results.fits', overwrite=True)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")
            

