import yaml
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np

from astropy.table import Table
from scipy.integrate import quad
from functools import partial

from src.LDDE_Luminosity_Functions import LDDE1, LDDE2, LDDE3, LDDE_BLL1, N_S_integrand, LDDE_BLL2
from src.SED_Models import nuLnu_LP, nuLnu_SBPL

start = time.time()

config_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Scripts/Configs/'

parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
out_dir = config['Out_dir'] + 'N_S_Count_Plots/'
    
Catalog = Table.read(config['Catalog_dir'] + config['Catalog_Name'])
LP_FSRQ_Mask = (Catalog[config['columns']['SED_Type']] == 'LP') & (Catalog[config['columns']['Class']] == 'FSRQ')
FSRQ_Catalog = Catalog[LP_FSRQ_Mask]

SBPL_BLL_Mask = (Catalog[config['columns']['SED_Type']] == 'SBPL') &  (Catalog[config['columns']['Class']] == 'BLL')
BLL_Catalog = Catalog[SBPL_BLL_Mask]

LP_BLL_Mask = (Catalog[config['columns']['SED_Type']] == 'LP') & (Catalog[config['columns']['Class']] == 'BLL')
BLL_LP_Catalog = Catalog[LP_BLL_Mask]

F_lim = list(np.logspace(-13, -9, 30)) + [1e-12, 1e-11, 1e-10]
F_lim.sort()

GRAMS_Flux_Bins = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

BLL_cmap = plt.get_cmap('Reds')
BLL_colors = [BLL_cmap(x) for x in np.linspace(0.4, 0.9, 3)]

print("Calculating counts for LP type BLLs...")               
N_per_sensitivity_per_band_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]
N_per_sensitivity_per_band_LDDE_BLL2_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]


for i in range(len(BLL_LP_Catalog)):
    Lum_bin = BLL_LP_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = BLL_LP_Catalog[config['columns']['L0']].tolist()[i]
    z = BLL_LP_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = BLL_LP_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = BLL_LP_Catalog[config['columns']['LP_Beta']].tolist()[i]
    print(i)
    nuLnu_LP_bin = partial(nuLnu_LP, L0=L0, alpha=Alpha, beta=Beta, E0=0.017)
    
    for f in range(len(GRAMS_Flux_Bins)):  
        for s in range(len(F_lim)):
            F_lim_band = F_lim[s]
            
            if i != len(BLL_LP_Catalog) - 1:
                log_L_max = Lum_bin[1]
            elif i == len(BLL_LP_Catalog) - 1:
                log_L_max = 52
            
            N_S_integrand_bll1 = N_S_integrand(nuLnu=nuLnu_LP_bin, LDDE=LDDE_BLL1, Energy=GRAMS_Flux_Bins[f], z_min=0.03, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max, i=i)
            N_LDDE_BLL1 = quad(N_S_integrand_bll1, 0.03, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
            
            N_S_integrand_bll2 = N_S_integrand(nuLnu=nuLnu_LP_bin, LDDE=LDDE_BLL2, Energy=GRAMS_Flux_Bins[f], z_min=0.03, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max, i=i)
            N_LDDE_BLL2 = quad(N_S_integrand_bll2, 0.03, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi

            print(N_LDDE_BLL1, N_LDDE_BLL2)
            N_per_sensitivity_per_band_LDDE_BLL1_LP[s][f] += N_LDDE_BLL1
            N_per_sensitivity_per_band_LDDE_BLL2_LP[s][f] += N_LDDE_BLL2

print("Calculating counts for HSP type BLLs...")
N_per_sensitivity_per_band_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]
N_per_sensitivity_per_band_LDDE_BLL2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]


for i in range(len(BLL_Catalog)):
    Lum_bin = BLL_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 =  BLL_Catalog[config['columns']['L0']].tolist()[i]
    Eb =  BLL_Catalog[config['columns']['Break_Energy']].tolist()[i]
    Index1 =  BLL_Catalog[config['columns']['Index1']].tolist()[i]
    Index2 =  BLL_Catalog[config['columns']['Index2']].tolist()[i]
    Beta = BLL_Catalog[config['columns']['Curvature']].tolist()[i]
    z = BLL_Catalog[config['columns']['Redshift']].tolist()[i]
    print(i)
    nuLnu_SBPL_bin = partial(nuLnu_SBPL, L0=L0, Eb=Eb, alpha1=Index1, alpha2=Index2, s=Beta)
    
    for f in range(len(GRAMS_Flux_Bins)):
        
        for s in range(len(F_lim)):
            F_lim_band = F_lim[s]
            log_L_max = Lum_bin[1]
            
            N_S_integrand_bll = N_S_integrand(nuLnu=nuLnu_SBPL_bin, LDDE=LDDE_BLL1, Energy=GRAMS_Flux_Bins[f], z_min=0.03, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max, i=i)
            N_LDDE_BLL1 = quad(N_S_integrand_bll, 0.03, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
            
            N_S_integrand_bll2 = N_S_integrand(nuLnu=nuLnu_SBPL_bin, LDDE=LDDE_BLL2, Energy=GRAMS_Flux_Bins[f], z_min=0.03, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max, i=i)
            N_LDDE_BLL2 = quad(N_S_integrand_bll2, 0.03, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
            
            print(N_LDDE_BLL1, N_LDDE_BLL2)
            N_per_sensitivity_per_band_LDDE_BLL1[s][f] += N_LDDE_BLL1
            N_per_sensitivity_per_band_LDDE_BLL2[s][f] += N_LDDE_BLL2

print("Calculating counts for FSRQs...")
N_per_sensitivity_per_band_LDDE1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]
N_per_sensitivity_per_band_LDDE2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]
N_per_sensitivity_per_band_LDDE3 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(F_lim))]

FSRQ_cmap = plt.get_cmap('Blues')
FSRQ_colors = [FSRQ_cmap(x) for x in np.linspace(0.4, 0.9, 5)]

for i in range(len(FSRQ_Catalog)):
    Lum_bin = FSRQ_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = FSRQ_Catalog[config['columns']['L0']].tolist()[i]
    z = FSRQ_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = FSRQ_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = FSRQ_Catalog[config['columns']['LP_Beta']].tolist()[i]
    
    nuLnu_LP_bin = partial(nuLnu_LP, L0=L0, alpha=Alpha, beta=Beta, E0=0.017)

    for f in range(len(GRAMS_Flux_Bins)):

        for s in range(len(F_lim)):
            F_lim_band = F_lim[s]

            if i != len(FSRQ_Catalog)-1:
                log_L_max1 = Lum_bin[1]
                log_L_max2 = Lum_bin[1]
                log_L_max3 = Lum_bin[1]
            elif i == len(FSRQ_Catalog) - 1:
                log_L_max1 = 50
                log_L_max2 = np.log10(7.3e48)
                log_L_max3 = 50
                
            N_S_integrand1 = N_S_integrand(nuLnu=nuLnu_LP_bin, LDDE=LDDE1, Energy=GRAMS_Flux_Bins[f], z_min=0.0, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max1, i=i)
            N_S_integrand2 = N_S_integrand(nuLnu=nuLnu_LP_bin, LDDE=LDDE2, Energy=GRAMS_Flux_Bins[f], z_min=0.0001, z_max=5, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max2, i=i)
            N_S_integrand3 = N_S_integrand(nuLnu=nuLnu_LP_bin, LDDE=LDDE3, Energy=GRAMS_Flux_Bins[f], z_min=0.0, z_max=6, F_lim=F_lim_band, Lum_bin=Lum_bin, log_L_max=log_L_max3, i=i)

            N_LDDE1 = quad(N_S_integrand1, 0.0, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
            N_LDDE2 = quad(N_S_integrand2, 0.0001, 5, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
            N_LDDE3 = quad(N_S_integrand3, 0.0, 6, epsabs=1e-6, epsrel=1e-4)[0] * 4 * np.pi
        
            N_per_sensitivity_per_band_LDDE1[s][f] += N_LDDE1
            N_per_sensitivity_per_band_LDDE2[s][f] += N_LDDE2
            N_per_sensitivity_per_band_LDDE3[s][f] += N_LDDE3
            print(N_LDDE1, N_LDDE2, N_LDDE3)


N_S_bins_LDDE1 = list(zip(*N_per_sensitivity_per_band_LDDE1))
N_S_bins_LDDE2 = list(zip(*N_per_sensitivity_per_band_LDDE2))
N_S_bins_LDDE3 = list(zip(*N_per_sensitivity_per_band_LDDE3))
N_S_bins_LDDE_BLL1 = list(zip(*N_per_sensitivity_per_band_LDDE_BLL1))
N_S_bins_LDDE_BLL1_LP = list(zip(*N_per_sensitivity_per_band_LDDE_BLL1_LP))
N_S_bins_LDDE_BLL2 = list(zip(*N_per_sensitivity_per_band_LDDE_BLL2))
N_S_bins_LDDE_BLL2_LP = list(zip(*N_per_sensitivity_per_band_LDDE_BLL2_LP))

N_S_bins_LDDE_BLL1_all = []
N_S_bins_LDDE_BLL2_all = []
for s in range(len(N_S_bins_LDDE1)):
    N_S_bins_LDDE_BLL1_all.append([N_S_bins_LDDE_BLL1[s][f] + N_S_bins_LDDE_BLL1_LP[s][f] for f in range(len(N_S_bins_LDDE_BLL1[s]))])
    N_S_bins_LDDE_BLL2_all.append([N_S_bins_LDDE_BLL2[s][f] + N_S_bins_LDDE_BLL2_LP[s][f] for f in range(len(N_S_bins_LDDE_BLL1[s]))])
    
LF_Data_Table = Table([F_lim, N_S_bins_LDDE1[0], N_S_bins_LDDE1[1], N_S_bins_LDDE1[2], N_S_bins_LDDE1[3], N_S_bins_LDDE1[4], N_S_bins_LDDE1[5], N_S_bins_LDDE1[6], N_S_bins_LDDE1[7], N_S_bins_LDDE1[8], N_S_bins_LDDE2[0], N_S_bins_LDDE2[1], N_S_bins_LDDE2[2], N_S_bins_LDDE2[3], N_S_bins_LDDE2[4], N_S_bins_LDDE2[5], N_S_bins_LDDE2[6], N_S_bins_LDDE2[7], N_S_bins_LDDE2[8], N_S_bins_LDDE3[0], N_S_bins_LDDE3[1], N_S_bins_LDDE3[2], N_S_bins_LDDE3[3], N_S_bins_LDDE3[4], N_S_bins_LDDE3[5], N_S_bins_LDDE3[6], N_S_bins_LDDE3[7], N_S_bins_LDDE3[8], N_S_bins_LDDE_BLL1_all[0], N_S_bins_LDDE_BLL1_all[1], N_S_bins_LDDE_BLL1_all[2], N_S_bins_LDDE_BLL1_all[3], N_S_bins_LDDE_BLL1_all[4], N_S_bins_LDDE_BLL1_all[5], N_S_bins_LDDE_BLL1_all[6], N_S_bins_LDDE_BLL1_all[7], N_S_bins_LDDE_BLL1_all[8], N_S_bins_LDDE_BLL2_all[0], N_S_bins_LDDE_BLL2_all[1], N_S_bins_LDDE_BLL2_all[2], N_S_bins_LDDE_BLL2_all[3], N_S_bins_LDDE_BLL2_all[4], N_S_bins_LDDE_BLL2_all[5], N_S_bins_LDDE_BLL2_all[6], N_S_bins_LDDE_BLL2_all[7], N_S_bins_LDDE_BLL2_all[8]], names=['Sensitivity', 'Toda_0_2', 'Toda_0_5','Toda_1_0', 'Toda_2_0', 'Toda_5_0', 'Toda_10_0', 'Toda_20_0', 'Toda_50_0', 'Toda_100_0', 'Rajguru_0_2', 'Rajguru_0_5', 'Rajguru_1_0', 'Rajguru_2_0', 'Rajguru_5_0', 'Rajguru_10_0','Rajguru_20_0', 'Rajguru_50_0', 'Rajguru_100_0', 'Marcotulli_0_2', 'Marcotulli_0_5', 'Marcotulli_1_0', 'Marcotulli_2_0', 'Marcotulli_5_0', 'Marcotulli_10_0', 'Marcotulli_20_0', 'Marcotulli_50_0', 'Marcotulli_100_0', 'Ajello_BLL_0_2', 'Ajello_BLL_0_5', 'Ajello_BLL_1_0', 'Ajello_BLL_2_0', 'Ajello_BLL_5_0', 'Ajello_BLL_10_0', 'Ajello_BLL_20_0', 'Ajello_BLL_50_0', 'Ajello_BLL_100_0', 'Ajello_BLL2_0_2', 'Ajello_BLL2_0_5', 'Ajello_BLL2_1_0', 'Ajello_BLL2_2_0', 'Ajello_BLL2_5_0', 'Ajello_BLL2_10_0', 'Ajello_BLL2_20_0', 'Ajello_BLL2_50_0', 'Ajello_BLL2_100_0'])
LF_Data_Table.write(config['Catalog_dir'] + 'N_S_LF_Output.fits', overwrite=True)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")