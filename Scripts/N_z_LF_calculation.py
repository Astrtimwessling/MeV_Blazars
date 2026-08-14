import yaml
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np

from astropy.table import Table, hstack
from scipy.integrate import quad
from functools import partial

from src.LDDE_Luminosity_Functions import LDDE1, LDDE2, LDDE3, LDDE_BLL1, dN_dz, LDDE_BLL2
from src.SED_Models import nuLnu_LP, nuLnu_SBPL

start = time.time()

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

Sensitivity_Data = Table.read(config['Catalog_dir'] + 'Sensitivity_Data_Table.fits')

GRAMS_Balloon_Energy = Sensitivity_Data['Energy'].tolist()

GRAMS_Balloon_Sensitivity = Sensitivity_Data['GRAMS_Balloon_Sensitivity'].tolist()
GRAMS_Satellite_Sensitivity = Sensitivity_Data['GRAMS_Satellite_Sensitivity'].tolist()
newASTROGAM_sensitivity = Sensitivity_Data['newASTROGAM_Sensitivity'].tolist()
AMEGO_X_sensitivity = Sensitivity_Data['AMEGO_X_Sensitivity'].tolist()

z_space = list(np.linspace(0, 8, 100)) + [1, 2, 3, 4, 5, 6, 7]
z_space_bll = list(np.linspace(0, 6, 100)) + [1, 1.5, 2.0, 2.5, 3, 4, 5]
z_space_bll.sort()
z_space.sort()

GRAMS_Flux_Bins = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

FSRQ_cmap = plt.get_cmap('Blues')
FSRQ_colors = [FSRQ_cmap(x) for x in np.linspace(0.4, 0.9, 5)]

print("Calculating counts for FSRQs...")

N_z_GRAMS_Balloon_LDDE_FSRQ1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]   
N_z_GRAMS_Satellite_LDDE_FSRQ1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_newASTROGAM_LDDE_FSRQ1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_AMEGO_X_LDDE_FSRQ1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))] 

N_z_GRAMS_Balloon_LDDE_FSRQ2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]   
N_z_GRAMS_Satellite_LDDE_FSRQ2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_newASTROGAM_LDDE_FSRQ2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_AMEGO_X_LDDE_FSRQ2 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))] 

N_z_GRAMS_Balloon_LDDE_FSRQ3 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]   
N_z_GRAMS_Satellite_LDDE_FSRQ3 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_newASTROGAM_LDDE_FSRQ3 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_AMEGO_X_LDDE_FSRQ3 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))] 

for i in range(len(FSRQ_Catalog)):
    Lum_bin = FSRQ_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = FSRQ_Catalog[config['columns']['L0']].tolist()[i]
    z = FSRQ_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = FSRQ_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = FSRQ_Catalog[config['columns']['LP_Beta']].tolist()[i]
    
    nuLnu_LP_bin = partial(nuLnu_LP, L0=L0, alpha=Alpha, beta=Beta, E0=0.017)

    for f in range(len(GRAMS_Flux_Bins)):
        for zi in range(len(z_space)):
            if i != len(FSRQ_Catalog)-1:
                log_L_max1 = Lum_bin[1]
                log_L_max2 = Lum_bin[1]
            elif i == len(FSRQ_Catalog) - 1:
                log_L_max1 = 50
                log_L_max2 = np.log10(7.3e48)
                
            dN_dz1 = partial(dN_dz, LDDE=LDDE1, log_L_max=log_L_max1, i=i, Lum_bin=Lum_bin, nuLnu=nuLnu_LP_bin)
            dN_dz2 = partial(dN_dz, LDDE=LDDE2, log_L_max=log_L_max2, i=i, Lum_bin=Lum_bin, nuLnu=nuLnu_LP_bin)
            dN_dz3 = partial(dN_dz, LDDE=LDDE3, log_L_max=log_L_max1, i=i, Lum_bin=Lum_bin, nuLnu=nuLnu_LP_bin)
        
            N_z_GRAMS_Balloon1 = quad(dN_dz1, z_space[zi], 8, args=(GRAMS_Balloon_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_GRAMS_Satellite1 = quad(dN_dz1, z_space[zi], 8, args=(GRAMS_Satellite_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_newASTROGAM1 = quad(dN_dz1, z_space[zi], 8, args=(newASTROGAM_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_AMEGO_X1 = quad(dN_dz1, z_space[zi], 8, args=(AMEGO_X_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
                
            N_z_GRAMS_Balloon2 = quad(dN_dz2, z_space[zi], 8, args=(GRAMS_Balloon_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_GRAMS_Satellite2 = quad(dN_dz2, z_space[zi], 8, args=(GRAMS_Satellite_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_newASTROGAM2 = quad(dN_dz2, z_space[zi], 8, args=(newASTROGAM_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_AMEGO_X2 = quad(dN_dz2, z_space[zi], 8, args=(AMEGO_X_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            
            N_z_GRAMS_Balloon3 = quad(dN_dz3, z_space[zi], 8, args=(GRAMS_Balloon_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_GRAMS_Satellite3 = quad(dN_dz3, z_space[zi], 8, args=(GRAMS_Satellite_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_newASTROGAM3 = quad(dN_dz3, z_space[zi], 8, args=(newASTROGAM_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_AMEGO_X3 = quad(dN_dz3, z_space[zi], 8, args=(AMEGO_X_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            
            N_z_GRAMS_Balloon_LDDE_FSRQ1[zi][f] += N_z_GRAMS_Balloon1
            N_z_GRAMS_Satellite_LDDE_FSRQ1[zi][f] += N_z_GRAMS_Satellite1
            N_z_newASTROGAM_LDDE_FSRQ1[zi][f] += N_z_newASTROGAM1
            N_z_AMEGO_X_LDDE_FSRQ1[zi][f] += N_z_AMEGO_X1
            
            N_z_GRAMS_Balloon_LDDE_FSRQ2[zi][f] += N_z_GRAMS_Balloon2
            N_z_GRAMS_Satellite_LDDE_FSRQ2[zi][f] += N_z_GRAMS_Satellite2
            N_z_newASTROGAM_LDDE_FSRQ2[zi][f] += N_z_newASTROGAM2
            N_z_AMEGO_X_LDDE_FSRQ2[zi][f] += N_z_AMEGO_X2             
            
            N_z_GRAMS_Balloon_LDDE_FSRQ3[zi][f] += N_z_GRAMS_Balloon3
            N_z_GRAMS_Satellite_LDDE_FSRQ3[zi][f] += N_z_GRAMS_Satellite3
            N_z_newASTROGAM_LDDE_FSRQ3[zi][f] += N_z_newASTROGAM3
            N_z_AMEGO_X_LDDE_FSRQ3[zi][f] += N_z_AMEGO_X3   

BLL_cmap = plt.get_cmap('Reds')
BLL_colors = [BLL_cmap(x) for x in np.linspace(0.4, 0.9, 3)]

print("Calculating counts for SBPL type BLLs...")         

N_z_GRAMS_Balloon_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]   
N_z_GRAMS_Satellite_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_newASTROGAM_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_AMEGO_X_LDDE_BLL1 = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))] 

for i in range(len(BLL_Catalog)):
    Lum_bin = BLL_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 =  BLL_Catalog[config['columns']['L0']].tolist()[i]
    Eb =  BLL_Catalog[config['columns']['Break_Energy']].tolist()[i]
    Index1 =  BLL_Catalog[config['columns']['Index1']].tolist()[i]
    Index2 =  BLL_Catalog[config['columns']['Index2']].tolist()[i]
    Beta = BLL_Catalog[config['columns']['Curvature']].tolist()[i]
    
    nuLnu_SBPL_bin = partial(nuLnu_SBPL, L0=L0, Eb=Eb, alpha1=Index1, alpha2=Index2, s=Beta)
        
    for f in range(len(GRAMS_Flux_Bins)):
        for zi in range(len(z_space_bll)):
            log_L_max = Lum_bin[1]

            dN_dz_BLL_SBPL = partial(dN_dz, LDDE=LDDE_BLL2, log_L_max=log_L_max, i=i, Lum_bin=Lum_bin, nuLnu=nuLnu_SBPL_bin)
            
            N_z_GRAMS_Balloon = quad(dN_dz_BLL_SBPL, z_space_bll[zi], 6, args=(GRAMS_Balloon_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_GRAMS_Satellite = quad(dN_dz_BLL_SBPL, z_space_bll[zi], 6, args=(GRAMS_Satellite_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_newASTROGAM = quad(dN_dz_BLL_SBPL, z_space_bll[zi], 6, args=(newASTROGAM_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_AMEGO_X = quad(dN_dz_BLL_SBPL, z_space_bll[zi], 6, args=(AMEGO_X_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
        
            N_z_GRAMS_Balloon_LDDE_BLL1[zi][f] += N_z_GRAMS_Balloon
            N_z_GRAMS_Satellite_LDDE_BLL1[zi][f] += N_z_GRAMS_Satellite
            N_z_newASTROGAM_LDDE_BLL1[zi][f] += N_z_newASTROGAM
            N_z_AMEGO_X_LDDE_BLL1[zi][f] += N_z_AMEGO_X

print("Calculating counts for LP type BLLs...")           

N_z_GRAMS_Balloon_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]   
N_z_GRAMS_Satellite_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_newASTROGAM_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]
N_z_AMEGO_X_LDDE_BLL1_LP = [[0 for _ in range(len(GRAMS_Flux_Bins))] for _ in range(len(z_space))]     

for i in range(len(BLL_LP_Catalog)):
    Lum_bin = BLL_LP_Catalog[config['columns']['lum_bin']].tolist()[i]
    L0 = BLL_LP_Catalog[config['columns']['L0']].tolist()[i]
    z = BLL_LP_Catalog[config['columns']['Redshift']].tolist()[i]
    Alpha = BLL_LP_Catalog[config['columns']['LP_Alpha']].tolist()[i]
    Beta = BLL_LP_Catalog[config['columns']['LP_Beta']].tolist()[i]
    
    nuLnu_LP_bin = partial(nuLnu_LP, L0=L0, alpha=Alpha, beta=Beta, E0=0.017)
    
    for f in range(len(GRAMS_Flux_Bins)):
        for zi in range(len(z_space_bll)):
            if i != len(BLL_LP_Catalog) - 1:
                log_L_max = Lum_bin[1]
            elif i == len(BLL_LP_Catalog) - 1:
                log_L_max = 52
            
            dN_dz_BLL_LP = partial(dN_dz, LDDE=LDDE_BLL2, log_L_max=log_L_max, i=i, Lum_bin=Lum_bin, nuLnu=nuLnu_LP_bin)
                
            N_z_GRAMS_Balloon = quad(dN_dz_BLL_LP, z_space_bll[zi], 6, args=(GRAMS_Balloon_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_GRAMS_Satellite = quad(dN_dz_BLL_LP, z_space_bll[zi], 6, args=(GRAMS_Satellite_Sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_newASTROGAM = quad(dN_dz_BLL_LP, z_space_bll[zi], 6, args=(newASTROGAM_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi
            N_z_AMEGO_X = quad(dN_dz_BLL_LP, z_space_bll[zi], 6, args=(AMEGO_X_sensitivity[f], GRAMS_Flux_Bins[f]))[0] * 4 * np.pi

            N_z_GRAMS_Balloon_LDDE_BLL1_LP[zi][f] += N_z_GRAMS_Balloon
            N_z_GRAMS_Satellite_LDDE_BLL1_LP[zi][f] += N_z_GRAMS_Satellite
            N_z_newASTROGAM_LDDE_BLL1_LP[zi][f] += N_z_newASTROGAM
            N_z_AMEGO_X_LDDE_BLL1_LP[zi][f] += N_z_AMEGO_X
        
N_z_GRAMS_Balloon_BLL_LDDE_SBPL = list(zip(*N_z_GRAMS_Balloon_LDDE_BLL1))
N_z_GRAMS_Satellite_BLL_LDDE_SBPL = list(zip(*N_z_GRAMS_Satellite_LDDE_BLL1))
N_z_newASTROGAM_BLL_LDDE_SBPL = list(zip(*N_z_newASTROGAM_LDDE_BLL1))
N_z_AMEGO_X_BLL_LDDE_SBPL = list(zip(*N_z_AMEGO_X_LDDE_BLL1))

N_z_GRAMS_Balloon_BLL_LDDE_LP = list(zip(*N_z_GRAMS_Balloon_LDDE_BLL1_LP))
N_z_GRAMS_Satellite_BLL_LDDE_LP = list(zip(*N_z_GRAMS_Satellite_LDDE_BLL1_LP))
N_z_newASTROGAM_BLL_LDDE_LP = list(zip(*N_z_newASTROGAM_LDDE_BLL1_LP))
N_z_AMEGO_X_BLL_LDDE_LP = list(zip(*N_z_AMEGO_X_LDDE_BLL1_LP)) 

N_z_GRAMS_Balloon_FSRQ_LDDE1 = list(zip(*N_z_GRAMS_Balloon_LDDE_FSRQ1))
N_z_GRAMS_Satellite_FSRQ_LDDE1 = list(zip(*N_z_GRAMS_Satellite_LDDE_FSRQ1))
N_z_newASTROGAM_FSRQ_LDDE1 = list(zip(*N_z_newASTROGAM_LDDE_FSRQ1))
N_z_AMEGO_X_FSRQ_LDDE1 = list(zip(*N_z_AMEGO_X_LDDE_FSRQ1))

N_z_GRAMS_Balloon_FSRQ_LDDE2 = list(zip(*N_z_GRAMS_Balloon_LDDE_FSRQ2))
N_z_GRAMS_Satellite_FSRQ_LDDE2 = list(zip(*N_z_GRAMS_Satellite_LDDE_FSRQ2))
N_z_newASTROGAM_FSRQ_LDDE2 = list(zip(*N_z_newASTROGAM_LDDE_FSRQ2))
N_z_AMEGO_X_FSRQ_LDDE2 = list(zip(*N_z_AMEGO_X_LDDE_FSRQ2))

N_z_GRAMS_Balloon_FSRQ_LDDE3 = list(zip(*N_z_GRAMS_Balloon_LDDE_FSRQ3))
N_z_GRAMS_Satellite_FSRQ_LDDE3 = list(zip(*N_z_GRAMS_Satellite_LDDE_FSRQ3))
N_z_newASTROGAM_FSRQ_LDDE3 = list(zip(*N_z_newASTROGAM_LDDE_FSRQ3))
N_z_AMEGO_X_FSRQ_LDDE3 = list(zip(*N_z_AMEGO_X_LDDE_FSRQ3))

N_z_GRAMS_Balloon_BLL_LDDE_all = []
N_z_GRAMS_Satellite_BLL_LDDE_all = []
N_z_newASTROGAM_BLL_LDDE_all = []
N_z_AMEGO_X_BLL_LDDE_all = []

for s in range(len(N_z_GRAMS_Balloon_BLL_LDDE_SBPL)):
    N_z_GRAMS_Balloon_BLL_LDDE_all.append([N_z_GRAMS_Balloon_BLL_LDDE_SBPL[s][f] + N_z_GRAMS_Balloon_BLL_LDDE_LP[s][f] for f in range(len(N_z_GRAMS_Balloon_BLL_LDDE_SBPL[s]))])
    N_z_GRAMS_Satellite_BLL_LDDE_all.append([N_z_GRAMS_Satellite_BLL_LDDE_SBPL[s][f] + N_z_GRAMS_Satellite_BLL_LDDE_LP[s][f] for f in range(len(N_z_GRAMS_Satellite_BLL_LDDE_SBPL[s]))])
    N_z_newASTROGAM_BLL_LDDE_all.append([N_z_newASTROGAM_BLL_LDDE_SBPL[s][f] + N_z_newASTROGAM_BLL_LDDE_LP[s][f] for f in range(len(N_z_newASTROGAM_BLL_LDDE_SBPL[s]))])
    N_z_AMEGO_X_BLL_LDDE_all.append([N_z_AMEGO_X_BLL_LDDE_SBPL[s][f] + N_z_AMEGO_X_BLL_LDDE_LP[s][f] for f in   range(len(N_z_AMEGO_X_BLL_LDDE_SBPL[s]))])

GRAMS_Balloon_Names_LDDE_BLL1 = [f'N_z_GRAMS_Balloon_{GRAMS_Flux_Bins[s]}_LDDE_BLL_MeV' for s in range(len(GRAMS_Flux_Bins))]
GRAMS_Satellite_Names_LDDE_BLL1 = [f'N_z_GRAMS_Satellite_{GRAMS_Flux_Bins[s]}_LDDE_BLL_MeV' for s in range(len(GRAMS_Flux_Bins))]
newASTROGAM_Names_LDDE_BLL1 = [f'N_z_newASTROGAM_{GRAMS_Flux_Bins[s]}_LDDE_BLL_MeV' for s in range(len(GRAMS_Flux_Bins))]
AMEGO_X_Names_LDDE_BLL1 = [f'N_z_AMEGO_X_{GRAMS_Flux_Bins[s]}_LDDE_BLL_MeV' for s in range(len(GRAMS_Flux_Bins))]

GRAMS_Balloon_Names_LDDE_FSRQ1 = [f'N_z_GRAMS_Balloon_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ_MeV' for s in range(len(GRAMS_Flux_Bins))]
GRAMS_Satellite_Names_LDDE_FSRQ1 = [f'N_z_GRAMS_Satellite_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ_MeV' for s in range(len(GRAMS_Flux_Bins))]
newASTROGAM_Names_LDDE_FSRQ1 = [f'N_z_newASTROGAM_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ_MeV' for s in range(len(GRAMS_Flux_Bins))]
AMEGO_X_Names_LDDE_FSRQ1 = [f'N_z_AMEGO_X_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ_MeV' for s in range(len(GRAMS_Flux_Bins))]

GRAMS_Balloon_Names_LDDE_FSRQ2 = [f'N_z_GRAMS_Balloon_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ2_MeV' for s in range(len(GRAMS_Flux_Bins))]
GRAMS_Satellite_Names_LDDE_FSRQ2 = [f'N_z_GRAMS_Satellite_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ2_MeV' for s in range(len(GRAMS_Flux_Bins))]
newASTROGAM_Names_LDDE_FSRQ2 = [f'N_z_newASTROGAM_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ2_MeV' for s in range(len(GRAMS_Flux_Bins))]
AMEGO_X_Names_LDDE_FSRQ2 = [f'N_z_AMEGO_X_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ2_MeV' for s in range(len(GRAMS_Flux_Bins))]

GRAMS_Balloon_Names_LDDE_FSRQ3 = [f'N_z_GRAMS_Balloon_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ3_MeV' for s in range(len(GRAMS_Flux_Bins))]
GRAMS_Satellite_Names_LDDE_FSRQ3 = [f'N_z_GRAMS_Satellite_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ3_MeV' for s in range(len(GRAMS_Flux_Bins))]
newASTROGAM_Names_LDDE_FSRQ3 = [f'N_z_newASTROGAM_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ3_MeV' for s in range(len(GRAMS_Flux_Bins))]
AMEGO_X_Names_LDDE_FSRQ3 = [f'N_z_AMEGO_X_{GRAMS_Flux_Bins[s]}_LDDE_FSRQ3_MeV' for s in range(len(GRAMS_Flux_Bins))]

All_Names = GRAMS_Balloon_Names_LDDE_BLL1 + GRAMS_Satellite_Names_LDDE_BLL1 + newASTROGAM_Names_LDDE_BLL1 + AMEGO_X_Names_LDDE_BLL1 + GRAMS_Balloon_Names_LDDE_FSRQ1 + GRAMS_Satellite_Names_LDDE_FSRQ1 + newASTROGAM_Names_LDDE_FSRQ1 + AMEGO_X_Names_LDDE_FSRQ1 + GRAMS_Balloon_Names_LDDE_FSRQ2 + GRAMS_Satellite_Names_LDDE_FSRQ2 + newASTROGAM_Names_LDDE_FSRQ2 + AMEGO_X_Names_LDDE_FSRQ2

GRAMS_Balloon_N_z_Data_Output_Table_BLL = Table(N_z_GRAMS_Balloon_BLL_LDDE_all, names=GRAMS_Balloon_Names_LDDE_BLL1)
GRAMS_Satellite_N_z_Data_Output_Table_BLL = Table(N_z_GRAMS_Satellite_BLL_LDDE_all, names=GRAMS_Satellite_Names_LDDE_BLL1)
newASTROGAM_N_z_Data_Output_Table_BLL = Table(N_z_newASTROGAM_BLL_LDDE_all, names=newASTROGAM_Names_LDDE_BLL1)
AMEGO_X_N_z_Data_Output_Table_BLL = Table(N_z_AMEGO_X_BLL_LDDE_all, names=AMEGO_X_Names_LDDE_BLL1)

GRAMS_Balloon_N_z_Data_Output_Table_FSRQ1 = Table(N_z_GRAMS_Balloon_FSRQ_LDDE1, names=GRAMS_Balloon_Names_LDDE_FSRQ1)
GRAMS_Satellite_N_z_Data_Output_Table_FSRQ1 = Table(N_z_GRAMS_Satellite_FSRQ_LDDE1, names=GRAMS_Satellite_Names_LDDE_FSRQ1)
newASTROGAM_N_z_Data_Output_Table_FSRQ1 = Table(N_z_newASTROGAM_FSRQ_LDDE1, names=newASTROGAM_Names_LDDE_FSRQ1)
AMEGO_X_N_z_Data_Output_Table_FSRQ1 = Table(N_z_AMEGO_X_FSRQ_LDDE1, names=AMEGO_X_Names_LDDE_FSRQ1)

GRAMS_Balloon_N_z_Data_Output_Table_FSRQ2 = Table(N_z_GRAMS_Balloon_FSRQ_LDDE2, names=GRAMS_Balloon_Names_LDDE_FSRQ2)
GRAMS_Satellite_N_z_Data_Output_Table_FSRQ2 = Table(N_z_GRAMS_Satellite_FSRQ_LDDE2, names=GRAMS_Satellite_Names_LDDE_FSRQ2)
newASTROGAM_N_z_Data_Output_Table_FSRQ2 = Table(N_z_newASTROGAM_FSRQ_LDDE2, names=newASTROGAM_Names_LDDE_FSRQ2)
AMEGO_X_N_z_Data_Output_Table_FSRQ2 = Table(N_z_AMEGO_X_FSRQ_LDDE2, names=AMEGO_X_Names_LDDE_FSRQ2)

GRAMS_Balloon_N_z_Data_Output_Table_FSRQ3 = Table(N_z_GRAMS_Balloon_FSRQ_LDDE3, names=GRAMS_Balloon_Names_LDDE_FSRQ3)
GRAMS_Satellite_N_z_Data_Output_Table_FSRQ3 = Table(N_z_GRAMS_Satellite_FSRQ_LDDE3, names=GRAMS_Satellite_Names_LDDE_FSRQ3)
newASTROGAM_N_z_Data_Output_Table_FSRQ3 = Table(N_z_newASTROGAM_FSRQ_LDDE3, names=newASTROGAM_Names_LDDE_FSRQ3)
AMEGO_X_N_z_Data_Output_Table_FSRQ3 = Table(N_z_AMEGO_X_FSRQ_LDDE3, names=AMEGO_X_Names_LDDE_FSRQ3)

Redshift_Table = Table([z_space, z_space_bll], names=['Redshift', 'BLL_Redshift'])

N_z_Data_Output_Table = hstack([Redshift_Table, GRAMS_Balloon_N_z_Data_Output_Table_BLL, GRAMS_Satellite_N_z_Data_Output_Table_BLL, newASTROGAM_N_z_Data_Output_Table_BLL, AMEGO_X_N_z_Data_Output_Table_BLL, GRAMS_Balloon_N_z_Data_Output_Table_FSRQ1, GRAMS_Satellite_N_z_Data_Output_Table_FSRQ1, newASTROGAM_N_z_Data_Output_Table_FSRQ1, AMEGO_X_N_z_Data_Output_Table_FSRQ1, GRAMS_Balloon_N_z_Data_Output_Table_FSRQ2, GRAMS_Satellite_N_z_Data_Output_Table_FSRQ2, newASTROGAM_N_z_Data_Output_Table_FSRQ2, AMEGO_X_N_z_Data_Output_Table_FSRQ2, GRAMS_Balloon_N_z_Data_Output_Table_FSRQ3, GRAMS_Satellite_N_z_Data_Output_Table_FSRQ3, newASTROGAM_N_z_Data_Output_Table_FSRQ3, AMEGO_X_N_z_Data_Output_Table_FSRQ3])
N_z_Data_Output_Table.write(config['Catalog_dir'] + 'N_z_LF_output.fits', overwrite=True)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")