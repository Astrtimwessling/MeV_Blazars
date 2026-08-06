from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/"
outdir = '/home/alab_student/Tim/Projects/MeV_Blazars/Outputs'

# Import N(>z) calculations from luminosity function models: 

N_z_LF_data_results = Table.read(catalog_dir + 'N_z_LF_output.fits')
N_z_LF_redshifts = N_z_LF_data_results['Redshift'].tolist()

# Import Sensitivity Information for GRAMS Balloon, GRAMS Satellite, AMEGO-X, newASTROGAM, and COSI.

# GRAMS Balloon and Satellite: https://arxiv.org/pdf/1901.03430 (Figure 7.)
# AMEGO-X: https://arxiv.org/pdf/2208.04990 (Figure 3.)
# newASTROGAM: https://arxiv.org/pdf/2507.08133 (Figure 1.)
# COSI: https://arxiv.org/pdf/2308.12362 (Right Panel, Figure 2.)

Sensitivity_Data = Table.read(catalog_dir + 'Sensitivity_Data_Table.fits')

Select_Energy_Values = Sensitivity_Data['Energy'].tolist()
GRAMS_Balloon_Sensitivity = Sensitivity_Data['GRAMS_Balloon_Sensitivity'].tolist()
GRAMS_Satellite_Sensitivity = Sensitivity_Data['GRAMS_Satellite_Sensitivity'].tolist()
AMEGO_X_Sensitivity = Sensitivity_Data['AMEGO_X_Sensitivity'].tolist()
newASTROGAM_Sensitivity = Sensitivity_Data['newASTROGAM_Sensitivity'].tolist()

Sensitivity_Curves = [GRAMS_Balloon_Sensitivity, GRAMS_Satellite_Sensitivity, AMEGO_X_Sensitivity, newASTROGAM_Sensitivity]
Instrument_Names = ['GRAMS Balloon', 'GRAMS Satellite', 'AMEGO-X', 'newASTROGAM']

# Extrapolate SED modeled fluxes of cross-matched sources with known redshifts

MeV_Blazar_Catalog_SED_Fits = Table.read(catalog_dir + "MeV_Blazar_Catalog_v2_SED_Fits.fits")
Mask = (MeV_Blazar_Catalog_SED_Fits['MeV_Model'] != 'None')
MeV_Blazar_Catalog_SED_Fits = MeV_Blazar_Catalog_SED_Fits[Mask]

Swift_Redshift = MeV_Blazar_Catalog_SED_Fits['Swift_Redshift'].tolist()
Fermi_Redshift = MeV_Blazar_Catalog_SED_Fits['Fermi_Redshift'].tolist()

Redshift = [Swift_Redshift[x] if Swift_Redshift[x] != 'None' else Fermi_Redshift[x] for x in range(len(Swift_Redshift))]

MeV_Estimated_Fluxes = [list(x) for x in MeV_Blazar_Catalog_SED_Fits['MeV_Flux_Estimates'].tolist()]
Blazar_Classes = MeV_Blazar_Catalog_SED_Fits['Fermi_Type'].tolist()

Fluxes_Redshifts_Classes = list(zip(Redshift, MeV_Estimated_Fluxes, Blazar_Classes))

Fluxes_Known_all = list(zip(*[x[1] for x in Fluxes_Redshifts_Classes if x[0] != 'None']))
Redshifts_Known_all = [float(x[0]) for x in Fluxes_Redshifts_Classes if x[0] != 'None']

Fluxes_Known_FSRQ = list(zip(*[x[1] for x in Fluxes_Redshifts_Classes if x[0] != 'None' and x[2].upper() == 'FSRQ']))
Redshifts_Known_FSRQ = [float(x[0]) for x in Fluxes_Redshifts_Classes if x[0] != 'None' and x[2].upper() == 'FSRQ']

Fluxes_Known_BLL = list(zip(*[x[1] for x in Fluxes_Redshifts_Classes if x[0] != 'None' and x[2].upper() == 'BLL']))
Redshifts_Known_BLL = [float(x[0]) for x in Fluxes_Redshifts_Classes if x[0] != 'None' and x[2].upper() == 'BLL']

# Plot joint cumulative redshift counts N(>z) from data and model.

redshift_space = np.linspace(0.0001, 3.5, 30)
sens = np.array(redshift_space)
edges = np.zeros(len(sens) + 1)
edges[1:-1] = np.sqrt(sens[:-1] * sens[1:])
edges[0] = sens[0]**2 / edges[1]
edges[-1] = sens[-1]**2 / edges[-2]

for i in range(len(Select_Energy_Values)):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    N_z_LF_data_all = [[N_z_LF_data_results.columns[n].tolist() for n in [m+i for m in [1, 10, 19, 28]]], [N_z_LF_data_results.columns[n].tolist() for n in [m+i+36 for m in [37, 46, 55, 64]]], [N_z_LF_data_results.columns[n].tolist() for n in [m+i for m in [109, 118, 127, 136]]]]

    for s in range(len(Sensitivity_Curves)):
        fluxes_redshifts_all_detectable = [Redshifts_Known_all[x] for x in range(len(Redshifts_Known_all)) if Fluxes_Known_all[i][x] >= Sensitivity_Curves[s][i]]
        fluxes_redshifts_fsrq_detectable = [Redshifts_Known_FSRQ[x] for x in range(len(Redshifts_Known_FSRQ)) if Fluxes_Known_FSRQ[i][x] >= Sensitivity_Curves[s][i]]
        fluxes_redshifts_bll_detectable = [Redshifts_Known_BLL[x] for x in range(len(Redshifts_Known_BLL)) if Fluxes_Known_BLL[i][x] >= Sensitivity_Curves[s][i]]
        
        N_z_all = [sum([x >= redshift_space[z] for x in fluxes_redshifts_all_detectable]) for z in range(len(redshift_space))]
        N_z_fsrq = [sum([x >= redshift_space[z] for x in fluxes_redshifts_fsrq_detectable]) for z in range(len(redshift_space))]
        N_z_bll = [sum([x >= redshift_space[z] for x in fluxes_redshifts_bll_detectable]) for z in range(len(redshift_space))]
        
        ax = axes.flat[s]
        ax.stairs(N_z_fsrq, edges, color='blue', label='FSRQ', baseline=None)
        ax.stairs(N_z_bll, edges, color='red', label='BLL', baseline=None)
        ax.axhline(y=1, color='black', linestyle='--', label='1 Source Detection')
        
        lit_labels = ['Ajello, M. (2013) LAT BLL LDDE', 'Rajguru et al. (2025) LAT FSRQ LDDE', 'Marcotulli et al. (2022) BAT FSRQ LDDE']
        colors = ['red', 'blue', 'blue']
        linestyles = ['--', '--', '-.']
                
        
        for l in range(len(lit_labels)):
            y_data = np.asarray(N_z_LF_data_all[l][s]).flatten()
            ax.plot(N_z_LF_redshifts, y_data, label=lit_labels[l], color=colors[l], linestyle=linestyles[l])

        ax.set_title(f'{Instrument_Names[s]}')
        ax.set_yscale('log')
        ax.set_xlim(0, None)

    for ax in axes.flat:
        ax.label_outer()

    fig.supxlabel(r'redshift $z$', y=0.03, fontsize=17)
    fig.supylabel(r'N(>$z$)' + f'[{Select_Energy_Values[i]} MeV]', fontsize=15)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=13, frameon=False, columnspacing=1.2, handletextpad=0.5)

    plt.ylim(0.5, 1000)
    plt.xscale('log')
    plt.xlim(1e-1, 0.7e1)
    
    plt.tight_layout()
    plt.savefig(outdir + f'/N_z_Plots/pdfs/Cumulative_N_z_{Select_Energy_Values[i]}.png')
    plt.savefig(outdir + f'/N_z_Plots/pngs/Cumulative_N_z_{Select_Energy_Values[i]}.pdf')
    plt.show()
    plt.close(fig)