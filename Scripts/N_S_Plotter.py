from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Output/"
outdir = '/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/N_S_Plots'

# Extract N(>S) results for each luminosity function model.

Luminosity_Function_N_S_Results = Table.read(catalog_dir + 'Monochromatic_Luminosity_Function_Results.fits')

N_S_Rajguru = [Luminosity_Function_N_S_Results.columns[i].tolist() for i in [10,11,12,13,14,15,16,17,18]]
N_S_Marcotulli = [Luminosity_Function_N_S_Results.columns[i].tolist() for i in [19,20,21,22,23,24,25,26,27]]
N_S_Ajello = [Luminosity_Function_N_S_Results.columns[i].tolist() for i in [28,29,30,31,32,33,34,35,36]]

Sensitivity_List = Luminosity_Function_N_S_Results['Sensitivity'].tolist()

# Extract N(>S) results from SED modeling of cross-matched sources.

MeV_Blazar_Catalog_SED_Fits = Table.read(catalog_dir + "MeV_Blazar_Catalog_v2_SED_Fits.fits")
Mask = (MeV_Blazar_Catalog_SED_Fits['MeV_Model'] != 'None')
MeV_Blazar_Catalog_SED_Fits = MeV_Blazar_Catalog_SED_Fits[Mask]

MeV_Estimated_Fluxes = [list(x) for x in MeV_Blazar_Catalog_SED_Fits['MeV_Flux_Estimates'].tolist()]
Blazar_Classes = MeV_Blazar_Catalog_SED_Fits['Fermi_Type'].tolist()
MeV_Flux_Estimates_By_Band = list(zip(*MeV_Estimated_Fluxes))

FSRQ_Flux_Estimates = list(zip(*[MeV_Estimated_Fluxes[i] for i in range(len(MeV_Estimated_Fluxes)) if Blazar_Classes[i].upper() == 'FSRQ']))
BLL_Flux_Estimates = list(zip(*[MeV_Estimated_Fluxes[i] for i in range(len(MeV_Estimated_Fluxes)) if Blazar_Classes[i].upper() == 'BLL']))

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
COSI_Sensitivity = Sensitivity_Data['COSI_Sensitivity'].tolist()

# Plot N(>S) for selected energies.

selected_indices = [1, 2, 4, 5, 7, 8]

sens = np.array(Sensitivity_List)
edges = np.zeros(len(sens) + 1)
edges[1:-1] = np.sqrt(sens[:-1] * sens[1:])
edges[0] = sens[0]**2 / edges[1]
edges[-1] = sens[-1]**2 / edges[-2]

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 15))
axes = axes.flatten()

for ax, i in zip(axes, selected_indices):
    N_S_Counts_All = [len([x for x in MeV_Flux_Estimates_By_Band[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_FSRQ = [len([x for x in FSRQ_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_BLL = [len([x for x in BLL_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    
    ax.stairs(N_S_Counts_All, edges, color='black', label='All', baseline=None)
    ax.stairs(N_S_Counts_FSRQ, edges, color='blue', label='FSRQ', baseline=None)
    ax.stairs(N_S_Counts_BLL, edges, color='red', label='BLL', baseline=None)
    
    ax.plot(Sensitivity_List, N_S_Marcotulli[i], label='Marcotulli al. (2022) BAT FSRQ LDDE', linestyle='--', color='blue')
    ax.plot(Sensitivity_List, N_S_Rajguru[i], label='Rajguru et al. (2025) LAT FSRQ LDDE', linestyle='-.', color='blue')
    ax.plot(Sensitivity_List, N_S_Ajello[i], label='Ajello et al. (2013) LAT BLL LDDE', linestyle='--', color='red')
    
    ax.axvline(GRAMS_Balloon_Sensitivity[i], linestyle='--', color='grey')
    ax.axvline(GRAMS_Satellite_Sensitivity[i], linestyle='-.', color='grey')
    ax.axvline(newASTROGAM_Sensitivity[i], linestyle='--', color='magenta')
    ax.axvline(AMEGO_X_Sensitivity[i], linestyle='--', color='orange')
        
    if i <= 4:
        ax.axvline(COSI_Sensitivity[i], linestyle='--', color='Green')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.axhline(y=1, color='black', linestyle='--', label='1 Source Detection')
    ax.tick_params(labelsize=14)
    ax.set_title(f"{Select_Energy_Values[i]} MeV", fontsize=17)
    
for ax in axes:
    ax.label_outer()

fig.supxlabel(r'$S$ [erg/cm$^{2}$/s]', y=0.01, fontsize=17)
fig.supylabel(r'N(>S) [4$\pi$ str]', fontsize=15)

handles, labels = axes[0].get_legend_handles_labels()

extra_lines = [
    plt.Line2D([0], [0], color='grey', linestyle='--', label='GRAMS Balloon'),
    plt.Line2D([0], [0], color='grey', linestyle='-.', label='GRAMS Satellite'),
    plt.Line2D([0], [0], color='orange', linestyle='--', label='AMEGO-X'),
    plt.Line2D([0], [0], color='magenta', linestyle='--', label='newASTROGAM'),
    plt.Line2D([0], [0], color='green', linestyle='--', label='COSI'),
]

handles += extra_lines
labels += [l.get_label() for l in extra_lines]

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.055), ncol=4, fontsize=13, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)

plt.ylim(0.5,10000)
plt.tight_layout()

plt.savefig(outdir + '/N_S_MeV_Blazars_Forecast.png', bbox_inches='tight', dpi=400)
plt.savefig(outdir + '/N_S_MeV_Blazars_Forecast.pdf', bbox_inches='tight', dpi=400)

plt.show()
plt.close()

selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

sens = np.array(Sensitivity_List)
edges = np.zeros(len(sens) + 1)
edges[1:-1] = np.sqrt(sens[:-1] * sens[1:])
edges[0] = sens[0]**2 / edges[1]
edges[-1] = sens[-1]**2 / edges[-2]

for i in selected_indices:
    fig, ax = plt.subplots(figsize=(11, 6))

    N_S_Counts_All = [len([x for x in MeV_Flux_Estimates_By_Band[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_FSRQ = [len([x for x in FSRQ_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_BLL = [len([x for x in BLL_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]

    ax.stairs(N_S_Counts_All, edges, color='black', label='All', baseline=None)
    ax.stairs(N_S_Counts_FSRQ, edges, color='blue', label='FSRQ', baseline=None)
    ax.stairs(N_S_Counts_BLL, edges, color='red', label='BLL', baseline=None)

    ax.plot(Sensitivity_List, N_S_Marcotulli[i], label='Marcotulli al. (2022) BAT FSRQ LDDE', linestyle='--', color='blue')
    ax.plot(Sensitivity_List, N_S_Rajguru[i], label='Rajguru et al. (2025) LAT FSRQ LDDE', linestyle='-.', color='blue')
    ax.plot(Sensitivity_List, N_S_Ajello[i], label='Ajello et al. (2013) LAT BLL LDDE', linestyle='--', color='red')

    ax.axvline(GRAMS_Balloon_Sensitivity[i], linestyle='--', color='grey')
    ax.axvline(GRAMS_Satellite_Sensitivity[i], linestyle='-.', color='grey')
    ax.axvline(newASTROGAM_Sensitivity[i], linestyle='--', color='magenta')
    ax.axvline(AMEGO_X_Sensitivity[i], linestyle='--', color='orange')

    if i <= 4:
        ax.axvline(COSI_Sensitivity[i], linestyle='--', color='Green')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.axhline(y=1, color='black', linestyle='--', label='1 Source Detection')
    ax.tick_params(labelsize=14)
    ax.set_title(f"{Select_Energy_Values[i]} MeV", fontsize=17)

    ax.set_xlabel(r'$S$ [erg/cm$^{2}$/s]', fontsize=17)
    ax.set_ylabel(r'N(>S) [4$\pi$ str]', fontsize=15)
    ax.set_ylim(0.5, 10000)

    handles, labels = ax.get_legend_handles_labels()

    extra_lines = [
        plt.Line2D([0], [0], color='grey', linestyle='--', label='GRAMS Balloon'),
        plt.Line2D([0], [0], color='grey', linestyle='-.', label='GRAMS Satellite'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='AMEGO-X'),
        plt.Line2D([0], [0], color='magenta', linestyle='--', label='newASTROGAM'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='COSI'),
    ]

    handles += extra_lines
    labels += [l.get_label() for l in extra_lines]

    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=13, frameon=False, handletextpad=0.5)

    plt.tight_layout()

    energy_label = str(Select_Energy_Values[i]).replace('.', '_')  # safe for filenames
    plt.savefig(f'{outdir}/N_S_MeV_Blazars_Forecast_{energy_label}_MeV.png', bbox_inches='tight', dpi=400)
    #plt.savefig(f'{outdir}/N_S_MeV_Blazars_Forecast_{energy_label}MeV.pdf', bbox_inches='tight', dpi=400)

    plt.show()
    plt.close(fig)
    
selected_indices = [2]

sens = np.array(Sensitivity_List)
edges = np.zeros(len(sens) + 1)
edges[1:-1] = np.sqrt(sens[:-1] * sens[1:])
edges[0] = sens[0]**2 / edges[1]
edges[-1] = sens[-1]**2 / edges[-2]

for i in selected_indices:
    fig, ax = plt.subplots(figsize=(11, 6))

    N_S_Counts_All = [len([x for x in MeV_Flux_Estimates_By_Band[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_FSRQ = [len([x for x in FSRQ_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]
    N_S_Counts_BLL = [len([x for x in BLL_Flux_Estimates[i] if x >= Sensitivity_List[s]]) for s in range(len(Sensitivity_List))]

    ax.stairs(N_S_Counts_All, edges, color='black', label='All', baseline=None)
    ax.stairs(N_S_Counts_FSRQ, edges, color='blue', label='FSRQ', baseline=None)
    ax.stairs(N_S_Counts_BLL, edges, color='red', label='BLL', baseline=None)

    ax.plot(Sensitivity_List, N_S_Marcotulli[i], label='Marcotulli al. (2022) BAT FSRQ LDDE', linestyle='--', color='blue')
    ax.plot(Sensitivity_List, N_S_Rajguru[i], label='Rajguru et al. (2025) LAT FSRQ LDDE', linestyle='-.', color='blue')
    ax.plot(Sensitivity_List, N_S_Ajello[i], label='Ajello et al. (2013) LAT BLL LDDE', linestyle='--', color='red')

    ax.axvline(4.4e-11, linestyle='--', color='grey')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.axhline(y=1, color='black', linestyle='--', label='1 Source Detection')
    ax.tick_params(labelsize=14)
    ax.set_title(f"{Select_Energy_Values[i]} MeV", fontsize=17)

    ax.set_xlabel(r'$S$ [erg/cm$^{2}$/s]', fontsize=17)
    ax.set_ylabel(r'N(>S) [4$\pi$ str]', fontsize=15)
    ax.set_ylim(0.5, 10000)

    handles, labels = ax.get_legend_handles_labels()

    extra_lines = [
        plt.Line2D([0], [0], color='grey', linestyle='--', label='GRAMS ULDB' + r'(S(1 MeV) = 4.4e-11 $\mathrm{erg/cm^2/s}$)'),
    ]

    handles += extra_lines
    labels += [l.get_label() for l in extra_lines]

    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=13, frameon=False, handletextpad=0.5)

    plt.tight_layout()

    energy_label = str(Select_Energy_Values[i]).replace('.', '_')
    plt.savefig(f'{outdir}/GRAMS_APRA_N_S_{energy_label}_MeV.pdf', bbox_inches='tight', dpi=400)

    plt.show()
    plt.close(fig)