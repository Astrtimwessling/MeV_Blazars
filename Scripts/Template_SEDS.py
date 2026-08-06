import matplotlib.pyplot as plt
import numpy as np 

from astropy.table import Table, vstack
from functools import partial

from src.Luminosity_Formulas import K_Corrected_Band_luminosity, d_L
from src.SED_Models import nuFnu_LP, nuFnu_SBPL, nuLnu_LP, Fit_nuLnu_LP, Fit_nuLnu_SBPL

def equal_count_bins(data, n_bins):
    data = np.array(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(data, quantiles)
    return bins

def dLdE(dnde, z, log_E1, log_E2):
    
    E_rest_grid = np.logspace(log_E1, log_E2, 100)
    E_obs = E_rest_grid / (1 + z)

    dnde_obs_vals = np.array([dnde(x) for x in E_obs])
    dnde_rest = (1 + z) * dnde_obs_vals

    E2_dnde_rest = E_rest_grid**2 * dnde_rest * 1.602176634e-6
    all_E2_dnde_space.append(E2_dnde_rest)
    
    Luminosity_Distance = d_L(z) * 3.085677581e24
    L_SED = 4 * np.pi * Luminosity_Distance**2 * E2_dnde_rest
    
    return L_SED

def Jackknife_technique(SEDs):
    N = SEDs.shape[0]
        
    jackknife_means = []

    for x in range(N):
    
        jk_logs = np.delete(SEDs, x, axis=0)
        jk_mean = np.mean(jk_logs, axis=0)
        jackknife_means.append(jk_mean)
    
    jackknife_means = np.array(jackknife_means)
    jk_bar = np.mean(jackknife_means, axis=0)
    jk_var = (N-1)/N * np.sum((jackknife_means - jk_bar)**2, axis=0)
    jk_err = np.sqrt(jk_var)
    
    return jk_err

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Data/"
out_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/Template_SEDS/"

params_catalog = Table.read(catalog_dir + 'SED_Selected_Model_Params.fits')

fig, axes = plt.subplots(1, 3, figsize=(30, 9), sharex=True)

ax_fsrq = axes[0]
ax_bll_lp = axes[1]
ax_bll_hsp = axes[2]

FSRQ_LP_Mask = (params_catalog['Model'] == 'LP') & ((params_catalog['Class'] == 'fsrq') | (params_catalog['Class'] == 'FSRQ'))
FSRQ_LP_Table = params_catalog[FSRQ_LP_Mask]

BLL_LP_Mask = (params_catalog['Model'] == 'LP') & ((params_catalog['Class'] == 'bll') | (params_catalog['Class'] == 'BLL'))
BLL_LP_Table = params_catalog[BLL_LP_Mask]

BLL_SBPL_Mask = (params_catalog['Model'] == 'SBPL') & ((params_catalog['Class'] == 'bll') | (params_catalog['Class'] == 'BLL'))
BLL_SBPL_Table = params_catalog[BLL_SBPL_Mask]

print(f"Making FSRQ SED Templates for {len(FSRQ_LP_Table)} sources...")

k_corr_lum_LP_FSRQ = []

for i in range(len(FSRQ_LP_Table)):
    params = [float(x) for x in [FSRQ_LP_Table['E0'][i], FSRQ_LP_Table['N0'][i], FSRQ_LP_Table['LP_Alpha'][i], FSRQ_LP_Table['LP_Beta'][i], FSRQ_LP_Table['Redshift'][i]]]
    nuFnu_LP_source = partial(nuFnu_LP, E0=params[0], N0=params[1], alpha=params[2], beta=params[3])
    k_corr_lum = K_Corrected_Band_luminosity(nuFnu = nuFnu_LP_source, z=params[4], E1=params[0], E2=1e6)
    k_corr_lum_LP_FSRQ.append(k_corr_lum)
    
FSRQ_LP_Lum_bins = equal_count_bins(k_corr_lum_LP_FSRQ,5)
FSRQ_LP_Table['k_corr_lum'] = k_corr_lum_LP_FSRQ

E_rest_grid = np.logspace(-2, 6, 100)

avg_SED_all_FSRQ = []

FSRQ_cmap = plt.get_cmap('Blues')
FSRQ_colors = [FSRQ_cmap(x) for x in np.linspace(0.4, 0.9, 5)]

Avg_FSRQ_SED_L_bin = []
Avg_FSRQ_SED_L0 = []
Avg_FSRQ_SED_alpha = []
Avg_FSRQ_SED_beta = []
Avg_FSRQ_SED_class = []
Avg_FSRQ_SED_SED_Class = []
Avg_FSRQ_SED_median_redshift = []

for i in range(len(FSRQ_LP_Lum_bins)-1):
    if i == len(FSRQ_LP_Lum_bins) - 2:
        Lum_mask = (FSRQ_LP_Table['k_corr_lum'] >= FSRQ_LP_Lum_bins[i]) & ((FSRQ_LP_Table['k_corr_lum'] <= FSRQ_LP_Lum_bins[i+1]))
    else:
        Lum_mask = (FSRQ_LP_Table['k_corr_lum'] >= FSRQ_LP_Lum_bins[i]) & ((FSRQ_LP_Table['k_corr_lum'] < FSRQ_LP_Lum_bins[i+1]))
        
    Avg_FSRQ_SED_L_bin.append([round(np.log10(FSRQ_LP_Lum_bins[i]),2), round(np.log10(FSRQ_LP_Lum_bins[i+1]),2)])
    Avg_FSRQ_SED_class.append('FSRQ')
    Avg_FSRQ_SED_SED_Class.append('LP')

    Masked_FSRQ_LP_Table = FSRQ_LP_Table[Lum_mask]
    print(round(np.log10(FSRQ_LP_Lum_bins[i]),2), round(np.log10(FSRQ_LP_Lum_bins[i+1]),2),len(Masked_FSRQ_LP_Table))
    med_z = np.median([float(x) for x in Masked_FSRQ_LP_Table['Redshift'].tolist()])
    Avg_FSRQ_SED_median_redshift.append(med_z)
    all_E2_dnde_space = []
    all_L_dnde_space = []
    
    for z in range(len(Masked_FSRQ_LP_Table)):
        E0 = float(Masked_FSRQ_LP_Table['E0'][z].tolist())
        N0 = float(Masked_FSRQ_LP_Table['N0'][z].tolist())
        LP_Alpha = float(Masked_FSRQ_LP_Table['LP_Alpha'][z])
        LP_Beta = float(Masked_FSRQ_LP_Table['LP_Beta'][z])
        redshift = float(Masked_FSRQ_LP_Table['Redshift'][z])
        
        nuFnu_LP_source = partial(nuFnu_LP, E0=E0, N0=N0, alpha=LP_Alpha, beta=LP_Beta)
        L_SED = dLdE(dnde=nuFnu_LP_source, z=redshift, log_E1=-2, log_E2=6)
        all_L_dnde_space.append(L_SED)
    
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    avg_SED_all_FSRQ.append(list(avg_SED))
    
    jackknife_errors = Jackknife_technique(log_SEDs)
    
    popt = Fit_nuLnu_LP(avg_SED=avg_SED, alpha_guess=2.0, beta_guess=0.2, energy_grid=E_rest_grid)
    L0_fit, alpha_fit, beta_fit = popt
    print(popt)
    
    Avg_FSRQ_SED_alpha.append(alpha_fit)
    Avg_FSRQ_SED_beta.append(beta_fit)
    Avg_FSRQ_SED_L0.append(L0_fit)
    
    model_SED = nuLnu_LP(E_rest_grid, *popt)

    E_peak = E_rest_grid[np.argmax(model_SED)]
    L_peak = np.max(model_SED)
    print(E_peak)
    
    ax_fsrq.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(FSRQ_LP_Lum_bins[i]),1)}-{round(np.log10(FSRQ_LP_Lum_bins[i+1]),1)}',color=FSRQ_colors[i])
    #ax_fsrq.plot(E_rest_grid, model_SED, linestyle='--', color='black')
    ax_fsrq.fill_between(E_rest_grid, 10**(avg_log_SED-jackknife_errors), 10**(avg_log_SED+jackknife_errors), alpha=0.3, color=FSRQ_colors[i])
    
    if z == len(Masked_FSRQ_LP_Table) - 1:
        handles, labels = ax_fsrq.get_legend_handles_labels()
        ax_fsrq.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
        ncol=3, fontsize=18, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)
    
    ax_fsrq.set_xscale('log')
    ax_fsrq.set_yscale('log')
    ax_fsrq.set_xlabel('Rest Frame Energy [MeV]',fontsize=21)
    ax_fsrq.set_ylabel(r'$E^2$dL/dE [$erg/s$]',fontsize=21)
    ax_fsrq.set_title("FSRQs", fontsize=24)
    ax_fsrq.tick_params(labelsize=18)
    #ax_fsrq.set_ylim(7e41, 1e50)

Avg_FSRQ_params_Table = Table([Avg_FSRQ_SED_L_bin, Avg_FSRQ_SED_class, Avg_FSRQ_SED_SED_Class, Avg_FSRQ_SED_median_redshift, Avg_FSRQ_SED_L0, Avg_FSRQ_SED_alpha, Avg_FSRQ_SED_beta], names=['Luminosity_Bin', 'Class', 'SED_Type', 'Redshift', 'L0', 'alpha', 'beta'])

# BLL LSP Template SEDS
print(f"Making LP BLL SED Templates for {len(BLL_LP_Table)} sources...")

BLL_cmap = plt.get_cmap('Reds')
BLL_colors = [BLL_cmap(x) for x in np.linspace(0.4, 0.9, 3)]

k_corr_lum_LP_BLL = []

for i in range(len(BLL_LP_Table)):
    Name = BLL_LP_Table['Name'][i]
    E0 = float(BLL_LP_Table['E0'][i].tolist())
    N0 = float(BLL_LP_Table['N0'][i].tolist())
    LP_Alpha = float(BLL_LP_Table['LP_Alpha'][i])
    LP_Beta = float(BLL_LP_Table['LP_Beta'][i])
    z = float(BLL_LP_Table['Redshift'][i])
    
    nuFnu_LP_source = partial(nuFnu_LP, E0=E0, N0=N0, alpha=LP_Alpha, beta=LP_Beta)
    k_corr_lum = K_Corrected_Band_luminosity(nuFnu = nuFnu_LP_source, z=z, E1=E0, E2=1e6)
    print(Name, k_corr_lum)
    k_corr_lum_LP_BLL.append(k_corr_lum)
    
BLL_LP_Lum_bins = equal_count_bins(k_corr_lum_LP_BLL,3)
BLL_LP_Table['k_corr_lum'] = k_corr_lum_LP_BLL

avg_SED_all_BLL_LP = []

Avg_BLL_LP_SED_L_bin = []
Avg_BLL_LP_SED_L0 = []
Avg_BLL_LP_SED_alpha = []
Avg_BLL_LP_SED_beta = []
Avg_BLL_LP_SED_class = []
Avg_BLL_LP_SED_SED_Class = []
Avg_BLL_LP_SED_median_redshift = []

for i in range(len(BLL_LP_Lum_bins)-1):
    if i == len(BLL_LP_Lum_bins) - 2:
        Lum_mask = (BLL_LP_Table['k_corr_lum'] >= BLL_LP_Lum_bins[i]) & ((BLL_LP_Table['k_corr_lum'] <= BLL_LP_Lum_bins[i+1]))
    else:
        Lum_mask = (BLL_LP_Table['k_corr_lum'] >= BLL_LP_Lum_bins[i]) & ((BLL_LP_Table['k_corr_lum'] < BLL_LP_Lum_bins[i+1]))
    
    Avg_BLL_LP_SED_L_bin.append([round(np.log10(BLL_LP_Lum_bins[i]),2), round(np.log10(BLL_LP_Lum_bins[i+1]),2)])
    Avg_BLL_LP_SED_class.append('BLL')
    Avg_BLL_LP_SED_SED_Class.append('LP')
    
    Masked_BLL_LP_Table = BLL_LP_Table[Lum_mask]
    print(round(np.log10(BLL_LP_Lum_bins[i]),2), round(np.log10(BLL_LP_Lum_bins[i+1]),2),len(Masked_BLL_LP_Table))
    med_z = np.median([float(x) for x in Masked_BLL_LP_Table['Redshift'].tolist()])
    Avg_BLL_LP_SED_median_redshift.append(med_z)
    all_E2_dnde_space = []
    all_L_dnde_space = []
    
    for z in range(len(Masked_BLL_LP_Table)):
        E0 = float(Masked_BLL_LP_Table['E0'][z].tolist())
        N0 = float(Masked_BLL_LP_Table['N0'][z].tolist())
        LP_Alpha = float(Masked_BLL_LP_Table['LP_Alpha'][z])
        LP_Beta = float(Masked_BLL_LP_Table['LP_Beta'][z])
        redshift = float(Masked_BLL_LP_Table['Redshift'][z])

        nuFnu_LP_source = partial(nuFnu_LP, E0=E0, N0=N0, alpha=LP_Alpha, beta=LP_Beta)
        L_SED = dLdE(dnde=nuFnu_LP_source, z=redshift, log_E1=-2, log_E2=6)
        all_L_dnde_space.append(L_SED)
        
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    avg_SED_all_BLL_LP.append(list(avg_SED))
    
    jackknife_errors = Jackknife_technique(log_SEDs)
    
    popt = Fit_nuLnu_LP(avg_SED=avg_SED, alpha_guess=2.0, beta_guess=0.2, energy_grid=E_rest_grid)
    L0_fit, alpha_fit, beta_fit = popt
    
    model_SED = nuLnu_LP(E_rest_grid, *popt)
    E_peak = E_rest_grid[np.argmax(model_SED)]
    L_peak = np.max(model_SED)
    print(E_peak)
    
    Avg_BLL_LP_SED_alpha.append(alpha_fit)
    Avg_BLL_LP_SED_beta.append(beta_fit)
    Avg_BLL_LP_SED_L0.append(L0_fit)
    
    ax_bll_lp.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(BLL_LP_Lum_bins[i]),1)}-{round(np.log10(BLL_LP_Lum_bins[i+1]),1)}',color=BLL_colors[i])
    #ax_bll_lp.plot(E_rest_grid, model_SED, linestyle='--', color='black')
    ax_bll_lp.fill_between(E_rest_grid, 10**(avg_log_SED-jackknife_errors), 10**(avg_log_SED+jackknife_errors), alpha=0.3, color=BLL_colors[i])
    
    if z == len(Masked_BLL_LP_Table) - 1:
        handles, labels = ax_bll_lp.get_legend_handles_labels()
        ax_bll_lp.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
        ncol=2, fontsize=18, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)
    
    ax_bll_lp.set_xscale('log')
    ax_bll_lp.set_yscale('log')
    ax_bll_lp.set_xlabel('Rest Frame Energy [MeV]', fontsize=21)
    ax_bll_lp.set_ylabel(r'$E^2$dL/dE [$erg/s$]', fontsize=21)
    ax_bll_lp.set_ylim(7e41, 1e50)
    ax_bll_lp.set_title("LP BLLs", fontsize=24)
    ax_bll_lp.tick_params(labelsize=18)

Avg_BLL_LSP_params_Table = Table([Avg_BLL_LP_SED_L_bin, Avg_BLL_LP_SED_class, Avg_BLL_LP_SED_SED_Class,Avg_BLL_LP_SED_median_redshift, Avg_BLL_LP_SED_L0, Avg_BLL_LP_SED_alpha, Avg_BLL_LP_SED_beta], names=['Luminosity_Bin', 'Class', 'SED_Type','Redshift', 'L0', 'alpha', 'beta'])

# BLL HSP Template SEDS

k_corr_lum_SBPL_BLL = []
print(f"Making SBPL BLL SED Templates for {len(BLL_SBPL_Table)} sources...")

for i in range(len(BLL_SBPL_Table)):
    E0 = float(BLL_SBPL_Table['E0'][i].tolist())
    N0 = float(BLL_SBPL_Table['N0'][i].tolist())
    SBPL_Index1 = float(BLL_SBPL_Table['SBPL_Index1'][i])
    SBPL_Index2 = float(BLL_SBPL_Table['SBPL_Index2'][i])
    SBPL_Break_Energy = float(BLL_SBPL_Table['SBPL_Break_Energy'][i])
    SBPL_Beta = float(BLL_SBPL_Table['SBPL_Beta'][i])
    z = float(BLL_SBPL_Table['Redshift'][i])
    
    nuFnu_SBPL_source = partial(nuFnu_SBPL, E0=E0, N0=N0, Gam1=SBPL_Index1, Gam2=SBPL_Index2, beta=-SBPL_Beta, E_Break=SBPL_Break_Energy)
    k_corr_lum = K_Corrected_Band_luminosity(nuFnu = nuFnu_SBPL_source, z=z, E1=E0, E2=1e6)
    k_corr_lum_SBPL_BLL.append(k_corr_lum)
        
BLL_SBPL_Lum_bins = equal_count_bins(k_corr_lum_SBPL_BLL,3)

BLL_SBPL_Table['k_corr_lum'] = k_corr_lum_SBPL_BLL

Avg_BLL_HSP_SED_L_bin = []
Avg_BLL_HSP_SED_L0 = []
Avg_BLL_HSP_SED_Eb = []
Avg_BLL_HSP_SED_alpha1 = []
Avg_BLL_HSP_SED_alpha2 = []
Avg_BLL_HSP_beta = []
Avg_BLL_HSP_SED_class = []
Avg_BLL_HSP_SED_SED_Class = []
Avg_BLL_HSP_SED_median_redshift = []

for i in range(len(BLL_SBPL_Lum_bins)-1):
    if i == len(BLL_SBPL_Lum_bins)-2:
        Lum_mask = (BLL_SBPL_Table['k_corr_lum'] >= BLL_SBPL_Lum_bins[i]) & ((BLL_SBPL_Table['k_corr_lum'] <= BLL_SBPL_Lum_bins[i+1]))
    else:
        Lum_mask = (BLL_SBPL_Table['k_corr_lum'] >= BLL_SBPL_Lum_bins[i]) & ((BLL_SBPL_Table['k_corr_lum'] < BLL_SBPL_Lum_bins[i+1]))    
    Masked_BLL_SBPL_Table = BLL_SBPL_Table[Lum_mask]
    print(round(np.log10(BLL_SBPL_Lum_bins[i]),2), round(np.log10(BLL_SBPL_Lum_bins[i+1]),2),len(Masked_BLL_SBPL_Table))
    med_z = np.median([float(x) for x in Masked_BLL_SBPL_Table['Redshift'].tolist()])
    Avg_BLL_HSP_SED_median_redshift.append(med_z)
    
    Avg_BLL_HSP_SED_L_bin.append([round(np.log10(BLL_SBPL_Lum_bins[i]),2), round(np.log10(BLL_SBPL_Lum_bins[i+1]),2)])
    
    Avg_BLL_HSP_SED_class.append('BLL')
    Avg_BLL_HSP_SED_SED_Class.append('SBPL')
    
    all_E2_dnde_space = []
    all_L_dnde_space = []
    
    for z in range(len(Masked_BLL_SBPL_Table)):
        E0 = float(Masked_BLL_SBPL_Table['E0'][z].tolist())
        N0 = float(Masked_BLL_SBPL_Table['N0'][z].tolist())
        SBPL_Index1 = float(Masked_BLL_SBPL_Table['SBPL_Index1'][z])
        SBPL_Index2 = float(Masked_BLL_SBPL_Table['SBPL_Index2'][z])
        SBPL_Break_Energy = float(Masked_BLL_SBPL_Table['SBPL_Break_Energy'][z])
        SBPL_Beta = float(Masked_BLL_SBPL_Table['SBPL_Beta'][z])
        redshift = float(Masked_BLL_SBPL_Table['Redshift'][z])

        nuFnu_SBPL_source = partial(nuFnu_SBPL, E0=E0, N0=N0, Gam1=SBPL_Index1, Gam2=SBPL_Index2, beta=-SBPL_Beta, E_Break=SBPL_Break_Energy)
        L_SED = dLdE(dnde=nuFnu_SBPL_source, z=redshift, log_E1=-2, log_E2=6)
        all_L_dnde_space.append(L_SED)
        
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    
    jackknife_errors = Jackknife_technique(log_SEDs)
        
    popt = Fit_nuLnu_SBPL(avg_SED, alpha1_guess=1.5, alpha2_guess=2.5, beta_guess=-2, energy_grid=E_rest_grid)

    L0_fit, Eb_fit, alpha1_fit, alpha2_fit, beta_fit = popt
    print(f'L0: {L0_fit}, Eb: {Eb_fit}, alpha1: {alpha1_fit}, alpha2: {alpha2_fit}, beta: {beta_fit}')
    Avg_BLL_HSP_SED_L0.append(L0_fit)
    Avg_BLL_HSP_SED_Eb.append(Eb_fit)
    Avg_BLL_HSP_SED_alpha1.append(alpha1_fit)
    Avg_BLL_HSP_SED_alpha2.append(alpha2_fit)
    Avg_BLL_HSP_beta.append(beta_fit)
    ax_bll_hsp.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(BLL_SBPL_Lum_bins[i]),1)}-{round(np.log10(BLL_SBPL_Lum_bins[i+1]),1)}',color=BLL_colors[i])
    #ax_bll_hsp.plot(E_rest_grid, nuLnu_SBPL(E_rest_grid, *popt), linestyle='--', color='black')
    ax_bll_hsp.fill_between(E_rest_grid, 10**(avg_log_SED-jackknife_errors), 10**(avg_log_SED+jackknife_errors), alpha=0.3, color=BLL_colors[i])
    
    if z == len(Masked_BLL_SBPL_Table) - 1:
        handles, labels = ax_bll_hsp.get_legend_handles_labels()
        ax_bll_hsp.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
        ncol=2, fontsize=18, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)
    
    ax_bll_hsp.set_xscale('log')
    ax_bll_hsp.set_yscale('log')
    ax_bll_hsp.set_xlabel('Rest Frame Energy [MeV]', fontsize=21)
    ax_bll_hsp.set_ylabel(r'$E^2$dL/dE [$erg/s$]',fontsize=21)
    ax_bll_hsp.set_title('SBPL BLLs', fontsize=24)
    ax_bll_hsp.tick_params(labelsize=18)
    ax_bll_hsp.set_ylim(7e41, 1e50)

fig.tight_layout()

plt.savefig(out_dir + 'SED_Templates.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir + 'SED_Templates.pdf', dpi=300, bbox_inches='tight')

plt.show()
plt.close()

Avg_BLL_HSP_params_Table = Table([Avg_BLL_HSP_SED_L_bin, Avg_BLL_HSP_SED_class, Avg_BLL_HSP_SED_SED_Class, Avg_BLL_HSP_SED_median_redshift, Avg_BLL_HSP_SED_L0, Avg_BLL_HSP_SED_Eb, Avg_BLL_HSP_SED_alpha1, Avg_BLL_HSP_SED_alpha2, Avg_BLL_HSP_beta], names=['Luminosity_Bin', 'Class', 'SED_Type', 'Redshift', 'L0', 'Eb', 'alpha1', 'alpha2', 'beta'])

SED_Template_Params_Table = vstack([Avg_FSRQ_params_Table, Avg_BLL_LSP_params_Table, Avg_BLL_HSP_params_Table])
SED_Template_Params_Table.write(catalog_dir + 'SED_Template_Params_Table.fits', overwrite=True)