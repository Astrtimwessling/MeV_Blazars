import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np 

from astropy.table import Table, vstack
from scipy.integrate import quad
from scipy.optimize import curve_fit

def energy_flux(dnde, E1, E2):

    def integrand(logE):
        E = np.exp(logE)
        return E**2 * dnde(E)

    result = quad(integrand, np.log(E1), np.log(E2))[0]

    return result * 1.602176634e-6

def equal_count_bins(data, n_bins):
    data = np.array(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(data, quantiles)
    return bins

def Luminosity_distance(z):
    c = 3 * (10 ** 5) # km/s
    H_0 = 73 # km/s
    omega_m = 0.3
    def f(x):
        return 1 / np.sqrt(omega_m * (1+x)**3 + 1 - omega_m)

    integral, _ = quad(f, 1, 1 + z)
    return (1 + z) * (c / H_0) * integral 

def nuLnu_LP(E, L0, alpha, beta, E0=1.0):
    return L0 * (E / E0) ** (2 - alpha - beta * np.log(E / E0))

def dLdE(E, L0, alpha, beta, E0=1.0):
    return nuLnu_LP(E, L0, alpha, beta, E0) / E

catalog_dir = "/home/alab_student/Tim/Projects/Catalogs/"
out_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/Template_SEDS/"

params_catalog = Table.read(catalog_dir + 'Model_Param_LF_Masked.fits')

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

# FSRQ Template SEDS
print(f"Making FSRQ SED Templates for {len(FSRQ_LP_Table)} sources...")

k_corr_lum_LP_FSRQ = []

for i in range(len(FSRQ_LP_Table)):
    E0 = float(FSRQ_LP_Table['E0'][i].tolist())
    
    N0 = float(FSRQ_LP_Table['N0'][i].tolist())
    LP_Alpha = float(FSRQ_LP_Table['LP_Alpha'][i])
    LP_Beta = float(FSRQ_LP_Table['LP_Beta'][i])
    z = float(FSRQ_LP_Table['Redshift'][i])
    
    def dnde(E):
        return N0 * (E / E0)**(-LP_Alpha - LP_Beta * np.log(E / E0))
    
    total_int_flux = energy_flux(dnde, E0, 1e6)
    total_int_flux_rest = energy_flux(dnde, E0*(1+z), 1e6*(1+z))
    
    k_corr = total_int_flux_rest / total_int_flux
    
    d_L = Luminosity_distance(z)
    
    K_corr_lum = total_int_flux * k_corr * ((d_L*3.085677581e24)**2) * 4 * np.pi
    k_corr_lum_LP_FSRQ.append(K_corr_lum)
    
FSRQ_LP_Lum_bins = equal_count_bins(k_corr_lum_LP_FSRQ,5)

FSRQ_LP_Table['k_corr_lum'] = k_corr_lum_LP_FSRQ

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
    
        def dnde(E):
            return N0 * (E / E0)**(-LP_Alpha - LP_Beta * np.log(E / E0))
        
        E_rest_grid = np.logspace(-2, 6, 100)
        E_obs = E_rest_grid / (1 + redshift)

        dnde_obs_vals = np.array([dnde(x) for x in E_obs])
        dnde_rest = (1 + redshift) * dnde_obs_vals

        E2_dnde_rest = E_rest_grid**2 * dnde_rest * 1.602176634e-6
        all_E2_dnde_space.append(E2_dnde_rest)
        
        d_L = Luminosity_distance(redshift) * 3.085677581e24
        L_SED = 4 * np.pi * d_L**2 * E2_dnde_rest
        all_L_dnde_space.append(L_SED)
    
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    avg_SED_all_FSRQ.append(list(avg_SED))
    
    N = log_SEDs.shape[0]
    
    jackknife_means = []
    
    for x in range(N):
    
        jk_logs = np.delete(log_SEDs, x, axis=0)
        jk_mean = np.mean(jk_logs, axis=0)
        jackknife_means.append(jk_mean)
    
    jackknife_means = np.array(jackknife_means)
    jk_bar = np.mean(jackknife_means, axis=0)
    jk_var = (N-1)/N * np.sum((jackknife_means - jk_bar)**2, axis=0)
    jk_err = np.sqrt(jk_var)
    
    def log_model(E, L0, alpha, beta):
        return np.log10(nuLnu_LP(E, L0, alpha, beta))

    # Initial guesses
    L0_guess = np.max(avg_SED)
    alpha_guess = 2.0
    beta_guess = 0.2

    p0 = [L0_guess, alpha_guess, beta_guess]

    popt, pcov = curve_fit(log_model, E_rest_grid, np.log10(avg_SED), p0=p0, maxfev=10000)

    L0_fit, alpha_fit, beta_fit = popt
    
    Avg_FSRQ_SED_alpha.append(alpha_fit)
    Avg_FSRQ_SED_beta.append(beta_fit)
    Avg_FSRQ_SED_L0.append(L0_fit)
    
    model_SED = 10**log_model(E_rest_grid, *popt)

    E_peak = E_rest_grid[np.argmax(model_SED)]
    L_peak = np.max(model_SED)
    print(E_peak)
        
    ax_fsrq.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(FSRQ_LP_Lum_bins[i]),1)}-{round(np.log10(FSRQ_LP_Lum_bins[i+1]),1)}',color=FSRQ_colors[i])
    ax_fsrq.fill_between(E_rest_grid, 10**(avg_log_SED-jk_err), 10**(avg_log_SED+jk_err), alpha=0.3, color=FSRQ_colors[i])

    #ax_fsrq.plot(E_rest_grid, 10**log_model(E_rest_grid, *popt), color='black', linestyle='--')
    
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
    
#plt.savefig(out_dir + 'FSRQ_SED_Templates.png', dpi=300, bbox_inches='tight')
#plt.close()

Avg_FSRQ_params_Table = Table([Avg_FSRQ_SED_L_bin, Avg_FSRQ_SED_class, Avg_FSRQ_SED_SED_Class, Avg_FSRQ_SED_median_redshift, Avg_FSRQ_SED_L0, Avg_FSRQ_SED_alpha, Avg_FSRQ_SED_beta], names=['Luminosity_Bin', 'Class', 'SED_Type', 'Redshift', 'L0', 'alpha', 'beta'])

# BLL LSP Template SEDS
print(f"Making LP BLL SED Templates for {len(BLL_LP_Table)} sources...")

BLL_cmap = plt.get_cmap('Reds')
BLL_colors = [BLL_cmap(x) for x in np.linspace(0.4, 0.9, 3)]

k_corr_lum_LP_BLL = []

for i in range(len(BLL_LP_Table)):
    E0 = float(BLL_LP_Table['E0'][i].tolist())
    N0 = float(BLL_LP_Table['N0'][i].tolist())
    LP_Alpha = float(BLL_LP_Table['LP_Alpha'][i])
    LP_Beta = float(BLL_LP_Table['LP_Beta'][i])
    z = float(BLL_LP_Table['Redshift'][i])
    
    def dnde(E):
        return N0 * (E / E0)**(-LP_Alpha - LP_Beta * np.log(E / E0))
    
    total_int_flux = energy_flux(dnde, E0, 1e6)
    total_int_flux_rest = energy_flux(dnde, E0*(1+z), 1e6*(1+z))
    
    k_corr = total_int_flux_rest / total_int_flux
    
    d_L = Luminosity_distance(z)
    
    K_corr_lum = total_int_flux * k_corr * ((d_L*3.085677581e24)**2) * 4 * np.pi
    k_corr_lum_LP_BLL.append(K_corr_lum)
    
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

        def dnde(E):
            return N0 * (E / E0)**(-LP_Alpha - LP_Beta * np.log(E / E0))
        
        E_rest_grid = np.logspace(-2, 6, 100)
        E_obs = E_rest_grid / (1 + redshift)

        dnde_obs_vals = np.array([dnde(x) for x in E_obs])
        dnde_rest = (1 + redshift) * dnde_obs_vals

        E2_dnde_rest = E_rest_grid**2 * dnde_rest * 1.602176634e-6
        all_E2_dnde_space.append(E2_dnde_rest)
        
        d_L = Luminosity_distance(redshift) * 3.085677581e24
        L_SED = 4 * np.pi * d_L**2 * E2_dnde_rest
        all_L_dnde_space.append(L_SED)
    
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    
    N = log_SEDs.shape[0]

    jackknife_means = []

    for x in range(N):

        jk_logs = np.delete(log_SEDs, x, axis=0)
        jk_mean = np.mean(jk_logs, axis=0)
        jackknife_means.append(jk_mean)

    jackknife_means = np.array(jackknife_means)
    jk_bar = np.mean(jackknife_means, axis=0)
    jk_var = (N-1)/N * np.sum((jackknife_means - jk_bar)**2, axis=0)
    jk_err = np.sqrt(jk_var)
    
    avg_SED_all_BLL_LP.append(list(avg_SED))
    
    def log_model(E, L0, alpha, beta):
        return np.log10(nuLnu_LP(E, L0, alpha, beta))

    # Initial guesses
    L0_guess = np.max(avg_SED)
    alpha_guess = 2.0
    beta_guess = 0.2

    p0 = [L0_guess, alpha_guess, beta_guess]

    popt, pcov = curve_fit(log_model, E_rest_grid, np.log10(avg_SED), p0=p0, maxfev=10000)

    L0_fit, alpha_fit, beta_fit = popt
    
    model_SED = 10**log_model(E_rest_grid, *popt)

    E_peak = E_rest_grid[np.argmax(model_SED)]
    L_peak = np.max(model_SED)
    print(E_peak)
    
    Avg_BLL_LP_SED_alpha.append(alpha_fit)
    Avg_BLL_LP_SED_beta.append(beta_fit)
    Avg_BLL_LP_SED_L0.append(L0_fit)
    
    ax_bll_lp.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(BLL_LP_Lum_bins[i]),1)}-{round(np.log10(BLL_LP_Lum_bins[i+1]),1)}',color=BLL_colors[i])
    ax_bll_lp.fill_between(E_rest_grid, 10**(avg_log_SED-jk_err), 10**(avg_log_SED+jk_err), alpha=0.3, color=BLL_colors[i])
    #ax_bll_lp.plot(E_rest_grid, 10**log_model(E_rest_grid, *popt), color='black', linestyle='--')
    
    if z == len(Masked_BLL_LP_Table) - 1:
        handles, labels = ax_bll_lp.get_legend_handles_labels()
        ax_bll_lp.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
        ncol=2, fontsize=18, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)
    
    ax_bll_lp.set_xscale('log')
    ax_bll_lp.set_yscale('log')
    ax_bll_lp.set_xlabel('Rest Frame Energy [MeV]', fontsize=21)
    ax_bll_lp.set_ylabel(r'$E^2$dL/dE [$erg/s$]', fontsize=21)
    ax_bll_lp.set_ylim(0.9e42,6e47)
    ax_bll_lp.set_title("LP BLLs", fontsize=24)
    ax_bll_lp.tick_params(labelsize=18)

#plt.savefig(out_dir + 'LSP_BLL_SED_Templates.png', bbox_inches='tight', dpi=300)
#plt.close()

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
    
    def dnde(E, E0=E0, N0=N0, Gam1=SBPL_Index1, Gam2=SBPL_Index2, beta=-SBPL_Beta, E_Break=SBPL_Break_Energy):
        term1 = (E / E0)**(-Gam1)
        term2 = (1 + (E / E_Break)**((Gam2 - Gam1) / beta))**(-beta)
        return N0 * term1 * term2
    
    total_int_flux = energy_flux(dnde, E0, 1e6)
    total_int_flux_rest = energy_flux(dnde, E0*(1+z), 1e6*(1+z))
    
    k_corr = total_int_flux_rest / total_int_flux
    
    d_L = Luminosity_distance(z)
    
    K_corr_lum = total_int_flux * k_corr * ((d_L*3.085677581e24)**2) * 4 * np.pi
    k_corr_lum_SBPL_BLL.append(K_corr_lum)
    
BLL_SBPL_Lum_bins = equal_count_bins(k_corr_lum_SBPL_BLL,3)

BLL_SBPL_Table['k_corr_lum'] = k_corr_lum_SBPL_BLL

Avg_BLL_HSP_SED_L_bin = []
Avg_BLL_HSP_SED_L0 = []
Avg_BLL_HSP_SED_Eb = []
Avg_BLL_HSP_SED_alpha1 = []
Avg_BLL_HSP_SED_alpha2 = []
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

        def dnde(E, E0=E0, N0=N0, Gam1=SBPL_Index1, Gam2=SBPL_Index2, beta=-SBPL_Beta, E_Break=SBPL_Break_Energy):
            term1 = (E / E0)**(-Gam1)
            term2 = (1 + (E / E_Break)**((Gam2 - Gam1) / beta))**(-beta)
            return N0 * term1 * term2
    
        E_rest_grid = np.logspace(-2, 6, 100)
        E_obs = E_rest_grid / (1 + redshift)

        dnde_obs_vals = np.array([dnde(x) for x in E_obs])
        dnde_rest = (1 + redshift) * dnde_obs_vals

        E2_dnde_rest = E_rest_grid**2 * dnde_rest * 1.602176634e-6
        all_E2_dnde_space.append(E2_dnde_rest)
        
        d_L = Luminosity_distance(redshift) * 3.085677581e24
        L_SED = 4 * np.pi * d_L**2 * E2_dnde_rest
        all_L_dnde_space.append(L_SED)
    
    log_SEDs = np.log10(all_L_dnde_space)
    avg_log_SED = np.mean(log_SEDs, axis=0)
    avg_SED = 10**avg_log_SED
    
    N = log_SEDs.shape[0]
    
    jackknife_means = []
    
    for x in range(N):
    
        jk_logs = np.delete(log_SEDs, x, axis=0)
        jk_mean = np.mean(jk_logs, axis=0)
        jackknife_means.append(jk_mean)
    
    jackknife_means = np.array(jackknife_means)
    jk_bar = np.mean(jackknife_means, axis=0)
    jk_var = (N-1)/N * np.sum((jackknife_means - jk_bar)**2, axis=0)
    jk_err = np.sqrt(jk_var)
        
    
    def nuLnu_SBPL(E, L0, Eb, alpha1, alpha2, s):
        term = (E / Eb)
        return L0 * term**(-alpha1) * (1 + term**((alpha2 - alpha1)/s))**(-s)
    
    def SBPL_fit(E, L0, Eb, alpha1, alpha2):
        s = -1
        return nuLnu_SBPL(E, L0, Eb, alpha1, alpha2, s)
    
    def log_SBPL(E, L0, Eb, alpha1, alpha2):
        return np.log10(SBPL_fit(E, L0, Eb, alpha1, alpha2))

    
    L0_guess = np.max(avg_SED)
    Eb_guess = E_rest_grid[np.argmax(avg_SED)]
    alpha1_guess = 1.5
    alpha2_guess = 2.5

    p0 = [L0_guess, Eb_guess, alpha1_guess, alpha2_guess]

    popt, pcov = curve_fit(log_SBPL, E_rest_grid, np.log10(avg_SED), p0=p0, maxfev=10000)

    L0_fit, Eb_fit, alpha1_fit, alpha2_fit = popt
    print(f'L0: {L0_fit}, Eb: {Eb_fit}, alpha1: {alpha1_fit}, alpha2: {alpha2_fit}')
    Avg_BLL_HSP_SED_L0.append(L0_fit)
    Avg_BLL_HSP_SED_Eb.append(Eb_fit)
    Avg_BLL_HSP_SED_alpha1.append(alpha1_fit)
    Avg_BLL_HSP_SED_alpha2.append(alpha2_fit)
    
    ax_bll_hsp.plot(E_rest_grid, avg_SED, label =f'{round(np.log10(BLL_SBPL_Lum_bins[i]),1)}-{round(np.log10(BLL_SBPL_Lum_bins[i+1]),1)}',color=BLL_colors[i])
    ax_bll_hsp.fill_between(E_rest_grid, 10**(avg_log_SED-jk_err), 10**(avg_log_SED+jk_err), alpha=0.3, color=BLL_colors[i])

    #ax_bll_hsp.plot(E_rest_grid, 10**log_SBPL(E_rest_grid, *popt), color='black', linestyle='--')
    
    if z == len(Masked_BLL_SBPL_Table) - 1:
        handles, labels = ax_bll_hsp.get_legend_handles_labels()
        ax_bll_hsp.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
        ncol=2, fontsize=18, frameon=False, reverse=False, columnspacing=1.2, handletextpad=0.5)
    
    #ax_bll_hsp.legend(loc='lower left', fontsize=9)
    ax_bll_hsp.set_xscale('log')
    ax_bll_hsp.set_yscale('log')
    ax_bll_hsp.set_xlabel('Rest Frame Energy [MeV]', fontsize=21)
    ax_bll_hsp.set_ylabel(r'$E^2$dL/dE [$erg/s$]',fontsize=21)
    ax_bll_hsp.set_title('SBPL BLLs', fontsize=24)
    ax_bll_hsp.tick_params(labelsize=18)

fig.tight_layout()

plt.savefig(out_dir + 'SED_Templates.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir + 'SED_Templates.pdf', dpi=300, bbox_inches='tight')

plt.show()
plt.close()

Avg_BLL_HSP_params_Table = Table([Avg_BLL_HSP_SED_L_bin, Avg_BLL_HSP_SED_class, Avg_BLL_HSP_SED_SED_Class, Avg_BLL_HSP_SED_median_redshift, Avg_BLL_HSP_SED_L0, Avg_BLL_HSP_SED_Eb, Avg_BLL_HSP_SED_alpha1, Avg_BLL_HSP_SED_alpha2], names=['Luminosity_Bin', 'Class', 'SED_Type', 'Redshift', 'L0', 'Eb', 'alpha1', 'alpha2'])

SED_Template_Params_Table = vstack([Avg_FSRQ_params_Table, Avg_BLL_LSP_params_Table, Avg_BLL_HSP_params_Table])
SED_Template_Params_Table.write(catalog_dir + 'SED_Template_Params_Table.fits', overwrite=True)