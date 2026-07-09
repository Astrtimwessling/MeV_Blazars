import yaml
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np 
import math
import os
import time
import argparse

from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel, BrokenPowerLawSpectralModel, SmoothBrokenPowerLawSpectralModel
from sklearn.metrics import r2_score


start = time.time()

config_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Scripts/Configs/'
out_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/Blazar_Fitter_out/'

parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

Catalog = Table.read(config['Catalog_dir'] + config['Catalog_Name'])

Sensitivity_Data = Table.read(config['Catalog_dir'] + 'sensitivity_all_Erg.csv')

source_names = Catalog[config['columns']['source_name']].tolist()
source_class = Catalog[config['columns']['source_class']].tolist() 
SED_class = Catalog[config['columns']['source_SED_class']].tolist()
energy_bands = Catalog[config['columns']['energy_bands']].tolist()
energy_flux_bands = Catalog[config['columns']['energy_flux']].tolist()
energy_flux_bands_err = Catalog[config['columns']['energy_flux_err']].tolist()

alpha_results = []
alpha_results_err = []
beta_results = []
beta_results_err = []

IC_peak_energy = []
IC_peak_energy_err = []
IC_peak_energy_flux = []
IC_peak_energy_flux_err = []
GRAMS_E2dnde_MeV_results = []
eASTROGAM_E2dnde_MeV_results = []

x_ray_index_results = []
x_ray_index_err_results = []
gamma_ray_index_results = []
gamma_ray_index_err_results = []
ebreak_results = []
ebreak_results_err = []
break_flux_results = []

chi2_results = []
chi2_results_LAT = []
chi2_results_BAT = []
swift_r2_results = []
fermi_r2_results = []

for i in range(len(Catalog)):

    energy = energy_bands[i]
    flux = energy_flux_bands[i]
    flux_err = energy_flux_bands_err[i]

    if config['Fitting_X_ray_data'] == False: 
        energy_band = energy[8:]
        flux_band = flux[8:]
        flux_band_err = [flux_err[0][8:], flux_err[1][8:]]
    
    elif config['Fitting_Gamma_ray_data'] == False:
        energy_band = energy[:8]
        flux_band = flux[:8]
        flux_band_err = [flux_err[0][:8], flux_err[1][:8]]

    else:
        energy_band = energy
        flux_band = flux
        flux_band_err = flux_err

    is_ul = []

    for x in range(len(flux_band)):
        if flux_band[x] < 0:
            is_ul.append(True)
        else:
            is_ul.append(False)

    fluxpoints_table = Table([energy_band, flux_band, flux_band_err[0], flux_band_err[1], is_ul], names = ['e_ref', 'e2dnde', 'e2dnde_errn', 'e2dnde_errp', 'is_ul'])
    fluxpoints_table['e_ref'].unit = config['columns']['energy_units']
    fluxpoints_table['e2dnde'].unit = config['columns']['energy_flux_units']
    fluxpoints_table['e2dnde_errn'].unit = config['columns']['energy_flux_units']
    fluxpoints_table['e2dnde_errp'].unit = config['columns']['energy_flux_units']

    flux_points = FluxPoints.from_table(fluxpoints_table, sed_type="e2dnde")
    dataset = FluxPointsDataset(data = flux_points, name = "Fermi")
    dataset_MeV = Datasets([dataset])
    Flux_ref = str(flux_band[0] / (energy_band[0] ** 2)) + config['columns']['diff_flux_units']

    if source_class[i] == 'fsrq' or source_class[i] == 'FSRQ':
        plot_out_dir = out_dir + config['out_dir'] + config['fsrq_out']
        os.makedirs(plot_out_dir, exist_ok=True)
    elif source_class[i] == 'bll' or source_class[i] == 'BLL':
        plot_out_dir = out_dir + config['out_dir'] + config['bll_out']
        os.makedirs(plot_out_dir, exist_ok=True)
    elif source_class[i] == 'bcu' or source_class[i] == 'BCU':
        plot_out_dir = out_dir + config['out_dir'] + config['bcu_out']
        os.makedirs(plot_out_dir, exist_ok=True)

    if config['fit_parameters']['model'] == 'Log Parabola':
        if source_class[i] == 'fsrq' or source_class[i] == 'FSRQ':
            a = config['fit_parameters']['fsrq_alpha']
            b = config['fit_parameters']['fsrq_beta']
        elif source_class[i] == 'bll' or source_class[i] == 'BLL':
            a = config['fit_parameters']['bll_alpha']
            b = config['fit_parameters']['bll_beta']
        elif source_class[i] == 'bcu' or source_class[i] == 'BCU':
            a = config['fit_parameters']['bcu_alpha']
            b = config['fit_parameters']['bcu_beta']

        ndof = 2
        
        model = LogParabolaSpectralModel(alpha=a, amplitude=Flux_ref, reference=str(energy_band[0]) + config['columns']['energy_units'], beta=b)
        skymodel = SkyModel(spectral_model=model, name="j1507-lp")
        dataset_MeV.models = skymodel

        fitter = Fit()

        result_log_parabola = fitter.run(datasets=dataset_MeV)

        alpha_results.append(model.alpha.value)
        alpha_results_err.append(model.alpha.error)
        beta_results.append(model.beta.value)
        beta_results_err.append(model.beta.error)

        ref = model.reference.value
        ref_er = model.reference.error
        
        GRAMS_E2dnde_MeV_modeled = []
        for E in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        
            E2dnde_MeV = (float(E) * u.Unit(config['columns']['energy_units']))**2 * model(E * u.Unit(config['columns']['energy_units']))
            GRAMS_E2dnde_MeV_modeled.append(E2dnde_MeV.value)
        
        GRAMS_E2dnde_MeV_results.append(GRAMS_E2dnde_MeV_modeled)
        
        eASTROGAM_E2dnde_MeV_modeled = []
        for E in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        
            E2dnde_MeV = (float(E) * u.Unit(config['columns']['energy_units']))**2 * model(E * u.Unit(config['columns']['energy_units']))
            eASTROGAM_E2dnde_MeV_modeled.append(E2dnde_MeV.value)
        
        eASTROGAM_E2dnde_MeV_results.append(eASTROGAM_E2dnde_MeV_modeled)

        if model.beta.value >= 0:
            E_peak = ref*math.exp((2-model.alpha.value)/(2*model.beta.value))
            term1 = (ref_er / ref)**2
            term2 = (model.alpha.error / (2 * model.beta.value))**2
            term3 = (((2 - model.alpha.value) * model.beta.error) / (2 * model.beta.value**2))**2
            E_peak_err = E_peak * math.sqrt(term1 + term2 + term3)
            IC_peak_energy.append(float(E_peak))
            IC_peak_energy_err.append(float(E_peak_err))
            IC_peak_flux = (float(E_peak) * u.Unit(config['columns']['energy_units']))**2 * model(float(E_peak) * u.Unit(config['columns']['energy_units']))
            IC_peak_energy_flux.append(IC_peak_flux)
            
            alpha = model.alpha.value
            beta = model.beta.value
            norm = model.amplitude.value
            ref = model.reference.value
            E_unit = model.reference.unit
            
            alpha_err = model.alpha.error
            beta_err = model.beta.error
            norm_err = model.amplitude.error
            ref_err = 0
            
            # Define small perturbations for finite difference
            def F(alpha_, beta_, norm_, ref_):
                E_peak_ = ref_ * np.exp((2 - alpha_) / (2 * beta_))
                log_ratio_ = np.log(E_peak_ / ref_)
                return (E_peak_**2 * norm_ * (E_peak_ / ref_)**(-alpha_ - beta_ * log_ratio_))

            # Partial derivatives via central differences
            dFdalpha = (F(alpha + alpha_err, beta, norm, ref) - F(alpha - alpha_err, beta, norm, ref)) / (2 * alpha_err)
            dFdbeta  = (F(alpha, beta + beta_err, norm, ref) - F(alpha, beta - beta_err, norm, ref)) / (2 * beta_err)
            dFdN0    = (F(alpha, beta, norm + norm_err, ref) - F(alpha, beta, norm - norm_err, ref)) / (2 * norm_err)
            dFdref   = 0

            # Combine in quadrature
            F_peak_err = np.sqrt((dFdalpha * alpha_err)**2 + (dFdbeta * beta_err)**2 + (dFdN0 * norm_err)**2) * E_unit**2 * model.amplitude.unit
            IC_peak_energy_flux_err.append(F_peak_err)
            #print(IC_peak_flux,F_peak_err)

        else:
            IC_peak_energy.append('None')
            IC_peak_energy_err.append('None')
            IC_peak_energy_flux.append('None')
            IC_peak_energy_flux_err.append('None')
            

    elif config['fit_parameters']['model'] == 'Broken Power Law':
        x_ray_index = float(Catalog[config['columns']['x_ray_index']][i])
        gamma_ray_index = float(Catalog[config['columns']['gamma_ray_index']][i])

        model = BrokenPowerLawSpectralModel(amplitude=Flux_ref, index1=x_ray_index, index2=gamma_ray_index, ebreak=config['fit_parameters']['break_energy'] * u.Unit(config['columns']['energy_units']), reference=str(energy_band[0]) + config['columns']['energy_units'])

        skymodel = SkyModel(spectral_model=model, name="j1507-bpl")
        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)

        x_ray_index_results.append(model.index1.value)
        x_ray_index_err_results.append(model.index1.error)
        gamma_ray_index_results.append(model.index2.value)
        gamma_ray_index_err_results.append(model.index2.error)
        ebreak_results.append(model.ebreak.value)
        ebreak_results_err.append(model.ebreak.error)
        ndof=4

    elif config['fit_parameters']['model'] == 'Smooth Broken Power Law':
        x_ray_index = float(Catalog[config['columns']['x_ray_index']][i])
        gamma_ray_index = float(Catalog[config['columns']['gamma_ray_index']][i])

        model = SmoothBrokenPowerLawSpectralModel(amplitude=Flux_ref, index1=x_ray_index, index2=gamma_ray_index, ebreak=config['fit_parameters']['break_energy'] * u.Unit(config['columns']['energy_units']), beta=config['fit_parameters']['curvature'], reference=str(energy_band[0]) + config['columns']['energy_units'])

        skymodel = SkyModel(spectral_model=model, name="j1507-sbpl")
        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)

        x_ray_index_results.append(model.index1.value)
        x_ray_index_err_results.append(model.index1.error)
        gamma_ray_index_results.append(model.index2.value)
        gamma_ray_index_err_results.append(model.index2.error)
        #print(model.beta.value)
        ebreak_results.append(model.ebreak.value)
        #print(model.ebreak.value)
        ebreak_results_err.append(model.ebreak.error)
        ndof=3
        
        GRAMS_E2dnde_MeV_modeled = []
        for E in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        
            E2dnde_MeV = (float(E) * u.Unit(config['columns']['energy_units']))**2 * model(E * u.Unit(config['columns']['energy_units']))
            GRAMS_E2dnde_MeV_modeled.append(E2dnde_MeV.value)
        
        GRAMS_E2dnde_MeV_results.append(GRAMS_E2dnde_MeV_modeled)
        
        eASTROGAM_E2dnde_MeV_modeled = []
        for E in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        
            E2dnde_MeV = (float(E) * u.Unit(config['columns']['energy_units']))**2 * model(E * u.Unit(config['columns']['energy_units']))
            eASTROGAM_E2dnde_MeV_modeled.append(E2dnde_MeV.value)
        
        eASTROGAM_E2dnde_MeV_results.append(eASTROGAM_E2dnde_MeV_modeled)
        
        Break_Flux = (float(model.ebreak.value) * u.Unit(config['columns']['energy_units']))**2 * model(float(model.ebreak.value) * u.Unit(config['columns']['energy_units']))
        break_flux_results.append(Break_Flux)

    Modeled_E2dnde = []
    for x in energy_band:
        E2dnde = (x * u.Unit(config['columns']['energy_units']))**2 * model(x * u.Unit(config['columns']['energy_units']))
        Modeled_E2dnde.append(E2dnde.value)
    
    chi2_vals = []

    for z in range(len(Modeled_E2dnde)):
        if flux_band[z] > Modeled_E2dnde[z]:
            point_err = flux_band_err[0][z]
            if point_err == 0:
                point_err = flux_band_err[1][z]
    
        elif flux_band[z] < Modeled_E2dnde[z]:
            point_err = flux_band_err[1][z]
            if point_err == 0:
                point_err = flux_band_err[0][z]

        if flux_band[z] > 0:
            chi2_value = ((flux_band[z] - Modeled_E2dnde[z]) ** 2) / (point_err**2)
            #print(chi2_value)
            chi2_vals.append(chi2_value)

    ax = plt.subplot()
    kwargs = {"ax": ax, "sed_type": "e2dnde"}
    ax.yaxis.set_units(u.erg / (u.cm**2 * u.s))

    energy_bounds = [1e-2, 0.5e6] * u.Unit(config['columns']['energy_units'])
    dataset.models[0].spectral_model.plot(energy_bounds=energy_bounds, color="k", **kwargs)
    dataset.models[0].spectral_model.plot_error(energy_bounds=energy_bounds, **kwargs)
    ax.set_xlim(energy_bounds)

    chi2_score = np.sum(chi2_vals) / (len(flux_band) - ndof)
    #print(chi2_score)
    chi2_results.append(chi2_score)
    chi2_results_LAT.append(np.sum(chi2_vals[8:]) / (len(flux_band) - ndof))
    chi2_results_BAT.append(np.sum(chi2_vals[:8]) / (len(flux_band) - ndof))

    r2_score_swift = r2_score(flux_band[:8], Modeled_E2dnde[:8])
    swift_r2_results.append(r2_score_swift)
    r2_score_fermi = r2_score(flux_band[8:], Modeled_E2dnde[8:])
    fermi_r2_results.append(r2_score_fermi)
    
    if config['Show_sensitivity_curve'] == True:
        GRAMS_Balloon_Energy = Sensitivity_Data['_GRAMS_Balloon_x'].tolist()[:9]
        GRAMS_Ballon_Sensitivity = [(x / 1.602176634e-6)  for x in Sensitivity_Data['_GRAMS_Balloon_y'].tolist()[:9]]

        GRAMS_Satellite_Energy = Sensitivity_Data['_GRAMS_Satellite_x'].tolist()[:9]
        GRAMS_Satellite_Sensitivity = [(x / 1.602176634e-6)  for x in Sensitivity_Data['_GRAMS_Satellite_y'].tolist()[:9]]

        e_ASTROGAM_energy = Sensitivity_Data['e-Astrogam_x'].tolist()
        e_ASTROGAM_sensitivity = [(x / 1.602176634e-6)  for x in Sensitivity_Data['e-Astrogam_y'].tolist()]

        #COSI_energy = Sensitivity_Data['COSI_x'].tolist()
        #COSI_sensitivity = Sensitivity_Data['COSI_y'].tolist()

        plt.plot(GRAMS_Balloon_Energy, GRAMS_Ballon_Sensitivity, color='red',  markersize=3)
        plt.plot(GRAMS_Satellite_Energy, GRAMS_Satellite_Sensitivity, color='blue',  markersize=3)
        plt.plot(e_ASTROGAM_energy,e_ASTROGAM_sensitivity, color='magenta',markersize=3,linestyle='--')
        #plt.plot(COSI_energy, COSI_sensitivity, color='green',markersize=3,linestyle='--')


        plt.xscale('log')
        plt.yscale('log')

        #plt.legend(loc = 'lower right')
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        
    flux_erg = [x * 1.602176634e-6 for x in flux]
    flux_erg_err = [[x * 1.602176634e-6 for x in flux_err[0]], [x * 1.602176634e-6 for x in flux_err[1]]]
    
    plt.title(f"{source_names[i]} ({source_class[i].upper()}, {SED_class[i]})", fontsize=14)

    is_ul = np.array(is_ul)
    
    det_swift = ~is_ul[:8]
    ul_swift = is_ul[:8]
    
    det_fermi = ~is_ul[8:]
    ul_fermi = is_ul[8:]
    
    energy = np.array(energy)
    flux_erg = np.array(flux_erg)
    flux_erg_err = [np.array(flux_erg_err[0]), np.array(flux_erg_err[1])]
    
    energy_ul_swift = energy[:8][ul_swift]
    flux_erg_ul_swift = flux_erg[:8][ul_swift]
    flux_err_erg_ul_swift = flux_erg_err[1][:8][ul_swift]
    print(flux_erg_ul_swift, flux_err_erg_ul_swift)
    upper_lim_swift = flux_erg_ul_swift + flux_err_erg_ul_swift
    print(upper_lim_swift)
    
    energy_ul_fermi = energy[8:][ul_fermi]
    flux_erg_ul_fermi = flux_erg[8:][ul_fermi]
    flux_err_erg_ul_fermi = flux_erg_err[1][8:][ul_fermi]
    upper_lim_fermi = flux_erg_ul_fermi + flux_err_erg_ul_fermi
    print(upper_lim_fermi)
    
    plt.errorbar(energy[:8][det_swift], flux_erg[:8][det_swift], yerr=[flux_erg_err[0][:8][det_swift], flux_erg_err[1][:8][det_swift]], fmt='o',markersize=3,color='red',label='Swift')
    plt.errorbar(energy[8:][det_fermi], flux_erg[8:][det_fermi], yerr=[flux_erg_err[0][-8:][det_fermi], flux_erg_err[1][-8:][det_fermi]], fmt='o',markersize=3,color='blue',label='Fermi')
    for x, y in zip(energy_ul_swift, upper_lim_swift):
            plt.annotate('', xy=(x, y/3), xytext=(x, y), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
    for x, y in zip(energy_ul_fermi, upper_lim_fermi):
            plt.annotate('', xy=(x, y/3), xytext=(x, y), arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Energy ({config['columns']['energy_units']})", fontsize=14)
    plt.ylabel(r"$E^2 \frac{dN}{dE}$" + r" [$erg/cm^{2}/s$]", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    #plt.ylim(1e-15, 1e-9)
    plt.savefig(plot_out_dir + source_names[i] + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(plot_out_dir + source_names[i] + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    
if config['fit_parameters']['model'] == 'Log Parabola':
    Output = Table([source_names, source_class, SED_class, alpha_results, alpha_results_err, beta_results, beta_results_err, IC_peak_energy, IC_peak_energy_err, IC_peak_energy_flux,IC_peak_energy_flux_err, GRAMS_E2dnde_MeV_results, eASTROGAM_E2dnde_MeV_results,chi2_results, chi2_results_BAT, chi2_results_LAT, swift_r2_results, fermi_r2_results], names=['Name', 'Class','SED_Class', 'Alpha', 'Alpha_Err', 'Beta', 'Beta_Err', 'IC_Peak_Energy', 'IC_Peak_Energy_Err','IC_Peak_Energy_Flux','IC_Peak_Energy_Flux_Err','Flux_MeV_GRAMS','Flux_MeV_eASTROGAM', 'Chi2','Chi2_BAT','Chi2_LAT','r2_BAT','r2_LAT'])
if config['fit_parameters']['model'] == 'Broken Power Law':
    Output = Table([source_names, source_class, SED_class, x_ray_index_results, x_ray_index_err_results, gamma_ray_index_results, gamma_ray_index_err_results, ebreak_results, ebreak_results_err, chi2_results, chi2_results_BAT, chi2_results_LAT, swift_r2_results, fermi_r2_results], names=['Name', 'Class','SED_Class', 'X_Ray_Index', 'X_Ray_Index_Err', 'Gamma_Ray_Index', 'Gamma_Ray_Index_Err', 'Break_Energy', 'Break_Energy_Err', 'Chi2','Chi2_BAT','Chi2_LAT','r2_BAT','r2_LAT'])
if config['fit_parameters']['model'] == 'Smooth Broken Power Law':
    Output = Table([source_names, source_class, SED_class, x_ray_index_results, x_ray_index_err_results, gamma_ray_index_results, gamma_ray_index_err_results, ebreak_results, ebreak_results_err, break_flux_results, GRAMS_E2dnde_MeV_results,eASTROGAM_E2dnde_MeV_results, chi2_results, chi2_results_BAT, chi2_results_LAT, swift_r2_results, fermi_r2_results], names=['Name', 'Class','SED_Class', 'X_Ray_Index', 'X_Ray_Index_Err', 'Gamma_Ray_Index', 'Gamma_Ray_Index_Err', 'Break_Energy', 'Break_Energy_Err','Break_Energy_Flux','Flux_MeV_GRAMS','Flux_MeV_eASTROGAM' ,'Chi2','Chi2_BAT','Chi2_LAT','r2_BAT','r2_LAT'])
Output.write(config['Catalog_dir'] + config['out_file_name'], overwrite=True)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")