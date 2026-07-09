import yaml
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np 
import os
import argparse
import time
import math

from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel, PowerLawSpectralModel, BrokenPowerLawSpectralModel, SmoothBrokenPowerLawSpectralModel
from sklearn.metrics import r2_score
from scipy.optimize import brentq
from scipy.integrate import quad

start = time.time()


config_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Scripts/Configs/'
out_dir = '/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/Flux_Fitter_out/'

parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

Catalog = Table.read(config['Catalog_dir'] + config['Catalog_Name'])

source_names = Catalog[config['columns']['source_name']].tolist()
source_class = Catalog[config['columns']['source_class']].tolist() 
energy_bands = Catalog[config['columns']['energy_bands']].tolist()
energy_flux_bands = Catalog[config['columns']['energy_flux']].tolist()
energy_flux_bands_err = Catalog[config['columns']['energy_flux_err']].tolist()
Fermi_redshifts = Catalog[config['columns']['Fermi_Redshift']].tolist()
Swift_redshifts = Catalog[config['columns']['Swift_Redshift']].tolist()

Redshifts = []

for z in range(len(Fermi_redshifts)):
    if Fermi_redshifts[z] != 'None':
        Redshifts.append(Fermi_redshifts[z])
    elif Fermi_redshifts[z] == 'None' and Swift_redshifts[z] != 'None':
        Redshifts.append(Swift_redshifts[z])
    else:
        Redshifts.append('None')
        
        
omega_m = 0.3
c = 3 * (10**5) # km/s
H_0 = 73 # km/s

def Luminosity_distance(z):
    def f(x):
        return 1 / np.sqrt(omega_m * (x**3) + 1 - omega_m)

    integral, _ = quad(f, 1, 1 + z)
    return (1 + z) * (c / H_0) * integral 

def luminosity(d_L, flux):
    L = 4 * math.pi * (d_L * 3.085677581e24)**2 * flux 

    return L

index_results = []
index_results_err = []
alpha_results = []
alpha_results_err = []
beta_results = []
beta_results_err = []

integral_energy_flux_results = []
integral_photon_flux_results = []
integral_energy_flux_err_results = []
integral_photon_flux_err_results = []
chi2_results = []
CGrB_Flux_Bins_all = []
GRAMS_Flux_Bins_all = []
k_corrections_all = []

x_ray_index_results = []
x_ray_index_err_results = []
gamma_ray_index_results = []
gamma_ray_index_err_results = []
ebreak_results = []
ebreak_results_err = []
r2_results = []

reference_results = []
amplitude_results = []
z_max_array_results = []
GRAMS_Red_Flux_Bins_all = []


def convert_to_diff_flux(energy_bands, energy_flux):
    diff_flux = [x / (y ** 2) for x,y in zip(energy_flux, energy_bands)]
    return diff_flux

for i in range(len(Catalog)):

    energy = energy_bands[i]
    flux = energy_flux_bands[i]
    flux_err = energy_flux_bands_err[i]

    if config['Fitting_X_ray_data'] == False: 
        energy_band = energy[8:]
        flux_band = convert_to_diff_flux(energy[8:],flux[8:])
        flux_band_err = [convert_to_diff_flux(energy[8:],flux_err[0][8:]), convert_to_diff_flux(energy[8:],flux_err[0][8:])]
    
    elif config['Fitting_Gamma_ray_data'] == False:
        energy_band = energy[:8]
        flux_band = flux[:8]
        flux_band_err = [flux_err[0][:8], flux_err[1][:8]]

    else:
        energy_band = energy
        flux_band = convert_to_diff_flux(energy,flux)
        flux_band_err = [convert_to_diff_flux(energy,flux_err[0]), convert_to_diff_flux(energy,flux_err[1])]

    is_ul = []

    for x in range(len(flux_band)):
        if flux_band[x] < 0:
            is_ul.append(True)
        else:
            is_ul.append(False)

    fluxpoints_table = Table([energy_band, flux_band, flux_band_err[0], flux_band_err[1], is_ul], names = ['e_ref', 'dnde', 'dnde_errn', 'dnde_errp','is_ul'])
    fluxpoints_table['e_ref'].unit = config['columns']['energy_units']
    fluxpoints_table['dnde'].unit = config['columns']['diff_flux_units']
    fluxpoints_table['dnde_errn'].unit = config['columns']['diff_flux_units']
    fluxpoints_table['dnde_errp'].unit = config['columns']['diff_flux_units']

    flux_points = FluxPoints.from_table(fluxpoints_table, sed_type="dnde")
    dataset = FluxPointsDataset(data = flux_points, name = "Fermi")
    Flux_ref = flux_band[0] * u.Unit(config['columns']['diff_flux_units'])

    if source_class[i] == 'fsrq' or source_class[i] == 'FSRQ':
        plot_out_dir = out_dir + config['out_dir'] + config['fsrq_out']
        os.makedirs(plot_out_dir, exist_ok=True)
    elif source_class[i] == 'bll' or source_class[i] == 'BLL':
        plot_out_dir = out_dir + config['out_dir'] + config['bll_out']
        os.makedirs(plot_out_dir, exist_ok=True)
    elif source_class[i] == 'bcu' or source_class[i] == 'BCU':
        plot_out_dir = out_dir + config['out_dir'] + config['bcu_out']
        os.makedirs(plot_out_dir, exist_ok=True)

    dataset_MeV = Datasets([dataset])

    if config['fit_parameters']['model'] == 'Power Law':

        model = PowerLawSpectralModel(index=config['fit_parameters']['index'], amplitude=Flux_ref, reference=str(energy_band[0]) + config['columns']['energy_units'])
        skymodel = SkyModel(spectral_model=model, name="j1507-pl")

        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)
        ax = plt.subplot()
        ax.yaxis.set_units(u.Unit("MeV-1 cm-2 s-1"))
        kwargs = {"ax": ax, "sed_type": "dnde"}

        index_results.append(model.index.value)
        index_results_err.append(model.index.error)
        ndof = 2

    elif config['fit_parameters']['model'] == 'Log Parabola':

        if source_class[i] == 'fsrq' or source_class[i] == 'FSRQ':
            a = config['fit_parameters']['fsrq_alpha']
            b = config['fit_parameters']['fsrq_beta']

        elif source_class[i] == 'bll' or source_class[i] == 'BLL':
            a = config['fit_parameters']['bll_alpha']
            b = config['fit_parameters']['bll_beta']

        elif source_class[i] == 'bcu' or source_class[i] == 'BCU':
            a = config['fit_parameters']['bcu_alpha']
            b = config['fit_parameters']['bcu_beta']

        model = LogParabolaSpectralModel(alpha=a, amplitude=Flux_ref, reference=str(energy_band[0]) + config['columns']['energy_units'], beta=b)
        skymodel = SkyModel(spectral_model=model, name="j1507-lp")

        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)
        ax = plt.subplot()
        ax.yaxis.set_units(u.Unit("MeV-1 cm-2 s-1"))
        kwargs = {"ax": ax, "sed_type": "dnde"}

        alpha_results.append(model.alpha.value)
        alpha_results_err.append(model.alpha.error)
        beta_results.append(model.beta.value)
        beta_results_err.append(model.beta.error)
        reference_results.append(model.reference.value)
        amplitude_results.append(model.amplitude.value)
        ndof = 2

    elif config['fit_parameters']['model'] == 'Broken Power Law':
        x_ray_index = float(Catalog[config['columns']['x_ray_index']][i])
        gamma_ray_index = float(Catalog[config['columns']['gamma_ray_index']][i])

        model = BrokenPowerLawSpectralModel(amplitude=Flux_ref, index1=x_ray_index, index2=gamma_ray_index, ebreak=config['fit_parameters']['break_energy'] * u.Unit(config['columns']['energy_units']), reference=str(energy_band[0]) + config['columns']['energy_units'])

        skymodel = SkyModel(spectral_model=model, name="j1507-bpl")
        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)

        ax = plt.subplot()
        ax.yaxis.set_units(u.Unit("MeV-1 cm-2 s-1"))
        kwargs = {"ax": ax, "sed_type": "dnde"}

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

        model = SmoothBrokenPowerLawSpectralModel(amplitude=Flux_ref, index1=x_ray_index, index2=gamma_ray_index, ebreak=config['fit_parameters']['break_energy'] * u.Unit(config['columns']['energy_units']), beta = config['fit_parameters']['curvature'],reference=str(energy_band[0]) + config['columns']['energy_units'])
        skymodel = SkyModel(spectral_model=model, name="j1507-sbpl")
        dataset_MeV.models = skymodel
        fitter = Fit()
        result_model = fitter.run(datasets=dataset_MeV)
        print(energy_band[0])
        ax = plt.subplot()
        ax.yaxis.set_units(u.Unit("MeV-1 cm-2 s-1"))
        kwargs = {"ax": ax, "sed_type": "dnde"}

        x_ray_index_results.append(model.index1.value)
        x_ray_index_err_results.append(model.index1.error)
        gamma_ray_index_results.append(model.index2.value)
        gamma_ray_index_err_results.append(model.index2.error)
        ebreak_results.append(model.ebreak.value)
        ebreak_results_err.append(model.ebreak.error)
        beta_results.append(model.beta.value)
        print(model.beta.value)
        beta_results_err.append(model.beta.error)
        reference_results.append(model.reference.value)
        amplitude_results.append(model.amplitude.value)
        
        ndof=3

    Modeled_dnde = []
    for x in energy_band:
        dnde = (model(x * u.Unit(config['columns']['energy_units'])))
        Modeled_dnde.append(dnde.value)

    chi2_vals = []

    for z in range(len(Modeled_dnde)):
        if flux_band[z] > Modeled_dnde[z]:
            point_err = flux_band_err[0][z]
            if point_err == 0:
                point_err = flux_band_err[1][z]
    
        elif flux_band[z] < Modeled_dnde[z]:
            point_err = flux_band_err[1][z]
            if point_err == 0:
                point_err = flux_band_err[0][z]

        if flux_band[z] > 0:
            chi2_value = ((flux_band[z] - Modeled_dnde[z]) ** 2) / (point_err**2)
            #print(chi2_value)
            #print(chi2_value)
            chi2_vals.append(chi2_value)

    chi2_score = np.sum(chi2_vals) / (len(flux_band) - ndof)
    chi2_results.append(chi2_score)
    #print(chi2_score)
    
    r2_stat = r2_score(flux_band,Modeled_dnde)
    #print(r2_stat)
    r2_results.append(r2_stat)


    energy_bounds = [1e-2, 0.5e6] * u.Unit(config['columns']['energy_units'])
    model.plot(energy_bounds=energy_bounds, color="k", **kwargs)
    model.plot_error(energy_bounds=energy_bounds, **kwargs)

    # Compute total flux across 0.1 MeV to 100 MeV

    integral_energy_flux_err = model.energy_flux_error(energy_min=0.1 * u.Unit(config['columns']['energy_units']), energy_max=100 * u.Unit(config['columns']['energy_units']), epsilon=0.0001)
    integral_energy_flux = integral_energy_flux_err[0]
    integral_energy_flux_error = integral_energy_flux_err[1]
    integral_energy_flux_results.append(integral_energy_flux)
    integral_energy_flux_err_results.append(integral_energy_flux_error)
    
    #print(Redshifts[i])
    if Redshifts[i] != 'None':
        integral_energy_flux_err_rest = model.energy_flux_error(energy_min=0.1 * (1 + float(Redshifts[i])) * u.Unit(config['columns']['energy_units']), energy_max=100 * (1 + float(Redshifts[i])) * u.Unit(config['columns']['energy_units']), epsilon=0.0001)
        integral_energy_flux_rest = integral_energy_flux_err_rest[0]
        k_correction = integral_energy_flux_rest / integral_energy_flux
        #print(k_correction)
        k_corrections_all.append(k_correction.value)
        
    else:
        k_corrections_all.append('None')
    
    CGrB_Bins = [(0.1, 1.0), (0.2,5.0), (1,10), (10,100)]
    
    CGrB_Flux_Bins = []
    
    for b in range(len(CGrB_Bins)):
        integral_energy_flux_err = model.energy_flux_error(energy_min=CGrB_Bins[b][0] * u.Unit(config['columns']['energy_units']), energy_max=CGrB_Bins[b][1] * u.Unit(config['columns']['energy_units']), epsilon=0.0001)
        integral_energy_flux = integral_energy_flux_err[0]
        CGrB_Flux_Bins.append(integral_energy_flux.value)
        
    CGrB_Flux_Bins_all.append(CGrB_Flux_Bins)

    #z_max_arrays = []
    
    GRAMS_Flux_Bins = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    GRAMS_Int_Flux_Bins = []
    
    for b in range(len(GRAMS_Flux_Bins)-1):
        integral_energy_flux_err = model.energy_flux_error(energy_min=GRAMS_Flux_Bins[b] * u.Unit(config['columns']['energy_units']), energy_max=GRAMS_Flux_Bins[b+1] * u.Unit(config['columns']['energy_units']), epsilon=0.0001)
        integral_energy_flux = integral_energy_flux_err[0]
        GRAMS_Int_Flux_Bins.append(integral_energy_flux.value)
    
            
    GRAMS_Flux_Bins_all.append(GRAMS_Int_Flux_Bins)
    
    GRAMS_Red_Flux_Bins = [0.2, 0.5, 1, 5, 10, 50, 100]
    GRAMS_Red_Int_Flux_Bins = []
    
    for b in range(len(GRAMS_Red_Flux_Bins)-1):
        integral_energy_flux_err = model.energy_flux_error(energy_min=GRAMS_Red_Flux_Bins[b] * u.Unit(config['columns']['energy_units']), energy_max=GRAMS_Red_Flux_Bins[b+1] * u.Unit(config['columns']['energy_units']), epsilon=0.0001)
        integral_energy_flux = integral_energy_flux_err[0]
        GRAMS_Red_Int_Flux_Bins.append(integral_energy_flux.value)
    

    GRAMS_Red_Flux_Bins_all.append(GRAMS_Red_Int_Flux_Bins)
        
    #z_max_array_results.append(z_max_arrays)
    
    integral_photon_flux_err = model.integral_error(energy_min=0.1 * u.Unit(config['columns']['energy_units']), energy_max=100 * u.Unit(config['columns']['energy_units']), epsilon = 0.0001)
    integral_photon_flux = integral_photon_flux_err[0]
    integral_photon_flux_error = integral_photon_flux_err[1]
    integral_photon_flux_err_results.append(integral_photon_flux_error)
    integral_photon_flux_results.append(integral_photon_flux)
    
    print(source_names[i])

    if config['Fit_modeled_data'] == True:
        plt.errorbar(energy, flux_band, yerr=[flux_band_err[0], flux_band_err[1]], fmt='o',markersize=3,color='purple',label='Modeled Data')

    else:
        plt.errorbar(energy[:8], flux_band[:8], yerr=[flux_band_err[0][:8], flux_band_err[1][:8]], fmt='o',markersize=3,color='red',label='Swift')
        plt.errorbar(energy[8:], flux_band[8:], yerr=[flux_band_err[0][-8:], flux_band_err[1][-8:]], fmt='o',markersize=3,color='blue',label='Fermi')
    
    plt.title(f"{source_names[i]} ({source_class[i]})")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Energy ({config['columns']['energy_units']})")
    plt.ylabel(f"dN/dE ({config['columns']['diff_flux_units']})")
    plt.legend(loc = 'upper right')
    plt.savefig(plot_out_dir + source_names[i] + '.png', bbox_inches='tight')

    plt.close()    

if config['fit_parameters']['model'] == 'Power Law':  
    Output = Table([source_names, source_class, index_results, index_results_err, integral_energy_flux_results,integral_energy_flux_err_results, integral_photon_flux_results, integral_photon_flux_err_results, chi2_results, r2_results], names=['Name', 'Class', 'Index', 'Index_Err', 'Energy_Flux', 'Energy_Flux_Err','Photon_Flux','Photon_Flux_Err','Chi2','r2'])
elif config['fit_parameters']['model'] == 'Log Parabola':
    Output = Table([source_names, source_class, alpha_results, alpha_results_err, beta_results, beta_results_err, reference_results, amplitude_results, integral_energy_flux_results, integral_energy_flux_err_results, integral_photon_flux_results, integral_photon_flux_err_results, CGrB_Flux_Bins_all, GRAMS_Flux_Bins_all, GRAMS_Red_Flux_Bins_all, chi2_results,r2_results], names=['Name', 'Class', 'Alpha', 'Alpha_Err', 'Beta', 'Beta_Err','Reference','Amplitude', 'Energy_Flux', 'Energy_Flux_Err','Photon_Flux','Photon_Flux_Err','CGrB_Flux_Bins','GRAMS_Int_Flux_Bins','GRAMS_Red_Int_Flux_Bins', 'Chi2','r2'])
elif config['fit_parameters']['model'] == 'Broken Power Law':
    Output = Table([source_names, source_class, x_ray_index_results, x_ray_index_err_results, gamma_ray_index_results, gamma_ray_index_err_results, ebreak_results,ebreak_results_err, integral_energy_flux_results,integral_energy_flux_err_results, integral_photon_flux_results, integral_photon_flux_err_results, chi2_results, r2_results], names=['Name', 'Class', 'X_Ray_Index', 'X_Ray_Index_Err','Gamma_Ray_Index','Gamma_Ray_Index_Err','Break_Energy','Break_Energy_Err','Energy_Flux', 'Energy_Flux_Err','Photon_Flux','Photon_Flux_Err','Chi2','r2'])
elif config['fit_parameters']['model'] == 'Smooth Broken Power Law':
    Output = Table([source_names, source_class, x_ray_index_results, x_ray_index_err_results, gamma_ray_index_results, gamma_ray_index_err_results, ebreak_results,ebreak_results_err, beta_results, beta_results_err, reference_results, amplitude_results, integral_energy_flux_results,integral_energy_flux_err_results, integral_photon_flux_results, integral_photon_flux_err_results, CGrB_Flux_Bins_all, GRAMS_Flux_Bins_all, GRAMS_Red_Flux_Bins_all, k_corrections_all, chi2_results,r2_results], names=['Name', 'Class', 'X_Ray_Index', 'X_Ray_Index_Err','Gamma_Ray_Index','Gamma_Ray_Index_Err','Break_Energy','Break_Energy_Err','Beta','Beta_Err','Reference','Amplitude','Energy_Flux', 'Energy_Flux_Err','Photon_Flux','Photon_Flux_Err','CGrB_Flux_Bins','GRAMS_Int_Flux_Bins', 'GRAMS_Red_Int_Flux_Bins', 'K_corrections','Chi2','r2'])
Output.write(config['Catalog_dir'] + config['out_file_name'], overwrite=True)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")