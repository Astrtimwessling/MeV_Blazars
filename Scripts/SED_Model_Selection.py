from astropy.table import Table,vstack
import numpy as np

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Data/"
Blazar_Fitter_Results_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/Blazar_Fitter_out/"

LP_Blazar_Fitting_Results = Table.read(Blazar_Fitter_Results_dir + "Matched_Blazar_Fits_LP/MeV_Blazar_Fits_LP_results.fits")
SBPL_Blazar_Fitting_Results = Table.read(Blazar_Fitter_Results_dir + "Matched_Blazar_Fits_SBPL/MeV_Blazar_Fits_SBPL_results.fits")

MeV_Blazar_Catalog = Table.read(catalog_dir + "MeV_Blazar_Catalog_v2.fits")

Selected_Models = []
Selected_Model_Flux_Estimates = []
Selected_Model_MeV_Energies = []
Selected_Model_Chi2 = []

Swift_Redshift = MeV_Blazar_Catalog['Swift_Redshift'].tolist()
Fermi_Redshift = MeV_Blazar_Catalog['Fermi_Redshift'].tolist()
Redshift = [Swift_Redshift[x] if Swift_Redshift[x] != 'None' else Fermi_Redshift[x] for x in range(len(Swift_Redshift))]

Reference_Energy = [x[0] for x in MeV_Blazar_Catalog['Energy'].tolist()]

LP_Names = []
LP_Class = []
LP_SED_Class = []
LP_Model = []
LP_Redshift = []
LP_Reference_Energy = []
LP_Reference_Flux = []
LP_Alpha = []
LP_Beta = []

LP_chi2 = []
LP_opposite_chi2 = []

SBPL_Names = []
SBPL_Class = []
SBPL_SED_Class = []
SBPL_Model = []
SBPL_Redshift = []
SBPL_Reference_Energy = []
SBPL_Reference_Flux = []
SBPL_Index_1 = []
SBPL_Index_2 = []
SBPL_Break_Energy = []
SBPL_Beta = []

SBPL_chi2 = []
SBPL_opposite_chi2 = []

for i in range(len(MeV_Blazar_Catalog)):
    blazar_class = MeV_Blazar_Catalog['Fermi_Type'][i]
    blazar_SED_class = MeV_Blazar_Catalog['SED_Class'][i]
    blazar_name = MeV_Blazar_Catalog['Fermi_Counterpart_Name'][i]
    Selected_Model_MeV_Energies.append([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    
    if (blazar_class.upper() == 'FSRQ') or (blazar_class.upper() == 'BLL' and blazar_SED_class == 'LSP') or (blazar_class.upper() == 'BCU' and blazar_SED_class == 'LSP') or (blazar_name == 'PMN J1310-5552') or (blazar_name == 'S5 0716+71'):
        Model = 'LP' 
        Estimated_Flux = [x * 1.602176634e-6 for x in LP_Blazar_Fitting_Results['Flux_MeV_GRAMS'][i].tolist()]
        Chi2 = LP_Blazar_Fitting_Results['Chi2'][i]
        Opposite_Chi2 = SBPL_Blazar_Fitting_Results['Chi2'][i]
        
        Selected_Models.append(Model)
        Selected_Model_Flux_Estimates.append(Estimated_Flux)
        Selected_Model_Chi2.append(str(Chi2))
        LP_chi2.append(Chi2)
        SBPL_opposite_chi2.append(Opposite_Chi2)
        
        if Redshift[i] != 'None':
            LP_Names.append(blazar_name)
            LP_Class.append(blazar_class)
            LP_SED_Class.append(blazar_SED_class)
            LP_Model.append(Model)
            LP_Redshift.append(Redshift[i])
            LP_Reference_Energy.append(Reference_Energy[i])
            LP_Reference_Flux.append(LP_Blazar_Fitting_Results['N0'][i]/(Reference_Energy[i]**2))
            LP_Alpha.append(LP_Blazar_Fitting_Results['Alpha'][i])
            LP_Beta.append(LP_Blazar_Fitting_Results['Beta'][i])
        
    elif (blazar_class.upper() == 'BLL' and blazar_SED_class == 'HSP') or (blazar_class.upper() == 'BCU' and blazar_SED_class == 'HSP' and blazar_name != 'PMN J1310-5552') or (blazar_class.upper() == 'BCU' and blazar_SED_class == 'None') or (blazar_name == 'RXC J0934.4-1721'):
        Model = 'SBPL' 
        Estimated_Flux = [x * 1.602176634e-6 for x in SBPL_Blazar_Fitting_Results['Flux_MeV_GRAMS'][i].tolist()]
        Chi2 = SBPL_Blazar_Fitting_Results['Chi2'][i]
        Opposite_Chi2 = LP_Blazar_Fitting_Results['Chi2'][i]
        
        Selected_Models.append(Model)
        Selected_Model_Flux_Estimates.append(Estimated_Flux)
        Selected_Model_Chi2.append(str(Chi2))
        SBPL_chi2.append(Chi2)
        LP_opposite_chi2.append(Opposite_Chi2)
        
        if Redshift[i] != 'None':
            SBPL_Names.append(blazar_name)
            SBPL_Class.append(blazar_class)
            SBPL_SED_Class.append(blazar_SED_class)
            SBPL_Model.append(Model)
            SBPL_Redshift.append(Redshift[i])
            SBPL_Reference_Energy.append(Reference_Energy[i])
            SBPL_Reference_Flux.append(SBPL_Blazar_Fitting_Results['N0'][i]/(Reference_Energy[i]**2))
            SBPL_Index_1.append(SBPL_Blazar_Fitting_Results['X_Ray_Index'][i])
            SBPL_Index_2.append(SBPL_Blazar_Fitting_Results['Gamma_Ray_Index'][i])
            SBPL_Break_Energy.append(SBPL_Blazar_Fitting_Results['Break_Energy'][i])
            SBPL_Beta.append(1)
            
    else:
        Selected_Models.append('None')
        Selected_Model_Flux_Estimates.append([])
        Selected_Model_Chi2.append('None')
        print(LP_Blazar_Fitting_Results['Chi2'][i], SBPL_Blazar_Fitting_Results['Chi2'][i])
        
print(np.mean(LP_chi2))
print(np.mean(SBPL_chi2))

print(np.mean(LP_opposite_chi2))
print(np.mean(SBPL_opposite_chi2))
        
MeV_Blazar_Catalog['MeV_Model'] = Selected_Models
MeV_Blazar_Catalog['MeV_Energies'] = Selected_Model_MeV_Energies
MeV_Blazar_Catalog['MeV_Flux_Estimates'] = Selected_Model_Flux_Estimates
MeV_Blazar_Catalog['MeV_Chi2'] = Selected_Model_Chi2

MeV_Blazar_Catalog.write(catalog_dir + "MeV_Blazar_Catalog_v2_SED_Fits.fits", overwrite=True)

SBPL_Model_Params_Table = Table([SBPL_Names, SBPL_Class, SBPL_SED_Class, SBPL_Model, SBPL_Redshift, SBPL_Reference_Energy, SBPL_Reference_Flux, SBPL_Index_1, SBPL_Index_2, SBPL_Break_Energy, SBPL_Beta],names=['Name','Class','SED_Class','Model','Redshift','E0','N0','SBPL_Index1','SBPL_Index2','SBPL_Break_Energy','SBPL_Beta'])
LP_Model_Params_Table = Table([LP_Names, LP_Class, LP_SED_Class, LP_Model, LP_Redshift, LP_Reference_Energy, LP_Reference_Flux, LP_Alpha, LP_Beta], names=['Name','Class','SED_Class','Model','Redshift','E0','N0','LP_Alpha','LP_Beta'])

Model_Params_Table = vstack([LP_Model_Params_Table, SBPL_Model_Params_Table])
Model_Params_Table.write(catalog_dir + 'SED_Selected_Model_Params.fits', overwrite=True)