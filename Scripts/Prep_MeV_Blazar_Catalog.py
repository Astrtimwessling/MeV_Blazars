import numpy as np

from astropy.table import Table

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Data/"

# Import Catalogs

Cross_Match_Catalog = Table.read(catalog_dir + 'cross_match_updated.fits')
Swift_157m_Catalog = Table.read(catalog_dir + "Swift_BAT_157m_Catalog_FITS.fits")
Fermi_LAT_4FGL_DR4 = Table.read(catalog_dir + '4FGLDR4.fit', hdu=1)
Fermi_LAC_DR3 = Table.read(catalog_dir + "Fermi_LAC_DR3.fits")

print(Cross_Match_Catalog.colnames)

# Apply Firm Blazar Mask to Cross-Matched Catalog

Firm_Blazar_Mask = [x in ['BCU', 'BLL','FSRQ','fsrq','bll','bcu'] and y in ['BZQ','BZB','BZG','BZU', 'Beamed AGN', 'FSRQ', 'BZQ/Lense'] for x,y in zip(Cross_Match_Catalog['fermi_category_type'].tolist(), Cross_Match_Catalog['bat_category_type'].tolist())]

Firmly_Cross_Matched_Blazars = Cross_Match_Catalog[Firm_Blazar_Mask]

# Extrapolate Extra Fermi-LAT Blazar Data

Fermi_Names = Firmly_Cross_Matched_Blazars['fermi_name_counterpart'].tolist()

fermi_redshifts = []
fermi_SED_classes = []
syn_peak_energies = []
syn_peak_flux = []
flux_peak_ratio = []

for i in Fermi_Names:
    
    # Extrapolate extra information from 4FGL-DR4
    
    LAT_catalog_mask = (i.ljust(28) == Fermi_LAT_4FGL_DR4['ASSOC1'])
    LAT_catalog_entry = Fermi_LAT_4FGL_DR4[LAT_catalog_mask]
    
    Flux_Ratio = (max(LAT_catalog_entry['Flux_History'].tolist()[0]) / np.mean(LAT_catalog_entry['Flux_History'].tolist()))
    flux_peak_ratio.append(Flux_Ratio)
    
    # Extrapolate information from 4FGL-DR3

    AGN_catalog_mask = (i == Fermi_LAC_DR3['ASSOC1'])
    AGN_catalog_entry = Fermi_LAC_DR3[AGN_catalog_mask]

    fermi_z = AGN_catalog_entry['Redshift'].tolist()
    if fermi_z == [] or fermi_z == [-np.inf]:
        fermi_z = ['None']
        fermi_redshifts.append(fermi_z[0])
    else:
        fermi_redshifts.append(fermi_z[0])

    fermi_SED_class = AGN_catalog_entry['SED_class'].tolist()
    if fermi_SED_class == [] or fermi_SED_class == ['']:
        fermi_SED_class = ['None'] 
        fermi_SED_classes.append(fermi_SED_class[0])
    else:
        fermi_SED_classes.append(fermi_SED_class[0])

    nu_syn = AGN_catalog_entry['nu_syn'].tolist() # Hz
    if nu_syn == [] or nu_syn == [0.0]:
        nu_syn = ['None']
        syn_peak_energies.append(nu_syn[0])
    else:
        nu_syn = [nu_syn[0] * 4.135667696e-21] # MeV
        syn_peak_energies.append(nu_syn[0])

    nuFnu_syn = AGN_catalog_entry['nuFnu_syn'].tolist() # erg / (cm^2 / s)
    if nuFnu_syn == [] or nuFnu_syn == [0.0]:
        nuFnu_syn = ['None']
        syn_peak_flux.append(nuFnu_syn[0])
    else:
        nuFnu_syn = [nuFnu_syn[0] * 624.151]
        syn_peak_flux.append(nuFnu_syn[0])
        
# Extrapolate extra Swift-BAT Blazar Data
        
Swift_names = Firmly_Cross_Matched_Blazars['bat_name_counterpart'].tolist()

Swift_Luminosity = []
Swift_Redshift = []

for i in Swift_names:

    Swift_catalog_mask = (i == Swift_157m_Catalog['COUNTERPART_NAME'])
    Swift_catalog_entry = Swift_157m_Catalog[Swift_catalog_mask]

    swift_lum = Swift_catalog_entry['LUM'].tolist()
    if swift_lum == ['NA'] or swift_lum == []:
        swift_lum = ['None']
        Swift_Luminosity.append(swift_lum[0])
    else:
        swift_lum = [float(swift_lum[0])]
        Swift_Luminosity.append(swift_lum[0])

    swift_z = Swift_catalog_entry['REDSHIFT'].tolist()
    if swift_z == ['NA'] or swift_z == []:
        swift_z = ['None']
        Swift_Redshift.append(swift_z[0])
    else:
        swift_z = [float(swift_z[0])]
        Swift_Redshift.append(swift_z[0])
        
# Import Selected Columns from Cross-Matched Catalog
        
Imported_Colnames = ['flag', 'fermi_category_type', 'bat_category_type', 'bat_pindex', 'bat_pindex_errm', 'bat_pindex_errp', 'fermi_pindex_PL', 'fermi_pindex_PL_err','bat_flux', 'bat_flux_errm', 'bat_flux_errp', 'fermi_flux', 'fermi_flux_err', 'Energy', 'Energy_Flux', 'Energy_Flux_Err', 'fermi_ra', 'fermi_dec', 'bat_ra', 'bat_dec']
Imported_Columns = [Firmly_Cross_Matched_Blazars[x].tolist() for x in Imported_Colnames]

Fermi_pindex_err = ['None' if i is None else i for i in Imported_Columns[7]]
Swift_pIndex_err = list(zip(Imported_Columns[4], Imported_Columns[5]))
Swift_Flux_err = list(zip(Imported_Columns[9], Imported_Columns[10]))

catalog_id = list(range(1, len(Firmly_Cross_Matched_Blazars)+1))

# Assemble!

MeV_Blazar_Catalog = Table([catalog_id, Imported_Columns[0], Fermi_Names, Swift_names, Imported_Columns[1], Imported_Columns[2], fermi_SED_classes, Imported_Columns[16], Imported_Columns[17], Imported_Columns[18], Imported_Columns[19], fermi_redshifts, Swift_Redshift, Imported_Columns[6], Fermi_pindex_err,  Imported_Columns[3], Swift_pIndex_err,  Imported_Columns[11], Imported_Columns[12], Imported_Columns[8], Swift_Flux_err, Swift_Luminosity, flux_peak_ratio, syn_peak_energies, syn_peak_flux, Imported_Columns[13], Imported_Columns[14], Imported_Columns[15]], names = ['id','Match_Flag', 'Fermi_Counterpart_Name', 'Swift_Counterpart_Name', 'Fermi_Type', 'Swift_Type', 'SED_Class', 'Fermi_RA', 'Fermi_DEC','Swift_RA', 'Swift_DEC', 'Fermi_Redshift', 'Swift_Redshift', 'Fermi_Photon_Index', 'Fermi_Photon_Index_Err', 'Swift_Photon_Index', 'Swift_Photon_Index_Err', 'Fermi_Flux', 'Fermi_Flux_err', 'Swift_Flux', 'Swift_Flux_err', 'Log_Swift_Luminosity', 'Flux_Ratio', 'Syn_Peak_Energy', 'Syn_Peak_Flux', 'Energy', 'Energy_Flux', 'Energy_Flux_Err'])

MeV_Blazar_Catalog['Fermi_RA'].unit = 'deg'
MeV_Blazar_Catalog['Fermi_DEC'].unit = 'deg'
MeV_Blazar_Catalog['Swift_RA'].unit = 'deg'
MeV_Blazar_Catalog['Swift_DEC'].unit = 'deg'
MeV_Blazar_Catalog['Log_Swift_Luminosity'].unit = 'erg s-1'
MeV_Blazar_Catalog['Syn_Peak_Energy'].unit = 'MeV'
MeV_Blazar_Catalog['Syn_Peak_Flux'].unit = 'MeV / (cm^2 s)'
MeV_Blazar_Catalog['Energy'].unit = 'MeV'
MeV_Blazar_Catalog['Energy_Flux'].unit = 'MeV / (cm^2 s)'
MeV_Blazar_Catalog['Energy_Flux_Err'].unit = 'MeV / (cm^2 s)'

MeV_Blazar_Catalog.write(catalog_dir + 'MeV_Blazar_Catalog_v2.fits', overwrite=True)
