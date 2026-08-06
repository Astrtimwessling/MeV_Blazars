from astropy.table import Table
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from matplotlib.lines import Line2D

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

catalog_dir = "/home/alab_student/Tim/Projects/MeV_Blazars/Data/"
outdir = '/home/alab_student/Tim/Projects/MeV_Blazars/Outputs/'

# Import flux estimates from SED modeling of cross matched sources.

MeV_Blazar_Catalog_SED_Fits = Table.read(catalog_dir + "MeV_Blazar_Catalog_v2_SED_Fits.fits")
Mask = (MeV_Blazar_Catalog_SED_Fits['MeV_Model'] != 'None')
MeV_Blazar_Catalog_SED_Fits = MeV_Blazar_Catalog_SED_Fits[Mask]

MeV_Estimated_Fluxes = [list(x) for x in MeV_Blazar_Catalog_SED_Fits['MeV_Flux_Estimates'].tolist()]
Blazar_Classes = MeV_Blazar_Catalog_SED_Fits['Fermi_Type'].tolist()
MeV_Flux_Estimates_By_Band = list(zip(*MeV_Estimated_Fluxes))

MeV_Energies = MeV_Blazar_Catalog_SED_Fits['MeV_Energies'].tolist()[0]

FSRQ_Flux_Estimates = list(zip(*[MeV_Estimated_Fluxes[i] for i in range(len(MeV_Estimated_Fluxes)) if Blazar_Classes[i].upper() == 'FSRQ']))
BLL_Flux_Estimates = list(zip(*[MeV_Estimated_Fluxes[i] for i in range(len(MeV_Estimated_Fluxes)) if Blazar_Classes[i].upper() == 'BLL']))
BCU_Flux_Estimates = list(zip(*[MeV_Estimated_Fluxes[i] for i in range(len(MeV_Estimated_Fluxes)) if Blazar_Classes[i].upper() == 'BCU']))

# Import RA and DEC coordinates for each cross-matched blazar.

Blazar_Coords = list(zip(MeV_Blazar_Catalog_SED_Fits['Swift_RA'].tolist(), MeV_Blazar_Catalog_SED_Fits['Swift_DEC'].tolist()))

FSRQ_Coords = [Blazar_Coords[x] for x in range(len(Blazar_Coords)) if Blazar_Classes[x].upper() == 'FSRQ']
BLL_Coords = [Blazar_Coords[x] for x in range(len(Blazar_Coords)) if Blazar_Classes[x].upper() == 'BLL']
BCU_Coords = [Blazar_Coords[x] for x in range(len(Blazar_Coords)) if Blazar_Classes[x].upper() == 'BCU']

RA_Sources = [[x[0] for x in FSRQ_Coords], [x[0] for x in BLL_Coords], [x[0] for x in BCU_Coords]]
DEC_Sources = [[x[1] for x in FSRQ_Coords], [x[1] for x in BLL_Coords], [x[1] for x in BCU_Coords]]
Flux_Estimates = [FSRQ_Flux_Estimates, BLL_Flux_Estimates, BCU_Flux_Estimates]

# Make one 3x3 grid with the extragalactic sky at all 9 selected energies.

fig, axes = plt.subplots(3, 3, figsize=(18, 14), subplot_kw={'projection': 'mollweide'}, facecolor='white')
axes = axes.flatten()
norm = LogNorm(vmin=1e-12, vmax=0.3e-9)

Blazar_Label = ['FSRQ', 'BLL', 'BCU']
Blazar_Marker = ['o', 's', '^']
legend_elements = [Line2D([0], [0], marker=Blazar_Marker[i], color='w', markerfacecolor='black', markersize=6, label=Blazar_Label[i]) for i in range(len(Blazar_Label))]

for i, ax in enumerate(axes):
    if i >= len(MeV_Flux_Estimates_By_Band):
        ax.set_visible(False)
        continue

    ax.set_facecolor('white')

    ax.plot(np.zeros(100), np.linspace(-np.pi/2, np.pi/2, 100), color='black', linewidth=0.5)
    ax.plot(np.linspace(-np.pi, np.pi, 500), np.zeros(500), color='black',linewidth=0.5)

    for x in range(len(RA_Sources)):

        RA = np.array(RA_Sources[x])
        DEC = np.array(DEC_Sources[x])
        Flux = np.array(Flux_Estimates[x][i])

        coords_icrs = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame='icrs')
        coords_gal = coords_icrs.galactic
        l = coords_gal.l.wrap_at(180*u.deg).radian
        l = -l
        b = coords_gal.b.radian
        sc = ax.scatter(l, b, c=Flux, cmap='inferno', norm=norm, s=35, marker=Blazar_Marker[x], edgecolors='none')

    for b_deg in [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]:
        b = np.deg2rad(b_deg)
        l_line = np.linspace(-np.pi, np.pi, 500)
        b_line = np.full_like(l_line, b)
        ax.plot(l_line, b_line, color='black', alpha=0.3, linewidth=1, linestyle=':')

    for l_deg in [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150]:

        l = -np.deg2rad(l_deg)
        b_line = np.linspace(-np.pi/2, np.pi/2, 500)
        l_line = np.full_like(b_line, l)
        ax.plot(l_line, b_line, color='black', alpha=0.3, linewidth=1, linestyle=':')

    b_deg_list = [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75]

    for b_deg in b_deg_list:
        ax.text(0.0, np.deg2rad(b_deg) + 0.05, f"{b_deg}°", color='black', ha='right', va='center', fontsize=7)

    l_deg_list = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]

    for l_deg in l_deg_list:
        ax.text(-np.deg2rad(l_deg) - 0.1, 0.0, f"{l_deg}°", color='black', ha='center', va='bottom', fontsize=7)

    ax.set_title(f"{MeV_Energies[i]} MeV", color='black', fontsize=11, pad=12)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.subplots_adjust(hspace=0.18,wspace=0.08, bottom=0.17)

cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', shrink=0.5, pad=0.03)
cbar.set_label(r"Energy Flux [erg cm$^{-2}$ s$^{-1}$]", fontsize=12)

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.51, 0.18), ncol=3, fontsize=11, frameon=False, handletextpad=0.01, columnspacing=1.5)

plt.savefig(outdir + 'Extragalactic_Skymaps/Extragalactic_Skymap_Grid.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Make a large, individual extragalactic skymap for each selected energy.

for i in range(len(MeV_Flux_Estimates_By_Band)):
    fig = plt.figure(figsize=(10,6), facecolor='white')
    ax = fig.add_subplot(111, projection="mollweide", facecolor='white')
    
    ax.plot(np.zeros(100), np.linspace(-np.pi/2, np.pi/2, 100),
        color='black', linewidth=0.5)
    
    ax.plot(np.linspace(-np.pi, np.pi, 500), np.zeros(500),
        color='black', linewidth=0.5)
    
    for x in range(len(RA_Sources)):
        RA = np.array(RA_Sources[x])
        DEC = np.array(DEC_Sources[x])
        Flux = np.array(Flux_Estimates[x][i])

        coords_icrs = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame='icrs')
        coords_gal = coords_icrs.galactic

        l = coords_gal.l.wrap_at(180*u.deg).radian
        l=-l
        b = coords_gal.b.radian
        norm = LogNorm(vmin=1e-12, vmax=0.3e-9)
        sc = ax.scatter(l, b, c=Flux, cmap='inferno', norm=norm, s=20, marker = Blazar_Marker[x], label=Blazar_Label[x])
    
    for b_deg in [-90, -75, -60, -45, -30,-15, 0,15, 30,45, 60,75,90]:  
        b = np.deg2rad(b_deg)
        l_line = np.linspace(-np.pi, np.pi, 500)
        b_line = np.full_like(l_line, b)
        ax.plot(l_line, b_line, color='black', alpha=0.4, linewidth=1, linestyle=':')
        
    b_deg_list = [-75, -60, -45, -30,-15, 15, 30,45, 60,75]   
    b_rad_list = np.deg2rad(b_deg_list)

    for b_rad, b_deg in zip(b_rad_list, b_deg_list):
        ax.text(0.0, b_rad+0.07, f"{b_deg}°", color='black',
        ha='right', va='center', fontsize=10, zorder=20)
        
    for l_deg in [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150]:
        l = -np.deg2rad(l_deg) 
        b_line = np.linspace(-np.pi/2, np.pi/2, 500)
        l_line = np.full_like(b_line, l)
        ax.plot(l_line, b_line, color='black', alpha=0.4, linewidth=1,linestyle=':')
        
    l_deg_list = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
    l_rad_list = -np.deg2rad(l_deg_list)  
    for l_rad, l_deg in zip(l_rad_list, l_deg_list):
            ax.text(l_rad - 0.07, 0.0, f"{l_deg}°", color='black',
            ha='center', va='bottom', fontsize=10, zorder=20)
    
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.07, shrink=0.4)
    
    cbar.set_label(f"Energy Flux ({MeV_Energies[i]} MeV) " r"[erg cm$^{-2}$ s$^{-1}$]")
    
    ax.set_xlabel(r"$l$ (°)")
    ax.set_ylabel(r"$b$ (°)")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(outdir + 'Extragalactic_Skymaps/' + f'Blazar_E2Point_Skymap_{MeV_Energies[i]} MeV.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()