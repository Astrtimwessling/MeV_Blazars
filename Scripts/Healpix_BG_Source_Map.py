import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as mcolors
import argparse
import yaml
import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
outdir = config['Out_dir']
os.makedirs(outdir, exist_ok=True)

BG_Catalog = fits.open(config['Catalog_dir'] + config['BG_Catalog'])
Source_Catalog = Table.read(config['Catalog_dir'] + config['Source_Catalog'])

BG_data = BG_Catalog[1].data
BG_Flux = BG_data[config['BG_Columns']['BG_Flux']]

BG_Flux_Map = BG_Flux.flatten()

RA = np.array(Source_Catalog[config['Source_Columns']['RA']].tolist())
DEC = np.array(Source_Catalog[config['Source_Columns']['DEC']].tolist())
Source_Flux = np.array(Source_Catalog[config['Source_Columns']['Source_Flux']].tolist())

coords = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame='icrs')
l = coords.galactic.l.deg
b = coords.galactic.b.deg

nside = 64

theta = np.radians(90.0 - b)  # colatitude
phi = np.radians(l % 360.0)   # ensure 0 ≤ φ < 2π
ipix_src = hp.ang2pix(nside, theta, phi)

combined_map = np.copy(BG_Flux_Map)
for pix, flux in zip(ipix_src, Source_Flux):
    combined_map[pix] += flux
    
    
deconvolved_map = hp.smoothing(combined_map, fwhm=np.radians(5.0))
hp.mollview(
    deconvolved_map,
    unit='erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$',
    cmap='inferno',
    title= f"{config['BG_Data']} BG + Point Source ({config['E_min']}-{config['E_max']} MeV)")

hp.graticule(dpar=15, dmer=30, color='white', alpha=0.4)

for lon in np.arange(-90, 90, 15):
    hp.projtext(
        np.radians(lon - 90),        # longitude in radians
        np.radians(180),    # slightly above south pole
        f"{lon}°",
        color="white",
        fontsize=7,
        ha="right",
        va="bottom"
    )

for lat in np.arange(0, 360, 30):
    if lat != 0:
        hp.projtext(
            np.radians(90), np.radians(lat),
            f"{lat}°", color="white", fontsize=7,
            ha="left", va="top"
        )


# Axis labels
plt.text(0, -np.radians(59), "Galactic Longitude (°)",
         color='black', fontsize=11, ha='center', va='top', transform=plt.gca().transData)
plt.text(np.radians(-120), 0, "Galactic Latitude (°)",
         color='black', fontsize=11, ha='center', va='center',
         rotation=90, transform=plt.gca().transData)

plt.savefig(os.path.join(outdir, f"{config['Out_name']}_{config['BG_Data']}_({config['E_min']}-{config['E_max']}).png"), dpi=500, bbox_inches='tight')