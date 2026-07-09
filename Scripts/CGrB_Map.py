import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn
import matplotlib.colors as mcolors
import argparse
import yaml
import os

from astropy.coordinates import SkyCoord
from astropy.io import fits


parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
outdir = config['Out_dir']
os.makedirs(outdir, exist_ok=True)
Catalog = fits.open(config['Catalog_dir'] + config['Catalog_Name'])
    
data = Catalog[1].data
print(data.columns[0])
header = Catalog[1].header

flux = data['T']
flux_map = flux.flatten()

nside = header['NSIDE']
npix = hp.nside2npix(nside)
print(f"NSIDE = {nside}, NPIX = {npix}, len(flux_map) = {len(flux_map)}")

assert len(flux_map) == npix, "Flux length doesn't match NSIDE resolution!"

hp.mollview(flux_map,coord=['G'], unit='erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$',cmap='inferno', title='CGrB Map' + ' ' + '(' + config['E_min'] + '-' + config['E_max'] + 'MeV' + ')',min=1e-13,max=1e-10,norm='log' )
hp.graticule()

plt.savefig(os.path.join(outdir, f"{config['Out_name']}({config['E_min']}-{config['E_max']}).png"), dpi=300, bbox_inches='tight')