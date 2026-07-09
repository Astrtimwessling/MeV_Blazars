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
from astropy.table import Table


parser = argparse.ArgumentParser(description="Run script with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

outdir = config['Out_dir']
os.makedirs(outdir, exist_ok=True)
Catalog = Table.read(config['Catalog_dir'] + config['Catalog_Name'])

RA = np.array(Catalog[config['Columns']['RA']])
DEC = np.array(Catalog[config['Columns']['DEC']].tolist())
Flux = np.array(Catalog[config['Columns']['Flux']].tolist())

nside = 128
npix = hp.nside2npix(nside)

coords = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame='icrs')
l = coords.galactic.l.deg
b = coords.galactic.b.deg

theta = np.radians(90.0 - b)
phi = np.radians(l)

pix_indices = hp.ang2pix(nside, theta, phi)

flux_map = np.zeros(npix)
counts = np.zeros(npix)

for p, f in zip(pix_indices, Flux):
    flux_map[p] += f
    counts[p] += 1

flux_map[counts > 0] /= counts[counts > 0]

flux_map[counts == 0] = hp.UNSEEN

obs_map = flux_map
nside = hp.get_nside(obs_map)
beam_fwhm = np.radians(3.0)

lmax = 3 * nside - 1
alm = hp.map2alm(obs_map, lmax=lmax)

ls = np.arange(lmax+1)
sigma = beam_fwhm / np.sqrt(8 * np.log(2))
B_l = np.exp(-0.5 * ls * (ls + 1) * sigma**2)

B_l_safe = np.where(B_l < 1e-3, 1e-3, B_l)

alm_deconv = hp.almxfl(alm, 1.0 / B_l_safe)

deconv_map = hp.alm2map(alm_deconv, nside)

smoothed_map = hp.smoothing(deconv_map, fwhm=np.radians(5.0))  

stretch = 0.1
norm = mcolors.SymLogNorm(linthresh=stretch, vmin=np.min(smoothed_map), vmax=np.max(smoothed_map))

hp.mollview(smoothed_map,title=config['Map_Energy']+ config['Energy_Units'],unit='MeV cm$^{-2}$ s$^{-1}$', cmap='coolwarm', norm=norm)

plt.savefig(outdir + config['Out_name'] + '_' + config['Map_Energy'] + '_' + config['Energy_Units'] + '.png', dpi=300)