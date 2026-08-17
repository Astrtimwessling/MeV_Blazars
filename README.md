# Observation Forecasting for Blazars by Future MeV $\gamma$-Ray Instruments

This is the repository associated with **Detection Forecasting for Blazars at MeV Energies with Updated Fermi-LAT Swift-BAT Cross Match** (Wessling, T., et al. 2026 - Manuscript in prep.)

<img width="4753" height="6365" alt="N_S_MeV_Blazars_Forecast" src="https://github.com/user-attachments/assets/601281fc-6254-4c2e-a595-23d9debbc5bc" />

With the longstanding gap in the MeV $\gamma$-ray regime, and a new suite of MeV instruments with improved sensitivity and detection capabilities currently propsed, the era of MeV astronomy is right around the corner.
These MeV instruments will have exciting abilities to observe many types of sources at MeV $\gamma$ ray energies. In particular, blazars will be of great interest to MeV instruments, given their prevalence in the $\gamma$ ray sky. Through this work, we motivate the study of blazars at MeV energies by demonstrating the exciting capabilties of next generation MeV $\gamma$-ray instruments. 

## Table of contents
1. [Data](#Data)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Contact](#Contact)
5. [Acknowledgements](#Acknowledgements)
   
## Data/Catalogs

The base data catalogs/files needed to utilize the pipeline can be downloaded from this [google drive](https://drive.google.com/drive/folders/1hJM_sq7k2ErQKk6xug273z-aH1Lh4eJ2). 

The catalogs used to synthesize the firmly matched blazar catalog are the following:

1. Fourth Fermi LAT Source Catalog Data Release 4: ```4FGLDR4.fit``` ([source link](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/))
2. Swift BAT 157 Month Source Catalog: ```Swift_BAT_157m_Catalog_FITS.fits``` ([source link](https://swift.gsfc.nasa.gov/results/bs157mon/))
3. Fourth Fermi-LAT AGN Catalog Data Release: ```Fermi_LAC_DR3.fits``` ([source link](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR3/))
4. Updated Cross-Match Catalog: ```cross_match_updated.fits```

NOTE 1: SED data for Swift-BAT sources was independently produced through XSPEC utilizing available ```.pha``` files. <br>
NOTE 2: Updated cross-match methodology and results will be presented in another repository (WIP)  <br>

The google drive also contains two files containing the sensitivity information for next generation MeV $\gamma$-ray insturments:

1. ```Sensitivity_Curves_Data.csv```: Contains full sensitivity curves for future instruments.
2. ```Sensitivity_Data_Table.fits```: Contains sensitivity values for future instruments at selected energies (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, and 100.0 MeV)

Sensitivity curve information was taken from the following sources:

[GRAMS Balloon and Satellite](https://arxiv.org/pdf/1901.03430) (Figure 7.) <br>
[AMEGO-X](https://arxiv.org/pdf/2208.04990) (Figure 3.) <br>
[newASTROGAM](https://arxiv.org/pdf/2507.08133) (Figure 1.) <br>
[COSI](https://arxiv.org/pdf/2308.12362) (Right Panel, Figure 2.)

## Requirements

To install the specifications required to utilize this code base, run the following command. The versions used for packages installed are listed in ```requirements.txt```.

```
conda create --name [new_env] python=3.10.12
conda activate [new_env]
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

To begin, download all files from the google drive and place them in a folder on the same level as ```Scripts```. In each script that will be run, input the path to the data, as well as the path to which the outputs will go. Run the commands in the following order to execute the full pipeline.

### Runner Scripts

Synthesize the firmly matched MeV blazar catalog. This is the collection of sources used in this work. This script will output the file ```MeV_Blazar_Catalog_v2.fits```.

```
Python3 Prep_MeV_Blazar_Catalog.py
```
Run SED modeling on cross-matched spectra. We use two phenomenological models in this work. Each model has a config file which can be found in ```Scripts/Configs```. Execute each to fit every source with each model. The fitter script will output plots showing the modeled SED and raw SED data by blazar class in individual folders. It will also output a file called ```MeV_Blazar_Fits_LP\SBPL_results.fits```, containing the best fit model parameters for each source, and the $\chi^2$ metric used to evaluate fit quality.

```
Python3 Blazar_Fitter.py --config Configs/Blazar_Fitter_config_LP.yaml
Python3 Blazar_Fitter.py --config Configs/Blazar_Fitter_config_SBPL.yaml
```
The figure below shows an example of a source fit with the log parabola model (left) and a source fit with the smooth broken power law model (right).

<img width="3570" height="1509" alt="Example_SEDS" src="https://github.com/user-attachments/assets/62956445-de9b-4309-a7dc-195c0973f0c4" />

Once all fitting is completed, run the SED selection script to export model choice and flux estimates for each source. This script adds the model information and estimated flux points from the selected model as columns onto ```MeV_Blazar_Catalog_v2.fits``` and saves it as  ```MeV_Blazar_Catalog_v2_SED_Fits.fits```.

```
Python3 SED_Model_Selection.py
```

From the selected SED models, we make luminosity binned averaged SEDs for FSRQs, LP BLLs, and SBPL BLLs. Best fit parameters for the averaged SEDs are saved as ```SED_Template_Params_Table.fits```. The averaged blazar SED output plot based on our individual source SED modeling is shown below.

```
Python3 Template_SEDS.py
```

<img width="8959" height="2654" alt="SED_Templates" src="https://github.com/user-attachments/assets/e8003c7d-5871-4114-a50c-55d757a43fd8" />

We utilize the averaged SEDs to compute the k-corrected luminosity required as the lower bound for cumulative source count, N(>S), calculations, and cumulative redshift count, N(>z), calculations. In this work, we utilize LDDE luminosity functions derived from different source populations in the liturature:

1. Fermi-LAT FSRQs: [Rajguru, G., et al. (2025)](https://arxiv.org/pdf/2510.05515)
2. Fermi-LAT BLLs: [Ajello, M., et al. (2013)](https://arxiv.org/pdf/1310.0006)
3. Swift-BAT FSRQs: [Toda, K., et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-4357/ac937f/pdf)

The luminosity functions and their respective integrand functions are defined in ```/src/LDDE_Luminosity_Functions.py```. Note that for the Fermi-LAT BLL luminosity function, we utilize $LDDE_2$ from Ajello et al. (2013). 


Running the following the command calculates the cumulative source count at a given sensitivity, N(>S), at sensitivities between $10^{-13}$ and $10^{-9}$ $\mathrm{erg/cm^2/s}$. This calculation outputs a file called ```N_S_LF_Output.fits```, where the first column are the sensitivity values, and subsequent columns are the counts for each luminosity function at each selected energy.

```
Python3 N_S_LF_Calculation.py --config Configs/LF_SED_Template_Params.yaml
```
Similarly, running the following command calculates the cumulative redshift counts, N(>z), at redshifts between z = 0 and z = 8. The output file for this script is called ```N_z_LF_Output.fits```, where the first column is the redshift values, and the subsequent columns are the number of sources beyond that redshift, z, as determined by the luminosity function calculation.

```
Python3 N_z_LF_Calculation.py --config Configs/LF_SED_Template_Params.yaml
```

### Plotter Scripts

The following scripts are used to make plots that will be shown in the upcoming Wessling, T., et al. 2026 paper. 


The following script produces skymaps for the cross-matched blazar population, with flux values represented by a colorbar. The script produces both large, individual skymaps for each of the selected energy, and one 3x3 grid of all of the selected energies. 

```
Python3 Cross_Match_Blazar_Skymap.py
```
Here we show an example of a large single energy skymap at 1 MeV, annotated to highlight the brightest sources:

<img width="4600" height="2880" alt="Annotated_EG_Skymap_TNR" src="https://github.com/user-attachments/assets/665ba6ad-d692-4b7b-91f9-11960573d988" />

To create the figure shown at the top of this README, we run the following script:
```
Python3 N_S_Plotter.py
```
The output of this script plots the joint forcast at each energy, including the number of detectable cross-matched sources and cumulative source counts determined from luminosity functions across the $10^{-13}$ and $10^{-9}$ $\mathrm{erg/cm^2/s}$ range of sensitivities. The sensitivities at each of our selected energies are shown as verticle lines on each respective plot. An additional plot is also created containing a plot at 6 of the selected energies. This figure will be shown in the text of the paper.


Similarly, we make plots using the following command for the cumulative redshift estimates and cross-matched redshift detectabilities at each energy for the sensitivity limit of each instrument at that energy. 

```
Python3 N_z_Plotter.py
```
<p align="center">
<img width="800" height="700" alt="Cumulative_N_z_1 0" src="https://github.com/user-attachments/assets/85dbd5a4-f558-417e-a283-47591b9787e5" />
</p>

### Notebooks
This repository also contains two jupyter notebooks that we use for results analysis.


The first of these notebooks has code that does simple population analysis of the cross matched sample.

```
Cross_Matched_Blazars.ipynb
```
<p align="center">
<img width="600" height="450" alt="LAT_BAT_index_comparison" src="https://github.com/user-attachments/assets/bed29168-9e31-48ef-ac90-06c7d07641bc" />
</p>

The other notebook is used analysis of the forecasting results, determining the estimates of detectable counts for future MeV $\gamma$-ray instruments.

```
MeV_Blazar_Forecasting.ipynb
```

## Contact

Please contact me at wessling-resnick.t@northeastern.edu for questions or comments.

## Acknowledgements

This work was supported by funding for the Gamma-Ray and AntiMatter Survey collaboration through the NASA APRA grant, No.22-APRA22-0128 (80NSSC23K1661), and the Alfred P. Sloan Foundation in the US, as well as the Japan Society for the Promotion of Science (JSPS) in Japan. In addition, this work was supported by the Northeastern University Undergraduate Research and Fellowships office by funding through the PEAK Ascent award (Summer 2024), PEAK Summit awards (Fall 2024, Spring 2025), and the AJC Merit Scholarship.
