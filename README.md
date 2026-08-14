# Observation Forecasting for Blazars by Future MeV $\gamma$-Ray Instruments

This is the repository associated with **Detection Forecasting for Blazars at MeV Energies with Updated Fermi-LAT Swift-BAT Cross Match** (Wessling, T., et al. 2026 - Manuscript in prep.)

<img width="4753" height="6365" alt="N_S_MeV_Blazars_Forecast" src="https://github.com/user-attachments/assets/601281fc-6254-4c2e-a595-23d9debbc5bc" />

With the longstanding gap in the MeV $\gamma$-ray regime, and a new suite of MeV instruments with improved sensitivity and detection capabilities currently propsed, the era of MeV astronomy is right around the corner.
These MeV instruments will have exciting abilities to observe many types of sources at MeV $\gamma$ ray energies. In particular, blazars will be of great interest to MeV instruments, given their prevalence in the $\gamma$ ray sky.

## Table of contents
1. [Data](#Data)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Contact](#Contact)
5. [Project Status](#Status)
6. [Acknowledgements](#Acknowledgements)
   

## Data/Catalogs

## Requirements

## Usage


### Runner Scripts

```
Blazar_Fitter.py --config Configs/Blazar_Fitter_config_LP.yaml
```

```
Blazar_Fitter.py --config Configs/Blazar_Fitter_config_SBPL.yaml
```
<img width="3570" height="1509" alt="Example_SEDS" src="https://github.com/user-attachments/assets/62956445-de9b-4309-a7dc-195c0973f0c4" />

```
SED_Model_Selection.py
```

```
Template_SEDS.py
```

<img width="8959" height="2654" alt="SED_Templates" src="https://github.com/user-attachments/assets/e8003c7d-5871-4114-a50c-55d757a43fd8" />

```
N_S_LF_Calculation.py --config Configs/LF_SED_Template_Params.yaml
```

```
N_z_LF_Calculation.py --config Configs/LF_SED_Template_Params.yaml
```

### Plotter Scripts

```
Cross_Match_Blazar_Skymap.py
```
<img width="4600" height="2880" alt="Annotated_EG_Skymap_TNR" src="https://github.com/user-attachments/assets/665ba6ad-d692-4b7b-91f9-11960573d988" />


```
N_S_Plotter.py
```

```
N_z_Plotter.py
```

### Notebooks

```
Cross_Matched_Blazars.ipynb
```

```
MeV_Blazar_Forecasting.ipynb
```

## Contact

Please contact me at wessling-resnick.t@northeastern.edu for questions or comments.

## Status

## Acknowledgements
