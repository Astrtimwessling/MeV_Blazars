import numpy as np

from scipy.optimize import curve_fit

# Log Parabola

def nuLnu_LP(E, L0, alpha, beta, E0=1.0):
    return L0 * (E / E0) ** (2 - alpha - beta * np.log(E / E0))

def nuFnu_LP(E, E0, N0, alpha, beta):
    return N0 * (E / E0)**(-alpha - beta * np.log(E / E0))

def Fit_nuLnu_LP(avg_SED, alpha_guess, beta_guess, energy_grid):
    def log_model(E, L0, alpha, beta):
        return np.log10(nuLnu_LP(E, L0, alpha, beta))
    
    L0_guess = np.max(avg_SED)

    p0 = [L0_guess, alpha_guess, beta_guess]

    popt, pcov = curve_fit(log_model, energy_grid, np.log10(avg_SED), p0=p0, maxfev=10000)

    L0_fit, alpha_fit, beta_fit = popt
    
    return popt
    
# Smooth Broken Power Law

def nuLnu_SBPL(E, L0, Eb, alpha1, alpha2, s):
    term = (E / Eb)
    return L0 * term**(-alpha1) * (1 + term**((alpha2 - alpha1)/s))**(-s)

def nuFnu_SBPL(E, E0, N0, Gam1, Gam2, beta, E_Break):
        term1 = (E / E0)**(-Gam1)
        term2 = (1 + (E / E_Break)**((Gam2 - Gam1) / beta))**(-beta)
        return N0 * term1 * term2
    
def Fit_nuLnu_SBPL(avg_SED, alpha1_guess, alpha2_guess, energy_grid, beta_guess):
    def log_SBPL(E, L0, Eb, alpha1, alpha2, s):
        return np.log10(nuLnu_SBPL(E, L0, Eb, alpha1, alpha2, s))
    
    L0_guess = np.max(avg_SED)
    Eb_guess = energy_grid[np.argmax(avg_SED)]

    p0 = [L0_guess, Eb_guess, alpha1_guess, alpha2_guess, beta_guess]

    popt, pcov = curve_fit(log_SBPL, energy_grid, np.log10(avg_SED), p0=p0, maxfev=10000)
    
    return popt