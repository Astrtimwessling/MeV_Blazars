import matplotlib.pyplot as plt
from matplotlib import rc

def set_rc_params(fontsize=None):
    #print ("Setting Matplotlib RC parameters...")
    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)
    rc('font',**{'family':'serif'})

    ## Try to use LaTeX for plotting, otherwise use simple text
    try:
        import subprocess
        subprocess.run(['latex', '--version'],
                        capture_output=True, check=True)
        rc('text', usetex=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        rc('text', usetex=False)
        print("LaTeX not found. Using simple text rendering")

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize-2})
    plt.rcParams.update({'ytick.labelsize': fontsize-2})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams.update({'figure.labelsize': fontsize})

    return