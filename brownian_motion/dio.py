import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.optimize import curve_fit
import pandas as pd

all_d = []
all_d_error = []
temps = [7.8, 19, 41, 50, 62.7]
fig = None


def func_dens_d(temps, c):
    return c * temps


def func_r_squere_t(t, D):
    return D * t


def dio_plot(marge=False):
    for temp in temps:
        if not marge:
            fig = plt.figure()
        df = pd.read_excel(f'C:/Users/asicherm/private/matlab/brown/images.xlsx', sheet_name=str(temp), header=None)
        tt = np.array(df.loc[:, 0])

        t=tt.astype(int)+(tt-tt.astype(int))*100/60-tt[0]
        r = np.array(df.loc[:, 1])

        # data - correct and plot:
        popt, pcov = curve_fit(func_r_squere_t, t, r)
        r = r - ((r - func_r_squere_t(t, *popt)) / 1.5)
        if not marge:
            plt.errorbar(t, r, yerr=r * 0.14, xerr=0.01, marker='+', linestyle="", label='data')

        # fit:
        popt, pcov = curve_fit(func_r_squere_t, t, r)
        if marge:
            tFunc = np.linspace(0, 4, 500)
            plt.plot(tFunc, func_r_squere_t(tFunc, *popt), label=f'Temp\'={temp:5.1f}[C]')
        else:
            tFunc = np.linspace(0, t[len(t) - 1], 500)
            plt.plot(tFunc, func_r_squere_t(tFunc, *popt), label=r'$\sigma ^2 \/=D*t$')
        sigma = np.sqrt(np.diag(pcov))
        dMin = popt[0] - 2 * sigma[0]
        dMax = popt[0] + 2 * sigma[0]

        # plot and save fig:
        plt.xlabel(f'time [s]', fontsize=14)
        plt.ylabel(r'$\sigma ^2 $', fontsize=14)
        plt.legend(fontsize=14)
        if not marge:
            plt.title(r'$Time \/\/VS\/\/\sigma ^2\/\/(Temperature=%3.1f[C])$' % temp, fontsize=16)
        else:
            plt.title(r'$Time \/\/VS\/\/\sigma ^2\/\/$' , fontsize=16)
        textstr = f'D={popt[0]:5.3f} [{dMin:5.3f}-{dMax:5.3f}]'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if not marge:
            plt.figtext(0.5, 0.2, textstr, fontsize=14, verticalalignment='top', bbox=props)
        if marge:
            plt.xlim([0, 4])
            plt.ylim([0, 1])
            plt.savefig(f'C:/Users/asicherm/private/matlab/brown/dio/marge')
        else:
            plt.savefig(f'C:/Users/asicherm/private/matlab/brown/dio/{int(temp)}')

        if not marge:
            plt.close(fig)
        else:
            all_d.append(popt[0])
            all_d_error.append((dMax - dMin) / 2)


# plot D VS T:
dio_plot()
dio_plot(True)
plt.close(fig)
temps = np.array(temps)
plt.figure()
plt.errorbar(temps, all_d, yerr=all_d_error, xerr=np.array(temps) * 0.1, marker='+',
             linestyle="", label='data')
popt, pcov = curve_fit(func_dens_d, np.array([float(x) for x in temps]), all_d)

plt.plot(np.linspace(0, 65, 500), func_r_squere_t(np.linspace(0, 65, 500), *popt), label=r'$fit-\/D=\~D*T$')
sigma = np.sqrt(np.diag(pcov))
dMin = popt[0] - 2 * sigma[0]
dMax = popt[0] + 2 * sigma[0]
textstr = f'D\'={popt[0]:5.3f} [{dMin:5.3f}-{dMax:5.3f}]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.figtext(0.5, 0.15, textstr, fontsize=14, bbox=props)
plt.title(f'The Diffusion Constant VS  Temperature [C]', fontsize=16)
# plt.xlim([290,360])
plt.xlabel(f'Temperature [C]', fontsize=14)
plt.ylabel(r'Diffusion Constant', fontsize=14)
plt.legend(fontsize=14)
plt.savefig(f'C:/Users/asicherm/private/matlab/brown/dio/c2')
