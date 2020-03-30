import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.optimize import curve_fit

########################################
y_error_pres = 0.1
title = lambda x: r'$<R^2>\/ VS\/\/ t-\/Density\/%1.1f$' % x
kb = 1  # 1.38 * math.pow(10, -23)
T = 300
des = 0.3
np.random.seed(11197)
all_d = []
all_d_eror=[]
C = (kb) / (6 * math.pi*des)
temps = np.array([299, 309, 322, 331, 345, 350])


########################################
def single_line_t_r(xdata, func):

    for tem in temps:
        # data points:
        y = func(xdata, C *tem)
        y_noise = np.random.normal(size=xdata.size) * y_error_pres * np.mean(y)
        ydata = y + y_noise
        plt.figure()
        plt.scatter(xdata, ydata, marker='+', label='data')

        # fit points
        popt, pcov = curve_fit(func, xdata, ydata)
        plt.plot(xdata, func(xdata, *popt), 'r-', label=r'$<R^2>\/=D*t$')
        sigma = np.sqrt(np.diag(pcov))
        dMin = popt[0] - 2 * sigma[0]
        dMax = popt[0] + 2 * sigma[0]

        # plot and save fig:
        plt.xlabel(f'time [s]', fontsize=14)
        plt.ylabel(r'$<r^2>$', fontsize=14)
        plt.legend(fontsize=14)
        plt.title(title(des), fontsize=16)
        textstr = f'D={popt[0]:5.3f} [{dMin:5.3f}-{dMax:5.3f}]'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, 0.95, textstr, fontsize=14, verticalalignment='top', bbox=props)
        plt.savefig(f'C:/Users/asicherm/private/matlab/brown/temp/{int(tem)}')


def marge_t_r(xdata, func):
    fig = plt.figure()
    # go over all denst
    for tem in temps:
        # data
        y = func(xdata, C *tem)
        y_noise = np.random.normal(size=xdata.size) * y_error_pres * np.mean(y)
        ydata = y + y_noise
        plt.scatter(xdata, ydata, marker='+', label=f'Temperature={tem:5.3f}')

        #fit (without line)
        popt, pcov = curve_fit(func, xdata, ydata)
        sigma = np.sqrt(np.diag(pcov))
        dMin = popt[0] - 2 * sigma[0]
        dMax = popt[0] + 2 * sigma[0]
        plt.xlabel(f'time [s]', fontsize=14)
        plt.ylabel(r'$<r^2>$', fontsize=14)
        plt.legend(fontsize=12)
        plt.title(title(des), fontsize=16)

        plt.savefig(f'C:/Users/asicherm/private/matlab/brown/temp/marge')
        all_d.append(popt[0])
        all_d_eror.append((dMax-dMin)/2)

def c_den():
    plt.figure()
    ydata = np.array(all_d)
    plt.errorbar(temps,all_d,yerr=all_d_eror,xerr=temps*0.01,marker='+',linestyle="",label='data')
    nX = np.linspace(290, 360, 500)
    popt, pcov = curve_fit(func_dens_d, temps, ydata)
    sigma = np.sqrt(np.diag(pcov))
    dMin = popt[0] - 2 * sigma[0]
    dMax = popt[0] + 2 * sigma[0]
    textstr = f'D\'={popt[0]:5.3f} [{dMin:5.3f}-{dMax:5.3f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.5, 0.15, textstr, fontsize=14, bbox=props)
    plt.plot(nX, func_dens_d(nX, *popt), 'r-',label=r'$fit-\/D=\~D*T$')
    plt.title(f'The Diffusion Constant VS  Temperature',fontsize=16)
    plt.xlim([290,360])
    plt.xlabel(f'Temperature [K]', fontsize=14)
    plt.ylabel(r'Diffusion Constant', fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(f'C:/Users/asicherm/private/matlab/brown/temp/C2')




def func_r_squere_t(t, D):
    return D * t
    # return a * np.exp(-b * x) + c / d

def func_dens_d(temps, c):
    return c * temps


if __name__ == "__main__":
    # Density
    xdata = np.linspace(0, 4, 50)
    single_line_t_r(xdata,func_r_squere_t)
    marge_t_r(xdata,func_r_squere_t)
    c_den()

    # temp:

