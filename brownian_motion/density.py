import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.optimize import curve_fit

########################################
y_error_pres = 0.2
title = lambda x: r'$<R^2>\/ VS\/\/ t-\/Density\/%1.1f$' % x
kb = 1  # 1.38 * math.pow(10, -23)
T = 300
des = 0.05
np.random.seed(11197)
all_d = []
all_d_eror=[]
C = (kb * T) / (6 * math.pi)
dens = np.array([0.05, 0.1, 0.3, 0.4, 0.6, 0.75, 0.85, 0.95])


########################################
def single_line_t_r(xdata, func):

    for des in dens:
        # data
        y = func(xdata, C / des)
        # data points:
        y = func(xdata, C / des)
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
        plt.savefig(f'C:/Users/asicherm/private/matlab/brown/ruk/{int(des * 100)}')


def marge_t_r(xdata, func):
    fig = plt.figure()
    # go over all denst
    for des in dens:
        # data
        y = func(xdata, C / des)
        y_noise = np.random.normal(size=xdata.size) * y_error_pres * np.mean(y)
        ydata = y + y_noise
        plt.scatter(xdata, ydata, marker='+', label=f'Density{des:5.3f}')

        #fit (without line)
        popt, pcov = curve_fit(func, xdata, ydata)
        sigma = np.sqrt(np.diag(pcov))
        dMin = popt[0] - 2 * sigma[0]
        dMax = popt[0] + 2 * sigma[0]
        plt.xlabel(f'time [s]', fontsize=14)
        plt.ylabel(r'$<r^2>$', fontsize=14)
        plt.legend(fontsize=12)
        plt.title(title(des), fontsize=16)

        plt.savefig(f'C:/Users/asicherm/private/matlab/brown/ruk/marge')
        all_d.append(popt[0])
        all_d_eror.append((dMax-dMin)/2)

def c_den():
    plt.figure()

    ydata = np.array(all_d)
    plt.errorbar(dens,all_d,yerr=all_d_eror,xerr=dens*0.001,marker='+',linestyle="",label='data')
    nX = np.linspace(0, 1, 500)
    popt, pcov = curve_fit(func_dens_d, dens, ydata)
    sigma = np.sqrt(np.diag(pcov))
    dMin = popt[0] - 2 * sigma[0]
    dMax = popt[0] + 2 * sigma[0]
    textstr = f'D\'={popt[0]:5.3f} [{dMin:5.3f}-{dMax:5.3f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.45, 0.25, textstr, fontsize=14, bbox=props)
    plt.plot(nX, func_dens_d(nX, *popt), 'r-',label=r'$fit-\/D=\~D/\eta$')
    plt.title(f'The Diffusion Constant VS  Density',fontsize=16)
    plt.ylim([0,400])
    plt.xlabel(f'Density', fontsize=14)
    plt.ylabel(r'Diffusion Constant', fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(f'C:/Users/asicherm/private/matlab/brown/ruk/C2')




def func_r_squere_t(t, D):
    return D * t
    # return a * np.exp(-b * x) + c / d

def func_dens_d(dens, c):
    return c / dens


if __name__ == "__main__":
    # Density
    xdata = np.linspace(0, 4, 50)
    single_line_t_r(xdata,func_r_squere_t)
    marge_t_r(xdata,func_r_squere_t)
    c_den()

    # temp:

