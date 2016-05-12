from __future__ import division, print_function
import numpy
from numpy import exp, sum, average, ones
from matplotlib import pyplot as plot
import matplotlib as mpl

# Python 2.7 compatibility
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas


def plot_text_formatter(fonttype='sans-serif', fontsize=7, axislabel=7, xtick=7, ytick=7, label_pad=2):
    """
    Formats plot text.
    :param fonttype: Font type
    :param fontsize:  Font size
    :param axislabel: Axis label size
    :param xtick: xtick label size
    :param ytick: ytick label size
    """
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['font.family'] = fonttype
    mpl.rcParams['axes.labelsize'] = axislabel
    mpl.rcParams['xtick.labelsize'] = xtick
    mpl.rcParams['ytick.labelsize'] = ytick
    mpl.rcParams['axes.labelpad'] = label_pad

numpy.set_printoptions(precision=3)

# ### Specify Grid

# Our one-dimensional domain has unit length and we define `J = 100` equally spaced
# grid points in this domain.
# This divides our domain into `J-1` subintervals, each of length `dx`.

# In[1]:

L = 1.
J = 100
dx = float(L) / float(J - 1)
x_grid = numpy.array([j * dx for j in range(J)])

# ### Specify System Parameters and the Reaction Term


def sigma_D(T):
    D = 1.5e-2 * (T ** 0.5) * exp(-4870 / 1.9872 / T)
    return D * dt / (2 * dx * dx)

# Carbohhdrates are the sum of Cellulose, Glucomannan and Xylan


def carbo_sum(C, G ,X):

    return C+G+X

# Concentration calculations for NaOH and Na2S

MM_NaOH = 23 + 16 + 1
MM_Na2S = 23 * 2 + 32
MM_Na2O = 23 * 2 + 16


def COH(AA, S):
    return AA * (2 * MM_NaOH / MM_Na2O) * (1 - S)


def C_S(AA, S):
    return AA * (2 * MM_NaOH / MM_Na2O) * S

# Specify the Initial Concentrations

# In[]:


def initial(xi):
    xi /= 100
    return numpy.array([xi for i in range(0,J)])

# Create Matrices

# The matrices that we need to construct are all tridiagonal so they are easy to construct with

# In[6]:

A_L = numpy.identity(J)
B_L = A_L
A_C = numpy.identity(J)
B_C = A_C

# Solve the System Iteratively

# In[]:


config = ConfigParser()
configfile = 'config.cfg'

if os.path.exists(configfile):
    config.read('config.cfg')
else:
    message = ("Cannot find config file {0}. "
               "Try copying sample_config.cfg to {0}.").format(configfile)
    raise EnvironmentError(message)

datadir = os.path.expanduser(config.get('paths', 'datadir'))
Data_file = os.path.join(datadir, 'RFP 0339 - Pre-treatment part two.xlsx')

# Create object from which the data can be read from
data = pandas.read_excel(Data_file, sheetname = "PULPING", skiprows=4, skipfooter=21)

# Create pdf document to save figures to
Kappa_Lig_Carbo_plot = PdfPages('Kappa_Lignin_Carbo.pdf')


def Lignin(A, Ea, a, b, L, CA, CS, T):
    R = 8.314
    return A*exp(Ea/R*(1/443.15 - 1/T))*(CA**a)*(CS**b)*L


def Carbo(A, Ea, a, b, k2, C, CA, CS, T):
    R = 8.314
    return A*exp(Ea/R*(1/443.15 - 1/T))*((CA**a)*(CS**b) + k2)*C


def f_vec(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC):

    # Lignin consumption

    dL1dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Lignin(0.1,50000,0,0.06,L1,CA,CS,TC)
    dL2dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Lignin(0.1,127000,0.48,0.39,L2,CA,CS,TC)
    dL3dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Lignin(0.0047,127000,0.2,0,L3,CA,CS,TC)

    # Carbohydrate consumption

        # 1) Cellulose

    dC1dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.06,50000,0.1,0,0,C1,CA,CS,TC)
    dC2dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.054,144000,1,0,0.22,C2,CA,CS,TC)
    dC3dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(6.4e-4,144000,1,0,0.42,C3,CA,CS,TC)

        # 2) Glucomannan

    dG1dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.06,50000,0.1,0,0,G1,CA,CS,TC)
    dG2dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.054,144000,1,0,0.22,G2,CA,CS,TC)
    dG3dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(6.4e-4,144000,1,0,0.42,G3,CA,CS,TC)

        # 3) Xylan

    dX1dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.06,50000,0.1,0,0,X1,CA,CS,TC)
    dX2dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(0.054,144000,1,0,0.22,X2,CA,CS,TC)
    dX3dt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: Carbo(6.4e-4,144000,1,0,0.42,X3,CA,CS,TC)

    return dL1dt, dL2dt, dL3dt, dC1dt, dC2dt, dC3dt, dG1dt, dG2dt, dG3dt, dX1dt, dX2dt, dX3dt


# In[]

def temp(t):
    """ Temperature function
    """

    # Heating time in minutes
    th = 120
    # Starting temperature
    To = 273 + 2

    # Gradient at which heat changes degC/min
    # The gradient is specified such that a temp 
    # of 170 degC is reached at 'th'
    m = ((170 + 273) - To) / 120

    if t <= th:
        # Heating time
        T = To + t * m
    else:
        # Cooking time
        T = To + th * m

    return T


def pulp_yield(L, C):
    return sum(L, axis=1)+sum(C, axis=1)

# In[]:

C_bulk = 0.5
SF2 = 0.001
cnt = 0

for index, row in data.iterrows():
    cnt+=1
    print('Execution count: {}'.format(cnt))

    AA = row['[AA]        g/L Na2O']
    Sulf = 0.3264
    tf = row['Tmax C'] + 273
    th = row['to Tmax min']
    T = row['total min'] # cook time
    K_exp = row['Kappa number']
    Run = row['COOK']

    CS = C_S(AA, Sulf) / MM_Na2S  # Molar [mol/L]
    OH = COH(AA, Sulf) / MM_NaOH  # Molar [mol/L]

    N = 1000
    dt = float(T) / float(N - 1)
    t_grid = numpy.array([n * dt for n in range(N)])

    L1 = initial(9)
    L2 = initial(19)
    L3 = initial(1.5)

    C1 = initial(2.4)
    C2 = initial(4.2)
    C3 = initial(36.9)

    G1 = initial(10.3)
    G2 = initial(3.4)
    G3 = initial(6.1)

    X1 = initial(0.9)
    X2 = initial(1.7)
    X3 = initial(4.6)

    CA = initial(OH)

    L_record = []
    C_record = []
    G_record = []
    X_record = []

    L1_record = []
    L2_record = []
    L3_record = []
    C1_record = []
    C2_record = []
    C3_record = []
    G1_record = []
    G2_record = []
    G3_record = []
    X1_record = []
    X2_record = []
    X3_record = []

    L1_record.append(L1)
    L2_record.append(L2)
    L3_record.append(L3)
    C1_record.append(C1)
    C2_record.append(C2)
    C3_record.append(C3)
    G1_record.append(G1)
    G2_record.append(G2)
    G3_record.append(G3)
    X1_record.append(X1)
    X2_record.append(X2)
    X3_record.append(X3)

    L_record.append(L1+L2+L3)
    C_record.append(C1+C2+C3)
    G_record.append(G1+G2+G3)
    X_record.append(X1+X2+X3)

    Carbo_record = [carbo_sum(C1+C2+C3, G1+G2+G3, X1+X2+X3)]
    Kappa_record = []


    def Kappa(L, CH):
        return 500*(L/(L+CH))+2
    L = L1+L2+L3
    Carb = C1+C2+C3+G1+G2+G3+X1+X2+X3
    Kappa_record.append(Kappa(L, Carb))

    for ti in range(1, N):
        t = ti*(T/N)
        TC = temp(t)
        sigma = sigma_D(TC)

        vec_L1,vec_L2,vec_L3,vec_C1,vec_C2,vec_C3,vec_G1,vec_G2,vec_G3,vec_X1,vec_X2,vec_X3 = f_vec(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC)

        L_new = numpy.linalg.solve(A_L, B_L.dot(L1) - vec_L1(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        L2_new = numpy.linalg.solve(A_L, B_L.dot(L2) - vec_L2(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        L3_new = numpy.linalg.solve(A_L, B_L.dot(L3) - vec_L3(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))

        C_new = numpy.linalg.solve(A_C, B_C.dot(C1) - vec_C1(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        C2_new = numpy.linalg.solve(A_L, B_C.dot(C2) - vec_C2(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        C3_new = numpy.linalg.solve(A_L, B_C.dot(C3) - vec_C3(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))

        G_new = numpy.linalg.solve(A_C, B_C.dot(G1) - vec_G1(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        G2_new = numpy.linalg.solve(A_C, B_C.dot(G2) - vec_G2(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        G3_new = numpy.linalg.solve(A_C, B_C.dot(G3) - vec_G3(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))

        X_new = numpy.linalg.solve(A_C, B_C.dot(X1) - vec_X1(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        X2_new = numpy.linalg.solve(A_C, B_C.dot(X2) - vec_X2(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
        X3_new = numpy.linalg.solve(A_C, B_C.dot(X3) - vec_X3(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))

        L1 = L_new
        L2 = L2_new
        L3 = L3_new
        L = L1+L2+L3

        C1 = C_new
        C2 = C2_new
        C3 = C3_new
        C = C1+C2+C3

        G1 = G_new
        G2 = G2_new
        G3 = G3_new
        G = G1+G2+G3

        X1 = X_new
        X2 = X2_new
        X3 = X3_new
        X = X1+X2+X3

        L1_record.append(L1)
        L2_record.append(L2)
        L3_record.append(L3)
        L_record.append(L)

        C1_record.append(C1)
        C2_record.append(C2)
        C3_record.append(C3)
        C_record.append(C)

        G1_record.append(G1)
        G2_record.append(G2)
        G3_record.append(G3)
        G_record.append(G)

        X1_record.append(X1)
        X2_record.append(X2)
        X3_record.append(X3)
        X_record.append(X)

        Alkali = OH*ones(N)
        Sulfide = CS*ones(N)
        Carbo_record.append(carbo_sum(C, G, X))
        Kappa_record.append(Kappa(L, C))

    Kappa_average = average(Kappa_record, axis = 1)
    # ### Plot the Numerical Solution

    # Let us take a look at the numerical solution we attain after `N` time steps.

    # In[]:

    print('Average Kappa number: {}'.format(Kappa_average[-1]))

    plot_text_formatter()

    fig = plot.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Cook number: {}'.format(Run))
    ax2 = ax1.twinx()

    # ax1.plot()
    ax1.set_xlabel('time [min]')
    ax2.set_ylabel('$\kappa$')
    ax1.set_ylabel('Lignin [% mass]')
    l1 = ax1.plot(t_grid, Kappa_average, 'r-.', label='$\kappa$')
    l2 = ax2.plot(t_grid, sum(L_record, axis=1), 'b-', label='$L_{tot}$')

    if type(K_exp) == float:
        ax1.plot(T, K_exp, 'rx')

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines])

    ax3 = fig.add_subplot(2,2,2)
    ax3.set_xlabel('time [min]')
    ax3.set_ylabel('Carbohydrates [% mass]')
    l3 = ax3.plot(t_grid, sum(Carbo_record, axis=1), 'g-', label='$Carb_{tot}$')
    l4 = ax3.plot(t_grid, sum(C_record, axis=1), 'g--', label='$Cellulose$')
    l5 = ax3.plot(t_grid, sum(G_record, axis=1), 'g-.', label='$Glucoman.$')
    l6 = ax3.plot(t_grid, sum(X_record, axis=1), 'g:', label='$Xylan$')
    lines2 = l3 + l4 + l5 + l6
    ax3.legend(lines2, [l.get_label() for l in lines2])
    fig.subplots_adjust(hspace = 0.33)
    fig.subplots_adjust(wspace = 0.33)

    ax11 = fig.add_subplot(2,2,3)
    ax11.set_xlabel('time [min]')
    ax11.set_ylabel('Yield')
    ax11.plot(t_grid, pulp_yield(L_record, Carbo_record),'r', label = 'Pulp Yield')
    ax11.legend()

    ax22 = fig.add_subplot(2,2,4)
    ax22.set_xlabel('time [min]')
    ax22.set_ylabel('Residual alkali')
    l7 = ax22.plot(t_grid, Alkali, 'm', label='OH')
    ax33 = ax22.twinx()
    ax33.set_ylabel('Sulfide')
    l8 = ax33.plot(t_grid, Sulfide,'m-.', label='CS')
    lines3 = l7 + l8
    ax22.legend(lines3, [l.get_label() for l in lines3])

    Kappa_Lig_Carbo_plot.savefig()
    plot.close()

Kappa_Lig_Carbo_plot.close()
