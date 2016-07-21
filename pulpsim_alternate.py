from __future__ import division, print_function
import numpy
from numpy import exp, sum, average, ones, array
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
from tqdm import tqdm


def plot_text_formatter(fonttype='sans-serif', fontsize=7, axislabel=9, xtick=9, ytick=9, label_pad=2):
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
    mpl.rcParams['figure.titlesize'] = 12

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

# Carbohhdrates are the sum of Cellulose, Glucomannan and Xylan


def carbo_sum(C, G ,X):

    return C+G+X

# Concentration calculations for NaOH and Na2S

MM_NaOH = 23 + 16 + 1
MM_Na2S = 23 * 2 + 32
MM_Na2O = 23 * 2 + 16

OD = 750  # Oven dried wood charge which is 750g in all cases
LW = 4.1

def COH(AA, S):
    """

    Parameters
    ----------
    LW : Liquor to Wood ratio in kg/kg
    AA : Active Alkali to Oven Dried (OD) wood %
    S : Sulfidity as a %

    Returns
    -------
    [OH] : Concentration of NaOH of the liquor in units g/L Na2O

    """

    AA_Na2O = (AA/100)*(OD/(OD*LW))*1000 # Active alkali in concentration units g/L Na2O
    return AA_Na2O * (2 * MM_NaOH / MM_Na2O) * (1 - S)


def C_S(AA, S):

    """

    Parameters
    ----------
    AA : Active alkali to OD wood %
    S : Sulfidity as a %

    Returns
    -------
    [HS] - Hydrogen sulfide concentration in units g/L Na2O

    """
    AA_Na2O = (AA/100)*(OD/(OD*LW))*1000
    return AA_Na2O * (2 * MM_NaOH / MM_Na2O) * S

# Specify the Initial Concentrations


def initial(xi):
    xi /= 100
    return numpy.array([xi for i in range(0,J)])

# Create Matrices

# The matrices that we need to construct are all tridiagonal so they are easy to construct with

A_L = numpy.identity(J)
B_L = A_L
A_C = numpy.identity(J)
B_C = A_C

# Unpack the data from experimental results

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
data = pandas.read_excel(Data_file, sheetname="PULPING", skiprows=4, skipfooter=4)

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


# Define Temperature ramp function

def temp(t, th, Tf):
    """ Temperature function
    t - at a specific time
    th - heating time
    Tf - final temperature

    Parameters
    ----------
    t : time in minutes
    th : heating time in minutes (time to reach maximum temperature)
    Tf : Final temperature in Kelvin

    Returns
    -------
    T : Temperature at time t in Kelvin
    """

    # Starting temperature
    To = 273 + 25

    # Gradient at which heat changes degC/min
    # The gradient is specified such that a temp 
    # of 170 degC is reached at 'th'
    m = (Tf - To) / th

    if t <= th:
        # Heating time
        T = To + t * m
    else:
        # Cooking time
        T = To + th * m

    return T


def pulp_yield(L, C):
    return sum(L, axis=1)+sum(C, axis=1)


def Kappa(L, CH):
    return 500*(L/(L+CH))+2

# Create counter for iterations

cnt = 0

# Parity plot empty lists

# Model values
parity_lig = []
parity_K = []
parity_yield = []
parity_xylan = []
parity_mannose = []

# Experimental values
exp_lig = []
exp_K = []
exp_yield = []
exp_xylan = []
exp_mannose = []

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    cnt+=1

    AA = row['AA to OD wood        %']
    Sulf = 0.3264
    Tf = row['Tmax C'] + 273
    th = row['to Tmax min']
    t_tot = row['total min'] # cook time
    K_exp = row['Kappa number']
    Run = row['COOK']
    Yield = row[r'total yield']
    Lig = row['Insoluble Lignin']
    xylose = row['Xylose']
    mannose = row['Mannose']

    if Run == 3117 or Run == 3124 or Run == 3126:
        continue

    # Experimental results
    exp_lig.append(Lig)
    exp_K.append(K_exp)
    exp_yield.append(Yield)
    exp_xylan.append(xylose)
    exp_mannose.append(mannose)

    CS = C_S(AA, Sulf) / MM_Na2S  # Molar [mol/L]
    OH = COH(AA, Sulf) / MM_NaOH  # Molar [mol/L]

    N = 1000
    dt = float(t_tot) / float(N - 1)
    t_grid = numpy.array([n * dt for n in range(N)])

    # The three species of each compound L,C,G and X. The composition of each species in a compound are given in

    # research on Picea Abies and i will assume the same ratio for each spieces in the compounds L,C,G andX.

    # Fengel and Wenger = FW_i,
    # Lindsrom and Lingdgren = LL_i,
    # where i = L,C,G,X

    FW_L = 28
    FW_C = 40.4
    FW_G = 22.2
    FW_X = 8.9

    LL_L = 29.5
    LL_C = 43.5
    LL_G = 19.8
    LL_X = 7.2

    # Weight_i = W_i
    W_L = FW_L / LL_L
    W_C = FW_C / LL_C
    W_G = FW_G / LL_G
    W_X = FW_X / LL_X

    L1 = initial(9) * W_L
    L2 = initial(19) * W_L
    L3 = initial(1.5) * W_L

    C1 = initial(2.4) * W_C
    C2 = initial(4.2) * W_C
    C3 = initial(36.9) * W_C

    G1 = initial(10.3) * W_G
    G2 = initial(3.4) * W_G
    G3 = initial(6.1) * W_G

    X1 = initial(0.9) * W_X
    X2 = initial(1.7) * W_X
    X3 = initial(4.6) * W_X

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

    L = L1+L2+L3
    Carb = C1+C2+C3+G1+G2+G3+X1+X2+X3
    Kappa_record.append(Kappa(L, Carb))

    for ti in range(1, N):
        t = ti*(t_tot / N)
        TC = temp(t, th, Tf)

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

    # Parity plot
    parity_K.append(Kappa_average[-1])
    parity_lig.append(sum(L_record, axis=1)[-1])
    parity_yield.append(pulp_yield(L_record, Carbo_record)[-1])
    parity_xylan.append(sum(X_record, axis=1)[-1])
    parity_mannose.append(sum(G_record, axis=1)[-1])

    # Let us take a look at the numerical solution we attain after `N` time steps.
    plot_text_formatter()

    fig = plot.figure()
    plot.suptitle('Cook number: {}'.format(Run))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()

    # Kappa number and Lignin plot
    ax1.set_xlabel('time [min]')
    ax1.set_ylabel('$\kappa$')
    ax2.set_ylabel('Lignin [% mass]')

    l1 = ax1.plot(t_grid, Kappa_average, 'r-.', label='$\kappa$')
    l2 = ax2.plot(t_grid, sum(L_record, axis=1), 'b-', label='$L_{tot}$')
    ax1.plot(t_tot, K_exp, 'rx')  # Kappa experimental
    ax2.plot(t_tot, Lig, 'bx')    # Lignin experimental
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines])

    # Carbohydrates plotting
    ax3 = fig.add_subplot(2,2,2)
    ax3.set_xlabel('time [min]')
    ax3.set_ylabel('Carbohydrates [% mass]')
    l3 = ax3.plot(t_grid, sum(Carbo_record, axis=1), 'darkgreen', label='$Carb_{tot}$')
    l4 = ax3.plot(t_grid, sum(C_record, axis=1), 'mediumseagreen', label='$Cellulose$')
    l5 = ax3.plot(t_grid, sum(G_record, axis=1), 'limegreen', label='$Glucoman.$')
    l6 = ax3.plot(t_grid, sum(X_record, axis=1), 'darkseagreen', label='$Xylan$')
    lines2 = l3 + l4 + l5 + l6
    ax3.legend(lines2, [l.get_label() for l in lines2])
    ax3.plot(t_tot, mannose, marker='x', color='limegreen')    # Mannose experimental
    ax3.plot(t_tot, xylose, marker='x', color='darkseagreen')  # Xylan experimental
    fig.subplots_adjust(hspace=0.33)
    fig.subplots_adjust(wspace=0.33)

    # Yield plot
    ax11 = fig.add_subplot(2,2,3)
    ax11.set_xlabel('time [min]')
    ax11.set_ylabel('Yield')
    ax11.plot(t_tot, Yield, 'rx')  # Yield experimental
    ax11.plot(t_grid, pulp_yield(L_record, Carbo_record), 'r', label='Pulp Yield')
    ax11.legend()

    # Alkali
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


def parity_error(experiment, model):
    return max([abs((E-M)/E) for E, M in zip(experiment, model)])


def error_plot(axis, error, limit):
    y = x = [0, limit]
    axis.plot(x, y)
    axis.plot([0, limit], [0, limit*(1-error)], 'r-.')
    axis.plot([0, limit], [0, limit*(1+error)], 'k-.')


fig = plot.figure()
# plot.suptitle('parity plot')
ax1 = fig.add_subplot(3, 2, 1)
ax1.scatter(exp_lig, parity_lig)
error_plot(ax1, parity_error(exp_lig, parity_lig), 30)
ax1.set_xlabel('Lignin exp')
ax1.set_ylabel('Lignin model')


ax2 = fig.add_subplot(3, 2, 2)
ax2.scatter(exp_K, parity_K)
error_plot(ax2, parity_error(exp_K, parity_K), 160)
ax2.set_xlabel('$\kappa$ exp')
ax2.set_ylabel('$\kappa$')

ax3 = fig.add_subplot(3, 2, 3)
ax3.scatter(exp_yield, parity_yield)
error_plot(ax3, parity_error(exp_yield, parity_yield), 80)
ax3.set_xlabel('Yield exp')
ax3.set_ylabel('Yield model')

ax4 = fig.add_subplot(3, 2, 4)
ax4.scatter(exp_xylan, parity_xylan)
error_plot(ax4, parity_error(exp_xylan, parity_xylan), 10)
ax4.set_xlabel('Xylan exp')
ax4.set_ylabel('Xylan model')

ax5 = fig.add_subplot(3, 2, 5)
ax5.scatter(exp_mannose, parity_mannose)
error_plot(ax5, parity_error(exp_mannose, parity_mannose), 14)
ax5.set_xlabel('Mannose exp')
ax5.set_ylabel('Mannose model')

plot.tight_layout()
Kappa_Lig_Carbo_plot.savefig()
plot.close()
Kappa_Lig_Carbo_plot.close()
