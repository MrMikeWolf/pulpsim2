from __future__ import division
import numpy
from numpy import exp, sum, average
import os
import ConfigParser
import pandas
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plot

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

# Equally, we define `N = 1000` equally spaced grid points on our time domain of length `T = 200` thus dividing our
# time domain into `N-1` intervals of length `dt`.

# In[]:

T = 280

# ### Specify System Parameters and the Reaction Term

# State initial starting conditions

# In[]:

MM_NaOH = 23 + 16 + 1
MM_Na2S = 23 * 2 + 32
MM_Na2O = 23 * 2 + 16


def COH(AA, S):
    return AA * (2 * MM_NaOH / MM_Na2O) * (1 - S)

def C_S(AA, S):
    return AA * (2 * MM_NaOH / MM_Na2O) * S

def sigma_D(T):
    D = 1.5e-2 * (T ** 0.5) * exp(-4870 / 1.9872 / T)
    return D * dt / (2 * dx * dx)

def Kappa(L,CH):
    return 500*(L/(L+CH))+2

# Create Matrices

# The matrices that we need to construct are all tridiagonal so they are easy to construct with

# In[6]:

A_L = numpy.identity(J)
B_L = A_L
A_C = numpy.identity(J)
B_C = A_C

def ACA(sigma):
    return numpy.diagflat([-sigma for i in range(J - 1)], -1) + numpy.diagflat(
        [1. + sigma] + [1. + 2. * sigma for i in range(J - 2)] + [1. + sigma]) + numpy.diagflat(
        [-sigma for i in range(J - 1)], 1)

def BCA(sigma):
    return numpy.diagflat([sigma for i in range(J - 1)], -1) + numpy.diagflat(
        [1. - sigma] + [1. - 2. * sigma for i in range(J - 2)] + [1. - sigma]) + numpy.diagflat(
        [sigma for i in range(J - 1)], 1)

SF = 1


def temp(t,th, tf):
    """ Temperature function
    th: heating time
    tf: final temperature
    """
    # Starting temperature
    To = 273 + 25

    # Gradient at which heat changes degC/min
    # The gradient is specified such that a temp 
    # of 170 degC is reached at 'th'
    m = (tf - To )/ th

    if t <= th:
        # Heating time
        T = To + t * m
    else:
        # Cooking time
        T = To + th * m

    return T


# In[]:
count = 0
K_average = []
T_list = []

# Use config file to set path to the data file directory

config = ConfigParser.ConfigParser()
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
data = pandas.read_excel(Data_file, sheetname = "PULPING" ,skiprows = 4, skip_footer = 4)

# Create pdf document to save figures to
Conc_plot = PdfPages('Lignin.pdf')

for index, row in data.iterrows():

    AA = row['[AA]        g/L Na2O']
    Sulf = 0.3264
    tf = row['Tmax C'] + 273
    th = row['to Tmax min']
    T = row['at Tmax min'] # cook time
    T_list.append(T)

    N = 1000
    dt = float(T) / float(N - 1)
    t_grid = numpy.array([n * dt for n in range(N)])

    CS = C_S(AA, Sulf) / MM_Na2S  # Molar [mol/L]
    OH = COH(AA, Sulf) / MM_NaOH  # Molar [mol/L]

    L = numpy.array([0.27 for i in range(0, J)])
    CC = numpy.array([0.677 for i in range(0, J)])
    CA = numpy.array([OH] + [0 for i in range(1, J)])

    def f_vec(L, CC, CA, TC):
        if sum(L) >= 22:

            dLdt = lambda L, CC, CA, TC: dt * (36.2 * TC ** 0.5 * exp(-4807.69 / TC)) * L
            dCCdt = lambda L, CC, CA, TC: dt * 2.53 * 36.2 * T ** 0.5 * exp(-4807.69 / TC) * L * (CA ** 0.11)
            dCAdt = lambda L, CC, CA, TC: dt * SF * (
                (-4.78e-3 * 36.2 * T ** 0.5 * exp(-4807.69 / TC)) * L + 1.81e-2 * 2.53 *
                36.2 * T ** 0.5 * exp(-4807.69 / TC) * L * (CA ** 0.11))

        elif sum(L) >= 2.5:

            dLdt = lambda L, CC, CA, TC: dt * (
                exp(35.19 - 17200 / TC) * CA + (exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4))) * L
            dCCdt = lambda L, CC, CA, TC: dt * (
                0.47 * (exp(35.19 - 17200 / TC) * CA + (exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4)))) * L
            dCAdt = lambda L, CC, CA, TC: dt * SF * (-4.78e-3 * (exp(35.19 - 17200 / TC) * CA * L +
                                                                 exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4) * L)
                                                     + 1.81e-2 * 0.47 * (exp(35.19 - 17200 / TC) * CA * L + (
                exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4) * L)))

        else:

            dLdt = lambda L, CC, CA, TC: dt * (exp(19.64 - 10804 / TC)) * L * (CA ** 0.7)
            dCCdt = lambda L, CC, CA, TC: dt * 2.19 * (exp(19.64 - 10804 / TC)) * L * (CA ** 0.7)
            dCAdt = lambda L, CC, CA, TC: dt * SF * (-4.78e-3 * (
                exp(19.64 - 10804 / TC) * (CA ** 0.7) * L + 1.81e-2 * 2.19 * exp(19.64 - 10804 / TC) * (CA ** 0.7) * L))

        return dLdt, dCCdt, dCAdt

    C_bulk = OH
    SF2 = 0.001

    L_record = []
    CC_record = []
    CA_record = []
    CA_bulk_record = []
    K = []

    count += 1
    print(count)

    L_record.append(L)
    CC_record.append(CC)
    CA_record.append(CA)
    for ti in range(1, N):
        t = ti*(T/N)
        TC = temp(t, th, tf)

        vec_L, vec_CC, vec_CA = f_vec(L, CC, CA, TC)

        sigma = sigma_D(TC)
        A_CA = ACA(sigma)
        B_CA = BCA(sigma)

        L_new = numpy.linalg.solve(A_L, B_L.dot(L) - vec_L(L, CC, CA, TC))
        CC_new = numpy.linalg.solve(A_C, B_C.dot(CC) - vec_CC(L, CC, CA, TC))
        CA_new = numpy.linalg.solve(A_CA, B_CA.dot(CA) - vec_CA(L, CC, CA, TC))

        val = (dt / dx) * SF2 * (CA[1] - CA[0])
        C_bulk += val
        CA_bulk_record.append(C_bulk)

        L = L_new
        CC = CC_new
        CA = CA_new

        # The liquor penetration rate is infinite (CAi bulk = CAi of first compartment)
        CA_new[0] = C_bulk
        CA[CA < 0] = 0
        K.append(Kappa(L,CC))

        L_record.append(L)
        CC_record.append(CC)
        CA_record.append(CA)

    K_average.append(average(K[-1]))
    plot.figure()
    plot.title('Final heating temp. = {} K'.format(tf))
    plot.xlabel('time [min]')
    plot.ylabel('Lignin [% on wood]')
    plot.plot(t_grid, sum(L_record, axis= 1))
    Conc_plot.savefig()

Conc_plot.close()
data['Kappa lit.'] = K_average
data.to_excel('Kappa_comparison.xlsx')