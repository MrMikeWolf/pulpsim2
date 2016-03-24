from __future__ import division
import numpy
from numpy import multiply, add, power, exp
from matplotlib import pyplot as plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
N = 1000
dt = float(T) / float(N - 1)
t_grid = numpy.array([n * dt for n in range(N)])

x_grid, t_grid = numpy.meshgrid(x_grid, t_grid)

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


# Specify the Initial Concentrations

# In[]:

AA = 148.5  # g Na2O/L
Sulf = 0.32  # Sulfdity(%) = Na2S/(NaOH+Na2S)

CS = C_S(AA, Sulf) / MM_Na2S  # Molar [mol/L]
OH = COH(AA, Sulf) / MM_NaOH  # Molar [mol/L]

L = numpy.array([0.27 for i in range(0, J)])
CC = numpy.array([0.776 for i in range(0, J)])
CA = numpy.array([OH] + [0 for i in range(1, J)])
# Let us plot our initial condition for confirmation:

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


# Solve the System Iteratively

# In[]:

SF = 1

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


# In[]

def temp(t):
    """ Temperature function
    """

    # Heating time in minutes
    th = 45
    # Starting temperature
    To = 273 + 25

    # Gradient at which heat changes degC/min
    # The gradient is specified such that a temp 
    # of 170 degC is reached at 'th'
    m = ((25 + 90 + 273) - To) / 120

    if t <= th:
        # Heating time
        T = To + t * m
    else:
        # Cooking time
        T = To + th * m

    return T


# In[]:

C_bulk = OH
SF2 = 0.001

L_record = []
CC_record = []
CA_record = []
CA_bulk_record = []

L_record.append(L)
CC_record.append(CC)
CA_record.append(CA)
for ti in range(1, N):
    TC = temp(ti)

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

    L_record.append(L)
    CC_record.append(CC)
    CA_record.append(CA)

# ### Plot the Numerical Solution

# Let us take a look at the numerical solution we attain after `N` time steps.

# In[]:

plot.xlabel('t')
plot.ylabel('concentration')

L_record = numpy.array(L_record)
CC_record = numpy.array(CC_record)
CA_record = numpy.array(CA_record)

plot.plot(t_grid[:, 0], L_record[:, 0])
# plot.plot(t_grid, CC_record[:,0])
plot.plot(t_grid[:, 0], CA_record[:, 0])

# In[]:

fig, ax = plot.subplots()
plot.xlabel('x')
plot.ylabel('t')
heatmap = ax.pcolor(x_grid, t_grid, CA_record, vmin=0., vmax=0.5)

plot.show()
