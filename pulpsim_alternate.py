from __future__ import division
import numpy
from numpy import multiply, add, power, exp
from matplotlib import pyplot as plot

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

# In[2]:

T = 200
N = 1000
dt = float(T) / float(N - 1)
t_grid = numpy.array([n * dt for n in range(N)])

# ### Specify System Parameters and the Reaction Term

# State initial starting conditions

# In[3]:

D = 1.1952024623899164e-05 * 10  # diffusion rate (only hydroxide)

sigma = float(D * dt) / float((2. * dx * dx))

# Specify the Initial Concentrations

# In[4]:


L = numpy.array([0.27 for i in range(0, J)])
CC = numpy.array([0.776 for i in range(0, J)])
CA = numpy.array([1.5625] + [0 for i in range(1, J)])
C = numpy.array([L, CC, CA])
# Let us plot our initial condition for confirmation:

# In[5]:

# plot.xlabel('x')
# plot.ylabel('concentration')
# plot.plot(x_grid, L)
# plot.plot(x_grid, C)
# plot.plot(x_grid, CA)

# Create Matrices

# The matrices that we need to construct are all tridiagonal so they are easy to construct with

# In[6]:

A_L = numpy.identity(J)
B_L = A_L
A_C = numpy.identity(J)
B_C = A_C

A_CA = numpy.diagflat([-sigma for i in range(J - 1)], -1) + numpy.diagflat(
    [1. + sigma] + [1. + 2. * sigma for i in range(J - 2)] + [1. + sigma]) + numpy.diagflat(
    [-sigma for i in range(J - 1)], 1)

B_CA = numpy.diagflat([sigma for i in range(J - 1)], -1) + numpy.diagflat(
    [1. - sigma] + [1. - 2. * sigma for i in range(J - 2)] + [1. - sigma]) + numpy.diagflat(
    [sigma for i in range(J - 1)], 1)

# Solve the System Iteratively

# In[21]:

CS = 1.5625
SF = 1


def f_vec(C, TC):
    L, CC, CA = C

    C[C < 0] = 0

    if L[0] >= .25:

        dLdt = lambda L, CC, CA, TC: dt * (36.2 * TC ** 0.5 * exp(-4807.69 / TC)) * L
        dCCdt = lambda L, CC, CA, TC: dt * 2.53 * 36.2 * T ** 0.5 * exp(-4807.69 / TC) * L * (CA ** 0.11)
        dCAdt = lambda L, CC, CA, TC: dt * SF * (
            (-4.78e-3 * 36.2 * T ** 0.5 * exp(-4807.69 / TC)) * L + 1.81e-2 * 2.53 *
            36.2 * T ** 0.5 * exp(-4807.69 / TC) * L * (CA ** 0.11))

    elif L[0] >= .025:

        dLdt = lambda L, CC, CA, TC: dt * (
            exp(35.19 - 17200 / TC) * CA + (exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4))) * L
        dCCdt = lambda L, CC, CA, TC: dt * (
            0.47(exp(35.19 - 17200 / TC) * CA + (exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4)))) * L
        dCAdt = lambda L, CC, CA, TC: dt * SF * (-4.78e-3 * (exp(35.19 - 17200 / TC) * CA * L +
                                                             exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4)*L)
                                                 + 1.81e-2 * 0.47 * (exp(35.19 - 17200 / TC) * CA* L + (
            exp(29.23 - 14400 / TC) * (CA ** 0.5) * (CS ** 0.4)*L)))

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
    th = 120
    # Starting temperature
    To = 273 + 80

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


# In[]:

L_record = []
CC_record = []
CA_record = []

L_record.append(L)
CC_record.append(CC)
CA_record.append(CA)
for ti in range(1, N):
    TC = temp(ti)

    vec_L, vec_CC, vec_CA = f_vec(C, TC)

    L_new = numpy.linalg.solve(A_L, B_L.dot(L) - vec_L(L, CC, CA, TC))
    CC_new = numpy.linalg.solve(A_C, B_C.dot(CC) - vec_CC(L, CC, CA, TC))
    CA_new = numpy.linalg.solve(A_CA, B_CA.dot(CA) - vec_CA(L, CC, CA, TC))

    L = L_new
    CC = CC_new
    CA = CA_new
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

plot.plot(t_grid, L_record[:, 0])
# plot.plot(t_grid, CC_record[:,0])
plot.plot(t_grid, CA_record[:, 0])

# In[]:

fig, ax = plot.subplots()
plot.xlabel('x')
plot.ylabel('t')
heatmap = ax.pcolor(x_grid, t_grid, CA_record, vmin=0., vmax=1.5)
plot.show()