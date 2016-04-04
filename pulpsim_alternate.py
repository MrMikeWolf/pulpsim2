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

T = 200
N = 1000
dt = float(T) / float(N - 1)
t_grid = numpy.array([n * dt for n in range(N)])

x_grid, t_grid = numpy.meshgrid(x_grid, t_grid)


# ### Specify System Parameters and the Reaction Term

# State initial starting conditions

# In[]:

def sigma_D(T):
    D = 1.5e-2 * (T ** 0.5) * exp(-4870 / 1.9872 / T)
    return D * dt / (2 * dx * dx)


# Specify the Initial Concentrations

# In[]:

def initial(xi):    
    xi /=100    
    return numpy.array([xi for i in range(0,J)])

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

L = numpy.array([L1,L2,L3])
C = numpy.array([C1,C2,C3])
G = numpy.array([G1,G2,G3])
X = numpy.array([X1,X2,X3])
CA = initial(0.1)

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

CS = 0.05
SF = 1

def Lignin(A,Ea,a,b,L,CA,CS,T):
    R = 8.314
    return -A*exp(Ea/R*(1/443.15 - 1/T))*(CA**a)*(CS**b)*L

def Carbo(A,Ea,a,b,k2,C,CA,CS,T):
    R = 8.314
    return -A*exp(Ea/R*(1/443.15 - 1/T))*((CA**a)*(CS**b) + k2)*C




def f_vec(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC):  
    
    # Lignin consumption
    
    dL1dt = Lignin(0.1,50000,0,0.06,L1,CA,CS,TC)
    dL2dt = Lignin(0.1,127000,0.48,0.39,L2,CA,CS,TC)
    dL3dt = Lignin(0.0047,127000,0.2,0,L3,CA,CS,TC)
    dLdt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: dt *(dL1dt + dL2dt + dL3dt)

    # Carbohydrate consumption
        # 1) Cellulose

    dC1dt = Carbo(0.06,50000,0.1,0,0,C1,CA,CS,TC)
    dC2dt = Carbo(0.054,144000,1,0,0.22,C2,CA,CS,TC)
    dC3dt = Carbo(6.4e-4,144000,1,0,0.42,C3,CA,CS,TC)
    
    dCdt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: dt * (dC1dt + dC2dt + dC3dt)

        # 2) Glucomannan

    dG1dt = Carbo(0.06,50000,0.1,0,0,G1,CA,CS,TC)
    dG2dt = Carbo(0.054,144000,1,0,0.22,G2,CA,CS,TC)
    dG3dt = Carbo(6.4e-4,144000,1,0,0.42,G3,CA,CS,TC)
    
    dGdt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: dt * (dG1dt + dG2dt + dG3dt)

        # 3) Xylan

    dX1dt = Carbo(0.06,50000,0.1,0,0,X1,CA,CS,TC)
    dX2dt = Carbo(0.054,144000,1,0,0.22,X2,CA,CS,TC)
    dX3dt = Carbo(6.4e-4,144000,1,0,0.42,X3,CA,CS,TC)
    
    dXdt = lambda L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC: dt * (dX1dt + dX2dt + dX3dt)

       

    return dLdt, dCdt, dGdt, dXdt


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

C_bulk = 0.5
SF2 = 0.001

L_record = []
C_record = []
G_record = []
X_record = []

CA_bulk_record = []

L_record.append(L)
C_record.append(C)
G_record.append(G)
X_record.append(X)

for ti in range(1, N):
    TC = temp(ti)
    sigma = sigma_D(TC)
    A_CA = ACA(sigma)
    B_CA = BCA(sigma)

    L1,L2,L3 = L
    C1,C2,C3 = C
    G1,G2,G3 = G    
    X1,X2,X3 = X

    vec_L, vec_C, vec_G, vec_X = f_vec(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC)
#    vec_L2, vec_C2, vec_G2, vec_X2 = f_vec(L2, C2, G2, X2, CA, TC)
#    vec_L3, vec_C3, vec_G3, vec_X3 = f_vec(L3, C3, G3, X3, CA, TC)


    L_new = numpy.linalg.solve(A_L, B_L.dot(L1) - vec_L(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
#    L2_new = numpy.linalg.solve(A_L, B_L.dot(L2) - vec_L2(L2, C, G, X, CA, TC))
#    L3_new = numpy.linalg.solve(A_L, B_L.dot(L3) - vec_L3(L3, C, G, X, CA, TC))
    
    
    C_new = numpy.linalg.solve(A_C, B_C.dot(C1) - vec_C(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
#    C2_new = numpy.linalg.solve(A_L, B_L.dot(C2) - vec_C2(L, C2, G, X, CA, TC))
#    C3_new = numpy.linalg.solve(A_L, B_L.dot(C3) - vec_C3(L, C3, G, X, CA, TC))
    
    
    G_new = numpy.linalg.solve(A_C, B_C.dot(G1) - vec_G(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
#    G2_new = numpy.linalg.solve(A_C, B_C.dot(G2) - vec_G2(L, C, G2, X, CA, TC))   
#    G3_new = numpy.linalg.solve(A_C, B_C.dot(G3) - vec_G3(L, C, G3, X, CA, TC))
    
    
    X_new = numpy.linalg.solve(A_C, B_C.dot(X1) - vec_X(L1,L2,L3, C1,C2,C3, G1,G2,G3, X1,X2,X3, CA, TC))
#    X2_new = numpy.linalg.solve(A_C, B_C.dot(X2) - vec_X2(L, C, G, X2, CA, TC))
#    X3_new = numpy.linalg.solve(A_C, B_C.dot(X3) - vec_X3(L, C, G, X3, CA, TC))

    val = (dt / dx) * SF2 * (CA[1] - CA[0])
    C_bulk += val
    CA_bulk_record.append(C_bulk)

    L = L_new
#    L2 = L2_new
#    L3 = L3_new
    
    C = C_new
#    C2 = C2_new
#    C3 = C3_new

    G = G_new
#    G2 = G2_new
#    G3 = G3_new
    
    X = X_new
#    X2 = X2_new
#    X3 = X3_new

    # The liquor penetration rate is infinite (CAi bulk = CAi of first compartment)

    L = numpy.array([L1,L2,L3])
    C = numpy.array([C1,C2,C3])
    G = numpy.array([G1,G2,G3])
    X = numpy.array([X1,X2,X3])
    
    L_record.append(L)
    C_record.append(C)
    G_record.append(G)
    X_record.append(X)

# ### Plot the Numerical Solution

# Let us take a look at the numerical solution we attain after `N` time steps.

# In[]:

plot.xlabel('t')
plot.ylabel('concentration')

L_record = numpy.array(L_record)
C_record = numpy.array(C_record)
G_record = numpy.array(G_record)
X_record = numpy.array(X_record)

plot.plot(t_grid[:, 0], L_record[:, 0])
# plot.plot(t_grid, CC_record[:,0])
#plot.plot(t_grid[:, 0], CA_record[:, 0])

# In[]:

#fig, ax = plot.subplots()
#plot.xlabel('x')
#plot.ylabel('t')
#heatmap = ax.pcolor(x_grid, t_grid, CA_record, vmin=0., vmax=0.5)

plot.show()