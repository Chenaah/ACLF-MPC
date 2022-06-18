from cProfile import label
from cmath import sin
import math
from turtle import color, pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pdb
#  m1=2.2, m2=2.5, k1=1.1, k2=1.5, b=1.4
def system_2_15(a, t, m1, m2, k1, k2, b):
    """ 2 mass translational system ex.2.15 """
    if t <= 0.1:
        f = 1000
    else:
        f = 0
    return np.array([a[2],
                     a[3],
                     (-k1 * (a[0] - a[1]) - b * (a[2] - a[3])) * (1 / m1),
                     ((-k2 * a[1]) - k1 * (a[1] - a[0]) - b * (a[3] - a[2]) + f) * (1 / m2)])


# m1=2, m2=3, k1=2, k2=1, b1=1.9, b2=1.5
def system_2_1(a, t, m1, m2, k1, k2, b1, b2):
    """ 2 mass translational system ex.2.1 """
    if t < 0:
        f = 0
    else:
        f = 1
    return np.array([a[2],
                     a[3],
                     (-(k1 + k2) * a[0] - b1 * a[2] + f + k2 * a[1]) * (1 / m1),
                     (-k2 * a[1] + k2 * a[0] - b2 * a[3]) * (1 / m2)])

# ==============================================================
# simulation harness

# time step
h = 0.01

# simulation time
time = np.arange(0, 40, h)

# initial conditions
y0 = np.array([0, 0, 0, 0])

# initialize yk
yk = y0

# solve system of equations args=(m1, m2, k1, k2, b1, b2)
y = odeint(system_2_1, y0, time, args=(1.2, 1.2, 0.9, 0.4, 0.8, 0.4))

# convert list to numpy array
y = np.array(y)

# ==============================================================
# plot result

fig, ax = plt.subplots()
ax.plot(time, y[:, 0])
ax.plot(time, y[:, 1])

plt.show()