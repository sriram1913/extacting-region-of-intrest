import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import math
from matplotlib import cm, colors
import plotly.graph_objects as go


def psi_R(r, n, l):
    coeff = np.sqrt((2.0 / n) ** 3 * sp.factorial(n - l - 1) / (2.0 * n * sp.factorial(n + l)))

    laguerre = sp.assoc_laguerre(2.0 * r / n, n - l - 1, 2 * l + 1)

    return coeff * np.exp(-r / n) * (2.0 * r / n) ** l * laguerre


def psi_ang(theta, phi, l, m):
    sphHarm = sp.sph_harm(m, l,theta,phi)

    return sphHarm.real


def HFunc(r, theta, phi, n, l, m):
    return psi_R(r, n, l) * psi_ang(theta, phi, l, m)


def func(n,l,m):
    """
    n = int(input("enter principal quantum number:"))
    l = int(input("enter azimuthal quantum number:"))
    m = int(input("enter magnetic quantum number:")) """

    limit = 4 * (n + l)

    x_1d = np.linspace(-limit, limit, 100)
    z_1d = np.linspace(-limit, limit, 100)
    y_1d = np.linspace(-limit, limit, 100)

    x, y, z = np.meshgrid(x_1d, y_1d, z_1d)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2),z)
    theta = np.arctan2(y, x) +np.pi

    print(n ,l,m)

    #psi_nlm = HFunc(r,theta,phi,n, l,m)
    psi_nlm=psi_R(r,n,l)
    psi_nlm = abs(psi_nlm) ** 2
    k = psi_nlm.max()


    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=psi_nlm.flatten(),
        isomin=0,
        isomax=k,
        opacity=0.1,
        surface_count=25,
    ))
    fig.update_layout(
        title="Radial plot for n="+str(n)+",l="+str(l)+",m="+str(m))
    fig.show()

    fig.write_image("Radial plot for n="+str(n)+",l="+str(l)+",m="+str(m)+".png")

'''
for n in range(1,4):
    for l in range(0,n):
        for m in range(-l,l+1):
            func(n,l,m)
'''
#for n in range(1,4):
 #   for l in range(0,n):
func(5,4,0)



#print(psi_R())
