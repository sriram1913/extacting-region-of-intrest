import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.special as sp
import math
from matplotlib import cm, colors
import plotly.graph_objects as go

n = int(input("enter principal quantum number:"))
l = int(input("enter azimuthal quantum number:"))
m = int(input("enter magnetic quantum number:"))

dz = 200
zmin = -6 * (n + l)
zmax = 6* (n + l)
x = numpy.linspace(zmin, zmax, dz)
y = numpy.linspace(zmin, zmax, dz)
z = numpy.linspace(zmin, zmax, dz)
X, Y, Z = numpy.meshgrid(x, y,z)

#data = hydrogen_wf(n, l, m, X, Y, Z)
R=numpy.sqrt(X**2+Y**2+Z**2)
data=sp.genlaguerre(n,l)(R) *numpy.exp(-1*R/(n))
data = abs(data) ** 2
k=data.max()
fig = go.Figure(data=go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=data.flatten(),
    isomin=0,
    isomax=k,
    opacity=0.1,
    surface_count=25,
))

fig.show()
