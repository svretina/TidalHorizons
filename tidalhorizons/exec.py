#!/usr/bin/env python3

import numpy as np
from kuibit.simdir import SimDir
from tidalhorizons import spharm as sp
from tidalhorizons import global_vars as glb
from tidalhorizons import load_horizons as lh


## Generate the Gauss Legendre grid (θ, φ)
lmax = 120
thetagl, phigl = sp.GaussLegendreMesh(
    lmax, coords="spherical", radians=True
)

## Get the coordinates of the horizon in spherical coordinates
hor = lh.Horizon()
r, theta, phi = hor.get_spherical_coordinates()


## interpolate r values of the horizon to the thetagl, phigl grid
rgl = lh.rthetaphi(r, theta, phi, thetagl, phigl)

## Transform the GL grid to Cartesian coordinates
x, y, z = lh.sph_to_cart(rgl, thetagl, phigl)

## Load some GridFunction
gf = SimDir(glb.horizons_path).gf
vars3D = gf.xyz
var = vars3D.fields.alp[0][0][0]

## Evaluate the GridFunction at the cartesian GL grid
vargl = np.empty_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vargl[i,j] = var((x[i,j], y[i,j], z[i,j]))

coefs = sp.decompose(vargl, lmax=lmax)
print(vargl.shape)
print(coefs.shape)
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# xh, yh, zh = hor.get_cartesian_coordinates()
# ax.scatter(xh, yh, zh, color="grey", marker="o", alpha=0.2, s=100)
# ax.scatter(x, y, z, color="r", marker="^", s=2)
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_zlim(-2,2)
# plt.show()
