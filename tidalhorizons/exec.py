#!/usr/bin/env python3

import numpy as np
from kuibit.simdir import SimDir
from tidalhorizons import spharm as sp
from tidalhorizons import global_vars as glb
from tidalhorizons import load_horizons as lh


## Generate the Gauss Legendre grid (θ, φ)
lmax = 20
thetagl, phigl = sp.GaussLegendreMesh(lmax, coords="spherical", radians=True)

## Get the coordinates of the horizon in spherical coordinates
hor = lh.Horizon("ks-mclachlan")
r, theta, phi = hor.spherical_coordinates

## interpolate r values of the horizon to the thetagl, phigl grid
rgl = lh.rthetaphi(r, theta, phi, thetagl, phigl)

## Transform the GL grid to Cartesian coordinates
xgl, ygl, zgl = lh.sph_to_cart(rgl, thetagl, phigl)
# ax.scatter(x, y, z, color="r", marker="^", s=2)

## Load some GridFunction
horizon_path = f"{glb.horizons_path}/ks-mclachlan"
gf = SimDir(horizon_path).gf

vars2D = gf.xy
var = vars2D.fields.qlm_psi2[0][0].merge_refinement_levels(resample=True)

xvar = vars2D.fields.qlm_x[0][0].merge_refinement_levels(resample=True)
yvar = vars2D.fields.qlm_y[0][0].merge_refinement_levels(resample=True)
zvar = vars2D.fields.qlm_z[0][0].merge_refinement_levels(resample=True)
rvar, thvar, phvar = lh.cart_to_sph(xvar.data, yvar.data, zvar.data)

real_part = np.empty(var.data.shape)
imag_part = np.empty(var.data.shape)
for i in range(var.data.shape[0]):
    for j in range(var.data.shape[1]):
        real_part[i, j] = var.data[i, j][0]
        imag_part[i, j] = var.data[i, j][1]


# Evaluate the GridFunction at the cartesian GL grid
rvargl = np.empty(xgl.shape)
ivargl = np.empty(xgl.shape)

for i in range(xgl.shape[0]):
    for j in range(xgl.shape[1]):
        rvargl[i, j] = lh.rthetaphi(real_part, thvar, phvar, thetagl[i, j], phigl[i, j])
        ivargl[i, j] = lh.rthetaphi(imag_part, thvar, phvar, thetagl[i, j], phigl[i, j])
vargl = rvargl + 1j * ivargl
# print(rvar)
coefs = sp.decompose(vargl, lmax=lmax)
print(coefs[0])

# import matplotlib.pyplot as plt
# from matplotlib import cm, colors

# # from sklearn import preprocessing
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")


# # norm = cm.colors.Normalize(vmax=abs(vargl).max(), vmin=abs(vargl).min())
# # fcolors = vargl
# # ax.plot_surface(
# #     xgl, ygl, zgl, rstride=1, cstride=1, norm=norm, facecolors=cm.seismic(fcolors)
# # )
# # ax.scatter(xgl, ygl, zgl, color="g", marker="x", s=2)

# norm = cm.colors.Normalize(vmax=abs(real_part).max(), vmin=abs(real_part).min())
# fcolors = real_part
# ax.plot_surface(
#     xvar.data,
#     yvar.data,
#     zvar.data,
#     rstride=1,
#     cstride=1,
#     norm=norm,
#     facecolors=cm.seismic(fcolors),
# )
# ax.scatter(xvar.data, yvar.data, zvar.data, color="r", marker="^", s=2)


# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)
# plt.show()
