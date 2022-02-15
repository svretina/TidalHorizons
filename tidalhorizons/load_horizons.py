#!/usr/bin/env python

import numpy as np
from scipy import interpolate
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir
from tidalhorizons import global_vars as glb
from scipy.interpolate import SmoothSphereBivariateSpline as ssbs
from scipy.interpolate import LSQSphereBivariateSpline as lsbs


def cart_to_sph(x, y, z):
    """Transforms cartesian coordinates to spherical
    theta \in [0, pi]
    phi \in [0, 2pi]
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x) + np.pi
    return r, theta, phi


def sph_to_cart(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def rthetaphi(r, theta, phi, thetanew, phinew, s=1e-5):
    if theta.ndim > 1:
        theta = theta.ravel()
    if phi.ndim > 1:
        phi = phi.ravel()
    if r.ndim > 1:
        r = r.ravel()
    interpolator = ssbs(theta, phi, r, s=s)
    rnew = interpolator(thetanew, phinew, grid=False)
    return rnew


class Horizon:
    def __init__(self):
        self.hor = SimDir(glb.horizons_path).horizons
        self.ahindex = self.hor.available_apparent_horizons[0]
        self.ah = self.hor.get_apparent_horizon(self.ahindex)

    def get_cartesian_coordinates(self):
        (px, py, pz) = self.ah.shape_at_iteration(0)
        return px, py, pz

    def get_spherical_coordinates(self):
        patches = self.get_cartesian_coordinates()
        shape_xyz = np.asarray(
            [
                np.concatenate([patch for patch in patches[dim]])
                for dim in range(3)
            ]
        )
        x = shape_xyz[0, :, :]
        y = shape_xyz[1, :, :]
        z = shape_xyz[2, :, :]
        r, th, ph = cart_to_sph(x, y, z)
        return r, th, ph


if __name__ == "__main__":

    hor = Horizon()

    r, th, ph = hor.get_spherical_coordinates()
    rprime = rthetaphi(r, th, ph, th, ph)
    print(r - rprime)
    exit()
    ## try the bisplrep function again
    ## use interpolator here to get r' for θ',φ'
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), "x")
    plt.show()
    exit()

    # r, theta, phi = hor.get_spherical_coordinates()
    gf = SimDir(glb.horizons_path).gf
    vars3D = gf.xyz
    var = vars3D.fields.alp[0][0][0]
