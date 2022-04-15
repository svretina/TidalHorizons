#!/usr/bin/env python

import numpy as np
from scipy import interpolate
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir
from tidalhorizons import global_vars as glb
from scipy.interpolate import SmoothSphereBivariateSpline as ssbs


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


## TODO more accurate interpolation
def rthetaphi(r, theta, phi, thetanew, phinew, s=1e-5):
    if theta.ndim > 1:
        theta = theta.ravel()
    if phi.ndim > 1:
        phi = phi.ravel()
    if r.ndim > 1:
        r = r.ravel()
    interpolator = ssbs(theta, phi, r, w=None, s=s)
    rnew = interpolator(thetanew, phinew, grid=False)
    if np.any(rnew) < 0:
        raise ValueError("radius cannot be negative")
    return rnew


class Horizon:
    def __init__(self, folder=None):
        if folder is None:
            horizon_path = glb.horizons_path
        else:
            horizon_path = f"{glb.horizons_path}/{folder}"
        self.hor = SimDir(horizon_path).horizons
        self.ahindex = self.hor.available_apparent_horizons[0]
        self.ah = self.hor.get_apparent_horizon(self.ahindex)

    @property
    def patches_cartesian_coordinates(self):
        (px, py, pz) = self.ah.shape_at_iteration(0)
        return px, py, pz

    @property
    def cartesian_coordinates(self):
        patches = self.patches_cartesian_coordinates
        shape_xyz = np.asarray(
            [np.concatenate([patch for patch in patches[dim]]) for dim in range(3)]
        )
        x = shape_xyz[0, :, :]
        y = shape_xyz[1, :, :]
        z = shape_xyz[2, :, :]
        return x, y, z

    @property
    def spherical_coordinates(self):
        x, y, z = self.cartesian_coordinates
        r, th, ph = cart_to_sph(x, y, z)
        return r, th, ph


if __name__ == "__main__":
    pass
if __name__ == "__main__":

    hor = Horizon()
    r, th, ph = hor.spherical_coordinates
    rprime = rthetaphi(r, th, ph, th, ph)
    # print(r - rprime)
    # exit()
    ## use interpolator here to get r' for θ',φ'
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y, z = hor.cartesian_coordinates
    ax.scatter(np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel(), "x")
    plt.show()
