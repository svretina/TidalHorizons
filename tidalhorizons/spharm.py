#!/usr/bin/env python

import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt


def GaussLegendreMesh(lmax, coords="geo", radians=False):
    latitude, longitude = pysh.expand.GLQGridCoord(
        lmax=lmax, extend=False
    )
    if coords == "geo":
        latitute, longitude = np.meshgrid(latitude, longitude)
        return latitude, longitude
    elif coords == "spherical":
        theta = 90 - latitude
        phi = longitude
        if radians:
            theta = np.radians(theta)
            phi = np.radians(phi)
        theta, phi = np.meshgrid(theta, phi)
        return theta, phi


def Spherical_Harmonic(l, m, lmax, kind="real"):
    coeffs = pysh.SHCoeffs.from_zeros(
        lmax, normalization="ortho", csphase=-1, kind=kind
    )
    coeffs.set_coeffs(values=[1], ls=[l], ms=[m])
    grid = coeffs.expand(grid="GLQ")
    return grid


def plot_SH(SH, show=True):
    fig, ax = SH.plot3d(show=True)
    ax.set_box_aspect([1, 1, 1])
    if show:
        plt.show()


def decompose(data, lmax=None):
    if isinstance(
        data,
        (
            pysh.shclasses.GLQComplexGrid,
            pysh.shclasses.GLQRealGrid,
            pysh.shclasses.DHComplexGrid,
            pysh.shclasses.DHRealGrid,
        ),
    ):
        cilm = data.expand(normalization="ortho", csphase=-1)
        return cilm.coeffs
    else:
        zero, w = pysh.expand.SHGLQ(lmax=lmax)
        if np.iscomplexobj(data):
            try:
                cilm = pysh.expand.SHExpandGLQC(
                    data.T,
                    w,
                    zero,
                    norm=4,
                    csphase=-1,
                    lmax_calc=lmax,
                )
            except:
                cilm = pysh.expand.SHExpandGLQC(
                    data, w, zero, norm=4, csphase=-1, lmax_calc=lmax
                )
        else:
            try:
                cilm = pysh.expand.SHExpandGLQ(
                    data.T,
                    w,
                    zero,
                    norm=4,
                    csphase=-1,
                    lmax_calc=lmax,
                )
            except:
                cilm = pysh.expand.SHExpandGLQ(
                    data, w, zero, norm=4, csphase=-1, lmax_calc=lmax
                )
        cilm[np.abs(cilm) < 1e-10] = 0
        return cilm


if __name__ == "__main__":
    lmax = 2
    theta, phi = GaussLegendreMesh(
        lmax, coords="spherical", radians=True
    )
    l = 1
    m = 0
    # n = 0
    # print(np.degrees(2*np.pi*n+np.arccos(-1/np.sqrt(3))))
    # print(np.degrees(2*np.pi*n+np.arccos(1/np.sqrt(3))))
    import scipy

    ylm = scipy.special.sph_harm(m, l, phi, theta)
    Ylm = Spherical_Harmonic(l, m, lmax, "complex")
    Ylm2 = Ylm * Spherical_Harmonic(2, m, lmax, "complex")
    print(Ylm2.data.sum())
    exit()
    coefs = decompose(ylm)
    print(coefs)

    Coefs = decompose(Ylm)
    print(Coefs)
