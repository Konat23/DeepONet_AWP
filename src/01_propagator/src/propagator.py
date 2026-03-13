# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:34:21 2021

@author: Alejandra Hernandez
"""
import numpy as np
from numba import njit  # Cambiado a njit para forzar modo nopython
from .get_CPML import (
    get_CPML,
)  # Asegúrate de que esta función también sea compatible con Numba


@njit
def propagator(m, src, Ix0, Iz0, dx, dz, dt, max_offset, frec, CPMLimit):
    """
    Simulates wave propagation in a 2D medium using finite differences and
    Convolutional Perfectly Matched Layer (C-PML) boundary conditions.
    """
    m = np.ascontiguousarray(np.transpose(m))  # Más eficiente para Numba
    max_ix = max_offset / dx
    Nx, Nz = m.shape
    Nt = len(src)

    # initialize arrays con tipos explícitos
    P = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    P_tmp = np.zeros((Nx, Nz), dtype=np.float64)
    P_Iz0 = np.zeros((Nt, Nx), dtype=np.float64)
    d2P_dt2 = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    dP_dx = np.zeros((Nx, Nz), dtype=np.float64)
    dP_dz = np.zeros((Nx, Nz), dtype=np.float64)
    F_dPdx = np.zeros((Nx, Nz), dtype=np.float64)
    F_dPdz = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdx2 = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdz2 = np.zeros((Nx, Nz), dtype=np.float64)
    v2 = m**2

    one_over_dx2 = 1.0 / dx**2  # Usar 1.0 para asegurar float
    one_over_dz2 = 1.0 / dz**2
    one_over_dx = 1.0 / dx
    one_over_dz = 1.0 / dz
    Vmax = np.max(m)
    Vmin = np.min(m)
    w_l_ = Vmin / frec
    Points_per_wavelength = w_l_ / dx
    Courant_number = Vmax * dt / dx * np.sqrt(2.0)

    # Functions for C-PML conditions
    R = 1e-3
    Vcpml = Vmax
    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec
    )

    for it in range(1, Nt - 1):
        # Snapshot
        P_tmp[:, :] = P[:, :, it]

        # Free-surface conditions
        d2P_dt2[:, 1, it] = 0.0

        # spatial finite differences
        for ix in range(CPMLimit + 1, Nx - CPMLimit):
            for iz in range(2, Nz - CPMLimit):
                d2P_dx2 = (
                    P_tmp[ix + 1, iz] + P_tmp[ix - 1, iz] - 2.0 * P_tmp[ix, iz]
                ) * one_over_dx2
                d2P_dz2 = (
                    P_tmp[ix, iz + 1] + P_tmp[ix, iz - 1] - 2.0 * P_tmp[ix, iz]
                ) * one_over_dz2
                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        P_tmp[Nx - 2, :] = 0.0
        P_tmp[1, :] = 0.0
        P_tmp[:, Nz - 2] = 0.0

        # C-PML conditions
        for ix in range(1, CPMLimit + 1):
            for iz in range(1, Nz - 1):
                dP_dx[ix, iz] = (P_tmp[ix + 1, iz] - P_tmp[ix, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz + 1] - P_tmp[ix, iz]) * one_over_dz

                F_dPdx[ix, iz] = (
                    F_dPdx[ix, iz] * b_x_half[ix] + a_x_half[ix] * dP_dx[ix, iz]
                )
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]
                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz[ix, iz]
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix, iz] - dP_dx[ix - 1, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz] - dP_dz[ix, iz - 1])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]
                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]
                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        for ix in range(Nx - 2, Nx - CPMLimit - 2, -1):
            for iz in range(1, Nz - 1):
                dP_dx[ix, iz] = (P_tmp[ix, iz] - P_tmp[ix - 1, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz + 1] - P_tmp[ix, iz]) * one_over_dz

                F_dPdx[ix, iz] = (
                    F_dPdx[ix, iz] * b_x_half[ix - 1] + a_x_half[ix - 1] * dP_dx[ix, iz]
                )
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]
                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz[ix, iz]
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix + 1, iz] - dP_dx[ix, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz] - dP_dz[ix, iz - 1])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]
                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]

                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        for ix in range(CPMLimit + 1, Nx - CPMLimit - 1):
            for iz in range(Nz - 2, Nz - CPMLimit - 2, -1):
                dP_dx[ix, iz] = (P_tmp[ix + 1, iz] - P_tmp[ix, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz] - P_tmp[ix, iz - 1]) * one_over_dz

                F_dPdx[ix, iz] = F_dPdx[ix, iz] * b_x[ix] + a_x[ix] * dP_dx[ix, iz]
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]
                F_dPdz[ix, iz] = (
                    F_dPdz[ix, iz] * b_z_half[iz - 1] + a_z_half[iz - 1] * dP_dz[ix, iz]
                )
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix, iz] - dP_dx[ix - 1, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz + 1] - dP_dz[ix, iz])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]
                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]

                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        # Time integration
        P[:, :, it + 1] = (
            dt**2 * v2 * d2P_dt2[:, :, it] + 2.0 * P[:, :, it] - P[:, :, it - 1]
        )

        # Source injection
        P[Ix0, Iz0, it + 1] += src[it + 1]

        # Surface component selection
        P_Iz0[it, :] = P[:, Iz0, it]

    # Cálculo de los índices de salida
    start_idx = int(max(21, Ix0 - max_ix))
    end_idx = int(min(Nx - 20, Ix0 + max_ix)) + 1
    o = np.arange(start_idx, end_idx)
    o2 = end_idx - start_idx

    Pt_slice = P[o, Iz0, :]
    Pt_reshaped = Pt_slice.reshape((o2, Nt))
    Pt = np.ascontiguousarray(Pt_reshaped.T)

    return Pt, P, d2P_dt2
