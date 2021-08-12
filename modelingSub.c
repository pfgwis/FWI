#include <string.h>

#include "xdfwi.h"

extern Param_t *Param;

void cpu_scale_df(int iStage) {
    int iX, iY, iZ, iXYZ;

    for (iZ = 0; iZ < Param->zNum - Param->pmlNum; iZ++) {
        for (iY = Param->pmlNum; iY < Param->yNum - Param->pmlNum; iY++) {
            for (iX = Param->pmlNum; iX < Param->xNum - Param->pmlNum; iX++) {
                iXYZ = iZ * Param->xyNum + iY * Param->xNum + iX;

                Param->dsxx[iXYZ] *= Param->A[iStage];
                Param->dsxy[iXYZ] *= Param->A[iStage];
                Param->dsxz[iXYZ] *= Param->A[iStage];
                Param->dsyy[iXYZ] *= Param->A[iStage];
                Param->dsyz[iXYZ] *= Param->A[iStage];
                Param->dszz[iXYZ] *= Param->A[iStage];
                Param->dvx[iXYZ]  *= Param->A[iStage];
                Param->dvy[iXYZ]  *= Param->A[iStage];
                Param->dvz[iXYZ]  *= Param->A[iStage];
            }
        }
    }

    for (iXYZ = 0; iXYZ < Param->pmlNumTotal; iXYZ++) {
        Param->dsxx_xpml[iXYZ] *= Param->A[iStage];
        Param->dsxy_xpml[iXYZ] *= Param->A[iStage];
        Param->dsxz_xpml[iXYZ] *= Param->A[iStage];
        Param->dsyy_xpml[iXYZ] *= Param->A[iStage];
        Param->dszz_xpml[iXYZ] *= Param->A[iStage];
        Param->dvx_xpml[iXYZ]  *= Param->A[iStage];
        Param->dvy_xpml[iXYZ]  *= Param->A[iStage];
        Param->dvz_xpml[iXYZ]  *= Param->A[iStage];

        Param->dsxx_ypml[iXYZ] *= Param->A[iStage];
        Param->dsxy_ypml[iXYZ] *= Param->A[iStage];
        Param->dsyy_ypml[iXYZ] *= Param->A[iStage];
        Param->dsyz_ypml[iXYZ] *= Param->A[iStage];
        Param->dszz_ypml[iXYZ] *= Param->A[iStage];
        Param->dvx_ypml[iXYZ]  *= Param->A[iStage];
        Param->dvy_ypml[iXYZ]  *= Param->A[iStage];
        Param->dvz_ypml[iXYZ]  *= Param->A[iStage];

        Param->dsxx_zpml[iXYZ] *= Param->A[iStage];
        Param->dsxz_zpml[iXYZ] *= Param->A[iStage];
        Param->dsyy_zpml[iXYZ] *= Param->A[iStage];
        Param->dsyz_zpml[iXYZ] *= Param->A[iStage];
        Param->dszz_zpml[iXYZ] *= Param->A[iStage];
        Param->dvx_zpml[iXYZ]  *= Param->A[iStage];
        Param->dvy_zpml[iXYZ]  *= Param->A[iStage];
        Param->dvz_zpml[iXYZ]  *= Param->A[iStage];
    }
}

void cpu_update_f(int iStage) {
    int iX, iY, iZ, iXYZ;

    for (iZ = 0; iZ < Param->zNum - Param->pmlNum; iZ++) {
        for (iY = Param->pmlNum; iY < Param->yNum - Param->pmlNum; iY++) {
            for (iX = Param->pmlNum; iX < Param->xNum - Param->pmlNum; iX++) {
                iXYZ = iZ * Param->xyNum + iY * Param->xNum + iX;

                Param->sxx[iXYZ] += Param->B[iStage] * Param->dsxx[iXYZ];
                Param->sxy[iXYZ] += Param->B[iStage] * Param->dsxy[iXYZ];
                Param->sxz[iXYZ] += Param->B[iStage] * Param->dsxz[iXYZ];
                Param->syy[iXYZ] += Param->B[iStage] * Param->dsyy[iXYZ];
                Param->syz[iXYZ] += Param->B[iStage] * Param->dsyz[iXYZ];
                Param->szz[iXYZ] += Param->B[iStage] * Param->dszz[iXYZ];
                Param->vx[iXYZ]  += Param->B[iStage] * Param->dvx[iXYZ];
                Param->vy[iXYZ]  += Param->B[iStage] * Param->dvy[iXYZ];
                Param->vz[iXYZ]  += Param->B[iStage] * Param->dvz[iXYZ];
            }
        }
    }

    for (iXYZ = 0; iXYZ < Param->pmlNumTotal; iXYZ++) {
        Param->sxx_xpml[iXYZ] += Param->B[iStage] * Param->dsxx_xpml[iXYZ];
        Param->sxy_xpml[iXYZ] += Param->B[iStage] * Param->dsxy_xpml[iXYZ];
        Param->sxz_xpml[iXYZ] += Param->B[iStage] * Param->dsxz_xpml[iXYZ];
        Param->syy_xpml[iXYZ] += Param->B[iStage] * Param->dsyy_xpml[iXYZ];
        Param->szz_xpml[iXYZ] += Param->B[iStage] * Param->dszz_xpml[iXYZ];
        Param->vx_xpml[iXYZ]  += Param->B[iStage] * Param->dvx_xpml[iXYZ];
        Param->vy_xpml[iXYZ]  += Param->B[iStage] * Param->dvy_xpml[iXYZ];
        Param->vz_xpml[iXYZ]  += Param->B[iStage] * Param->dvz_xpml[iXYZ];

        Param->sxx_ypml[iXYZ] += Param->B[iStage] * Param->dsxx_ypml[iXYZ];
        Param->sxy_ypml[iXYZ] += Param->B[iStage] * Param->dsxy_ypml[iXYZ];
        Param->syy_ypml[iXYZ] += Param->B[iStage] * Param->dsyy_ypml[iXYZ];
        Param->syz_ypml[iXYZ] += Param->B[iStage] * Param->dsyz_ypml[iXYZ];
        Param->szz_ypml[iXYZ] += Param->B[iStage] * Param->dszz_ypml[iXYZ];
        Param->vx_ypml[iXYZ]  += Param->B[iStage] * Param->dvx_ypml[iXYZ];
        Param->vy_ypml[iXYZ]  += Param->B[iStage] * Param->dvy_ypml[iXYZ];
        Param->vz_ypml[iXYZ]  += Param->B[iStage] * Param->dvz_ypml[iXYZ];

        Param->sxx_zpml[iXYZ] += Param->B[iStage] * Param->dsxx_zpml[iXYZ];
        Param->sxz_zpml[iXYZ] += Param->B[iStage] * Param->dsxz_zpml[iXYZ];
        Param->syy_zpml[iXYZ] += Param->B[iStage] * Param->dsyy_zpml[iXYZ];
        Param->syz_zpml[iXYZ] += Param->B[iStage] * Param->dsyz_zpml[iXYZ];
        Param->szz_zpml[iXYZ] += Param->B[iStage] * Param->dszz_zpml[iXYZ];
        Param->vx_zpml[iXYZ]  += Param->B[iStage] * Param->dvx_zpml[iXYZ];
        Param->vy_zpml[iXYZ]  += Param->B[iStage] * Param->dvy_zpml[iXYZ];
        Param->vz_zpml[iXYZ]  += Param->B[iStage] * Param->dvz_zpml[iXYZ];
    }
}

void cpu_pml_combine() {
    int iPml, iXYZ;

    for (iPml = 0; iPml < Param->pmlNumTotal; iPml++) {
        iXYZ = iPml_to_iXYZ(iPml);

        Param->sxx[iXYZ] = Param->sxx_xpml[iPml] + Param->sxx_ypml[iPml] + Param->sxx_zpml[iPml];
        Param->sxy[iXYZ] = Param->sxy_xpml[iPml] + Param->sxy_ypml[iPml];
        Param->sxz[iXYZ] = Param->sxz_xpml[iPml] + Param->sxz_zpml[iPml];
        Param->syy[iXYZ] = Param->syy_xpml[iPml] + Param->syy_ypml[iPml] + Param->syy_zpml[iPml];
        Param->syz[iXYZ] = Param->syz_ypml[iPml] + Param->syz_zpml[iPml];
        Param->szz[iXYZ] = Param->szz_xpml[iPml] + Param->szz_ypml[iPml] + Param->szz_zpml[iPml];
        Param->vx[iXYZ]  = Param->vx_xpml[iPml]  + Param->vx_ypml[iPml]  + Param->vx_zpml[iPml];
        Param->vy[iXYZ]  = Param->vy_xpml[iPml]  + Param->vy_ypml[iPml]  + Param->vy_zpml[iPml];
        Param->vz[iXYZ]  = Param->vz_xpml[iPml]  + Param->vz_ypml[iPml]  + Param->vz_zpml[iPml];
    }
}

void cpu_x_derivative(int isForward) {
    int iX, iY, iZ, iXYZ, iPml, fdIndex, iIndex, iRegion;
    float dvxdx, dvydx, dvzdx, dsxxdx, dsxydx, dsxzdx;

    for (iZ = 0; iZ < Param->zNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                iXYZ = iZ * Param->xyNum + iY * Param->xNum + iX;

                dvxdx  = 0.f;
                dvydx  = 0.f;
                dvzdx  = 0.f;
                dsxxdx = 0.f;
                dsxydx = 0.f;
                dsxzdx = 0.f;

                if (isForward) {
                    for (fdIndex = -1; fdIndex <= 3; fdIndex++) {
                        if (iX + fdIndex < 0 || iX + fdIndex >= Param->xNum) {
                            continue;
                        }

                        iIndex = iXYZ + fdIndex;

                        dsxxdx += Param->sxx[iIndex] * Param->fd_x[fdIndex + 1];
                        dsxydx += Param->sxy[iIndex] * Param->fd_x[fdIndex + 1];
                        dsxzdx += Param->sxz[iIndex] * Param->fd_x[fdIndex + 1];
                        dvxdx  += Param->vx[iIndex]  * Param->fd_x[fdIndex + 1];
                        dvydx  += Param->vy[iIndex]  * Param->fd_x[fdIndex + 1];
                        dvzdx  += Param->vz[iIndex]  * Param->fd_x[fdIndex + 1];
                    }
                } else {
                    for (fdIndex = -3; fdIndex <= 1; fdIndex++) {
                        if (iX + fdIndex < 0 || iX + fdIndex >= Param->xNum) {
                            continue;
                        }

                        iIndex = iXYZ + fdIndex;

                        dsxxdx -= Param->sxx[iIndex] * Param->fd_x[1 - fdIndex];
                        dsxydx -= Param->sxy[iIndex] * Param->fd_x[1 - fdIndex];
                        dsxzdx -= Param->sxz[iIndex] * Param->fd_x[1 - fdIndex];
                        dvxdx  -= Param->vx[iIndex]  * Param->fd_x[1 - fdIndex];
                        dvydx  -= Param->vy[iIndex]  * Param->fd_x[1 - fdIndex];
                        dvzdx  -= Param->vz[iIndex]  * Param->fd_x[1 - fdIndex];
                    }
                }

                iRegion = get_pml_region_iXYZ(iX, iY, iZ);

                if (iRegion >= 0) {
                    iPml = iXYZ_to_iPml(iX, iY, iZ, iRegion);

                    Param->dvx_xpml[iPml]  += Param->rho[iXYZ] * dsxxdx;
                    Param->dvy_xpml[iPml]  += Param->rho[iXYZ] * dsxydx;
                    Param->dvz_xpml[iPml]  += Param->rho[iXYZ] * dsxzdx;
                    Param->dsxx_xpml[iPml] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvxdx;
                    Param->dsyy_xpml[iPml] += Param->lambda[iXYZ] * dvxdx;
                    Param->dszz_xpml[iPml] += Param->lambda[iXYZ] * dvxdx;
                    Param->dsxy_xpml[iPml] += Param->mu[iXYZ] * dvydx;
                    Param->dsxz_xpml[iPml] += Param->mu[iXYZ] * dvzdx;

                    Param->dvx_xpml[iPml]  -= Param->pml_dx[iX] * Param->vx_xpml[iPml];
                    Param->dvy_xpml[iPml]  -= Param->pml_dx[iX] * Param->vy_xpml[iPml];
                    Param->dvz_xpml[iPml]  -= Param->pml_dx[iX] * Param->vz_xpml[iPml];
                    Param->dsxx_xpml[iPml] -= Param->pml_dx[iX] * Param->sxx_xpml[iPml];
                    Param->dsyy_xpml[iPml] -= Param->pml_dx[iX] * Param->syy_xpml[iPml];
                    Param->dszz_xpml[iPml] -= Param->pml_dx[iX] * Param->szz_xpml[iPml];
                    Param->dsxy_xpml[iPml] -= Param->pml_dx[iX] * Param->sxy_xpml[iPml];
                    Param->dsxz_xpml[iPml] -= Param->pml_dx[iX] * Param->sxz_xpml[iPml];
                } else {
                    Param->dvx[iXYZ]  += Param->rho[iXYZ] * dsxxdx;
                    Param->dvy[iXYZ]  += Param->rho[iXYZ] * dsxydx;
                    Param->dvz[iXYZ]  += Param->rho[iXYZ] * dsxzdx;
                    Param->dsxx[iXYZ] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvxdx;
                    Param->dsyy[iXYZ] += Param->lambda[iXYZ] * dvxdx;
                    Param->dszz[iXYZ] += Param->lambda[iXYZ] * dvxdx;
                    Param->dsxy[iXYZ] += Param->mu[iXYZ] * dvydx;
                    Param->dsxz[iXYZ] += Param->mu[iXYZ] * dvzdx;
                }
            }
        }
    }
}

void cpu_y_derivative(int isForward) {
    int iX, iY, iZ, iXYZ, iPml, fdIndex, iIndex, iRegion;
    float dvxdy, dvydy, dvzdy, dsxydy, dsyydy, dsyzdy;

    for (iZ = 0; iZ < Param->zNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                iXYZ = iZ * Param->xyNum + iY * Param->xNum + iX;

                dvxdy  = 0.f;
                dvydy  = 0.f;
                dvzdy  = 0.f;
                dsxydy = 0.f;
                dsyydy = 0.f;
                dsyzdy = 0.f;

                if (isForward) {
                    for (fdIndex = -1; fdIndex <= 3; fdIndex++) {
                        if (iY + fdIndex < 0 || iY + fdIndex >= Param->yNum) {
                            continue;
                        }

                        iIndex = iXYZ + fdIndex * Param->xNum;

                        dsxydy += Param->sxy[iIndex] * Param->fd_y[fdIndex + 1];
                        dsyydy += Param->syy[iIndex] * Param->fd_y[fdIndex + 1];
                        dsyzdy += Param->syz[iIndex] * Param->fd_y[fdIndex + 1];
                        dvxdy  += Param->vx[iIndex]  * Param->fd_y[fdIndex + 1];
                        dvydy  += Param->vy[iIndex]  * Param->fd_y[fdIndex + 1];
                        dvzdy  += Param->vz[iIndex]  * Param->fd_y[fdIndex + 1];
                    }
                } else {
                    for (fdIndex = -3; fdIndex <= 1; fdIndex++) {
                        if (iY + fdIndex < 0 || iY + fdIndex >= Param->yNum) {
                            continue;
                        }

                        iIndex = iXYZ + fdIndex * Param->xNum;

                        dsxydy -= Param->sxy[iIndex] * Param->fd_y[1 - fdIndex];
                        dsyydy -= Param->syy[iIndex] * Param->fd_y[1 - fdIndex];
                        dsyzdy -= Param->syz[iIndex] * Param->fd_y[1 - fdIndex];
                        dvxdy  -= Param->vx[iIndex]  * Param->fd_y[1 - fdIndex];
                        dvydy  -= Param->vy[iIndex]  * Param->fd_y[1 - fdIndex];
                        dvzdy  -= Param->vz[iIndex]  * Param->fd_y[1 - fdIndex];
                    }
                }

                iRegion = get_pml_region_iXYZ(iX, iY, iZ);

                if (iRegion >= 0) {
                    iPml = iXYZ_to_iPml(iX, iY, iZ, iRegion);

                    Param->dvx_ypml[iPml]  += Param->rho[iXYZ] * dsxydy;
                    Param->dvy_ypml[iPml]  += Param->rho[iXYZ] * dsyydy;
                    Param->dvz_ypml[iPml]  += Param->rho[iXYZ] * dsyzdy;
                    Param->dsxx_ypml[iPml] += Param->lambda[iXYZ] * dvydy;
                    Param->dsyy_ypml[iPml] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvydy;
                    Param->dszz_ypml[iPml] += Param->lambda[iXYZ] * dvydy;
                    Param->dsxy_ypml[iPml] += Param->mu[iXYZ] * dvxdy;
                    Param->dsyz_ypml[iPml] += Param->mu[iXYZ] * dvzdy;

                    Param->dvx_ypml[iPml]  -= Param->pml_dy[iY] * Param->vx_ypml[iPml];
                    Param->dvy_ypml[iPml]  -= Param->pml_dy[iY] * Param->vy_ypml[iPml];
                    Param->dvz_ypml[iPml]  -= Param->pml_dy[iY] * Param->vz_ypml[iPml];
                    Param->dsxx_ypml[iPml] -= Param->pml_dy[iY] * Param->sxx_ypml[iPml];
                    Param->dsyy_ypml[iPml] -= Param->pml_dy[iY] * Param->syy_ypml[iPml];
                    Param->dszz_ypml[iPml] -= Param->pml_dy[iY] * Param->szz_ypml[iPml];
                    Param->dsxy_ypml[iPml] -= Param->pml_dy[iY] * Param->sxy_ypml[iPml];
                    Param->dsyz_ypml[iPml] -= Param->pml_dy[iY] * Param->syz_ypml[iPml];
                } else {
                    Param->dvx[iXYZ]  += Param->rho[iXYZ] * dsxydy;
                    Param->dvy[iXYZ]  += Param->rho[iXYZ] * dsyydy;
                    Param->dvz[iXYZ]  += Param->rho[iXYZ] * dsyzdy;
                    Param->dsxx[iXYZ] += Param->lambda[iXYZ] * dvydy;
                    Param->dsyy[iXYZ] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvydy;
                    Param->dszz[iXYZ] += Param->lambda[iXYZ] * dvydy;
                    Param->dsxy[iXYZ] += Param->mu[iXYZ] * dvxdy;
                    Param->dsyz[iXYZ] += Param->mu[iXYZ] * dvzdy;
                }
            }
        }
    }
}

void cpu_z_derivative(int isForward) {
    int iX, iY, iZ, iXYZ, iPml, fdIndex, iIndex, iRegion;
    float dvxdz, dvydz, dvzdz, dsxzdz, dsyzdz, dszzdz;

    for (iZ = 0; iZ < Param->zNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                iXYZ = iZ * Param->xyNum + iY * Param->xNum + iX;

                dvxdz  = 0.f;
                dvydz  = 0.f;
                dvzdz  = 0.f;
                dsxzdz = 0.f;
                dsyzdz = 0.f;
                dszzdz = 0.f;

                if (isForward) {
                    for (fdIndex = -1; fdIndex <= 3; fdIndex++) {
                        if (iZ + fdIndex < 0) {
                            iIndex = (-(iZ + fdIndex)) * Param->xyNum + iY * Param->xNum + iX;

                            dsxzdz -= Param->sxz[iIndex] * Param->fd_z[fdIndex + 1];
                            dsyzdz -= Param->syz[iIndex] * Param->fd_z[fdIndex + 1];
                            dszzdz -= Param->szz[iIndex] * Param->fd_z[fdIndex + 1];
                        } else if (iZ + fdIndex >= Param->zNum) {
                            continue;
                        } else {
                            iIndex = iXYZ + fdIndex * Param->xyNum;

                            dsxzdz += Param->sxz[iIndex] * Param->fd_z[fdIndex + 1];
                            dsyzdz += Param->syz[iIndex] * Param->fd_z[fdIndex + 1];
                            dszzdz += Param->szz[iIndex] * Param->fd_z[fdIndex + 1];
                            if (iZ > 0) {
                                dvxdz += Param->vx[iIndex] * Param->fd_z[fdIndex + 1];
                                dvydz += Param->vy[iIndex] * Param->fd_z[fdIndex + 1];
                                dvzdz += Param->vz[iIndex] * Param->fd_z[fdIndex + 1];
                            }
                        }
                    }
                } else {
                    for (fdIndex = -3; fdIndex <= 1; fdIndex++) {
                        if (iZ + fdIndex < 0) {
                            iIndex = (-(iZ + fdIndex)) * Param->xyNum + iY * Param->xNum + iX;

                            dsxzdz += Param->sxz[iIndex] * Param->fd_z[1 - fdIndex];
                            dsyzdz += Param->syz[iIndex] * Param->fd_z[1 - fdIndex];
                            dszzdz += Param->szz[iIndex] * Param->fd_z[1 - fdIndex];
                        } else if (iZ + fdIndex >= Param->zNum) {
                            continue;
                        } else {
                            iIndex = iXYZ + fdIndex * Param->xyNum;

                            dsxzdz -= Param->sxz[iIndex] * Param->fd_z[1 - fdIndex];
                            dsyzdz -= Param->syz[iIndex] * Param->fd_z[1 - fdIndex];
                            dszzdz -= Param->szz[iIndex] * Param->fd_z[1 - fdIndex];
                            if (iZ > 2) {
                                dvxdz -= Param->vx[iIndex] * Param->fd_z[1 - fdIndex];
                                dvydz -= Param->vy[iIndex] * Param->fd_z[1 - fdIndex];
                                dvzdz -= Param->vz[iIndex] * Param->fd_z[1 - fdIndex];
                            }
                        }
                    }
                }

                iRegion = get_pml_region_iXYZ(iX, iY, iZ);

                if (iRegion >= 0) {
                    iPml = iXYZ_to_iPml(iX, iY, iZ, iRegion);

                    Param->dvx_zpml[iPml]  += Param->rho[iXYZ] * dsxzdz;
                    Param->dvy_zpml[iPml]  += Param->rho[iXYZ] * dsyzdz;
                    Param->dvz_zpml[iPml]  += Param->rho[iXYZ] * dszzdz;
                    Param->dsxx_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                    Param->dsyy_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                    Param->dszz_zpml[iPml] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvzdz;
                    Param->dsxz_zpml[iPml] += Param->mu[iXYZ] * dvxdz;
                    Param->dsyz_zpml[iPml] += Param->mu[iXYZ] * dvydz;

                    Param->dvx_zpml[iPml]  -= Param->pml_dz[iZ] * Param->vx_zpml[iPml];
                    Param->dvy_zpml[iPml]  -= Param->pml_dz[iZ] * Param->vy_zpml[iPml];
                    Param->dvz_zpml[iPml]  -= Param->pml_dz[iZ] * Param->vz_zpml[iPml];
                    Param->dsxx_zpml[iPml] -= Param->pml_dz[iZ] * Param->sxx_zpml[iPml];
                    Param->dsyy_zpml[iPml] -= Param->pml_dz[iZ] * Param->syy_zpml[iPml];
                    Param->dszz_zpml[iPml] -= Param->pml_dz[iZ] * Param->szz_zpml[iPml];
                    Param->dsxz_zpml[iPml] -= Param->pml_dz[iZ] * Param->sxz_zpml[iPml];
                    Param->dsyz_zpml[iPml] -= Param->pml_dz[iZ] * Param->syz_zpml[iPml];
                } else {
                    Param->dvx[iXYZ]  += Param->rho[iXYZ] * dsxzdz;
                    Param->dvy[iXYZ]  += Param->rho[iXYZ] * dsyzdz;
                    Param->dvz[iXYZ]  += Param->rho[iXYZ] * dszzdz;
                    Param->dsxx[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                    Param->dsyy[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                    Param->dszz[iXYZ] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvzdz;
                    Param->dsxz[iXYZ] += Param->mu[iXYZ] * dvxdz;
                    Param->dsyz[iXYZ] += Param->mu[iXYZ] * dvydz;
                }
            }
        }
    }
}

void cpu_free_surface(int isForward) {
    int iX, iY, iZ, iXYZ, iPml, iRegion, fdIndex, iIndex;
    float dvxdz, dvydz, dvzdz, dvxdx, dvydy, dvzdy, dvzdx;

    for (iY = 0; iY < Param->yNum; iY++) {
        for (iX = 0; iX < Param->xNum; iX++) {
            iXYZ = iY * Param->xNum + iX;

            //x derivative
            dvxdx  = 0.f;
            dvzdx  = 0.f;
            if (isForward) {
                for (fdIndex = -1; fdIndex <= 3; fdIndex++) {
                    if (iX + fdIndex < 0 || iX + fdIndex >= Param->xNum) {
                        continue;
                    }

                    iIndex = iXYZ + fdIndex;

                    dvxdx += Param->vx[iIndex] * Param->fd_x[fdIndex + 1];
                    dvzdx += Param->vz[iIndex] * Param->fd_x[fdIndex + 1];
                }
            } else {
                for (fdIndex = -3; fdIndex <= 1; fdIndex++) {
                    if (iX + fdIndex < 0 || iX + fdIndex >= Param->xNum) {
                        continue;
                    }

                    iIndex = iXYZ + fdIndex;

                    dvxdx -= Param->vx[iIndex] * Param->fd_x[1 - fdIndex];
                    dvzdx -= Param->vz[iIndex] * Param->fd_x[1 - fdIndex];
                }
            }

            //y derivative
            dvydy  = 0.f;
            dvzdy  = 0.f;
            if (isForward) {
                for (fdIndex = -1; fdIndex <= 3; fdIndex++) {
                    if (iY + fdIndex < 0 || iY + fdIndex >= Param->yNum) {
                        continue;
                    }

                    iIndex = iXYZ + fdIndex * Param->xNum;

                    dvydy += Param->vy[iIndex] * Param->fd_y[fdIndex + 1];
                    dvzdy += Param->vz[iIndex] * Param->fd_y[fdIndex + 1];
                }
            } else {
                for (fdIndex = -3; fdIndex <= 1; fdIndex++) {
                    if (iY + fdIndex < 0 || iY + fdIndex >= Param->yNum) {
                        continue;
                    }

                    iIndex = iXYZ + fdIndex * Param->xNum;

                    dvydy -= Param->vy[iIndex] * Param->fd_y[1 - fdIndex];
                    dvzdy -= Param->vz[iIndex] * Param->fd_y[1 - fdIndex];
                }
            }

            iRegion = get_pml_region_iXYZ(iX, iY, 0);

            if (iRegion >= 0) {
                iPml = iXYZ_to_iPml(iX, iY, 0, iRegion);

                dvxdz = Param->pml_dx[iX] / Param->mu[iXYZ] * Param->sxz_xpml[iPml] - dvzdx;
                dvydz = Param->pml_dy[iY] / Param->mu[iXYZ] * Param->syz_ypml[iPml] - dvzdy;
                dvzdz = (Param->pml_dx[iX] * Param->szz_xpml[iPml] + Param->pml_dy[iY] * Param->szz_ypml[iPml] - Param->lambda[iXYZ] * (dvxdx + dvydy)) / (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]);

                /*
                dvxdz = -Param->dsxz_xpml[iPml] / Param->mu[iXYZ];
                dvydz = -Param->dsyz_ypml[iPml] / Param->mu[iXYZ];
                dvzdz = -(Param->dszz_xpml[iPml] + Param->dszz_ypml[iPml]) / (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]);
                */

                Param->dsxz_zpml[iPml] = -Param->dsxz_xpml[iPml];
                Param->dsyz_zpml[iPml] = -Param->dsyz_ypml[iPml];

                Param->dsxx_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                Param->dsyy_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                Param->dszz_zpml[iPml]  = -(Param->dszz_xpml[iPml] + Param->dszz_ypml[iPml]);
            } else {
                dvxdz = -dvzdx;
                dvydz = -dvzdy;
                dvzdz = -Param->lambda[iXYZ] * (dvxdx + dvydy) / (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]);
                
                /*
                dvxdz = -Param->dsxz[iXYZ] / Param->mu[iXYZ];
                dvydz = -Param->dsyz[iXYZ] / Param->mu[iXYZ];
                dvzdz = -Param->dszz[iXYZ] / (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]);
                */

                Param->dsxz[iXYZ] = 0.f;
                Param->dsyz[iXYZ] = 0.f;

                Param->dsxx[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                Param->dsyy[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                Param->dszz[iXYZ] = 0.f;
            }

            if (!isForward) {
                for (iZ = 1; iZ <= 2; iZ++) {
                    iXYZ += Param->xyNum;

                    dvxdz = -0.5f * dvxdz + 0.25f / Param->dz * Param->theDeltaT * (Param->vx[iXYZ + Param->xyNum] + 4 * Param->vx[iXYZ] - 5 * Param->vx[iXYZ - Param->xyNum]);
                    dvydz = -0.5f * dvydz + 0.25f / Param->dz * Param->theDeltaT * (Param->vy[iXYZ + Param->xyNum] + 4 * Param->vy[iXYZ] - 5 * Param->vy[iXYZ - Param->xyNum]);
                    dvzdz = -0.5f * dvzdz + 0.25f / Param->dz * Param->theDeltaT * (Param->vz[iXYZ + Param->xyNum] + 4 * Param->vz[iXYZ] - 5 * Param->vz[iXYZ - Param->xyNum]);
                    
                    if (iRegion >= 0) {
                        iPml += Param->pmlMap[iRegion][4];

                        Param->dsxx_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                        Param->dsyy_zpml[iPml] += Param->lambda[iXYZ] * dvzdz;
                        Param->dszz_zpml[iPml] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvzdz;
                        Param->dsxz_zpml[iPml] += Param->mu[iXYZ] * dvxdz;
                        Param->dsyz_zpml[iPml] += Param->mu[iXYZ] * dvydz;
                    } else {
                        Param->dsxx[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                        Param->dsyy[iXYZ] += Param->lambda[iXYZ] * dvzdz;
                        Param->dszz[iXYZ] += (Param->lambda[iXYZ] + 2 * Param->mu[iXYZ]) * dvzdz;
                        Param->dsxz[iXYZ] += Param->mu[iXYZ] * dvxdz;
                        Param->dsyz[iXYZ] += Param->mu[iXYZ] * dvydz;
                    }
                } 
            }
        }
    }
}

void memory_set_zeros() {
    //set initial memories to zeros
    memset(Param->sxx, 0, sizeof(float) * Param->xyzNum);
    memset(Param->sxy, 0, sizeof(float) * Param->xyzNum);
    memset(Param->sxz, 0, sizeof(float) * Param->xyzNum);
    memset(Param->syy, 0, sizeof(float) * Param->xyzNum);
    memset(Param->syz, 0, sizeof(float) * Param->xyzNum);
    memset(Param->szz, 0, sizeof(float) * Param->xyzNum);
    memset(Param->vx,  0, sizeof(float) * Param->xyzNum);
    memset(Param->vy,  0, sizeof(float) * Param->xyzNum);
    memset(Param->vz,  0, sizeof(float) * Param->xyzNum);

    memset(Param->sxx_xpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->sxy_xpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->sxz_xpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->syy_xpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->szz_xpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vx_xpml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vy_xpml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vz_xpml,  0, sizeof(float) * Param->pmlNumTotal);

    memset(Param->sxx_ypml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->sxy_ypml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->syy_ypml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->syz_ypml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->szz_ypml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vx_ypml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vy_ypml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vz_ypml,  0, sizeof(float) * Param->pmlNumTotal);

    memset(Param->sxx_zpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->sxz_zpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->syy_zpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->syz_zpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->szz_zpml, 0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vx_zpml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vy_zpml,  0, sizeof(float) * Param->pmlNumTotal);
    memset(Param->vz_zpml,  0, sizeof(float) * Param->pmlNumTotal);
}
