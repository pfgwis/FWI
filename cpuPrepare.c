#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "xdfwi.h"

extern Param_t *Param;

//local function
void cpu_allocate_all_field_memory();
void cpu_set_field_parameter();
void cpu_set_modeling_parameter();

void cpu_init() {
    cpu_allocate_all_field_memory();

    cpu_set_field_parameter();

    cpu_set_modeling_parameter();
}

void cpu_allocate_all_field_memory() {
    Param->lambda = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->mu     = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->rho    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->sxx    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->sxy    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->sxz    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->syy    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->syz    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->szz    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->vx     = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->vy     = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->vz     = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dsxx   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dsxy   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dsxz   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dsyy   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dsyz   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dszz   = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dvx    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dvy    = (float *)calloc(Param->xyzNum, sizeof(float));
    Param->dvz    = (float *)calloc(Param->xyzNum, sizeof(float));

    if (Param->lambda  == NULL ||
        Param->mu      == NULL ||
        Param->rho     == NULL ||
        Param->sxx     == NULL ||
        Param->sxy     == NULL ||
        Param->sxz     == NULL ||
        Param->syy     == NULL ||
        Param->syz     == NULL ||
        Param->szz     == NULL ||
        Param->vx      == NULL ||
        Param->vy      == NULL ||
        Param->vz      == NULL ||
        Param->dsxx    == NULL ||
        Param->dsxy    == NULL ||
        Param->dsxz    == NULL ||
        Param->dsyy    == NULL ||
        Param->dsyz    == NULL ||
        Param->dszz    == NULL ||
        Param->dvx     == NULL ||
        Param->dvy     == NULL ||
        Param->dvz     == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for the fields\n");
    }

    Param->sxx_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->sxy_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->sxz_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->syy_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->szz_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vx_xpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vy_xpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vz_xpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxx_xpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxy_xpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxz_xpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsyy_xpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dszz_xpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvx_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvy_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvz_xpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));

    Param->sxx_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->sxy_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->syy_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->syz_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->szz_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vx_ypml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vy_ypml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vz_ypml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxx_ypml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxy_ypml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsyy_ypml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsyz_ypml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dszz_ypml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvx_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvy_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvz_ypml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));

    Param->sxx_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->sxz_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->syy_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->syz_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->szz_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vx_zpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vy_zpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->vz_zpml   = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxx_zpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsxz_zpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsyy_zpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dsyz_zpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dszz_zpml = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvx_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvy_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));
    Param->dvz_zpml  = (float *)calloc(Param->pmlNumTotal, sizeof(float));

    if (Param->sxx_xpml  == NULL ||
        Param->sxy_xpml  == NULL ||
        Param->sxz_xpml  == NULL ||
        Param->syy_xpml  == NULL ||
        Param->szz_xpml  == NULL ||
        Param->vx_xpml   == NULL ||
        Param->vy_xpml   == NULL ||
        Param->vz_xpml   == NULL ||
        Param->dsxx_xpml == NULL ||
        Param->dsxy_xpml == NULL ||
        Param->dsxz_xpml == NULL ||
        Param->dsyy_xpml == NULL ||
        Param->dszz_xpml == NULL ||
        Param->dvx_xpml  == NULL ||
        Param->dvy_xpml  == NULL ||
        Param->dvz_xpml  == NULL ||
        Param->sxx_ypml  == NULL ||
        Param->sxy_ypml  == NULL ||
        Param->syy_ypml  == NULL ||
        Param->syz_ypml  == NULL ||
        Param->szz_ypml  == NULL ||
        Param->vx_ypml   == NULL ||
        Param->vy_ypml   == NULL ||
        Param->vz_ypml   == NULL ||
        Param->dsxx_ypml == NULL ||
        Param->dsxy_ypml == NULL ||
        Param->dsyy_ypml == NULL ||
        Param->dsyz_ypml == NULL ||
        Param->dszz_ypml == NULL ||
        Param->dvx_ypml  == NULL ||
        Param->dvy_ypml  == NULL ||
        Param->dvz_ypml  == NULL ||
        Param->sxx_zpml  == NULL ||
        Param->sxz_zpml  == NULL ||
        Param->syy_zpml  == NULL ||
        Param->syz_zpml  == NULL ||
        Param->szz_zpml  == NULL ||
        Param->vx_zpml   == NULL ||
        Param->vy_zpml   == NULL ||
        Param->vz_zpml   == NULL ||
        Param->dsxx_zpml == NULL ||
        Param->dsxz_zpml == NULL ||
        Param->dsyy_zpml == NULL ||
        Param->dsyz_zpml == NULL ||
        Param->dszz_zpml == NULL ||
        Param->dvx_zpml  == NULL ||
        Param->dvy_zpml  == NULL ||
        Param->dvz_zpml  == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for the pml fields\n");
    }
}

void cpu_set_field_parameter() {
    int iX, iY, iZ, iXYZ;
    //float x, y, z;
    float vp, vs, rho;

    //query the velocity value at the grid point

    // init the input velocity model
    //velocity_model_init();

    if (Param->myID == 0) {
        fprintf(stdout, "\nStart to query value for the simulation domain from the velocity model.\n");
    }

    iXYZ = 0;
    for (iZ = 0; iZ < Param->zNum; iZ++) {
        //z = iZ * Param->dz;
        for (iY = 0; iY < Param->yNum; iY++) {
            //y = Param->yMin + iY * Param->dy;
            for (iX = 0; iX < Param->xNum; iX++) {
                //x = Param->xMin + iX * Param->dx;

                //the model query part could be moved here
                /*
                velocity_model_query(x, y, z, &vp, &vs, &rho);
                if (vs < 500 || vp < 500) {
                    vp = 500.f * vp / vs;
                    vs = 500.f;
                    vp = vs * sqrt(3.);
                } if (vp > 4000) {
                    vs = 4000.f * vs / vp;
                    vp = 4000.f;
                    vs = vp / sqrt(3.);
                }
                */

                vp = 4000.f;
                vs = 4000.f / sqrt(3.);
                rho = 2000.f;

                Param->lambda[iXYZ] = rho * (vp * vp -  2 * vs * vs);
                Param->mu[iXYZ]     = rho * vs * vs;
                Param->rho[iXYZ]    = 1.f / rho; //using 1/rho instead of rho

                iXYZ++;
            }
        }
    }

    //velocity_model_delete();
}

void cpu_set_modeling_parameter() {
    int   iIndex;
    float d0x, d0y, d0z, dtOverDx, dtOverDy, dtOverDz;

    //finite difference coefficient
    dtOverDx = Param->theDeltaT / Param->dx;
    Param->fd_x[0] = -0.30874f * dtOverDx;
    Param->fd_x[1] = -0.6326f  * dtOverDx;
    Param->fd_x[2] = 1.2330f   * dtOverDx;
    Param->fd_x[3] = -0.3334f  * dtOverDx;
    Param->fd_x[4] = 0.04168f  * dtOverDx;

    dtOverDy = Param->theDeltaT / Param->dy;
    Param->fd_y[0] = -0.30874f * dtOverDy;
    Param->fd_y[1] = -0.6326f  * dtOverDy;
    Param->fd_y[2] = 1.2330f   * dtOverDy;
    Param->fd_y[3] = -0.3334f  * dtOverDy;
    Param->fd_y[4] = 0.04168f  * dtOverDy;

    dtOverDz = Param->theDeltaT / Param->dz;
    Param->fd_z[0] = -0.30874f * dtOverDz;
    Param->fd_z[1] = -0.6326f  * dtOverDz;
    Param->fd_z[2] = 1.2330f   * dtOverDz;
    Param->fd_z[3] = -0.3334f  * dtOverDz;
    Param->fd_z[4] = 0.04168f  * dtOverDz;

    //runge kutta coefficient
    Param->nRKStage = 5;

    Param->A[0] = 0;
    Param->A[1] = -567301805773.f / 1357537059087.f;
    Param->A[2] = -2404267990393.f / 2016746695238.f;
    Param->A[3] = -3550918686646.f /2091501179385.f;
    Param->A[4] = -1275806237668.f/842570457699.f;

    Param->B[0] = 1432997174477.f/9575080441755.f;
    Param->B[1] = 5161836677717.f/13612068292357.f;
    Param->B[2] = 1720146321549.f/2090206949498.f;
    Param->B[3] = 3134564353537.f/4481467310338.f;
    Param->B[4] = 2277821191437.f/14882151754819.f;

    Param->nRKStage = 3;

    Param->A[0] = 0;
    Param->A[1] = -5.f / 9.f;
    Param->A[2] = -153.f / 128.f;

    Param->B[0] = 1.f/3.f;
    Param->B[1] = 15.f/16.f;
    Param->B[2] = 8.f/15.f;

    //pml initialization
    Param->pml_dx = (float *)calloc(Param->xNum, sizeof(float));
    Param->pml_dy = (float *)calloc(Param->yNum, sizeof(float));
    Param->pml_dz = (float *)calloc(Param->zNum, sizeof(float));
    if (Param->pml_dx == NULL || Param->pml_dx == NULL || Param->pml_dz == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for the pml\n");
    }

    //  * Param->theDeltaT
    d0x = 3.f * 4000.f / 2.f / Param->pmlNum / Param->dx * 3 * Param->theDeltaT * 3;
    for (iIndex = 0; iIndex < Param->xNum; iIndex++) {
        if (iIndex < Param->pmlNum) {
            Param->pml_dx[iIndex] = d0x * pow((float)(Param->pmlNum - iIndex) / Param->pmlNum, 2);
        } else if (iIndex >= Param->xNum - Param->pmlNum) {
            Param->pml_dx[iIndex] = Param->pml_dx[Param->xNum - 1 - iIndex];
        }
    }

    d0y = 3.f * 4000.f / 2.f / Param->pmlNum / Param->dy * 3 * Param->theDeltaT * 3;
    for (iIndex = 0; iIndex < Param->yNum; iIndex++) {
        if (iIndex < Param->pmlNum) {
            Param->pml_dy[iIndex] = d0y * pow((float)(Param->pmlNum - iIndex) / Param->pmlNum, 2);
        } else if (iIndex >= Param->yNum - Param->pmlNum) {
            Param->pml_dy[iIndex] = Param->pml_dy[Param->yNum - 1 - iIndex];
        }
    }

    d0z = 3.f * 4000.f / 2.f / Param->pmlNum / Param->dz * 3 * Param->theDeltaT * 3;
    for (iIndex = Param->zNum - Param->pmlNum; iIndex < Param->zNum; iIndex++) {
        Param->pml_dz[iIndex] = d0z * pow((float)(Param->pmlNum - (Param->zNum - 1 - iIndex)) / Param->pmlNum, 2);
    }
}

void cpu_free_all_field_memory() {
    free(Param->lambda);
    free(Param->mu);
    free(Param->rho);

    free(Param->sxx);
    free(Param->sxy);
    free(Param->sxz);
    free(Param->syy);
    free(Param->syz);
    free(Param->szz);
    free(Param->vx);
    free(Param->vy);
    free(Param->vz);
    free(Param->dsxx);
    free(Param->dsxy);
    free(Param->dsxz);
    free(Param->dsyy);
    free(Param->dsyz);
    free(Param->dszz);
    free(Param->dvx);
    free(Param->dvy);
    free(Param->dvz);

    free(Param->sxx_xpml);
    free(Param->sxy_xpml);
    free(Param->sxz_xpml);
    free(Param->syy_xpml);
    free(Param->szz_xpml);
    free(Param->vx_xpml);
    free(Param->vy_xpml);
    free(Param->vz_xpml);
    free(Param->dsxx_xpml);
    free(Param->dsxy_xpml);
    free(Param->dsxz_xpml);
    free(Param->dsyy_xpml);
    free(Param->dszz_xpml);
    free(Param->dvx_xpml);
    free(Param->dvy_xpml);
    free(Param->dvz_xpml);

    free(Param->sxx_ypml);
    free(Param->sxy_ypml);
    free(Param->syy_ypml);
    free(Param->syz_ypml);
    free(Param->szz_ypml);
    free(Param->vx_ypml);
    free(Param->vy_ypml);
    free(Param->vz_ypml);
    free(Param->dsxx_ypml);
    free(Param->dsxy_ypml);
    free(Param->dsyy_ypml);
    free(Param->dsyz_ypml);
    free(Param->dszz_ypml);
    free(Param->dvx_ypml);
    free(Param->dvy_ypml);
    free(Param->dvz_ypml);

    free(Param->sxx_zpml);
    free(Param->sxz_zpml);
    free(Param->syy_zpml);
    free(Param->syz_zpml);
    free(Param->szz_zpml);
    free(Param->vx_zpml);
    free(Param->vy_zpml);
    free(Param->vz_zpml);
    free(Param->dsxx_zpml);
    free(Param->dsxz_zpml);
    free(Param->dsyy_zpml);
    free(Param->dsyz_zpml);
    free(Param->dszz_zpml);
    free(Param->dvx_zpml);
    free(Param->dvy_zpml);
    free(Param->dvz_zpml);
}
