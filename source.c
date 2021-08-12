#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "xdfwi.h"

extern Param_t *Param;


//local function
static void compute_source_time_function(float delay, float rTs, float rTp);
static float compute_stf_time(float delay, float rTs, float rTp, float time);

/**
 * compute_source_function: compute the slip source function
 */
static void compute_source_time_function(float delay, float rTs, float rTp) {
    float t1, T;
    int   iTime;

    for (iTime = 0; iTime <= Param->theTotalTimeSteps; iTime++) {
        T = Param->theDeltaT*iTime;   
        
        if (T >= delay) {
            t1 = pow((T - rTs) * PI / rTp, 2);

            // need multiply by the 1 / density
            Param->mySourceTimeFunction[iTime] = (1 - 2 * t1) * exp(-t1) * 2 * Param->theDeltaT * 1e5;
        }     
        else {
            Param->mySourceTimeFunction[iTime] = 0;
        }
    }
}

/**
 * compute_source_function: compute the source function at a time t
 */
static float compute_stf_time(float delay, float rTs, float rTp, float time) {
    float t1;
    
    if (time >= delay) {
        t1 = pow((time - rTs) * PI / rTp, 2);

        return -(1 - 2 * t1) * exp(-t1) * 2 * Param->theDeltaT * 1e5;
    }     
    else {
        return 0.f;
    }
}

/**
 *  init the vertical force source
 */
int source_init() {
    float sx, sy;
    float mySrcKx[KAISER_LEN * 2];
    float mySrcKy[KAISER_LEN * 2];

    Param->theSx = Param->theSrcLat;
    Param->theSy = Param->theSrcLon;

    sx = (Param->theSx - Param->xMin) / Param->dx;
    sy = (Param->theSy - Param->yMin) / Param->dy;

    Param->theSourceX = (int)(sx);
    Param->theSourceY = (int)(sy);
    Param->theSourceZ = 0;

    sx -= Param->theSourceX;
    sy -= Param->theSourceY;

    Param->mySourceIndexXYZ = Param->theSourceZ * Param->xyNum + Param->theSourceY * Param->xNum + Param->theSourceX;

    /* Compute the source displacement and broadcast them */
    Param->mySourceTimeFunction = (float*)calloc(Param->theTotalTimeSteps, sizeof(float));
    if (Param->mySourceTimeFunction == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for Param.mySourceTimeFunction\n");
    }

    compute_source_time_function(0, 15, 10);

    //kaiser windown sinc function
    Param->theSourceX -= KAISER_LEN;
    sx -= KAISER_LEN;

    Param->theSourceY -= KAISER_LEN;
    sy -= KAISER_LEN;

    for (int i = 0; i < KAISER_LEN * 2; i++) {
        mySrcKx[i] = kaiser_sinc(sx + i);
        mySrcKy[i] = kaiser_sinc(sy + i);
    }

    if (Param->useGPU) {
        int bytes = KAISER_LEN * 2 * sizeof(float);
        if (launch_cudaMalloc((void**)&Param->d_srcKx, bytes) != 0 ||
            launch_cudaMalloc((void**)&Param->d_srcKy, bytes) != 0) {
            xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the source\n");
        }

        if (launch_cudaMemcpy(Param->d_srcKx, mySrcKx, bytes, 0) != 0 ||
            launch_cudaMemcpy(Param->d_srcKy, mySrcKy, bytes, 0) != 0) {
            xd_abort(__func__, "launch_cudaMemcpy() failed", "Memory copy from host to device failed for the sourcex\n");
        }
    }

    return 0;
}

void source_delete() {
    free(Param->mySourceTimeFunction);

    if (Param->useGPU) {
        launch_cudaFree(Param->d_srcKx);
        launch_cudaFree(Param->d_srcKy);
    }
}

void source_inject_forward_multiple_stage(int iTime, int iStage) {
    if (Param->useGPU) {
        float time = (iTime + Param->C[iStage]) * Param->theDeltaT;
        //float stf = compute_stf_time(0.f, 15.f, 10.f, time);

	//time = iTime * Param->theDeltaT;
        float stf = compute_stf_time(0.f, 20.f, 15.f, time);

        launch_source_inject_forward(Param->d_dvz, Param->theSourceX, Param->theSourceY, Param->d_srcKx, Param->d_srcKy, stf);
    }
}

void source_inject_forward(int iTime) {
    int iX, iY, iZ, iXYZ;

    if (Param->useGPU) {
        //launch_source_inject_forward(Param->d_vz, Param->theSourceX, Param->theSourceY, Param->theSx, Param->theSy, Param->mySourceTimeFunction[iTime]);
    } else {
        for (iZ = -5; iZ < 5; iZ++) {
            for (iY = -5; iY < 5; iY++) {
                for (iX = -5; iX <= 5; iX++) {
                    iXYZ = (Param->theSourceZ + iZ) * Param->xyNum + (Param->theSourceY + iY) * Param->xNum + Param->theSourceX + iX;
                    Param->vz[iXYZ] += Param->mySourceTimeFunction[iTime] * exp(-pow(iX, 2) - pow(iY, 2) - pow(iZ, 2));
                }
            }
        }
    }
}

void source_inject_backward(int iTime) {
    int iStation, iX, iY, iZ, iXYZ;

    if (Param->useGPU) {
        launch_source_inject_backward(Param->d_station_iX, Param->d_station_iY, Param->d_station_x, Param->d_station_y, Param->d_station_vz + iTime * Param->myNumberOfStations, Param->d_dvz, Param->myNumberOfStations);
    } else {
        for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
            for (iZ = -5; iZ < 5; iZ++) {
                for (iY = -5; iY < 5; iY++) {
                    for (iX = -5; iX <= 5; iX++) {
                        iXYZ = (Param->myStations[iStation].iY + iY) * Param->xNum + Param->myStations[iStation].iX + iX;
                        Param->vz[iXYZ] += Param->myStations[iStation].vz[iTime] * exp(-pow(iX, 2) - pow(iY, 2) - pow(iZ, 2));
                    }
                }
            }
        }
    }
}
