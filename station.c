#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "xdfwi.h"

extern Param_t* Param;

/**
 * read seismic data and init stations
 */
int station_init() {
    int iStation, itmp, totalStation;
    float ftmp;
    float x, y;

    char dataFileName[MAX_FILENAME_LEN];
    sprintf(dataFileName, "../data/blocktest/station.%d", Param->theSrcID);
    FILE *stationFp;

    stationFp = fopen(dataFileName, "r");
    if (stationFp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open station input files!\n");
    }

    fscanf(stationFp, "%d", &totalStation);
    fscanf(stationFp, "%d", &itmp);
    fscanf(stationFp, "%f", &ftmp);

    Param->myNumberOfStations = totalStation;
/*
    for (int i = 0; i < totalStation; i++) {
        fscanf(stationFp, "%f %f", &x, &y);
        Param->myNumberOfStations++;
        if (sqrt(pow(x - Param->theSx, 2) + pow(y - Param->theSy, 2)) > 50000.f) {
            Param->myNumberOfStations++;
        }
    }

    rewind(stationFp);
    fscanf(stationFp, "%d", &totalStation);
    fscanf(stationFp, "%d", &itmp);
    fscanf(stationFp, "%f", &ftmp);
*/
    Param->myStations = (station_t *)malloc(Param->myNumberOfStations * sizeof(station_t));
    if (Param->myStations == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for Param.myStations\n");
    }

    iStation = 0;
    for (int i = 0; i < totalStation; i++) {
        fscanf(stationFp, "%f %f", &Param->myStations[iStation].lon, &Param->myStations[iStation].lat);
        //latlon2xy_azequaldist(Param->theSrcLat, Param->theSrcLon, Param->myStations[iStation].lat, Param->myStations[iStation].lon, &x, &y);
        x = Param->myStations[iStation].lat;
        y = Param->myStations[iStation].lon;
        if (x < Param->xMin || x > Param->xMax || y < Param->yMin || y > Param->yMax) {
            Param->myNumberOfStations--;
            continue;
        }
  //      if (sqrt(pow(x - Param->theSx, 2) + pow(y - Param->theSy, 2)) > 50000.f) {
            Param->myStations[iStation].iX = (int)((x - Param->xMin) / Param->dx);
            Param->myStations[iStation].iY = (int)((y - Param->yMin) / Param->dy);
            Param->myStations[iStation].x  = (x - Param->xMin) / Param->dx - Param->myStations[iStation].iX;
            Param->myStations[iStation].y  = (y - Param->yMin) / Param->dy - Param->myStations[iStation].iY;
            Param->myStations[iStation].dis = sqrt(pow(x - Param->theSx, 2) + pow(y - Param->theSy, 2));
            Param->myStations[iStation].indexXYZ = Param->myStations[iStation].iY * Param->xNum + Param->myStations[iStation].iX;

            Param->myStations[iStation].id = iStation;

            Param->myStations[iStation].vz = (float *)malloc(Param->theTotalTimeSteps * sizeof(float));
            if (Param->myStations[iStation].vz == NULL) {
                xd_abort(__func__, "malloc() failed", "Memory allocation failed for Param->myStations.vz\n");
            }

            iStation++;
    //    }
    }

    if (Param->useGPU) {
        int   *iX = (int *)malloc(Param->myNumberOfStations * sizeof(int));
        int   *iY = (int *)malloc(Param->myNumberOfStations * sizeof(int));
        float *xc = (float *)malloc(Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float));
        float *yc = (float *)malloc(Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float));
        if (iX == NULL || iY == NULL || xc == NULL || yc == NULL) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for station\n");
        }

        for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
            iX[iStation] = Param->myStations[iStation].iX - KAISER_LEN;
            iY[iStation] = Param->myStations[iStation].iY - KAISER_LEN;
            for (int i = 0; i < KAISER_LEN * 2; i++) {
                xc[iStation * KAISER_LEN * 2 + i] = kaiser_sinc(Param->myStations[iStation].x - KAISER_LEN + i);
                yc[iStation * KAISER_LEN * 2 + i] = kaiser_sinc(Param->myStations[iStation].y - KAISER_LEN + i);
            }
        }

        if (launch_cudaMalloc((void**)&Param->d_station_iX, Param->myNumberOfStations * sizeof(int))   != 0 ||
            launch_cudaMalloc((void**)&Param->d_station_iY, Param->myNumberOfStations * sizeof(int))   != 0 ||
            launch_cudaMalloc((void**)&Param->d_station_x,  Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float)) != 0 ||
            launch_cudaMalloc((void**)&Param->d_station_y,  Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float)) != 0 ||
            launch_cudaMalloc((void**)&Param->d_station_vz, Param->myNumberOfStations * sizeof(float) * Param->theTotalTimeSteps) !=0) {
            xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the stations\n");
        }

        if (launch_cudaMemset(Param->d_station_vz,   0, Param->myNumberOfStations * sizeof(float) * Param->theTotalTimeSteps) != 0) {
            xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for stations\n");
        }

        if (launch_cudaMemcpy(Param->d_station_iX, iX, Param->myNumberOfStations * sizeof(int),   0) != 0 ||
            launch_cudaMemcpy(Param->d_station_iY, iY, Param->myNumberOfStations * sizeof(int),   0) != 0 ||
            launch_cudaMemcpy(Param->d_station_x,  xc, Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float), 0) != 0 ||
            launch_cudaMemcpy(Param->d_station_y,  yc, Param->myNumberOfStations * KAISER_LEN * 2 * sizeof(float), 0) != 0) {
            xd_abort(__func__, "launch_cudaMemcpy() failed", "Memory copy from host to device failed for the station index\n");
        }

        free(iX);
        free(iY);
        free(xc);
        free(yc);
    }

    fclose(stationFp);

    return 0;
}

void station_download() {
    int iTime, iStation;
    int bytes = sizeof(float) * Param->theTotalTimeSteps * Param->myNumberOfStations;

    float *station = (float *)malloc(bytes);
    if (station == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for station\n");
    }

    if (launch_cudaMemcpy(station, Param->d_station_vz, bytes, 1) != 0) {
        xd_abort(__func__, "launch_cudaMemcpy() failed", "Memory copy from device to host failed for the station\n");
    }

    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
            Param->myStations[iStation].vz[iTime] = station[iTime * Param->myNumberOfStations + iStation];
        }
    }

    free(station);
}

void station_output_qc() {
    FILE* fp;
    char stationFileName[MAX_FILENAME_LEN];
    int iStation, iTime;

    sprintf(stationFileName, "station.%d", Param->theSrcID);
    fp = fopen(stationFileName, "w");
    if (fp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open station output files!\n");
    }

    fprintf(fp, "%d %d %f\n", Param->myNumberOfStations, Param->theTotalTimeSteps, Param->theDeltaT);

    for (iStation = 0 ; iStation < Param->myNumberOfStations; iStation++) {
        fprintf(fp, "%f %f\n", Param->myStations[iStation].lon, Param->myStations[iStation].lat);
    }

    for (iStation = 0 ; iStation < Param->myNumberOfStations; iStation++) {
        for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
          fprintf(fp, "%f\n", Param->myStations[iStation].vz[iTime]);
        }
    }

    fclose(fp);
}

void station_extract(int step) {
    //need be modified if station is not on the grid
    if (Param->useGPU) {
        launch_station_extract(Param->d_station_iX, Param->d_station_iY, Param->d_station_x, Param->d_station_y, Param->d_station_vz + step * Param->myNumberOfStations, Param->d_vz, Param->myNumberOfStations);
    } else {
        for (int iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
            Param->myStations[iStation].vz[step] = Param->vz[Param->myStations[iStation].indexXYZ];
        }
    }
}

void station_delete() {
    int i;

    /* free stations */
    for (i = 0; i < Param->myNumberOfStations; i++) {
        free(Param->myStations[i].vz);
    }

    if (Param->myStations != NULL) {
        free(Param->myStations);
    }

    if (Param->d_station_iX != NULL) {
        launch_cudaFree(Param->d_station_iX);
        launch_cudaFree(Param->d_station_iY);
        launch_cudaFree(Param->d_station_x);
        launch_cudaFree(Param->d_station_y);
        launch_cudaFree(Param->d_station_vz);
    }
}
