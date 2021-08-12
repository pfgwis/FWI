#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xdfwi.h"

void wavefield_extract_init() {
    int iX, iY, iZ, totalSteps;
    float coord[3], dt;
    char fileName[MAX_FILENAME_LEN];

    /* open the wavefield output file*/
    sprintf(fileName, "wavefield.%d", Param->myID);
    Param->theWavefieldOutFp = fopen(fileName, "wb");
    if (Param->theWavefieldOutFp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the wavefield extract file\n");
    }

    //ouptut the time inforamtion
    totalSteps = (Param->theTotalTimeSteps - 1) / Param->theOutputRate + 1;
    dt = Param->theDeltaT * Param->theOutputRate;
    fwrite(&totalSteps, sizeof(int), 1, Param->theWavefieldOutFp);
    fwrite(&dt, sizeof(float), 1, Param->theWavefieldOutFp);

    //output the coordinate information
    fwrite(&Param->xNum, sizeof(int), 1, Param->theWavefieldOutFp);
    fwrite(&Param->yNum, sizeof(int), 1, Param->theWavefieldOutFp);
    fwrite(&Param->zNum, sizeof(int), 1, Param->theWavefieldOutFp);
    for (iZ = 0; iZ < Param->zNum; iZ++) {
        coord[2] = iZ * Param->dz;
        for (iY = 0; iY < Param->yNum; iY++) {
            coord[1] = Param->yMin + iY * Param->dy;
            for (iX = 0; iX < Param->xNum; iX++) {
                coord[0] = Param->xMin + iX * Param->dx;

                fwrite(coord, sizeof(float), 3, Param->theWavefieldOutFp);
            }
        }
    }

    if(launch_cudaMallocHost((void**)&Param->vz, Param->xyzNum * sizeof(float)) != 0) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for the host fields memory\n");
    }
}

void wavefield_extract() {
    //output the vz
    fwrite(Param->vz, sizeof(float), Param->xyzNum, Param->theWavefieldOutFp);
}
