#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "xdfwi.h"

extern Param_t* Param;

//local functions
static void velocity_model_read();
static void velocity_model_broadcast();
static float interpolate_linear_general(float x, int numsamples, float *xdiscrete, float *discretefunction);

int velocity_model_init() {
    if (Param->myID == 0) {
        fprintf(stdout, "\nStart to read velocity model: %s\n", Param->theVelocityModelPath);
        velocity_model_read();
    }

    //velocity_model_broadcast();

    return 0;
}

static void velocity_model_read() { 
    FILE* velocityModelFp;

    velocityModelFp = fopen(Param->theVelocityModelPath, "rb");
    if (velocityModelFp == NULL) {
        xd_abort(__func__, "fopen() fail", "Error open the velocity model file: %s\n", Param->theVelocityModelPath);
    }

    /* read the grid information */
    fread(&Param->theVelocityModel.xNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.yNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.zNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.xMin, sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.yMin, sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.dx,   sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.dz,   sizeof(float), 1, velocityModelFp);    
    
    Param->theVelocityModel.dy = Param->theVelocityModel.dx;
    Param->theVelocityModel.xMax = Param->theVelocityModel.xMin + (Param->theVelocityModel.xNum - 1) * Param->theVelocityModel.dx;
    Param->theVelocityModel.yMax = Param->theVelocityModel.yMin + (Param->theVelocityModel.yNum - 1) * Param->theVelocityModel.dy;
    Param->theVelocityModel.gridNum = Param->theVelocityModel.xNum * Param->theVelocityModel.yNum * Param->theVelocityModel.zNum;

    /* print the velocity model information */
    fprintf(stdout, "\tModel domainX: (min: %f, dx: %f, max:%f, num:%d)\n", Param->theVelocityModel.xMin,
            Param->theVelocityModel.dx, Param->theVelocityModel.xMax, Param->theVelocityModel.xNum);
    fprintf(stdout, "\tModel domainY: (min: %f, dy: %f, max:%f, num:%d)\n", Param->theVelocityModel.yMin,
            Param->theVelocityModel.dy, Param->theVelocityModel.yMax, Param->theVelocityModel.yNum);
    fprintf(stdout, "\tModel domainZ: (dz: %f, num:%d)\n", Param->theVelocityModel.dz, Param->theVelocityModel.zNum);

    /* read profile at each grid point */
    Param->theVelocityModel.vp  = (float *)malloc(Param->theVelocityModel.gridNum * sizeof(float));
    Param->theVelocityModel.vs  = (float *)malloc(Param->theVelocityModel.gridNum * sizeof(float));
    Param->theVelocityModel.rho = (float *)malloc(Param->theVelocityModel.gridNum * sizeof(float));
    if (Param->theVelocityModel.vp == NULL || Param->theVelocityModel.vs == NULL || Param->theVelocityModel.rho == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for theVelocityModel\n");
    }

    //fread(Param->theVelocityModel.vp,  sizeof(float), Param->theVelocityModel.gridNum, velocityModelFp);
    fread(Param->theVelocityModel.vs,  sizeof(float), Param->theVelocityModel.gridNum, velocityModelFp);
    //fread(Param->theVelocityModel.rho, sizeof(float), Param->theVelocityModel.gridNum, velocityModelFp);

    /*
    for (int iZ = 0; iZ < Param->theVelocityModel.zNum; iZ++) {
        for (int iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            for (int iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                int index = (iZ * Param->theVelocityModel.yNum + iY) * Param->theVelocityModel.xNum + iX;

                Param->theVelocityModel.vs[index] *= 1.f + sin(iX / 50.f * PI) * sin(iY / 50.f * PI) * 0.05f;
		        if (Param->theVelocityModel.vs[index] < 2000.f) {
			        Param->theVelocityModel.vs[index] = 2000.f;
		        }
            }
        }
    }
    */

    fclose(velocityModelFp);

    // read new updated model
    /*
    velocityModelFp = fopen("./model.iter_p", "rb");
    if (velocityModelFp == NULL) {
        xd_abort(__func__, "fopen() fail", "Error open the velocity model file: %s\n", Param->theVelocityModelPath);
    }

    fread(&Param->theVelocityModel.xNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.yNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.zNum, sizeof(int),   1, velocityModelFp);
    fread(&Param->theVelocityModel.xMin, sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.yMin, sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.dx,   sizeof(float), 1, velocityModelFp);
    fread(&Param->theVelocityModel.dz,   sizeof(float), 1, velocityModelFp);    

    fread(Param->theVelocityModel.vs,  sizeof(float), Param->theVelocityModel.gridNum, velocityModelFp);

    fclose(velocityModelFp);

    */

    return;
}

void velocity_model_write() {
    return;
}

/*
static void velocity_model_broadcast() {
    float float_message[6];
    int    int_message[4];
    int    iProfile;

    if (Param->myID == 0) {
        float_message[0] = Param->theVelocityModel.xMin;
        float_message[1] = Param->theVelocityModel.xMax;
        float_message[2] = Param->theVelocityModel.dx;
        float_message[3] = Param->theVelocityModel.yMin;
        float_message[4] = Param->theVelocityModel.yMax;
        float_message[5] = Param->theVelocityModel.dy;
        int_message[0] = Param->theVelocityModel.xNum;
        int_message[1] = Param->theVelocityModel.yNum;
        int_message[2] = Param->theVelocityModel.gridNum;
        int_message[3] = Param->theVelocityModel.depthNum;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (Param->myID == 0) {
        fprintf(stdout, "Start to broadcast the velocity model\n");
    }

    MPI_Bcast(float_message, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(int_message   , 4, MPI_INT, 0, MPI_COMM_WORLD);

    Param->theVelocityModel.xMin     = float_message[0];
    Param->theVelocityModel.xMax     = float_message[1];
    Param->theVelocityModel.dx       = float_message[2];
    Param->theVelocityModel.yMin     = float_message[3];
    Param->theVelocityModel.yMax     = float_message[4];
    Param->theVelocityModel.dy       = float_message[5];
    Param->theVelocityModel.xNum     = int_message[0];
    Param->theVelocityModel.yNum     = int_message[1];
    Param->theVelocityModel.gridNum  = int_message[2];
    Param->theVelocityModel.depthNum = int_message[3];

    if (Param->myID != 0) {
        Param->theVelocityModel.depth = (float *)malloc(Param->theVelocityModel.depthNum * sizeof(float));
        if (Param->theVelocityModel.depth == NULL) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for theVelocityModel\n");
        }

        Param->theVelocityModel.vp  = (float **)malloc(Param->theVelocityModel.gridNum * sizeof(float *));
        Param->theVelocityModel.vs  = (float **)malloc(Param->theVelocityModel.gridNum * sizeof(float *));
        Param->theVelocityModel.rho = (float **)malloc(Param->theVelocityModel.gridNum * sizeof(float *));
        if (Param->theVelocityModel.vp == NULL || Param->theVelocityModel.vs == NULL || Param->theVelocityModel.rho == NULL) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for theVelocityModel\n");
        }

        for (iProfile = 0; iProfile < Param->theVelocityModel.gridNum; iProfile++) {
            Param->theVelocityModel.vp[iProfile]  = (float *)malloc(Param->theVelocityModel.depthNum * sizeof(float));
            Param->theVelocityModel.vs[iProfile]  = (float *)malloc(Param->theVelocityModel.depthNum * sizeof(float));
            Param->theVelocityModel.rho[iProfile] = (float *)malloc(Param->theVelocityModel.depthNum * sizeof(float));
            if (Param->theVelocityModel.vp[iProfile] == NULL || Param->theVelocityModel.vs[iProfile] == NULL || Param->theVelocityModel.rho[iProfile] == NULL) {
                xd_abort(__func__, "malloc() failed", "Memory allocation failed for theVelocityModel\n");
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(Param->theVelocityModel.depth, Param->theVelocityModel.depthNum, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    for (iProfile = 0; iProfile < Param->theVelocityModel.gridNum; iProfile++) {
        MPI_Bcast(Param->theVelocityModel.vp[iProfile], Param->theVelocityModel.depthNum, MPI_FLOAT, 0, MPI_COMM_WORLD); 
        MPI_Bcast(Param->theVelocityModel.vs[iProfile], Param->theVelocityModel.depthNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(Param->theVelocityModel.rho[iProfile], Param->theVelocityModel.depthNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    return;
} */

void velocity_model_query(float x, float y, int iZ, float *vp, float *vs, float *rho) {
    int nodeX[4], nodeY[4], iNode, index;  
    float csi, etha, phi;
    float velP, velS, density;
    static float csiI[4]  = {-1, -1,  1,  1};
    static float ethaI[4] = {-1,  1,  1, -1};

    /* if the query is out of the database return the value of the last point 
     in the direction that has not been passed if none it will return the 
     value of the upper right corner */    
    if(Param->theVelocityModel.xMin > x) x = Param->theVelocityModel.xMin;
    if(Param->theVelocityModel.xMax < x) x = Param->theVelocityModel.xMax;
    if(Param->theVelocityModel.yMin > y) y = Param->theVelocityModel.yMin;
    if(Param->theVelocityModel.yMax < y) y = Param->theVelocityModel.yMax;

    /* now do the interpolation
     * node scheme: 
     *                  1----------------2
     *                  |                |
     *         Y        |                |
     *         ^        |                |
     *         |        |                | 
     *         |        |                | 
     *         |        0----------------3
     *         ----->X
     *
     *-----------------------------------*/

    /* search in the grid for the node 0 */
    nodeX[0] = (int)floor((x - Param->theVelocityModel.xMin) / Param->theVelocityModel.dx); 
    nodeY[0] = (int)floor((y - Param->theVelocityModel.yMin) / Param->theVelocityModel.dy);
        
    nodeX[1] = nodeX[0];
    nodeY[1] = nodeY[0] + 1;
    nodeX[2] = nodeX[0] + 1;
    nodeY[2] = nodeY[0] + 1;
    nodeX[3] = nodeX[0] + 1;
    nodeY[3] = nodeY[0];

    /* compute csi and etha */  
    csi  = Param->theVelocityModel.xMin + (nodeX[0] + 0.5) * Param->theVelocityModel.dx;
    etha = Param->theVelocityModel.yMin + (nodeY[0] + 0.5) * Param->theVelocityModel.dy;  
    csi  = 2 * (x - csi)  / Param->theVelocityModel.dx;
    etha = 2 * (y - etha) / Param->theVelocityModel.dy;
    
    /* loop over all nodes and compute the value of z interpolated */
    *vp  = 0;
    *vs  = 0;
    *rho = 0;
 
    for (iNode = 0; iNode < 4; iNode++) {
        index = (iZ * Param->theVelocityModel.yNum + nodeY[iNode]) * Param->theVelocityModel.xNum + nodeX[iNode];

        velP = Param->theVelocityModel.vp[index];
        velS = Param->theVelocityModel.vs[index];
        density = Param->theVelocityModel.rho[index];
        
        phi = phi_local(iNode, csi, etha, &(csiI[0]), &(ethaI[0]));     
    
        *vs  += phi * velS;
        *vp  += phi * velP;
        *rho += phi * density;
    }

    return;
}

void velocity_model_delete() {
    free(Param->theVelocityModel.vp);
    free(Param->theVelocityModel.vs);
    free(Param->theVelocityModel.rho);
}

/*
 * interpolate_linear_general: naive implementation to interpolate linearly a function 
 *                             if the x is larger or smaller than the one 
 *                             supported by the function
 *                             the last or first value will be assigned
 **/
static float interpolate_linear_general(float x, int numsamples, float *xdiscrete, float *discretefunction) {
    float k;
    int iInterval = 0;

    if (x >= xdiscrete[numsamples - 1])
    return discretefunction[numsamples - 1];
    if (x <= xdiscrete[0])
    return discretefunction[0]; 

    /* locate the interval using a n complexity this has to change */
    while (x >= xdiscrete[iInterval + 1]) iInterval++;
        
    /* k = (y1 -y0) / (x1 - x0) */
    k = (discretefunction[iInterval + 1] - discretefunction[iInterval]) / (xdiscrete[iInterval + 1] - xdiscrete[iInterval]);

    /* y(x) = y0 + k * (x - x0) */
    return k * (x - xdiscrete[iInterval]) + discretefunction[iInterval]; 
}

/*
 *   phi_local: local shape function
 *
 *                 1----------------2
 *                 |       ^etha    |
 *                 |       |   csi  |
 *                 |        ---->   |
 *                 |                | 
 *                 |                | 
 *                 0----------------3
 *
 *   input:       i - shape function number
 *              csi - horizontal coord
 *            ethai - vertical coord
 *      ethai  csii - arrays with the convention of orintation in csi and etha
 **/
float phi_local(int i, float csi, float etha, float *csii, float *ethai) {
        return 0.25 * ( 1 + csii[i] * csi) * ( 1 + ethai[i]* etha);
}
