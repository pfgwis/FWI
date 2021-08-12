#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "xdfwi.h"

extern Param_t *Param;

static void kernel_smooth();
static void kernel_station_clip();

void kernel_init() {
	char fileName[MAX_FILENAME_LEN];

    if (Param->useGPU) {
        int bytes = Param->theVelocityModel.xNum * Param->theVelocityModel.yNum * Param->kernelZNum;
        Param->Kmu_all = (float *)calloc(bytes, sizeof(float));

        if(Param->Kmu_all == NULL) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for the host kernel memory\n");
        }

        bytes = Param->xyzNumKernel * sizeof(float);
        if(launch_cudaMallocHost((void**)&Param->Klambda, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->Kmu,     bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->pk,      bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->gk,      bytes) != 0) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for the host kernel memory\n");
        }

        if (launch_cudaMalloc((void**)&Param->d_Klambda, bytes) != 0 ||
            launch_cudaMalloc((void**)&Param->d_Kmu,     bytes) != 0) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for the device kernel memory\n");
        }

        bytes *= (Param->theTotalTimeSteps - Param->theOutputStartTimeStep) / Param->theOutputRate + 1;

        if(launch_cudaMallocHost((void**)&Param->sxx, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->sxy, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->sxz, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->syy, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->syz, bytes) != 0 ||
           launch_cudaMallocHost((void**)&Param->szz, bytes) != 0) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for the host kernel memory\n");
        }
    } else {
        Param->Klambda = (float *)calloc(Param->xyzNum, sizeof(float));
        Param->Kmu     = (float *)calloc(Param->xyzNum, sizeof(float));
        //Param->Krho    = (float *)calloc(Param->xyzNum, sizeof(float));

        if (Param->Klambda == NULL || Param->Kmu == NULL || Param->Krho == NULL) {
            xd_abort(__func__, "malloc() failed", "Memory allocation failed for the fields\n");
        }

        /* open the kernel output file*/
        sprintf(fileName, "kernel.%d", Param->myID);
        Param->kernelOutputFp = fopen(fileName, "w+b");
        if (Param->kernelOutputFp == NULL) {
            xd_abort(__func__, "fopen() failed", "Cannot open the kernel output files\n");
        }
    }
}

/**
  *  Output the needed wavefield to calculate kernel for the current timestep. 
  *  Each processor will create a seperated file to save the result.
  */
void kernel_forward_extract() {
	fwrite(Param->sxx, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
	fwrite(Param->sxy, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    fwrite(Param->sxz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
	fwrite(Param->syy, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    fwrite(Param->syz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    fwrite(Param->szz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    //fwrite(Param->vx,  sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    //fwrite(Param->vy,  sizeof(float), Param->xyzNum, Param->kernelOutputFp);
    //fwrite(Param->vz,  sizeof(float), Param->xyzNum, Param->kernelOutputFp);
}

void* kernel_forward_extract_multi_thread(void* nullPointer) {
    /*
    float totalModelingTime = 0.f;

    totalModelingTime -= MPI_Wtime();
    for (int i = 0; i < 10; i++) {
        fwrite(Param->sxx, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
        fwrite(Param->sxy, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
        fwrite(Param->sxz, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
        fwrite(Param->syy, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
        fwrite(Param->syz, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
        fwrite(Param->szz, sizeof(float), Param->xyzNumNonPml, Param->kernelOutputFp);
    }
    totalModelingTime += MPI_Wtime();
    fprintf(stdout, "Array output: %f\n", totalModelingTime);

    float *largeMemory;
    largeMemory = (float *)malloc(sizeof(float) * Param->xyzNumNonPml * 6 * 100);
    if (largeMemory == NULL) {
        fprintf(stdout, "Cannot alloate memory for largeMemory\n");
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for the fields\n");
    } else {
        fprintf(stdout, "%fGB memory allocated successfully\n", sizeof(float) * Param->xyzNumNonPml * 600.f / 1024. /1024. / 1024.);
    }
    free(largeMemory);
*/
    return NULL;
}

void kernel_backward_xcoor() {
    if (Param->useGPU) {
        launch_kernel_backward_xcoor(Param->d_sxx, Param->d_sxy, Param->d_sxz, Param->d_syy, Param->d_syz, Param->d_szz, Param->d_dsxx, Param->d_dsxy, Param->d_dsxz, Param->d_dsyy, Param->d_dsyz, Param->d_dszz, Param->d_Klambda, Param->d_Kmu, Param->kernelZNum);
    } else {
    	int iXYZ;
        long int offset = -sizeof(float) * 6 * Param->xyzNum;

        //read forward wavefield stress
        fseek(Param->kernelOutputFp, offset, SEEK_CUR);
        fread(Param->dsxx, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        fread(Param->dsxy, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        fread(Param->dsxz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        fread(Param->dsyy, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        fread(Param->dsyz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        fread(Param->dszz, sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        //fread(Param->dvx,  sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        //fread(Param->dvy,  sizeof(float), Param->xyzNum, Param->kernelOutputFp);
        //fread(Param->vz,   sizeof(float), Param->xyzNum, Param->kernelOutputFp);

        for (iXYZ = 0; iXYZ < Param->xyzNum; iXYZ++) {
            //Krho
            //Param->Krho[iXYZ] += Param->vx[iXYZ] * Param->dvx[iXYZ];
            //Param->Krho[iXYZ] += Param->vy[iXYZ] * Param->dvy[iXYZ];
            //Param->Krho[iXYZ] += Param->vz[iXYZ] * Param->dvz[iXYZ];

        	//Kmu
        	Param->Kmu[iXYZ] += Param->sxx[iXYZ] * Param->dsxx[iXYZ];
            Param->Kmu[iXYZ] += Param->sxy[iXYZ] * Param->dsxy[iXYZ];
            Param->Kmu[iXYZ] += Param->sxz[iXYZ] * Param->dsxz[iXYZ];
            Param->Kmu[iXYZ] += Param->syy[iXYZ] * Param->dsyy[iXYZ];
            Param->Kmu[iXYZ] += Param->syz[iXYZ] * Param->dsyz[iXYZ];
            Param->Kmu[iXYZ] += Param->szz[iXYZ] * Param->dszz[iXYZ];

            //Klambda
            Param->Klambda[iXYZ] += (Param->sxx[iXYZ] + Param->syy[iXYZ] + Param->szz[iXYZ]) * (Param->dsxx[iXYZ] + Param->dsyy[iXYZ] + Param->dszz[iXYZ]);
        }

        fseek(Param->kernelOutputFp, offset, SEEK_CUR);
    }
}

void kernel_processing() {
    if (Param->useGPU) {
        //launch_station_clip(Param->d_Klambda, Param->d_Kmu, Param->d_station_iX, Param->d_station_iY, Param->myNumberOfStations);
        //launch_add_kernel(Param->d_Klambda0, Param->d_Kmu0, Param->d_Klambda, Param->d_Kmu, Param->kernelZNum);
        int bytes = Param->xyzNumKernel * sizeof(float);

        if (launch_cudaMemcpy(Param->Klambda, Param->d_Klambda, bytes, 1) != 0 ||
            launch_cudaMemcpy(Param->Kmu,     Param->d_Kmu,     bytes, 1) != 0) {
            xd_abort(__func__, "malloc() failed", "Cannot copy kernel from device to host\n");
        }

        //kernel_station_clip();

        //kernel_smooth();

        if (launch_cudaMemcpy(Param->d_Klambda, Param->Klambda, bytes, 0) != 0 ||
            launch_cudaMemcpy(Param->d_Kmu,     Param->Kmu,     bytes, 0) != 0) {
            xd_abort(__func__, "malloc() failed", "Cannot copy kernel from host to device\n");
        }
    }
}

void kernel_add_all() {
    float x, y, lat, lon;
    int nodeX[4], nodeY[4], iNode, index;  
    float csi, etha, phi, kv;
    static float csiI[4]  = {-1, -1,  1,  1};
    static float ethaI[4] = {-1,  1,  1, -1};

    int bytes = Param->xyzNumKernel * sizeof(float);

    if (launch_cudaMemcpy(Param->Kmu, Param->d_Kmu, bytes, 1) != 0) {
        xd_abort(__func__, "malloc() failed", "Cannot copy kernel from device to host\n");
    }

    for (int iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (int iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            lon = Param->theVelocityModel.yMin + Param->theVelocityModel.dy * iY;
            for (int iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                lat = Param->theVelocityModel.xMin + Param->theVelocityModel.dx * iX;

                latlon2xy_azequaldist(Param->theSrcLat, Param->theSrcLon, lat, lon, &x, &y);

                if(Param->xMin > x) continue;
                if(Param->xMax <= x) continue;
                if(Param->yMin > y) continue;
                if(Param->yMax <= y) continue;

                /* search in the grid for the node 0 */
                nodeX[0] = (int)floor((x - Param->xMin) / Param->dx); 
                nodeY[0] = (int)floor((y - Param->yMin) / Param->dx);
                    
                nodeX[1] = nodeX[0];
                nodeY[1] = nodeY[0] + 1;
                nodeX[2] = nodeX[0] + 1;
                nodeY[2] = nodeY[0] + 1;
                nodeX[3] = nodeX[0] + 1;
                nodeY[3] = nodeY[0];

                /* compute csi and etha */  
                csi  = Param->xMin + (nodeX[0] + 0.5) * Param->dx;
                etha = Param->yMin + (nodeY[0] + 0.5) * Param->dx;  
                csi  = 2 * (x - csi)  / Param->dx;
                etha = 2 * (y - etha) / Param->dx;
                
                /* loop over all nodes and compute the value of z interpolated */
                kv  = 0;
             
                for (iNode = 0; iNode < 4; iNode++) {
                    index = (iZ * Param->yNum + nodeY[iNode]) * Param->xNum + nodeX[iNode];
                    
                    phi = phi_local(iNode, csi, etha, &(csiI[0]), &(ethaI[0]));

                    kv += phi * Param->Kmu[index];
                }

                index = (iZ * Param->theVelocityModel.yNum + iY) * Param->theVelocityModel.xNum + iX;

                Param->Kmu_all[index] += kv;
            }
        }
    }
}

static void kernel_station_clip() {
    int iX, iY, iZ, iSx, iSy;
    float x, y;
    FILE *stationFp;

    stationFp = fopen("info.TA", "r");
    if (stationFp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open station input files!\n");
    }

    //processing the kernel, clip the station
    while (fscanf(stationFp, "%f %f", &x, &y) != EOF) {
        iSx = (int)((x - Param->xMin) / Param->dx);
        iSy = (int)((y - Param->yMin) / Param->dy);

        for (iZ = 0; iZ < 4; iZ++) {
            float clip_value = 0.f;
            for (iY = -8; iY <= 8; iY++) {
                for (iX = -8; iX <= 8; iX++) {
                    if ((iX <= -5 || iX >= 5) && (iY <= -5 || iY >= 5)) {
                        int index = iZ * Param->xyNum + (iY + iSy) * Param->xNum + iX + iSx;
                        if (clip_value < fabs(Param->Kmu[index])) {
                            clip_value = fabs(Param->Kmu[index]);
                        }
                    }
                }
            }

            for (iY = -4; iY < 4; iY++) {
                for (iX = -4; iX < 4; iX++) {
                    int index = iZ * Param->xyNum + (iY + iSy) * Param->xNum + iX + iSx;
                    if (fabs(Param->Kmu[index]) > clip_value) {
                        if (Param->Kmu[index] > 0) {
                            Param->Kmu[index] = clip_value;
                        } else {
                            Param->Kmu[index] = -clip_value;
                        }
                    }
                }
            }
        }
    }

    fclose(stationFp);
}

static void kernel_smooth() {
    int iX, iY, iZ;
    //smooth width in xy plane
    int width = 5;

    float *Kmu_O = (float *)malloc(Param->xyzNumKernel * sizeof(float));
    if (Kmu_O == NULL) {
        xd_abort(__func__, "fopen() failed", "Not enough memory for kernel smooth\n");
    }

    memcpy(Kmu_O, Param->Kmu, Param->xyzNumKernel * sizeof(float));
    Param->kernelMax = 0.f;
    Param->kernelMean = 0.f;
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                float total = 0.f;
                for (int sX = -width; sX <= width; sX++) {
                    for (int sY = -width; sY <= width; sY++) {
                        int index = (iZ * Param->yNum + (iY + sY)) * Param->xNum + iX + sX;

                        if (index >= 0 && index < Param->xyzNumKernel) {
                            total += Kmu_O[index];
                        }
                    }
                }
                Param->Kmu[(iZ * Param->yNum + iY) * Param->xNum + iX] = total / (width + 1) / (width + 1);
                if (Param->kernelMax < fabs(total / (width + 1) / (width + 1))) {
                    Param->kernelMax = fabs(total / (width + 1) / (width + 1));
                }
                if (iZ <= 1) {
                    Param->kernelMean += fabs(total / (width + 1) / (width + 1));
                }
            }
        }
    }

    Param->kernelMean /= Param->xyNum * 2;

    free(Kmu_O);
}

void kernel_finalize() {
    if (Param->useGPU) {
        launch_kernel_finalize(Param->d_Klambda, Param->d_Kmu, Param->d_lambda, Param->d_mu, Param->kernelZNum);
    } else {
        int iXYZ;

        for (iXYZ = 0; iXYZ < Param->xyzNum; iXYZ++) {
            //Param->Krho[iXYZ] *= -1.f;
            Param->Klambda[iXYZ] /= pow(3.f * Param->lambda[iXYZ] + 2.f * Param->mu[iXYZ], 2);
            Param->Kmu[iXYZ] /= pow(Param->mu[iXYZ], 2) * 2.f;
            Param->Kmu[iXYZ] -= 2 * Param->lambda[iXYZ] * (3.f * Param->lambda[iXYZ] + 4.f * Param->mu[iXYZ]) * Param->Klambda[iXYZ];
        }
    }
    
}

void kernel_delete() {
    if (Param->useGPU) {
        launch_cudaFreeHost(Param->Klambda);
        launch_cudaFreeHost(Param->Kmu);
        launch_cudaFreeHost(Param->sxx);
        launch_cudaFreeHost(Param->sxy);
        launch_cudaFreeHost(Param->sxz);
        launch_cudaFreeHost(Param->syy);
        launch_cudaFreeHost(Param->syz);
        launch_cudaFreeHost(Param->szz);
        launch_cudaFreeHost(Param->pk);
        launch_cudaFreeHost(Param->gk);

        launch_cudaFree(Param->d_Klambda);
        launch_cudaFree(Param->d_Kmu);

        free(Param->Kmu_all);
    } else {
        free(Param->Klambda);
        free(Param->Kmu);
        fclose(Param->kernelOutputFp);
    }

    //free(Param->Krho);
}

float get_step_length_CG() {
    float step0;
    float cScale = 2e10;

    // set the initial search direction equal to minus the initial gradient of the misfit function
    if (Param->currentIteration == 0) {
        for (int i = 0; i < Param->xyzNumKernel; i++) {
            Param->Kmu[i] *= Param->mu[i] / cScale;
            Param->gk[i] = Param->Kmu[i];
            Param->pk[i] = -Param->Kmu[i];
        }
    } else {
        float beta = 0;
        float gknorm2 = 0;

        for (int i = 0; i < Param->xyzNumKernel; i++) {
            Param->Kmu[i] *= Param->mu[i] / cScale;
            gknorm2 += Param->gk[i] * Param->gk[i];
            beta += Param->Kmu[i] * (Param->Kmu[i] - Param->gk[i]);
            Param->gk[i] = Param->Kmu[i];
        }
        beta /= gknorm2;

        for (int i = 0; i < Param->xyzNumKernel; i++) {
            Param->pk[i] = -Param->Kmu[i] + Param->pk[i] * beta;
        }
    }

    //initial test of the step
    float g0 = 0;
    for (int i = 0; i < Param->xyzNumKernel; i++) {
        g0 += Param->gk[i] * Param->pk[i];
    }

    step0 = Param->totalMisfit / g0 * 1e18 * 2;

    int bytes = Param->xyzNumKernel * sizeof(float);
    if (launch_cudaMemcpy(Param->d_Kmu, Param->pk, bytes, 0) != 0) {
        xd_abort(__func__, "malloc() failed", "Cannot copy kernel from host to device\n");
    }

    return step0;
}

float get_initial_step_length() {
    float referenceVelocity = 3000.f;
    float velocityChange, muChange;
    Param->averageMisfit /= Param->currentNumberOfStation;

    Param->averageMisfit /= 5;
    //using 3000m/s as reference velocity, make sure the test velocity change can reach 1/5 of the averageMisfit
    velocityChange = 1000.f / (1000.f / referenceVelocity - Param->averageMisfit) - referenceVelocity;
    muChange = 1.f - pow(1.f - velocityChange / referenceVelocity, 2);

    if (Param->myID == 0) {
        fprintf(stdout, "AverageMisfitPerKm: %f, velocityChangeAbsolute: %f, muChangePercentage: %f, kernelMax: %e, kernelMean: %f\n", Param->averageMisfit * 5, velocityChange, muChange, Param->kernelMax, Param->kernelMean);
        fprintf(stdout, "Step length for mu: %e\n", muChange / Param->kernelMax * 2);
    }

    return muChange / Param->kernelMax * 2;
}

void update_model() {
    float *vs = (float *)malloc(sizeof(float) * Param->xyzNumKernel);
    //int iX, iY, iZ;
    char fileName[MAX_FILENAME_LEN];
    FILE *outputModel;
    float gridMu, vsNew;

    // read initial model
    /*
    float *vs_all = (float *)malloc(sizeof(float) * Param->xyzNumKernel);

    FILE *inputModel;

    inputModel = fopen("initial.model", "rb");
    if (inputModel == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the model output file\n");
    }

    //output the coordinate information
    fread(&Param->xNum, sizeof(int), 1, inputModel);
    fread(&Param->yNum, sizeof(int), 1, inputModel);
    fread(&Param->kernelZNum, sizeof(int), 1, inputModel);
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                fread(coord, sizeof(float), 3, inputModel);
            }
        }
    }

    fread(vs_all, sizeof(float), Param->xyzNumKernel, inputModel);
*/
    for (int iIndex = 0; iIndex < Param->theVelocityModel.xNum * Param->theVelocityModel.yNum * Param->kernelZNum; iIndex++) {
        //int iZ = iIndex / Param->xyNum + 1;

        // scale the kernel with sqrt(iZ)
        //Param->mu[iIndex] += Param->Kmu[iIndex] * Param->stepLength * sqrt(iZ);

        gridMu = Param->theVelocityModel.vs[iIndex] * Param->theVelocityModel.vs[iIndex] * Param->theVelocityModel.rho[iIndex];
        gridMu += Param->Kmu_all[iIndex] * Param->stepLength;
        vsNew = sqrt(gridMu / Param->theVelocityModel.rho[iIndex]);

        if (vsNew < 2000.f) {
            vsNew = 2000.f;
        }

        if (vsNew > 8500.f / sqrt(3.)) {
            vsNew = 8500.f / sqrt(3.);
        }

        Param->theVelocityModel.vs[iIndex] = vsNew;

        /*
        if (fabs(vs[iIndex] - vs_all[iIndex]) / vs_all[iIndex] > 0.1) {
            vs[iIndex] *= 1 + copysign(0.1, vs[iIndex] - vs_all[iIndex]);
            Param->mu[iIndex] = vs[iIndex] * vs[iIndex] / Param->rho[iIndex];
        }
        */
    }

    /* output the new model */
    sprintf(fileName, "model.iter_%d", Param->currentIteration);
    outputModel = fopen(fileName, "wb");
    if (outputModel == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the model output file\n");
    }

    fwrite(&Param->theVelocityModel.xNum, sizeof(int), 1, outputModel);
    fwrite(&Param->theVelocityModel.yNum, sizeof(int), 1, outputModel);
    fwrite(&Param->zNum, sizeof(int), 1, outputModel);
    fwrite(&Param->theVelocityModel.xMin, sizeof(float), 1, outputModel);
    fwrite(&Param->theVelocityModel.yMin, sizeof(float), 1, outputModel);
    fwrite(&Param->theVelocityModel.dx, sizeof(float), 1, outputModel);
    fwrite(&Param->theVelocityModel.dz, sizeof(float), 1, outputModel);

    fwrite(Param->theVelocityModel.vs, sizeof(float), Param->theVelocityModel.gridNum, outputModel);

    free(vs);

    //free(vs_all);

    fclose(outputModel);

    //fclose(inputModel);
}

void update_model_CG() {
    float *vs = (float *)malloc(sizeof(float) * Param->xyzNumKernel);
    int iX, iY, iZ;
    char fileName[MAX_FILENAME_LEN];
    FILE *outputModel;
    float coord[3];

    for (int iIndex = 0; iIndex < Param->xyzNumKernel; iIndex++) {
        Param->mu[iIndex] += Param->pk[iIndex] * Param->stepLength;

        vs[iIndex] = sqrt(Param->mu[iIndex] * Param->rho[iIndex]);

        if (vs[iIndex] < 2000.f) {
            vs[iIndex] = 2000.f;
            Param->mu[iIndex] = 2000.f * 2000.f / Param->rho[iIndex];
        }
    }

    if (Param->useGPU) {
        if (launch_cudaMemcpy(Param->d_mu, Param->mu, sizeof(float) * Param->xyzNumKernel, 0) != 0) {
            xd_abort(__func__, "malloc() failed", "Cannot copy mu from host to device\n");
        }
    }

    /* open the kernel output file*/
    sprintf(fileName, "model.iter_%d", Param->currentIteration);
    outputModel = fopen(fileName, "wb");
    if (outputModel == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the model output file\n");
    }

    //output the coordinate information
    fwrite(&Param->xNum, sizeof(int), 1, outputModel);
    fwrite(&Param->yNum, sizeof(int), 1, outputModel);
    fwrite(&Param->kernelZNum, sizeof(int), 1, outputModel);
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        coord[2] = iZ * Param->dz;
        for (iY = 0; iY < Param->yNum; iY++) {
            coord[1] = Param->yMin + iY * Param->dy;
            for (iX = 0; iX < Param->xNum; iX++) {
                coord[0] = Param->xMin + iX * Param->dx;

                fwrite(coord, sizeof(float), 3, outputModel);
            }
        }
    }

    fwrite(vs, sizeof(float), Param->xyzNumKernel, outputModel);

    fclose(outputModel);

    free(vs);
}

void kernel_output_qc() {
	int iX, iY, iZ;
	char fileName[MAX_FILENAME_LEN];
	FILE  *outputFpMu;
	float coord[3];

	/* open the kernel output file*/
    /*
    sprintf(fileName, "kernel_lambda.iter%d", Param->currentIteration);
    outputFpLambda = fopen(fileName, "wb");
    if (outputFpLambda == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the kernel output file\n");
    }
    */
    sprintf(fileName, "kernel_mu.iter%d", Param->currentIteration);
    outputFpMu = fopen(fileName, "wb");
    if (outputFpMu == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the kernel output file\n");
    }

    //output the coordinate information
    //fwrite(&Param->xNum, sizeof(int), 1, outputFpLambda);
    //fwrite(&Param->yNum, sizeof(int), 1, outputFpLambda);
    //fwrite(&Param->kernelZNum, sizeof(int), 1, outputFpLambda);
    fwrite(&Param->theVelocityModel.xNum, sizeof(int), 1, outputFpMu);
    fwrite(&Param->theVelocityModel.yNum, sizeof(int), 1, outputFpMu);
    fwrite(&Param->kernelZNum, sizeof(int), 1, outputFpMu);
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        coord[2] = iZ * Param->dz;
        for (iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            coord[1] = Param->theVelocityModel.yMin + iY * Param->theVelocityModel.dy;
            for (iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                coord[0] = Param->theVelocityModel.xMin + iX * Param->theVelocityModel.dx;

                //fwrite(coord, sizeof(float), 3, outputFpLambda);
                fwrite(coord, sizeof(float), 3, outputFpMu);
            }
        }
    }

    //fwrite(Param->Klambda, sizeof(float), Param->xyzNumKernel, outputFpLambda);
    fwrite(Param->Kmu_all, sizeof(float), Param->theVelocityModel.xNum * Param->theVelocityModel.yNum * Param->kernelZNum, outputFpMu);

    //fclose(outputFpLambda);
    fclose(outputFpMu);
}

void read_kernel() {
    int iX, iY, iZ;
    FILE *outputFpMu;
    float coord[3];

    /* open the kernel output file*/
    outputFpMu = fopen("kernel_mu.iter0_p", "rb");
    if (outputFpMu == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open the kernel input file\n");
    }

    //output the coordinate information
    fread(&Param->theVelocityModel.xNum, sizeof(int), 1, outputFpMu);
    fread(&Param->theVelocityModel.yNum, sizeof(int), 1, outputFpMu);
    fread(&Param->kernelZNum, sizeof(int), 1, outputFpMu);
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            for (iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                fread(coord, sizeof(float), 3, outputFpMu);
            }
        }
    }

    fread(Param->Kmu_all, sizeof(float), Param->theVelocityModel.xNum * Param->theVelocityModel.yNum * Param->kernelZNum, outputFpMu);

    // scale with the maximum
    int iXYZ = 0;
    float maxf = 0;
    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            for (iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                if (fabs(Param->Kmu_all[iXYZ]) > maxf) {
                    maxf = fabs(Param->Kmu_all[iXYZ]);
                }
                iXYZ++;
            }
        }
    }

    iXYZ = 0;

    for (iZ = 0; iZ < Param->kernelZNum; iZ++) {
        for (iY = 0; iY < Param->theVelocityModel.yNum; iY++) {
            for (iX = 0; iX < Param->theVelocityModel.xNum; iX++) {
                Param->Kmu_all[iXYZ] /= -maxf;
                iXYZ++;
            }
        }
    }

    fclose(outputFpMu);
}
