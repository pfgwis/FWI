#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "xdfwi.h"

extern Param_t* Param;

//local functions
static void print_memory_statistic();

int modeling_domain_init() {
    if (Param->myID == 0) {
        fprintf(stdout, "\nStart to init the simulation domain.\n");
    }

    Param->xyNum  = Param->xNum * Param->yNum;
    Param->xzNum  = Param->xNum * Param->zNum;
    Param->yzNum  = Param->yNum * Param->zNum;
    Param->xyzNum = Param->xyNum * Param->zNum;

    Param->kernelZNum   = 20;
    Param->xyzNumKernel = Param->xyNum * Param->kernelZNum;

    int bytes = Param->xyzNum * sizeof(float);

    Param->lambda  = (float *)malloc(bytes);
    Param->mu      = (float *)malloc(bytes);
    Param->rho     = (float *)malloc(bytes);

    if (Param->lambda == NULL || Param->mu == NULL || Param->rho == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for lambda\n");
    }

    //pml_map_init();

    fprintf(stdout, "xNum: %d, yNum: %d, zNum: %d, totalNum: %d\n", Param->xNum, Param->yNum, Param->zNum, Param->xyzNum);

    print_memory_statistic();

    return 0;
}

static void print_memory_statistic() {
    float sizeOfOneCubeXYZMB = Param->xyzNum * sizeof(float) / 1024. / 1024.;
    float totalMemory = 0.f;

    //total wavefield: 21 cubes
    totalMemory += 21 * sizeOfOneCubeXYZMB;

    //inversion will have 3 more cubes
    if (Param->theSimulationMode != 0) {
        totalMemory += 2 * sizeOfOneCubeXYZMB;
    }

    //pml
    totalMemory += (Param->xyNum + 2.f * Param->xzNum + 2.f * Param->yzNum) * Param->pmlNum * sizeOfOneCubeXYZMB * 12.f / Param->xyzNum;

    if (Param->myID == 0) {
        fprintf(stdout, "Total GPU memory needed is %.1fMB.\n", totalMemory);
    }

    totalMemory = 0.f;

    if (Param->theSimulationMode != 0) {
        totalMemory += 6 * ((Param->theTotalTimeSteps - Param->theOutputStartTimeStep) / Param->theOutputRate + 1) * (Param->xyzNumKernel * sizeof(float) / 1024.f / 1024.f);

        if (Param->myID == 0) {
            fprintf(stdout, "Total CPU memory needed is %.1fGB.\n", totalMemory / 1024.f);
        }
    }
}
