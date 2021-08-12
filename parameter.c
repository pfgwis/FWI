#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "xdfwi.h"

extern Param_t* Param;

//local functions
static int parameter_read();
static void parameter_broadcast();

int parameter_init(int argc, char** argv) {
    //xdfwi running syntax
    if (argc != 2) {
        if (Param->myID == 0) {
            fprintf(stderr, "Usage: xdfwi <parameters.in>\n\n");
            MPI_Finalize();
            exit(1);
        }
    }

    //get the input parameter file name
    strcpy(Param->theParametersInputFileName, argv[1]);

    if (Param->myID == 0) {
        fprintf(stdout, "Start to init parameters.\n");

        if (parameter_read() != 0) {
            xd_abort(__func__, "parameter_read() fail", "Error init input parameters\n");
        }
    }

    parameter_broadcast();

    //parameter with default value
    Param->pmlNum = 10;
    //Param->xMin = -320000.;
    //Param->yMin = -400000.;

    // we are doing the vertical resolution test
    Param->vMin = 4700.f;
    Param->vMax = 8500.f;

    //modeling domain
    Param->dx = Param->vMin / Param->theFreq / 8.f;
    Param->dx = 2500.f;
    Param->dy = Param->dx;
    Param->dz = Param->dx;
    Param->theDeltaT = 0.45f * Param->dx / Param->vMax;
    Param->theOutputRate = (int)(1.f / Param->theFreq / Param->theDeltaT / 6.f);
    //Param->theOutputRate = 500;
    Param->theOutputStartTimeStep = (int)(20.f / Param->theDeltaT);
    //Param->theOutputStartTimeStep = 0;
    fprintf(stdout, "dx: %f, dt: %f, theOutputRate: %d\n", Param->dx, Param->theDeltaT, Param->theOutputRate);
    return 0;
}

/**
 * Open material database and initialize various static global variables.
 */
static int parameter_read() {
    FILE  *fp;
    int   output_station, output_wavefield, simulation_mode, core_mode;

    //obtain the specficiation of the simulation
    fp = fopen(Param->theParametersInputFileName, "r");
    if (fp == NULL) {
        xd_abort(__func__, "fopen() fail", "Error open the input parameter file: %s\n", Param->theParametersInputFileName);
    }

    if ((parsetext(fp, "simulation_wave_max_freq_hz", 'f', &Param->theFreq)             != 0) ||
        (parsetext(fp, "velocity_model_path",         's', Param->theVelocityModelPath) != 0) ||
        (parsetext(fp, "x_direction_num",             'i', &Param->xNum)                != 0) ||
        (parsetext(fp, "y_direction_num",             'i', &Param->yNum)                != 0) ||
        (parsetext(fp, "z_direction_num",             'i', &Param->zNum)                != 0) ||
        (parsetext(fp, "total_time_step",             'i', &Param->theTotalTimeSteps)   != 0)) {
            fprintf(stderr, "Error reading parameters from %s\n", Param->theParametersInputFileName);
            return -1;
    }

    //optional parameters
    if (parsetext(fp, "use_GPU", 'i', &core_mode) != 0)
        Param->useGPU = 0;
    else
        Param->useGPU = core_mode;

    if (parsetext(fp, "output_station", 'i', &output_station) != 0)
        Param->outputStation = 0;
    else
        Param->outputStation = output_station;

    if (parsetext(fp, "output_wavefield", 'i', &output_wavefield) != 0)
        Param->outputWavefield = 0;
    else
        Param->outputWavefield = output_wavefield;

    if (parsetext(fp, "simulation_mode", 'i', &simulation_mode) != 0)
        Param->theSimulationMode = 0;
    else
        Param->theSimulationMode = simulation_mode;

    /* Sanity check */
    if (Param->theFreq <= 0) {
        fprintf(stderr, "Illegal frequency value %f\n", Param->theFreq);
        return -1;
    }

    return 0;
}

static void parameter_broadcast() {
    int   int_message[7];
    float float_message[1];

    float_message[0]  = Param->theFreq;

    MPI_Bcast(float_message, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    Param->theFreq = float_message[0];

    int_message[0] = Param->outputStation;
    int_message[1] = Param->outputWavefield;
    int_message[2] = Param->theSimulationMode;
    int_message[3] = Param->useGPU;
    int_message[4] = Param->xNum;
    int_message[5] = Param->yNum;
    int_message[6] = Param->zNum;

    MPI_Bcast(int_message, 7, MPI_INT, 0, MPI_COMM_WORLD);

    Param->outputStation     = int_message[0];
    Param->outputWavefield   = int_message[1];
    Param->theSimulationMode = int_message[2];
    Param->useGPU            = int_message[3];
    Param->xNum              = int_message[4];
    Param->yNum              = int_message[5];
    Param->zNum              = int_message[6];
}

void parameter_free() {
    if (Param->useGPU) {
        cuda_free_all_field_memory();
    } else {
        cpu_free_all_field_memory();
    }

    if (Param->outputWavefield) {
        fclose(Param->theWavefieldOutFp);
        if (Param->useGPU) {
            launch_cudaFreeHost(Param->vz);
        }
    }

    if (Param->theSimulationMode != 0) {
        kernel_delete();
    }

    velocity_model_delete();

    free(Param->lambda);
    free(Param->mu);
    free(Param->rho);

    free(Param);
}
