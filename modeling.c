#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "xdfwi.h"

extern Param_t* Param;

//local function
void iso_modeling_cpu(int iTime);
void iso_modeling_gpu(int iTime);

void forward_modeling_gpu_memory() {
    int   iTime, iStage, isForward, iOutput = 0;

    if (Param->outputWavefield) {
        wavefield_extract_init();
    }

    cuda_set_field_memory_zero();

    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        station_extract(iTime);

        if (iTime >= Param->theOutputStartTimeStep && (iTime - Param->theOutputStartTimeStep) % Param->theOutputRate == 0) {
            cuda_download_wavefield(iOutput);
            iOutput++;
        }

        for (iStage = 0; iStage < Param->nRKStage; iStage++) {
            isForward = (iTime + iStage) % 2;

            cuda_df_dxyz(isForward);
            cuda_pml(isForward, iStage);
            cuda_free_surface(isForward);
            source_inject_forward_multiple_stage(iTime, iStage);

            if (iStage == 0 && iTime % Param->theOutputRate == 0) {
                launch_cuda_stream_synchronize();
                if (Param->outputWavefield) {
                    wavefield_extract();
                }
            }

            cuda_update_f(iStage);
        }
    }

    if (Param->outputStation || Param->theSimulationMode) {
        station_download();
    }

    launch_cuda_device_synchronize();

    if (Param->outputStation) {
        station_output_qc();
    }
}

void backward_modeling_gpu() {
    int iTime, iStage, iOutput, isForward;

    compute_residual();

    upload_residual();

    cuda_set_field_memory_zero();

    iOutput = (Param->theTotalTimeSteps - Param->theOutputStartTimeStep) / Param->theOutputRate;

    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        if (Param->theTotalTimeSteps - iTime - 1 < Param->theOutputStartTimeStep) {
            break;
        }

        /* kernel calculate */
        if ((Param->theTotalTimeSteps - iTime - 1 - Param->theOutputStartTimeStep) % Param->theOutputRate == 0) {
            cuda_upload_wavefield(iOutput);
            kernel_backward_xcoor();
            iOutput--;
        }

        for (iStage = 0; iStage < Param->nRKStage; iStage++) {
            isForward = (iTime + iStage) % 2;

            cuda_df_dxyz(isForward);
            cuda_pml(isForward, iStage);
            cuda_free_surface(isForward);
            source_inject_backward(iTime * Param->nRKStage + iStage);

            cuda_update_f(iStage);
        }
    }

    //kernel_processing();

    launch_cuda_device_synchronize();
}

void forward_modeling_gpu() {
    int   iTime, iStage, isForward;

    cuda_set_field_memory_zero();

    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        station_extract(iTime);

        for (iStage = 0; iStage < Param->nRKStage; iStage++) {
            isForward = (iTime + iStage) % 2;

            cuda_df_dxyz(isForward);
            cuda_pml(isForward, iStage);
            cuda_free_surface(isForward);
            source_inject_forward_multiple_stage(iTime, iStage);
            cuda_update_f(iStage);
        }
    }

    station_download();

    launch_cuda_device_synchronize();
}

void forward_modeling_cpu() {
    int    iTime;
    double stationExtractTime = 0;
    double kernelForwardExtractTime = 0;
    double sourceInjectTime = 0;
    double modelingTime = 0;
    double totalModelingTime = 0;
    double wavefieldExtractTime = 0;

    if (Param->myID == 0) {
        fprintf(stdout ,"\nStart forward modeling:\n");
    }

    if (Param->outputWavefield) {
        wavefield_extract_init();
    }

    totalModelingTime -= MPI_Wtime();
    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        if (iTime % 1 == 0) {
            fprintf(stdout ,"Time step: %d\n", iTime);
        }
        
        //wavefield extract: vy1 at timestep t
        if (Param->outputWavefield && iTime % Param->theOutputRate == 0) {
            wavefieldExtractTime -= MPI_Wtime();
            wavefield_extract();
            wavefieldExtractTime += MPI_Wtime();
        }

        /* Output the forward wavefield for calculating kernel and residual later */
        if (Param->theSimulationMode != 0 && iTime % Param->theOutputRate == 0) {
            kernelForwardExtractTime -= MPI_Wtime();
            kernel_forward_extract();
            kernelForwardExtractTime += MPI_Wtime();
        }

        //station extract: vx1 at timestep t
        stationExtractTime -= MPI_Wtime();
        station_extract(iTime);
        stationExtractTime += MPI_Wtime();

        //source inject at time step t
        sourceInjectTime -= MPI_Wtime();
        source_inject_forward(iTime);
        sourceInjectTime += MPI_Wtime();

        modelingTime -= MPI_Wtime();
        iso_modeling_cpu(iTime);
        modelingTime += MPI_Wtime();
    }

    totalModelingTime += MPI_Wtime();

    if (Param->outputStation) {
        station_output_qc();
    }

    //if (Param->theSimulationMode != 0) print_time_status(&kernelForwardExtractTime, "\tKernel extract time");
    if (Param->outputWavefield) {
        print_time_status(&wavefieldExtractTime, "Wavefield extract time");
    }
    print_time_status(&stationExtractTime, "Station extract time");
    print_time_status(&sourceInjectTime, "Source injecting time");
    print_time_status(&modelingTime, "Modeling time");
    print_time_status(&totalModelingTime, "Total modeling time");
}

void backward_modeling_cpu() {
    int    iTime;
    double kernelBackwardXcorrTime = 0;
    double sourceInjectTime = 0;
    double modelingTime = 0;
    double totalModelingTime = 0;

    if (Param->myID == 0) {
        fprintf(stdout ,"\nStart backward modeling:\n");
    }

    compute_residual();

    memory_set_zeros();

    totalModelingTime -= MPI_Wtime();
    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        if (iTime % 50 == 0) {
            fprintf(stdout ,"Time step: %d\n", iTime);
        }

        /* kernel calculate */
        if ((Param->theTotalTimeSteps - iTime - 1) % Param->theOutputRate == 0) {
            kernelBackwardXcorrTime -= MPI_Wtime();
            kernel_backward_xcoor();
            kernelBackwardXcorrTime += MPI_Wtime();
        }

        //source inject at time step t
        sourceInjectTime -= MPI_Wtime();
        source_inject_backward(iTime);
        sourceInjectTime += MPI_Wtime();

        modelingTime -= MPI_Wtime();
        iso_modeling_cpu(iTime);
        modelingTime += MPI_Wtime();
    }
    totalModelingTime += MPI_Wtime();

    kernel_finalize();

    kernel_output_qc();

    print_time_status(&kernelBackwardXcorrTime, "Kernel backward crosscorrelation time");
    print_time_status(&sourceInjectTime, "Source injecting time");
    print_time_status(&modelingTime, "Modeling time");
    print_time_status(&totalModelingTime, "Total modeling time");
}

void iso_modeling_gpu(int iTime) {
    
}

void iso_modeling_cpu(int iTime) {
    int iStage, isForward;

    for (iStage = 0; iStage < Param->nRKStage; iStage++) {
        isForward = (iTime + iStage) % 2;
        
        cpu_scale_df(iStage);

        cpu_x_derivative(isForward);
        cpu_y_derivative(isForward);
        cpu_z_derivative(isForward);

        cpu_free_surface(isForward);

        cpu_update_f(iStage);

        cpu_pml_combine();
    }
}
