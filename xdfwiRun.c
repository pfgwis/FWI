#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "xdfwi.h"

extern Param_t *Param;

int xdfwi_run() {
    FILE *srcFp;

    srcFp = fopen("source.info", "r");
    if (srcFp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open source input files!\n");
    }

    velocity_model_init();

    modeling_domain_init();

    if (Param->useGPU) {
        cuda_init();

        if (Param->theSimulationMode != 0) {
            Param->totalIteration = 1;
            kernel_init();

            Param->allMisfit = (float *)malloc(sizeof(int) * 20000);
        } else {
            Param->totalIteration = 1;
        }

        for (Param->currentIteration = 0; Param->currentIteration < Param->totalIteration; Param->currentIteration++) {
            fprintf(stdout, "Iteration: %d\n", Param->currentIteration);

            Param->totalMisfit = 0.f;
            Param->averageMisfit = 0.f;
            Param->currentNumberOfStation = 0;

            // continious form last iteration
            if (0) {
                read_kernel();
                Param->stepLength = 7.752223e08;
                //launch_update_model(Param->stepLength, Param->d_Kmu, Param->d_mu, Param->d_rho, Param->kernelZNum);
                update_model();
            }

            rewind(srcFp);
            Param->theSrcID = 0;
            while (fscanf(srcFp, "%f %f", &Param->theSrcLon, &Param->theSrcLat) != EOF) {
                if (Param->theSrcID % 200 != 0) {
                    Param->theSrcID++;
                    continue;
                }

                if (Param->theSimulationMode != 0) {
                    if (launch_cudaMemset(Param->d_Kmu, 0, Param->xyzNumKernel * sizeof(float)) != 0) {
                        xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the kernel\n");
                    }
                }
                
                if (Param->myID == 0 && Param->theSrcID % 1 == 0) {
                    //fprintf(stdout, "%d... of 244\n", Param->theSrcID + 1);
                }

                cuda_set_field_parameter();

                source_init();

                station_init();

                if (Param->myNumberOfStations == 0) {
                    source_delete();
                    //station_delete();
                    Param->theSrcID++;
                    continue;
                }

                if (Param->theSimulationMode != 0) {
                    forward_modeling_gpu_memory();

                    backward_modeling_gpu();

                    kernel_finalize();

                    kernel_add_all();
                } else {
                    forward_modeling_gpu_memory();
                }

                source_delete();

                station_delete();

                Param->theSrcID++;
            }

            if (Param->theSimulationMode != 0) {
                //kernel_processing();

                kernel_output_qc();

                for (int iS = 0; iS < Param->currentNumberOfStation; iS++) {
                    fprintf(stdout, "%f\n", Param->allMisfit[iS]);
                }
/*
                read_kernel();

                // calulate the step length
                //Param->stepLength = get_step_length_CG();

                fprintf(stdout, "Initail step length: %e\n", Param->stepLength);

                float oldMisfit[12];
                oldMisfit[0] = 1310.47;
                //fprintf(stdout, "Initial old misfit: %f, number of stations: %d\n", Param->totalMisfit, Param->currentNumberOfStation);

		        Param->stepLength = 5e8;
                for (int iCG = 0; iCG <= 1; iCG++) {
                    //launch_update_model(Param->stepLength, Param->d_Kmu, Param->d_mu, Param->d_rho, Param->kernelZNum);
                    update_model();

                    rewind(srcFp);
                    Param->theSrcID = 0;

                    Param->totalMisfit = 0.f;
                    Param->averageMisfit = 0.f;
                    Param->currentNumberOfStation = 0;
                    while (fscanf(srcFp, "%f %f", &Param->theSrcLon, &Param->theSrcLat) != EOF) {
                        if (Param->theSrcID % 1 != 0) {
                            Param->theSrcID++;
                            continue;
                        }

                        if (Param->myID == 0 && Param->theSrcID % 50 == 0) {
                            //fprintf(stdout, "%d... of 244\n", Param->theSrcID + 1);
                        }

                        cuda_set_field_parameter();

                        source_init();

                        station_init();

                        if (Param->myNumberOfStations == 0) {
                            source_delete();
                            //station_delete();
                            Param->theSrcID++;
                            continue;
                        }

                        forward_modeling_gpu();

                        compute_residual();

                        source_delete();

                        station_delete();

                        Param->theSrcID++;
                    }

                    oldMisfit[iCG + 1] = Param->totalMisfit;

                    fprintf(stdout, "Misfit: %f\n", Param->totalMisfit);
                }

                // interpolation to get the minimum
                if ( fabs(oldMisfit[2] + oldMisfit[0] - 2 * oldMisfit[1]) > 1e-6 ) {
                    Param->stepLength *= (1 - (oldMisfit[2] - oldMisfit[0]) / ((oldMisfit[2] + oldMisfit[0]) - 2 * oldMisfit[1]) / 2);
                }

                fprintf(stdout, "New step length: %e\n", Param->stepLength);

                //update_model_CG();
                */
            }
        }

        if (Param->allMisfit) {
            free(Param->allMisfit);
        }
    } else {
        cpu_init();
        if (Param->theSimulationMode != 0) {
            kernel_init();
        }
        forward_modeling_cpu();
        if (Param->theSimulationMode != 0) {
            backward_modeling_cpu();
        }
    }

    fclose(srcFp);

    return 0;
}
