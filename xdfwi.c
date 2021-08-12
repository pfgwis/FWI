#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "xdfwi.h"

//make parameters as a global variable
extern Param_t* Param;

/*
 * Program's standard entry point.
 */
int main(int argc, char* argv[]) {
    double theTotalRunningTime = -MPI_Wtime();

    //MPI initialization
    MPI_Init(&argc, &argv);

    Param = (Param_t *)malloc(sizeof(Param_t));
    if (Param == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for Param\n");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &Param->myID);
    MPI_Comm_size(MPI_COMM_WORLD, &Param->theGroupSize);

    if (Param->myID == 0) {
        fprintf(stdout, "Starting XDFWI 1.0 on %d PEs\n\n", Param->theGroupSize);
    }

    //init parameters from the input file
    parameter_init(argc, argv);

    theTotalRunningTime = -MPI_Wtime();
    //running xdfwi
    xdfwi_run();

    theTotalRunningTime += MPI_Wtime();
    print_time_status(&theTotalRunningTime, "XDFWI total running time");
    fprintf(stdout, "XDFWI finished successfully\n");

    // free the parameters
    parameter_free();

    MPI_Finalize();

    return 0;
}
