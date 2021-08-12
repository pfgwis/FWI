#include <stdio.h>
#include <stdlib.h>

#include "cudaKernel.cuh"

static cudaStream_t dataTransferStream;
static cudaStream_t modelingStream;

extern "C"
int launch_cudaMalloc(void **devPtr, size_t size) {
	if (cudaMalloc(devPtr, size) != cudaSuccess) {
		return -1;
	}

	return 0;
}

extern "C"
int  launch_cudaMallocHost(void **hostPtr, size_t size) {
    if (cudaMallocHost(hostPtr, size) != cudaSuccess) {
        return -1;
    }

    return 0;
}

extern "C"
int launch_cudaMemset(void *devPtr, int value, size_t count) {
	if (cudaMemset(devPtr, value, count) != cudaSuccess) {
		return -1;
	}

	return 0;
}

extern "C"
int launch_cudaMemcpy(void *dst, const void *src, size_t count, int direction) {
	if (direction == 0) {
		if (cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) != cudaSuccess) {
			return -1;
		}
	} else {
		if (cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) != cudaSuccess) {
			return -1;
		}
	}

	return 0;
}

extern "C"
void launch_cudaMemcpyAsync(void *dst, const void *src, size_t count, int direction) {
    if (direction == 0) {
        if (cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, dataTransferStream) != cudaSuccess) {
            fprintf(stdout, "Error cudaMemcpyAsync from host to device!\n");
            exit(0);
        }
    } else {
        if (cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, dataTransferStream) != cudaSuccess) {
            fprintf(stdout, "Error cudaMemcpyAsync from device to host!\n");
            exit(0);
        }
    }
}

extern "C"
void launch_cudaFree(void *devPtr) {
	cudaFree(devPtr);
}

extern "C"
void launch_cudaFreeHost(void *hostPtr) {
    cudaFreeHost(hostPtr);
}

extern "C"
void launch_dstress_dxy(float *fx_in, float *fy_in, float *f_out, int isForward) {
    if (isForward) {
        dstress_dxy_forward<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>>(fx_in, fy_in, f_out);
    } else {
        dstress_dxy_backward<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>>(fx_in, fy_in, f_out);
    }
}

extern "C"
void launch_dstress_dz(float *f_in, float *f_out, int isForward) {
    if (isForward) {
        dstress_dz_forward<<<dim3(NX/32, NY, 1), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    } else {
        dstress_dz_backward<<<dim3(NX/32, NY, 1), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    }
}

extern "C"
void launch_dvxy_dxy(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *dszz_out, int isForward) {
    if (isForward) {
        dvxy_dxy_forward<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>> (vx_in, vy_in, lambda_in, dsxy_out, dsxx_out, dsyy_out, dszz_out);
    } else {
        dvxy_dxy_backward<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>> (vx_in, vy_in, lambda_in, dsxy_out, dsxx_out, dsyy_out, dszz_out);
    }
}

extern "C"
void launch_dvxz_dxz(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *dszz_out, int isForward) {
    if (isForward) {
        dvxz_dxz_forward<<<dim3(NX/32, NY, NZ/32), dim3(32, 32, 1), 0, modelingStream>>> (vx_in, vz_in, lambda_in, dsxz_out, dsxx_out, dsyy_out, dszz_out);
    } else {
        dvxz_dxz_backward<<<dim3(NX/32, NY, NZ/32), dim3(32, 32, 1), 0, modelingStream>>> (vx_in, vz_in, lambda_in, dsxz_out, dsxx_out, dsyy_out, dszz_out);
    }
}

extern "C"
void launch_dvelocity_dy(float *f_in, float *f_out, int isForward) {
    if (isForward) {
        dvelocity_dy_forward_32x32<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    } else {
        dvelocity_dy_backward_32x32<<<dim3(NX/32, NY/32, NZ), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    }
}

extern "C"
void launch_dvelocity_dz(float *f_in, float *f_out, int isForward) {
    if (isForward) {
        dvelocity_dz_forward<<<dim3(NX/32, NY, 1), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    } else {
        dvelocity_dz_backward<<<dim3(NX/32, NY, 1), dim3(32, 32, 1), 0, modelingStream>>>(f_in, f_out);
    }
}

extern "C"
void launch_pml_frontv(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_frontv_forward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(vx, vy, vz, vxx, vyx, vzx, dvxx, dvyx, dvzx, dsxx, dsyy, dszz, dsxy, dsxz, lambda_in, scaleA, scaleB);    
    } else {
        pml_frontv_backward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(vx, vy, vz, vxx, vyx, vzx, dvxx, dvyx, dvzx, dsxx, dsyy, dszz, dsxy, dsxz, lambda_in, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_fronts(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_fronts_forward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(sxx, sxy, sxz, sxxx, sxyx, sxzx, dsxxx, dsxyx, dsxzx, dvx, dvy, dvz, scaleA, scaleB);
    } else {
        pml_fronts_backward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(sxx, sxy, sxz, sxxx, sxyx, sxzx, dsxxx, dsxyx, dsxzx, dvx, dvy, dvz, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_backv(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_backv_forward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(vx, vy, vz, vxx, vyx, vzx, dvxx, dvyx, dvzx, dsxx, dsyy, dszz, dsxy, dsxz, lambda_in, scaleA, scaleB);
    } else {
        pml_backv_backward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(vx, vy, vz, vxx, vyx, vzx, dvxx, dvyx, dvzx, dsxx, dsyy, dszz, dsxy, dsxz, lambda_in, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_backs(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_backs_forward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(sxx, sxy, sxz, sxxx, sxyx, sxzx, dsxxx, dsxyx, dsxzx, dvx, dvy, dvz, scaleA, scaleB);
    } else {
        pml_backs_backward<<<dim3(NY/32, NZ, 1), dim3(32, 32, 1), 0, modelingStream>>>(sxx, sxy, sxz, sxxx, sxyx, sxzx, dsxxx, dsxyx, dsxzx, dvx, dvy, dvz, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_leftv(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_leftv_forward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxy, vyy, vzy, dvxy, dvyy, dvzy, dsxx, dsyy, dszz, dsxy, dsyz, lambda_in, scaleA, scaleB);
    } else {
        pml_leftv_backward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxy, vyy, vzy, dvxy, dvyy, dvzy, dsxx, dsyy, dszz, dsxy, dsyz, lambda_in, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_lefts(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_lefts_forward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxy, syy, syz, sxyy, syyy, syzy, dsxyy, dsyyy, dsyzy, dvx, dvy, dvz, scaleA, scaleB);
    
    } else {
        pml_lefts_backward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxy, syy, syz, sxyy, syyy, syzy, dsxyy, dsyyy, dsyzy, dvx, dvy, dvz, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_rightv(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_rightv_forward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxy, vyy, vzy, dvxy, dvyy, dvzy, dsxx, dsyy, dszz, dsxy, dsyz, lambda_in, scaleA, scaleB);
    } else {
        pml_rightv_backward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxy, vyy, vzy, dvxy, dvyy, dvzy, dsxx, dsyy, dszz, dsxy, dsyz, lambda_in, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_rights(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_rights_forward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxy, syy, syz, sxyy, syyy, syzy, dsxyy, dsyyy, dsyzy, dvx, dvy, dvz, scaleA, scaleB);
    
    } else {
        pml_rights_backward<<<dim3(NX/32, NZ, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxy, syy, syz, sxyy, syyy, syzy, dsxyy, dsyyy, dsyzy, dvx, dvy, dvz, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_bottomv(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_bottomv_forward<<<dim3(NX/32, NY, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxz, vyz, vzz, dvxz, dvyz, dvzz, dsxx, dsyy, dszz, dsxz, dsyz, lambda_in, scaleA, scaleB);
    } else {
        pml_bottomv_backward<<<dim3(NX/32, NY, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(vx, vy, vz, vxz, vyz, vzz, dvxz, dvyz, dvzz, dsxx, dsyy, dszz, dsxz, dsyz, lambda_in, scaleA, scaleB);
    }
}

extern "C"
void launch_pml_bottoms(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward) {
    if (isForward) {
        pml_bottoms_forward<<<dim3(NX/32, NY, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxz, syz, szz, sxzz, syzz, szzz, dsxzz, dsyzz, dszzz, dvx, dvy, dvz, scaleA, scaleB);   
    } else {
        pml_bottoms_backward<<<dim3(NX/32, NY, 1), dim3(32, PML_NUM, 1), 0, modelingStream>>>(sxz, syz, szz, sxzz, syzz, szzz, dsxzz, dsyzz, dszzz, dvx, dvy, dvz, scaleA, scaleB);
    }
}

extern "C"
void launch_free_surface(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in, float *vx, float *vy, float *vz, int isForward) {
    if (isForward) {
        free_surface_forward<<<NY, NX, 0, modelingStream>>>(dsxz, dsyz, dszz, dsxx, dsyy, lambda_in);
    } else {
        free_surface_backward<<<NY, NX, 0, modelingStream>>>(dsxz, dsyz, dszz, dsxx, dsyy, lambda_in, vx, vy, vz);
    }
}

extern "C"
int launch_set_modeling_parameter(float dt, float dx) {
    if (set_constant_memory(dt, dx)           != 0           ||
        cudaStreamCreate(&dataTransferStream) != cudaSuccess ||
        cudaStreamCreate(&modelingStream)     != cudaSuccess) {
        return -1;
    }

    return 0;
}

extern "C"
void launch_kernel_backward_xcoor(float *sxx1, float *sxy1, float *sxz1, float *syy1, float *syz1, float *szz1, float *sxx2, float *sxy2, float *sxz2, float *syy2, float *syz2, float *szz2, float *Klambda, float *Kmu, int zNum) {
    backward_xcoor<<<dim3(NY, zNum, 1), NX, 0, modelingStream>>>(sxx1, sxy1, sxz1, syy1, syz1, szz1, sxx2, sxy2, sxz2, syy2, syz2, szz2, Klambda, Kmu);
}

extern "C"
void launch_kernel_finalize(float *Klambda, float *Kmu, float *lambda, float *mu, int zNum) {
    kernel_processing<<<dim3(NY, zNum, 1), NX, 0, modelingStream>>>(Klambda, Kmu, lambda, mu);
}

extern "C"
void launch_station_clip(float *Klambda0, float *Kmu0, int *iX, int *iY, int numberOfStations) {
    station_clip<<<dim3(numberOfStations, 5), dim3(16, 16), 0, modelingStream>>>(Klambda0, Kmu0, iX, iY);
}

extern "C"
void launch_add_kernel(float *Klambda0, float *Kmu0, float *Klambda, float *Kmu, int zNum) {
    add_kernel<<<dim3(NY, zNum, 1), NX, 0, modelingStream>>>(Klambda0, Kmu0, Klambda, Kmu);
}

extern "C"
void launch_update_stress(float *sxx, float *dsxx, float *syy, float *dsyy, float *szz, float *dszz, float *sxy, float *dsxy, float *sxz, float *dsxz, float *syz, float *dsyz, float *mu, float scaleA, float scaleB) {
    update_stress<<<dim3(NY, NZ, 1), NX, 0, modelingStream>>>(sxx, dsxx, syy, dsyy, szz, dszz, sxy, dsxy, sxz, dsxz, syz, dsyz, mu, scaleA, scaleB);
}

extern "C"
void launch_update_velocity(float *fx, float *dfx, float *fy, float *dfy, float *fz, float *dfz, float *rho, float scaleA, float scaleB) {
    update_velocity<<<dim3(NY, NZ, 1), NX, 0, modelingStream>>>(fx, dfx, fy, dfy, fz, dfz, rho, scaleA, scaleB);
}

extern "C"
void launch_source_inject_forward(float *f, int x, int y, float *skx, float *sky, float stf) {
    source_inject<<<1, dim3(16, 16, 1), 0, modelingStream>>>(f, x, y, skx, sky, stf);
}

extern "C"
void launch_source_inject_forward_gaussian(float *f, int x, int y, float skx, float sky, float stf) {
    source_inject_gaussian<<<1, dim3(16, 16, 1), 0, modelingStream>>>(f, x, y, skx, sky, stf);
}

extern "C"
void launch_source_inject_backward(int *iX, int *iY, float *kx, float *ky, float *vz, float *f, int numberOfStations) {
    source_inject_station<<<numberOfStations, dim3(16, 16, 1), 0, modelingStream>>>(iX, iY, kx, ky, vz, f);
}

extern "C"
void launch_source_inject_backward_gaussian(int *iX, int *iY, float *kx, float *ky, float *vz, float *f, int numberOfStations) {
    source_inject_station_gaussian<<<numberOfStations, dim3(16, 16, 1), 0, modelingStream>>>(iX, iY, kx, ky, vz, f);
}

extern "C"
void launch_station_extract(int *iX, int *iY, float *x, float *y, float *vz_out, float *vz_in, int myNumberOfStations) {
    station_extract<<<myNumberOfStations, dim3(16, 16, 1), 0, modelingStream>>>(iX, iY, x, y, vz_out, vz_in);
}

extern "C"
void launch_update_model(float stepLength, float *Kmu, float *mu, float *rho, int zNum) {
    update_mu<<<dim3(NY, zNum, 1), NX, 0, modelingStream>>>(stepLength, Kmu, mu, rho);
}

extern "C"
void launch_cuda_device_synchronize() {
    cudaDeviceSynchronize();
}

extern "C"
void launch_cuda_stream_synchronize() {
    cudaStreamSynchronize(dataTransferStream);
    cudaStreamSynchronize(modelingStream);
}

extern "C"
void launch_delete_modeling_parameter() {
    cudaStreamDestroy(dataTransferStream);
    cudaStreamDestroy(modelingStream);
}

/*
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice Name: %s\n", prop.name);
    printf("Total global memory: %fGB\n", prop.totalGlobalMem/1024./1024./1024.);
    printf("Total const memory: %u\n", prop.totalConstMem);
    printf("Shared memory per block: %u\n", prop.sharedMemPerBlock);
    printf("Texture alignment: %u\n", prop.textureAlignment);
    printf("Regs per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %u\n", prop.maxThreadsPerBlock);
    printf("Multi processor counts: %d\n", prop.multiProcessorCount);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    cudaEvent_t startEvent, stopEvent;
    float time;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    
    for (int i = 0; i < 1000; i++) {
        dstress_dx_forward<<<dim3(NY/32, NZ, 1), dim3(32, 32)>>>(f_in, f_out);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);

    fprintf(stdout, "Total running time: %f\n", time);
    fprintf(stdout, "Bandwidth (GB/s): %f\n", 2 * 256. * 256. * 256. * sizeof(float) * 1000. / 1024. / 1024 / time);
*/