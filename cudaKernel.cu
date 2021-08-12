#include "cudaKernel.cuh"

//fd coefficients
__constant__ float c_fd[5];
__constant__ float c_fd_fs[3];

__constant__ float c_d0;
__constant__ float c_beta0;
__constant__ float c_alpha0;
__constant__ float c_alpha01;

__constant__ float c_dt;

int set_constant_memory(float dt, float dx) {
    float fd[5];
    float d0, beta0, alpha0, alpha01;
    float dtOverDx = dt / dx;

    fd[0] = -0.30874f * dtOverDx;
    fd[1] = -0.6326f  * dtOverDx;
    fd[2] = 1.2330f   * dtOverDx;
    fd[3] = -0.3334f  * dtOverDx;
    fd[4] = 0.04168f  * dtOverDx;

    if (cudaMemcpyToSymbol(c_fd, fd, 5 * sizeof(float)) != cudaSuccess) {
        return -1;
    }

    fd[0] = 0.25f  * dtOverDx;
    fd[1] = 1.00f  * dtOverDx;
    fd[2] = -1.25f * dtOverDx;

    if (cudaMemcpyToSymbol(c_fd_fs, fd, 3 * sizeof(float)) != cudaSuccess) {
        return -1;
    }

    d0 = 4.5f * 8500.f / PML_NUM / PML_NUM / PML_NUM / dx;
    beta0 = (8.0f - 1.f) / PML_NUM / PML_NUM;
    alpha0 = 3.14f * 0.033f;
    alpha01 = alpha0 / PML_NUM;

    if (cudaMemcpyToSymbol(c_d0,      &d0,      sizeof(float)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_beta0,   &beta0,   sizeof(float)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_alpha0,  &alpha0,  sizeof(float)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_alpha01, &alpha01, sizeof(float)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_dt, &dt, sizeof(float)) != cudaSuccess) {
        return -1;
    }

    return 0;
}

__global__ void dstress_dz_forward(float *f_in, float *f_out) {
    __shared__ float s_f[NZ + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;

    int si = threadIdx.x;

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[k + 1][si] = f_in[globalIdx];
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        s_f[0][si]      = -s_f[2][si];
        s_f[NZ + 1][si] = 0.f;
        s_f[NZ + 2][si] = 0.f;
        s_f[NZ + 3][si] = 0.f;
    }

    __syncthreads();

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        float dsdz = c_fd[0] * s_f[k][si]     + 
                     c_fd[1] * s_f[k + 1][si] +
                     c_fd[2] * s_f[k + 2][si] +
                     c_fd[3] * s_f[k + 3][si] +
                     c_fd[4] * s_f[k + 4][si];

        f_out[globalIdx] += dsdz;
    }
}

__global__ void dstress_dz_backward(float *f_in, float *f_out) {
    __shared__ float s_f[NZ + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;

    int si = threadIdx.x;

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[k + 3][si] = f_in[globalIdx];
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        s_f[0][si]      = -s_f[6][si];
        s_f[1][si]      = -s_f[5][si];
        s_f[2][si]      = -s_f[4][si];
        s_f[NZ + 3][si] = 0.f;
    }

    __syncthreads();

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        float dsdz = -(c_fd[4] * s_f[k][si]     + 
                       c_fd[3] * s_f[k + 1][si] +
                       c_fd[2] * s_f[k + 2][si] +
                       c_fd[1] * s_f[k + 3][si] +
                       c_fd[0] * s_f[k + 4][si]);

        f_out[globalIdx] += dsdz;
    }
}

/*
__global__ void dvelocity_dy_forward(float *f_in, float *f_out) {
    __shared__ float s_f[NY + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int k = blockIdx.y;

    int si = threadIdx.x;

    for (int j = threadIdx.y; j < NY; j += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[j + 1][si] = f_in[globalIdx];
    }

    if (threadIdx.y == 0) {
        s_f[0][si]      = 0.f;
        s_f[NY + 1][si] = 0.f;
        s_f[NY + 2][si] = 0.f;
        s_f[NY + 3][si] = 0.f;
    }

    __syncthreads();

    for (int j = threadIdx.y; j < NY; j += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        float dvdy = c_fd[0] * s_f[j][si]     + 
                     c_fd[1] * s_f[j + 1][si] +
                     c_fd[2] * s_f[j + 2][si] +
                     c_fd[3] * s_f[j + 3][si] +
                     c_fd[4] * s_f[j + 4][si];

        f_out[globalIdx] += dvdy;
    }
}

__global__ void dvelocity_dy_backward(float *f_in, float *f_out) {
    __shared__ float s_f[NY + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int k = blockIdx.y;

    int si = threadIdx.x;

    for (int j = threadIdx.y; j < NY; j += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[j + 3][si] = f_in[globalIdx];
    }

    if (threadIdx.y == 0) {
        s_f[0][si]      = 0.f;
        s_f[1][si]      = 0.f;
        s_f[2][si]      = 0.f;
        s_f[NY + 3][si] = 0.f;
    }

    __syncthreads();

    for (int j = threadIdx.y; j < NY; j += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        float dvdy = -(c_fd[4] * s_f[j][si]     + 
                       c_fd[3] * s_f[j + 1][si] +
                       c_fd[2] * s_f[j + 2][si] +
                       c_fd[1] * s_f[j + 3][si] +
                       c_fd[0] * s_f[j + 4][si]);

        f_out[globalIdx] += dvdy;
    }
}
*/

__global__ void dvelocity_dy_forward_32x32(float *f_in, float *f_out) {
    __shared__ float s_f[36][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    s_f[sj + 1][si] = f_in[globalIdx];

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_f[0][si] = 0.f;
        } else {
            s_f[0][si] = f_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_f[33][si] = 0.f;
            s_f[34][si] = 0.f;
            s_f[35][si] = 0.f;
        } else {
            s_f[33][si] = f_in[globalIdx + STRIDEY1];
            s_f[34][si] = f_in[globalIdx + STRIDEY2];
            s_f[35][si] = f_in[globalIdx + STRIDEY3];
        }
    }

    __syncthreads();

    float dvdy = c_fd[0] * s_f[sj][si]     + 
                 c_fd[1] * s_f[sj + 1][si] +
                 c_fd[2] * s_f[sj + 2][si] +
                 c_fd[3] * s_f[sj + 3][si] +
                 c_fd[4] * s_f[sj + 4][si];

    f_out[globalIdx] += dvdy;
}

__global__ void dvelocity_dy_backward_32x32(float *f_in, float *f_out) {
     __shared__ float s_f[36][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    s_f[sj + 3][si] = f_in[globalIdx];

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_f[0][si] = 0.f;
            s_f[1][si] = 0.f;
            s_f[2][si] = 0.f;
        } else {
            s_f[0][si] = f_in[globalIdx - STRIDEY3];
            s_f[1][si] = f_in[globalIdx - STRIDEY2];
            s_f[2][si] = f_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_f[35][si] = 0.f;
        } else {
            s_f[35][si] = f_in[globalIdx + STRIDEY1];
        }
    }

    __syncthreads();

    float dvdy = -(c_fd[4] * s_f[sj][si]     + 
                   c_fd[3] * s_f[sj + 1][si] +
                   c_fd[2] * s_f[sj + 2][si] +
                   c_fd[1] * s_f[sj + 3][si] +
                   c_fd[0] * s_f[sj + 4][si]);

    f_out[globalIdx] += dvdy;
}

__global__ void dvelocity_dz_forward(float *f_in, float *f_out) {
    __shared__ float s_f[NZ + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;

    int si = threadIdx.x;

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[k + 1][si] = f_in[globalIdx];
    }

    if (threadIdx.y == 0) {
        s_f[NZ + 1][si] = 0.f;
        s_f[NZ + 2][si] = 0.f;
        s_f[NZ + 3][si] = 0.f;
    }

    __syncthreads();

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        if (k >= 1) {
            float dvdz = c_fd[0] * s_f[k][si]     + 
                         c_fd[1] * s_f[k + 1][si] +
                         c_fd[2] * s_f[k + 2][si] +
                         c_fd[3] * s_f[k + 3][si] +
                         c_fd[4] * s_f[k + 4][si];

            f_out[globalIdx] += dvdz;
        }
    }
}

__global__ void dvelocity_dz_backward(float *f_in, float *f_out) {
    __shared__ float s_f[NZ + 4][32];

    //gloabl memory index
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;

    int si = threadIdx.x;

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        s_f[k + 3][si] = f_in[globalIdx];
    }

    if (threadIdx.y == 0) {
        s_f[0][si]      = 0.f;
        s_f[1][si]      = 0.f;
        s_f[2][si]      = 0.f;
        s_f[NZ + 3][si] = 0.f;
    }

    __syncthreads();

    for (int k = threadIdx.y; k < NZ; k += 32) {
        int globalIdx = (k * NY + j) * NX + i;

        if (k >= 3) {
            float dvdz = -(c_fd[4] * s_f[k][si]     + 
                           c_fd[3] * s_f[k + 1][si] +
                           c_fd[2] * s_f[k + 2][si] +
                           c_fd[1] * s_f[k + 3][si] +
                           c_fd[0] * s_f[k + 4][si]);

            f_out[globalIdx] += dvdz;
        }
    }
}

__global__ void dstress_dxy_forward(float *fx_in, float *fy_in, float *f_out) {
    __shared__ float s_x[32][36];
    __shared__ float s_y[36][32];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_x[sj][0] = 0.f;
        } else {
            s_x[sj][0] = fx_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_x[sj][33] = 0.f;
            s_x[sj][34] = 0.f;
            s_x[sj][35] = 0.f;
        } else {
            s_x[sj][33] = fx_in[globalIdx + 1];
            s_x[sj][34] = fx_in[globalIdx + 2];
            s_x[sj][35] = fx_in[globalIdx + 3];
        }
    }

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_y[0][si] = 0.f;
        } else {
            s_y[0][si] = fy_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_y[33][si] = 0.f;
            s_y[34][si] = 0.f;
            s_y[35][si] = 0.f;
        } else {
            s_y[33][si] = fy_in[globalIdx + STRIDEY1];
            s_y[34][si] = fy_in[globalIdx + STRIDEY2];
            s_y[35][si] = fy_in[globalIdx + STRIDEY3];
        }
    }

    s_x[sj][si + 1] = fx_in[globalIdx];
    s_y[sj + 1][si] = fy_in[globalIdx];

    __syncthreads();

    float dfxdx = c_fd[0] * s_x[sj][si]     + 
                  c_fd[1] * s_x[sj][si + 1] +
                  c_fd[2] * s_x[sj][si + 2] +
                  c_fd[3] * s_x[sj][si + 3] +
                  c_fd[4] * s_x[sj][si + 4];

    float dfydy = c_fd[0] * s_y[sj][si]     + 
                  c_fd[1] * s_y[sj + 1][si] +
                  c_fd[2] * s_y[sj + 2][si] +
                  c_fd[3] * s_y[sj + 3][si] +
                  c_fd[4] * s_y[sj + 4][si];

    f_out[globalIdx] += dfxdx + dfydy;
}

__global__ void dstress_dxy_backward(float *fx_in, float *fy_in, float * f_out) {
    __shared__ float s_x[32][36];
    __shared__ float s_y[36][32];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_x[sj][0] = 0.f;
            s_x[sj][1] = 0.f;
            s_x[sj][2] = 0.f;
        } else {
            s_x[sj][0] = fx_in[globalIdx - 3];
            s_x[sj][1] = fx_in[globalIdx - 2];
            s_x[sj][2] = fx_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_x[sj][35] = 0.f;
        } else {
            s_x[sj][35] = fx_in[globalIdx + 1];
        }
    }

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_y[0][si] = 0.f;
            s_y[1][si] = 0.f;
            s_y[2][si] = 0.f;
        } else {
            s_y[0][si] = fy_in[globalIdx - STRIDEY3];
            s_y[1][si] = fy_in[globalIdx - STRIDEY2];
            s_y[2][si] = fy_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_y[35][si] = 0.f;
        } else {
            s_y[35][si] = fy_in[globalIdx + STRIDEY1];
        }
    }

    s_x[sj][si + 3] = fx_in[globalIdx];
    s_y[sj + 3][si] = fy_in[globalIdx];

    __syncthreads();

    float dfxdx = -(c_fd[4] * s_x[sj][si]     + 
                    c_fd[3] * s_x[sj][si + 1] +
                    c_fd[2] * s_x[sj][si + 2] +
                    c_fd[1] * s_x[sj][si + 3] +
                    c_fd[0] * s_x[sj][si + 4]);

    float dfydy = -(c_fd[4] * s_y[sj][si]     + 
                    c_fd[3] * s_y[sj + 1][si] +
                    c_fd[2] * s_y[sj + 2][si] +
                    c_fd[1] * s_y[sj + 3][si] +
                    c_fd[0] * s_y[sj + 4][si]);

    f_out[globalIdx] += dfxdx + dfydy;
}

__global__ void dvxy_dxy_forward(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *dszz_out) {
    __shared__ float s_x[36][36];
    __shared__ float s_y[36][36];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_x[sj + 1][0] = 0.f;

            s_y[sj + 1][0] = 0.f;
        } else {
            s_x[sj + 1][0] = vx_in[globalIdx - 1];

            s_y[sj + 1][0] = vy_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_x[sj + 1][33] = 0.f;
            s_x[sj + 1][34] = 0.f;
            s_x[sj + 1][35] = 0.f;

            s_y[sj + 1][33] = 0.f;
            s_y[sj + 1][34] = 0.f;
            s_y[sj + 1][35] = 0.f;
        } else {
            s_x[sj + 1][33] = vx_in[globalIdx + 1];
            s_x[sj + 1][34] = vx_in[globalIdx + 2];
            s_x[sj + 1][35] = vx_in[globalIdx + 3];

            s_y[sj + 1][33] = vy_in[globalIdx + 1];
            s_y[sj + 1][34] = vy_in[globalIdx + 2];
            s_y[sj + 1][35] = vy_in[globalIdx + 3];
        }
    }

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_x[0][si + 1] = 0.f;

            s_y[0][si + 1] = 0.f;
        } else {
            s_x[0][si + 1] = vx_in[globalIdx - STRIDEY1];

            s_y[0][si + 1] = vy_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_x[33][si + 1] = 0.f;
            s_x[34][si + 1] = 0.f;
            s_x[35][si + 1] = 0.f;

            s_y[33][si + 1] = 0.f;
            s_y[34][si + 1] = 0.f;
            s_y[35][si + 1] = 0.f;
        } else {
            s_x[33][si + 1] = vx_in[globalIdx + STRIDEY1];
            s_x[34][si + 1] = vx_in[globalIdx + STRIDEY2];
            s_x[35][si + 1] = vx_in[globalIdx + STRIDEY3];

            s_y[33][si + 1] = vy_in[globalIdx + STRIDEY1];
            s_y[34][si + 1] = vy_in[globalIdx + STRIDEY2];
            s_y[35][si + 1] = vy_in[globalIdx + STRIDEY3];
        }
    }

    s_x[sj + 1][si + 1] = vx_in[globalIdx];
    s_y[sj + 1][si + 1] = vy_in[globalIdx];

    __syncthreads();
    float lambda = lambda_in[globalIdx];
    float dvxdx  = c_fd[0] * s_x[sj + 1][si]     +
                   c_fd[1] * s_x[sj + 1][si + 1] +
                   c_fd[2] * s_x[sj + 1][si + 2] +
                   c_fd[3] * s_x[sj + 1][si + 3] +
                   c_fd[4] * s_x[sj + 1][si + 4];

    float dvydy  = c_fd[0] * s_y[sj][si + 1]     +
                   c_fd[1] * s_y[sj + 1][si + 1] +
                   c_fd[2] * s_y[sj + 2][si + 1] +
                   c_fd[3] * s_y[sj + 3][si + 1] +
                   c_fd[4] * s_y[sj + 4][si + 1];

    float dvxdy  = c_fd[0] * s_x[sj][si + 1]     +
                   c_fd[1] * s_x[sj + 1][si + 1] +
                   c_fd[2] * s_x[sj + 2][si + 1] +
                   c_fd[3] * s_x[sj + 3][si + 1] +
                   c_fd[4] * s_x[sj + 4][si + 1];

    float dvydx  = c_fd[0] * s_y[sj + 1][si]     +
                   c_fd[1] * s_y[sj + 1][si + 1] +
                   c_fd[2] * s_y[sj + 1][si + 2] +
                   c_fd[3] * s_y[sj + 1][si + 3] +
                   c_fd[4] * s_y[sj + 1][si + 4];

    dsxy_out[globalIdx] += dvxdy + dvydx;
    dsxx_out[globalIdx] += (lambda + 2.f) * dvxdx + lambda * dvydy;
    dsyy_out[globalIdx] += (lambda + 2.f) * dvydy + lambda * dvxdx;
    dszz_out[globalIdx] += lambda * (dvxdx + dvydy);
}

__global__ void dvxy_dxy_backward(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *dszz_out) {
    __shared__ float s_x[36][36];
    __shared__ float s_y[36][36];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    int k = blockIdx.z;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_x[sj + 3][0] = 0.f;
            s_x[sj + 3][1] = 0.f;
            s_x[sj + 3][2] = 0.f;

            s_y[sj + 3][0] = 0.f;
            s_y[sj + 3][1] = 0.f;
            s_y[sj + 3][2] = 0.f;
        } else {
            s_x[sj + 3][0] = vx_in[globalIdx - 3];
            s_x[sj + 3][1] = vx_in[globalIdx - 2];
            s_x[sj + 3][2] = vx_in[globalIdx - 1];

            s_y[sj + 3][0] = vy_in[globalIdx - 3];
            s_y[sj + 3][1] = vy_in[globalIdx - 2];
            s_y[sj + 3][2] = vy_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_x[sj + 3][35] = 0.f;

            s_y[sj + 3][35] = 0.f;
        } else {
            s_x[sj + 3][35] = vx_in[globalIdx + 1];

            s_y[sj + 3][35] = vy_in[globalIdx + 1];
        }
    }

    if (sj == 0) {
        if (blockIdx.y == 0) {
            s_x[0][si + 3] = 0.f;
            s_x[1][si + 3] = 0.f;
            s_x[2][si + 3] = 0.f;

            s_y[0][si + 3] = 0.f;
            s_y[1][si + 3] = 0.f;
            s_y[2][si + 3] = 0.f;
        } else {
            s_x[0][si + 3] = vx_in[globalIdx - STRIDEY3];
            s_x[1][si + 3] = vx_in[globalIdx - STRIDEY2];
            s_x[2][si + 3] = vx_in[globalIdx - STRIDEY1];

            s_y[0][si + 3] = vy_in[globalIdx - STRIDEY3];
            s_y[1][si + 3] = vy_in[globalIdx - STRIDEY2];
            s_y[2][si + 3] = vy_in[globalIdx - STRIDEY1];
        }
    }

    if (sj == 31) {
        if (blockIdx.y == gridDim.y - 1) {
            s_x[35][si + 3] = 0.f;

            s_y[35][si + 3] = 0.f;
        } else {
            s_x[35][si + 3] = vx_in[globalIdx + STRIDEY1];

            s_y[35][si + 3] = vy_in[globalIdx + STRIDEY1];
        }
    }

    s_x[sj + 3][si + 3] = vx_in[globalIdx];
    s_y[sj + 3][si + 3] = vy_in[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdx  = -(c_fd[4] * s_x[sj + 3][si]     +
                     c_fd[3] * s_x[sj + 3][si + 1] +
                     c_fd[2] * s_x[sj + 3][si + 2] +
                     c_fd[1] * s_x[sj + 3][si + 3] +
                     c_fd[0] * s_x[sj + 3][si + 4]);

    float dvydy  = -(c_fd[4] * s_y[sj][si + 3]     +
                     c_fd[3] * s_y[sj + 1][si + 3] +
                     c_fd[2] * s_y[sj + 2][si + 3] +
                     c_fd[1] * s_y[sj + 3][si + 3] +
                     c_fd[0] * s_y[sj + 4][si + 3]);

    float dvxdy  = -(c_fd[4] * s_x[sj][si + 3]     +
                     c_fd[3] * s_x[sj + 1][si + 3] +
                     c_fd[2] * s_x[sj + 2][si + 3] +
                     c_fd[1] * s_x[sj + 3][si + 3] +
                     c_fd[0] * s_x[sj + 4][si + 3]);

    float dvydx  = -(c_fd[4] * s_y[sj + 3][si]     +
                     c_fd[3] * s_y[sj + 3][si + 1] +
                     c_fd[2] * s_y[sj + 3][si + 2] +
                     c_fd[1] * s_y[sj + 3][si + 3] +
                     c_fd[0] * s_y[sj + 3][si + 4]);

    dsxy_out[globalIdx] += dvxdy + dvydx;
    dsxx_out[globalIdx] += (lambda + 2.f) * dvxdx + lambda * dvydy;
    dsyy_out[globalIdx] += (lambda + 2.f) * dvydy + lambda * dvxdx;
    dszz_out[globalIdx] += lambda * (dvxdx + dvydy);
}

__global__ void dvxz_dxz_forward(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *dszz_out) {
    __shared__ float s_x[36][32];
    __shared__ float s_z[36][36];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z * 32 + threadIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_z[sj + 1][0] = 0.f;
        } else {
            s_z[sj + 1][0] = vz_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_z[sj + 1][33] = 0.f;
            s_z[sj + 1][34] = 0.f;
            s_z[sj + 1][35] = 0.f;
        } else {
            s_z[sj + 1][33] = vz_in[globalIdx + 1];
            s_z[sj + 1][34] = vz_in[globalIdx + 2];
            s_z[sj + 1][35] = vz_in[globalIdx + 3];
        }
    }

    if (sj == 0) {
        if (blockIdx.z == 0) {
            s_x[0][si] = 0.f;

            s_z[0][si + 1] = 0.f;
        } else {
            s_x[0][si] = vx_in[globalIdx - STRIDEZ1];

            s_z[0][si + 1] = vz_in[globalIdx - STRIDEZ1];
        }
    }

    if (sj == 31) {
        if (blockIdx.z == gridDim.z - 1) {
            s_x[33][si] = 0.f;
            s_x[34][si] = 0.f;
            s_x[35][si] = 0.f;

            s_z[33][si + 1] = 0.f;
            s_z[34][si + 1] = 0.f;
            s_z[35][si + 1] = 0.f;
        } else {
            s_x[33][si] = vx_in[globalIdx + STRIDEZ1];
            s_x[34][si] = vx_in[globalIdx + STRIDEZ2];
            s_x[35][si] = vx_in[globalIdx + STRIDEZ3];

            s_z[33][si + 1] = vz_in[globalIdx + STRIDEZ1];
            s_z[34][si + 1] = vz_in[globalIdx + STRIDEZ2];
            s_z[35][si + 1] = vz_in[globalIdx + STRIDEZ3];
        }
    }

    s_x[sj + 1][si]     = vx_in[globalIdx];
    s_z[sj + 1][si + 1] = vz_in[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];
    float dvzdz  = c_fd[0] * s_z[sj][si + 1]     +
                   c_fd[1] * s_z[sj + 1][si + 1] +
                   c_fd[2] * s_z[sj + 2][si + 1] +
                   c_fd[3] * s_z[sj + 3][si + 1] +
                   c_fd[4] * s_z[sj + 4][si + 1];

    float dvxdz = c_fd[0] * s_x[sj][si]     +
                  c_fd[1] * s_x[sj + 1][si] +
                  c_fd[2] * s_x[sj + 2][si] +
                  c_fd[3] * s_x[sj + 3][si] +
                  c_fd[4] * s_x[sj + 4][si];

    float dvzdx = c_fd[0] * s_z[sj + 1][si]     +
                  c_fd[1] * s_z[sj + 1][si + 1] +
                  c_fd[2] * s_z[sj + 1][si + 2] +
                  c_fd[3] * s_z[sj + 1][si + 3] +
                  c_fd[4] * s_z[sj + 1][si + 4];

    if (k >= 1) {
        dsxz_out[globalIdx] += dvxdz + dvzdx;
        dsxx_out[globalIdx] += lambda * dvzdz;
        dsyy_out[globalIdx] += lambda * dvzdz;
        dszz_out[globalIdx] += (lambda + 2.f) * dvzdz;
    } else {
        dsxz_out[globalIdx] += dvzdx;
    }
}

__global__ void dvxz_dxz_backward(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *dszz_out) {
    __shared__ float s_x[36][32];
    __shared__ float s_z[36][36];

    //gloabl i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z * 32 + threadIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    int si = threadIdx.x;
    int sj = threadIdx.y;

    if (si == 0) {
        if (blockIdx.x == 0) {
            s_z[sj + 3][0] = 0.f;
            s_z[sj + 3][1] = 0.f;
            s_z[sj + 3][2] = 0.f;
        } else {
            s_z[sj + 3][0] = vz_in[globalIdx - 3];
            s_z[sj + 3][1] = vz_in[globalIdx - 2];
            s_z[sj + 3][2] = vz_in[globalIdx - 1];
        }
    }

    if (si == 31) {
        if (blockIdx.x == gridDim.x - 1) {
            s_z[sj + 3][35] = 0.f;
        } else {
            s_z[sj + 3][35] = vz_in[globalIdx + 1];
        }
    }

    if (sj == 0) {
        if (blockIdx.z == 0) {
            s_x[0][si] = 0.f;
            s_x[1][si] = 0.f;
            s_x[2][si] = 0.f;

            s_z[0][si + 3] = 0.f;
            s_z[1][si + 3] = 0.f;
            s_z[2][si + 3] = 0.f;
        } else {
            s_x[0][si] = vx_in[globalIdx - STRIDEZ3];
            s_x[1][si] = vx_in[globalIdx - STRIDEZ2];
            s_x[2][si] = vx_in[globalIdx - STRIDEZ1];

            s_z[0][si + 3] = vz_in[globalIdx - STRIDEZ3];
            s_z[1][si + 3] = vz_in[globalIdx - STRIDEZ2];
            s_z[2][si + 3] = vz_in[globalIdx - STRIDEZ1];
        }
    }

    if (sj == 31) {
        if (blockIdx.z == gridDim.z - 1) {
            s_x[35][si] = 0.f;

            s_z[35][si + 3] = 0.f;
        } else {
            s_x[35][si] = vx_in[globalIdx + STRIDEZ1];

            s_z[35][si + 3] = vz_in[globalIdx + STRIDEZ1];
        }
    }

    s_x[sj + 3][si]     = vx_in[globalIdx];
    s_z[sj + 3][si + 3] = vz_in[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvzdz = -(c_fd[4] * s_z[sj][si + 3]     +
                    c_fd[3] * s_z[sj + 1][si + 3] +
                    c_fd[2] * s_z[sj + 2][si + 3] +
                    c_fd[1] * s_z[sj + 3][si + 3] +
                    c_fd[0] * s_z[sj + 4][si + 3]);

    float dvxdz = -(c_fd[4] * s_x[sj][si]     +
                    c_fd[3] * s_x[sj + 1][si] +
                    c_fd[2] * s_x[sj + 2][si] +
                    c_fd[1] * s_x[sj + 3][si] +
                    c_fd[0] * s_x[sj + 4][si]);

    float dvzdx = -(c_fd[4] * s_z[sj + 3][si]     +
                    c_fd[3] * s_z[sj + 3][si + 1] +
                    c_fd[2] * s_z[sj + 3][si + 2] +
                    c_fd[1] * s_z[sj + 3][si + 3] +
                    c_fd[0] * s_z[sj + 3][si + 4]);

    if (k >= 3) {
        dsxz_out[globalIdx] += dvxdz + dvzdx;
        dsxx_out[globalIdx] += lambda * dvzdz;
        dsyy_out[globalIdx] += lambda * dvzdz;
        dszz_out[globalIdx] += (lambda + 2.f) * dvzdz;
    } else {
        dsxz_out[globalIdx] += dvzdx;
    }
}

__global__ void pml_frontv_forward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[32][PML_NUM + 4];
    __shared__ float s_vy[32][PML_NUM + 4];
    __shared__ float s_vz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_vx[sj][0] = 0.f;

        s_vy[sj][0] = 0.f;

        s_vz[sj][0] = 0.f;
    }

    if (threadIdx.x < PML_NUM + 3) {
        s_vx[sj][i + 1] = vx[globalIdx];
        s_vy[sj][i + 1] = vy[globalIdx];
        s_vz[sj][i + 1] = vz[globalIdx];
    }

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    if (threadIdx.x < PML_NUM) {
        float dvxdx = c_fd[0] * s_vx[sj][i]     +
                      c_fd[1] * s_vx[sj][i + 1] +
                      c_fd[2] * s_vx[sj][i + 2] +
                      c_fd[3] * s_vx[sj][i + 3] +
                      c_fd[4] * s_vx[sj][i + 4];

        float dvydx = c_fd[0] * s_vy[sj][i]     +
                      c_fd[1] * s_vy[sj][i + 1] +
                      c_fd[2] * s_vy[sj][i + 2] +
                      c_fd[3] * s_vy[sj][i + 3] +
                      c_fd[4] * s_vy[sj][i + 4];

        float dvzdx = c_fd[0] * s_vz[sj][i]     +
                      c_fd[1] * s_vz[sj][i + 1] +
                      c_fd[2] * s_vz[sj][i + 2] +
                      c_fd[3] * s_vz[sj][i + 3] +
                      c_fd[4] * s_vz[sj][i + 4];

        i = PML_NUM - i;

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dsxx[globalIdx] = dsxx[globalIdx] + (lambda + 2.f) * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dszz[globalIdx] = dszz[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dsxy[globalIdx] = dsxy[globalIdx] + dvydx * (beta - 1.f) - vyx[pmlIdx] * beta * c_dt;
        dsxz[globalIdx] = dsxz[globalIdx] + dvzdx * (beta - 1.f) - vzx[pmlIdx] * beta * c_dt;

        dvxx[pmlIdx] += dbeta * dvxdx - alphadbeta * vxx[pmlIdx];
        dvyx[pmlIdx] += dbeta * dvydx - alphadbeta * vyx[pmlIdx];
        dvzx[pmlIdx] += dbeta * dvzdx - alphadbeta * vzx[pmlIdx];

        vxx[pmlIdx]  += dvxx[pmlIdx] * scaleB;
        dvxx[pmlIdx] *= scaleA;

        vyx[pmlIdx]  += dvyx[pmlIdx] * scaleB;
        dvyx[pmlIdx] *= scaleA;

        vzx[pmlIdx]  += dvzx[pmlIdx] * scaleB;
        dvzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_frontv_backward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[32][PML_NUM + 4];
    __shared__ float s_vy[32][PML_NUM + 4];
    __shared__ float s_vz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_vx[sj][0] = 0.f;
        s_vx[sj][1] = 0.f;
        s_vx[sj][2] = 0.f;

        s_vy[sj][0] = 0.f;
        s_vy[sj][1] = 0.f;
        s_vy[sj][2] = 0.f;

        s_vz[sj][0] = 0.f;
        s_vz[sj][1] = 0.f;
        s_vz[sj][2] = 0.f;
    }

    if (threadIdx.x < PML_NUM + 1) {
        s_vx[sj][i + 3] = vx[globalIdx];
        s_vy[sj][i + 3] = vy[globalIdx];
        s_vz[sj][i + 3] = vz[globalIdx];
    }

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    if (threadIdx.x < PML_NUM) {
        float dvxdx = -(c_fd[4] * s_vx[sj][i]     +
                        c_fd[3] * s_vx[sj][i + 1] +
                        c_fd[2] * s_vx[sj][i + 2] +
                        c_fd[1] * s_vx[sj][i + 3] +
                        c_fd[0] * s_vx[sj][i + 4]);

        float dvydx = -(c_fd[4] * s_vy[sj][i]     +
                        c_fd[3] * s_vy[sj][i + 1] +
                        c_fd[2] * s_vy[sj][i + 2] +
                        c_fd[1] * s_vy[sj][i + 3] +
                        c_fd[0] * s_vy[sj][i + 4]);

        float dvzdx = -(c_fd[4] * s_vz[sj][i]     +
                        c_fd[3] * s_vz[sj][i + 1] +
                        c_fd[2] * s_vz[sj][i + 2] +
                        c_fd[1] * s_vz[sj][i + 3] +
                        c_fd[0] * s_vz[sj][i + 4]);

        i = PML_NUM - i;

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dsxx[globalIdx] = dsxx[globalIdx] + (lambda + 2.f) * (dvxdx * (beta - 1.f) -vxx[pmlIdx] * beta * c_dt);
        dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dszz[globalIdx] = dszz[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dsxy[globalIdx] = dsxy[globalIdx] + dvydx * (beta - 1.f) - vyx[pmlIdx] * beta * c_dt;
        dsxz[globalIdx] = dsxz[globalIdx] + dvzdx * (beta - 1.f) - vzx[pmlIdx] * beta * c_dt;

        dvxx[pmlIdx] += dbeta * dvxdx - alphadbeta * vxx[pmlIdx];
        dvyx[pmlIdx] += dbeta * dvydx - alphadbeta * vyx[pmlIdx];
        dvzx[pmlIdx] += dbeta * dvzdx - alphadbeta * vzx[pmlIdx];

        vxx[pmlIdx]  += dvxx[pmlIdx] * scaleB;
        dvxx[pmlIdx] *= scaleA;

        vyx[pmlIdx]  += dvyx[pmlIdx] * scaleB;
        dvyx[pmlIdx] *= scaleA;

        vzx[pmlIdx]  += dvzx[pmlIdx] * scaleB;
        dvzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_fronts_forward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxx[32][PML_NUM + 4];
    __shared__ float s_sxy[32][PML_NUM + 4];
    __shared__ float s_sxz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_sxx[sj][0] = 0.f;

        s_sxy[sj][0] = 0.f;

        s_sxz[sj][0] = 0.f;
    }

    if (threadIdx.x < PML_NUM + 3) {
        s_sxx[sj][i + 1] = sxx[globalIdx];
        s_sxy[sj][i + 1] = sxy[globalIdx];
        s_sxz[sj][i + 1] = sxz[globalIdx];
    }

    __syncthreads();

    if (threadIdx.x < PML_NUM) {
        float dsxxdx = c_fd[0] * s_sxx[sj][i]     +
                       c_fd[1] * s_sxx[sj][i + 1] +
                       c_fd[2] * s_sxx[sj][i + 2] +
                       c_fd[3] * s_sxx[sj][i + 3] +
                       c_fd[4] * s_sxx[sj][i + 4];

        float dsxydx = c_fd[0] * s_sxy[sj][i]     +
                       c_fd[1] * s_sxy[sj][i + 1] +
                       c_fd[2] * s_sxy[sj][i + 2] +
                       c_fd[3] * s_sxy[sj][i + 3] +
                       c_fd[4] * s_sxy[sj][i + 4];

        float dsxzdx = c_fd[0] * s_sxz[sj][i]     +
                       c_fd[1] * s_sxz[sj][i + 1] +
                       c_fd[2] * s_sxz[sj][i + 2] +
                       c_fd[3] * s_sxz[sj][i + 3] +
                       c_fd[4] * s_sxz[sj][i + 4];

        i = PML_NUM - i;

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxxdx - sxxx[pmlIdx] * beta * c_dt;
        dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsxydx - sxyx[pmlIdx] * beta * c_dt;
        dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsxzdx - sxzx[pmlIdx] * beta * c_dt;

        dsxxx[pmlIdx] += dbeta * dsxxdx - alphadbeta * sxxx[pmlIdx];
        dsxyx[pmlIdx] += dbeta * dsxydx - alphadbeta * sxyx[pmlIdx];
        dsxzx[pmlIdx] += dbeta * dsxzdx - alphadbeta * sxzx[pmlIdx];

        sxxx[pmlIdx]  += dsxxx[pmlIdx] * scaleB;
        dsxxx[pmlIdx] *= scaleA;

        sxyx[pmlIdx]  += dsxyx[pmlIdx] * scaleB;
        dsxyx[pmlIdx] *= scaleA;

        sxzx[pmlIdx]  += dsxzx[pmlIdx] * scaleB;
        dsxzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_fronts_backward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxx[32][PML_NUM + 4];
    __shared__ float s_sxy[32][PML_NUM + 4];
    __shared__ float s_sxz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_sxx[sj][0] = 0.f;
        s_sxx[sj][1] = 0.f;
        s_sxx[sj][2] = 0.f;

        s_sxy[sj][0] = 0.f;
        s_sxy[sj][1] = 0.f;
        s_sxy[sj][2] = 0.f;

        s_sxz[sj][0] = 0.f;
        s_sxz[sj][1] = 0.f;
        s_sxz[sj][2] = 0.f;
    }

    if (threadIdx.x < PML_NUM + 1) {
        s_sxx[sj][i + 3] = sxx[globalIdx];
        s_sxy[sj][i + 3] = sxy[globalIdx];
        s_sxz[sj][i + 3] = sxz[globalIdx];
    }

    __syncthreads();

    if (threadIdx.x < PML_NUM) {
        float dsxxdx = -(c_fd[4] * s_sxx[sj][i]     +
                         c_fd[3] * s_sxx[sj][i + 1] +
                         c_fd[2] * s_sxx[sj][i + 2] +
                         c_fd[1] * s_sxx[sj][i + 3] +
                         c_fd[0] * s_sxx[sj][i + 4]);

        float dsxydx = -(c_fd[4] * s_sxy[sj][i]     +
                         c_fd[3] * s_sxy[sj][i + 1] +
                         c_fd[2] * s_sxy[sj][i + 2] +
                         c_fd[1] * s_sxy[sj][i + 3] +
                         c_fd[0] * s_sxy[sj][i + 4]);

        float dsxzdx = -(c_fd[4] * s_sxz[sj][i]     +
                         c_fd[3] * s_sxz[sj][i + 1] +
                         c_fd[2] * s_sxz[sj][i + 2] +
                         c_fd[1] * s_sxz[sj][i + 3] +
                         c_fd[0] * s_sxz[sj][i + 4]);

        i = PML_NUM - i;

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxxdx - sxxx[pmlIdx] * beta * c_dt;
        dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsxydx - sxyx[pmlIdx] * beta * c_dt;
        dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsxzdx - sxzx[pmlIdx] * beta * c_dt;

        dsxxx[pmlIdx] += dbeta * dsxxdx - alphadbeta * sxxx[pmlIdx];
        dsxyx[pmlIdx] += dbeta * dsxydx - alphadbeta * sxyx[pmlIdx];
        dsxzx[pmlIdx] += dbeta * dsxzdx - alphadbeta * sxzx[pmlIdx];

        sxxx[pmlIdx]  += dsxxx[pmlIdx] * scaleB;
        dsxxx[pmlIdx] *= scaleA;

        sxyx[pmlIdx]  += dsxyx[pmlIdx] * scaleB;
        dsxyx[pmlIdx] *= scaleA;

        sxzx[pmlIdx]  += dsxzx[pmlIdx] * scaleB;
        dsxzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_backv_forward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[32][PML_NUM + 4];
    __shared__ float s_vy[32][PML_NUM + 4];
    __shared__ float s_vz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i + NX_PML;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_vx[sj][0]           = vx[globalIdx - 1];
        s_vx[sj][PML_NUM + 1] = 0.f;
        s_vx[sj][PML_NUM + 2] = 0.f;
        s_vx[sj][PML_NUM + 3] = 0.f;

        s_vy[sj][0]           = vy[globalIdx - 1];
        s_vy[sj][PML_NUM + 1] = 0.f;
        s_vy[sj][PML_NUM + 2] = 0.f;
        s_vy[sj][PML_NUM + 3] = 0.f;

        s_vz[sj][0]           = vz[globalIdx - 1];
        s_vz[sj][PML_NUM + 1] = 0.f;
        s_vz[sj][PML_NUM + 2] = 0.f;
        s_vz[sj][PML_NUM + 3] = 0.f;
    }

    if (threadIdx.x < PML_NUM) {
        s_vx[sj][i + 1] = vx[globalIdx];
        s_vy[sj][i + 1] = vy[globalIdx];
        s_vz[sj][i + 1] = vz[globalIdx];
    }

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    if (threadIdx.x < PML_NUM) {
        float dvxdx = c_fd[0] * s_vx[sj][i]     +
                      c_fd[1] * s_vx[sj][i + 1] +
                      c_fd[2] * s_vx[sj][i + 2] +
                      c_fd[3] * s_vx[sj][i + 3] +
                      c_fd[4] * s_vx[sj][i + 4];

        float dvydx = c_fd[0] * s_vy[sj][i]     +
                      c_fd[1] * s_vy[sj][i + 1] +
                      c_fd[2] * s_vy[sj][i + 2] +
                      c_fd[3] * s_vy[sj][i + 3] +
                      c_fd[4] * s_vy[sj][i + 4];

        float dvzdx = c_fd[0] * s_vz[sj][i]     +
                      c_fd[1] * s_vz[sj][i + 1] +
                      c_fd[2] * s_vz[sj][i + 2] +
                      c_fd[3] * s_vz[sj][i + 3] +
                      c_fd[4] * s_vz[sj][i + 4];

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dsxx[globalIdx] = dsxx[globalIdx] + (lambda + 2.f) * (dvxdx * (beta - 1.f) -vxx[pmlIdx] * beta * c_dt);
        dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dszz[globalIdx] = dszz[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dsxy[globalIdx] = dsxy[globalIdx] + dvydx * (beta - 1.f) - vyx[pmlIdx] * beta * c_dt;
        dsxz[globalIdx] = dsxz[globalIdx] + dvzdx * (beta - 1.f) - vzx[pmlIdx] * beta * c_dt;

        dvxx[pmlIdx] += dbeta * dvxdx - alphadbeta * vxx[pmlIdx];
        dvyx[pmlIdx] += dbeta * dvydx - alphadbeta * vyx[pmlIdx];
        dvzx[pmlIdx] += dbeta * dvzdx - alphadbeta * vzx[pmlIdx];

        vxx[pmlIdx]  += dvxx[pmlIdx] * scaleB;
        dvxx[pmlIdx] *= scaleA;

        vyx[pmlIdx]  += dvyx[pmlIdx] * scaleB;
        dvyx[pmlIdx] *= scaleA;

        vzx[pmlIdx]  += dvzx[pmlIdx] * scaleB;
        dvzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_backv_backward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[32][PML_NUM + 4];
    __shared__ float s_vy[32][PML_NUM + 4];
    __shared__ float s_vz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i + NX_PML;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_vx[sj][0]           = vx[globalIdx - 3];
        s_vx[sj][1]           = vx[globalIdx - 2];
        s_vx[sj][2]           = vx[globalIdx - 1];
        s_vx[sj][PML_NUM + 3] = 0.f;

        s_vy[sj][0]           = vy[globalIdx - 3];
        s_vy[sj][1]           = vy[globalIdx - 2];
        s_vy[sj][2]           = vy[globalIdx - 1];
        s_vy[sj][PML_NUM + 3] = 0.f;

        s_vz[sj][0]           = vz[globalIdx - 3];
        s_vz[sj][1]           = vz[globalIdx - 2];
        s_vz[sj][2]           = vz[globalIdx - 1];
        s_vz[sj][PML_NUM + 3] = 0.f;
    }

    if (threadIdx.x < PML_NUM) {
        s_vx[sj][i + 3] = vx[globalIdx];
        s_vy[sj][i + 3] = vy[globalIdx];
        s_vz[sj][i + 3] = vz[globalIdx];
    }

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    if (threadIdx.x < PML_NUM) {
        float dvxdx = -(c_fd[4] * s_vx[sj][i]     +
                        c_fd[3] * s_vx[sj][i + 1] +
                        c_fd[2] * s_vx[sj][i + 2] +
                        c_fd[1] * s_vx[sj][i + 3] +
                        c_fd[0] * s_vx[sj][i + 4]);

        float dvydx = -(c_fd[4] * s_vy[sj][i]     +
                        c_fd[3] * s_vy[sj][i + 1] +
                        c_fd[2] * s_vy[sj][i + 2] +
                        c_fd[1] * s_vy[sj][i + 3] +
                        c_fd[0] * s_vy[sj][i + 4]);

        float dvzdx = -(c_fd[4] * s_vz[sj][i]     +
                        c_fd[3] * s_vz[sj][i + 1] +
                        c_fd[2] * s_vz[sj][i + 2] +
                        c_fd[1] * s_vz[sj][i + 3] +
                        c_fd[0] * s_vz[sj][i + 4]);

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dsxx[globalIdx] = dsxx[globalIdx] + (lambda + 2.f) * (dvxdx * (beta - 1.f) -vxx[pmlIdx] * beta * c_dt);
        dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dszz[globalIdx] = dszz[globalIdx] + lambda * (dvxdx * (beta - 1.f) - vxx[pmlIdx] * beta * c_dt);
        dsxy[globalIdx] = dsxy[globalIdx] + dvydx * (beta - 1.f) - vyx[pmlIdx] * beta * c_dt;
        dsxz[globalIdx] = dsxz[globalIdx] + dvzdx * (beta - 1.f) - vzx[pmlIdx] * beta * c_dt;

        dvxx[pmlIdx] += dbeta * dvxdx - alphadbeta * vxx[pmlIdx];
        dvyx[pmlIdx] += dbeta * dvydx - alphadbeta * vyx[pmlIdx];
        dvzx[pmlIdx] += dbeta * dvzdx - alphadbeta * vzx[pmlIdx];

        vxx[pmlIdx]  += dvxx[pmlIdx] * scaleB;
        dvxx[pmlIdx] *= scaleA;

        vyx[pmlIdx]  += dvyx[pmlIdx] * scaleB;
        dvyx[pmlIdx] *= scaleA;

        vzx[pmlIdx]  += dvzx[pmlIdx] * scaleB;
        dvzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_backs_forward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxx[32][PML_NUM + 4];
    __shared__ float s_sxy[32][PML_NUM + 4];
    __shared__ float s_sxz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i + NX_PML;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_sxx[sj][0]           = sxx[globalIdx - 1];
        s_sxx[sj][PML_NUM + 1] = 0.f;
        s_sxx[sj][PML_NUM + 2] = 0.f;
        s_sxx[sj][PML_NUM + 3] = 0.f;

        s_sxy[sj][0]           = sxy[globalIdx - 1];
        s_sxy[sj][PML_NUM + 1] = 0.f;
        s_sxy[sj][PML_NUM + 2] = 0.f;
        s_sxy[sj][PML_NUM + 3] = 0.f;

        s_sxz[sj][0]           = sxz[globalIdx - 1];
        s_sxz[sj][PML_NUM + 1] = 0.f;
        s_sxz[sj][PML_NUM + 2] = 0.f;
        s_sxz[sj][PML_NUM + 3] = 0.f;
    }

    if (threadIdx.x < PML_NUM) {
        s_sxx[sj][i + 1] = sxx[globalIdx];
        s_sxy[sj][i + 1] = sxy[globalIdx];
        s_sxz[sj][i + 1] = sxz[globalIdx];
    }

    __syncthreads();

    if (threadIdx.x < PML_NUM) {

        float dsxxdx = c_fd[0] * s_sxx[sj][i]     +
                       c_fd[1] * s_sxx[sj][i + 1] +
                       c_fd[2] * s_sxx[sj][i + 2] +
                       c_fd[3] * s_sxx[sj][i + 3] +
                       c_fd[4] * s_sxx[sj][i + 4];

        float dsxydx = c_fd[0] * s_sxy[sj][i]     +
                       c_fd[1] * s_sxy[sj][i + 1] +
                       c_fd[2] * s_sxy[sj][i + 2] +
                       c_fd[3] * s_sxy[sj][i + 3] +
                       c_fd[4] * s_sxy[sj][i + 4];

        float dsxzdx = c_fd[0] * s_sxz[sj][i]     +
                       c_fd[1] * s_sxz[sj][i + 1] +
                       c_fd[2] * s_sxz[sj][i + 2] +
                       c_fd[3] * s_sxz[sj][i + 3] +
                       c_fd[4] * s_sxz[sj][i + 4];

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxxdx - sxxx[pmlIdx] * beta * c_dt;
        dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsxydx - sxyx[pmlIdx] * beta * c_dt;
        dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsxzdx - sxzx[pmlIdx] * beta * c_dt;

        dsxxx[pmlIdx] += dbeta * dsxxdx - alphadbeta * sxxx[pmlIdx];
        dsxyx[pmlIdx] += dbeta * dsxydx - alphadbeta * sxyx[pmlIdx];
        dsxzx[pmlIdx] += dbeta * dsxzdx - alphadbeta * sxzx[pmlIdx];

        sxxx[pmlIdx]  += dsxxx[pmlIdx] * scaleB;
        dsxxx[pmlIdx] *= scaleA;

        sxyx[pmlIdx]  += dsxyx[pmlIdx] * scaleB;
        dsxyx[pmlIdx] *= scaleA;

        sxzx[pmlIdx]  += dsxzx[pmlIdx] * scaleB;
        dsxzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_backs_backward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxx[32][PML_NUM + 4];
    __shared__ float s_sxy[32][PML_NUM + 4];
    __shared__ float s_sxz[32][PML_NUM + 4];
    
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x * 32 + threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i + NX_PML;
    int pmlIdx    = (k * NY + j) * PML_NUM + i;

    //shared index
    int sj = threadIdx.y;

    if (threadIdx.x == 0) {
        s_sxx[sj][0]           = sxx[globalIdx - 3];
        s_sxx[sj][1]           = sxx[globalIdx - 2];
        s_sxx[sj][2]           = sxx[globalIdx - 1];
        s_sxx[sj][PML_NUM + 3] = 0.f;

        s_sxy[sj][0]           = sxy[globalIdx - 3];
        s_sxy[sj][1]           = sxy[globalIdx - 2];
        s_sxy[sj][2]           = sxy[globalIdx - 1];
        s_sxy[sj][PML_NUM + 3] = 0.f;

        s_sxz[sj][0]           = sxz[globalIdx - 3];
        s_sxz[sj][1]           = sxz[globalIdx - 2];
        s_sxz[sj][2]           = sxz[globalIdx - 1];
        s_sxz[sj][PML_NUM + 3] = 0.f;
    }

    if (threadIdx.x < PML_NUM) {
        s_sxx[sj][i + 3] = sxx[globalIdx];
        s_sxy[sj][i + 3] = sxy[globalIdx];
        s_sxz[sj][i + 3] = sxz[globalIdx];
    }

    __syncthreads();

    if (threadIdx.x < PML_NUM) {

        float dsxxdx = -(c_fd[4] * s_sxx[sj][i]     +
                         c_fd[3] * s_sxx[sj][i + 1] +
                         c_fd[2] * s_sxx[sj][i + 2] +
                         c_fd[1] * s_sxx[sj][i + 3] +
                         c_fd[0] * s_sxx[sj][i + 4]);

        float dsxydx = -(c_fd[4] * s_sxy[sj][i]     +
                         c_fd[3] * s_sxy[sj][i + 1] +
                         c_fd[2] * s_sxy[sj][i + 2] +
                         c_fd[1] * s_sxy[sj][i + 3] +
                         c_fd[0] * s_sxy[sj][i + 4]);

        float dsxzdx = -(c_fd[4] * s_sxz[sj][i]     +
                         c_fd[3] * s_sxz[sj][i + 1] +
                         c_fd[2] * s_sxz[sj][i + 2] +
                         c_fd[1] * s_sxz[sj][i + 3] +
                         c_fd[0] * s_sxz[sj][i + 4]);

        float beta       = 1.f / (1.f + c_beta0 * i * i);
        float d          = c_d0 * i * i;
        float alpha      = c_alpha0 - c_alpha01 * i;
        float dbeta      = d * beta;
        float alphadbeta = (alpha + dbeta) * c_dt;

        dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxxdx - sxxx[pmlIdx] * beta * c_dt;
        dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsxydx - sxyx[pmlIdx] * beta * c_dt;
        dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsxzdx - sxzx[pmlIdx] * beta * c_dt;

        dsxxx[pmlIdx] += dbeta * dsxxdx - alphadbeta * sxxx[pmlIdx];
        dsxyx[pmlIdx] += dbeta * dsxydx - alphadbeta * sxyx[pmlIdx];
        dsxzx[pmlIdx] += dbeta * dsxzdx - alphadbeta * sxzx[pmlIdx];

        sxxx[pmlIdx]  += dsxxx[pmlIdx] * scaleB;
        dsxxx[pmlIdx] *= scaleA;

        sxyx[pmlIdx]  += dsxyx[pmlIdx] * scaleB;
        dsxyx[pmlIdx] *= scaleA;

        sxzx[pmlIdx]  += dsxzx[pmlIdx] * scaleB;
        dsxzx[pmlIdx] *= scaleA;
    }
}

__global__ void pml_leftv_forward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = 0.f;
        s_vx[PML_NUM + 1][si] = vx[globalIdx + STRIDEY1];
        s_vx[PML_NUM + 2][si] = vx[globalIdx + STRIDEY2];
        s_vx[PML_NUM + 3][si] = vx[globalIdx + STRIDEY3];

        s_vy[0][si]           = 0.f;
        s_vy[PML_NUM + 1][si] = vy[globalIdx + STRIDEY1];
        s_vy[PML_NUM + 2][si] = vy[globalIdx + STRIDEY2];
        s_vy[PML_NUM + 3][si] = vy[globalIdx + STRIDEY3];

        s_vz[0][si]           = 0.f;
        s_vz[PML_NUM + 1][si] = vz[globalIdx + STRIDEY1];
        s_vz[PML_NUM + 2][si] = vz[globalIdx + STRIDEY2];
        s_vz[PML_NUM + 3][si] = vz[globalIdx + STRIDEY3];
    }

    s_vx[j + 1][si] = vx[globalIdx];
    s_vy[j + 1][si] = vy[globalIdx];
    s_vz[j + 1][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdy = c_fd[0] * s_vx[j][si]     +
                  c_fd[1] * s_vx[j + 1][si] +
                  c_fd[2] * s_vx[j + 2][si] +
                  c_fd[3] * s_vx[j + 3][si] +
                  c_fd[4] * s_vx[j + 4][si];

    float dvydy = c_fd[0] * s_vy[j][si]     +
                  c_fd[1] * s_vy[j + 1][si] +
                  c_fd[2] * s_vy[j + 2][si] +
                  c_fd[3] * s_vy[j + 3][si] +
                  c_fd[4] * s_vy[j + 4][si];

    float dvzdy = c_fd[0] * s_vz[j][si]     +
                  c_fd[1] * s_vz[j + 1][si] +
                  c_fd[2] * s_vz[j + 2][si] +
                  c_fd[3] * s_vz[j + 3][si] +
                  c_fd[4] * s_vz[j + 4][si];

    j = PML_NUM - j;

    float beta       = 1.f / (1.f + c_beta0 * j * j);
    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + (lambda + 2.f) * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsxy[globalIdx] = dsxy[globalIdx] + dvxdy * (beta - 1.f) - vxy[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvzdy * (beta - 1.f) - vzy[pmlIdx] * beta * c_dt;

    dvxy[pmlIdx] += dbeta * dvxdy - alphadbeta * vxy[pmlIdx];
    dvyy[pmlIdx] += dbeta * dvydy - alphadbeta * vyy[pmlIdx];
    dvzy[pmlIdx] += dbeta * dvzdy - alphadbeta * vzy[pmlIdx];

    vxy[pmlIdx]  += dvxy[pmlIdx] * scaleB;
    dvxy[pmlIdx] *= scaleA;

    vyy[pmlIdx]  += dvyy[pmlIdx] * scaleB;
    dvyy[pmlIdx] *= scaleA;

    vzy[pmlIdx]  += dvzy[pmlIdx] * scaleB;
    dvzy[pmlIdx] *= scaleA;
}

__global__ void pml_leftv_backward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = 0.f;
        s_vx[1][si]           = 0.f;
        s_vx[2][si]           = 0.f;
        s_vx[PML_NUM + 3][si] = vx[globalIdx + STRIDEY1];

        s_vy[0][si]           = 0.f;
        s_vy[1][si]           = 0.f;
        s_vy[2][si]           = 0.f;
        s_vy[PML_NUM + 3][si] = vy[globalIdx + STRIDEY1];

        s_vz[0][si]           = 0.f;
        s_vz[1][si]           = 0.f;
        s_vz[2][si]           = 0.f;
        s_vz[PML_NUM + 3][si] = vz[globalIdx + STRIDEY1];
    }

    s_vx[j + 3][si] = vx[globalIdx];
    s_vy[j + 3][si] = vy[globalIdx];
    s_vz[j + 3][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdy = -(c_fd[4] * s_vx[j][si]     +
                    c_fd[3] * s_vx[j + 1][si] +
                    c_fd[2] * s_vx[j + 2][si] +
                    c_fd[1] * s_vx[j + 3][si] +
                    c_fd[0] * s_vx[j + 4][si]);

    float dvydy = -(c_fd[4] * s_vy[j][si]     +
                    c_fd[3] * s_vy[j + 1][si] +
                    c_fd[2] * s_vy[j + 2][si] +
                    c_fd[1] * s_vy[j + 3][si] +
                    c_fd[0] * s_vy[j + 4][si]);

    float dvzdy = -(c_fd[4] * s_vz[j][si]     +
                    c_fd[3] * s_vz[j + 1][si] +
                    c_fd[2] * s_vz[j + 2][si] +
                    c_fd[1] * s_vz[j + 3][si] +
                    c_fd[0] * s_vz[j + 4][si]);

    j = PML_NUM - j;

    float beta  = 1.f / (1.f + c_beta0 * j * j);    

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + (lambda + 2.f) * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsxy[globalIdx] = dsxy[globalIdx] + dvxdy * (beta - 1.f) - vxy[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvzdy * (beta - 1.f) - vzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dvxy[pmlIdx] += dbeta * dvxdy - alphadbeta * vxy[pmlIdx];
    dvyy[pmlIdx] += dbeta * dvydy - alphadbeta * vyy[pmlIdx];
    dvzy[pmlIdx] += dbeta * dvzdy - alphadbeta * vzy[pmlIdx];

    vxy[pmlIdx]  += dvxy[pmlIdx] * scaleB;
    dvxy[pmlIdx] *= scaleA;

    vyy[pmlIdx]  += dvyy[pmlIdx] * scaleB;
    dvyy[pmlIdx] *= scaleA;

    vzy[pmlIdx]  += dvzy[pmlIdx] * scaleB;
    dvzy[pmlIdx] *= scaleA;
}

__global__ void pml_lefts_forward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxy[PML_NUM + 4][32];
    __shared__ float s_syy[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxy[0][si]           = 0.f;
        s_sxy[PML_NUM + 1][si] = sxy[globalIdx + STRIDEY1];
        s_sxy[PML_NUM + 2][si] = sxy[globalIdx + STRIDEY2];
        s_sxy[PML_NUM + 3][si] = sxy[globalIdx + STRIDEY3];

        s_syy[0][si]           = 0.f;
        s_syy[PML_NUM + 1][si] = syy[globalIdx + STRIDEY1];
        s_syy[PML_NUM + 2][si] = syy[globalIdx + STRIDEY2];
        s_syy[PML_NUM + 3][si] = syy[globalIdx + STRIDEY3];

        s_syz[0][si]           = 0.f;
        s_syz[PML_NUM + 1][si] = syz[globalIdx + STRIDEY1];
        s_syz[PML_NUM + 2][si] = syz[globalIdx + STRIDEY2];
        s_syz[PML_NUM + 3][si] = syz[globalIdx + STRIDEY3];
    }

    s_sxy[j + 1][si] = sxy[globalIdx];
    s_syy[j + 1][si] = syy[globalIdx];
    s_syz[j + 1][si] = syz[globalIdx];

    __syncthreads();

    float dsxydy = c_fd[0] * s_sxy[j][si]     +
                   c_fd[1] * s_sxy[j + 1][si] +
                   c_fd[2] * s_sxy[j + 2][si] +
                   c_fd[3] * s_sxy[j + 3][si] +
                   c_fd[4] * s_sxy[j + 4][si];

    float dsyydy = c_fd[0] * s_syy[j][si]     +
                   c_fd[1] * s_syy[j + 1][si] +
                   c_fd[2] * s_syy[j + 2][si] +
                   c_fd[3] * s_syy[j + 3][si] +
                   c_fd[4] * s_syy[j + 4][si];

    float dsyzdy = c_fd[0] * s_syz[j][si]     +
                   c_fd[1] * s_syz[j + 1][si] +
                   c_fd[2] * s_syz[j + 2][si] +
                   c_fd[3] * s_syz[j + 3][si] +
                   c_fd[4] * s_syz[j + 4][si];

    j = PML_NUM - j;

    float beta  = 1.f / (1.f + c_beta0 * j * j);

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxydy - sxyy[pmlIdx] * beta * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyydy - syyy[pmlIdx] * beta * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsyzdy - syzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxyy[pmlIdx] += dbeta * dsxydy - alphadbeta * sxyy[pmlIdx];
    dsyyy[pmlIdx] += dbeta * dsyydy - alphadbeta * syyy[pmlIdx];
    dsyzy[pmlIdx] += dbeta * dsyzdy - alphadbeta * syzy[pmlIdx];

    sxyy[pmlIdx]  += dsxyy[pmlIdx] * scaleB;
    dsxyy[pmlIdx] *= scaleA;

    syyy[pmlIdx]  += dsyyy[pmlIdx] * scaleB;
    dsyyy[pmlIdx] *= scaleA;

    syzy[pmlIdx]  += dsyzy[pmlIdx] * scaleB;
    dsyzy[pmlIdx] *= scaleA;
}

__global__ void pml_lefts_backward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxy[PML_NUM + 4][32];
    __shared__ float s_syy[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxy[0][si]           = 0.f;
        s_sxy[1][si]           = 0.f;
        s_sxy[2][si]           = 0.f;
        s_sxy[PML_NUM + 3][si] = sxy[globalIdx + STRIDEY1];

        s_syy[0][si]           = 0.f;
        s_syy[1][si]           = 0.f;
        s_syy[2][si]           = 0.f;
        s_syy[PML_NUM + 3][si] = syy[globalIdx + STRIDEY1];

        s_syz[0][si]           = 0.f;
        s_syz[1][si]           = 0.f;
        s_syz[2][si]           = 0.f;
        s_syz[PML_NUM + 3][si] = syz[globalIdx + STRIDEY1];
    }

    s_sxy[j + 3][si] = sxy[globalIdx];
    s_syy[j + 3][si] = syy[globalIdx];
    s_syz[j + 3][si] = syz[globalIdx];

    __syncthreads();

    float dsxydy = -(c_fd[4] * s_sxy[j][si]     +
                     c_fd[3] * s_sxy[j + 1][si] +
                     c_fd[2] * s_sxy[j + 2][si] +
                     c_fd[1] * s_sxy[j + 3][si] +
                     c_fd[0] * s_sxy[j + 4][si]);

    float dsyydy = -(c_fd[4] * s_syy[j][si]     +
                     c_fd[3] * s_syy[j + 1][si] +
                     c_fd[2] * s_syy[j + 2][si] +
                     c_fd[1] * s_syy[j + 3][si] +
                     c_fd[0] * s_syy[j + 4][si]);

    float dsyzdy = -(c_fd[4] * s_syz[j][si]     +
                     c_fd[3] * s_syz[j + 1][si] +
                     c_fd[2] * s_syz[j + 2][si] +
                     c_fd[1] * s_syz[j + 3][si] +
                     c_fd[0] * s_syz[j + 4][si]);

    j = PML_NUM - j;

    float beta  = 1.f / (1.f + c_beta0 * j * j);

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxydy - sxyy[pmlIdx] * beta * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyydy - syyy[pmlIdx] * beta * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsyzdy - syzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxyy[pmlIdx] += dbeta * dsxydy - alphadbeta * sxyy[pmlIdx];
    dsyyy[pmlIdx] += dbeta * dsyydy - alphadbeta * syyy[pmlIdx];
    dsyzy[pmlIdx] += dbeta * dsyzdy - alphadbeta * syzy[pmlIdx];

    sxyy[pmlIdx]  += dsxyy[pmlIdx] * scaleB;
    dsxyy[pmlIdx] *= scaleA;

    syyy[pmlIdx]  += dsyyy[pmlIdx] * scaleB;
    dsyyy[pmlIdx] *= scaleA;

    syzy[pmlIdx]  += dsyzy[pmlIdx] * scaleB;
    dsyzy[pmlIdx] *= scaleA;
}

__global__ void pml_rightv_forward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j + NY_PML) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = vx[globalIdx - STRIDEY1];
        s_vx[PML_NUM + 1][si] = 0.f;
        s_vx[PML_NUM + 2][si] = 0.f;
        s_vx[PML_NUM + 3][si] = 0.f;

        s_vy[0][si]           = vy[globalIdx - STRIDEY1];
        s_vy[PML_NUM + 1][si] = 0.f;
        s_vy[PML_NUM + 2][si] = 0.f;
        s_vy[PML_NUM + 3][si] = 0.f;

        s_vz[0][si]           = vz[globalIdx - STRIDEY1];
        s_vz[PML_NUM + 1][si] = 0.f;
        s_vz[PML_NUM + 2][si] = 0.f;
        s_vz[PML_NUM + 3][si] = 0.f;
    }

    s_vx[j + 1][si] = vx[globalIdx];
    s_vy[j + 1][si] = vy[globalIdx];
    s_vz[j + 1][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdy = c_fd[0] * s_vx[j][si]     +
                  c_fd[1] * s_vx[j + 1][si] +
                  c_fd[2] * s_vx[j + 2][si] +
                  c_fd[3] * s_vx[j + 3][si] +
                  c_fd[4] * s_vx[j + 4][si];

    float dvydy = c_fd[0] * s_vy[j][si]     +
                  c_fd[1] * s_vy[j + 1][si] +
                  c_fd[2] * s_vy[j + 2][si] +
                  c_fd[3] * s_vy[j + 3][si] +
                  c_fd[4] * s_vy[j + 4][si];

    float dvzdy = c_fd[0] * s_vz[j][si]     +
                  c_fd[1] * s_vz[j + 1][si] +
                  c_fd[2] * s_vz[j + 2][si] +
                  c_fd[3] * s_vz[j + 3][si] +
                  c_fd[4] * s_vz[j + 4][si];

    float beta  = 1.f / (1.f + c_beta0 * j * j);    

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + (lambda + 2.f) * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsxy[globalIdx] = dsxy[globalIdx] + dvxdy * (beta - 1.f) - vxy[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvzdy * (beta - 1.f) - vzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dvxy[pmlIdx] += dbeta * dvxdy - alphadbeta * vxy[pmlIdx];
    dvyy[pmlIdx] += dbeta * dvydy - alphadbeta * vyy[pmlIdx];
    dvzy[pmlIdx] += dbeta * dvzdy - alphadbeta * vzy[pmlIdx];

    vxy[pmlIdx]  += dvxy[pmlIdx] * scaleB;
    dvxy[pmlIdx] *= scaleA;

    vyy[pmlIdx]  += dvyy[pmlIdx] * scaleB;
    dvyy[pmlIdx] *= scaleA;

    vzy[pmlIdx]  += dvzy[pmlIdx] * scaleB;
    dvzy[pmlIdx] *= scaleA;
}

__global__ void pml_rightv_backward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j + NY_PML) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = vx[globalIdx - STRIDEY3];
        s_vx[1][si]           = vx[globalIdx - STRIDEY2];
        s_vx[2][si]           = vx[globalIdx - STRIDEY1];
        s_vx[PML_NUM + 3][si] = 0.f;

        s_vy[0][si]           = vy[globalIdx - STRIDEY3];
        s_vy[1][si]           = vy[globalIdx - STRIDEY2];
        s_vy[2][si]           = vy[globalIdx - STRIDEY1];
        s_vy[PML_NUM + 3][si] = 0.f;

        s_vz[0][si]           = vz[globalIdx - STRIDEY3];
        s_vz[1][si]           = vz[globalIdx - STRIDEY2];
        s_vz[2][si]           = vz[globalIdx - STRIDEY1];
        s_vz[PML_NUM + 3][si] = 0.f;
    }

    s_vx[j + 3][si] = vx[globalIdx];
    s_vy[j + 3][si] = vy[globalIdx];
    s_vz[j + 3][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdy = -(c_fd[4] * s_vx[j][si]     +
                    c_fd[3] * s_vx[j + 1][si] +
                    c_fd[2] * s_vx[j + 2][si] +
                    c_fd[1] * s_vx[j + 3][si] +
                    c_fd[0] * s_vx[j + 4][si]);

    float dvydy = -(c_fd[4] * s_vy[j][si]     +
                    c_fd[3] * s_vy[j + 1][si] +
                    c_fd[2] * s_vy[j + 2][si] +
                    c_fd[1] * s_vy[j + 3][si] +
                    c_fd[0] * s_vy[j + 4][si]);

    float dvzdy = -(c_fd[4] * s_vz[j][si]     +
                    c_fd[3] * s_vz[j + 1][si] +
                    c_fd[2] * s_vz[j + 2][si] +
                    c_fd[1] * s_vz[j + 3][si] +
                    c_fd[0] * s_vz[j + 4][si]);

    float beta  = 1.f / (1.f + c_beta0 * j * j);    

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + (lambda + 2.f) * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + lambda * (dvydy * (beta - 1.f) - vyy[pmlIdx] * beta * c_dt);
    dsxy[globalIdx] = dsxy[globalIdx] + dvxdy * (beta - 1.f) - vxy[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvzdy * (beta - 1.f) - vzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dvxy[pmlIdx] += dbeta * dvxdy - alphadbeta * vxy[pmlIdx];
    dvyy[pmlIdx] += dbeta * dvydy - alphadbeta * vyy[pmlIdx];
    dvzy[pmlIdx] += dbeta * dvzdy - alphadbeta * vzy[pmlIdx];

    vxy[pmlIdx]  += dvxy[pmlIdx] * scaleB;
    dvxy[pmlIdx] *= scaleA;

    vyy[pmlIdx]  += dvyy[pmlIdx] * scaleB;
    dvyy[pmlIdx] *= scaleA;

    vzy[pmlIdx]  += dvzy[pmlIdx] * scaleB;
    dvzy[pmlIdx] *= scaleA;
}

__global__ void pml_rights_forward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxy[PML_NUM + 4][32];
    __shared__ float s_syy[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j + NY_PML) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxy[0][si]           = sxy[globalIdx - STRIDEY1];
        s_sxy[PML_NUM + 1][si] = 0.f;
        s_sxy[PML_NUM + 2][si] = 0.f;
        s_sxy[PML_NUM + 3][si] = 0.f;

        s_syy[0][si]           = syy[globalIdx - STRIDEY1];
        s_syy[PML_NUM + 1][si] = 0.f;
        s_syy[PML_NUM + 2][si] = 0.f;
        s_syy[PML_NUM + 3][si] = 0.f;

        s_syz[0][si]           = syz[globalIdx - STRIDEY1];
        s_syz[PML_NUM + 1][si] = 0.f;
        s_syz[PML_NUM + 2][si] = 0.f;
        s_syz[PML_NUM + 3][si] = 0.f;
    }

    s_sxy[j + 1][si] = sxy[globalIdx];
    s_syy[j + 1][si] = syy[globalIdx];
    s_syz[j + 1][si] = syz[globalIdx];

    __syncthreads();

    float dsxydy = c_fd[0] * s_sxy[j][si]     +
                   c_fd[1] * s_sxy[j + 1][si] +
                   c_fd[2] * s_sxy[j + 2][si] +
                   c_fd[3] * s_sxy[j + 3][si] +
                   c_fd[4] * s_sxy[j + 4][si];

    float dsyydy = c_fd[0] * s_syy[j][si]     +
                   c_fd[1] * s_syy[j + 1][si] +
                   c_fd[2] * s_syy[j + 2][si] +
                   c_fd[3] * s_syy[j + 3][si] +
                   c_fd[4] * s_syy[j + 4][si];

    float dsyzdy = c_fd[0] * s_syz[j][si]     +
                   c_fd[1] * s_syz[j + 1][si] +
                   c_fd[2] * s_syz[j + 2][si] +
                   c_fd[3] * s_syz[j + 3][si] +
                   c_fd[4] * s_syz[j + 4][si];

    float beta = 1.f / (1.f + c_beta0 * j * j);

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxydy - sxyy[pmlIdx] * beta * c_dt * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyydy - syyy[pmlIdx] * beta * c_dt * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsyzdy - syzy[pmlIdx] * beta * c_dt * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxyy[pmlIdx] += dbeta * dsxydy - alphadbeta * sxyy[pmlIdx];
    dsyyy[pmlIdx] += dbeta * dsyydy - alphadbeta * syyy[pmlIdx];
    dsyzy[pmlIdx] += dbeta * dsyzdy - alphadbeta * syzy[pmlIdx];

    sxyy[pmlIdx]  += dsxyy[pmlIdx] * scaleB;
    dsxyy[pmlIdx] *= scaleA;

    syyy[pmlIdx]  += dsyyy[pmlIdx] * scaleB;
    dsyyy[pmlIdx] *= scaleA;

    syzy[pmlIdx]  += dsyzy[pmlIdx] * scaleB;
    dsyzy[pmlIdx] *= scaleA;
}

__global__ void pml_rights_backward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxy[PML_NUM + 4][32];
    __shared__ float s_syy[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j + NY_PML) * NX + i;
    int pmlIdx    = (k * PML_NUM + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxy[0][si]           = sxy[globalIdx - STRIDEY3];
        s_sxy[1][si]           = sxy[globalIdx - STRIDEY2];
        s_sxy[2][si]           = sxy[globalIdx - STRIDEY1];
        s_sxy[PML_NUM + 3][si] = 0.f;

        s_syy[0][si]           = syy[globalIdx - STRIDEY3];
        s_syy[1][si]           = syy[globalIdx - STRIDEY2];
        s_syy[2][si]           = syy[globalIdx - STRIDEY1];
        s_syy[PML_NUM + 3][si] = 0.f;

        s_syz[0][si]           = syz[globalIdx - STRIDEY3];
        s_syz[1][si]           = syz[globalIdx - STRIDEY2];
        s_syz[2][si]           = syz[globalIdx - STRIDEY1];
        s_syz[PML_NUM + 3][si] = 0.f;
    }

    s_sxy[j + 3][si] = sxy[globalIdx];
    s_syy[j + 3][si] = syy[globalIdx];
    s_syz[j + 3][si] = syz[globalIdx];

    __syncthreads();

    float dsxydy = -(c_fd[4] * s_sxy[j][si]     +
                     c_fd[3] * s_sxy[j + 1][si] +
                     c_fd[2] * s_sxy[j + 2][si] +
                     c_fd[1] * s_sxy[j + 3][si] +
                     c_fd[0] * s_sxy[j + 4][si]);

    float dsyydy = -(c_fd[4] * s_syy[j][si]     +
                     c_fd[3] * s_syy[j + 1][si] +
                     c_fd[2] * s_syy[j + 2][si] +
                     c_fd[1] * s_syy[j + 3][si] +
                     c_fd[0] * s_syy[j + 4][si]);

    float dsyzdy = -(c_fd[4] * s_syz[j][si]     +
                     c_fd[3] * s_syz[j + 1][si] +
                     c_fd[2] * s_syz[j + 2][si] +
                     c_fd[1] * s_syz[j + 3][si] +
                     c_fd[0] * s_syz[j + 4][si]);

    float beta = 1.f / (1.f + c_beta0 * j * j);

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxydy - sxyy[pmlIdx] * beta * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyydy - syyy[pmlIdx] * beta * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dsyzdy - syzy[pmlIdx] * beta * c_dt;

    float d          = c_d0 * j * j;
    float alpha      = c_alpha0 - c_alpha01 * j;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxyy[pmlIdx] += dbeta * dsxydy - alphadbeta * sxyy[pmlIdx];
    dsyyy[pmlIdx] += dbeta * dsyydy - alphadbeta * syyy[pmlIdx];
    dsyzy[pmlIdx] += dbeta * dsyzdy - alphadbeta * syzy[pmlIdx];

    sxyy[pmlIdx]  += dsxyy[pmlIdx] * scaleB;
    dsxyy[pmlIdx] *= scaleA;

    syyy[pmlIdx]  += dsyyy[pmlIdx] * scaleB;
    dsyyy[pmlIdx] *= scaleA;

    syzy[pmlIdx]  += dsyzy[pmlIdx] * scaleB;
    dsyzy[pmlIdx] *= scaleA;
}

__global__ void pml_bottomv_forward(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.y;
    int globalIdx = ((k + NZ_PML) * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = vx[globalIdx - STRIDEZ1];
        s_vx[PML_NUM + 1][si] = 0.f;
        s_vx[PML_NUM + 2][si] = 0.f;
        s_vx[PML_NUM + 3][si] = 0.f;

        s_vy[0][si]           = vy[globalIdx - STRIDEZ1];
        s_vy[PML_NUM + 1][si] = 0.f;
        s_vy[PML_NUM + 2][si] = 0.f;
        s_vy[PML_NUM + 3][si] = 0.f;

        s_vz[0][si]           = vz[globalIdx - STRIDEZ1];
        s_vz[PML_NUM + 1][si] = 0.f;
        s_vz[PML_NUM + 2][si] = 0.f;
        s_vz[PML_NUM + 3][si] = 0.f;
    }

    s_vx[k + 1][si] = vx[globalIdx];
    s_vy[k + 1][si] = vy[globalIdx];
    s_vz[k + 1][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdz = c_fd[0] * s_vx[k][si]     +
                  c_fd[1] * s_vx[k + 1][si] +
                  c_fd[2] * s_vx[k + 2][si] +
                  c_fd[3] * s_vx[k + 3][si] +
                  c_fd[4] * s_vx[k + 4][si];

    float dvydz = c_fd[0] * s_vy[k][si]     +
                  c_fd[1] * s_vy[k + 1][si] +
                  c_fd[2] * s_vy[k + 2][si] +
                  c_fd[3] * s_vy[k + 3][si] +
                  c_fd[4] * s_vy[k + 4][si];

    float dvzdz = c_fd[0] * s_vz[k][si]     +
                  c_fd[1] * s_vz[k + 1][si] +
                  c_fd[2] * s_vz[k + 2][si] +
                  c_fd[3] * s_vz[k + 3][si] +
                  c_fd[4] * s_vz[k + 4][si];

    float beta  = 1.f / (1.f + c_beta0 * k * k);    

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + (lambda + 2.f) * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dsxz[globalIdx] = dsxz[globalIdx] + dvxdz * (beta - 1.f) - vxz[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvydz * (beta - 1.f) - vyz[pmlIdx] * beta * c_dt;

    float d          = c_d0 * k * k;
    float alpha      = c_alpha0 - c_alpha01 * k;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dvxz[pmlIdx] += dbeta * dvxdz - alphadbeta * vxz[pmlIdx];
    dvyz[pmlIdx] += dbeta * dvydz - alphadbeta * vyz[pmlIdx];
    dvzz[pmlIdx] += dbeta * dvzdz - alphadbeta * vzz[pmlIdx];

    vxz[pmlIdx]  += dvxz[pmlIdx] * scaleB;
    dvxz[pmlIdx] *= scaleA;

    vyz[pmlIdx]  += dvyz[pmlIdx] * scaleB;
    dvyz[pmlIdx] *= scaleA;

    vzz[pmlIdx]  += dvzz[pmlIdx] * scaleB;
    dvzz[pmlIdx] *= scaleA;
}

__global__ void pml_bottomv_backward(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB) {
    __shared__ float s_vx[PML_NUM + 4][32];
    __shared__ float s_vy[PML_NUM + 4][32];
    __shared__ float s_vz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.y;
    int globalIdx = ((k + NZ_PML) * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_vx[0][si]           = vx[globalIdx - STRIDEZ3];
        s_vx[1][si]           = vx[globalIdx - STRIDEZ2];
        s_vx[2][si]           = vx[globalIdx - STRIDEZ1];
        s_vx[PML_NUM + 3][si] = 0.f;

        s_vy[0][si]           = vy[globalIdx - STRIDEZ3];
        s_vy[1][si]           = vy[globalIdx - STRIDEZ2];
        s_vy[2][si]           = vy[globalIdx - STRIDEZ1];
        s_vy[PML_NUM + 3][si] = 0.f;

        s_vz[0][si]           = vz[globalIdx - STRIDEZ3];
        s_vz[1][si]           = vz[globalIdx - STRIDEZ2];
        s_vz[2][si]           = vz[globalIdx - STRIDEZ1];
        s_vz[PML_NUM + 3][si] = 0.f;
    }

    s_vx[k + 3][si] = vx[globalIdx];
    s_vy[k + 3][si] = vy[globalIdx];
    s_vz[k + 3][si] = vz[globalIdx];

    __syncthreads();

    float lambda = lambda_in[globalIdx];

    float dvxdz = -(c_fd[4] * s_vx[k][si]     +
                    c_fd[3] * s_vx[k + 1][si] +
                    c_fd[2] * s_vx[k + 2][si] +
                    c_fd[1] * s_vx[k + 3][si] +
                    c_fd[0] * s_vx[k + 4][si]);

    float dvydz = -(c_fd[4] * s_vy[k][si]     +
                    c_fd[3] * s_vy[k + 1][si] +
                    c_fd[2] * s_vy[k + 2][si] +
                    c_fd[1] * s_vy[k + 3][si] +
                    c_fd[0] * s_vy[k + 4][si]);

    float dvzdz = -(c_fd[4] * s_vz[k][si]     +
                    c_fd[3] * s_vz[k + 1][si] +
                    c_fd[2] * s_vz[k + 2][si] +
                    c_fd[1] * s_vz[k + 3][si] +
                    c_fd[0] * s_vz[k + 4][si]);

    float beta  = 1.f / (1.f + c_beta0 * k * k);    

    dsxx[globalIdx] = dsxx[globalIdx] + lambda * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dsyy[globalIdx] = dsyy[globalIdx] + lambda * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dszz[globalIdx] = dszz[globalIdx] + (lambda + 2.f) * (dvzdz * (beta - 1.f) - vzz[pmlIdx] * beta * c_dt);
    dsxz[globalIdx] = dsxz[globalIdx] + dvxdz * (beta - 1.f) - vxz[pmlIdx] * beta * c_dt;
    dsyz[globalIdx] = dsyz[globalIdx] + dvydz * (beta - 1.f) - vyz[pmlIdx] * beta * c_dt;

    float d          = c_d0 * k * k;
    float alpha      = c_alpha0 - c_alpha01 * k;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dvxz[pmlIdx] += dbeta * dvxdz - alphadbeta * vxz[pmlIdx];
    dvyz[pmlIdx] += dbeta * dvydz - alphadbeta * vyz[pmlIdx];
    dvzz[pmlIdx] += dbeta * dvzdz - alphadbeta * vzz[pmlIdx];

    vxz[pmlIdx]  += dvxz[pmlIdx] * scaleB;
    dvxz[pmlIdx] *= scaleA;

    vyz[pmlIdx]  += dvyz[pmlIdx] * scaleB;
    dvyz[pmlIdx] *= scaleA;

    vzz[pmlIdx]  += dvzz[pmlIdx] * scaleB;
    dvzz[pmlIdx] *= scaleA;
}

__global__ void pml_bottoms_forward(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxz[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    __shared__ float s_szz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.y;
    int globalIdx = ((k + NZ_PML) * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxz[0][si]           = sxz[globalIdx - STRIDEZ1];
        s_sxz[PML_NUM + 1][si] = 0.f;
        s_sxz[PML_NUM + 2][si] = 0.f;
        s_sxz[PML_NUM + 3][si] = 0.f;

        s_syz[0][si]           = syz[globalIdx - STRIDEZ1];
        s_syz[PML_NUM + 1][si] = 0.f;
        s_syz[PML_NUM + 2][si] = 0.f;
        s_syz[PML_NUM + 3][si] = 0.f;

        s_szz[0][si]           = szz[globalIdx - STRIDEZ1];
        s_szz[PML_NUM + 1][si] = 0.f;
        s_szz[PML_NUM + 2][si] = 0.f;
        s_szz[PML_NUM + 3][si] = 0.f;
    }

    s_sxz[k + 1][si] = sxz[globalIdx];
    s_syz[k + 1][si] = syz[globalIdx];
    s_szz[k + 1][si] = szz[globalIdx];

    __syncthreads();

    float dsxzdz = c_fd[0] * s_sxz[k][si]     +
                   c_fd[1] * s_sxz[k + 1][si] +
                   c_fd[2] * s_sxz[k + 2][si] +
                   c_fd[3] * s_sxz[k + 3][si] +
                   c_fd[4] * s_sxz[k + 4][si];

    float dsyzdz = c_fd[0] * s_syz[k][si]     +
                   c_fd[1] * s_syz[k + 1][si] +
                   c_fd[2] * s_syz[k + 2][si] +
                   c_fd[3] * s_syz[k + 3][si] +
                   c_fd[4] * s_syz[k + 4][si];

    float dszzdz = c_fd[0] * s_szz[k][si]     +
                   c_fd[1] * s_szz[k + 1][si] +
                   c_fd[2] * s_szz[k + 2][si] +
                   c_fd[3] * s_szz[k + 3][si] +
                   c_fd[4] * s_szz[k + 4][si];

    float beta  = 1.f / (1.f + c_beta0 * k * k);    

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxzdz - sxzz[pmlIdx] * beta * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyzdz - syzz[pmlIdx] * beta * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dszzdz - szzz[pmlIdx] * beta * c_dt;

    float d          = c_d0 * k * k;
    float alpha      = c_alpha0 - c_alpha01 * k;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxzz[pmlIdx] += dbeta * dsxzdz - alphadbeta * sxzz[pmlIdx];
    dsyzz[pmlIdx] += dbeta * dsyzdz - alphadbeta * syzz[pmlIdx];
    dszzz[pmlIdx] += dbeta * dszzdz - alphadbeta * szzz[pmlIdx];

    sxzz[pmlIdx]  += dsxzz[pmlIdx] * scaleB;
    dsxzz[pmlIdx] *= scaleA;

    syzz[pmlIdx]  += dsyzz[pmlIdx] * scaleB;
    dsyzz[pmlIdx] *= scaleA;

    szzz[pmlIdx]  += dszzz[pmlIdx] * scaleB;
    dszzz[pmlIdx] *= scaleA;
}

__global__ void pml_bottoms_backward(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB) {
    __shared__ float s_sxz[PML_NUM + 4][32];
    __shared__ float s_syz[PML_NUM + 4][32];
    __shared__ float s_szz[PML_NUM + 4][32];
    
    //global i, j, k
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.y;
    int globalIdx = ((k + NZ_PML) * NY + j) * NX + i;
    int pmlIdx    = (k * NY + j) * NX + i;

    //shared memory index
    int si = threadIdx.x;

    if (threadIdx.y == 0) {
        s_sxz[0][si]           = sxz[globalIdx - STRIDEZ3];
        s_sxz[1][si]           = sxz[globalIdx - STRIDEZ2];
        s_sxz[2][si]           = sxz[globalIdx - STRIDEZ1];
        s_sxz[PML_NUM + 3][si] = 0.f;

        s_syz[0][si]           = syz[globalIdx - STRIDEZ3];
        s_syz[1][si]           = syz[globalIdx - STRIDEZ2];
        s_syz[2][si]           = syz[globalIdx - STRIDEZ1];
        s_syz[PML_NUM + 3][si] = 0.f;

        s_szz[0][si]           = szz[globalIdx - STRIDEZ3];
        s_szz[1][si]           = szz[globalIdx - STRIDEZ2];
        s_szz[2][si]           = szz[globalIdx - STRIDEZ1];
        s_szz[PML_NUM + 3][si] = 0.f;
    }

    s_sxz[k + 3][si] = sxz[globalIdx];
    s_syz[k + 3][si] = syz[globalIdx];
    s_szz[k + 3][si] = szz[globalIdx];

    __syncthreads();

    float dsxzdz = -(c_fd[4] * s_sxz[k][si]     +
                     c_fd[3] * s_sxz[k + 1][si] +
                     c_fd[2] * s_sxz[k + 2][si] +
                     c_fd[1] * s_sxz[k + 3][si] +
                     c_fd[0] * s_sxz[k + 4][si]);

    float dsyzdz = -(c_fd[4] * s_syz[k][si]     +
                     c_fd[3] * s_syz[k + 1][si] +
                     c_fd[2] * s_syz[k + 2][si] +
                     c_fd[1] * s_syz[k + 3][si] +
                     c_fd[0] * s_syz[k + 4][si]);

    float dszzdz = -(c_fd[4] * s_szz[k][si]     +
                     c_fd[3] * s_szz[k + 1][si] +
                     c_fd[2] * s_szz[k + 2][si] +
                     c_fd[1] * s_szz[k + 3][si] +
                     c_fd[0] * s_szz[k + 4][si]);

    float beta  = 1.f / (1.f + c_beta0 * k * k);    

    dvx[globalIdx] = dvx[globalIdx] + (beta - 1.f) * dsxzdz - sxzz[pmlIdx] * beta * c_dt;
    dvy[globalIdx] = dvy[globalIdx] + (beta - 1.f) * dsyzdz - syzz[pmlIdx] * beta * c_dt;
    dvz[globalIdx] = dvz[globalIdx] + (beta - 1.f) * dszzdz - szzz[pmlIdx] * beta * c_dt;

    float d          = c_d0 * k * k;
    float alpha      = c_alpha0 - c_alpha01 * k;
    float dbeta      = d * beta;
    float alphadbeta = (alpha + dbeta) * c_dt;

    dsxzz[pmlIdx] += dbeta * dsxzdz - alphadbeta * sxzz[pmlIdx];
    dsyzz[pmlIdx] += dbeta * dsyzdz - alphadbeta * syzz[pmlIdx];
    dszzz[pmlIdx] += dbeta * dszzdz - alphadbeta * szzz[pmlIdx];

    sxzz[pmlIdx]  += dsxzz[pmlIdx] * scaleB;
    dsxzz[pmlIdx] *= scaleA;

    syzz[pmlIdx]  += dsyzz[pmlIdx] * scaleB;
    dsyzz[pmlIdx] *= scaleA;

    szzz[pmlIdx]  += dszzz[pmlIdx] * scaleB;
    dszzz[pmlIdx] *= scaleA;
}

__global__ void free_surface_forward(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in) {
    //gloabl i, j
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalIdx = j * NX + i;

    float lambda = lambda_in[globalIdx];
    float dvzdz  = -dszz[globalIdx] / (lambda + 2.f);

    dsxx[globalIdx] += dvzdz * lambda;
    dsyy[globalIdx] += dvzdz * lambda;

    dsxz[globalIdx] = 0.f;
    dsyz[globalIdx] = 0.f;
    dszz[globalIdx] = 0.f;
}

__global__ void free_surface_backward(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in, float *vx, float *vy, float *vz) {
    __shared__ float s_vx[NX][4];
    __shared__ float s_vy[NX][4];
    __shared__ float s_vz[NX][4];

    //gloabl i, j
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalIdx = j * NX + i;

    s_vx[i][0] = vx[globalIdx];
    s_vx[i][1] = vx[globalIdx + STRIDEZ1];
    s_vx[i][2] = vx[globalIdx + STRIDEZ2];
    s_vx[i][3] = vx[globalIdx + STRIDEZ3];

    s_vy[i][0] = vy[globalIdx];
    s_vy[i][1] = vy[globalIdx + STRIDEZ1];
    s_vy[i][2] = vy[globalIdx + STRIDEZ2];
    s_vy[i][3] = vy[globalIdx + STRIDEZ3];

    s_vz[i][0] = vz[globalIdx];
    s_vz[i][1] = vz[globalIdx + STRIDEZ1];
    s_vz[i][2] = vz[globalIdx + STRIDEZ2];
    s_vz[i][3] = vz[globalIdx + STRIDEZ3];

    __syncthreads();

    float lambda = lambda_in[globalIdx];
    float dvxdz  = -dsxz[globalIdx];
    float dvydz  = -dsyz[globalIdx];
    float dvzdz  = -dszz[globalIdx] / (lambda + 2.f);

    dsxz[globalIdx] = 0.f;
    dsyz[globalIdx] = 0.f;
    dszz[globalIdx] = 0.f;

    dsxx[globalIdx] += dvzdz * lambda;
    dsyy[globalIdx] += dvzdz * lambda;

    globalIdx += STRIDEZ1;
    lambda = lambda_in[globalIdx];
    dvxdz = -0.5f * dvxdz + c_fd_fs[0] * s_vx[i][2] + c_fd_fs[1] * s_vx[i][1] + c_fd_fs[2] * s_vx[i][0];
    dvydz = -0.5f * dvydz + c_fd_fs[0] * s_vy[i][2] + c_fd_fs[1] * s_vy[i][1] + c_fd_fs[2] * s_vy[i][0];
    dvzdz = -0.5f * dvzdz + c_fd_fs[0] * s_vz[i][2] + c_fd_fs[1] * s_vz[i][1] + c_fd_fs[2] * s_vz[i][0];

    dsxx[globalIdx] += dvzdz * lambda;
    dsyy[globalIdx] += dvzdz * lambda;
    dszz[globalIdx] += dvzdz * (lambda + 2.f);
    dsxz[globalIdx] += dvxdz;
    dsyz[globalIdx] += dvydz;

    globalIdx += STRIDEZ1;
    lambda = lambda_in[globalIdx];
    dvxdz = -0.5f * dvxdz + c_fd_fs[0] * s_vx[i][3] + c_fd_fs[1] * s_vx[i][2] + c_fd_fs[2] * s_vx[i][1];
    dvydz = -0.5f * dvydz + c_fd_fs[0] * s_vy[i][3] + c_fd_fs[1] * s_vy[i][2] + c_fd_fs[2] * s_vy[i][1];
    dvzdz = -0.5f * dvzdz + c_fd_fs[0] * s_vz[i][3] + c_fd_fs[1] * s_vz[i][2] + c_fd_fs[2] * s_vz[i][1];

    dsxx[globalIdx] += dvzdz * lambda;
    dsyy[globalIdx] += dvzdz * lambda;
    dszz[globalIdx] += dvzdz * (lambda + 2.f);
    dsxz[globalIdx] += dvxdz;
    dsyz[globalIdx] += dvydz;
}

__global__ void backward_xcoor(float *sxx1, float *sxy1, float *sxz1, float *syy1, float *syz1, float *szz1, float *sxx2, float *sxy2, float *sxz2, float *syy2, float *syz2, float *szz2, float *Klambda, float *Kmu) {
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    float sxxf = sxx2[globalIdx];
    float sxyf = sxy2[globalIdx];
    float sxzf = sxz2[globalIdx];
    float syyf = syy2[globalIdx];
    float syzf = syz2[globalIdx];
    float szzf = szz2[globalIdx];

    float sxxb = sxx1[globalIdx];
    float sxyb = sxy1[globalIdx];
    float sxzb = sxz1[globalIdx];
    float syyb = syy1[globalIdx];
    float syzb = syz1[globalIdx];
    float szzb = szz1[globalIdx];

    Kmu[globalIdx]     += sxxf * sxxb + sxyf * sxyb + sxzf * sxzb + syyf * syyb + syzf * syzb + szzf * szzb;
    Klambda[globalIdx] += (sxxf + syyf + szzf) * (sxxb + syyb + szzb);

    sxx2[globalIdx] = 0.f;
    sxy2[globalIdx] = 0.f;
    sxz2[globalIdx] = 0.f;
    syy2[globalIdx] = 0.f;
    syz2[globalIdx] = 0.f;
    szz2[globalIdx] = 0.f;
}

__global__ void kernel_processing(float *Klambda, float *Kmu, float *lambda_in, float *mu_in) {
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    float mu     = mu_in[globalIdx];
    float lambda = lambda_in[globalIdx] * mu;

    float Klambda_out = Klambda[globalIdx];
    float Kmu_out     = Kmu[globalIdx];

    Klambda_out /= (3.f * lambda + 2.f * mu) * (3.f * lambda + 2.f * mu);
    Kmu_out /= 2.f * mu * mu;
    Kmu_out += 2.f * lambda * (3.f * lambda + 4.f * mu) * Klambda_out;

    Klambda[globalIdx] = Klambda_out;
    Kmu[globalIdx]     = Kmu_out;
}

__global__ void update_stress(float *sxx, float *dsxx, float *syy, float *dsyy, float *szz, float *dszz, float *sxy, float *dsxy, float *sxz, float *dsxz, float *syz, float *dsyz, float *mu, float scaleA, float scaleB) {
    //gloabl i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    float muScale = mu[globalIdx] * scaleB;

    sxx[globalIdx]  += dsxx[globalIdx] * muScale;
    dsxx[globalIdx] *= scaleA;

    syy[globalIdx]  += dsyy[globalIdx] * muScale;
    dsyy[globalIdx] *= scaleA;

    szz[globalIdx]  += dszz[globalIdx] * muScale;
    dszz[globalIdx] *= scaleA;

    sxy[globalIdx]  += dsxy[globalIdx] * muScale;
    dsxy[globalIdx] *= scaleA;

    sxz[globalIdx]  += dsxz[globalIdx] * muScale;
    dsxz[globalIdx] *= scaleA;

    syz[globalIdx]  += dsyz[globalIdx] * muScale;
    dsyz[globalIdx] *= scaleA;
}

__global__ void update_velocity(float *fx_in, float *dfx_in, float *fy_in, float *dfy_in, float *fz_in, float *dfz_in, float *rho, float scaleA, float scaleB) {
    //gloabl i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    float rhoScale = rho[globalIdx] * scaleB;

    fx_in[globalIdx]  += dfx_in[globalIdx] * rhoScale;
    dfx_in[globalIdx] *= scaleA;

    fy_in[globalIdx]  += dfy_in[globalIdx] * rhoScale;
    dfy_in[globalIdx] *= scaleA;

    fz_in[globalIdx]  += dfz_in[globalIdx] * rhoScale;
    dfz_in[globalIdx] *= scaleA;
}

//may be use a more general source
__global__ void source_inject(float *f, int x, int y, float* skx, float* sky, float stf) {
    //gloabl i, j, k
    int i = threadIdx.x;
    int j = threadIdx.y;
    int globalIdx = (j + y) * NX + (i + x);
    float scale = skx[i] * sky[j];
    //float scale = expf(-(i - 8) * (i - 8)) * expf(-(j - 8) * (j - 8));

    f[globalIdx] += stf * scale;
}

__global__ void source_inject_gaussian(float *f, int x, int y, float skx, float sky, float stf) {
    //gloabl i, j, k
    int i = threadIdx.x;
    int j = threadIdx.y;
    int globalIdx = (j + y) * NX + (i + x);
    float scale = expf(-(i + skx - 8) * (i + skx - 8)) * expf(-(j + sky - 8) * (j + sky - 8));

    f[globalIdx] += stf * scale;
}

__device__ inline void MyAtomicAdd (float *address, float value)
 {
   int oldval, newval, readback;
 
   oldval = __float_as_int(*address);
   newval = __float_as_int(__int_as_float(oldval) + value);
   while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
     {
      oldval = readback;
      newval = __float_as_int(__int_as_float(oldval) + value);
     }
 }

__global__ void source_inject_station_gaussian(int *iX, int *iY, float *kx, float *ky, float *vz, float *f) {
    //gloabl i, j, k
    int x = iX[blockIdx.x];
    int y = iY[blockIdx.x];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int globalIdx = (j + y) * NX + (i + x);
    float scale = expf(-(i + kx[blockIdx.x] - 8) * (i + kx[blockIdx.x] - 8)) * expf(-(j + ky[blockIdx.y] - 8) * (j + ky[blockIdx.y] - 8));
    scale *= vz[blockIdx.x];

    MyAtomicAdd(f + globalIdx, scale);
}

__global__ void source_inject_station(int *iX, int *iY, float *kx, float *ky, float *vz, float *f) {
    //gloabl i, j, k
    int x = iX[blockIdx.x];
    int y = iY[blockIdx.x];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int globalIdx = (j + y) * NX + (i + x);
    float scale = kx[blockIdx.x * 16 + i] * ky[blockIdx.x * 16 + j];
    //float scale = expf(-(i - 8) * (i - 8)) * expf(-(j - 8) * (j - 8));
    scale *= vz[blockIdx.x];

    MyAtomicAdd(f + globalIdx, scale);
}

__global__ void station_extract(int *iX, int *iY, float *kx, float *ky, float *vz_out, float *vz_in) {
    __shared__ float s_vz[16][16];

    int i = iX[blockIdx.x];
    int j = iY[blockIdx.x];

    int si = threadIdx.x;
    int sj = threadIdx.y;

    int globalIdx = (j + sj) * NX + i + si;

    float sx = kx[si + blockIdx.x * 16];
    float sy = ky[sj + blockIdx.x * 16];
    s_vz[sj][si] = vz_in[globalIdx] * sx * sy;

    __syncthreads();

    if (sj < 8) {
        s_vz[sj][si] += s_vz[sj + 8][si];
    }

    __syncthreads();

    if (sj < 4) {
        s_vz[sj][si] += s_vz[sj + 4][si];
    }

    __syncthreads();

    if (sj < 2) {
        s_vz[sj][si] += s_vz[sj + 2][si];
    }

    __syncthreads();

    if (sj == 0) {
        s_vz[0][si] += s_vz[1][si];
    }

    __syncthreads();

    if (sj == 0 && si < 8) {
        s_vz[0][si] += s_vz[0][si + 8];
    }

    __syncthreads();

    if (sj == 0 && si < 4) {
        s_vz[0][si] += s_vz[0][si + 4];
    }

    __syncthreads();

    if (sj == 0 && si < 2) {
        s_vz[0][si] += s_vz[0][si + 2];
    }

    __syncthreads();

    if (sj == 0 && si == 0) {
        vz_out[blockIdx.x] += s_vz[0][0] + s_vz[0][1];
    }
}

__global__ void station_clip(float *Klambda0, float *Kmu0, int *iX, int *iY) {
    int i = iX[blockIdx.x];
    int j = iY[blockIdx.x];

    int si = threadIdx.x;
    int sj = threadIdx.y;

    int globalIdx = (threadIdx.z * NY + j + sj) * NX + i + si;

    Klambda0[globalIdx] = 0.f;
    Kmu0[globalIdx] = 0.f;
}

__global__ void add_kernel(float *Klambda0, float *Kmu0, float *Klambda, float *Kmu) {
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    Klambda[globalIdx] += Klambda0[globalIdx];
    Kmu[globalIdx] += Kmu0[globalIdx];
}

__global__ void update_mu(float stepLength, float *Kmu, float *mu, float *rho) {
    //global i, j, k
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int globalIdx = (k * NY + j) * NX + i;

    float oldmu = mu[globalIdx];

    oldmu += Kmu[globalIdx] * stepLength;

    float lowermu = 4e6 / rho[globalIdx];
    if (oldmu < lowermu) {
        oldmu = lowermu;
    }

    mu[globalIdx] = oldmu;
}
