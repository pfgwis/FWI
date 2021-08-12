#ifndef __CUDAKERNEL_H
#define __CUDAKERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

#define NX 256
#define NY 320
#define NZ 32

#define STRIDEY1 NX
#define STRIDEY2 STRIDEY1 * 2
#define STRIDEY3 STRIDEY1 * 3

#define STRIDEZ1 NX * NY
#define STRIDEZ2 STRIDEZ1 * 2
#define STRIDEZ3 STRIDEZ1 * 3

#define SPONGE_NUM 50

#define PML_NUM 10
#define NX_PML (NX - PML_NUM)
#define NY_PML (NY - PML_NUM)
#define NZ_PML (NZ - PML_NUM)

int set_constant_memory(float dt, float dx);

__global__ void dstress_dxy_forward(float *fx_in, float *fy_in, float *f_out);
__global__ void dstress_dxy_backward(float *fx_in, float *fy_in, float *f_out);

__global__ void dstress_dz_forward(float *f_in, float *f_out);
__global__ void dstress_dz_backward(float *f_in, float *f_out);

__global__ void dvxy_dxy_forward(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *dszz_out);
__global__ void dvxy_dxy_backward(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *dszz_out);

__global__ void dvxz_dxz_forward(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *dszz_out);
__global__ void dvxz_dxz_backward(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *dszz_out);

__global__ void dvelocity_dy_forward(float *f_in, float *f_out);
__global__ void dvelocity_dy_backward(float *f_in, float *f_out);

__global__ void dvelocity_dy_forward_32x32(float *f_in, float *f_out);
__global__ void dvelocity_dy_backward_32x32(float *f_in, float *f_out);

__global__ void dvelocity_dz_forward(float *f_in, float *f_out);
__global__ void dvelocity_dz_backward(float *f_in, float *f_out);

__global__ void pml_frontv_forward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB);
__global__ void pml_frontv_backward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB);

__global__ void pml_fronts_forward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);
__global__ void pml_fronts_backward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);

__global__ void pml_backv_forward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB);
__global__ void pml_backv_backward(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB);

__global__ void pml_backs_forward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);
__global__ void pml_backs_backward(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);

__global__ void pml_backv_forward(float *vx, float *vy, float *vz, float *vxxb, float *vyxb, float *vzxb, float *dvxxb, float *dvyxb, float *dvzxb, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in);
__global__ void pml_backv_backward(float *vx, float *vy, float *vz, float *vxxb, float *vyxb, float *vzxb, float *dvxxb, float *dvyxb, float *dvzxb, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in);

__global__ void pml_backs_forward(float *sxx, float *sxy, float *sxz, float *sxxxb, float *sxyxb, float *sxzxb, float *dsxxxb, float *dsxyxb, float *dsxzxb, float *dvx, float *dvy, float *dvz);
__global__ void pml_backs_backward(float *sxx, float *sxy, float *sxz, float *sxxxb, float *sxyxb, float *sxzxb, float *dsxxxb, float *dsxyxb, float *dsxzxb, float *dvx, float *dvy, float *dvz);

__global__ void pml_leftv_forward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB);
__global__ void pml_leftv_backward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB);

__global__ void pml_lefts_forward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);
__global__ void pml_lefts_backward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);

__global__ void pml_rightv_forward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB);
__global__ void pml_rightv_backward(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB);

__global__ void pml_rights_forward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);
__global__ void pml_rights_backward(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);

__global__ void pml_bottomv_forward(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB);
__global__ void pml_bottomv_backward(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB);

__global__ void pml_bottoms_forward(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);
__global__ void pml_bottoms_backward(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB);

__global__ void free_surface_forward(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in);
__global__ void free_surface_backward(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in, float *vx, float *vy, float *vz);

__global__ void backward_xcoor(float *sxx1, float *sxy1, float *sxz1, float *syy1, float *syz1, float *szz1, float *sxx2, float *sxy2, float *sxz2, float *syy2, float *syz2, float *szz2, float *Klambda, float *Kmu);

__global__ void kernel_processing(float *Klambda, float *Kmu, float *lambda, float *mu);

__global__ void update_stress(float *sxx, float *dsxx, float *syy, float *dsyy, float *szz, float *dszz, float *sxy, float *dsxy, float *sxz, float *dsxz, float *syz, float *dsyz, float *mu, float scaleA, float scaleB);
__global__ void update_velocity(float *fx_in, float *dfx_in, float *fy_in, float *dfy_in, float *fz_in, float *dfz_in, float *rho, float scaleA, float scaleB);

__global__ void source_inject(float *f, int x, int y, float* skx, float* sky, float stf);

__global__ void source_inject_station(int *iX, int *iY, float *kx, float *ky, float *vz, float *f);

__global__ void station_extract(int *iX, int *iY, float *kx, float *ky, float *vz_out, float *vz_in);

__global__ void station_clip(float *Klambda0, float *Kmu0, int *iX, int *iY);

__global__ void add_kernel(float *Klambda0, float *Kmu0, float *Klambda, float *Kmu);

__global__ void update_mu(float stepLength, float *Kmu, float *rho, float *mu);

__global__ void source_inject_gaussian(float *f, int x, int y, float skx, float sky, float stf);

__global__ void source_inject_station_gaussian(int *iX, int *iY, float *kx, float *ky, float *vz, float *f);


#endif
