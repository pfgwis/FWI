#ifndef _XDFWI_H
#define _XDFWI_H

#include <stdio.h>

#include "def.h"

/**
 * velocityModel_t: velocity model structure
 */
typedef struct velocityModel_t {
    float   xMin;
    int     xNum;
    float   dx;
    float   xMax;
    float   yMin;
    int     yNum;
    float   dy;
    float   yMax;
    int     zNum;
    int     depthNum;
    int     gridNum;
    float   dz;
    float*  vp;
    float*  vs;
    float*  rho;
} velocityModel_t;

/**
 * station_t: Station data abstraction.
 */
typedef struct station_t {
    int    id;
    int    iX;
    int    iY;
    int    indexXYZ;
    float  x;
    float  y;
    float  lon;
    float  lat;
    float* vz;
    float  dis;
} station_t;

typedef struct Param_t {
    //required input parameters
    char  theParametersInputFileName[MAX_FILENAME_LEN];
    char  theVelocityModelPath[MAX_FILENAME_LEN];
    float theFreq;

    //MPI information
    int myID;
    int theGroupSize;

    //velocity model
    velocityModel_t theVelocityModel;
    float vMin;
    float vMax;

    //modeling paramters
    float theDeltaT;
    int   theTotalTimeSteps;
    int   theOutputStartTimeStep;

    //whole model description
    float  xMin;
    int    xNum;
    float  xMax;
    float  dx;
    float  yMin;
    int    yNum;
    float  yMax;
    float  dy;
    int    zNum;
    float  dz;
    int    xyNum;
    int    xzNum;
    int    yzNum;
    int    xyzNum;

    int   useGPU;

    //cuda parameters
    float *d_lambda;
    float *d_mu;
    float *d_rho;
    float *d_sxx;
    float *d_sxy;
    float *d_sxz;
    float *d_syy;
    float *d_syz;
    float *d_szz;
    float *d_dsxx;
    float *d_dsxy;
    float *d_dsxz;
    float *d_dsyy;
    float *d_dsyz;
    float *d_dszz;
    float *d_vx;
    float *d_vy;
    float *d_vz;
    float *d_dvx;
    float *d_dvy;
    float *d_dvz;

    //cuda pml
    float *d_vxx_front;
    float *d_vyx_front;
    float *d_vzx_front;
    float *d_dvxx_front;
    float *d_dvyx_front;
    float *d_dvzx_front;

    float *d_sxxx_front;
    float *d_sxyx_front;
    float *d_sxzx_front;
    float *d_dsxxx_front;
    float *d_dsxyx_front;
    float *d_dsxzx_front;

    float *d_vxx_back;
    float *d_vyx_back;
    float *d_vzx_back;
    float *d_dvxx_back;
    float *d_dvyx_back;
    float *d_dvzx_back;

    float *d_sxxx_back;
    float *d_sxyx_back;
    float *d_sxzx_back;
    float *d_dsxxx_back;
    float *d_dsxyx_back;
    float *d_dsxzx_back;

    float *d_vxy_left;
    float *d_vyy_left;
    float *d_vzy_left;
    float *d_dvxy_left;
    float *d_dvyy_left;
    float *d_dvzy_left;

    float *d_sxyy_left;
    float *d_syyy_left;
    float *d_syzy_left;
    float *d_dsxyy_left;
    float *d_dsyyy_left;
    float *d_dsyzy_left;

    float *d_vxy_right;
    float *d_vyy_right;
    float *d_vzy_right;
    float *d_dvxy_right;
    float *d_dvyy_right;
    float *d_dvzy_right;

    float *d_sxyy_right;
    float *d_syyy_right;
    float *d_syzy_right;
    float *d_dsxyy_right;
    float *d_dsyyy_right;
    float *d_dsyzy_right;

    float *d_vxz_bottom;
    float *d_vyz_bottom;
    float *d_vzz_bottom;
    float *d_dvxz_bottom;
    float *d_dvyz_bottom;
    float *d_dvzz_bottom;

    float *d_sxzz_bottom;
    float *d_syzz_bottom;
    float *d_szzz_bottom;
    float *d_dsxzz_bottom;
    float *d_dsyzz_bottom;
    float *d_dszzz_bottom;

    //cpu parameters
    float *lambda;
    float *mu;
    float *rho;
    float *sxx;
    float *sxy;
    float *sxz;
    float *syy;
    float *syz;
    float *szz;
    float *dsxx;
    float *dsxy;
    float *dsxz;
    float *dsyy;
    float *dsyz;
    float *dszz;
    float *vx;
    float *vy;
    float *vz;
    float *dvx;
    float *dvy;
    float *dvz;

    //pml parameter
    int   pmlNum;
    int   pmlNumTotal;
    int   pmlMap[17][6];
    float *pml_dx;
    float *pml_dy;
    float *pml_dz;

    //pml field
    float *sxx_xpml;
    float *sxy_xpml;
    float *sxz_xpml;
    float *syy_xpml;
    float *szz_xpml;
    float *vx_xpml;
    float *vy_xpml;
    float *vz_xpml;
    float *dsxx_xpml;
    float *dsxy_xpml;
    float *dsxz_xpml;
    float *dsyy_xpml;
    float *dszz_xpml;
    float *dvx_xpml;
    float *dvy_xpml;
    float *dvz_xpml;

    float *sxx_ypml;
    float *sxy_ypml;
    float *syy_ypml;
    float *syz_ypml;
    float *szz_ypml;
    float *vx_ypml;
    float *vy_ypml;
    float *vz_ypml;
    float *dsxx_ypml;
    float *dsxy_ypml;
    float *dsyy_ypml;
    float *dsyz_ypml;
    float *dszz_ypml;
    float *dvx_ypml;
    float *dvy_ypml;
    float *dvz_ypml;

    float *sxx_zpml;
    float *sxz_zpml;
    float *syy_zpml;
    float *syz_zpml;
    float *szz_zpml;
    float *vx_zpml;
    float *vy_zpml;
    float *vz_zpml;
    float *dsxx_zpml;
    float *dsxz_zpml;
    float *dsyy_zpml;
    float *dsyz_zpml;
    float *dszz_zpml;
    float *dvx_zpml;
    float *dvy_zpml;
    float *dvz_zpml;

    //fd coefficient
    float fd_x[5];
    float fd_y[5];
    float fd_z[5];

    //rk parameters
    int nRKStage;
    float A[6];
    float B[6];
    float C[6];

    //source
    int   theSourceX;
    int   theSourceY;
    int   theSourceZ;
    float theSx;
    float theSy;
    float theSrcLat;
    float theSrcLon;
    int   theSrcID;
    float *d_srcKx;
    float *d_srcKy;
    int   mySourceIndexXYZ;
    float *mySourceTimeFunction;

    //station
    int myNumberOfStations;
    station_t *myStations;
    int    outputStation;
    int    *d_station_iX;
    int    *d_station_iY;
    float  *d_station_x;
    float  *d_station_y;
    float  *d_station_vz;

    //wavefield output
    FILE *theWavefieldOutFp;
    int  outputWavefield;
    int  theOutputRate;

    //kernel parameters
    FILE  *kernelOutputFp;
    float *Klambda;
    float *Kmu;
    float *Kmu_all;
    float *Krho;
    float *d_Klambda;
    float *d_Kmu;
    float *pk;
    float *gk;
    int   kernelZNum;
    int   xyzNumKernel;

    int theSimulationMode;
    int currentIteration;
    int totalIteration;
    int currentNumberOfStation;
    float totalMisfit;
    float averageMisfit;
    float stepLength;
    float kernelMax;
    float kernelMean;
    float *allMisfit;
} Param_t;

Param_t* Param;

//paramters.c
int  parameter_init(int argc, char** argv);
void parameter_free();

//xdfwiRun.c
int xdfwi_run();

//velocityModel.c
int  velocity_model_init();
void velocity_model_write();
void velocity_model_query(float x, float y, int iZ, float *vp, float *vs, float *rho);
void velocity_model_delete();
float phi_local(int i, float csi, float etha, float *csii, float *ethai);

//modelingDomain.c
int modeling_domain_init();

//modeling.c
void forward_modeling_cpu();
void backward_modeling_cpu();
void forward_modeling_gpu();
void forward_modeling_gpu_memory();
void backward_modeling_gpu();

//modelingSub.c
void cpu_scale_df(int iStage);
void cpu_update_f(int iStage);
void cpu_x_derivative(int isForward);
void cpu_y_derivative(int isForward);
void cpu_z_derivative(int isForward);
void cpu_free_surface(int isForward);
void cpu_pml_exchange();
void cpu_pml_combine();
void memory_set_zeros();

//util.c
int  xd_abort(const char* fname, const char* perror_msg, const char* format, ...);
int  parsetext(FILE* fp, const char* querystring, const char type, void* result);
void print_time_status(double *myTime, const char* timeName);
void pml_map_init();
int  get_pml_region_iXYZ(int iX, int iY, int iZ);
int  get_pml_region_iPml(int iPml);
int  iXYZ_to_iPml(int iX, int iY, int iZ, int iRegion);
int  iPml_to_iXYZ(int iPml);
float kaiser_sinc(float x);

//source.c
int  source_init();
void source_delete();
void source_inject_forward(int iTime);
void source_inject_backward(int iTime);
void source_inject_forward_multiple_stage(int iTime, int iStage);

//wavefield.c
void wavefield_extract_init();
void wavefield_extract();

//station.c
int  station_init();
void station_delete();
void station_extract(int step);
void station_download();
void station_output_qc();

//cudaPrepare.c
void cuda_init();
void cuda_set_field_memory_zero();
void cuda_free_all_field_memory();
void cuda_df_dxyz(int isForward);
void cuda_free_surface(int isForward);
void cuda_update_f(int iStage);
void cuda_pml(int isForward, int iStage);
void cuda_download_wavefield(int iOutput);
void cuda_upload_wavefield(int iOutput);

//cpuPrepare.c
void cpu_init();
void cpu_free_all_field_memory();
void cuda_set_field_parameter();
void xy2latlon_azequaldist(float lat0, float lon0, float x, float y, float* lat1, float* lon1);
void latlon2xy_azequaldist(float lat0, float lon0, float lat1, float lon1, float* x, float* y);

//kernel.c
void kernel_init();
void kernel_forward_extract();
void *kernel_forward_extract_multi_thread(void* nullPointer);
void kernel_backward_xcoor();
void kernel_finalize();
void kernel_output_qc();
void kernel_delete();
void kernel_processing();
float get_initial_step_length();
void update_model();
float get_step_length_CG();
void update_model_CG();
void read_kernel();
void kernel_add_all();

//residual.c
void compute_residual();
void upload_residual();

//cudaKernelLaunch.cu
int  launch_cudaMalloc(void **devPtr, size_t size);
int  launch_cudaMallocHost(void **hostPtr, size_t size);
int  launch_cudaMemset(void *devPtr, int value, size_t count);
int  launch_cudaMemcpy(void *dst, const void *src, size_t count, int direction);
void launch_cudaMemcpyAsync(void *dst, const void *src, size_t count, int direction);
int  launch_set_modeling_parameter(float dt, float dx);
void launch_cudaFree(void *devPtr);
void launch_cudaFreeHost(void *hostPtr);
void launch_delete_modeling_parameter();
void launch_dstress_dxy(float *fx_in, float *fy_in, float * f_out, int isForward);
void launch_dstress_dz(float *f_in, float *f_out, int isForward);
void launch_dvxy_dxy(float *vx_in, float *vy_in, float *lambda_in, float *dsxy_out, float *dsxx_out, float *dsyy_out, float *szz_out, int isForward);
void launch_dvxz_dxz(float *vx_in, float *vz_in, float *lambda_in, float *dsxz_out, float *dsxx_out, float *dsyy_out, float *szz_out, int isForward);
void launch_dvelocity_dy(float *f_in, float *f_out, int isForward);
void launch_dvelocity_dz(float *f_in, float *f_out, int isForward);
void launch_free_surface(float *dsxz, float *dsyz, float *dszz, float *dsxx, float *dsyy, float *lambda_in, float *vx, float *vy, float *vz, int isForward);
void launch_update_stress(float *sxx, float *dsxx, float *syy, float *dsyy, float *szz, float *dszz, float *sxy, float *dsxy, float *sxz, float *dsxz, float *syz, float *dsyz, float *mu, float scaleA, float scaleB);
void launch_update_velocity(float *fx, float *dfx, float *fy, float *dfy, float *fz, float *dfz, float *rho, float scaleA, float scaleB);
void launch_source_inject_forward(float *f, int x, int y, float *skx, float *sky, float stf);
void launch_source_inject_backward(int *iX, int *iY, float *kx, float *ky, float *vz, float *f, int numberOfStations);
void launch_source_inject_forward_gaussian(float *f, int x, int y, float skx, float sky, float stf);
void launch_source_inject_backward_gaussian(int *iX, int *iY, float *kx, float *ky, float *vz, float *f, int numberOfStations);
void launch_cuda_device_synchronize();
void launch_cuda_stream_synchronize();
void launch_pml_leftv(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward);
void launch_pml_lefts(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward);
void launch_pml_rightv(float *vx, float *vy, float *vz, float *vxy, float *vyy, float *vzy, float *dvxy, float *dvyy, float *dvzy, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward);
void launch_pml_rights(float *sxy, float *syy, float *syz, float *sxyy, float *syyy, float *syzy, float *dsxyy, float *dsyyy, float *dsyzy, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward);
void launch_pml_bottomv(float *vx, float *vy, float *vz, float *vxz, float *vyz, float *vzz, float *dvxz, float *dvyz, float *dvzz, float *dsxx, float *dsyy, float *dszz, float *dsxz, float *dsyz, float *lambda_in, float scaleA, float scaleB, int isForward);
void launch_pml_bottoms(float *sxz, float *syz, float *szz, float *sxzz, float *syzz, float *szzz, float *dsxzz, float *dsyzz, float *dszzz, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward);
void launch_pml_frontv(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB, int isForward);
void launch_pml_fronts(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward);
void launch_pml_backv(float *vx, float *vy, float *vz, float *vxx, float *vyx, float *vzx, float *dvxx, float *dvyx, float *dvzx, float *dsxx, float *dsyy, float *dszz, float *dsxy, float *dsxz, float *lambda_in, float scaleA, float scaleB, int isForward);
void launch_pml_backs(float *sxx, float *sxy, float *sxz, float *sxxx, float *sxyx, float *sxzx, float *dsxxx, float *dsxyx, float *dsxzx, float *dvx, float *dvy, float *dvz, float scaleA, float scaleB, int isForward);
void launch_station_extract(int *iX, int *iY, float *x, float *y, float *vz_out, float *vz_in, int myNumberOfStations);
void launch_kernel_backward_xcoor(float *sxx1, float *sxy1, float *sxz1, float *syy1, float *syz1, float *szz1, float *sxx2, float *sxy2, float *sxz2, float *syy2, float *syz2, float *szz2, float *Klambda, float *Kmu, int zNum);
void launch_kernel_finalize(float *Klambda, float *Kmu, float *lambda, float *mu, int zNum);
void launch_station_clip(float *Klambda0, float *Kmu0, int *iX, int *iY, int clipWidth);
void launch_add_kernel(float *Klambda0, float *Kmu0, float *Klambda, float *Kmu, int kernelZNum);
void launch_update_model(float stepLength, float *Kmu, float *mu, float *rho, int kernelZNum);

#endif
