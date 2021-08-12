#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "xdfwi.h"

extern Param_t *Param;

//local function
void cuda_allocate_all_field_memory();
void cuda_set_field_parameter();
void cuda_set_modeling_parameter();

void cuda_init() {
	cuda_allocate_all_field_memory();

    //cuda_set_field_parameter();

    cuda_set_modeling_parameter();
}

void cuda_allocate_all_field_memory() {
	int bytes;

	bytes = Param->xyzNum * sizeof(float);

	if (launch_cudaMalloc((void**)&Param->d_lambda, bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_mu,     bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_rho,    bytes) != 0 ||
    	launch_cudaMalloc((void**)&Param->d_sxx,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_sxy,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_sxz,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_syy,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_syz,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_szz,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_vx,     bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_vy,     bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_vz,     bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dsxx,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dsxy,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dsxz,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dsyy,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dsyz,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dszz,   bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dvx,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dvy,    bytes) != 0 ||
        launch_cudaMalloc((void**)&Param->d_dvz,    bytes) != 0) {
        xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the fields\n");
    }

    //front and back
    bytes = Param->yzNum * Param->pmlNum * sizeof(float);

	if (launch_cudaMalloc((void**)&Param->d_vxx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vyx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vzx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvxx_front, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvyx_front, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvzx_front, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_sxxx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_sxyx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_sxzx_front,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxxx_front, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxyx_front, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxzx_front, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_vxx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vyx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vzx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvxx_back, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvyx_back, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvzx_back, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_sxxx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_sxyx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_sxzx_back,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxxx_back, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxyx_back, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxzx_back, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

    //left and right
	bytes = Param->xzNum * Param->pmlNum * sizeof(float);

	if (launch_cudaMalloc((void**)&Param->d_vxy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vyy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vzy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvxy_left, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvyy_left, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvzy_left, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_sxyy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_syyy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_syzy_left,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxyy_left, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsyyy_left, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsyzy_left, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_vxy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vyy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vzy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvxy_right, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvyy_right, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvzy_right, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_sxyy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_syyy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_syzy_right,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxyy_right, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsyyy_right, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsyzy_right, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	//bottom
	bytes = Param->xyNum * Param->pmlNum * sizeof(float);

	if (launch_cudaMalloc((void**)&Param->d_vxz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vyz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_vzz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvxz_bottom, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvyz_bottom, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dvzz_bottom, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}

	if (launch_cudaMalloc((void**)&Param->d_sxzz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_syzz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_szzz_bottom,  bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsxzz_bottom, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dsyzz_bottom, bytes) != 0 ||
		launch_cudaMalloc((void**)&Param->d_dszzz_bottom, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the pml fields\n");
	}
}

void cuda_set_field_memory_zero() {
	int bytes;

	bytes = Param->xyzNum * sizeof(float);

	if (launch_cudaMemset(Param->d_sxx,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_sxy,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_sxz,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_syy,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_syz,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_szz,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_vx,   0, bytes) != 0 ||
		launch_cudaMemset(Param->d_vy,   0, bytes) != 0 ||
		launch_cudaMemset(Param->d_vz,   0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dsxx, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dsxy, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dsxz, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dsyy, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dsyz, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dszz, 0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dvx,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dvy,  0, bytes) != 0 ||
		launch_cudaMemset(Param->d_dvz,  0, bytes) != 0) {
		xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the fields\n");
	}

	//front and back
	bytes = Param->yzNum * Param->pmlNum * sizeof(float);

    if (launch_cudaMemset(Param->d_vxx_front,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vyx_front,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vzx_front,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxxx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxyx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxzx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvxx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvyx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvzx_front,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxxx_front, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxyx_front, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxzx_front, 0, bytes) != 0) {
    	xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the pml fields\n");
    }

    if (launch_cudaMemset(Param->d_vxx_back,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vyx_back,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vzx_back,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxxx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxyx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxzx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvxx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvyx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvzx_back,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxxx_back, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxyx_back, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxzx_back, 0, bytes) != 0) {
    	xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the pml dfields\n");
    }

	//left and right
	bytes = Param->xzNum * Param->pmlNum * sizeof(float);

    if (launch_cudaMemset(Param->d_vxy_left,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vyy_left,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vzy_left,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxyy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_syyy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_syzy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvxy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvyy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvzy_left,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxyy_left, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsyyy_left, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsyzy_left, 0, bytes) != 0) {
    	xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the pml dfields\n");
    }

    if (launch_cudaMemset(Param->d_vxy_right,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vyy_right,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vzy_right,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxyy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_syyy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_syzy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvxy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvyy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvzy_right,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxyy_right, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsyyy_right, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsyzy_right, 0, bytes) != 0) {
    	xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the pml dfields\n");
    }

    //bottom
	bytes = Param->xyNum * Param->pmlNum * sizeof(float);

    if (launch_cudaMemset(Param->d_vxz_bottom,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vyz_bottom,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_vzz_bottom,   0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_sxzz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_syzz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_szzz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvxz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvyz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dvzz_bottom,  0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsxzz_bottom, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dsyzz_bottom, 0, bytes) != 0 ||
    	launch_cudaMemset(Param->d_dszzz_bottom, 0, bytes) != 0) {
    	xd_abort(__func__, "launch_cudaMemset() failed", "Memory setting to 0 failed for the pml dfields\n");
    }
}

void latlon2xy_azequaldist(float lat0, float lon0, float lat1, float lon1, float* x, float* y) {
    const float R  = 6378137;
    const float pi = 3.14159265358979323846;
    const float D0 = pi/180.0;

    lat0 *= D0;
    lon0 *= D0;
    lat1 *= D0;
    lon1 *= D0;

    float c = acos(sin(lat0)*sin(lat1)+cos(lat0)*cos(lat1)*cos(lon1-lon0));
    float k;

    if (fabs(c) < 1e-20) {
        k = 0;
    } else {
        k = c / sin(c);
    }

    *y = R*k*cos(lat1)*sin(lon1-lon0);
    *x = R*k*(cos(lat0)*sin(lat1)-sin(lat0)*cos(lat1)*cos(lon1-lon0));
}

void xy2latlon_azequaldist(float lat0, float lon0, float x, float y, float* lat1, float* lon1) {
    const float R  = 6378137;
    const float pi = 3.14159265358979323846;
    const float D0 = pi/180.0;

    float rho = sqrt(x*x+y*y);
    float c   = rho/R;

    lat0 *= D0;
    lon0 *= D0;

    if (rho == 0) {
        *lat1 = lat0;
        *lon1 = lon0;
    } else {
        *lat1 = asin(cos(c)*sin(lat0)+x*sin(c)*cos(lat0)/rho);
        *lon1 = lon0+atan(y*sin(c)/(rho*cos(lat0)*cos(c)-x*sin(lat0)*sin(c)));
    }

    *lat1 /= D0;
    *lon1 /= D0;
}

void cuda_set_field_parameter() {
	int iX, iY, iZ, iXYZ;
	int bytes = Param->xyzNum * sizeof(float);
    float vp, vs, rho;
    float x, y;

    Param->xMin = 0.;
    Param->xMax = 0.;

    Param->xMax = Param->xMin + Param->dx * (Param->xNum - 1);
    Param->yMax = Param->yMin + Param->dx * (Param->yNum - 1);

    iXYZ = 0;
    for (iZ = 0; iZ < Param->zNum; iZ++) {
        for (iY = 0; iY < Param->yNum; iY++) {
            for (iX = 0; iX < Param->xNum; iX++) {
                vs = Param->theVelocityModel.vs[iXYZ];

                //Param->lambda[iXYZ] = rho * (vp * vp -  2 * vs * vs);
                Param->mu[iXYZ]     = vs * vs;
		        //Param->lambda[iXYZ] = vp * vp / vs / vs - 2.f; // lambda / mu instead of lambda
                Param->lambda[iXYZ] = 1.f;
                Param->rho[iXYZ]    = 1.f; //using 1/rho instead of rho
                iXYZ++;
            }
        }
    }

    if (launch_cudaMemcpy(Param->d_lambda, Param->lambda, bytes, 0) != 0 ||
    	launch_cudaMemcpy(Param->d_mu,     Param->mu,     bytes, 0) != 0 ||
    	launch_cudaMemcpy(Param->d_rho,    Param->rho,    bytes, 0) != 0) {
    	xd_abort(__func__, "launch_cudaMemcpy() failed", "Memory copy from host to device failed for the fields\n");
    }
}

void cuda_set_modeling_parameter() {
    //we need to make sure that mx, my and mz must be integral multiples of sPencils (4)
    //mx and my mush be multiples of lPencils (32)
    /*
    if (Param->xNum % sPencils != 0 || Param->zNum % sPencils != 0 || Param->yNum % sPencils ||
        Param->xNum % lPencils != 0 || Param->zNum % lPencils != 0) {
        xd_abort(__func__, "", "Error model dimensions\n");
    }
    */

    if (launch_set_modeling_parameter(Param->theDeltaT, Param->dx) != 0) {
        xd_abort(__func__, "", "Error set modeling parameters\n");
    }

    //runge kutta coefficient
/*    Param->nRKStage = 6;

    Param->A[0] = -0.4919575f;
    Param->A[1] = -0.8946264f;
    Param->A[2] = -1.5526678f;
    Param->A[3] = -3.4077973f;
    Param->A[4] = -1.0742640f;
    Param->A[5] = 0.f;

    Param->B[0] = 0.1453095f;
    Param->B[1] = 0.4653797f;
    Param->B[2] = 0.4675397f;
    Param->B[3] = 0.7795279f;
    Param->B[4] = 0.3574327f;
    Param->B[5] = 0.15f;

    Param->C[0] = 0.f;
    Param->C[1] = 0.1453095f;
    Param->C[2] = 0.3817422f;
    Param->C[3] = 0.6367813f;
    Param->C[4] = 0.7560744f;
    Param->C[5] = 0.9271047f;

    Param->nRKStage = 5;

    Param->A[0] = -0.6913965f;
    Param->A[1] = -2.655155f;
    Param->A[2] = -0.8147688f;
    Param->A[3] = -0.6686587f;
    Param->A[4] = -0.f;

    Param->B[0] = 0.1f;
    Param->B[1] = 0.75f;
    Param->B[2] = 0.7f;
    Param->B[3] = 0.479313f;
    Param->B[4] = 0.310392f;

    Param->C[0] = 0.f;
    Param->C[1] = 0.1f;
    Param->C[2] = 0.3315201f;
    Param->C[3] = 0.4577796f;
    Param->C[4] = 0.8666528f;
*/
    Param->nRKStage = 3;

    Param->A[0] = -5.f / 9.f;
    Param->A[1] = -153.f / 128.f;
    Param->A[2] = 0.f;

    Param->B[0] = 1.f / 3.f;
    Param->B[1] = 15.f / 16.f;
    Param->B[2] = 8.f / 15.f;

    Param->C[0] = 0.f;
    Param->C[1] = 1.f / 3.f;
    Param->C[2] = 3.f / 4.f;
/*
    Param->nRKStage = 5;

    Param->A[0] = -567301805773. / 1357537059087.;
    Param->A[1] = -2404267990393. / 2016746695238.;
    Param->A[2] = -3550918686646. / 2091501179385.;
    Param->A[3] = -1275806237668. / 842570457699.;
    Param->A[4] = 0.;

    Param->B[0] = 1432997174477. / 9575080441755.;
    Param->B[1] = 5161836677717. / 13612068292357.;
    Param->B[2] = 1720146321549. / 2090206949498.;
    Param->B[3] = 3134564353537. / 4481467310338.;
    Param->B[4] = 2277821191437. / 14882151754819.;

    Param->C[0] = 0.;
    Param->C[1] = 1432997174477. / 9575080441755.;
    Param->C[2] = 2526269341429. / 6820363962896.;
    Param->C[3] = 2006345519317. / 3224310063776.;
    Param->C[4] = 2802321613138. / 2924317926251.;
    
    Param->nRKStage = 2;

    Param->A[0] = -1.f;
    Param->A[1] = 0.f;

    Param->B[0] = 1.f;
    Param->B[1] = 0.5f;

    Param->C[0] = 0.f;
    Param->C[1] = 0.5f;
        
    Param->nRKStage = 4;

    Param->A[0] = -1.f;
    Param->A[1] = -0.5f;
    Param->A[2] = -4.f;
    Param->A[3] = 0.f;

    Param->B[0] = 0.5f;
    Param->B[1] = 0.5f;
    Param->B[2] = 1.f;
    Param->B[3] = 1.f / 6.f;

    Param->C[0] = 0.f;
    Param->C[1] = 0.5f;
    Param->C[2] = 0.5f;
    Param->C[3] = 1.f;

    Param->nRKStage = 4;

    Param->A[0] = -5.f / 9.f;
    Param->A[1] = -5.f / 3.f;
    Param->A[2] = -5.f;
    Param->A[3] = 0.f;

    Param->B[0] = 1.f / 3.f;
    Param->B[1] = 1.f;
    Param->B[2] = 1.f;
    Param->B[3] = 1.f / 8.f;

    Param->C[0] = 0.f;
    Param->C[1] = 1.f / 3.f;
    Param->C[2] = 2.f / 3.f;
    Param->C[3] = 1.f;
*/
}

void cuda_free_all_field_memory() {
	launch_delete_modeling_parameter();

	launch_cudaFree(Param->d_lambda);
	launch_cudaFree(Param->d_mu);
	launch_cudaFree(Param->d_rho);
	launch_cudaFree(Param->d_sxx);
	launch_cudaFree(Param->d_sxy);
	launch_cudaFree(Param->d_sxz);
	launch_cudaFree(Param->d_syy);
	launch_cudaFree(Param->d_syz);
	launch_cudaFree(Param->d_szz);
	launch_cudaFree(Param->d_vx);
	launch_cudaFree(Param->d_vy);
	launch_cudaFree(Param->d_vz);
	launch_cudaFree(Param->d_dsxx);
	launch_cudaFree(Param->d_dsxy);
	launch_cudaFree(Param->d_dsxz);
	launch_cudaFree(Param->d_dsyy);
	launch_cudaFree(Param->d_dsyz);
	launch_cudaFree(Param->d_dszz);
	launch_cudaFree(Param->d_dvx);
	launch_cudaFree(Param->d_dvy);
	launch_cudaFree(Param->d_dvz);

	launch_cudaFree(Param->d_vxx_front);
	launch_cudaFree(Param->d_vyx_front);
	launch_cudaFree(Param->d_vzx_front);
	launch_cudaFree(Param->d_dvxx_front);
	launch_cudaFree(Param->d_dvyx_front);
	launch_cudaFree(Param->d_dvzx_front);

	launch_cudaFree(Param->d_sxxx_front);
	launch_cudaFree(Param->d_sxyx_front);
	launch_cudaFree(Param->d_sxzx_front);
	launch_cudaFree(Param->d_dsxxx_front);
	launch_cudaFree(Param->d_dsxyx_front);
	launch_cudaFree(Param->d_dsxzx_front);

	launch_cudaFree(Param->d_vxx_back);
	launch_cudaFree(Param->d_vyx_back);
	launch_cudaFree(Param->d_vzx_back);
	launch_cudaFree(Param->d_dvxx_back);
	launch_cudaFree(Param->d_dvyx_back);
	launch_cudaFree(Param->d_dvzx_back);

	launch_cudaFree(Param->d_sxxx_back);
	launch_cudaFree(Param->d_sxyx_back);
	launch_cudaFree(Param->d_sxzx_back);
	launch_cudaFree(Param->d_dsxxx_back);
	launch_cudaFree(Param->d_dsxyx_back);
	launch_cudaFree(Param->d_dsxzx_back);

	launch_cudaFree(Param->d_vxy_left);
	launch_cudaFree(Param->d_vyy_left);
	launch_cudaFree(Param->d_vzy_left);
	launch_cudaFree(Param->d_dvxy_left);
	launch_cudaFree(Param->d_dvyy_left);
	launch_cudaFree(Param->d_dvzy_left);

	launch_cudaFree(Param->d_sxyy_left);
	launch_cudaFree(Param->d_syyy_left);
	launch_cudaFree(Param->d_syzy_left);
	launch_cudaFree(Param->d_dsxyy_left);
	launch_cudaFree(Param->d_dsyyy_left);
	launch_cudaFree(Param->d_dsyzy_left);

	launch_cudaFree(Param->d_vxy_right);
	launch_cudaFree(Param->d_vyy_right);
	launch_cudaFree(Param->d_vzy_right);
	launch_cudaFree(Param->d_dvxy_right);
	launch_cudaFree(Param->d_dvyy_right);
	launch_cudaFree(Param->d_dvzy_right);

	launch_cudaFree(Param->d_sxyy_right);
	launch_cudaFree(Param->d_syyy_right);
	launch_cudaFree(Param->d_syzy_right);
	launch_cudaFree(Param->d_dsxyy_right);
	launch_cudaFree(Param->d_dsyyy_right);
	launch_cudaFree(Param->d_dsyzy_right);

	launch_cudaFree(Param->d_vxz_bottom);
	launch_cudaFree(Param->d_vyz_bottom);
	launch_cudaFree(Param->d_vzz_bottom);
	launch_cudaFree(Param->d_dvxz_bottom);
	launch_cudaFree(Param->d_dvyz_bottom);
	launch_cudaFree(Param->d_dvzz_bottom);

	launch_cudaFree(Param->d_sxzz_bottom);
	launch_cudaFree(Param->d_syzz_bottom);
	launch_cudaFree(Param->d_szzz_bottom);
	launch_cudaFree(Param->d_dsxzz_bottom);
	launch_cudaFree(Param->d_dsyzz_bottom);
	launch_cudaFree(Param->d_dszzz_bottom);

    free(Param->lambda);
    free(Param->mu);
    free(Param->rho);
}

void cuda_df_dxyz(int isForward) {
	launch_dstress_dxy(Param->d_sxx, Param->d_sxy, Param->d_dvx, isForward);
	launch_dstress_dz(Param->d_sxz, Param->d_dvx, isForward);

	launch_dstress_dxy(Param->d_sxy, Param->d_syy, Param->d_dvy, isForward);
	launch_dstress_dz(Param->d_syz, Param->d_dvy, isForward);

	launch_dstress_dxy(Param->d_sxz, Param->d_syz, Param->d_dvz, isForward);
	launch_dstress_dz(Param->d_szz, Param->d_dvz, isForward);

	launch_dvxy_dxy(Param->d_vx, Param->d_vy, Param->d_lambda, Param->d_dsxy, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, isForward);
	launch_dvxz_dxz(Param->d_vx, Param->d_vz, Param->d_lambda, Param->d_dsxz, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, isForward);

	launch_dvelocity_dy(Param->d_vz, Param->d_dsyz, isForward);
	launch_dvelocity_dz(Param->d_vy, Param->d_dsyz, isForward);
}

void cuda_pml(int isForward, int iStage) {
	launch_pml_frontv(Param->d_vx, Param->d_vy, Param->d_vz, Param->d_vxx_front, Param->d_vyx_front, Param->d_vzx_front, Param->d_dvxx_front, Param->d_dvyx_front, Param->d_dvzx_front, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, Param->d_dsxy, Param->d_dsxz, Param->d_lambda, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_fronts(Param->d_sxx, Param->d_sxy, Param->d_sxz, Param->d_sxxx_front, Param->d_sxyx_front, Param->d_sxzx_front, Param->d_dsxxx_front, Param->d_dsxyx_front, Param->d_dsxzx_front, Param->d_dvx, Param->d_dvy, Param->d_dvz, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_backv(Param->d_vx, Param->d_vy, Param->d_vz, Param->d_vxx_back, Param->d_vyx_back, Param->d_vzx_back, Param->d_dvxx_back, Param->d_dvyx_back, Param->d_dvzx_back, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, Param->d_dsxy, Param->d_dsxz, Param->d_lambda, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_backs(Param->d_sxx, Param->d_sxy, Param->d_sxz, Param->d_sxxx_back, Param->d_sxyx_back, Param->d_sxzx_back, Param->d_dsxxx_back, Param->d_dsxyx_back, Param->d_dsxzx_back, Param->d_dvx, Param->d_dvy, Param->d_dvz, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_leftv(Param->d_vx, Param->d_vy, Param->d_vz, Param->d_vxy_left, Param->d_vyy_left, Param->d_vzy_left, Param->d_dvxy_left, Param->d_dvyy_left, Param->d_dvzy_left, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, Param->d_dsxy, Param->d_dsyz, Param->d_lambda, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_lefts(Param->d_sxy, Param->d_syy, Param->d_syz, Param->d_sxyy_left, Param->d_syyy_left, Param->d_syzy_left, Param->d_dsxyy_left, Param->d_dsyyy_left, Param->d_dsyzy_left, Param->d_dvx, Param->d_dvy, Param->d_dvz, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_rightv(Param->d_vx, Param->d_vy, Param->d_vz, Param->d_vxy_right, Param->d_vyy_right, Param->d_vzy_right, Param->d_dvxy_right, Param->d_dvyy_right, Param->d_dvzy_right, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, Param->d_dsxy, Param->d_dsyz, Param->d_lambda, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_rights(Param->d_sxy, Param->d_syy, Param->d_syz, Param->d_sxyy_right, Param->d_syyy_right, Param->d_syzy_right, Param->d_dsxyy_right, Param->d_dsyyy_right, Param->d_dsyzy_right, Param->d_dvx, Param->d_dvy, Param->d_dvz, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_bottomv(Param->d_vx, Param->d_vy, Param->d_vz, Param->d_vxz_bottom, Param->d_vyz_bottom, Param->d_vzz_bottom, Param->d_dvxz_bottom, Param->d_dvyz_bottom, Param->d_dvzz_bottom, Param->d_dsxx, Param->d_dsyy, Param->d_dszz, Param->d_dsxz, Param->d_dsyz, Param->d_lambda, Param->A[iStage], Param->B[iStage], isForward);
	launch_pml_bottoms(Param->d_sxz, Param->d_syz, Param->d_szz, Param->d_sxzz_bottom, Param->d_syzz_bottom, Param->d_szzz_bottom, Param->d_dsxzz_bottom, Param->d_dsyzz_bottom, Param->d_dszzz_bottom, Param->d_dvx, Param->d_dvy, Param->d_dvz, Param->A[iStage], Param->B[iStage], isForward);
}

void cuda_free_surface(int isForward) {
	launch_free_surface(Param->d_dsxz, Param->d_dsyz, Param->d_dszz, Param->d_dsxx, Param->d_dsyy, Param->d_lambda, Param->d_vx, Param->d_vy, Param->d_vz, isForward);
}

void cuda_update_f(int iStage) {
	launch_update_stress(Param->d_sxx, Param->d_dsxx, Param->d_syy, Param->d_dsyy, Param->d_szz, Param->d_dszz, Param->d_sxz, Param->d_dsxz, Param->d_sxy, Param->d_dsxy, Param->d_syz, Param->d_dsyz, Param->d_mu, Param->A[iStage], Param->B[iStage]);
	launch_update_velocity(Param->d_vx, Param->d_dvx, Param->d_vy, Param->d_dvy, Param->d_vz, Param->d_dvz, Param->d_rho, Param->A[iStage], Param->B[iStage]);
}

void cuda_download_wavefield(int iOutput) {
	int bytes;
    int offset = iOutput * Param->xyzNumKernel;

	if (Param->outputWavefield) {
        bytes  = Param->xyzNum * sizeof(float);

		launch_cudaMemcpyAsync(Param->vz, Param->d_vz, bytes, 1);
	}

	if (Param->theSimulationMode != 0) {
        bytes  = Param->xyzNumKernel * sizeof(float);

		launch_cudaMemcpyAsync(Param->sxx + offset, Param->d_sxx, bytes, 1);
		launch_cudaMemcpyAsync(Param->sxy + offset, Param->d_sxy, bytes, 1);
		launch_cudaMemcpyAsync(Param->sxz + offset, Param->d_sxz, bytes, 1);
		launch_cudaMemcpyAsync(Param->syy + offset, Param->d_syy, bytes, 1);
		launch_cudaMemcpyAsync(Param->syz + offset, Param->d_syz, bytes, 1);
		launch_cudaMemcpyAsync(Param->szz + offset, Param->d_szz, bytes, 1);
	}
}

void cuda_upload_wavefield(int iOutput) {
    int bytes  = Param->xyzNumKernel * sizeof(float);
    int offset = iOutput * Param->xyzNumKernel;

    launch_cudaMemcpy(Param->d_dsxx, Param->sxx + offset, bytes, 0);
    launch_cudaMemcpy(Param->d_dsxy, Param->sxy + offset, bytes, 0);
    launch_cudaMemcpy(Param->d_dsxz, Param->sxz + offset, bytes, 0);
    launch_cudaMemcpy(Param->d_dsyy, Param->syy + offset, bytes, 0);
    launch_cudaMemcpy(Param->d_dsyz, Param->syz + offset, bytes, 0);
    launch_cudaMemcpy(Param->d_dszz, Param->szz + offset, bytes, 0);
}
