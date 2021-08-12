#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <math.h>

#include "sac.h"
#include "sacio.h"
#include "xdfwi.h"

extern Param_t *Param;

static void taper_signal(float *data, float dis) {
    int ibeg = (int)(dis / 3200.f / Param->theDeltaT);
    int iend = (int)((dis / 2800.f + 35.f) / Param->theDeltaT);

    int width = 10;

    //ibeg -= width;
    //iend += width;

    if (ibeg < 0) {
        ibeg = 0;
    }

    if (iend >= Param->theTotalTimeSteps) {
        iend = Param->theTotalTimeSteps - 1;
    }

    int r;
    for (int i = 0; i < Param->theTotalTimeSteps; i++) {
        float scale = 0;

        if (i >= ibeg && i <= iend) {
            if (i - ibeg < width) {
                r = i - ibeg;
                scale = 0.5 * (1 + cos(r * 1.f / width * PI - PI));
            } else if (iend - i < width) {
                r = iend - i;
                scale = 0.5 * (1 + cos(r * 1.f / width * PI));
            } else {
                scale = 1.f;
            }
        }

        data[i] *= scale;
    }
}

static void bandpass_filter_signal(float *data) {
    /* Local variables */
    double low, high, attenuation, transition_bandwidth;

    int order, passes;

    /*     Define the Maximum size of the data Array 
     * Filter Prototypes 
     * Filter Types 
     *     Define the Data Array of size MAX 
     *     Declare Variables used in the rsac1() subroutine 
     *     Define variables used in the filtering routine 
     */
    low    = 0.023;
    high   = 0.043;
    passes = 2;
    order  = 2;
    transition_bandwidth = 0.0;
    attenuation = 0.0;

    /*     Call xapiir ( Apply a IIR Filter ) 
     *        - yarray - Original Data 
     *        - nlen   - Number of points in yarray 
     *        - proto  - Prototype of Filter 
     *                 - SAC_FILTER_BUTTERWORK        - Butterworth 
     *                 - SAC_FILTER_BESSEL            - Bessel 
     *                 - SAC_FILTER_CHEBYSHEV_TYPE_I  - Chebyshev Type I 
     *                 - SAC_FILTER_CHEBYSHEV_TYPE_II - Chebyshev Type II 
     *        - transition_bandwidth (Only for Chebyshev Filter) 
     *                 - Bandwidth as a fraction of the lowpass prototype 
     *                   cutoff frequency 
     *        - attenuation (Only for Chebyshev Filter) 
     *                 - Attenuation factor, equals amplitude reached at 
     *                   stopband egde 
     *        - order  - Number of poles or order of the analog prototype 
     *                   4 - 5 should be ample 
     *                   Cannot exceed 10 
     *        - type   - Type of Filter 
     *                 - SAC_FILTER_BANDPASS 
     *                 - SAC_FILTER_BANDREJECT 
     *                 - SAC_FILTER_LOWPASS 
     *                 - SAC_FILTER_HIGHPASS 
     *        - low    - Low Frequency Cutoff [ Hertz ] 
     *                   Ignored on SAC_FILTER_LOWPASS 
     *        - high   - High Frequency Cutoff [ Hertz ] 
     *                   Ignored on SAC_FILTER_HIGHPASS 
     *        - delta  - Sampling Interval [ seconds ] 
     *        - passes - Number of passes 
     *                 - 1 Forward filter only 
     *                 - 2 Forward and reverse (i.e. zero-phase) filtering 
     */
    xapiir(data, Param->theTotalTimeSteps, SAC_BUTTERWORTH, transition_bandwidth, attenuation, order, SAC_BANDPASS, low, high, Param->theDeltaT, passes);
}

static float max_cross_correlation(float *sa, float *sb, int N, float *maxValue) {
    // sa: reference, sb: data
    int maxLag;
    float maxLagInterp;
    fftw_complex *sa_t, *sa_f;
    fftw_complex *sb_t, *sb_f;
    fftw_complex *xcorr_t, *xcorr_f;

    sa_t = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    sa_f = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    sb_t = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    sb_f = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    xcorr_t = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    xcorr_f = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));

    fftw_plan p1 = fftw_plan_dft_1d(2 * N - 1, sa_t, sa_f, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_1d(2 * N - 1, sb_t, sb_f, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan re = fftw_plan_dft_1d(2 * N - 1, xcorr_f, xcorr_t, FFTW_BACKWARD, FFTW_ESTIMATE);

    //load the data to complex array
    for (int i = 0; i < N - 1; i++) {
        sa_t[i] = 0;
    }

    for (int i = N - 1; i < 2 * N - 1; i++) {
        sa_t[i] = sa[i - N + 1];
    }

    for (int i = 0; i < N; i++) {
        sb_t[i] = sb[i];
    }

    for (int i = N; i < 2 * N - 1; i++) {
        sb_t[i] = 0;
    }


    fftw_execute(p1);
    fftw_execute(p2);

    float norm2a = 0.f, norm2b = 0.f;

    for (int i = 0; i < N; i++) {
        norm2a += sa[i] * sa[i];
        norm2b += sb[i] * sb[i];
    }

    fftw_complex scale = 1.0 / (2 * N - 1) / sqrt(norm2a) / sqrt(norm2b);
    for (int i = 0; i < 2 * N - 1; i++) {
        xcorr_f[i] = sa_f[i] * conj(sb_f[i]) * scale;
    }

    fftw_execute(re);

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(re);

    fftw_free(sa_t);
    fftw_free(sa_f);
    fftw_free(sb_t);
    fftw_free(sb_f);
    fftw_free(xcorr_t);
    fftw_free(xcorr_f);

    *maxValue = -100000.f;
    maxLag = 0;

    for (int i = 0; i < 2 * N - 1; i++) {
        if (creal(xcorr_t[i]) > *maxValue) {
            *maxValue = creal(xcorr_t[i]);
            maxLag = i;
        }
    }

    //interpolation to find better time lag
    maxLagInterp = N - 1 - maxLag - (creal(xcorr_t[maxLag + 1]) - creal(xcorr_t[maxLag - 1])) / 2. / (2. * creal(xcorr_t[maxLag]) - creal(xcorr_t[maxLag + 1]) - creal(xcorr_t[maxLag - 1]));

    return maxLagInterp;
}

//back propogate the data
void compute_residual() {
	int iStation, iTime;
	float tmp, scale, maxCorr, dataNorm;
    char dataFileName[MAX_FILENAME_LEN];
    FILE *dataFp;
    int dataNumberOfStations, dataTotalTimeSteps;
    float dataDeltaT;
    float *data;

    sprintf(dataFileName, "../data/cb0412/station.%d", Param->theSrcID);
    dataFp = fopen(dataFileName, "r");
    if (dataFp == NULL) {
        xd_abort(__func__, "fopen() failed", "Cannot open data files!\n");
    }

    fscanf(dataFp, "%d %d %f", &dataNumberOfStations, &dataTotalTimeSteps, &dataDeltaT);

    for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
        fscanf(dataFp, "%f %f", &tmp, &tmp);
    }

    data = (float *)malloc(sizeof(float) * dataNumberOfStations * dataTotalTimeSteps);
    if (data == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for data\n");
    }
    for (iStation = 0; iStation < dataNumberOfStations; iStation++) {
        for (iTime = 0; iTime < dataTotalTimeSteps; iTime++) {
            fscanf(dataFp, "%f", data + iStation * dataTotalTimeSteps + iTime);
            //data[iStation * dataTotalTimeSteps + iTime] *= -1;
        }
    }
    fclose(dataFp);

    float thisMisfit = 0.f;
    for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
        //filter synthetic and real data
        //taper_signal(data + iStation * Param->theTotalTimeSteps, Param->myStations[iStation].dis);
        //taper_signal(Param->myStations[iStation].vz, Param->myStations[iStation].dis);
        bandpass_filter_signal(data + iStation * Param->theTotalTimeSteps);
        bandpass_filter_signal(Param->myStations[iStation].vz);

        scale = max_cross_correlation(data + iStation * Param->theTotalTimeSteps, Param->myStations[iStation].vz, Param->theTotalTimeSteps, &maxCorr);
        scale *= Param->theDeltaT;

        for (iTime = 0; iTime < Param->theTotalTimeSteps / 2; iTime++) {
            tmp = Param->myStations[iStation].vz[iTime];
            Param->myStations[iStation].vz[iTime] = Param->myStations[iStation].vz[Param->theTotalTimeSteps - iTime - 1];
            Param->myStations[iStation].vz[Param->theTotalTimeSteps - iTime - 1] = tmp;
        }

        dataNorm = 0.f;
        for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
            dataNorm += Param->myStations[iStation].vz[iTime] * Param->myStations[iStation].vz[iTime];
        }

        Param->allMisfit[Param->currentNumberOfStation] = scale;
        Param->totalMisfit += scale * scale / 2;
	    thisMisfit += scale * scale / 2;
        for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
            Param->myStations[iStation].vz[iTime] *= scale / dataNorm;
        }

        Param->currentNumberOfStation++;
    }

    free(data);
}

void upload_residual() {
	int iTime, iStation, iStage;
    int bytes = sizeof(float) * Param->theTotalTimeSteps * Param->myNumberOfStations * Param->nRKStage;
    float value1, value2;

    float *station = (float *)malloc(bytes);
    if (station == NULL) {
        xd_abort(__func__, "malloc() failed", "Memory allocation failed for station\n");
    }

    for (iTime = 0; iTime < Param->theTotalTimeSteps; iTime++) {
        for (iStage = 0; iStage < Param->nRKStage; iStage++) {
            for (iStation = 0; iStation < Param->myNumberOfStations; iStation++) {
                value1 = Param->myStations[iStation].vz[iTime];
                if (iTime == Param->theTotalTimeSteps - 1) {
                    value2 = Param->myStations[iStation].vz[iTime];
                } else {
                    value2 = Param->myStations[iStation].vz[iTime + 1];
                }

                station[(iTime * Param->nRKStage + iStage) * Param->myNumberOfStations + iStation] = (value1 * (1.f - Param->C[iStage]) + value2 * Param->C[iStage]) * 1e5;
            }
        }
    }

    launch_cudaFree(Param->d_station_vz);

    if (launch_cudaMalloc((void**)&Param->d_station_vz, bytes) !=0) {
        xd_abort(__func__, "launch_cudaMalloc() failed", "Memory allocation failed for the stations\n");
    }

    if (launch_cudaMemcpy(Param->d_station_vz, station, bytes, 0) != 0) {
        xd_abort(__func__, "launch_cudaMemcpy() failed", "Memory copy from host to device failed for the station\n");
    }

    free(station);
}
