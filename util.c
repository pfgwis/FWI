#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <mpi.h>
#include <math.h>

#include "xdfwi.h"

#define LINESIZE 512

extern Param_t* Param;

int xd_abort(const char* fname, const char* perror_msg, const char* format, ...) {
    int my_errno = errno;

    fprintf(stderr, "\nPE %d called solver_abort() ", Param->myID);

    if (fname != NULL) {
        fprintf(stderr, "from %s: \n", fname);
    }

    if (format != NULL) {
        va_list ap;

        va_start(ap, format);
        vfprintf(stderr, format, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    if (perror_msg != NULL) {
        fprintf(stderr, "%s: %s\n", perror_msg, strerror(my_errno));
    }

    fprintf(stderr, "\n");

    MPI_Abort(MPI_COMM_WORLD, XD_ERROR);
    exit(1);

    return -1;
}

/**
 * Parse a text file and return the value of a match string.
 */
int parsetext(FILE* fp, const char* querystring, const char type, void* result) {
    const static char delimiters[] = " =\n\t";

    int32_t res = 0, found = 0;

    // start from the beginning
    rewind(fp);

    //look for the string until found
    while (!found) {
        char line[LINESIZE];
        char *name, *value;

        /* Read in one line */
        if (fgets(line, LINESIZE, fp) == NULL)
            break;

        name = strtok(line, delimiters);
        if ((name != NULL) && (strcmp(name, querystring) == 0)) {
            found = 1;
            value = strtok(NULL, delimiters);

            switch (type) {
                case 'i':
                    res = sscanf(value, "%d", (int *)result);
                    break;

                case 'f':
                    res = sscanf(value, "%f", (float *)result);
                    break;

                case 'd':
                    res = sscanf(value, "%lf", (double *)result);
                    break;

                case 's':
                    res = 1;
                    strcpy((char *)result, value);
                    break;

                case 'u':
                    res = sscanf(value, "%u", (uint32_t *)result);
                    break;

                default:
                    fprintf(stderr, "parsetext: unknown type %c\n", type);
                    return -1;
            }
        }
    }

    return (res == 1) ? 0 : -1;
}

void print_time_status(double *myTime, const char* timeName) {
    double myTimeMin, myTimeMax, myTimeAve;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(myTime, &myTimeMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(myTime, &myTimeMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(myTime, &myTimeAve, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (Param->myID == 0)
        fprintf(stdout, "%s: min: %.1fs, max: %.1fs, average: %.1fs\n", timeName, myTimeMin, myTimeMax, myTimeAve/Param->theGroupSize);
}

void pml_map_init() {
    //the order: left-front(0), left(1), left-back(2), front(3), back(4), right-front(5), right(6), right-back(7)
    //           left-front-bottom(8), left-bottom(9), left-back-bottom(10), front-bottom(11), bottom(12)
    //           back-bottom(13), right-front-bottom(14), right-bottom(15), right-back-bottom(16)
    //there are 17 regions totally
    //map = [initPmlIndex initX, initY, initZ, xyNum, xNum]

    //left-front
    Param->pmlMap[0][0] = 0;
    Param->pmlMap[0][1] = 0;
    Param->pmlMap[0][2] = 0;
    Param->pmlMap[0][3] = 0;
    Param->pmlMap[0][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[0][5] = Param->pmlNum;

    //left
    Param->pmlMap[1][0] = Param->pmlMap[0][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[0][0];
    Param->pmlMap[1][1] = Param->pmlNum;
    Param->pmlMap[1][2] = 0;
    Param->pmlMap[1][3] = 0;
    Param->pmlMap[1][4] = (Param->xNum - 2 * Param->pmlNum) * Param->pmlNum;
    Param->pmlMap[1][5] = Param->xNum - 2 * Param->pmlNum;

    //left-back
    Param->pmlMap[2][0] = Param->pmlMap[1][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[1][0];
    Param->pmlMap[2][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[2][2] = 0;
    Param->pmlMap[2][3] = 0;
    Param->pmlMap[2][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[2][5] = Param->pmlNum;

    //front
    Param->pmlMap[3][0] = Param->pmlMap[2][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[2][0];
    Param->pmlMap[3][1] = 0;
    Param->pmlMap[3][2] = Param->pmlNum;
    Param->pmlMap[3][3] = 0;
    Param->pmlMap[3][4] = Param->pmlNum * (Param->yNum - 2 * Param->pmlNum);
    Param->pmlMap[3][5] = Param->pmlNum;

    //back
    Param->pmlMap[4][0] = Param->pmlMap[3][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[3][0];
    Param->pmlMap[4][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[4][2] = Param->pmlNum;
    Param->pmlMap[4][3] = 0;
    Param->pmlMap[4][4] = Param->pmlNum * (Param->yNum - 2 * Param->pmlNum);
    Param->pmlMap[4][5] = Param->pmlNum;

    //right-front
    Param->pmlMap[5][0] = Param->pmlMap[4][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[4][0];
    Param->pmlMap[5][1] = 0;
    Param->pmlMap[5][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[5][3] = 0;
    Param->pmlMap[5][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[5][5] = Param->pmlNum;

    //right
    Param->pmlMap[6][0] = Param->pmlMap[5][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[5][0];
    Param->pmlMap[6][1] = Param->pmlNum;
    Param->pmlMap[6][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[6][3] = 0;
    Param->pmlMap[6][4] = (Param->xNum - 2 * Param->pmlNum) * Param->pmlNum;
    Param->pmlMap[6][5] = Param->xNum - 2 * Param->pmlNum;

    //right-back
    Param->pmlMap[7][0] = Param->pmlMap[6][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[6][0];
    Param->pmlMap[7][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[7][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[7][3] = 0;
    Param->pmlMap[7][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[7][5] = Param->pmlNum;

    //left-front-bottom
    Param->pmlMap[8][0] = Param->pmlMap[7][4] * (Param->zNum - Param->pmlNum) + Param->pmlMap[7][0];
    Param->pmlMap[8][1] = 0;
    Param->pmlMap[8][2] = 0;
    Param->pmlMap[8][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[8][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[8][5] = Param->pmlNum;

    //left-bottom
    Param->pmlMap[9][0] = Param->pmlMap[8][4] * Param->pmlNum + Param->pmlMap[8][0];
    Param->pmlMap[9][1] = Param->pmlNum;
    Param->pmlMap[9][2] = 0;
    Param->pmlMap[9][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[9][4] = (Param->xNum - 2 * Param->pmlNum) * Param->pmlNum;
    Param->pmlMap[9][5] = Param->xNum - 2 * Param->pmlNum;

    //left-back-bottom
    Param->pmlMap[10][0] = Param->pmlMap[9][4] * Param->pmlNum + Param->pmlMap[9][0];
    Param->pmlMap[10][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[10][2] = 0;
    Param->pmlMap[10][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[10][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[10][5] = Param->pmlNum;

    //front_bottom
    Param->pmlMap[11][0] = Param->pmlMap[10][4] * Param->pmlNum + Param->pmlMap[10][0];
    Param->pmlMap[11][1] = 0;
    Param->pmlMap[11][2] = Param->pmlNum;
    Param->pmlMap[11][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[11][4] = Param->pmlNum * (Param->yNum - 2 * Param->pmlNum);
    Param->pmlMap[11][5] = Param->pmlNum;

    //bottom
    Param->pmlMap[12][0] = Param->pmlMap[11][4] * Param->pmlNum + Param->pmlMap[11][0];
    Param->pmlMap[12][1] = Param->pmlNum;
    Param->pmlMap[12][2] = Param->pmlNum;
    Param->pmlMap[12][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[12][4] = (Param->xNum - 2 * Param->pmlNum) * (Param->yNum - 2 * Param->pmlNum);
    Param->pmlMap[12][5] = Param->xNum - 2 * Param->pmlNum;

    //back_bottom
    Param->pmlMap[13][0] = Param->pmlMap[12][4] * Param->pmlNum + Param->pmlMap[12][0];
    Param->pmlMap[13][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[13][2] = Param->pmlNum;
    Param->pmlMap[13][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[13][4] = Param->pmlNum * (Param->yNum - 2 * Param->pmlNum);
    Param->pmlMap[13][5] = Param->pmlNum;

    //front-right-bottom
    Param->pmlMap[14][0] = Param->pmlMap[13][4] * Param->pmlNum + Param->pmlMap[13][0];
    Param->pmlMap[14][1] = 0;
    Param->pmlMap[14][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[14][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[14][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[14][5] = Param->pmlNum;

    //right-bottom
    Param->pmlMap[15][0] = Param->pmlMap[14][4] * Param->pmlNum + Param->pmlMap[14][0];
    Param->pmlMap[15][1] = Param->pmlNum;
    Param->pmlMap[15][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[15][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[15][4] = (Param->xNum - 2 * Param->pmlNum) * Param->pmlNum;
    Param->pmlMap[15][5] = Param->xNum - 2 * Param->pmlNum;

    //right-back-bottom
    Param->pmlMap[16][0] = Param->pmlMap[15][4] * Param->pmlNum + Param->pmlMap[15][0];
    Param->pmlMap[16][1] = Param->xNum - Param->pmlNum;
    Param->pmlMap[16][2] = Param->yNum - Param->pmlNum;
    Param->pmlMap[16][3] = Param->zNum - Param->pmlNum;
    Param->pmlMap[16][4] = Param->pmlNum * Param->pmlNum;
    Param->pmlMap[16][5] = Param->pmlNum;

    Param->pmlNumTotal = Param->pmlMap[16][4] * Param->pmlNum + Param->pmlMap[16][0];
}

int get_pml_region_iXYZ(int iX, int iY, int iZ) {
    if (iZ < Param->zNum - Param->pmlNum) {
        if (iY < Param->pmlNum) {
            if (iX < Param->pmlNum) {
                //left-front
                return 0;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //left-back
                return 2;
            } else {
                //left
                return 1;
            }
        } else if (iY >= Param->yNum - Param->pmlNum) {
            if (iX < Param->pmlNum) {
                //right-front
                return 5;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //right-back
                return 7;
            } else {
                //right
                return 6;
            }
        } else {
            if (iX < Param->pmlNum) {
                //front
                return 3;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //back
                return 4;
            } else {
                //interior
                return -1;
            }
        }
    } else {
        if (iY < Param->pmlNum) {
            if (iX < Param->pmlNum) {
                //left-front-bottom
                return 8;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //left-back-bottom
                return 10;
            } else {
                //left-bottom
                return 9;
            }
        } else if (iY >= Param->yNum - Param->pmlNum) {
            if (iX < Param->pmlNum) {
                //right-front-bottom
                return 14;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //right-back-bottom
                return 16;
            } else {
                //right-bottom
                return 15;
            }
        } else {
            if (iX < Param->pmlNum) {
                //front-bottom
                return 11;
            } else if (iX >= Param->xNum - Param->pmlNum) {
                //back-bottom
                return 13;
            } else {
                //bottom
                return 12;
            }
        }
    }
}

int get_pml_region_iPml(int iPml) {
    int iRegion = 0;

    while (iPml >= Param->pmlMap[iRegion + 1][0]) {
        iRegion++;

        if (iRegion == 16) {
            break;
        }
    }

    return iRegion;
}

int iXYZ_to_iPml(int iX, int iY, int iZ, int iRegion) {
    //return the index of the pml
    int iPml;

    iPml  = Param->pmlMap[iRegion][0];
    iPml += (iZ - Param->pmlMap[iRegion][3]) * Param->pmlMap[iRegion][4];
    iPml += (iY - Param->pmlMap[iRegion][2]) * Param->pmlMap[iRegion][5];
    iPml += iX - Param->pmlMap[iRegion][1];

    return iPml;
}

int iPml_to_iXYZ(int iPml) {
    //convert iPml index to iXYZ index
    int iX, iY, iZ, iRegion;

    iRegion = get_pml_region_iPml(iPml);

    iPml -= Param->pmlMap[iRegion][0];
    iZ = iPml / Param->pmlMap[iRegion][4] + Param->pmlMap[iRegion][3];
    iPml %= Param->pmlMap[iRegion][4];
    iY = iPml / Param->pmlMap[iRegion][5] + Param->pmlMap[iRegion][2];
    iPml %= Param->pmlMap[iRegion][5];
    iX = iPml + Param->pmlMap[iRegion][1];

    return iZ * Param->xyNum + iY * Param->xNum + iX;
}

static float sinc(float x) {
    if (fabs(x) < 1e-8) {
        return 1.f;
    } else {
        return sin(PI * x) / PI / x;
    }
}

static float bessel0(float x) {
    float ax, ans, y;

    ax = fabs(x);

   if (ax < 3.75) {
        y = x / 3.75;
        y = y * y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
   } else {
        y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
   }

   return ans;
}

static float kaiser_window(float x) {
    return bessel0(KAISER_B * sqrt(1.f - (x / KAISER_LEN) * (x / KAISER_LEN))) / bessel0(KAISER_B);
}

float kaiser_sinc(float x) {
    return sinc(x) * kaiser_window(x);
}