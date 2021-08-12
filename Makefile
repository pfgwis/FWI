MPI_DIR      = /public/apps/openmpi/3.1.1
MPI_INCLUDE  = $(MPI_DIR)/include/
CC           = $(MPI_DIR)/bin/mpicc
CFLAGS       = -Wall -O3 -I/public/apps/fftw/3.3.8/include
NVCC         = nvcc -O3
LD           = $(MPI_DIR)/bin/mpic++
LDFLAGS      = -lm -lsac -lsacio -L/cm/shared/apps/cuda92/toolkit/9.2.88/lib64 -L/public/apps/fftw/3.3.8/lib -lfftw3 -L$(MPI_DIR)/lib -lcudart -L/public/apps/sac/101.6a/lib/

#
# Object modules 
#
OBJECTS = xdfwi.o xdfwiRun.o parameter.o velocityModel.o util.o wavefield.o modeling.o modelingDomain.o source.o station.o cudaPrepare.o cpuPrepare.o modelingSub.o kernel.o residual.o cudaKernel.o cudaKernelLaunch.o

TARGET = xdfwi

.PHONY: all tags clean cleanall

all: $(TARGET)

xdfwi: ${OBJECTS}
		$(LD) $^ \
		$(LDFLAGS) \
		-o $@

cudaKernel.o: cudaKernel.cu
	$(NVCC) -c cudaKernel.cu

cudaKernelLaunch.o: cudaKernelLaunch.cu
	$(NVCC) -c cudaKernelLaunch.cu

clean:
		rm -f $(OBJECTS) *.o

cleanall:
		rm -f $(OBJECTS) $(TARGET) *.o

TAGS:
		etags *.c *.h

tags:   TAGS
