

SHELL=/bin/sh
VPATH=


include ../Makefile.inc

OBJS=   NNTypes.o NNWeight.o NNLayer.o NNNetwork.o GpuTypes.o kernels.o kLoss.o kActivation.o kDelta.o  

COMMON_LIBS = $(MATH_LIBS) $(MPI_LIBS) $(CU_LIBS) $(CU_LOADLIBS)
all: ../lib/libdsstne.a

install: all 

../lib/libdsstne.a: $(OBJS)
	mkdir -p ../lib
	ar -cvr $@ $^

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc* ../lib/libdsstne.a

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*

# The following statement keeps the .f90 files from getting clobbered; this
# would happen because they are intermediates.

# Control the suffixes available; this essentially eliminates built-in
# inference rules, done here to improve portability.

.SUFFIXES:

.SUFFIXES: .cpp .cu .o

.cpp.o:
	$(CC) $(CFLAGS) $(CU_INCLUDES) -c $*.cpp

.cu.o:	GpuTypes.h
	$(NVCC) $(CU_FLAGS) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $*.cu

## CUDA dependencies

