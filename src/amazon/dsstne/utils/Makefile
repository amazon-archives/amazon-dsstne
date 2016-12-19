

SHELL=/bin/sh
VPATH=

# Platform-specific info should be found in ../config.h

include ../Makefile.inc

OBJS= Utils.o NetCDFhelper.o NNRecsGenerator.o Filters.o
LIB_DSSTNE=../lib/libdsstne.a

COMMON_LIBS = $(LIB_DSSTNE) $(MATH_LIBS) $(MPI_LIBS) $(CU_LIBS) $(CU_LOADLIBS)
COMMON_LIBS_UNIT_TEST = $(COMMON_LIBS) -l:libcppunit.a

all: generateNetCDF train predict encoder TestGPU

install: all 

../lib/libdsstne.a:
	cd ../engine && make


generateNetCDF: NetCDFGenerator.o NetCDFhelper.o Utils.o $(LIB_DSSTNE)
	mkdir -p ../bin
	$(LOAD) $(LOADFLAGS) -o $@  NetCDFGenerator.o NetCDFhelper.o Utils.o $(COMMON_LIBS)
	cp $@ ../bin/

train : $(OBJS) Train.o $(LIB_DSSTNE)
	mkdir -p ../bin
	$(LOAD) $(LOADFLAGS) -o $@ $(OBJS) Train.o $(COMMON_LIBS)
	cp $@ ../bin/

encoder : $(OBJS) main.o $(LIB_DSSTNE)
	mkdir -p ../bin
	$(LOAD) $(LOADFLAGS) -o $@ $(OBJS) main.o $(COMMON_LIBS)
	cp $@ ../bin/

predict : $(OBJS) Predict.o $(LIB_DSSTNE)
	mkdir -p ../bin
	$(LOAD) $(LOADFLAGS) -o $@ $(OBJS) Predict.o  $(COMMON_LIBS)
	cp $@ ../bin/

TestGPU : $(OBJS) ../../../../tst/gputests/TestDune.o
	$(LOAD) $(LOADFLAGS) -o $@ $(OBJS) TestDune.o  $(COMMON_LIBS_UNIT_TEST)


clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc* generateNetCDF train predict encoder TestGPU ../bin/generateNetCDF ../bin/train ../bin/predict ../bin/encoder

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

