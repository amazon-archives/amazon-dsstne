
SHELL=/bin/sh
VPATH=

# Platform-specific info should be found in ../config.h

include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_PREFIX := $(PREFIX)/include/amazon/dsstne/utils

OBJS_PREFIX := $(PREFIX)/tmp/utils/cpp
CU_OBJS_PREFIX := $(PREFIX)/tmp/utils/cuda
$(shell mkdir -p $(OBJS_PREFIX))
$(shell mkdir -p $(CU_OBJS_PREFIX))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_PREFIX)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_PREFIX)/%.o)
# objects for the shared lib (e.g. the cpp files that have a header)
OBJS := $(addprefix $(OBJS_PREFIX)/, $(addsuffix .o, $(basename $(HEADERS))))

LIB_PREFIX := $(PREFIX)/lib

LIB_DSSTNE :=$(PREFIX)/lib/libdsstne.a

BIN_PREFIX := $(PREFIX)/bin
$(shell mkdir -p $(BIN_PREFIX))

INCLUDES = $(CU_INCLUDES)
LIBS = $(MATH_LIBS) $(MPI_LIBS) $(CU_LIBS) 
LOAD_LIBS = $(CU_LOADLIBS)

LDFLAGS ?= -fPIC -Wl,--whole-archive $(LIB_DSSTNE) -Wl,--no-whole-archive

EXECUTABLES := \
	$(BIN_PREFIX)/generateNetCDF \
	$(BIN_PREFIX)/train \
	$(BIN_PREFIX)/predict \
	$(BIN_PREFIX)/encoder 

all: $(EXECUTABLES) $(LIB_PREFIX)/libdsstne_utils.so

install: all 

$(LIB_DSSTNE):
	cd ../engine && make
	
$(LIB_PREFIX)/libdsstne_utils.so: $(OBJS)
	$(info ========== Creating libdsstne_utils.so ==========)
	mkdir -p $(PREFIX)/lib
	$(CC) -shared $(LDFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)
	$(info ========== Copying amazon/dsstne/utils headers ==========)
	mkdir -p $(HEADERS_PREFIX)
	cp $(HEADERS) $(HEADERS_PREFIX)

$(BIN_PREFIX)/generateNetCDF: $(OBJS) $(LIB_DSSTNE) $(OBJS_PREFIX)/NetCDFGenerator.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_PREFIX)/train: $(OBJS) $(LIB_DSSTNE) $(OBJS_PREFIX)/Train.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_PREFIX)/encoder: $(OBJS) $(LIB_DSSTNE) $(OBJS_PREFIX)/main.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_PREFIX)/predict: $(OBJS) $(LIB_DSSTNE) $(OBJS_PREFIX)/Predict.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_PREFIX) $(CU_OBJS_PREFIX) $(BIN_PREFIX) $(HEADERS_PREFIX) $(LIB_PREFIX)/libdsstne_utils.so

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_PREFIX) $(CU_OBJS_PREFIX)

# Pattern rule for compiling C++
$(OBJS_PREFIX)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Pattern rule for compiling CUDA
$(CU_OBJS_PREFIX)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h
	$(NVCC) $(CU_FLAGS) $(INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

# FIXME TestGPU belongs with tst, move this into the Makefile of tst
TST_OBJS_PREFIX := $(PREFIX)/tmp/utils/tst

$(BIN_PREFIX)/TestGPU: $(OBJS) $(LIB_DSSTNE) $(TST_OBJS_PREFIX)/TestDune.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $(BIN_PREFIX)/$@ $(LOAD_LIBS) -l:libcppunit.a

$(TST_OBJS_PREFIX)/TestDune.o: ../../../../tst/gputests/TestDune.cpp 
		mkdir -p $(TST_OBJS_PREFIX)
		$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
# FIXME END