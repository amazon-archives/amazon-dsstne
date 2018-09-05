MODULE_ROOT = $(shell pwd)

BUILD_DIR ?= $(shell pwd)/../build
BIN_DIR = $(BUILD_DIR)/tst/bin
TEST_SOURCES_DIR = amazon
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/tst

SOURCES=$(shell find '$(TEST_SOURCES_DIR)' -type f -name '*.cpp')
OBJECTS := $(SOURCES:$(TEST_SOURCES_DIR)/%.cpp=$(OBJS_BUILD_DIR)/%.o)

DEP := $(OBJECTS:.o=.d)

$(info ========== SOURCES = $(SOURCES) =============)
$(info ========== OBJECTS = $(OBJECTS) =============)

DSSTNE_SRC_DIR = ../src/amazon/dsstne
LIB_DSSTNE = $(BUILD_DIR)/lib/libdsstne.a
LIB_DSSTNE_UTILS = $(BUILD_DIR)/lib/libdsstne_utils.so
 
INCLUDES = \
	-isystem /usr/local/cuda/include \
	-isystem /usr/include/jsoncpp \
	-isystem /usr/lib/openmpi/include \
	-isystem /usr/lib/openmpi/include/openmpi \
	-I../src # dsstne headers

LIB = \
	$(BUILD_DIR)/lib \
	/usr/local/cuda/lib64 \
	/usr/lib/openmpi/lib

LIBS ?= $(LIB:%=-L%)

LLIB = \
	cudnn \
	curand \
	cublas \
	cudart \
	jsoncpp \
	netcdf \
	netcdf_c++4 \
	blas \
	dl \
	stdc++ \
	mpi_cxx \
	mpi \
	cppunit \
	dsstne_utils 

LOAD_LIBS = $(LLIB:%=-l%)

CC = g++
CFLAGS ?=-Wall -std=c++11 -pthread 
LDFLAGS ?= -Wl,-rpath=/usr/local/cuda/lib64

all: $(BIN_DIR)/unittests

$(LIB_DSSTNE):
	cd $(DSSTNE_SRC_DIR)/engine && make

$(LIB_DSSTNE_UTILS):
	cd $(DSSTNE_SRC_DIR)/utils && make

$(BIN_DIR)/unittests: $(LIB_DSSTNE) $(LIB_DSSTNE_UTILS) $(OBJECTS)
	$(info ========== Building unittests =============)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) unittests.cpp $^ $(LIB_DSSTNE) -o $@ $(LOAD_LIBS)

run-tests: $(BIN_DIR)/unittests 
	$(info ========== Running unittests =============)
	$(BIN_DIR)/unittests

clean:
	rm -rf $(BIN_DIR)	

# Pattern rule for compiling C++
$(OBJS_BUILD_DIR)/%.o: $(TEST_SOURCES_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

#-include	 $(DEP)	

