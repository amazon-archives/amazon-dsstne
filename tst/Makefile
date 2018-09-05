MODULE_ROOT = $(shell pwd)

PREFIX ?= ../src/amazon/dsstne/build
BIN_DIR = $(PREFIX)/tst/bin
TEST_SOURCES_DIR = amazon
SOURCES=$(shell find '$(TEST_SOURCES_DIR)' -type f -name '*.cpp')

DSSTNE_SRC_DIR = ../src/amazon/dsstne
LIB_DSSTNE = $(PREFIX)/lib/libdsstne.a
LIB_DSSTNE_UTILS = $(PREFIX)/lib/libdsstne_utils.so
 
INC = \
	/usr/local/cuda/include \
	/usr/include/jsoncpp \
	/usr/lib/openmpi/include \
	/usr/lib/openmpi/include/openmpi \
	../src # dsstne headers

INCLUDES ?= $(INC:%=-I%)

LIB = \
	$(PREFIX)/lib \
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
CFLAGS ?=-Wall -fPIC -std=c++11 -pthread -Wl,-rpath=/usr/local/cuda/lib64

all: $(BIN_DIR)/unittests

$(LIB_DSSTNE):
	cd $(DSSTNE_SRC_DIR)/engine && make

$(LIB_DSSTNE_UTILS):
	cd $(DSSTNE_SRC_DIR)/utils && make

$(BIN_DIR)/unittests: $(LIB_DSSTNE) $(LIB_DSSTNE_UTILS)
	$(info ========== Building unittests =============)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) unittests.cpp $(SOURCES) $(LIB_DSSTNE) -o $@ $(LOAD_LIBS)

run-tests: $(BIN_DIR)/unittests 
	$(info ========== Running unittests =============)
	$(BIN_DIR)/unittests

clean:
	rm -rf $(BIN_DIR)	

clean-deps:
	$(info ==========  Cleaning Dependencies ==========)
	cd $(DSSTNE_SRC_DIR) && make clean
