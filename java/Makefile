MODULE_ROOT = $(shell pwd)

BUILD_DIR ?= ../build

DSSTNE_SRC_DIR = ../src/amazon/dsstne

SOURCES_DIR = src/main/native
TEST_DIR = src/test/native
OBJECTS_DIR = target/native/build
LIBRARY_DIR = target/lib
BIN_DIR = target/bin
LIBRARY = $(LIBRARY_DIR)/libdsstne_java.so
MAIN = main.o

INC = \
	/usr/lib/jvm/java-1.8.0-openjdk-amd64/include \
	/usr/lib/jvm/java-8-openjdk-amd64/include/linux \
	/usr/local/cuda/include \
	/usr/include/jsoncpp \
	/usr/lib/openmpi/include \
	/usr/lib/openmpi/include/openmpi \
	../src  # dsstne headers

INCLUDES ?= $(INC:%=-I%)

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
	dsstne_engine \
	dsstne_runtime

LOAD_LIBS = $(LLIB:%=-l%)

SOURCES=$(shell find '$(SOURCES_DIR)' -type f -name '*.cpp')
OBJECTS=$(SOURCES:$(SOURCES_DIR)/%.cpp=$(OBJECTS_DIR)/%.o)

CC = g++
CFLAGS ?=-Wall -fPIC -std=c++11 -pthread -O3
LDFLAGS ?= -fPIC

all: $(LIBRARY)
	mkdir -p $(BUILD_DIR)/lib
	cp -fp $(LIBRARY) $(BUILD_DIR)/lib

main:
	$(info ========== Building main =============)
	mkdir -p $(BIN_DIR)
	$(CC) -g $(CFLAGS) -Wl,-rpath=/usr/local/cuda/lib64 $(INCLUDES) $(LIBS) src/test/native/main.cpp -o $(BIN_DIR)/main.o $(LOAD_LIBS)

$(LIBRARY): $(OBJECTS) $(LIB_DSSTNE)
	$(info ==========  Building JNI Libs ==========)
	mkdir -p $(LIBRARY_DIR)
	$(CC) -shared $(LDFLAGS) $(LIBS) $(OBJECTS) -o $@ $(LOAD_LIBS)

$(OBJECTS_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	mkdir -p $(OBJECTS_DIR)
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -rf $(OBJECTS_DIR) $(LIBRARY) $(BIN_DIR)
