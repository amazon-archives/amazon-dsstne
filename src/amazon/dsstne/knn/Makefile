
SHELL=/bin/sh
VPATH=

include ../Makefile.inc

CC = g++

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/amazon/dsstne/knn

OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)

DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

CU_INCLUDES += -I../utils

LIB_BUILD_DIR := $(BUILD_DIR)/lib

all: $(LIB_BUILD_DIR)/libdsstne_knn.so

$(LIB_BUILD_DIR)/libdsstne_knn.so: $(OBJECTS) $(CU_OBJECTS)
	$(info ========== Creating libdsstne_knn.so ==========)
	mkdir -p $(BUILD_DIR)/lib
	$(CC) -shared $(LDFLAGS) $(CU_LIBS) $(OBJECTS) $(CU_OBJECTS) -o $@ $(CU_LOADLIBS)
	$(info ========== Copying amazon/dsstne/knn headers ==========)
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libdsstne_knn.a

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)

# The following statement keeps the .f90 files from getting clobbered; this
# would happen because they are intermediates.

# Pattern rule for compiling C++
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CU_INCLUDES) -c $< -o $@

# Pattern rule for compiling CUDA
$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu $(CU_OBJS_BUILD_DIR)/%.d
	$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

# Pattern rule for auto dependency generation of *.cu files
$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	$(NVCC) -E -std=c++11 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM" $(CU_INCLUDES)  $< -o $@

-include $(DEP)
-include $(CU_DEP)
