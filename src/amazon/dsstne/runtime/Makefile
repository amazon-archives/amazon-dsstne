
SHELL=/bin/sh
VPATH=

include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/amazon/dsstne/runtime

OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/runtime/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/runtime/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)

LIB_DSSTNE :=$(BUILD_DIR)/lib/libdsstne.a

LDFLAGS ?= -fPIC
#LDFLAGS ?= -fPIC -Wl,--whole-archive $(LIB_DSSTNE) -Wl,--no-whole-archive

LIBS = \
    $(CU_LIBS) \
    -L$(BUILD_DIR)/lib

LOAD_LIBS = \
    $(CU_LOADLIBS) \
    -ldsstne_engine

DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

LIB_BUILD_DIR := $(BUILD_DIR)/lib

all: $(LIB_BUILD_DIR)/libdsstne_runtime.so

$(LIB_BUILD_DIR)/libdsstne_runtime.so: $(OBJECTS) $(CU_OBJECTS)
	$(info ========== Creating libdsstne_runtime.so ==========)
	mkdir -p $(BUILD_DIR)/lib
	$(CC) -shared $(LDFLAGS) $(LIBS) $(OBJECTS) $(CU_OBJECTS) -o $@ $(LOAD_LIBS)
	$(info ========== Copying amazon/dsstne/runtime headers ==========)
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

clean:
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libdsstne_runtime.so

distclean:
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)

# Pattern rule for compiling C++
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CU_INCLUDES) -c $< -o $@

# Pattern rule for compiling CUDA
$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h  $(CU_OBJS_BUILD_DIR)/%.d
	$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

# Pattern rule for auto dependency generation of *.cu files
$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	$(NVCC) -E -std=c++11 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM" $(CU_INCLUDES)  $< -o $@

-include $(DEP)
-include $(CU_DEP)
