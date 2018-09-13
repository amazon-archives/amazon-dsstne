
SHELL=/bin/sh
VPATH=

include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_PREFIX := $(PREFIX)/include/amazon/dsstne/engine

OBJS_PREFIX := $(PREFIX)/tmp/engine/cpp
CU_OBJS_PREFIX := $(PREFIX)/tmp/engine/cuda
$(shell mkdir -p $(OBJS_PREFIX))
$(shell mkdir -p $(CU_OBJS_PREFIX))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_PREFIX)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_PREFIX)/%.o)

LIB_PREFIX := $(PREFIX)/lib

all: $(LIB_PREFIX)/libdsstne.a

install: all

$(LIB_PREFIX)/libdsstne.a: $(OBJECTS) $(CU_OBJECTS)
	$(info ========== Creating libdsstne.a ==========)
	mkdir -p $(LIB_PREFIX)
	ar -cvr $@ $^

	$(info ========== Copying amazon/dsstne/engine headers ==========)
	mkdir -p $(HEADERS_PREFIX)
	cp $(HEADERS) $(HEADERS_PREFIX)
clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc* 
	rm -rf $(OBJS_PREFIX) $(CU_OBJS_PREFIX) $(HEADERS_PREFIX) $(LIB_PREFIX)/libdsstne.a

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_PREFIX) $(CU_OBJS_PREFIX)

# The following statement keeps the .f90 files from getting clobbered; this
# would happen because they are intermediates.

# Pattern rule for compiling C++
$(OBJS_PREFIX)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CU_INCLUDES) -c $< -o $@

# Pattern rule for compiling CUDA
$(CU_OBJS_PREFIX)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h
	$(NVCC) $(CU_FLAGS) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

