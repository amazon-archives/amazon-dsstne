
SHELL=/bin/sh
VPATH=

# Platform-specific info should be found in ../config.h

include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/amazon/dsstne/utils

OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)
# objects for the shared lib (e.g. the cpp files that have a header)
OBJS := $(addprefix $(OBJS_BUILD_DIR)/, $(addsuffix .o, $(basename $(HEADERS))))

LIB_BUILD_DIR := $(BUILD_DIR)/lib

LIB_DSSTNE :=$(BUILD_DIR)/lib/libdsstne.a

BIN_BUILD_DIR := $(BUILD_DIR)/bin
$(shell mkdir -p $(BIN_BUILD_DIR))

INCLUDES = \
    $(CU_INCLUDES) \
    -I../engine

LIBS = \
    $(CU_LIBS) \

LOAD_LIBS = $(CU_LOADLIBS)

LDFLAGS ?= -fPIC -Wl,--whole-archive $(LIB_DSSTNE) -Wl,--no-whole-archive

EXECUTABLES := \
	$(BIN_BUILD_DIR)/generateNetCDF \
	$(BIN_BUILD_DIR)/train \
	$(BIN_BUILD_DIR)/predict \
	$(BIN_BUILD_DIR)/encoder

all: $(EXECUTABLES) $(LIB_BUILD_DIR)/libdsstne_utils.so

$(LIB_DSSTNE):
	cd ../engine && make

$(LIB_BUILD_DIR)/libdsstne_utils.so: $(OBJS)
	$(info ========== Creating libdsstne_utils.so ==========)
	mkdir -p $(BUILD_DIR)/lib
	$(CC) -shared $(LDFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)
	$(info ========== Copying amazon/dsstne/utils headers ==========)
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

$(BIN_BUILD_DIR)/generateNetCDF: $(OBJS) $(LIB_DSSTNE) $(OBJS_BUILD_DIR)/NetCDFGenerator.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/train: $(OBJS) $(LIB_DSSTNE) $(OBJS_BUILD_DIR)/Train.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/encoder: $(OBJS) $(LIB_DSSTNE) $(OBJS_BUILD_DIR)/main.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/predict: $(OBJS) $(LIB_DSSTNE) $(OBJS_BUILD_DIR)/Predict.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(BIN_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libdsstne_utils.so

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)

# Pattern rule for compiling C++
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# FIXME TestGPU belongs with tst, move this into the Makefile of tst
TST_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/tst

$(BIN_BUILD_DIR)/TestGPU: $(OBJS) $(LIB_DSSTNE) $(TST_OBJS_BUILD_DIR)/TestDune.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $(BIN_BUILD_DIR)/$@ $(LOAD_LIBS) -l:libcppunit.a

$(TST_OBJS_BUILD_DIR)/TestDune.o: ../../../../tst/gputests/TestDune.cpp
		mkdir -p $(TST_OBJS_BUILD_DIR)
		$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
# FIXME END