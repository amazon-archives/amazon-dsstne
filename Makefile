SHELL=/bin/sh
VPATH=

# Prefix of the directory to install dsstne.
PREFIX ?= $(shell pwd)/amazon-dsstne

# Build directory. Export it for sub-makefiles to use it
export BUILD_DIR ?= $(shell pwd)/build

all:
	cd src/amazon/dsstne/engine && make
	cd src/amazon/dsstne/utils && make
	cd tst && make
	cd java && make

install: all
	mkdir -p $(PREFIX)
	cp -rfp $(BUILD_DIR)/lib $(PREFIX)/lib
	cp -rfp $(BUILD_DIR)/bin $(PREFIX)/bin
	cp -rfp $(BUILD_DIR)/include $(PREFIX)/include

run-tests:
	cd tst && make run-tests

clean:
	cd src/amazon/dsstne/engine && make clean
	cd src/amazon/dsstne/utils && make clean
	cd tst && make clean

