SHELL=/bin/sh
VPATH=

# Prefix of the directory to install dsstne.
PREFIX ?= $(shell pwd)/amazon-dsstne

# Build directory. Export it for sub-makefiles to use it
export BUILD_DIR ?= $(shell pwd)/build

all: | engine runtime tests java

engine:
	cd src/amazon/dsstne/engine && make

runtime:
	cd src/amazon/dsstne/runtime && make

tests:
	cd tst && make

java: | engine runtime tests
	cd java && make

install: all
	mkdir -p $(PREFIX)
	cp -rfp $(BUILD_DIR)/lib $(PREFIX)
	cp -rfp $(BUILD_DIR)/bin $(PREFIX)
	cp -rfp $(BUILD_DIR)/include $(PREFIX)

run-tests:
	cd tst && make run-tests

clean:
	cd src/amazon/dsstne/engine && make clean
	cd src/amazon/dsstne/utils && make clean
	cd tst && make clean

.PHONY: engine runtime tests java clean 
