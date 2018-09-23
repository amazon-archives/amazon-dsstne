#
# THIS MAKEFILE IS DEPRECATED PLEASE USE THE ONE AT THE ROOT OF THE DSSTNE REPOSITORY!
# Keeping this here for backwards compatibility
#

$(warning **************************************************************************************)
$(warning ******************          USE OF DEPRECATED MAKEFILE              ******************)
$(warning ****************** PLEASE USE THE ONE AT THE ROOT OF THE REPOSITORY ******************)
$(warning **************************************************************************************)

SHELL=/bin/sh
VPATH=

# Prefix of the directory to install dsstne. Override-able by setting PREFIX before calling make
PREFIX ?= $(dir $(shell pwd))../../amazon-dsstne

all: $(BUILD_DIR)/lib/libdsstne.a $(BUILD_DIR)/bin/train $(BUILD_DIR)/bin/predict $(BUILD_DIR)/bin/generateNetCDF
	cd engine && make
	cd utils && make

$(BUILD_DIR)/lib/libdsstne.a:
	cd engine && make

install: all
	mkdir -p $(PREFIX)
	cp -rfp $(BUILD_DIR)/lib $(PREFIX)/lib
	cp -rfp $(BUILD_DIR)/bin $(PREFIX)/bin
	cp -rfp $(BUILD_DIR)/include $(PREFIX)/include

$(BUILD_DIR)/bin/train:
	cd utils && make

$(BUILD_DIR)/bin/predict:
	cd utils && make

$(BUILD_DIR)/bin/generateNetCDF:
	cd utils && make

clean:
	cd engine && make clean
	cd utils && make clean


