

SHELL=/bin/sh
VPATH=




all: lib/libdsstne.a  bin/train bin/predict bin/generateNetCDF

install: all 
lib/libdsstne.a:
	cd engine && make


bin/train:
	cd utils && make

bin/predict:
	cd utils && make

bin/generateNetCDF:
	cd utils && make


clean:
	cd engine && make clean
	cd utils && make clean


