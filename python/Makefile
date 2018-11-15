libdsstnecuda.a : dsstnecuda.c
	mpiCC -DOMPI_SKIP_MPICXX -std=c++0x -O3 -fPIC -I/usr/local/cuda/include -I/usr/include/jsoncpp -I../src/amazon/dsstne/engine -c dsstnecuda.c
	ar -cvr libdsstnecuda.a dsstnecuda.o
