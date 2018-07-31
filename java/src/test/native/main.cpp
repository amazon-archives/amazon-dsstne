/*
 * main.cpp
 *
 *  Created on: Aug 15, 2018
 *      Author: kiuk
 */
#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <tuple>
#include <mpi.h>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"
#include "amazon/dsstne/engine/NNNetwork.h"

int main(int argc, char** argv) {
  std::cout << "sizeof(char) = " << sizeof(char) << std::endl;
  std::cout << "sizof(uchar) = " << sizeof(unsigned char) << std::endl;

//  char *argv2 = "process";
//  getGpu().Startup(1, &argv2);
//  MPI_Init(&argc, &argv);

//  int _numprocs;
//  int _id;
//  MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
//  MPI_Comm_rank(MPI_COMM_WORLD, &_id);

//  std::cout << "initialized MPI " << _numprocs << ", " << _id << std::endl;

//  getGpu().Shutdown();
// getGpu().Startup(argc, argv);
// NNNetwork *network = LoadNeuralNetworkNetCDF(argv[1], 32);
// std::vector<const NNLayer*> inputLayers = network->GetLayers(NNLayer::Kind::Input);
// std::cout << "Input Layer Size: " << inputLayers.size() << std::endl;
// getGpu().Shutdown();
}
