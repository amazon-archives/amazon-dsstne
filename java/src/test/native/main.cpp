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

#include "amazon/dsstne/engine/DsstneContext.h"
#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"
#include "amazon/dsstne/engine/NNNetwork.h"

using namespace std;

int main(int argc, char** argv) {

    DsstneContext dc("/home/kiuk/tmp/gl.nc", 1);

    NNNetwork *network = dc.getNetwork();

    cout << "main.o: loaded network " << network->GetName() << endl;

    vector<const NNLayer*> inputLayers;
    vector<NNDataSetBase*> datasets;

    auto it = network->GetLayers(NNLayer::Kind::Input, inputLayers);
    for(; it != inputLayers.end(); ++it) {
        const NNLayer *layer = (*it);
        NNDataSetDescriptor desc;

        uint32_t x,y,z,w;
        tie(x,y,z,w) = layer->GetDimensions();
        desc._dim = NNDataSetDimensions(x,y,z);
        desc._name = layer->GetDataSetName();
        desc._dataType = NNDataSetEnums::DataType::Int;
        desc._attributes = NNDataSetEnums::None;
        desc._examples = network->GetBatch();
        desc._sparseDensity = 1;

        NNDataSetBase *ds = createNNDataSet(desc);
        datasets.push_back(ds);
    }

    network->LoadDataSets(datasets);
    it = network->GetLayers(NNLayer::Kind::Input, inputLayers);
    for(; it != inputLayers.end(); ++it) {
        const NNLayer *layer = (*it);
        const string &layerName = layer->GetName();
        const string &datasetName = layer->GetDataSetName();
        cout << "main.o: loaded " << datasetName << " to layer " << layerName << endl;
    }

    network->PredictBatch();
    network->SetPosition(0);

    const NNLayer *outputLayer = network->GetLayer("Output");
    NNFloat *dUnitBuffer = network->GetUnitBuffer("Output");

    uint32_t x,y,z,w;
    tie(x,y,z,w) = outputLayer->GetDimensions();
    uint32_t batchSize = network->GetBatch();
    size_t width = x * y * z;
    uint32_t k = 10;

    NNFloat *outputScores;
    uint32_t *outputIndexes;
    cudaMallocManaged(&outputScores, k * batchSize * sizeof(NNFloat));
    cudaMallocManaged(&outputIndexes, k * batchSize * sizeof(uint32_t));

    kCalculateTopK(dUnitBuffer, outputScores, outputIndexes, batchSize, width, k);
    cudaDeviceSynchronize();

    for(size_t i =0; i < batchSize; ++i)
    {
        for(size_t j=0; j<k; ++j) {
                printf("%u:%5.3f,", outputIndexes[i * k + j], outputScores[i*k + j]);
            }
    }

    printf("\n");

    cudaFree(outputScores);
    cudaFree(outputIndexes);

    for (const auto &d : datasets)
    {
        delete d;
    }

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
