/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

#include "DsstneContext.h"

namespace
{
const int ARGC = 1;
char *ARGV = "dsstne-faux-process";
const unsigned long SEED = 12134ULL;
}

using namespace std;

DsstneContext::DsstneContext(const string &networkFilename, uint32_t batchSize, int maxK):
    networkFilename(networkFilename),
    batchSize(batchSize),
    maxK(maxK)
{
    getGpu().Startup(ARGC, &ARGV);
    getGpu().SetRandomSeed(SEED);
    NNNetwork *network = LoadNeuralNetworkNetCDF(networkFilename, batchSize);
    getGpu().SetNeuralNetwork(network);

    vector<const NNLayer*> outputLayers;
    vector<const NNLayer*>::iterator it = network->GetLayers(NNLayer::Kind::Output, outputLayers);

    for(; it != outputLayers.end(); ++it) {
        const string &layerName = (*it)->GetName();

        size_t outputBufferLength;

        if(maxK == ALL) {
            // no topK, return all output layer
            uint32_t x,y,z,w;
            tie(x,y,z,w) = (*it)->GetDimensions();
            outputBufferLength = x * y * z * batchSize;
        } else {
            // FIXME this only works for 1-D outputs
            if((*it)->GetNumDimensions() > 1) {
                std::runtime_error("topK only supported on 1-D output layers");
            }
            outputBufferLength = maxK * batchSize;
        }

        printf("DsstneContext::DsstneContext: Allocating buffer of size %zu for output layer %s\n", outputBufferLength, layerName.c_str());
    }
}

DsstneContext::~DsstneContext()
{
    const string networkName = getGpu()._pNetwork->GetName();
    for(const auto &kv: dOutputScores)
    {
        delete(kv.second);
    }
    for(const auto &kv: dOutputIndexes)
    {
        delete(kv.second);
    }

    dOutputScores.clear();
    dOutputIndexes.clear();

    NNNetwork *network = getNetwork();
    for(const auto &layerName :network->GetLayers()){
        const NNLayer *layer = network->GetLayer(layerName);
        delete layer->GetDataSet();
    }

    delete network;

    getGpu().Shutdown();
    printf("DsstneContext::~DsstneContext: Destroyed context for network %s\n", networkName.c_str());
}

NNNetwork* DsstneContext::getNetwork() const
{
    return getGpu()._pNetwork;
}

void DsstneContext::initInputLayerDataSets(const vector<NNDataSetDescriptor> datasetDescriptors)
{
    vector<NNDataSetBase*> datasets;
    for(const auto &descriptor : datasetDescriptors) {
        NNDataSetBase *dataset = createNNDataSet(descriptor);
        datasets.push_back(dataset);
    }

    /*
     * LoadDataSet marks the network as "dirty" meaning that the next time the Predict()
     * method is called on the network, it will refresh the state of the network
     * which is expensive as it tries to re-allocate the GpuBuffers by calling the Shard()
     * method on the dataset.
     * Run through a prediction once on the dataset to prime (allocate) the GpuBuffers
     * and RefreshState() of the network once. Going forward we will avoid marking the network
     * as dirty.
     */
    NNNetwork *network = getNetwork();
    network->LoadDataSets(datasets);
    network->PredictBatch();
    network->SetPosition(0);
}

