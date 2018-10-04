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

DsstneContext::DsstneContext(const string &networkFilename, uint32_t batchSize, int maxK) :
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

    for (; it != outputLayers.end(); ++it)
    {
        const string &layerName = (*it)->GetName();

        if(maxK != ALL)
        {
            // FIXME this only works for 1-D outputs
            if ((*it)->GetNumDimensions() > 1)
            {
                std::runtime_error("topK only supported on 1-D output layers");
            }
            size_t outputBufferLength = maxK * batchSize;
            printf(
                "DsstneContext::DsstneContext: Allocating output score and index buffers, each of size %zu for output layer %s\n",

                outputBufferLength, layerName.c_str());
            GpuBuffer<NNFloat> *outputScores = new GpuBuffer<NNFloat>(outputBufferLength, false, false);
            GpuBuffer<uint32_t> *outputIndexes = new GpuBuffer<uint32_t>(outputBufferLength, false, false);

            dOutputScores[layerName] = outputScores;
            dOutputIndexes[layerName] = outputIndexes;
        }
    }
}

GpuBuffer<NNFloat>* DsstneContext::getOutputScoresBuffer(const std::string &layerName)
{
    return dOutputScores.at(layerName);
}

GpuBuffer<uint32_t>* DsstneContext::getOutputIndexesBuffer(const std::string &layerName)
{
    return dOutputIndexes.at(layerName);
}

DsstneContext::~DsstneContext()
{
    const string networkName = getNetwork()->GetName();
    for (const auto &kv : dOutputScores)
    {
        delete (kv.second);
    }
    for (const auto &kv : dOutputIndexes)
    {
        delete (kv.second);
    }

    dOutputScores.clear();
    dOutputIndexes.clear();

    delete getNetwork();
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
    for (const auto &descriptor : datasetDescriptors)
    {
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

