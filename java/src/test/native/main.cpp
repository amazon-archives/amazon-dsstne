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
#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <tuple>
#include <mpi.h>
#include <fstream>
#include <sstream>

#include "amazon/dsstne/engine/DsstneContext.h"
#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"
#include "amazon/dsstne/engine/NNNetwork.h"

using namespace std;

void readIndexFile(const string &indexFile, map<string, uint32_t> &indexMap, map<uint32_t, string> &rIndexMap)
{
    ifstream in(indexFile);
    if(!in) {
        throw runtime_error(indexFile + " does not exist");
    }
    string line;
    while (getline(in, line))
    {
        int idx = line.find_first_of('\t');
        string key = line.substr(0, idx);
        uint32_t index = stoul(line.substr(idx + 1, line.size()));
        indexMap[key] = index;
        rIndexMap[index] = key;
    }
    in.close();
}

void readInputFile(const string &inputFile, const map<string, uint32_t> &indexMap, uint64_t *sparseStart,
                   uint64_t *sparseEnd, uint32_t *sparseIndex, int *sparseData)
{
    ifstream in(inputFile);
    if(!in) {
        throw runtime_error(inputFile + " does not exist");
    }

    string line;
    int i = 0;

    sparseStart[0] = 0;
    while (getline(in, line))
    {
        if (i != 0)
        {
            sparseStart[i] = sparseEnd[i - 1];
        }

        int idx = line.find_first_of('\t');
        string v = line.substr(idx + 1, line.size());

        int j = 0;
        stringstream vs(v);
        string elem;
        while (getline(vs, elem, ':'))
        {
            int vidx = elem.find_first_of(',');
            string key = elem.substr(0, vidx);

            sparseIndex[sparseStart[i] + j] = indexMap.at(key);
            sparseData[sparseStart[i] + j] = 1;
            ++j;
        }
        sparseEnd[i] = sparseStart[i] + j;
        ++i;
    }
    in.close();
}

int main(int argc, char** argv)
{
    uint32_t k = 10;
    uint32_t batchSize = 32;
    float sparseDensity = 0.09;

    const string networkFile = argv[1];
    const string indexFile = argv[2];
    const string inputFile = argv[3];

    DsstneContext dc(networkFile, batchSize);

    NNNetwork *network = dc.getNetwork();
    cout << "main.o: loaded network " << network->GetName() << endl;

    vector<const NNLayer*> inputLayers;
    vector<NNDataSetDescriptor> datasets;

    auto it = network->GetLayers(NNLayer::Kind::Input, inputLayers);
    for (; it != inputLayers.end(); ++it)
    {
        const NNLayer *layer = (*it);
        NNDataSetDescriptor desc;

        uint32_t x, y, z, w;
        tie(x, y, z, w) = layer->GetDimensions();
        desc._dim = NNDataSetDimensions(x, y, z);
        desc._name = layer->GetDataSetName();
        desc._dataType = NNDataSetEnums::DataType::Int;
        desc._attributes = NNDataSetEnums::Attributes::Sparse;
        desc._examples = network->GetBatch();
        desc._sparseDensity = sparseDensity;

        datasets.push_back(desc);
    }

    dc.initInputLayerDataSets(datasets);

    map<string, uint32_t> indexes;
    map<uint32_t, string> rIndexes;
    readIndexFile(indexFile, indexes, rIndexes);
    cout << "Read " << indexes.size() << " indexes" << endl;

    const NNLayer* inputLayer = network->GetLayer("Input");
    uint32_t x, y, z, w;
    tie(x, y, z, w) = inputLayer->GetDimensions();
    size_t sparseDataLength = ((float) (x * y * z * batchSize)) * sparseDensity;
    NNDataSetBase* inputDataset = inputLayer->GetDataSet();

    NNFloat *outputScores;
    uint32_t *outputIndexes;
    cudaMallocManaged(&outputScores, k * batchSize * sizeof(NNFloat));
    cudaMallocManaged(&outputIndexes, k * batchSize * sizeof(uint32_t));

    for (int i = 0; i < 1; ++i)
    {
        uint64_t *sparseStart = (uint64_t*) calloc(batchSize, sizeof(uint64_t));
        uint64_t *sparseEnd = (uint64_t*) calloc(batchSize, sizeof(uint64_t));
        uint32_t *sparseIndex = (uint32_t*) calloc(sparseDataLength, sizeof(uint32_t));
        int *sparseData = (int*) calloc(sparseDataLength, sizeof(int));

        readInputFile(inputFile, indexes, sparseStart, sparseEnd, sparseIndex, sparseData);

        inputDataset->LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);

        free(sparseStart);
        free(sparseEnd);
        free(sparseIndex);
        free(sparseData);

        network->SetPosition(0);
        network->PredictBatch();

        const NNLayer *outputLayer = network->GetLayer("Output");
        NNFloat *dUnitBuffer = network->GetUnitBuffer("Output");

        tie(x, y, z, w) = outputLayer->GetDimensions();

        size_t width = x * y * z;

        kCalculateTopK(dUnitBuffer, outputScores, outputIndexes, batchSize, width, k);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < batchSize; ++i)
        {
            printf("%d\t", i+1);
            for (size_t j = 0; j < k; ++j)
            {

                const string &idx = rIndexes.at(outputIndexes[i * k + j]);
                printf("%s:%5.3f,", idx.c_str(), outputScores[i * k + j]);
            }
            printf("\n");
        }

    }

    cudaFree(outputScores);
    cudaFree(outputIndexes);
}
