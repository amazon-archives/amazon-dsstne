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

#ifndef DSSTNECONTEXT_H_
#define DSSTNECONTEXT_H_

#include "GpuTypes.h"
#include "NNTypes.h"
#include "NNLayer.h"

/**
 * Holds the context for an instance of NNNetwork,
 * which is the network itself along with data structures
 * needed to support operations (e.g. predict) on the network.
 * Currently can only support one context per process
 * since GpuContext is a static.
 */
class DsstneContext
{
 private:
    static const int ALL = -1;

    const std::string networkFilename;
    const uint32_t batchSize;
    const uint32_t maxK;

    std::map<string, GpuBuffer<NNFloat>*> dOutputScores;
    std::map<string, GpuBuffer<uint32_t>*> dOutputIndexes;

 public:
    DsstneContext(const std::string &networkFilename, uint32_t batchSize, int maxK = ALL);

    ~DsstneContext();

    NNNetwork* getNetwork() const;

    /**
     * Initializes empty datasets for the input layers given the NNDataSetDescriptors.
     */
    void initInputLayerDataSets(const std::vector<NNDataSetDescriptor> datasetDescriptors);

    GpuBuffer<NNFloat>* getOutputScoresBuffer(const std::string &layerName);

    GpuBuffer<uint32_t>* getOutputIndexesBuffer(const std::string &layerName);

    static DsstneContext* fromPtr(long ptr)
    {
        DsstneContext * dc = (DsstneContext *) ptr;
        if (dc == nullptr)
        {
            std::runtime_error("Cannot convert nullptr to DsstneContext");
        }
        return dc;
    }
};

#endif /* DSSTNECONTEXT_H_ */
