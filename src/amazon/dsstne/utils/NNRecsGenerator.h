/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef NNRECSGENERATOR_H
#define NNRECSGENERATOR_H

#include <vector>
#include <string>
#include <memory>

#include "GpuTypes.h"
#include "NNTypes.h"

class FilterConfig;
class NNNetwork;

class NNRecsGenerator
{
    std::unique_ptr<GpuBuffer<NNFloat>> pbKey;
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue;
    std::unique_ptr<GpuBuffer<NNFloat>> pFilteredOutput;
    std::vector<GpuBuffer<NNFloat>*> *vNodeFilters;
    std::string recsGenLayerLabel;
    std::string scorePrecision;
    
public:
    static const std::string DEFAULT_LAYER_RECS_GEN_LABEL;
    static const unsigned int TOPK_SCALAR;
    static const std::string DEFAULT_SCORE_PRECISION;

    NNRecsGenerator(unsigned int xBatchSize,
                    unsigned int xK, 
                    unsigned int xOutputBufferSize,
                    const std::string &layer = DEFAULT_LAYER_RECS_GEN_LABEL,
                    const std::string &precision = DEFAULT_SCORE_PRECISION);

    void generateRecs(const NNNetwork *network,
                      unsigned int topK,
                      const FilterConfig *filters,
                      const std::vector<std::string> &customerIndex,
                      const std::vector<std::string> &featureIndex);
};

#endif
