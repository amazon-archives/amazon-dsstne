/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef NN_SORT_H
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>

#include "GpuTypes.h"
#include "NNTypes.h"
#include "NNNetwork.h"
#include "Filters.h"

using namespace std;

class NNRecsGenerator
{
private :
    GpuBuffer<NNFloat>* pbKey ;
    GpuBuffer<unsigned int>* pbUIValue;
    GpuBuffer<NNFloat> *pFilteredOutput;
    vector <GpuBuffer<NNFloat>*> *vNodeFilters;
    string recsGenLayerLabel;
    string scorePrecision;
    
public:
    static const string DEFAULT_LAYER_RECS_GEN_LABEL;
    static const unsigned int TOPK_SCALAR;
    static const string DEFAULT_SCORE_PRECISION;

    NNRecsGenerator(unsigned int,
		unsigned int,
		unsigned int,
    string layer=DEFAULT_LAYER_RECS_GEN_LABEL,
    string precision=DEFAULT_SCORE_PRECISION);

    void generateRecs(NNNetwork *network,
                      int topK,
                      FilterConfig* filters,
                      vector<string> & customerIndex,
                      vector<string> & featureIndex);

    
    string getRecsLayerLabel();

    void reset();
    
    ~NNRecsGenerator()
    {
        reset();
    }
};
#endif
