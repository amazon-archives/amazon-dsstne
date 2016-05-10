/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef FILTERS_H
#include <json/json.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "Utils.h"
using namespace Json;
using namespace std;
class AbstractFilter
{
public:
    virtual void loadFilter(unordered_map<string, unsigned int>& ,
                            unordered_map<string, unsigned int>& ,
                            string ) = 0;
    virtual void applyFilter(float *,int ) = 0 ;
    virtual void applyFilter(float *, int, int, int) = 0;
    virtual string getFilterType() = 0;
protected:
    void updateRecords(float *,unordered_map<int,float> *);
    void updateRecords(float *,unordered_map<int,float> *, int, int);

};

class SamplesFilter : public AbstractFilter
{
private:
    vector<unordered_map<int,float>*> *samplefilters;

    void loadSingleFilter(unordered_map<string, unsigned int> &xMInput,
                          unordered_map<string, unsigned int> &xMSamples,
                          vector<unordered_map<int,float>*>&sampleFilters,
                          const string &filePath);
public:
    SamplesFilter()
    {
        samplefilters = NULL;
    }

    void loadFilter(unordered_map<string, unsigned int> &xMInput,
                    unordered_map<string, unsigned int> &xMSamples,
                    string filePath);

    void applyFilter(float *,int ) ;
    void applyFilter(float *,int, int, int);

    string getFilterType()
    {
        return "samplesFilterType";
    }
    ~SamplesFilter() ;
};

class FilterConfig
{
private:
    SamplesFilter *sampleFilter;
    string outputFileName;
public :
    FilterConfig()
    {
	sampleFilter = NULL;
    }

    ~FilterConfig()
    {
	delete(sampleFilter);
    }

    void setOutputFileName(string xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    string getOutputFileName()
    {
        return outputFileName;
    }

    void setSamplesFilter(SamplesFilter* xSampleFilter)
    {
        sampleFilter = xSampleFilter;
    }

    void applySamplesFilter(float *xInput, int xSampleIndex, int offSet, int width)
    {
	    if(sampleFilter != NULL) {
		    sampleFilter->applyFilter(xInput, xSampleIndex, offSet, width);
	    }
    }

};

/**
Parses the  filterConfig file which should be a json
and created the Filters based on the Indexes  given for the Input Layer mInput
and sampled mSamples
*/
FilterConfig* loadFilters(string , string ,
                                  unordered_map<string, unsigned int>& ,
                                  unordered_map<string, unsigned int>& );

#define FILTERS_H
#endif
