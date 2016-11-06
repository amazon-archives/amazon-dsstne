/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#include <json/json.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

#include "Filters.h"
#include "GpuTypes.h"
#include "NNTypes.h"

using namespace Json;
using namespace std;

AbstractFilter::~AbstractFilter()
{
}

/**
        Updates the records in the array with the filters
        x[Index] =  x[Index] * (filterValue for that index)
*/
void AbstractFilter::updateRecords(float *xArray,unordered_map<int,float> *xFilter)
{

    if (xFilter && xFilter->size() >0)
    {
        unordered_map<int,float>::iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index =  filterIter->first ;
            float value = filterIter->second;
            xArray[ index  ] = value *  xArray[ index ];
        }
    }
}

void AbstractFilter::updateRecords(float *xArray,unordered_map<int,float> *xFilter, int offSet, int width)
{
    /* Filter 
       @param xArray values to be filtered
       @param offSet the starting global index of current xArray
       @param width the length of xArray
    
    */
    

    if (xFilter && xFilter->size() >0)
    {
        unordered_map<int,float>::iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index =  filterIter->first ;
            float value = filterIter->second;
            // xArray global index [offset, offSet + width)
            // if index is falling outside of range, not changing xArray value
            // currently value is always zero in binary inputs
            if (index >=offSet && index < offSet + width) { 
                xArray[ index-offSet  ] = value *  xArray[ index-offSet ];
            }
        }
    }
}



int gSamplesLoggingInterval = 10000;

void SamplesFilter::loadSingleFilter(unordered_map<string, unsigned int> &xMInput,
                                     unordered_map<string, unsigned int> &xMSamples,
                                     vector<unique_ptr<unordered_map<int,float>>> &sampleFilters,
                                     const string &filePath) {
    ifstream samplesFile(filePath);
    timeval ts;
    gettimeofday(&ts, NULL);
    unordered_map<int,float>* customerSampleFilter =  nullptr;
    int samplesFilterCount = 0;
    vector<string> filters;
    if(samplesFile.good())
    {
        string line;
        int sample = -1;
        while(getline(samplesFile,line))
        {
            filters = split(line, ':');
            if(filters.size() > 0)
            {
                vector<string> vals =  split(filters[0],'\t');
                if(vals.size() >0)
                {
                    try
                    {
                        sample = xMSamples.at(vals[0]);
                        if(vals.size() > 1)
                        {
                            filters[0] = vals[1];
                        }
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            } //filters[i] is the ith Feature for that customer

            customerSampleFilter = new unordered_map<int,float>();
            for(int  i =0; i < filters.size(); ++i)
            {
                vector<string>  vals =  split(filters[i],',');
                if(vals.size() > 0)
                {
                    try
                    {
                        int key = xMInput.at(vals[0]);
                        float value =  0.0f;
                        if(vals.size() >1)
                        {
                            value =  atof(vals[1].c_str());
                            // This is hack for reading just the recs
                            // Because the current one has date
                            if( value > 10.0 )
                            {
                                value = 0.0f;
                            }
                        }
                        (*customerSampleFilter)[key] = value;
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }
            if(sample != -1)
            {
                sampleFilters[sample].reset(customerSampleFilter);
                ++samplesFilterCount;
                if(samplesFilterCount % gSamplesLoggingInterval ==0)
                {
                    timeval t2;
                    gettimeofday(&t2, NULL);
                    cout <<"Progress Parsing Filter "<< samplesFilterCount;
                    cout <<"Time " <<  elapsed_time(t2, ts)<<endl;
                    gettimeofday(&ts, NULL);

                }
            }
        }
    }
    else
    {
        cout << "Unable to read the file "<< filePath<<endl;
        throw std::invalid_argument("invalid sample filters " + filePath + ", exiting...");

    }

}

void SamplesFilter::loadFilter(unordered_map<string, unsigned int>& xMInput,
                               unordered_map<string, unsigned int>& xMSamples,
                               string filterFilePath)
{
    /**
     @param xMInput: $Feature , $GLOBAL_INDEX_FOR_FEATURES
     @param xMSamples: $CUST, $GLOBAL_INDEX_FOR_CUST
     @param filterFilePath: name of sample filter file. Samples filter should be as below:
                            $CUS    $FEATURE,$VALUE:$FEATURE,$VALUE
     
     

     TODO There is a hack currently where when the value is >10.0 i am assuming to zero
     The reason is currently watch Filters have watch dates as the first Suffix
    
 
     TODO:larger number of inserts time will be wasted for resizing the vector
     and the size is the maximum of the Samples. Might not be a good  use case when we have sparse
     Filter
    */


    samplefilters.reset(new vector<unique_ptr<unordered_map<int,float>>>(xMSamples.size()));

    vector<string> files;
    if (listFiles(filterFilePath, false, files) == 0) {
        cout << "Loading " << files.size() << " filter files" << endl;

        for (auto const &file: files) {
            cout << "\tLoading filter: " << file << endl;
            loadSingleFilter(xMInput, xMSamples, *samplefilters.get(), file);
        }
    }

    cout << "Info:SamplesFilter " << samplefilters->size() <<endl;
}

void SamplesFilter::applyFilter(float *xArray,int xSamplesIndex, int offSet, int width)
{
    unordered_map<int,float> *filter= (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray,filter, offSet, width);
}

void SamplesFilter::applyFilter(float *xArray,int xSamplesIndex)
{
    unordered_map<int,float> *filter= (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray,filter);
}

SamplesFilter::~SamplesFilter()
{
}

/**
Sample Filters.json
{
        "filters": [
                {"sampleFilters": "watches","nodeFilters": "primeFilters","outputFile":"primerecs" }
        ]
}
Takes a filters.json and parses the file and created the Filters
*/
FilterConfig* loadFilters(string samplesFilterFileName,string outputFileName,
                                  unordered_map<string, unsigned int>& xMInput,
                                  unordered_map<string, unsigned int>& xMSamples)
{
   
    
    Value index;
    Reader reader;
    FilterConfig *filterConfig  = new FilterConfig();
    SamplesFilter *samplesFilter = new SamplesFilter() ;
    samplesFilter->loadFilter(xMInput,xMSamples,samplesFilterFileName);
    filterConfig->setSamplesFilter(samplesFilter);
    filterConfig->setOutputFileName(outputFileName);
    //Cleaning up the existing file rather than appending the file
    FILE *fp =  fopen(outputFileName.c_str(),"w");
    fclose(fp);
    return filterConfig;
}

