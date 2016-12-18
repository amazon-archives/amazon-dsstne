/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

class AbstractFilter
{
public:
    virtual ~AbstractFilter() = default;

    virtual void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                            std::unordered_map<std::string, unsigned int> &xMSamples,
                            const std::string &filePath) = 0;

    virtual void applyFilter(float *xArray, int xSamplesIndex) const = 0;
    virtual void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const = 0;

    virtual std::string getFilterType() const = 0;

protected:
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter) const;
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter, int offset, int width) const;
};

class SamplesFilter : public AbstractFilter
{
    std::unique_ptr<std::vector<std::unique_ptr<std::unordered_map<int, float>>>> samplefilters;

    void loadSingleFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples,
                          std::vector<std::unique_ptr<std::unordered_map<int, float>>> &sampleFilters,
                          const std::string &filePath);

public:
    void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                    std::unordered_map<std::string, unsigned int> &xMSamples,
                    const std::string &filePath);

    void applyFilter(float *xArray, int xSamplesIndex) const;
    void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const;

    std::string getFilterType() const
    {
        return "samplesFilterType";
    }
};

class FilterConfig
{
    std::unique_ptr<SamplesFilter> sampleFilter;
    std::string outputFileName;

public :
    void setOutputFileName(const std::string &xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    std::string getOutputFileName() const
    {
        return outputFileName;
    }

    void setSamplesFilter(SamplesFilter *xSampleFilter)
    {
        sampleFilter.reset(xSampleFilter);
    }

    void applySamplesFilter(float *xInput, int xSampleIndex, int offset, int width) const
    {
        if (sampleFilter)
        {
            sampleFilter->applyFilter(xInput, xSampleIndex, offset, width);
        }
    }
};

/**
 * Parses a filterConfig file, which should be in JSON format
 *
 * This filter will be created based on the indexes given for the input layer,
 * mInput, and samples, mSamples.
 *
 * Sample Filters.json:
 *
 *    "filters": [
 *        {"sampleFilters": "watches", "nodeFilters": "primeFilters", "outputFile":"primerecs" }
 *    ]
 */
FilterConfig* loadFilters(const std::string &samplesFilterFileName,
                          const std::string &outputFileName,
                          std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples);

#endif
