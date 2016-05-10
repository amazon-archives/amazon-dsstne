/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <netcdf>
#include <sys/time.h>
#include <unordered_map>
#include <stdexcept>

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

int gLoggingRate = 10000;

void loadIndex(unordered_map<string, unsigned int> &mLabelToIndex,
               string indexFileName) {
    string line;
    ifstream is(indexFileName);
    while (getline(is, line)) {
        vector<string> vData = split(line, '\t');
        mLabelToIndex[vData[0]] = atoi(vData[1].c_str());
    }
    is.close();
}

void exportIndex(unordered_map<string, unsigned int> &mLabelToIndex, string indexFileName) {
    ofstream outputIndexStream(indexFileName);
    unordered_map<string, unsigned int>::iterator indexIterator;
    for (indexIterator = mLabelToIndex.begin(); indexIterator != mLabelToIndex.end(); indexIterator++) {
        outputIndexStream << indexIterator->first << "\t" << indexIterator->second << endl;
    }
    outputIndexStream.close();
}

void indexSingleFile(const string &samplesFileName,
                     const bool enableFeatureIndexUpdates,
                     unordered_map<string, unsigned int> &mFeatureIndex,
                     unordered_map<string, unsigned int> &mSampleIndex,
                     bool &featureIndexUpdated,
                     bool &sampleIndexUpdated,
                     map<unsigned int, vector<unsigned int>> &mSignals,
                     map<unsigned int, vector<float>> &mSignalValues) {

    ifstream inputFileStream(samplesFileName);
    timeval ts;
    gettimeofday(&ts, NULL);
    string line;
    int lineNumber = 0;

    while (getline(inputFileStream, line)) {
        lineNumber++;
        // ignore empty lines - there could be new lines at the end of the file
        if(line.empty()) {
            continue;
        }

        // Determine the first tab and split the line into 2 parts:
        //  1) customer/sample information: <customer_id>,<marketplace>
        //  2) data point tuples with <feature_label>,<score|date|value>
        int index = line.find('\t');
        if (index < 0) {
            cout << "Warning: Skipping over malformed line (" << line << ") at line " << lineNumber << " in file " << samplesFileName <<
            endl;
            continue;
        }

        string sampleLabel = line.substr(0, index);
        string dataString = line.substr(index + 1);

        // Check the sampleIndex and update it if required.
        unsigned int sampleIndex = 0;
        try {
            sampleIndex = mSampleIndex.at(sampleLabel);
        }
        catch (const std::out_of_range &oor) {
            unsigned int index = mSampleIndex.size();
            mSampleIndex[sampleLabel] = index;
            sampleIndex = mSampleIndex[sampleLabel];
            sampleIndexUpdated = true;
        }
        // Now process the dataPointTuples to extract signals and values
        vector<unsigned int> signals;  // Contains the index of for the feature
        vector<float> signalValue;     // Contains the value for the feature

        vector<string> dataPointTuples = split(dataString, ':');
        for (unsigned int i = 0; i < dataPointTuples.size(); i++) {
            string dataPoint = dataPointTuples[i];
            vector<string> dataElems = split(dataPoint, ',');

            if (dataElems.size() < 1 || dataElems[0].length() == 0) {
                // Skip over empty elements.
                continue;
            }

            if (dataElems.size() > 2) {
                cout << "Warning: Data point [" << dataPoint << "] at line " << lineNumber <<
                " has more than 1 value for feature. ";
                cout << "Keeping the first value and ignoring subsequent values." << endl;
            }

            string featureName = dataElems[0];
            float featureValue = 0.0;
            if (dataElems.size() > 1) {
                // Look for the optional value for the feature.
                // Since value for a feature can be int or float, its safer to parse float.
                featureValue = stof(dataElems[1]);
            }

            // Look up the index for the given feature.
            unsigned int featureIndex = 0;
            try {
                featureIndex = mFeatureIndex.at(featureName);
            }
            catch (const std::out_of_range &oor) {
                if (enableFeatureIndexUpdates) {
                    unsigned int index = mFeatureIndex.size();
                    mFeatureIndex[featureName] = index;
                    featureIndex = index;
                    featureIndexUpdated = true;
                } else {
                    // Ignore this data point if we are not allowed to
                    // update the feature index.
                    continue;
                }
            }
            // Update signals with this feature index
            signals.push_back(featureIndex);
            // Update the data with the value
            signalValue.push_back(featureValue);
        }

        mSignals[sampleIndex] = signals;
        mSignalValues[sampleIndex] = signalValue;
        if (mSampleIndex.size() % gLoggingRate == 0) {
            timeval t2;
            gettimeofday(&t2, NULL);
            cout << "Progress Parsing" << mSampleIndex.size();
            cout << "Time " << elapsed_time(t2, ts) << endl;
            gettimeofday(&ts, NULL);
        }
    }
}

void indexFile(const string &samplesPath,
               const bool enableFeatureIndexUpdates,
               unordered_map<string, unsigned int> &mFeatureIndex,
               unordered_map<string, unsigned int> &mSampleIndex,
               bool &featureIndexUpdated,
               bool &sampleIndexUpdated,
               vector<unsigned int> &vSparseStart,
               vector<unsigned int> &vSparseEnd,
               vector<unsigned int> &vSparseIndex,
               vector<float> &vSparseData) {

    featureIndexUpdated = false;
    sampleIndexUpdated = false;

    if (!fileExists(samplesPath)) {
        cout << "Error: " << samplesPath << " not found. Exiting" << endl;
        exit(1);
    }

    vector<string> files;

    // maps the index of samples -> signals
    // we buffer the entire content of the directory to align the samples when writing sparseIndex
    map<unsigned int, vector<unsigned int>> mSignals;
    map<unsigned int, vector<float>> mSignalValues;

    if (listFiles(samplesPath, false, files) == 0) {
        cout << "Indexing " << files.size() << " files" << endl;

        for (auto const &file: files) {
            cout << "\tIndexing file: " << file << endl;

            // read file and keep updating index maps
            indexSingleFile(file,
                            enableFeatureIndexUpdates,
                            mFeatureIndex,
                            mSampleIndex,
                            featureIndexUpdated,
                            sampleIndexUpdated,
                            mSignals,
                            mSignalValues);
        }
    }

    // Iterate through the signals to generate the signal indexes
    // This is done in a ordered map so that the same customers will have the same signal order
    map<unsigned int, vector<unsigned int>>::iterator mSignalsIter;
    map<unsigned int, vector<float>>::iterator mSignalValuesIter;
    for (mSignalsIter = mSignals.begin(); mSignalsIter != mSignals.end(); mSignalsIter++) {
        vSparseStart.push_back(vSparseIndex.size());
        vector<unsigned int> &signals = mSignalsIter->second;

        // Retrieve the corresponding value. Since we record both signal and signal values
        // we can be confident that (a) there is a valid vector of floats and (b) they are
        // of the same number of entries as signals.
        mSignalValuesIter = mSignalValues.find(mSignalsIter->first);
        vector<float> &signalValues = mSignalValuesIter->second;

        for (unsigned int i = 0; i < signals.size(); ++i) {
            vSparseIndex.push_back(signals[i]);
            vSparseData.push_back(signalValues[i]);
        }
        vSparseEnd.push_back(vSparseIndex.size());
    }
}

void generateNetCDFIndexes(const string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const string &outFeatureIndexFileName,
                           const string &outSampleIndexFileName,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseData) {

    bool featureIndexUpdated;
    bool sampleIndexUpdated;


    indexFile(samplesPath,
              enableFeatureIndexUpdates,
              mFeatureIndex,
              mSampleIndex,
              featureIndexUpdated,
              sampleIndexUpdated,
              vSparseStart,
              vSparseEnd,
              vSparseIndex,
              vSparseData);

    // Now export the updated indices files only if they were updated.
    if (featureIndexUpdated) {
        exportIndex(mFeatureIndex, outFeatureIndexFileName);
        cout << "Exported " << outFeatureIndexFileName << " with " << mFeatureIndex.size() << " entries." << endl;
    }

    if (sampleIndexUpdated) {
        exportIndex(mSampleIndex, outSampleIndexFileName);
        cout << "Exported " << outSampleIndexFileName << " with " << mSampleIndex.size() << " entries." << endl;
    }
}

void generateNetCDFIndexes(const string &samplesFileName,
                           const bool enableFeatureIndexUpdates,
                           const string &dataSetName,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseValue) {

    string outFeatureIndexFileName = dataSetName + FEATURE_INDEX_FILE_SUFFIX;
    string outSampleIndexFileName = dataSetName + SAMPLES_INDEX_FILE_SUFFIX;
    generateNetCDFIndexes(samplesFileName,
                          enableFeatureIndexUpdates,
                          outFeatureIndexFileName,
                          outSampleIndexFileName,
                          mFeatureIndex,
                          mSampleIndex,
                          vSparseIndex,
                          vSparseEnd,
                          vSparseIndex,
                          vSparseValue);

}

unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex) {
    // Make the maxFeatureIndex a Multiple of 32
    // Pre- Titan-X:
    // maxFeatureIndex = ((maxFeatureIndex + 31) >> 5) << 5;
    return ((maxFeatureIndex + 127) >> 7) << 7;
}

void writeNetCDFFile(vector<unsigned int> &vSparseStart,
                     vector<unsigned int> &vSparseEnd,
                     vector<unsigned int> &vSparseIndex,
                     vector<float> &vSparseData,
                     string fileName,
                     string datasetName,
                     unsigned int maxFeatureIndex) {

    cout << "Raw max index is: " << maxFeatureIndex << endl;
    maxFeatureIndex = roundUpMaxIndex(maxFeatureIndex);
    cout << "Rounded up max index to: " << maxFeatureIndex << endl;

    try {
        NcFile nc(fileName, NcFile::replace);
        if (nc.isNull()) {
            cout << "Error creating output file:" << fileName << endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, NNDataSetBase::Attributes::Sparse);
        nc.putAtt("kind0", ncUint, NNDataSetBase::Kind::Numeric);
        nc.putAtt("dataType0", ncUint, NNDataSetBase::DataType::Float);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", ncUint, examplesDim);
        NcVar sparseEndVar = nc.addVar("sparseEnd0", ncUint, examplesDim);
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", ncUint, sparseDataDim);
        NcVar sparseDataVar = nc.addVar("sparseData0", ncFloat, sparseDataDim);
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);
        sparseDataVar.putVar(&vSparseData[0]);

        cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << endl;
    } catch (std::exception &e) {
        cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}

void writeNetCDFFile(vector<unsigned int> &vSparseStart,
                     vector<unsigned int> &vSparseEnd,
                     vector<unsigned int> &vSparseIndex,
                     string fileName,
                     string datasetName,
                     unsigned int maxFeatureIndex) {
    // Make the maxFeatureIndex a Multuple of 32
    // Pre- Titan-X:
    // maxFeatureIndex = ((maxFeatureIndex + 31) >> 5) << 5;
    cout << "Raw max index is: " << maxFeatureIndex << endl;
    maxFeatureIndex = roundUpMaxIndex(maxFeatureIndex);
    cout << "Rounded up max index to: " << maxFeatureIndex << endl;

    try {
        NcFile nc(fileName, NcFile::replace);
        if (nc.isNull()) {
            cout << "Error creating output file:" << fileName << endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, (NNDataSetBase::Attributes::Sparse + NNDataSetBase::Attributes::Boolean));
        nc.putAtt("kind0", ncUint, NNDataSetBase::Kind::Numeric);
        nc.putAtt("dataType0", ncUint, NNDataSetBase::DataType::UInt);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", ncUint, examplesDim);
        NcVar sparseEndVar = nc.addVar("sparseEnd0", ncUint, examplesDim);
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", ncUint, sparseDataDim);
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);

        cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << endl;
    } catch (std::exception &e) {
        cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}




