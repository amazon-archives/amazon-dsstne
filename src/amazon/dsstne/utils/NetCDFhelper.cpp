/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <netcdf>
#include <sys/time.h>
#include <unordered_map>
#include <stdexcept>

#include "NNEnum.h"
#include "Utils.h"
#include "NetCDFhelper.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

int gLoggingRate = 10000;

bool loadIndex(unordered_map<string, unsigned int> &labelsToIndices, istream &inputStream,
               ostream &outputStream) {
    string line;
    unsigned int linesProcessed = 0;
    const size_t initialIndexSize = labelsToIndices.size();
    while (getline(inputStream, line)) {
        vector<string> vData = split(line, '\t');
        linesProcessed++;
        if (vData.size() == 2 && !vData[0].empty()) {
            labelsToIndices[vData[0]] = atoi(vData[1].c_str());
        } else {
            outputStream << "Error: line " << linesProcessed << " contains invalid data" << endl;
            return false;
        }
    }

    const size_t numEntriesAdded = labelsToIndices.size() - initialIndexSize;
    outputStream << "Number of lines processed: " << linesProcessed << endl;
    outputStream << "Number of entries added to index: " << numEntriesAdded << endl;
    if (linesProcessed != numEntriesAdded) {
        outputStream << "Error: Number of entries added to index not equal to number of lines processed" << endl;
        return false;
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << strerror(errno) << endl;
        return false;
    }

    return true;
}

bool loadIndexFromFile(unordered_map<string, unsigned int> &labelsToIndices, const string &inputFile,
                       ostream &outputStream) {
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        outputStream << "Error: Failed to open index file" << endl;
        return false;
    }

    return loadIndex(labelsToIndices, inputStream, outputStream);
}

void exportIndex(unordered_map<string, unsigned int> &mLabelToIndex, string indexFileName) {
    ofstream outputIndexStream(indexFileName);
    unordered_map<string, unsigned int>::iterator indexIterator;
    for (indexIterator = mLabelToIndex.begin(); indexIterator != mLabelToIndex.end(); indexIterator++) {
        outputIndexStream << indexIterator->first << "\t" << indexIterator->second << endl;
    }
    outputIndexStream.close();
}

bool parseSamples(istream &inputStream,
                  const bool enableFeatureIndexUpdates,
                  unordered_map<string, unsigned int> &mFeatureIndex,
                  unordered_map<string, unsigned int> &mSampleIndex,
                  bool &featureIndexUpdated,
                  bool &sampleIndexUpdated,
                  map<unsigned int, vector<unsigned int>> &mSignals,
                  map<unsigned int, vector<float>> &mSignalValues,
                  ostream &outputStream) {
    timeval tBegin;
    gettimeofday(&tBegin, NULL);
    timeval tReported = tBegin;
    string line;
    int lineNumber = 0;

    while (getline(inputStream, line)) {
        lineNumber++;
        // ignore empty lines - there could be new lines at the end of the file
        if (line.empty()) {
            continue;
        }

        // Determine the first tab and split the line into 2 parts:
        //  1) customer/sample information: <customer_id>,<marketplace>
        //  2) data point tuples with <feature_label>,<score|date|value>
        int index = line.find('\t');
        if (index < 0) {
            outputStream << "Warning: Skipping over malformed line (" << line << ") at line " << lineNumber << endl;
            continue;
        }

        string sampleLabel = line.substr(0, index);
        string dataString = line.substr(index + 1);

        // Check the sampleIndex and update it if required.
        unsigned int sampleIndex = 0;
        try {
            sampleIndex = mSampleIndex.at(sampleLabel);
        }
        catch (const out_of_range &oor) {
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

            if (dataElems.empty() || dataElems[0].length() == 0) {
                // Skip over empty elements.
                continue;
            }

            const size_t numDataElems = dataElems.size();
            if (numDataElems > 2) {
                outputStream << "Warning: Data point [" << dataPoint << "] at line " << lineNumber << " has more "
                             << "than 1 value for feature (actual value: " << numDataElems << "). "
                             << "Keeping the first value and ignoring subsequent values." << endl;
            }

            string featureName = dataElems[0];
            float featureValue = 0.0;
            if (numDataElems > 1) {
                // Look for the optional value for the feature.
                // Since value for a feature can be int or float, its safer to parse float.
                featureValue = stof(dataElems[1]);
            }

            // Look up the index for the given feature.
            unsigned int featureIndex = 0;
            try {
                featureIndex = mFeatureIndex.at(featureName);
            }
            catch (const out_of_range &oor) {
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
            timeval tNow;
            gettimeofday(&tNow, NULL);
            outputStream << "Progress Parsing (Sample " << mSampleIndex.size() << ", ";
            outputStream << "Time " << elapsed_time(tNow, tReported) << ", ";
            outputStream << "Total " << elapsed_time(tNow, tBegin) << ")" << endl;
            tReported = tNow;
        }
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << strerror(errno) << endl;
        return false;
    }

    return true;
}

bool importSamplesFromPath(const string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           bool &featureIndexUpdated,
                           bool &sampleIndexUpdated,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseData,
                           ostream &outputStream) {

    featureIndexUpdated = false;
    sampleIndexUpdated = false;

    if (!fileExists(samplesPath)) {
        outputStream << "Error: " << samplesPath << " not found." << endl;
        return false;
    }

    vector<string> files;

    // maps the index of samples -> signals
    // we buffer the entire content of the directory to align the samples when writing sparseIndex
    map<unsigned int, vector<unsigned int>> mSignals;
    map<unsigned int, vector<float>> mSignalValues;

    if (listFiles(samplesPath, false, files) == 0) {
        outputStream << "Indexing " << files.size() << " files" << endl;

        for (auto const &file: files) {
            outputStream << "\tIndexing file: " << file << endl;

            ifstream inputStream(file);
            if (!inputStream.is_open()) {
                outputStream << "Error: Failed to open index file" << endl;
                return false;
            }

            // read file and keep updating index maps
            if (!parseSamples(inputStream,
                              enableFeatureIndexUpdates,
                              mFeatureIndex,
                              mSampleIndex,
                              featureIndexUpdated,
                              sampleIndexUpdated,
                              mSignals,
                              mSignalValues,
                              outputStream)) {
                return false;
            }
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

    return true;
}

bool generateNetCDFIndexes(const string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const string &outFeatureIndexFileName,
                           const string &outSampleIndexFileName,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseData,
                           ostream &outputStream) {

    bool featureIndexUpdated;
    bool sampleIndexUpdated;

    if (!importSamplesFromPath(samplesPath,
              enableFeatureIndexUpdates,
              mFeatureIndex,
              mSampleIndex,
              featureIndexUpdated,
              sampleIndexUpdated,
              vSparseStart,
              vSparseEnd,
              vSparseIndex,
              vSparseData,
              cout)) {

        return false;
    }

    // Now export the updated indices files only if they were updated.
    if (featureIndexUpdated) {
        exportIndex(mFeatureIndex, outFeatureIndexFileName);
        cout << "Exported " << outFeatureIndexFileName << " with " << mFeatureIndex.size() << " entries." << endl;
    }

    if (sampleIndexUpdated) {
        exportIndex(mSampleIndex, outSampleIndexFileName);
        cout << "Exported " << outSampleIndexFileName << " with " << mSampleIndex.size() << " entries." << endl;
    }

    return true;
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
            throw runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, NNDataSetEnums::Sparse);
        nc.putAtt("kind0", ncUint, NNDataSetEnums::Numeric);
        nc.putAtt("dataType0", ncUint, NNDataSetEnums::Float);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        NcVar sparseDataVar = nc.addVar("sparseData0", ncFloat, sparseDataDim);
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);
        sparseDataVar.putVar(&vSparseData[0]);

        cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << endl;
    } catch (exception &e) {
        cout << "Caught exception: " << e.what() << "\n";
        throw runtime_error("Error writing to NetCDF file.");
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
            throw runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, (NNDataSetEnums::Sparse + NNDataSetEnums::Boolean));
        nc.putAtt("kind0", ncUint, NNDataSetEnums::Numeric);
        nc.putAtt("dataType0", ncUint, NNDataSetEnums::UInt);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);

        cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << endl;
    } catch (exception &e) {
        cout << "Caught exception: " << e.what() << "\n";
        throw runtime_error("Error writing to NetCDF file.");
    }
}


unsigned int align(size_t size) {
    return (unsigned int) ((size + 127) >> 7) << 7;
}

bool addDataToNetCDF(NcFile& nc, const long long dataIndex, const string& dataName,
                const map<string, unsigned int>& mFeatureNameToIndex,
                const vector<vector<unsigned int> >& vInputSamples,
                const vector<vector<unsigned int> >& vInputSamplesTime, const vector<vector<float> >& vInputSamplesData,
                const bool alignFeatureDimensionality, int& minDate, int& maxDate, const int featureDimensionality) {
    // convert strings to format which is supported by netcdf
    vector<string> vFeatureName(mFeatureNameToIndex.size());
    vector<char*> vFeatureNameC(vFeatureName.size());
    if (mFeatureNameToIndex.size()) {
        for (map<string, unsigned int>::const_iterator it = mFeatureNameToIndex.begin();
                        it != mFeatureNameToIndex.end(); it++) {
            vFeatureName[it->second] = it->first;
        }

        for (int i = 0; i < vFeatureNameC.size(); i++) {
            vFeatureNameC[i] = &(vFeatureName[i])[0];
        }
    }
    string sDataIndex = to_string(dataIndex);

    NcDim indToFeatureDim;
    if (vFeatureNameC.size()) {
        indToFeatureDim = nc.addDim((string("indToFeatureDim") + sDataIndex).c_str(), vFeatureNameC.size()); // number of Features
    }
    NcVar indToFeatureVar;
    if (vFeatureNameC.size()) {
        indToFeatureVar = nc.addVar((string("indToFeature") + sDataIndex).c_str(), "string", (string("indToFeatureDim") + sDataIndex).c_str()); // ind to Feature mapping
    }
    if (vFeatureNameC.size()) {
        indToFeatureVar.putVar(vector<size_t>(1, 0), vector<size_t>(1, mFeatureNameToIndex.size()),
                        vFeatureNameC.data());
    }

    unsigned long long numberSamples = 0;
    for (int i = 0; i < vInputSamples.size(); i++) {
        numberSamples += vInputSamples[i].size();
    }

    if (numberSamples) {
        vector<unsigned int> vSparseInputStart(vInputSamples.size());
        vector<unsigned int> vSparseInputEnd(vInputSamples.size());
        vector<unsigned int> vSparseInputIndex(0), vSparseInputTime(0);
        for (int i = 0; i < vInputSamples.size(); i++) {
            vSparseInputStart[i] = (unsigned int) vSparseInputIndex.size();
            for (int j = 0; j < vInputSamples[i].size(); j++) {
                vSparseInputIndex.push_back(vInputSamples[i][j]);
                if (vInputSamplesTime.size() && vInputSamplesTime[i].size()) {
                    vSparseInputTime.push_back(vInputSamplesTime[i][j]);
                    minDate = min(minDate, (int) vInputSamplesTime[i][j]);
                    maxDate = max(maxDate, (int) vInputSamplesTime[i][j]);
                }
            }
            vSparseInputEnd[i] = (unsigned int) vSparseInputIndex.size();
        }

        vector<float> vSparseData(vSparseInputIndex.size(), 1.f); // fill it with one
        if (vInputSamplesData.size()) { // if input data is specified
            int cnt = 0;
            for (int c = 0; c < vInputSamplesData.size(); c++) {
                const vector<float>& inputData = vInputSamplesData[c];
                for (int i = 0; i < inputData.size(); i++) {
                    vSparseData[cnt] = inputData[i];
                    cnt++;
                }
            }
        }
        cout << vSparseInputIndex.size() << " total input data points." << endl;
        cout << "write " << dataName << " " << sDataIndex << endl;

        unsigned int width = 0;

        if (featureDimensionality > 0 && mFeatureNameToIndex.size() == 0) { // special case when there is no index to Feature mapping
            width = featureDimensionality;
        } else {
            width = (unsigned int) mFeatureNameToIndex.size();
        }

        width = (alignFeatureDimensionality) ? align(width) : width;
        nc.putAtt((string("name") + sDataIndex).c_str(), dataName.c_str());
        if (vInputSamplesData.size()) {
            nc.putAtt((string("attributes") + sDataIndex).c_str(), ncUint, 1); // NNDataSetBase::Attributes::Sparse = 1
            nc.putAtt((string("kind") + sDataIndex).c_str(), ncUint, 0); // NNDataSetBase::Kind::Numeric = 0
            nc.putAtt((string("dataType") + sDataIndex).c_str(), ncUint, 4); // NNDataSetBase::DataType::Float = 4
        } else {
            nc.putAtt((string("attributes") + sDataIndex).c_str(), ncUint, 3); // Sparse(1) + Boolean(2)
            nc.putAtt((string("kind") + sDataIndex).c_str(), ncUint, 0); // NNDataSetBase::Kind::Numeric = 0
            nc.putAtt((string("dataType") + sDataIndex).c_str(), ncUint, 0); // UInt = 0,
        }

        nc.putAtt((string("dimensions") + sDataIndex).c_str(), ncUint, 1);
        nc.putAtt((string("width") + sDataIndex).c_str(), ncUint, width);
        NcDim examplesDim = nc.addDim((string("examplesDim") + sDataIndex).c_str(), vSparseInputStart.size()); // number of training samples
        NcDim sparseDataDim = nc.addDim((string("sparseDataDim") + sDataIndex).c_str(), vSparseInputIndex.size()); // number of all purchases
        NcVar sparseStartVar = nc.addVar((string("sparseStart") + sDataIndex).c_str(), "uint", (string("examplesDim") + sDataIndex).c_str());
        NcVar sparseEndVar = nc.addVar((string("sparseEnd") + sDataIndex).c_str(), "uint", (string("examplesDim") + sDataIndex).c_str());
        NcVar sparseIndexVar = nc.addVar((string("sparseIndex") + sDataIndex).c_str(), "uint", (string("sparseDataDim") + sDataIndex).c_str());
        NcVar sparseTimeVar;
        if (vSparseInputTime.size()) {
            sparseTimeVar = nc.addVar((string("sparseTime") + sDataIndex).c_str(), "uint", (string("sparseDataDim") + sDataIndex).c_str());
        }

        NcVar sparseDataVar;
        if (vInputSamplesData.size()) {
            sparseDataVar = nc.addVar((string("sparseData") + sDataIndex).c_str(), ncFloat, sparseDataDim);
        }

        sparseStartVar.putVar(&vSparseInputStart[0]);
        sparseEndVar.putVar(&vSparseInputEnd[0]);
        sparseIndexVar.putVar(&vSparseInputIndex[0]); // all purchases for all samples
        if (vSparseInputTime.size()) {
            sparseTimeVar.putVar(&vSparseInputTime[0]); // all time purchases for all samples
        }
        if (vInputSamplesData.size()) {
            sparseDataVar.putVar(&vSparseData[0]);
        }
        return true;
    } else {
        return false;
    }
}

// read index to feature name mapping
void readNetCDFindToFeature(const string& fname, const int n, vector<string>& vFeaturesStr) {
    NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        cout << "Error opening binary output file " << fname << endl;
        return;
    }

    string nstring = to_string((long long) n);
    vFeaturesStr.clear();

    NcDim indToFeatureDim = nc.getDim(string("indToFeatureDim") + nstring);
    if (indToFeatureDim.isNull()) { // it is not supported by pipeline, so no exception raise
        cout << "reading error indToFeatureDim" << endl;
        return;
    }

    NcVar indToFeatureVar = nc.getVar(string("indToFeature") + nstring);
    if (indToFeatureVar.isNull()) {
        cout << "reading error indToFeature" << endl;
        return;
    }

    vector<char*> vFeaturesChars;
    vFeaturesChars.resize(indToFeatureDim.getSize());
    indToFeatureVar.getVar(&vFeaturesChars[0]);
    vFeaturesStr.resize(indToFeatureDim.getSize());
    for (int i = 0; i < vFeaturesStr.size(); i++) {
        vFeaturesStr[i] = vFeaturesChars[i];
    }
}

// read samples name(id)
void readNetCDFsamplesName(const string& fname, vector<string>& vSamplesName) {

    NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        cout << "Error opening binary output file " << fname << endl;
        return;
    }

    vSamplesName.clear();

    NcDim samplesDim = nc.getDim("samplesDim");
    if (samplesDim.isNull()) { // it is not supported by pipeline, so no exception raise
        cout << "reading error examplesDim" << endl;
        return;
    }
    NcVar sparseSamplesVar = nc.getVar("samples");
    if (sparseSamplesVar.isNull()) { // it is not supported by pipeline, so no exception raise
        cout << "reading error sparseSamplesVar" << endl;
        return;
    }
    vector<char*> vSamplesChars;

    vSamplesChars.resize(samplesDim.getSize());
    vSamplesName.resize(samplesDim.getSize());
    sparseSamplesVar.getVar(&vSamplesChars[0]);
    for (int i = 0; i < vSamplesChars.size(); i++) {
        vSamplesName[i] = vSamplesChars[i];
    }
}

void writeNETCDF(const string& fileName, const vector<string>& vSamplesName,
                const map<string, unsigned int>& mInputFeatureNameToIndex, vector<vector<unsigned int> >& vInputSamples,
                const vector<vector<unsigned int> >& vInputSamplesTime, vector<vector<float> >& vInputSamplesData,
                const map<string, unsigned int>& mOutputFeatureNameToIndex, const vector<vector<unsigned int> >& vOutputSamples,
                const vector<vector<unsigned int> >& vOutputSamplesTime,
                const vector<vector<float> >& vOutputSamplesData, int& minInpDate, int& maxInpDate, int& minOutDate,
                int& maxOutDate, const bool alignFeatureDimensionality, const int datasetNum) {

    // Write NetCDF data set files
    NcFile nc(fileName, NcFile::replace);
    if (nc.isNull()) {
        cout << "Error opening binary output file" << endl;
        exit(2);
    }

    int countData = 0;
    // add features
    if (datasetNum >= 1) {
        // input data always has index zero
        if (addDataToNetCDF(nc, 0, "input", mInputFeatureNameToIndex, vInputSamples, vInputSamplesTime, vInputSamplesData,
                        alignFeatureDimensionality, minInpDate, maxInpDate)) {
            countData++;
        } else {
            cout << "failed to write input data";
            exit(1);
        }
    }
    if (datasetNum >= 2) {
        // output data always has index one
        if (addDataToNetCDF(nc, 1, "output", mOutputFeatureNameToIndex, vOutputSamples, vOutputSamplesTime, vOutputSamplesData,
                        alignFeatureDimensionality, minOutDate, maxOutDate)) {
            countData++;
        } else {
            cout << "failed to write output data";
            exit(1);
        }
    } else {
        cout << "number of data sets datasetNum " << datasetNum << " is not implemented";
        exit(1);
    }
    nc.putAtt("datasets", ncUint, countData);

    // add samples id
    vector<const char*> vSamplesChars(vSamplesName.size());
    for (int i = 0; i < vSamplesChars.size(); i++) {
        vSamplesChars[i] = &(vSamplesName[i])[0];
    }

    NcDim samplesDim = nc.addDim("samplesDim", vSamplesName.size()); // number of training samples (it has to be the same for input and output data sets)
    NcVar sparseSamplesVar = nc.addVar("samples", "string", "samplesDim");
    sparseSamplesVar.putVar(vector<size_t>(1, 0), vector<size_t>(1, vSamplesChars.size()), vSamplesChars.data());
}
