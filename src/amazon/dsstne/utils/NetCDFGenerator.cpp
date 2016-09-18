/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <sys/time.h>

#include "NetCDFhelper.h"
#include "Utils.h"

using namespace std;

string DATASET_TYPE_INDICATOR("indicator");
string DATASET_TYPE_ANALOG("analog");

void printUsageNetCDFGenerator() {
    cout << "NetCDFGenerator: Converts a text dataset file into a more compressed NetCDF file." << endl;
    cout <<
    "Usage: generateNetCDF -d <dataset_name> -i <input_text_file> -o <output_netcdf_file> -f <features_index> -s <samples_index> [-c] [-m]" <<
    endl;
    cout << "    -d dataset_name: (required) name for the dataset within the netcdf file." << endl;
    cout << "    -i input_text_file: (required) path to the input text file with records in data format." << endl;
    cout << "    -o output_netcdf_file: (required) path to the output netcdf file that we generate." << endl;
    cout << "    -f features_index: (required) path to the features index file to read-from/write-to." << endl;
    cout << "    -s samples_index: (required) path to the samples index file to read-from/write-to." << endl;
    cout <<
    "    -m : if set, we'll merge the feature index with new features found in the input_text_file. (Cannot be used with -c)." <<
    endl;
    cout << "    -c : if set, we'll create a new feature index from scratch. (Cannot be used with -m)." << endl;
    cout <<
    "    -t type: (default = 'indicator') the type of dataset to generate. Valid values are: ['indicator', 'analog']." <<
    endl;
    cout << endl;
}

int main(int argc, char **argv) {
    if (isArgSet(argc, argv, "-h")) {
        printUsageNetCDFGenerator();
        exit(1);
    }
    // Check + fetch required arguments
    string inputFile = getRequiredArgValue(argc, argv, "-i", "input text file to convert.", &printUsageNetCDFGenerator);
    string outputFile = getRequiredArgValue(argc, argv, "-o", "output netcdf file to generate.", &printUsageNetCDFGenerator);
    string datasetName = getRequiredArgValue(argc, argv, "-d", "dataset name for the netcdf metadata.", &printUsageNetCDFGenerator);
    string featureIndexFile = getRequiredArgValue(argc, argv, "-f", "feature index file.", &printUsageNetCDFGenerator);
    string sampleIndexFile = getRequiredArgValue(argc, argv, "-s", "samples index file.", &printUsageNetCDFGenerator);

    bool createFeatureIndex = isArgSet(argc, argv, "-c");
    if (createFeatureIndex) {
        cout << "Flag -c is set. Will create a new feature file and overwrite: " << featureIndexFile << endl;
    }

    bool mergeFeatureIndex = isArgSet(argc, argv, "-m");
    if (mergeFeatureIndex) {
        cout << "Flag -m is set. Will merge with existing feature file and overwrite: " << featureIndexFile << endl;
    }

    if (createFeatureIndex && mergeFeatureIndex) {
        cout << "Error: Cannot create (-c) and update existing (-u) feature index. Please select only one.";
        printUsageNetCDFGenerator();
        exit(1);
    }
    bool updateFeatureIndex = createFeatureIndex || mergeFeatureIndex;

    string dataType = getOptionalArgValue(argc, argv, "-t", "indicator");
    if (dataType.compare(DATASET_TYPE_INDICATOR) != 0 && dataType.compare(DATASET_TYPE_ANALOG) != 0) {
        cout << "Error: Unknown dataset type [" << dataType << "].";
        cout << " Please select one of {" << DATASET_TYPE_INDICATOR << "," << DATASET_TYPE_ANALOG << "}" << endl;
        exit(1);
    }
    cout << "Generating dataset of type: " << dataType << endl;

    // maps for feature and samples index.
    unordered_map<string, unsigned int> mFeatureIndex;
    unordered_map<string, unsigned int> mSampleIndex;

    // Start timing
    timeval timeStart;
    gettimeofday(&timeStart, NULL);


    // If sampleIndexFile exists, load it, otherwise we'll create it when we finish
    if (!fileExists(sampleIndexFile)) {
        cout << "Will create a new samples index file: " << sampleIndexFile << endl;
    } else {
        cout << "Loading sample index from: " << sampleIndexFile << endl;
        if (!loadIndexFromFile(mSampleIndex, sampleIndexFile, cout)) {
            exit(1);
        }
    }

    if (createFeatureIndex) {
        cout << "Will create a new features index file: " << featureIndexFile << endl;
    } else if (!fileExists(featureIndexFile)) {
        cout << "Error: Cannnot find a valid feature index file: " << featureIndexFile << endl;
        exit(1);
    } else {
        cout << "Loading feature index from: " << featureIndexFile << endl;
        if (!loadIndexFromFile(mFeatureIndex, featureIndexFile, cout)) {
            exit(1);
        }
    }

    // Generate a sparse matrix from inputFile.
    vector<unsigned int> vSparseStart;
    vector<unsigned int> vSparseEnd;
    vector<unsigned int> vSparseIndex;
    vector<float> vSparseData;


    // collects indices into the provided index maps, and writes them to a file if updated
    if (!generateNetCDFIndexes(inputFile,
                          updateFeatureIndex,
                          featureIndexFile,
                          sampleIndexFile,
                          mFeatureIndex,
                          mSampleIndex,
                          vSparseStart,
                          vSparseEnd,
                          vSparseIndex,
                          vSparseData,
                          cout)) {
        exit(1);
    }


    if (dataType.compare(DATASET_TYPE_ANALOG) == 0) {
        writeNetCDFFile(vSparseStart,
                        vSparseEnd,
                        vSparseIndex,
                        vSparseData,
                        outputFile,
                        datasetName,
                        mFeatureIndex.size());
    } else {
        // Default type is to assume indicator, so we don't retain the data values in the NetCDF file.
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputFile, datasetName, mFeatureIndex.size());
    }

    timeval timeEnd;
    gettimeofday(&timeEnd, NULL);
    cout << "Total time for generating NetCDF: " << elapsed_time(timeEnd, timeStart) << " secs. " << endl;
}
