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
#include <vector>
#include <map>
#include <netcdf>
#include <sys/time.h>
#include <unordered_map>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

/**
 * Loads an index from the given indexFileName files, assuming an entry on each line with a 
 * tab separating label and index. Used for feature and sample indices for a dataset.
 */
void loadIndex(unordered_map<string, unsigned int> &mLabelToIndex, string indexFileName);

/**
 * Exports an index to the given indexFileName files, writing an entry to each line with a 
 * tab separating label and index. Used for feature and sample indices for a dataset.
 */
void exportIndex(unordered_map<string, unsigned int> &mLabelToIndex, string indexFileName);

/**
 * Updates the index information from the samplesFile into the referenced data structures.
 * Use this along with exportIndex(mLabelToIndex, indexFileName).
 * featureIndexUpdated and sampleIndexUpdated will be true iff feature and sample indices have been updated (respectively)
 *
 */
void indexFile(const string &samplesPath,
               const bool enableFeatureIndexUpdates,
               unordered_map<string, unsigned int> &mFeatureIndex,
               unordered_map<string, unsigned int> &mSampleIndex,
               bool &featureIndexUpdated,
               bool &sampleIndexUpdated,
               vector<unsigned int> &vSparseStart,
               vector<unsigned int> &vSparseEnd,
               vector<unsigned int> &vSparseIndex,
               vector<float> &vSparseData);

/**
 * Generates a NetCDF index for a given dataset and exports them to respective files with 
 * specified names for for the index files. If enableFeatureIndexUpdates is set, and existing
 * feature index will be updated with any new entries found. Otherwise only samples index is updated.
 *
 * @param mFeatureIndex - map of feature to index.
 * @param mSampleIndex - map of sample id to index.
 * @param samplesFileName - the input text file with data in format: <customer_id>,<marketplace>TAB<feature,value>:<feature,value>:...
 * @param vSparseStart, vSparseEnd, vSparseIndex - returned set of indices for sparse representation of the input dataset.
 * @param enableFeatureIndexUpdates - if set, well update existing feature index with new entries.
 * @param outFeatureIndexFileName - the name of the file to export the feature index to.
 * @param outSampleIndexFileName - the name of tile to export the samples index to.
 */
void generateNetCDFIndexes(const string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const string &outFeatureIndexFileName,
                           const string &outSampleIndexFileName,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseData);

/**
 * Generates a NetCDF index for a given dataset and exports them to respective files, using 
 * the default names for feature and sample index files.
 * 
 * Same as generateNetCDFIndexes() except the default values are:
 *  - outFeatureIndexFileName = $datasetName.inputIndex
 *  - outSampleIndexFileName = $datasetName.sampleIndex
 */
void generateNetCDFIndexes(const string &samplesFileName,
                           const bool enableFeatureIndexUpdates,
                           const string &dataSetName,
                           unordered_map<string, unsigned int> &mFeatureIndex,
                           unordered_map<string, unsigned int> &mSampleIndex,
                           vector<unsigned int> &vSparseStart,
                           vector<unsigned int> &vSparseEnd,
                           vector<unsigned int> &vSparseIndex,
                           vector<float> &vSparseValue);

/**
 * Writes an NetCDFfile for a given sparse matrix of indices and values (start of sample, end of sample, samples array) for each sample.
 * The dataset within the file is indexed with dataset name. Note that maxFeatureIndex is the rounded up to multiple of 32.
 */
void writeNetCDFFile(vector<unsigned int> &vSparseStart,
                     vector<unsigned int> &vSparseEnd,
                     vector<unsigned int> &vSparseIndex,
                     vector<float> &vSparseValue,
                     string fileName,
                     string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * Writes an NetCDFfile for a given sparse matrix of indices only (start of sample, end of sample, samples array) for each sample.
 * The dataset within the file is indexed with dataset name. Note that maxFeatureIndex is the rounded up to multiple of 32.
 */
void writeNetCDFFile(vector<unsigned int> &vSparseStart,
                     vector<unsigned int> &vSparseEnd,
                     vector<unsigned int> &vSparseIndex,
                     string fileName,
                     string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * Rounds up the index to take advantage of aligned memory addressing.
 */
unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);

/**
 * Returns the absolute paths of all files in the directory. If the directory is a file, then returns a singleton
 * of the file itself.
 */
int listFiles(const string &dirname, const bool recursive, vector<string> &files);
