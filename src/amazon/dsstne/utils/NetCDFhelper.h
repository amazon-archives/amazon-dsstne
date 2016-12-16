/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#pragma once

#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <netcdf>

/**
 * Loads an index from the given input stream, assuming an entry on each line with a 
 * tab separating label and index. Used for feature and sample indices for a dataset.

 * This function performs additional error checks to ensure that the number of lines
 * processed matches the number of entries added to the index. This is will detect
 * data corruption issues, but will not necessarily identify the exact line that
 * is causing the issue.
 *
 * @param labelsToIndices  unordered_map into which new entries will be inserted
 * @param inputStream      input stream from which to read raw data
 * @param outputStream     output stream to be used for any status or error messages
 *
 * @return \c true if all input is processed successfully; \c false otherwise
 */
bool loadIndex(std::unordered_map<std::string, unsigned int> &mLabelToIndex, std::istream &inputStream,
               std::ostream &outputStream);

/**
 * Loads an index from the given input file, assuming an entry on each line with a
 * tab separating label and index. Used for feature and sample indices for a dataset.
 *
 * Error checking is as described for the loadIndex() function.
 *
 * @param labelsToIndices  unordered_map into which new entries will be inserted
 * @param inputFile        path to file from which to read raw data
 * @param outputStream     output stream to be used for any status or error messages
 *
 * @return  \c true if the entire input file was read successfully; \c false otherwise
 */
bool loadIndexFromFile(std::unordered_map<std::string, unsigned int> &labelsToIndices, const std::string &inputFile,
                       std::ostream &outputStream);

/**
 * Exports an index to the given indexFileName files, writing an entry to each line with a 
 * tab separating label and index. Used for feature and sample indices for a dataset.
 */
void exportIndex(std::unordered_map<std::string, unsigned int> &mLabelToIndex, std::string indexFileName);

/**
 * Parse sample data from the given input stream, and update the referenced sample/signal and
 * sample/signal-value data structures.
 *
 * Data imported into the sample/signal and sample/signal-value data structures can later be 
 * used to seed or update a sparse data index, appropriate for generating NetCDF files.
 *
 * @see importSamplesFromPath() for more documentation about return variables
 *
 * @return  \c true if all input is processed successfully; \c false otherwise
 */
bool parseSamples(std::istream &inputStream,
                  const bool enableFeatureIndexUpdates,
                  std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                  std::unordered_map<std::string, unsigned int> &mSampleIndex,
                  bool &featureIndexUpdated,
                  bool &sampleIndexUpdated,
                  std::map<unsigned int, std::vector<unsigned int>> &mSignals,
                  std::map<unsigned int, std::vector<float>> &mSignalValues,
                  std::ostream &outputStream);

/**
 * Import samples from a given file or directory, and update the referenced data structures.
 *
 * Use this along with exportIndex(mLabelToIndex, indexFileName) to convert a raw data
 * to NetCDF format.
 *
 * The output variables featureIndexUpdated and sampleIndexUpdated will be true iff feature
 * and sample indices have been updated (respectively)
 *
 * If enableFeatureIndexUpdates is set, the existing feature index will be updated with any
 * new entries found. Otherwise only the samples index will be updated.
 *
 * @return  \c true if the all input files were read successfully; \c false otherwise
 */
bool importSamplesFromPath(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           bool &featureIndexUpdated,
                           bool &sampleIndexUpdated,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream);

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
 * @param outputStream - output stream to be used for any status or error messages.
 *
 * @return  \c true if the all input files were read successfully; \c false otherwise
 */
bool generateNetCDFIndexes(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const std::string &outFeatureIndexFileName,
                           const std::string &outSampleIndexFileName,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream);

/**
 * Writes an NetCDFfile for a given sparse matrix of indices and values (start of sample, end of sample, samples array) for each sample.
 * The dataset within the file is indexed with dataset name. Note that maxFeatureIndex is the rounded up to multiple of 32.
 */
void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::vector<float> &vSparseValue,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * Writes an NetCDFfile for a given sparse matrix of indices only (start of sample, end of sample, samples array) for each sample.
 * The dataset within the file is indexed with dataset name. Note that maxFeatureIndex is the rounded up to multiple of 32.
 */
void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * Rounds up the index to take advantage of aligned memory addressing.
 */
unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex);

/**
 * Returns the absolute paths of all files in the directory. If the directory is a file, then returns a singleton
 * of the file itself.
 */
int listFiles(const std::string &dirname, const bool recursive, std::vector<std::string> &files);

/**
 * Writes self contained netcdf. It includes input/output features, feature map, samples id
 */
void writeNETCDF(const std::string& fileName, const std::vector<std::string>& vSamplesName,
                const std::map<std::string, unsigned int>& mInputFeatureNameToIndex, std::vector<std::vector<unsigned int> >& vInputSamples,
                const std::vector<std::vector<unsigned int> >& vInputSamplesTime, std::vector<std::vector<float> >& vInputSamplesData,
                const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex, const std::vector<std::vector<unsigned int> >& vOutputSamples,
                const std::vector<std::vector<unsigned int> >& vOutputSamplesTime,
                const std::vector<std::vector<float> >& vOutputSamplesData, int& minInpDate, int& maxInpDate,
                int& minOutDate, int& maxOutDate, const bool alignFeatureDimensionality, const int datasetNum);

/**
 * Reads samples id of "input" data
 */
void readNetCDFsamplesName(const std::string& fname, std::vector<std::string>& vSamplesName);

/**
 * Reads index to feature map of data with index "n"
 */
void readNetCDFindToFeature(const std::string& fname, const int n, std::vector<std::string>& vFeaturesStr);

/**
 * Align feature size
 */
unsigned int align(size_t size);

/**
 * Add data to netcdf
 */
bool addDataToNetCDF(netCDF::NcFile& nc, const long long dataIndex, const std::string& dataName,
                const std::map<std::string, unsigned int>& mFeatureNameToIndex,
                const std::vector<std::vector<unsigned int> >& vInputSamples,
                const std::vector<std::vector<unsigned int> >& vInputSamplesTime, const std::vector<std::vector<float> >& vInputSamplesData,
                const bool alignFeatureDimensionality, int& minDate, int& maxDate, const int featureDimensionality = -1);
