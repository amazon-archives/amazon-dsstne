/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <iosfwd>
#include <string>
#include <vector>
#include <unordered_map>

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
 * Updates the index information from the samplesFile into the referenced data structures.
 * Use this along with exportIndex(mLabelToIndex, indexFileName).
 * featureIndexUpdated and sampleIndexUpdated will be true iff feature and sample indices have been updated (respectively)
 *
 */
void indexFile(const std::string &samplesPath,
               const bool enableFeatureIndexUpdates,
               std::unordered_map<std::string, unsigned int> &mFeatureIndex,
               std::unordered_map<std::string, unsigned int> &mSampleIndex,
               bool &featureIndexUpdated,
               bool &sampleIndexUpdated,
               std::vector<unsigned int> &vSparseStart,
               std::vector<unsigned int> &vSparseEnd,
               std::vector<unsigned int> &vSparseIndex,
               std::vector<float> &vSparseData);

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
void generateNetCDFIndexes(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const std::string &outFeatureIndexFileName,
                           const std::string &outSampleIndexFileName,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData);

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
