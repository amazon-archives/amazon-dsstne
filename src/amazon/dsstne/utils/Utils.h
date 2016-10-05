/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#pragma once
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <map>

using std::string;
using std::vector;

const string INPUT_DATASET_SUFFIX = "_input";
const string OUTPUT_DATASET_SUFFIX = "_output";
const string NETCDF_FILE_EXTENTION = ".nc";
const unsigned long FIXED_SEED = 12134ull;

class CWMetric
{
public:
    static void updateMetrics(string metric, string value);
    static void updateMetrics(string metric, int value);
    static void updateMetrics(string metric, unsigned int value);
    static void updateMetrics(string metric, double value);
    static void updateMetrics(string metric, size_t value);
};

char* getCmdOption(char ** , char **, const std::string & );

bool cmdOptionExists(char** , char**, const std::string& );

/**
 * Helper function to return the value of a given argument. If it isn't supplied, errors out.
 * @param argc has count of arguments
 * @param argv the command line arguments.
 * @param flag for the argument of interest.
 * @param message message in event the argument is not defined.
 * @param usage callback error function to call and exit with error code 1.
 */
string getRequiredArgValue(int argc, char** argv, string flag, string message, void (*usage)());

/**
 * Helper function to return the value of a given argument. If it isn't supplied, return the defaultValue.
 * @param argc has count of arguments
 * @param argv the command line arguments.
 * @param flag for the argument of interest.
 * @param defaultValue to return in the event it is not overridden in the arguments.
 */
string getOptionalArgValue(int argc, char** argv, string flag, string defaultValue);

/**
 * Returns true if the argument flag is defined/set.
 * @param argc has count of arguments
 * @param argv the command line arguments.
 * @param flag for the argument of interest.
 */
bool isArgSet(int argc, char** argv, string flag);

bool fileExists(const std::string &);

/**
 * Return true if the file is a NetCDF file. 
 */
bool isNetCDFfile(const string &filename);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);

/**
 * Uses the swap technique to force clearing of memory allocated to the vector.
 * TODO: Templatize this function so that it can be used with ANY vector types.
 */
void forceClearVector(vector<unsigned int> &vectorToClear);
void forceClearVector(vector<float> &vectorToClear);

double elapsed_time(timeval x, timeval y);

/**
 * Returns true iff dirname is a directory
 */
bool isDirectory(const string &dirname);

/**
 * Returns true iff filename is a file
 */
bool isFile(const string &filename);

/**
 * Adds all files (not directories) under the specified dirname into files
 * Returns 0 if success and non-zero otherwise.
 * If the recursive flag is not set, only lists the first level files. Otherwise, it recurses into
 * all sub-directories.
 */
int listFiles(const string &dirname, const bool recursive, vector<string> &files);

// sort top K by keys and return top keys with top values
template<typename Tkey, typename Tval>
void topKsort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey = true);


// min max - inclusive
inline int rand(int min, int max) {
  return rand() % (max - min + 1) + min;
}

inline float rand(float min, float max) {
  float r = (float)rand() / (float)RAND_MAX;
  return min + r * (max - min);
}
