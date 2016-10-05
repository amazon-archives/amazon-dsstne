/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <cerrno>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>

#include "Utils.h"

using namespace std;

void CWMetric::updateMetrics(string metric, string value)
{
//Plugin to Plot Cloudwatch Metrics from the EC2 Instance so that as you cans see the training Progress
 /*
    static const string executable = "/usr/bin/cw_monitoring.py";
    ifstream f(executable.c_str());
    if ( f.good())
    {
        stringstream command ;
        command <<"python "<< executable << " --metric  "<< metric << " --value "<< value;
        std::cout << "Executing "<< command.str()<< std::endl;
        system(command.str().c_str());
    }
*/
}

void CWMetric::updateMetrics(string metric, int value)
{
    stringstream sValue;
    sValue << value;
    return CWMetric::updateMetrics(metric, sValue.str());

}

void CWMetric::updateMetrics(string metric, unsigned int value)
{
    stringstream sValue;
    sValue << value;
    return CWMetric::updateMetrics(metric, sValue.str());

}

void CWMetric::updateMetrics(string metric, double value)
{
    stringstream sValue;
    sValue << value;
    return CWMetric::updateMetrics(metric, sValue.str());
}

void CWMetric::updateMetrics(string metric, size_t value)
{
    stringstream sValue;
    sValue << value;
    return CWMetric::updateMetrics(metric, sValue.str());
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return find(begin, end, option) != end;
}

/**
 * Helper function to return the value of a given argument. If it isn't supplied, errors out.
 */
string getRequiredArgValue(int argc, char** argv, string flag, string message, void (*usage)()) 
{
    if(!cmdOptionExists(argv, argv+argc, flag))
    {
        std::cout << "Error: Missing required argument: " << flag << ": " << message << std::endl;
        usage();
        exit(1);
    } 
    else 
    {
        return string(getCmdOption(argv, argv + argc, flag));
    } 
}

/**
 * Helper function to return the value of a given argument. If it isn't supplied, return the defaultValue.
 */
string getOptionalArgValue(int argc, char** argv, string flag, string defaultValue) 
{
    if(!cmdOptionExists(argv, argv+argc, flag))
    {
        return defaultValue;
    } 
    else 
    {
        return string(getCmdOption(argv, argv + argc, flag));
    } 
}

/**
 * Returns true if the argument flag is defined/set.
 */
bool isArgSet(int argc, char** argv, string flag) {
    return cmdOptionExists(argv, argv+argc, flag);
}


/*
This is an utility function which checks the file actually exist
*/
bool fileExists(const std::string& fileName)
{
    ifstream stream(fileName.c_str());
    if(stream.good()) {
        return true;
    } else {
        return false;
    }
}

/**
 * Currently we simply use the file extension. Ideally we should use something like the unix 
 * file command to determine specific file type.
 */
bool isNetCDFfile(const string &filename) 
{
    size_t extIndex = filename.find_last_of(".");
    if (extIndex == string::npos) {
        return false;
    }

    string ext = filename.substr(extIndex);
    return (ext.compare(NETCDF_FILE_EXTENTION) == 0);
}

/*
This is the splitter which is used to split a  string
which is used majorly for splitting our data sets
*/
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void forceClearVector(vector<unsigned int> &vectorToClear)
{
    vectorToClear.clear();
    vector<unsigned int>(vectorToClear).swap(vectorToClear);
}

void forceClearVector(vector<float> &vectorToClear)
{
    vectorToClear.clear();
    vector<float>(vectorToClear).swap(vectorToClear);
}

double elapsed_time(timeval x, timeval y)
{
    const int uSecondsInSeconds = 1000000;

    timeval result;
    /* Perform the carry for the later subtraction by updating y. */
    if (x.tv_usec < y.tv_usec) {
        int nsec = (y.tv_usec - x.tv_usec) / uSecondsInSeconds + 1;
        y.tv_usec -= uSecondsInSeconds * nsec;
        y.tv_sec += nsec;
    }
    if (x.tv_usec - y.tv_usec > uSecondsInSeconds) {
        int nsec = (x.tv_usec - y.tv_usec) / uSecondsInSeconds;
        y.tv_usec += uSecondsInSeconds * nsec;
        y.tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_usec is certainly positive. */
    result.tv_sec = x.tv_sec - y.tv_sec;
    result.tv_usec = x.tv_usec - y.tv_usec;
    return (double) result.tv_sec + ((double) result.tv_usec / (double) uSecondsInSeconds);
}

bool isDirectory(const string &dirname) {
    struct stat buf;
    stat(dirname.c_str(), &buf);
    return S_ISDIR(buf.st_mode);
}

bool isFile(const string &filename) {
    struct stat buf;
    stat(filename.c_str(), &buf);
    return S_ISREG(buf.st_mode);
}

int listFiles(const string &dirname, const bool recursive, vector<string> &files) {
    if (isFile(dirname)) {
        files.push_back(dirname);
    } else if (isDirectory(dirname)) {
        DIR *dp;
        struct dirent *dirp;

        if ((dp = opendir(dirname.data())) != NULL) {
            string normalizedDirname = (dirname[dirname.length() - 1] != '/') ? dirname + "/" : dirname;

            while ((dirp = readdir(dp)) != NULL) {
                char *relativeChildFilePath = dirp->d_name;
                if (strcmp(relativeChildFilePath, ".") == 0 || strcmp(relativeChildFilePath, "..") == 0) {
                    continue;
                }
                string absoluteChildFilePath = normalizedDirname + relativeChildFilePath;

                if (recursive && isDirectory(absoluteChildFilePath)) {
                    listFiles(absoluteChildFilePath, recursive, files);
                } else {
                    files.push_back(absoluteChildFilePath);
                }
            }
	    closedir(dp);
        } else {
            std::cerr << "Error(" << errno << ") opening " << dirname << std::endl;
            return errno;
        }
    } else {
        return 1;
    }
    std::sort(files.begin(), files.end());
    return 0;
}


template<typename Tkey, typename Tval>
bool cmpFirst(const pair<Tkey, Tval>& left, const pair<Tkey, Tval>& right) {
  if (left.first > right.first) { // biggest comes first
    return true;
  } else {
    return false;
  }
}

template<typename Tkey, typename Tval>
bool cmpSecond(const pair<Tkey, Tval>& left, const pair<Tkey, Tval>& right) {
  if (left.second > right.second) { // biggest comes first
    return true;
  } else {
    return false;
  }
}

template<typename Tkey, typename Tval>
void topKsort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey) {
  if (!keys || !topKkeys || !topKvals) {
    cout << "null input array" << endl;
    exit(0);
  }
  vector<pair<Tkey, Tval> > data(size);
  if (vals) {
    for (int i = 0; i < size; i++) {
      data[i].first = keys[i];
      data[i].second = vals[i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      data[i].first = keys[i];
      data[i].second = i;
    }
  }

  if (sortByKey) {
    std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpFirst<Tkey, Tval>);
    std::sort(data.begin(), data.begin() + topK, cmpFirst<Tkey, Tval>);
  } else {
    std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpSecond<Tkey, Tval>);
    std::sort(data.begin(), data.begin() + topK, cmpSecond<Tkey, Tval>);
  }
  for (int i = 0; i < topK; i++) {
    topKkeys[i] = data[i].first;
    topKvals[i] = data[i].second;
  }
}

template
void topKsort<float, unsigned int>(float*, unsigned int*, const int, float*, unsigned int*, const int, const bool);

template
void topKsort<float, float>(float*, float*, const int, float*, float*, const int, const bool);

