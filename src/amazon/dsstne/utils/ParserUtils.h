/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#pragma once
#include <vector>
#include <string>
#include <map>
#include <set>
#include "sys/time.h"

using std::pair;
using std::vector;
using std::string;
using std::map;
using std::pair;
using std::set;

void writeTXT(const string& prefix, map<string, unsigned int>& mFeature, vector<string>& vCustomerName,
    const vector<vector<unsigned int> >& vCustomerInput, const vector<vector<unsigned int> >& vCustomerInputTime,
    const vector<vector<unsigned int> >& vCustomerOutput, const vector<vector<unsigned int> >& vCustomerOutputTime);

template<typename T>
void writeVector(const string& fname, const vector<T>& distribution, const int offset = 0);

void writeFeatureFrequency(const string& fname, const std::map<std::string, int>& mFeatureToFrequency);
void clipFeatureFrequency(const std::map<std::string, int>& mFeatureToFrequency, const double limit, map<string, unsigned int>& mFeature);

void writeDates(const string& fname, const int min_date_inp, const int max_date_inp, const int min_date_out,
    const int max_date_out);

void splitString(const std::string& str, const std::string& delimiter, std::vector<std::string>& line_strip);

void writeNETCDF(const string& fileName, const vector<string>& vCustomerName,
    const map<string, unsigned int>& mFeatureInput, const vector<vector<unsigned int> >& vCustomerInput, 
    const vector<vector<unsigned int> >& vCustomerInputTime, const vector<vector<float> >& vCustomerInputData,
    const map<string, unsigned int>& mFeatureOutput, const vector<vector<unsigned int> >& vCustomerOutput, 
    const vector<vector<unsigned int> >& vCustomerOutputTime, const vector<vector<float> >& vCustomerOutputData,
    const int local_layers, int& min_inp_date, int& max_inp_date, int& min_out_date, int& max_out_date, const bool align_feature_number = true);

void verifyNETCDF(const string& fileName, const map<string, unsigned int>& mFeature, const vector<string>& vCustomerName,
    const vector<vector<unsigned int> >& vCustomerInput, const vector<vector<unsigned int> >& vCustomerInputTime,
    const vector<vector<unsigned int> >& vCustomerOutput, const vector<vector<unsigned int> >& vCustomerOutputTime);

void readNETCDF(const string& fileName, vector<std::string>& vFeaturesStr, vector<string>& vCustomerName,
    vector<unsigned int>& vSparseInputStart, vector<unsigned int>& vSparseInputEnd,
    vector<unsigned int>& vSparseInputIndex, vector<unsigned int>& vSparseInputTime,
    vector<unsigned int>& vSparseOutputStart, vector<unsigned int>& vSparseOutputEnd,
    vector<unsigned int>& vSparseOutputIndex, vector<unsigned int>& vSparseOutputTime);

void addCustomerData(const string& customer, const set<string>& sInput, const set<string>& sOutput,
    map<string, unsigned int>& mFeature, map<string, unsigned int>& mInputTime, map<string, unsigned int>& mOutputTime,
    vector<vector<unsigned int> >& vCustomerInput, vector<vector<unsigned int> >& vCustomerInputTime,
    vector<vector<unsigned int> >& vCustomerOutput, vector<vector<unsigned int> >& vCustomerOutputTime,
    vector<string>& vCustomerName, const bool clip);

void addFEATUREvec(const string& s, map<string, unsigned int>& mFeature, map<string, unsigned int>& mTime,
    vector<unsigned int>& vFeatures, vector<unsigned int>& vTime, const bool clip);

void addFEATUREset(const string& feature, const unsigned int date, set<string>& sFeatures, map<string, unsigned int>& mTime,
    map<string, unsigned int>& mFeature, const bool clip);

int writeFeatureToInd(const string& fname, const map<string, unsigned int>& mFeature);

void readNetCDFindToFeature(const string& fname, const int n, vector<string>& vFeaturesStr);

void readNetCDFcustomers(const string& fname, const int n, vector<string>& vCustomerStr);

void checkFile(const std::ofstream& file, const string& fname);

void generateRegressionData(const string& FLAGS_path);

void generateClassificationData(const string& FLAGS_path);
