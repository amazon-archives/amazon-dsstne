/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <netcdf>
#include <limits.h>

#include "ParserUtils.h"
#include "NetCDFhelper.h"

using namespace netCDF;
using namespace netCDF::exceptions;
using std::cout;
using std::endl;

void checkFile(const std::ofstream& file, const string& fname) {
  if (!file.is_open()) {
    cout << "failed to open " << fname << endl;
    exit(2);
  }
}

void appendToFile(std::ofstream& file_inp, const string& inp_customer, const vector<unsigned int>& inp_features,
    const vector<unsigned int>& inp_times, const vector<std::string>& feature_ind_to_feature) {
  if (inp_features.size() != inp_times.size()) {
    cout << "inp_features.size() != inp_times.size()" << endl;
    exit(2);
  }
  string customer = inp_customer;
  // ignore market place (just print whatewer we have in text whcih is separated by \t)
  //string customer = inp_customer.substr(0, inp_customer.size() - 1);
  //string mkt = inp_customer.substr(inp_customer.size() - 1, inp_customer.size());
  //file_inp << customer << "," << mkt << "\t";
  file_inp << customer << "\t";
  for (int i = 0; i < inp_features.size(); i++) {
    int feature_ind = inp_features[i];
    string feature = feature_ind_to_feature[feature_ind];
    int feature_time = inp_times[i];
    file_inp << feature << "," << feature_time;
    if (i != inp_features.size() - 1) {
      file_inp << ":";
    }
  }
  file_inp << std::endl;
}

void writeTXT(const string& prefix, map<string, unsigned int>& mFeature, vector<string>& vCustomerName,
    const vector<vector<unsigned int> >& vCustomerInput, const vector<vector<unsigned int> >& vCustomerInputTime,
    const vector<vector<unsigned int> >& vCustomerOutput, const vector<vector<unsigned int> >& vCustomerOutputTime) {

  if (vCustomerInput.size() != vCustomerOutput.size()) {
    cout << "vCustomerInput.size() != vCustomerOutput.size()" << endl;
    exit(2);
  }
  string fname_inp = prefix + string("_inp");
  string fname_out = prefix + string("_out");
  std::ofstream file_inp(fname_inp.c_str());
  checkFile(file_inp, fname_inp);

  std::ofstream file_out(fname_out.c_str());
  checkFile(file_out, fname_out);

  vector<std::string> feature_ind_to_feature(mFeature.size());
  for (map<string, unsigned int>::iterator it = mFeature.begin(); it != mFeature.end(); it++) {
    feature_ind_to_feature[it->second] = it->first;
  }
  for (int c = 0; c < vCustomerInput.size(); c++) {
    const string customer = vCustomerName[c];

    const vector<unsigned int> &inp_features = vCustomerInput[c];
    const vector<unsigned int> &inp_times = vCustomerInputTime[c];
    appendToFile(file_inp, customer, inp_features, inp_times, feature_ind_to_feature);

    const vector<unsigned int> &out_features = vCustomerOutput[c];
    const vector<unsigned int> &out_times = vCustomerOutputTime[c];
    appendToFile(file_out, customer, out_features, out_times, feature_ind_to_feature);
  }
}

template<typename T>
void writeVector(const string& fname, const vector<T>& distribution, const int offset) {
  std::ofstream file_txt(fname.c_str());
  checkFile(file_txt, fname);
  for (int i = 0; i < distribution.size(); i++) {
    file_txt << i + offset << "\t" << distribution[i] << std::endl;
  }
}

template
void writeVector<int>(const string&, const vector<int>&, const int);

// Split string using delimiter. If output line_strip has single special character '\n', '\r', '\t' they will be ignored
inline void splitString(const std::string& str, const std::string& delimiter, std::vector<std::string>& line_strip) {
  line_strip.resize(0);
  size_t pos = 0;
  std::string token;
  std::string line = str;

  while ((pos = line.find(delimiter)) != std::string::npos) { // search for the next delimeter
    token = line.substr(0, pos);
    line_strip.push_back(token);
    line.erase(0, pos + delimiter.length());
  }
  if (line.size()) {
    if (line.size() == 1) {
      if (line[0] == '\n' || line[0] == '\r' || line[0] == '\t') {
        return;
      }
    }
    line_strip.push_back(line); // add remaining part
  }
}

inline bool cmp_frequency(const pair<string, int>& left, const pair<string, int>& right) {
  if (left.second > right.second) { // biggest comes first
    return true;
  } else {
    return false;
  }
}

void writeFeatureFrequency(const string& fname, const std::map<std::string, int>& mFeatureToFrequency) {
  vector<pair<string, int>> feature_frequency;
  for (map<std::string, int>::const_iterator iterator = mFeatureToFrequency.begin(); iterator != mFeatureToFrequency.end();
      iterator++) {
    feature_frequency.push_back(make_pair(iterator->first, iterator->second));
  }
  std::sort(feature_frequency.begin(), feature_frequency.end(), cmp_frequency);
  std::ofstream file_txt(fname.c_str());
  checkFile(file_txt, fname);
  for (int i = 0; i < feature_frequency.size(); i++) {
    file_txt << feature_frequency[i].first << " " << feature_frequency[i].second << std::endl;
  }
}

void clipFeatureFrequency(const std::map<std::string, int>& mFeatureToFrequency, const double limit, map<string, unsigned int>& mFeature) {
  vector<pair<string, int>> feature_frequency;
  for (map<std::string, int>::const_iterator iterator = mFeatureToFrequency.begin(); iterator != mFeatureToFrequency.end();
      iterator++) {
    feature_frequency.push_back(make_pair(iterator->first, iterator->second));
  }
  std::sort(feature_frequency.begin(), feature_frequency.end(), cmp_frequency);

  double totalSum = 0.f;
  for (int i = 0; i < feature_frequency.size(); i++) {
    totalSum += feature_frequency[i].second;
  }

  double limitSum = totalSum * limit / 100.f;
  float currentSum = 0.f;
  mFeature.clear(); // reset feature to index map
  for (int i = 0; i < feature_frequency.size(); i++) {
    string feature = feature_frequency[i].first;
    int frequency = feature_frequency[i].second;
    if (currentSum < limitSum && limit > 0.0) {
      mFeature[feature] = i; // create a new feature to index map for clipping features
    }
    currentSum += frequency;
  }
}

void writeDates(const string& fname, const int min_date_inp, const int max_date_inp, const int min_date_out,
    const int max_date_out) {
  std::ofstream file_txt(fname.c_str());
  checkFile(file_txt, fname);
  file_txt << "min_date_inp " << min_date_inp << std::endl;
  file_txt << "max_date_inp " << max_date_inp << std::endl;
  file_txt << "min_date_out " << min_date_out << std::endl;
  file_txt << "max_date_out " << max_date_out << std::endl;
}

void verifyNETCDF(const string& fileName, const map<string, unsigned int>& mFeature, const vector<string>& vCustomerName,
    const vector<vector<unsigned int> >& vCustomerInput, const vector<vector<unsigned int> >& vCustomerInputTime,
    const vector<vector<unsigned int> >& vCustomerOutput, const vector<vector<unsigned int> >& vCustomerOutputTime) {

  vector<std::string> vFeaturesStr_;
  vector<string> vCustomerName_;
  vector<unsigned int> vSparseInputStart_;
  vector<unsigned int> vSparseInputEnd_;
  vector<unsigned int> vSparseInputIndex_;
  vector<unsigned int> vSparseInputTime_;
  vector<unsigned int> vSparseOutputStart_;
  vector<unsigned int> vSparseOutputEnd_;
  vector<unsigned int> vSparseOutputIndex_;
  vector<unsigned int> vSparseOutputTime_;

  readNETCDF(fileName, vFeaturesStr_, vCustomerName_, vSparseInputStart_, vSparseInputEnd_, vSparseInputIndex_,
      vSparseInputTime_, vSparseOutputStart_, vSparseOutputEnd_, vSparseOutputIndex_, vSparseOutputTime_);

  vector<std::string> vFeaturesStr(mFeature.size());
  for (map<string, unsigned int>::const_iterator it = mFeature.begin(); it != mFeature.end(); it++) {
    vFeaturesStr[it->second] = it->first;
  }
  std::vector<char*> vFeaturesChars(vFeaturesStr.size());
  for (int i = 0; i < vFeaturesChars.size(); i++) {
    vFeaturesChars[i] = &(vFeaturesStr[i])[0];
  }

  std::vector<const char*> vCustomerChars(vCustomerName.size());
  for (int i = 0; i < vCustomerChars.size(); i++) {
    vCustomerChars[i] = &(vCustomerName[i])[0];
  }

  unsigned int width = align(mFeature.size());
  vector<unsigned int> vSparseInputStart(vCustomerName.size());
  vector<unsigned int> vSparseInputEnd(vCustomerName.size());
  vector<unsigned int> vSparseInputIndex(0), vSparseInputTime(0);
  vector<unsigned int> vSparseOutputStart(vCustomerName.size());
  vector<unsigned int> vSparseOutputEnd(vCustomerName.size());
  vector<unsigned int> vSparseOutputIndex(0), vSparseOutputTime(0);

  for (int i = 0; i < vCustomerName.size(); i++) {
    vSparseInputStart[i] = (unsigned int) vSparseInputIndex.size();
    for (int j = 0; j < vCustomerInput[i].size(); j++) {
      vSparseInputIndex.push_back(vCustomerInput[i][j]);
      vSparseInputTime.push_back(vCustomerInputTime[i][j]);
    }
    vSparseInputEnd[i] = (unsigned int) vSparseInputIndex.size();
  }
  cout << vSparseInputIndex.size() << " total input datapoints." << endl;

  for (int i = 0; i < vCustomerName.size(); i++) {
    vSparseOutputStart[i] = (unsigned int) vSparseOutputIndex.size();
    for (int j = 0; j < vCustomerOutput[i].size(); j++) {
      vSparseOutputIndex.push_back(vCustomerOutput[i][j]);
      vSparseOutputTime.push_back(vCustomerOutputTime[i][j]);
    }
    vSparseOutputEnd[i] = (unsigned int) vSparseOutputIndex.size();
  }

  if ((vFeaturesStr_ != vFeaturesStr) || (vCustomerName_ != vCustomerName) || (vSparseInputStart_ != vSparseInputStart) ||
      (vSparseInputEnd_ != vSparseInputEnd) || (vSparseInputIndex_ != vSparseInputIndex) ||
      (vSparseInputTime_ != vSparseInputTime) || (vSparseOutputStart_ != vSparseOutputStart) ||
      (vSparseOutputEnd_ != vSparseOutputEnd) || (vSparseOutputIndex_ != vSparseOutputIndex) ||
      (vSparseOutputTime_ == vSparseOutputTime)) {
    cout << "error comparison" << std::endl;
    exit(2);
  }
}

void readNETCDF(const string& fileName, vector<std::string>& vFeaturesStr, vector<string>& vCustomerName,
    vector<unsigned int>& vSparseInputStart, vector<unsigned int>& vSparseInputEnd,
    vector<unsigned int>& vSparseInputIndex, vector<unsigned int>& vSparseInputTime,
    vector<unsigned int>& vSparseOutputStart, vector<unsigned int>& vSparseOutputEnd,
    vector<unsigned int>& vSparseOutputIndex, vector<unsigned int>& vSparseOutputTime) {

  // Read back written data
  NcFile nc(fileName, NcFile::read);
  if (nc.isNull()) {
    cout << "Error opening binary output file" << endl;
    exit(2);
  }
  {
    NcGroupAtt datasetsAtt = nc.getAtt("datasets");
    NcGroupAtt name0Att = nc.getAtt("name0");
    NcGroupAtt attributes0Att = nc.getAtt("attributes0");
    NcGroupAtt kind0Att = nc.getAtt("kind0");
    NcGroupAtt dataType0Att = nc.getAtt("dataType0");
    NcGroupAtt dimensions0Att = nc.getAtt("dimensions0");
    NcGroupAtt width0Att = nc.getAtt("width0");

    NcDim examplesDim0 = nc.getDim("examplesDim0");
    NcDim sparseDataDim0 = nc.getDim("sparseDataDim0");
    NcDim indToFeatureDim0 = nc.getDim("indToFeatureDim0");

    NcVar sparseStart0Var = nc.getVar("sparseStart0");
    NcVar sparseEnd0Var = nc.getVar("sparseEnd0");
    NcVar sparseIndex0Var = nc.getVar("sparseIndex0");

    NcVar sparseTime0Var = nc.getVar("sparseTime0");
    NcVar indToFeature0Var = nc.getVar("indToFeature0");
    NcVar sparseCustomersVar = nc.getVar("Customers");

    std::vector<char*> vFeaturesChars_;
    std::vector<char*> vCustomerChars_;

    vSparseInputStart.resize(examplesDim0.getSize());
    vSparseInputEnd.resize(examplesDim0.getSize());
    vSparseInputIndex.resize(sparseDataDim0.getSize());
    vSparseInputTime.resize(sparseDataDim0.getSize());

    vCustomerName.resize(examplesDim0.getSize());
    vFeaturesStr.resize(indToFeatureDim0.getSize());

    vCustomerChars_.resize(examplesDim0.getSize());
    vFeaturesChars_.resize(indToFeatureDim0.getSize());

    sparseStart0Var.getVar(&vSparseInputStart[0]);
    sparseEnd0Var.getVar(&vSparseInputEnd[0]);
    sparseIndex0Var.getVar(&vSparseInputIndex[0]);
    sparseTime0Var.getVar(&vSparseInputTime[0]);

    indToFeature0Var.getVar(&vFeaturesChars_[0]);
    sparseCustomersVar.getVar(&vCustomerChars_[0]);

    for (int i = 0; i < vFeaturesStr.size(); i++) {
      vFeaturesStr[i] = vFeaturesChars_[i];
    }

    for (int i = 0; i < vCustomerChars_.size(); i++) {
      vCustomerName[i] = vCustomerChars_[i];
    }
  }

  {
    NcGroupAtt datasetsAtt = nc.getAtt("datasets");
    NcGroupAtt name1Att = nc.getAtt("name1");
    NcGroupAtt attributes1Att = nc.getAtt("attributes1");
    NcGroupAtt kind1Att = nc.getAtt("kind1");
    NcGroupAtt dataType1Att = nc.getAtt("dataType1");
    NcGroupAtt dimensions1Att = nc.getAtt("dimensions1");
    NcGroupAtt width1Att = nc.getAtt("width1");

    NcDim examplesDim1 = nc.getDim("examplesDim1");
    NcDim sparseDataDim1 = nc.getDim("sparseDataDim1");

    NcVar sparseStart1Var = nc.getVar("sparseStart1");
    NcVar sparseEnd1Var = nc.getVar("sparseEnd1");
    NcVar sparseIndex1Var = nc.getVar("sparseIndex1");

    NcVar sparseTime1Var = nc.getVar("sparseTime1");

    vSparseOutputStart.resize(examplesDim1.getSize());
    vSparseOutputEnd.resize(examplesDim1.getSize());
    vSparseOutputIndex.resize(sparseDataDim1.getSize());
    vSparseOutputTime.resize(sparseDataDim1.getSize());

    sparseStart1Var.getVar(&vSparseOutputStart[0]);
    sparseEnd1Var.getVar(&vSparseOutputEnd[0]);
    sparseIndex1Var.getVar(&vSparseOutputIndex[0]);
    sparseTime1Var.getVar(&vSparseOutputTime[0]);
  }
}

void addFEATUREvec(const string& s, map<string, unsigned int>& mFeature, map<string, unsigned int>& mTime,
    vector<unsigned int>& vFeatures, vector<unsigned int>& vTime, const bool clip) {

  if (!clip) {
    if (mFeature.find(s) == mFeature.end()) {
      unsigned int sz = (unsigned int) mFeature.size();
      mFeature[s] = sz;
    }
  } else {
    if (mFeature.find(s) == mFeature.end()) {
      cout << "feature has to be removed" << endl;
      exit(0);
    }
  }
  vFeatures.push_back(mFeature[s]);

  if (mTime.find(s) != mTime.end()) {
    vTime.push_back(mTime[s]);
  } else {
    cout << "error map" << endl;
    exit(0);
  }
}

void addFEATUREset(const string& feature, const unsigned int date, set<string>& sFeatures, map<string, unsigned int>& mTime,
    map<string, unsigned int>& mFeature, const bool clip) {
  if (clip) { // if limitation is enabled
    if (mFeature.find(feature) == mFeature.end()) { // if feature is not found
      return;
    }
  }
  if (sFeatures.count(feature) == 0) {
    sFeatures.insert(feature);
    mTime[feature] = date;
  } else {
    if (mTime[feature] < date) { // select the freshest purchase
      mTime[feature] = date;
    }
  }
}

int writeFeatureToInd(const string& fname, const map<string, unsigned int>& mFeature) {
  std::ofstream file(fname.c_str());
  checkFile(file, fname);
  int num_keys = 0;
  for (map<string, unsigned int>::const_iterator it = mFeature.begin(); it != mFeature.end(); it++) {
    string key = it->first;
    int ind = it->second;
    file << key << " " << ind << std::endl;
    num_keys++;
  }
  return num_keys;
}

