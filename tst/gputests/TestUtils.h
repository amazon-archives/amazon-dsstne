#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>

#include "GpuTypes.h"
#include "NNTypes.h"
#include "NetCDFhelper.h"

#define TEST_DATA_PATH "../../../../tst/test_data/"

enum TestDataType {
    Regression = 1,               // data for regression
    Classification = 2,           // data for classification
    ClassificationAnalog = 3,     // data for classification with analog input features
};

struct DataParameters {
    DataParameters() {
        numberOfSamples = 1024;
        inpFeatureDimensionality = 1;
        outFeatureDimensionality = 1;
        W0 = -2.f;
        B0 = 3.f;
    }
    int numberOfSamples;
    int inpFeatureDimensionality;
    int outFeatureDimensionality;
    float W0;
    float B0;
};

inline void generateTestData(const std::string& path, const TestDataType testDataType, const DataParameters& dataParameters, std::ostream& out) {
    std::vector<std::vector<unsigned int> > vSampleTestInput, vSampleTestInputTime;
    std::vector<std::vector<float> > vSampleTestInputData;
    std::vector<std::vector<unsigned int> > vSampleTestOutput, vSampleTestOutputTime;
    std::vector<std::vector<float> > vSampleTestOutputData;
    std::vector<std::string> vSamplesName(dataParameters.numberOfSamples);
    std::map<std::string, unsigned int> mFeatureNameToIndex;

    for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
        std::string feature_name = std::string("feature") + std::to_string((long long) d);
        mFeatureNameToIndex[feature_name] = d;
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        vSamplesName[s] = std::string("sample") + std::to_string((long long) s);
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        vector<unsigned int> inpFeatureIndex, inpTime;
        vector<float> inpFeatureValue;
        vector<unsigned int> outFeatureIndex, outTime;
        vector<float> outFeatureValue;

        switch (testDataType) {
        case Regression:
            for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
                inpFeatureIndex.push_back(d);
                inpFeatureValue.push_back((float) s);
                inpTime.push_back(s);
            }

            for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                outFeatureIndex.push_back(d);
                outFeatureValue.push_back(dataParameters.W0 * inpFeatureValue[d] + dataParameters.B0);
                outTime.push_back(s);
            }

            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputData.push_back(inpFeatureValue);
            vSampleTestInputTime.push_back(inpTime);
            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputData.push_back(outFeatureValue);
            vSampleTestOutputTime.push_back(outTime);
            break;
        case Classification:
            inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality); // one activation per sample
            inpTime.push_back(s);

            outFeatureIndex.push_back(s % dataParameters.outFeatureDimensionality); // one activation per sample
            outTime.push_back(s);

            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputTime.push_back(inpTime);
            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputTime.push_back(outTime);
            break;
        case ClassificationAnalog:
            inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality); // one activation per sample
            inpFeatureValue.push_back((float) s);
            inpTime.push_back(s);

            for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                outFeatureIndex.push_back(d);
                outFeatureValue.push_back(((s + d) % 2) + 1);
                outTime.push_back(s);
            }
            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputData.push_back(inpFeatureValue);
            vSampleTestInputTime.push_back(inpTime);

            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputData.push_back(outFeatureValue);
            vSampleTestOutputTime.push_back(outTime);

            break;
        default:
            out << "unsupported mode";
            exit(2);
        }
    }

    int minInpDate = std::numeric_limits<int>::max(), maxInpDate = std::numeric_limits<int>::min(),
                    minOutDate = std::numeric_limits<int>::max(), maxOutDate = std::numeric_limits<int>::min();
    const bool alignFeatureNumber = false;
    writeNETCDF(path + std::string("test.nc"), vSamplesName, mFeatureNameToIndex, vSampleTestInput, vSampleTestInputTime,
                    vSampleTestInputData, mFeatureNameToIndex, vSampleTestOutput, vSampleTestOutputTime,
                    vSampleTestOutputData, minInpDate, maxInpDate, minOutDate, maxOutDate, alignFeatureNumber, 2);
}

inline bool validateNeuralNetwork(const size_t batch, const std::string& modelPath, const TestDataType testDataType, const DataParameters& dataParameters, std::ostream& out) {
    out << "start validation of " << modelPath << std::endl;

    NNNetwork* pNetwork = NULL;
    std::vector<NNDataSetBase*> vDataSet;
    const std::string dataName = string("test.nc");
    const std::string dataPath(TEST_DATA_PATH);
    generateTestData(dataPath, testDataType, dataParameters, out);
    vDataSet = LoadNetCDF(dataPath + dataName);
    pNetwork = LoadNeuralNetworkJSON(modelPath, batch, vDataSet);
    pNetwork->LoadDataSets(vDataSet);
    pNetwork->SetCheckpoint("check", 1);
    pNetwork->SetTrainingMode(SGD);
    bool valid = pNetwork->Validate();
    if (valid) {
        out << "SUCCESFUL validation" << std::endl;
    } else {
        out << "FAILED validation" << std::endl;
    }

    int totalGPUMemory, totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    out << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    out << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    delete pNetwork;

    // Delete datasets
    for (auto p : vDataSet) {
        delete p;
    }
    return valid;
}
