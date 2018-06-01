/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#include <boost/filesystem.hpp>

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <netcdf>
#include <values.h>
#include <stdexcept>
#include <unordered_map>

#include "GpuTypes.h"
#include "NetCDFhelper.h"
#include "NNTypes.h"
#include "TensorboardMetricsLogger.h"
#include "Utils.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

void printUsageTrain() {
    cout << "Train: Trains a neural networks given a config and dataset." << endl;
    cout << "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_netcdf> -o <output_netcdf> [-b <batch_size>] [-e <num_epochs>]" << endl;
    cout << "    -c config_file: (required) the JSON config files with network training parameters." << endl;
    cout << "    -i input_netcdf: (required) path to the netcdf with dataset for the input of the network." << endl;
    cout << "    -o output_netcdf: (required) path to the netcdf with dataset for expected output of the network." << endl;
    cout << "    -n network_file: (required) the output trained neural network in NetCDF file." << endl;
    cout << "    -b batch_size: (default = 1024) the number records/input rows to process in a batch." << endl;
    cout << "    -e num_epochs: (default = 40) the number passes on the full dataset." << endl;
    cout << "    --logdir LOGDIR: (default = 'logs') a directory where Tensorboard-compatible logs will be emitted." << endl;
    cout << endl;
}

/**
Samples argument
./train -i input.netcdf -o output.netcdf -e 10 -c config.json -b 5 -n  network.nc

input is the data for the input layer
output is the data for the output layer
10 is the number of epochs to run
config.json is the Json configuration for the NN layer
5 is the batch size
network.nc is the Network dump after the training is done
*/

int main(int argc, char** argv)
{
    // Hyper parameters
    float alpha = stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));
    float lambda = stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));
    float lambda1 = stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));
    float mu = stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));
    float mu1 = stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));


   if (isArgSet(argc, argv, "-h")) {
        printUsageTrain();
        exit(1);
    }
   
    // Check all required arguments first. 

    string configFileName = getRequiredArgValue(argc, argv, "-c", "config file was not specified.", &printUsageTrain);
    if (! fileExists(configFileName)) {
        cout << "Error: Cannot read config file: " << configFileName << endl;
        return 1;
    } else {
        cout << "Train will use configuration file: " << configFileName << endl;
	}

    string inputDataFile = getRequiredArgValue(argc, argv, "-i", "input data file is not specified.", &printUsageTrain);
    if (! fileExists(inputDataFile)) {
        cout << "Error: Cannot read input feature index file: " << inputDataFile << endl;
        return 1;
    } else {
        cout << "Train will use input data file: " << inputDataFile << endl;
	}

    string outputDataFile = getRequiredArgValue(argc, argv, "-o", "output data  file is not specified.", &printUsageTrain);
    if (! fileExists(outputDataFile)) {
        cout << "Error: Cannot read output feature index file: " << outputDataFile << endl;
        return 1;
    } else {
        cout << "Train will use output data file: " << outputDataFile << endl;
	}
    
    string networkFileName = getRequiredArgValue(argc, argv, "-n", "the output network file path is not specified.", &printUsageTrain);
    if (fileExists(networkFileName)) {
        cout << "Error: Network file already exists: " << networkFileName << endl;
        return 1;
    } else {
        cout << "Train will produce networkFileName: " << networkFileName << endl;
	}

    boost::filesystem::path logdir = getOptionalArgValue(argc, argv, "--logdir", "logs");
    if (logdir.is_relative()) {
        logdir = boost::filesystem::absolute(logdir);
    }
    if (!boost::filesystem::exists(logdir)) {
        boost::filesystem::create_directory(logdir);
    }
    std::cout << "Train will log Tensorboard-compatible metrics to the following directory: " << logdir << std::endl;
    TensorboardMetricsLogger metrics(logdir);

    // Check optional arguments and use default values if not overidden
    unsigned int batchSize =  stoi(getOptionalArgValue(argc, argv, "-b", "1024"));
    cout << "Train will use batchSize: " << batchSize << endl;

    unsigned int epoch =  stoi(getOptionalArgValue(argc, argv, "-e", "40"));
    cout << "Train will use number of epochs: " << epoch << endl;
    cout << "Train alpha " << alpha << ", lambda " << lambda <<", mu "<< mu <<".Please check CDL.txt for meanings" << endl;
    cout << "Train alpha " << alpha << ", lambda " << lambda << ", lambda1 " << lambda1 << ", mu " << mu << ", mu1 " << mu1 << ".Please check CDL.txt for meanings" << endl;    
	
    // Initialize GPU network
    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    // Load the input and output dataset
    vector <NNDataSetBase*> vDataSetInput = LoadNetCDF(inputDataFile);
    vector <NNDataSetBase*> vDataSetOutput = LoadNetCDF(outputDataFile);

    // Merging to a single List for Loading it to Network
    vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());

    // Create a Neural network from the config
    NNNetwork* pNetwork = LoadNeuralNetworkJSON(configFileName, batchSize, vDataSetInput);
    
    // Load training data
    pNetwork->LoadDataSets(vDataSetInput);
    pNetwork->LoadDataSets(vDataSetOutput);
    pNetwork->SetCheckpoint(networkFileName, 10);

    // Save initialized network before train
    pNetwork->SetPosition(0);
    pNetwork->PredictBatch();
    pNetwork->SaveNetCDF("initial_network.nc");

    // Set to default training mode SGD.
    TrainingMode mode=SGD;
    pNetwork->SetTrainingMode(mode);

    auto const start = std::chrono::steady_clock::now();
    // Start Training
    for(unsigned int x = 0 ; x < epoch; ++x) {
        float error = pNetwork->Train(1, alpha, lambda, lambda1, mu, mu1);
        unsigned int epoch = x + 1;
        CWMetric::updateMetrics("Average_Error",error);
        CWMetric::updateMetrics("Epochs",epoch);
        metrics.scalar(epoch, "Average_Error", error);
    }
    auto const end = std::chrono::steady_clock::now();
    CWMetric::updateMetrics("Training_Time", elapsed_seconds(start, end));
    cout << "Total Training Time " << elapsed_seconds(start, end);

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << endl;
    cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << endl;
    CWMetric::updateMetrics("Training_GPU_usage", totalGPUMemory);
    // Save Neural network
    pNetwork->SaveNetCDF(networkFileName);
    delete pNetwork;
    getGpu().Shutdown();
    return 0;
}

