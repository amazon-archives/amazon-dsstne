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
#include <stdexcept>
#include <unordered_map>

#include <values.h>

#include "Utils.h"
#include "Filters.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "NNRecsGenerator.h"
#include "NetCDFhelper.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;
using namespace std;

unsigned int INTERVAL_REPORT_PROGRESS = 1000000;

/**
Extracts the Map whose key is the value and the value is the index to vector
Map : A 1
      B 0

Vector A,B
**/
void extractNNMapsToVectors(vector<string> &vVectors,
                            unordered_map<string, unsigned int>&  mMaps)
{
    unordered_map<string, unsigned int>::iterator mInputIter;
    for(mInputIter = mMaps.begin(); mInputIter != mMaps.end() ; ++mInputIter )
    {
        vVectors[mInputIter->second] = mInputIter->first;
    }
}

/**
 * A wrapper function to convert the TSV text file into a NetCDF file. The mSignalIndex will return the mappings for
 * all instances/signals/samples/customer id that were found in the text dataset. This looks common enough to move
 * to common utils.
 *
 * @param inputTextFile - input text file to process.
 * @param dataSetName - the name for the dataset to store in netcdf.
 * @param outputNCDFFile - the name of the output NetCDF file that we generate.
 * @param mFeatureIndex - feature index map used to translate features to indices for sparse representation.
 * @param mSignalsIndex - signals or instance index, updated as the text file is processed.
 */
void convertTextToNetCDF(string inputTextFile, 
                         string dataSetName, 
                         string outputNCDFFile, 
                         unordered_map<string, unsigned int> &mFeatureIndex,
                         unordered_map<string, unsigned int> &mSignalIndex,
                         string featureIndexFile, 
                         string sampleIndexFile)
{
    vector <unsigned int> vSparseStart;
    vector <unsigned int> vSparseEnd;
    vector <unsigned int> vSparseIndex;
    vector <float> vSparseData;

    if (!generateNetCDFIndexes(inputTextFile, false, featureIndexFile, sampleIndexFile, mFeatureIndex, mSignalIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, cout)) {
        exit(1);
    }

    // Only write binary data using a single CPU
    if (getGpu()._id==0){
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, 
            outputNCDFFile, dataSetName, mFeatureIndex.size());
    }

    // Delete unwanted memory now that we have produced the netCDF file.
    forceClearVector(vSparseStart);
    forceClearVector(vSparseEnd);
    forceClearVector(vSparseIndex);
    forceClearVector(vSparseData);
}

void printUsagePredict() {
    cout << "Predict: Generates predictions from a trained neural network given a signals/input dataset." << endl;
    cout << "Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_recs>] [-l layer] [-s input_signals_index] [-p score_precision]" << endl;
    cout << "    -b batch_size: (default = 1024) the number records/input rows to process in a batch." << endl;
    cout << "    -d dataset_name: (required) name for the dataset within the netcdf file." << endl;
    cout << "    -f samples filterFileName ." << endl;
    cout << "    -i input_feature_index: (required) path to the feature index file, used to tranform input signals to correct input feature vector." << endl;
    cout << "    -k num_recs: (default = 100) The number of predictions (sorted by score to generate). Ignored if -l flag is used." << endl;
    cout << "    -l layer: (default = Output) the network layer to use for predictions. If specified, the raw scores for each node in the layer is output in order." << endl;
    cout << "    -n network_file: (required) the trained neural network in NetCDF file." << endl;
    cout << "    -o output_feature_index: (required) path to the feature index file, used to tranform the network output feature vector to appropriate features." << endl;
    cout << "    -p score_precision: (default = 4.3f) precision of the scores in output" << endl;
    cout << "    -r input_text_file: (required) path to the file with input signal to use to generate predictions (i.e. recommendations)." << endl;
    cout << "    -s filename (required) . to put the output recs to." << endl;
    cout << endl;
}

int main(int argc, char** argv)
{
    if (isArgSet(argc, argv, "-h")) {
        printUsagePredict();
        exit(1);
    }

    // Check all required arguments first.
    string dataSetName = getRequiredArgValue(argc, argv, "-d", "dataset_name is not specified.", &printUsagePredict);
    // For prediction we automatically add the _INPUT_DATASET_SUFFIX.
    dataSetName += INPUT_DATASET_SUFFIX;

    string filtersFileName = getRequiredArgValue(argc, argv, "-f", "filters_json is not specified.", &printUsagePredict);
    if (! fileExists(filtersFileName)) {
        cout << "Error: Cannot read filter file: " << filtersFileName << endl;
        return 1;
    }

    string inputIndexFileName = getRequiredArgValue(argc, argv, "-i", "input features index file is not specified.", &printUsagePredict);
    if (! fileExists(inputIndexFileName)) {
        cout << "Error: Cannot read input feature index file: " << inputIndexFileName << endl;
        return 1;
    }

    string networkFileName = getRequiredArgValue(argc, argv, "-n", "network file is not specified.", &printUsagePredict);
    if (! fileExists(networkFileName)) {
        cout << "Error: Cannot read network file: " << networkFileName << endl;
        return 1;
    }

    string outputIndexFileName = getRequiredArgValue(argc, argv, "-o", "output features index file is not specified.", &printUsagePredict);
    if (! fileExists(outputIndexFileName)) {
        cout << "Error: Cannot read output feature index file: " << outputIndexFileName << endl;
        return 1;
    }

    string recsFileName = getRequiredArgValue(argc, argv, "-r", "input_text_file is not specified.", &printUsagePredict);
    if (! fileExists(recsFileName)) {
        cout << "Error: Cannot read input_text_file: " << recsFileName << endl;
        return 1;
    }

    string recsOutputFileName =  getRequiredArgValue(argc, argv, "-s", "filename to put the output recs to.", &printUsagePredict);



    // Check optional arguments and use default values if not overidden
    unsigned int batchSize =  stoi(getOptionalArgValue(argc, argv, "-b", "1024"));

    unsigned int topK =  stoi(getOptionalArgValue(argc, argv, "-k", "100"));
    if (topK >=128 ) {
	cout << "Error :Optimized topk Only works for top 128 . "<< topK<< " is greater" <<endl;
	return 1;
    }

    string scoreFormat = getOptionalArgValue(argc, argv, "-p", NNRecsGenerator::DEFAULT_SCORE_PRECISION);


    // Initialize GPU network
    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    // Start timing loading of data and network.
    timeval timePreProcessingStart;
    gettimeofday(&timePreProcessingStart, NULL);

    unordered_map<string, unsigned int> mInput;
    cout << "Loading input feature index from: " << inputIndexFileName << endl;
    if (!loadIndexFromFile(mInput, inputIndexFileName, cout)) {
        exit(1);
    }

    // Load the dataset text file and geneate the NetCDF
    unordered_map<string, unsigned int> mSignals;
    string inputNetCDFFileName;

    // Ensure we don't override other dataset files (i.e. from training)
    string dataSetFilesPrefix = dataSetName + "_predict";
    inputNetCDFFileName.assign(dataSetFilesPrefix + NETCDF_FILE_EXTENTION);

    string featureIndexFile = dataSetFilesPrefix + ".featuresIndex";
    string sampleIndexFile = dataSetFilesPrefix + ".samplesIndex";
    convertTextToNetCDF(recsFileName,
		    dataSetName,
		    inputNetCDFFileName,
		    mInput,
		    mSignals,
		    featureIndexFile,
		    sampleIndexFile);
    // TODO: We should look at avoiding generating/re-reading the netCDF since we have parsed it.
    // TODO: convertTextToNetCDF needs a better name. Now it parse the text into NetCDF file and write them out. A function should have all input/output
    // variables defined in the interface

    // Load the filter set
    if(getGpu()._id == 0 ){
        cout << "Number of network input nodes: " << mInput.size() << endl;
        cout << "Number of entries to generate predictions for: " << mSignals.size() << endl;
        CWMetric::updateMetrics("Signals_Size", mSignals.size());
    }

    vector <NNDataSetBase*> vDataSetInput = LoadNetCDF(inputNetCDFFileName);
    NNNetwork* pNetwork = LoadNeuralNetworkNetCDF(networkFileName, batchSize);
    pNetwork->LoadDataSets(vDataSetInput);

    // Generate an ordered vector of the signals/samples index, so that output are correctly labeled.
    vector<string> vSignals(mSignals.size());
    extractNNMapsToVectors(vSignals, mSignals);

    // For output recs, we cannot assume the input and output layers have identical
    // features or even ordering. So, we load the index for output layer.
    unordered_map<string, unsigned int> mOutput;
    cout << "Loading output feature index from: " << outputIndexFileName << endl;
    if (!loadIndexFromFile(mOutput, outputIndexFileName, cout)) {
        exit(1);
    }
    
    vector<string> vOutput(mOutput.size());
    extractNNMapsToVectors(vOutput, mOutput);
    FilterConfig* vFilterSet = loadFilters(filtersFileName,recsOutputFileName, mOutput, mSignals);
    // Delete the unwanted memory
    mInput.clear();
    mOutput.clear();
    mSignals.clear();

    timeval timePreProcessingEnd;
    gettimeofday(&timePreProcessingEnd, NULL);
    cout << "Total time for loading network and data is: " << elapsed_time(timePreProcessingEnd, timePreProcessingStart) << endl;

    string recsGenLayerLabel = "Output";
    // Now ready to generate recs for the dataset
    const NNLayer* pLayer = pNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    tie(lx, ly, lz, lw)            = pLayer->GetDimensions();
    unsigned int lBatch            = pNetwork->GetBatch();
    unsigned int outputBufferSize  = pNetwork->GetBufferSize(recsGenLayerLabel);

    NNRecsGenerator *nnRecsGenerator = new NNRecsGenerator(lBatch, topK, outputBufferSize, recsGenLayerLabel, scoreFormat);

    timeval timeRecsGenerationStart;
    gettimeofday(&timeRecsGenerationStart, NULL);

    timeval timeProgressReporterStart;
    gettimeofday(&timeProgressReporterStart, NULL);
    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
    {
        cout << "Predicting from position "<< pos << endl;

        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();
        nnRecsGenerator->generateRecs(pNetwork, topK, vFilterSet, vSignals, vOutput);
        if((pos % INTERVAL_REPORT_PROGRESS) < pNetwork->GetBatch()  && (pos/INTERVAL_REPORT_PROGRESS) > 0 && getGpu()._id == 0) {
            timeval timeProgressReporterEnd;
            gettimeofday(&timeProgressReporterEnd, NULL);
	    cout << "Elapsed time after " << pos <<" is "<<elapsed_time(timeProgressReporterEnd, timeProgressReporterStart)<<endl;
            CWMetric::updateMetrics("Prediction_Time_Progress", elapsed_time(timeProgressReporterEnd, timeProgressReporterStart));
            CWMetric::updateMetrics("Prediction_Progress",(unsigned int)pos);
            gettimeofday(&timeProgressReporterStart,NULL);
        }

    }
    timeval timeRecsGenerationEnd;
    gettimeofday(&timeRecsGenerationEnd, NULL);
    if (getGpu()._id == 0) {
        CWMetric::updateMetrics("Prediction_Time", elapsed_time(timeRecsGenerationEnd, timeRecsGenerationStart));
        cout << "Total time for Generating recs for " << pNetwork->GetExamples() << " was " <<  elapsed_time(timeRecsGenerationEnd, timeRecsGenerationStart) << endl;}

    delete(nnRecsGenerator);
    delete pNetwork;
    getGpu().Shutdown();
    return 0;
}
