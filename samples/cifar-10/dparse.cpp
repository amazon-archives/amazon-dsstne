// Parser to generate training and test data sets for CIFAR-10
//
// Step 1. Download CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
// Step 2. mv test_batch.bin test.bin
// Step 3. cat data_batch_*.bin > training.bin
// Step 4. g++ dparse.cpp -o dparse -lnetcdf -lnetcdf_c++4 --std=c++0x
// Step 5. ./dparse
// Step 6. Use the network in config.json with cifar-10-training.nc and cifar-10-test.nc, P@1~=81%

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <stdlib.h>
#include <netcdf>
#include <sys/time.h>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

double elapsed_time(timeval& x, timeval& y)
{
    timeval result;
  /* Perform the carry for the later subtraction by updating y. */
  if (x.tv_usec < y.tv_usec) {
    int nsec = (y.tv_usec - x.tv_usec) / 1000000 + 1;
    y.tv_usec -= 1000000 * nsec;
    y.tv_sec += nsec;
  }
  if (x.tv_usec - y.tv_usec > 1000000) {
    int nsec = (x.tv_usec - y.tv_usec) / 1000000;
    y.tv_usec += 1000000 * nsec;
    y.tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result.tv_sec = x.tv_sec - y.tv_sec;
  result.tv_usec = x.tv_usec - y.tv_usec;
  return (double)result.tv_sec + ((double)result.tv_usec / 1000000.0);
}


int main(int argc, char **argv)
{
    timeval ts;
    gettimeofday(&ts, NULL);
    static const uint32_t classes = 10;
    static const uint32_t training_images = 49920; //50000; //49920;
    static const uint32_t test_images = 9984; //10000; //9984;    
    static const uint32_t width = 32;
    static const uint32_t height = 32;
    static const uint32_t length = 3;
    static const uint32_t imageSize = width * height * length;

    // Read training data set
    ifstream is("training.bin", ios::binary | ios::in);
    if (!is)
    {
        printf("Error opening input file training.bin\n");
        exit(-1);
    }
    vector<uint8_t> vTrainingData(training_images * imageSize);
    vector<uint32_t> vTrainingSparseLabel(training_images);
    vector<uint32_t> vTrainingSparseLabelStart(training_images);
    vector<uint32_t> vTrainingSparseLabelEnd(training_images);    
    size_t pos = 0;
    for (size_t i = 0; i < training_images; i++)
    {
        uint8_t dummy;
        is.read((char*)&dummy, 1);
        vTrainingSparseLabel[i] = dummy;
        is.read((char*)&vTrainingData[pos], imageSize);
        vTrainingSparseLabelStart[i] = i;
        vTrainingSparseLabelEnd[i] = i + 1;
        pos += imageSize;
    }
    is.close();
    
    {
        // Write CIFAR training set 
        NcFile nc("cifar10_training.nc", NcFile::replace);
        if (nc.isNull())
            printf("Error opening binary output file.\n");
        nc.putAtt("datasets", ncUint, 2);
        
        nc.putAtt("name0", "input");
        nc.putAtt("attributes0", ncUint, 0);
        nc.putAtt("kind0", ncUint, 1);
        nc.putAtt("dataType0", ncUint, 8);
        nc.putAtt("dimensions0", ncUint, 3);
        nc.putAtt("width0", ncUint, width);
        nc.putAtt("height0", ncUint, height);
        nc.putAtt("length0", ncUint, length);
        NcDim examplesDim0      = nc.addDim("examplesDim0", training_images);
        NcDim dataDim0          = nc.addDim("dataDim0", training_images * imageSize); 
        NcVar dataVar0          = nc.addVar("data0", ncUint, dataDim0); 
        dataVar0.putVar(vTrainingData.data());
    
        // Write output data set
        nc.putAtt("name1", "output");
        nc.putAtt("attributes1", ncUint, 3);
        nc.putAtt("kind1", ncUint, 0);
        nc.putAtt("dataType1", ncUint, 0);
        nc.putAtt("dimensions1", ncUint, 1);
        nc.putAtt("width1", ncUint, classes);
        NcDim examplesDim1      = nc.addDim("examplesDim1", training_images);
        NcDim sparseDataDim1    = nc.addDim("sparseDataDim1", training_images); 
        NcVar sparseStartVar1   = nc.addVar("sparseStart1", ncUint, examplesDim1);
        NcVar sparseEndVar1     = nc.addVar("sparseEnd1", ncUint, examplesDim1);
        NcVar sparseIndexVar1   = nc.addVar("sparseIndex1", ncUint, sparseDataDim1);
        sparseStartVar1.putVar(vTrainingSparseLabelStart.data());
        sparseEndVar1.putVar(vTrainingSparseLabelEnd.data());
        sparseIndexVar1.putVar(vTrainingSparseLabel.data());      
    }
    
    // Read test data set
    is.open("test.bin", ios::binary | ios::in);
    if (!is)
    {
        printf("Error opening input file test.bin\n");
        exit(-1);
    }    
    vector<uint8_t> vTestData(test_images * imageSize);
    vector<uint32_t> vTestSparseLabel(test_images);
    vector<uint32_t> vTestSparseLabelStart(test_images);
    vector<uint32_t> vTestSparseLabelEnd(test_images);    
    pos = 0;
    for (size_t i = 0; i < test_images; i++)
    {
        uint8_t dummy;
        is.read((char*)&dummy, 1);
        vTestSparseLabel[i] = dummy; 
        is.read((char*)&vTestData[pos], imageSize);
        vTestSparseLabelStart[i] = i;
        vTestSparseLabelEnd[i] = i + 1;
        pos += imageSize;
    }
    is.close();
    
    {
        // Write CIFAR test set 
        NcFile nc("cifar10_test.nc", NcFile::replace);
        if (nc.isNull())
            printf("Error opening binary output file.\n");
        nc.putAtt("datasets", ncUint, 2);
        
        nc.putAtt("name0", "input");
        nc.putAtt("attributes0", ncUint, 0);
        nc.putAtt("kind0", ncUint, 1);
        nc.putAtt("dataType0", ncUint, 8);
        nc.putAtt("dimensions0", ncUint, 3);
        nc.putAtt("width0", ncUint, width);
        nc.putAtt("height0", ncUint, height);
        nc.putAtt("length0", ncUint, length);
        NcDim examplesDim0      = nc.addDim("examplesDim0", test_images);
        NcDim dataDim0          = nc.addDim("dataDim0", test_images * imageSize); 
        NcVar dataVar0          = nc.addVar("data0", ncUbyte, dataDim0); 
        dataVar0.putVar(vTestData.data());
 
        // Write output data set
        nc.putAtt("name1", "output");
        nc.putAtt("attributes1", ncUint, 3);
        nc.putAtt("kind1", ncUint, 0);
        nc.putAtt("dataType1", ncUint, 0);
        nc.putAtt("dimensions1", ncUint, 1);
        nc.putAtt("width1", ncUint, classes);
        NcDim examplesDim1      = nc.addDim("examplesDim1", test_images);
        NcDim sparseDataDim1    = nc.addDim("sparseDataDim1", test_images); 
        NcVar sparseStartVar1   = nc.addVar("sparseStart1", ncUint, examplesDim1);
        NcVar sparseEndVar1     = nc.addVar("sparseEnd1", ncUint, examplesDim1);
        NcVar sparseIndexVar1   = nc.addVar("sparseIndex1", ncUint, sparseDataDim1);
        sparseStartVar1.putVar(vTestSparseLabelStart.data());
        sparseEndVar1.putVar(vTestSparseLabelEnd.data());
        sparseIndexVar1.putVar(vTestSparseLabel.data());
    }






    timeval t1;
    gettimeofday(&t1, NULL);

    printf("Write %fs\n", elapsed_time(t1, ts));

    return 0;
}

