/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

#ifndef LIBKNN_KNN_HANDLE_H_
#define LIBKNN_KNN_HANDLE_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "DataReader.h"

namespace astdl
{
namespace knn
{

enum class DataType
{
  FP32 = 0, FP16 = 1
};

std::string getDataTypeString(DataType dataType);

DataType getDataTypeFromString(const std::string &dataTypeLiteral);

/**
 * Holds the data pointer and row, column, and element size
 * information for a 2-d array in either host or device memory.
 */
struct Matrix
{
    void *data;
    uint32_t numRows;
    int numColumns;
    size_t elementSize;
    cudaMemoryType memoryType;

    Matrix();

    Matrix(void *data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType);

    size_t getSizeInBytes();

    size_t getLength();
};

Matrix loadDataOnHost(DataReader *dataReader);

struct KnnData
{
    const int numGpus;
    const int batchSize;
    const int maxK;

    /*
     * cublas handles
     */
    std::vector<cublasHandle_t> cublasHandles;

    /*
     * data
     */
    std::vector<Matrix> dCollectionPartitions; // column major
    std::vector<Matrix> dInputBatches;
    std::vector<Matrix> dProducts;

    std::vector<Matrix> dResultScores;
    std::vector<Matrix> dResultIndexes;

    std::vector<Matrix> hResultScores;
    std::vector<Matrix> hResultIndexes;
    std::vector<std::vector<std::string>> hKeys;

    /*
     * tmp buffers
     */
    // used for converting float -> half when data type is FP16
    std::vector<Matrix> dInputBatchTmpBuffers;

    /*
     * number of rows padded per device.
     * we pad rows to multiples of 4 for cublasSgemm performance
     * actualNumRows = dCollectionPartitions[device].rows - collectionRowsPadded[device]
     */
    std::vector<uint32_t> collectionRowsPadded;

    /*
     * metric data
     */
    std::vector<float> elapsedSgemm;
    std::vector<float> elapsedTopK;

    /**
     * Stores elements in fp16 instead of fp32. Also uses fp16 math on V100.
     */
    const DataType dataType;

    KnnData(int numGpus, int batchSize, int maxK, DataType dataType);

    void load(int device, DataReader *dataReader);

    void load(const std::map<int, DataReader*> &deviceToData);

    void load(const std::map<int, std::string> &deviceToFile, char keyValDelim, char vecDelim);

    int getFeatureSize() const;

    ~KnnData();
};

Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize);

Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize);

void freeMatrix(const Matrix &matrix);

} // namespace knn
} // namepace astdl

#endif /* KNN_HANDLE_H_ */
