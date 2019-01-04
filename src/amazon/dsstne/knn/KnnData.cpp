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

#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "cudautil.h"
#include "KnnData.h"
#include "MathUtil.h"

namespace
{
static const int ROW_PADDING = 8;
static const std::unordered_map<std::string, astdl::knn::DataType> STRING_TO_DATA_TYPE = { { "fp32",
    astdl::knn::DataType::FP32 }, { "fp16", astdl::knn::DataType::FP16 }, };
}  // namespace

namespace astdl
{
namespace knn
{

Matrix::Matrix() :
    data(nullptr),
    numRows(0),
    numColumns(0),
    elementSize(0),
    memoryType(cudaMemoryTypeHost)
{
}

Matrix::Matrix(void *data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType) :
    data(data),
    numRows(numRows),
    numColumns(numColumns),
    elementSize(elementSize),
    memoryType(memoryType)
{
}

size_t Matrix::getSizeInBytes()
{
    size_t sizeInBytes = numRows * numColumns * elementSize;
    return sizeInBytes;
}

size_t Matrix::getLength()
{
    size_t length = numRows * numColumns;
    return length;
}

std::string getDataTypeString(DataType dataType)
{
    switch (dataType) {
        case DataType::FP32:
            return "fp32";
        case DataType::FP16:
            return "fp16";
        default:
            return "unknown";
    }
}

DataType getDataTypeFromString(const std::string &dataTypeLiteral)
{
    auto entry = STRING_TO_DATA_TYPE.find(dataTypeLiteral);
    if (entry == STRING_TO_DATA_TYPE.end())
    {
        std::stringstream msg;
        msg << "Unknown DataType " << dataTypeLiteral;
        throw std::invalid_argument(msg.str());
    }
    return entry->second;
}

Matrix loadDataOnHost(DataReader *dataReader)
{
    uint32_t rows = dataReader->getRows();
    int columns = dataReader->getColumns();
    size_t elementSize = sizeof(float);

    Matrix matrix = allocateMatrixOnHost(rows, columns, elementSize);

    std::string ignored;
    for (int rowNum = 0; dataReader->readRow(&ignored, ((float*) matrix.data) + (rowNum * columns)); ++rowNum)
    {
    }
    return matrix;
}

KnnData::KnnData(int numGpus, int batchSize, int maxK, DataType dataType) :
    numGpus(numGpus),
    batchSize(batchSize),
    maxK(maxK),
    dataType(dataType),
    dCollectionPartitions(numGpus),
    dInputBatches(numGpus),
    dProducts(numGpus),
    dResultScores(numGpus),
    dResultIndexes(numGpus),
    hResultScores(numGpus),
    hResultIndexes(numGpus),
    dInputBatchTmpBuffers(numGpus),
    collectionRowsPadded(numGpus),
    hKeys(numGpus),
    elapsedSgemm(numGpus),
    elapsedTopK(numGpus)
{
// sanity check number of GPUs
    int deviceCount = astdl::cuda_util::getDeviceCount();
    if (deviceCount < 1)
    {
        std::stringstream msg;
        msg << "No GPU device found on host. Device count is " << deviceCount;
        throw std::runtime_error(msg.str());
    }

    if (deviceCount < numGpus)
    {
        std::stringstream msg;
        msg << "Not enough GPUs on host. Required " << numGpus << ", found " << deviceCount;
        throw std::runtime_error(msg.str());
    }

    fprintf(stderr, "INFO: Initializing KnnData with numGpus = %d, batchSize = %d, maxK = %d, dataType = %s\n", numGpus,
            batchSize, maxK, getDataTypeString(dataType).c_str());
// fp16 mode only supported on sm >= 7
    if (dataType == DataType::FP16)
    {
        cudaDeviceProp deviceProp;
        CHECK_ERR(cudaGetDeviceProperties(&deviceProp, 0));
        int smMajor = deviceProp.major;
        int smMinor = deviceProp.minor;

        if (smMajor < 7)
        {
            fprintf(stderr, "WARNING: fp16 compute is not supported in sm %d.%d < 7. Only storing data in fp16.\n",
                    smMajor, smMinor);
        }
    }

    for (int i = 0; i < numGpus; ++i)
    {
        CHECK_ERR(cudaSetDevice(i));
        cublasHandle_t handle;
        STATUS_ERR(cublasCreate(&handle));

        if (dataType == DataType::FP16)
        {
            fprintf(stderr, "INFO: On device %d, setting cublas mode to CUBLAS_TENSOR_OP_MATH\n", i);
            STATUS_ERR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        }

        cublasHandles.push_back(handle);
    }
}

int KnnData::getFeatureSize() const
{
    return dCollectionPartitions[0].numColumns;
}

void KnnData::load(int device, DataReader *dataReader)
{
    CHECK_ERR(cudaSetDevice(device));

// pad by multiples of 4 for sgemm
    uint32_t actualRows = dataReader->getRows();
    uint32_t rows = ((actualRows + (ROW_PADDING - 1)) / ROW_PADDING) * ROW_PADDING;
    uint32_t rowsPadded = rows - actualRows;
    int columns = dataReader->getColumns();

// holds the data on host memory until we copy it over to the device
    Matrix hTmpMatrix = allocateMatrixOnHost(rows, columns, sizeof(float));
    size_t hTmpDataBytes = hTmpMatrix.getSizeInBytes();
    float *hTmpData = (float*) hTmpMatrix.data;

    collectionRowsPadded[device] = rowsPadded;

    std::string key;
    float vector[columns];
//    for (int rowNum = 0; dataReader->readRow(&key, hTmpData + (rowNum * columns)); ++rowNum)
    for (int rowNum = 0; dataReader->readRow(&key, vector); ++rowNum)
    {
        hKeys[device].push_back(key);
        // copy vector into hTmpData in column major format
        for(int j = 0; j < columns; ++j) {
            hTmpData[j * rows + rowNum] = vector[j];
        }
    }

    if (dataType == DataType::FP16)
    {
        dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(half));
        astdl::math::kFloatToHalf(hTmpData, hTmpDataBytes, (half*) dCollectionPartitions[device].data);

        dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(half));
        dInputBatchTmpBuffers[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
    } else
    {
        dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(float));
        CHECK_ERR(cudaMemcpy(dCollectionPartitions[device].data, hTmpData, hTmpDataBytes, cudaMemcpyHostToDevice));
        dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
    }

    free(hTmpData);

    dProducts[device] = allocateMatrixOnDevice(batchSize, rows, sizeof(float));
    dResultScores[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(float));
    dResultIndexes[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(uint32_t));

    hResultScores[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(float));
    hResultIndexes[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(uint32_t));

    size_t totalMemory;
    size_t freeMemory;
    astdl::cuda_util::getDeviceMemoryInfoInMb(device, &totalMemory, &freeMemory);

    fprintf(
        stderr,
        "INFO: loaded %zu (%zu padded) rows and %d columns into device %d. Used: %zu MB, Free: %zu MB, Total: %zu MB\n",
        actualRows, rows, columns, device, totalMemory - freeMemory, freeMemory, totalMemory);
}

void KnnData::load(const std::map<int, DataReader*> &deviceToData)
{
    omp_set_num_threads(numGpus);
#pragma omp parallel
    {
        int device = omp_get_thread_num();
        auto dataReader = deviceToData.find(device);
        if (dataReader == deviceToData.end())
        {
            std::stringstream msg;
            msg << "Data reader for device " << device << " not specified. Must specify readers for all " << numGpus
                << " devices";
            throw std::runtime_error(msg.str());
        }

        load(device, dataReader->second);
    }
}

void KnnData::load(const std::map<int, std::string> &deviceToFile, char keyValDelim, char vecDelim)
{
    std::map<int, DataReader*> deviceToData;
    for (const auto &entry : deviceToFile)
    {
        int device = entry.first;
        std::string file = entry.second;
        DataReader *dataReader = new TextFileDataReader(file, keyValDelim, vecDelim);
        deviceToData.insert( { device, dataReader });
    }

    load(deviceToData);

    for (const auto &entry : deviceToData)
    {
        delete entry.second;
    }
    deviceToData.clear();
}

KnnData::~KnnData()
{
    for (auto handle : cublasHandles)
    {
        cublasDestroy(handle);
    }
    for (auto dCollection : dCollectionPartitions)
    {
        freeMatrix(dCollection);
    }
    for (auto dInputBatch : dInputBatches)
    {
        freeMatrix(dInputBatch);
    }
    for (auto dProduct : dProducts)
    {
        freeMatrix(dProduct);
    }
    for (auto dResultScore : dResultScores)
    {
        freeMatrix(dResultScore);
    }
    for (auto dResultIndex : dResultIndexes)
    {
        freeMatrix(dResultIndex);
    }
    for (auto hResultScore : hResultScores)
    {
        freeMatrix(hResultScore);
    }
    for (auto hResultIndex : hResultIndexes)
    {
        freeMatrix(hResultIndex);
    }
    for (auto hKey : hKeys)
    {
        hKey.clear();
    }

    for (auto dInputBatchTmpBuffer : dInputBatchTmpBuffers)
    {
        freeMatrix(dInputBatchTmpBuffer);
    }

    cublasHandles.clear();
    dCollectionPartitions.clear();
    dInputBatches.clear();
    dProducts.clear();
    dResultScores.clear();
    dResultIndexes.clear();
    hKeys.clear();
    elapsedSgemm.clear();
    elapsedTopK.clear();
}

Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize)
{
    void *data = malloc(numRows * numColumns * elementSize);
    return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeHost);
}

Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize)
{
    void *data;
    CHECK_ERR(cudaMalloc(&data, numRows * numColumns * elementSize));
    return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeDevice);
}

void freeMatrix(const Matrix &matrix)
{
    if (matrix.data != nullptr)
    {
        switch (matrix.memoryType) {
            case cudaMemoryTypeDevice:
                CHECK_ERR(cudaFree(matrix.data))
                break;
            case cudaMemoryTypeHost:
                free(matrix.data);
                break;
            default:
                std::stringstream msg;
                msg << "Unknown memory type " << matrix.memoryType;
                throw std::invalid_argument(msg.str());
        }
    }
}
}  // namespace knn
}  // namespace astdl

