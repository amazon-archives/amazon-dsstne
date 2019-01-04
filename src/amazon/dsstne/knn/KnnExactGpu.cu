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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sstream>

#include "cudautil.h"
#include "KnnExactGpu.h"
#include "MathUtil.h"
#include "topk.h"

namespace astdl
{
namespace knn
{

Knn::Knn(KnnData *data) :
    data(data)
{
}

KnnExactGpu::KnnExactGpu(KnnData *data) :
    Knn(data)
{
}

void KnnExactGpu::search(int k, const float *inputs, int size, std::string *keys, float *scores)
{
  int maxK = data->maxK;
  int batchSize = data->batchSize;
  int numGpus = data->numGpus;

  if (k > maxK)
  {
    std::stringstream msg;
    msg << "k = " << k << " is > maxK = " << maxK;
    throw std::invalid_argument(msg.str());
  }

  if(size > batchSize)
  {
      std::stringstream msg;
      msg << "size = " << size << " is > batchSize = " << batchSize;
  }

  // only process "size" (subset) of batch
  batchSize = size;

  // results from each GPU
  std::vector<float*> allScores(numGpus);
  std::vector<uint32_t*> allIndexes(numGpus);

  omp_set_num_threads(numGpus);
#pragma omp parallel
  {
    int device = omp_get_thread_num();
    CHECK_ERR(cudaSetDevice(device));

    cublasHandle_t handle = data->cublasHandles[device];
    Matrix dCollectionPartition = data->dCollectionPartitions[device];
    Matrix dInputBatch = data->dInputBatches[device];
    Matrix dProducts = data->dProducts[device];
    Matrix dResultScores = data->dResultScores[device];
    Matrix dResultIndexes = data->dResultIndexes[device];
    Matrix hResultScores = data->hResultScores[device];
    Matrix hResultIndexes = data->hResultIndexes[device];
    uint32_t paddedRows = data->collectionRowsPadded[device];

    void *dA = dCollectionPartition.data; // asin x feature (M x K)
    void *dB = dInputBatch.data; // query x feature (N x K)
    void *dC = dProducts.data; // (scores x query)^t (M x N)

    float *dScores = (float*) dResultScores.data;
    uint32_t *dIndexes = (uint32_t*) dResultIndexes.data;

    float *hScores = (float*) hResultScores.data;
    uint32_t *hIndexes = (uint32_t*) hResultIndexes.data;

    uint32_t aRows = dCollectionPartition.numRows; // asin size (M)
    uint32_t bRows = batchSize; // batch size (N); size == dInputBatch.numRows if we are using the full batch
    uint32_t cRows = batchSize; // batch size (N); size == dProduct.numRows if we are using the full batch
    int aColumns = dCollectionPartition.numColumns; // feature size (K)
    int bColumns = dInputBatch.numColumns; // feature size (K)
    int cColumns = dProducts.numColumns; // asin size (M)


    cudaDataType aType;
    cudaDataType bType;
    cudaDataType cType = CUDA_R_32F;

    // copy input vectors to device
    if (data->dataType == astdl::knn::DataType::FP16)
    {
      aType = CUDA_R_16F;
      bType = CUDA_R_16F;

      Matrix tmpBuffer = data->dInputBatchTmpBuffers[device];
      astdl::math::kFloatToHalf(inputs, dInputBatch.getLength() * sizeof(float), (half*) dB, (float*) tmpBuffer.data,
          tmpBuffer.getSizeInBytes());
    } else if(data->dataType == astdl::knn::DataType::FP32)
    { // fp32
      aType = CUDA_R_32F;
      bType = CUDA_R_32F;
      CHECK_ERR(cudaMemcpy(dB, inputs, dInputBatch.getSizeInBytes(), cudaMemcpyHostToDevice));
    } else {
      throw std::runtime_error("Uknown data type");
    }

    static const cublasOperation_t transa = CUBLAS_OP_N;
    static const cublasOperation_t transb = CUBLAS_OP_N;
    static const float alpha = 1.0f;
    static const float beta = 0.0f;

    cudaEvent_t start, stop;
    float elapsed;
    CHECK_ERR(cudaEventCreate(&start));
    CHECK_ERR(cudaEventCreate(&stop));
    CHECK_ERR(cudaEventRecord(start, 0));
    STATUS_ERR(
        cublasSgemmEx(handle, transa, transb, aRows, bRows, aColumns, &alpha, dA, aType, aRows, dB, bType,
            bColumns, &beta, dC, cType, cColumns));
    CHECK_ERR(cudaEventRecord(stop, 0));
    CHECK_ERR(cudaEventSynchronize(stop));
    CHECK_ERR(cudaEventElapsedTime(&elapsed, start, stop));
    data->elapsedSgemm[device] = elapsed;

    CHECK_ERR(cudaEventRecord(start, 0));
    kCalculateTopK((float*) dC, dScores, dIndexes, cRows, cColumns, paddedRows, maxK);
    CHECK_ERR(cudaEventRecord(stop, 0));
    CHECK_ERR(cudaEventSynchronize(stop));
    CHECK_ERR(cudaEventElapsedTime(&elapsed, start, stop));
    data->elapsedTopK[device] = elapsed;

    // copy top k back to host
    CHECK_ERR(cudaMemcpy(hScores, dScores, hResultScores.getSizeInBytes(), cudaMemcpyDeviceToHost));
    CHECK_ERR(cudaMemcpy(hIndexes, dIndexes, hResultIndexes.getSizeInBytes(), cudaMemcpyDeviceToHost));

    allScores[device] = hScores;
    allIndexes[device] = hIndexes;
  }

  mergeKnn(k, batchSize, maxK, numGpus, allScores, allIndexes, data->hKeys, scores, keys);
}

void mergeKnn(int k, int batchSize, int width, int numGpus, const std::vector<float*> &allScores,
    const std::vector<uint32_t*> &allIndexes, const std::vector<std::vector<std::string>> &allKeys, float *scores,
    std::string *keys)
{

  for (int i = 0; i < batchSize; ++i)
  {
    // keeps track of the index if each array as we perform a merge; initialize to rowStartIdx
    int posIdxs[numGpus];
    for (int n = 0; n < numGpus; n++)
    {
      posIdxs[n] = i * width;
    }
    for (int col = 0; col < k; col++)
    {
      // initialize minValue and minIndex to the posIdx of GPU 0 results
      int deviceId_0 = 0;
      int posIdx_0 = posIdxs[deviceId_0];
      float maxVal = allScores[deviceId_0][posIdx_0];
      uint32_t maxIdx = allIndexes[deviceId_0][posIdx_0];
      int maxDeviceId = deviceId_0;

      for (int deviceId = 0; deviceId < numGpus; deviceId++)
      {
        int posIdx = posIdxs[deviceId];
        if (maxVal < allScores[deviceId][posIdx])
        {
          maxVal = allScores[deviceId][posIdx];
          maxIdx = allIndexes[deviceId][posIdx];
          maxDeviceId = deviceId;
        }
      }
      ++posIdxs[maxDeviceId];
      scores[i * k + col] = maxVal;
      keys[i * k + col] = allKeys[maxDeviceId][maxIdx];
    }
  }
}
} // namespace knn
} // namespace astdl
