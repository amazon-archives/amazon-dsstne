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

#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cudautil.h"

namespace astdl
{
namespace math
{
/**
 * Copies size elements from src[0...size] to dst[0...size], converting each float value
 * in src to half in dst.
 */
__global__ void kFloatToHalf_kernel(const float *src, size_t length, half *dst)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length)
  {
    dst[idx] = __float2half(src[idx]);
  }
}

void kFloatToHalf(const float *hSource, size_t sourceSizeInBytes, half *dDest, float *dBuffer, size_t bufferSizeInBytes)
{
  if(sourceSizeInBytes % sizeof(float) != 0)
  {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(float)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0)
  {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }

  dim3 threads(128);
  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 blocks((bufferLen + (threads.x - 1)) / threads.x);

  size_t srcLeftBytes = sourceSizeInBytes;
  size_t offset = 0;

  while (srcLeftBytes > 0)
  {
    size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
    size_t cpyLength = cpyBytes / sizeof(float);

    CHECK_ERR(cudaMemcpy(dBuffer, hSource + offset, cpyBytes, cudaMemcpyHostToDevice));
    LAUNCH_ERR((kFloatToHalf_kernel<<<blocks, threads>>>(dBuffer, cpyLength, dDest + offset)));

    offset += cpyLength;
    srcLeftBytes -= cpyBytes;
  }
}

void kFloatToHalf(const float *hSource, size_t sourceSizeInBytes, half *dDest, size_t bufferSizeInBytes)
{
  float *dBuffer;
  CHECK_ERR(cudaMalloc(&dBuffer, bufferSizeInBytes));
  kFloatToHalf(hSource, sourceSizeInBytes, dDest, dBuffer, bufferSizeInBytes);
  CHECK_ERR(cudaFree(dBuffer));
}

__global__ void kHalfToFloat_kernel(const half *src, size_t length, float *dst)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length)
  {
    dst[idx] = __half2float(src[idx]);
  }
}

void kHalfToFloat(const half *dSource, size_t sourceSizeInBytes, float *hDest, size_t bufferSizeInBytes)
{
  if(sourceSizeInBytes % sizeof(half) != 0) {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(half)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0)
  {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }


  dim3 threads(128);
  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 blocks((bufferLen + (threads.x - 1)) / threads.x);

  float *dBuffer;
  CHECK_ERR(cudaMalloc(&dBuffer, bufferLen * sizeof(float)));

  size_t sourceLength = sourceSizeInBytes / sizeof(half);
  size_t srcLeftBytes = sourceLength * sizeof(float);
  size_t offset = 0;

  while (srcLeftBytes > 0)
  {
    size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
    size_t cpyLength = cpyBytes / sizeof(float);

    LAUNCH_ERR((kHalfToFloat_kernel<<<blocks, threads>>>(dSource + offset, cpyLength, dBuffer)));
    CHECK_ERR(cudaMemcpy(hDest + offset, dBuffer, cpyBytes, cudaMemcpyDeviceToHost));

    offset += cpyLength;
    srcLeftBytes -= cpyBytes;
  }

  CHECK_ERR(cudaFree(dBuffer));
}
} // namespace math
} // namespace astdl
