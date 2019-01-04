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

#ifndef LIBKNN_CUDAUTIL_H_
#define LIBKNN_CUDAUTIL_H_

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static void CHECK_ERR2(cudaError_t e, const char *fname, int line)
{
  if (e != cudaSuccess)
  {
    fprintf(stderr, "FATAL ERROR: cuda failure(%d): %s in %s#%d\n", e, cudaGetErrorString(e), fname,
        line);
    exit(-1);
  }
}

static void STATUS_ERR2(cublasStatus_t e, const char *fname, int line)
{
  if (e != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "FATAL ERROR: cublas failure %d in %s#%d\n", e, fname, line);
    exit(-1);
  }
}

static void LAUNCH_ERR2(const char *kernelName, const char *fname, int line)
{
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
  {
    fprintf(stderr, "FATAL ERROR: %s launching kernel: %s\n in %s#%d\n", cudaGetErrorString(e), kernelName, fname, line);
    exit(-1);
  }
}

#define CHECK_ERR(e) {CHECK_ERR2(e, __FILE__, __LINE__);}

#define STATUS_ERR(e) {STATUS_ERR2(e, __FILE__, __LINE__);}

#define LAUNCH_ERR(expression) { \
    expression; \
    LAUNCH_ERR2(#expression, __FILE__, __LINE__); \
  }

namespace astdl
{
namespace cuda_util
{
void printMemInfo(const char *header = "");

void getDeviceMemoryInfoInMb(int device, size_t *total, size_t *free);

int getDeviceCount();

/**
 * Returns true if the host has GPUs, false otherwise.
 */
bool hasGpus();

} // namespace cuda
} // namespace astdl

/**
 * When used as the first statement of a function, returns immediately
 * if the host does not have any GPUs.
 */
#define REQUIRE_GPU if(!astdl::cuda_util::hasGpus()) return;

/**
 * When used as the first statement of a function, returns immediately
 * if the host does not have at least numGpus.
 */
#define REQUIRE_GPUS(numGpus) if(astdl::cuda_util::getDeviceCount() < numGpus) return;

#endif /* LIBKNN_CUDAUTIL_H_ */
