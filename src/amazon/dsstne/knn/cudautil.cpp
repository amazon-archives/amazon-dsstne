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

#include <cuda_runtime.h>

#include "cudautil.h"

namespace astdl
{
namespace cuda_util
{
void printMemInfo(const char *header)
{
  int bytesInMb = 1024 * 1024;
  int device;
  size_t free;
  size_t total;
  CHECK_ERR(cudaGetDevice(&device));
  CHECK_ERR(cudaMemGetInfo(&free, &total));

  long freeMb = free / bytesInMb;
  long usedMb = (total - free) / bytesInMb;
  long totalMb = total / bytesInMb;

  printf("--%-50s GPU [%d] Mem Used: %-6ld MB. Free: %-6ld MB. Total: %-6ld MB\n", header, device, usedMb, freeMb,
      totalMb);
}

void getDeviceMemoryInfoInMb(int device, size_t *total, size_t *free) {
  static const int bytesInMb = 1024 * 1024;
  size_t freeInBytes;
  size_t totalInBytes;
  CHECK_ERR(cudaGetDevice(&device));
  CHECK_ERR(cudaMemGetInfo(&freeInBytes, &totalInBytes));
  *total = totalInBytes / bytesInMb;
  *free = freeInBytes / bytesInMb;
}

int getDeviceCount()
{
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "** ERROR (%d - %s) calling cudaGetDeviceCount()."
        " The host probably does not have any GPUs or the driver is not installed."
        " Returning -1\n", err, cudaGetErrorString(err));
    return -1;
  } else
  {
    return deviceCount;
  }
}

bool hasGpus()
{
  int deviceCount = getDeviceCount();
  return deviceCount > 0;
}

} // namespace cudautil
} // namespace astdl

