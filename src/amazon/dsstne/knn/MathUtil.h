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

#ifndef LIBKNN_MATHUTIL_H_
#define LIBKNN_MATHUTIL_H_

#include <cuda_fp16.h>

namespace astdl
{
namespace math
{
/**
 * Converts and loads the fp32 array on the host to fp16 array on device.
 * Temporarily allocates and frees a copy buffer on the device of length bufferSizeInBytes (must be a multiple of 4).
 */
void kFloatToHalf(const float *hSource, size_t sourceLength, half *dDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);

/**
 * Converts and loads the fp32 array on the host to fp16 array on device.
 * Uses the provided dBuffer on the device to copy the source floats on the host to device.
 */
void kFloatToHalf(const float *hSource, size_t sourceSizeInBytes, half *dDest, float *dBuffer, size_t bufferSizeInBytes);

/**
 * Converts and loads the fp16 array on device to fp32 array on host.
 * bufferSizeInBytes MUST be multiples of 4.
 */
void kHalfToFloat(const half *dSource, size_t sourceLength, float *hDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);
} // namespace math
} // namespace astdl
#endif /* LIBKNN_MATHUTIL_H_ */
