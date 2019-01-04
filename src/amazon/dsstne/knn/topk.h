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

#ifndef LIBKNN_TOPK_H_
#define LIBKNN_TOPK_H_

#include <stdint.h>
#include <stdio.h>

#define NNFloat float

static const float MAX_VALUE        = 999999999999999.0f;


static const int SM_3X_THREADS_PER_BLOCK                        = 128;
static const int SM_5X_THREADS_PER_BLOCK                        = 128;
static const int SM_6X_THREADS_PER_BLOCK                        = 128;

#if (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
    }
#endif

/*
 * TODO use kCalculateTopK from amazon/dsstne/engine/kernels.h
 * The reason why we have two versions is that this version accounts
 * for row padding which is required to make rows a multiple of 4 or 8
 * to enable cublas to run the fp16 sgemm kernels on tensorcores.
 */
void kCalculateTopK(NNFloat* pOutput, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k);

#endif /* LIBKNN_TOPK_H_ */
