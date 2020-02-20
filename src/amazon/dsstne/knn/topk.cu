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

#include "../engine/bitonic.h"
#include "topk.h"
#include <limits>

/*
 * TODO use the bitonic sorter from amazon/dsstne/engine/bitonic.h
 * The reason why we have two versions is that this version accounts
 * for row padding which is required to make rows a multiple of 4 or 8
 * to enable cublas to run the fp16 sgemm kernels on tensorcores.
 */

static __global__ void
LAUNCH_BOUNDS()
kCalculateTopK_kernel(NNFloat* pOutputBuffer, NNFloat* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch,
    uint32_t width, uint32_t widthPadding, uint32_t k)
{
__shared__ volatile NNFloat sKey[160 * 4];
__shared__ volatile uint32_t sValue[160 * 4];

    uint32_t dataWidth = width - widthPadding;
    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t tgx                    = threadIdx.x & 31;


    if (pos < batch)
    {
        NNFloat *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> 5;
        volatile NNFloat* psKey     = &sKey[160 * offset];
        volatile uint32_t* psValue  = &sValue[160 * offset];

        // Initialize values to
        NNFloat k0                  = -MAX_VALUE;
        NNFloat k1                  = -MAX_VALUE;
        NNFloat k2                  = -MAX_VALUE;
        NNFloat k3                  = -MAX_VALUE;
        NNFloat k4                  = -MAX_VALUE;
        NNFloat k5                  = -MAX_VALUE;
        NNFloat k6                  = -MAX_VALUE;
        NNFloat k7                  = -MAX_VALUE;
        uint32_t v0                 = 0;
        uint32_t v1                 = 0;
        uint32_t v2                 = 0;
        uint32_t v3                 = 0;
        uint32_t v4                 = 0;
        uint32_t v5                 = 0;
        uint32_t v6                 = 0;
        uint32_t v7                 = 0;

        // Read first 128 elements into registers
        uint32_t wpos               = tgx;
        if (wpos < dataWidth)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += 32;
        if (wpos < dataWidth)
        {
            k1                      = pOutput[wpos];
            v1                      = wpos;
        }
        wpos                       += 32;
        if (wpos < dataWidth)
        {
            k2                      = pOutput[wpos];
            v2                      = wpos;
        }
        wpos                       += 32;
        if (wpos < dataWidth)
        {
            k3                      = pOutput[wpos];
            v3                      = wpos;
        }

        // Run through remainder of data
        NNFloat minValue            = -MAX_VALUE;
        uint32_t rpos               = 128;
        uint32_t bufferSize         = 0;
        NNFloat key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < dataWidth)
        {
            // Read block of data
            unsigned wpos           = rpos + tgx;
            NNFloat key             = -MAX_VALUE;
            uint32_t value          = wpos;
            if (wpos < dataWidth)
            {
                key                 = pOutput[wpos];
            }

            // Add values > minValue to shared memory buffer
            uint32_t count          = __ballot(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            // Check if buffer is full
            if (bufferSize >= 128)
            {
                // Sort 256 elements
                k4                  = psKey[tgx];
                v4                  = psValue[tgx];
                k5                  = psKey[tgx + 32];
                v5                  = psValue[tgx + 32];
                k6                  = psKey[tgx + 2 * 32];
                v6                  = psValue[tgx + 2 * 32];
                k7                  = psKey[tgx + 3 * 32];
                v7                  = psValue[tgx + 3 * 32];
                bool flag;
                BITONICSORT256_256();

                // Set minValue to the new min in the warp queue (registers sorted in descending order)
                // last register in the last lane (31) holds the smallest value
                minValue = SHFL(k3, 31);

                // Shift members in shared memory to beginning
                bufferSize         -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 128];
                    psValue[tgx]    = psValue[tgx + 128];
                }
            }

            // Advance to next block of data
            rpos                    += 32;
        }

        // Do final sort if buffer has any remaining data
        if ((bufferSize > 0) || (dataWidth <= 128))
        {
            // Store sentinel values in registers
            k4                       = -MAX_VALUE;
            k5                       = -MAX_VALUE;
            k6                       = -MAX_VALUE;
            k7                       = -MAX_VALUE;
            v4                       = 0;
            v5                       = 0;
            v6                       = 0;
            v7                       = 0;

            // Load last block of unsorted data into registers
            if (tgx < bufferSize)
            {
                k4                   = psKey[tgx];
                v4                   = psValue[tgx];
            }
            if (tgx + 32 < bufferSize)
            {
                k5                   = psKey[tgx + 32];
                v5                   = psValue[tgx + 32];
            }
            if (tgx + 2 * 32 < bufferSize)
            {
                k6                   = psKey[tgx + 2 * 32];
                v6                   = psValue[tgx + 2 * 32];
            }
            if (tgx + 3 * 32 < bufferSize)
            {
                k7                   = psKey[tgx + 3 * 32];
                v7                   = psValue[tgx + 3 * 32];
            }

            BITONICSORT256_256();
        }

        // Copy results to key and value pointers
        NNFloat* pKey                = pKeyBuffer + pos * k;
        uint32_t* pValue             = pValueBuffer + pos * k;
        wpos                         = tgx;
        if (wpos < k)
        {
            pKey[wpos]               = k0;
            pValue[wpos]             = v0;
        }
        wpos                        += 32;
        if (wpos < k)
        {
            pKey[wpos]               = k1;
            pValue[wpos]             = v1;
        }
        wpos                        += 32;
        if (wpos < k)
        {
            pKey[wpos]               = k2;
            pValue[wpos]             = v2;
        }
        wpos                        += 32;
        if (wpos < k)
        {
            pKey[wpos]               = k3;
            pValue[wpos]             = v3;
        }
    }
}

void kCalculateTopK(NNFloat* pOutput, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k)
{
    uint32_t blocks                 = (batch + 3) / 4;
    kCalculateTopK_kernel<<<blocks, 128>>>(pOutput, pKey, pValue, batch, width, widthPadding, k);
    LAUNCHERROR("kCalculateTopK_kernel");
}
