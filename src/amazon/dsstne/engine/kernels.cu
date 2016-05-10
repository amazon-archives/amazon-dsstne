/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include <limits>

static __constant__ GpuData cData;

__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

void SetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

void GetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy From cData failed");
}


uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

// Scales and biases a weight matrix previously generated
__global__ void
LAUNCH_BOUNDS()
kScaleAndBias_kernel(NNFloat* pData, uint64_t size, NNFloat scale, NNFloat bias)
{
    uint64_t offset             = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < size)
    {
        NNFloat value           = pData[offset];
        pData[offset]           = scale * value - bias;
    }
}

void kScaleAndBias(NNFloat* pData, uint64_t size, NNFloat scale, NNFloat bias)
{
    uint32_t blocks             = CalculateBlocks(size);
    kScaleAndBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size, scale, bias);
    LAUNCHERROR("kScaleAndBias_kernel");
}


// Initializes hidden or output unit with bias of single incoming unit
__global__ void
LAUNCH_BOUNDS()
kClearUnit_kernel(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]              = pBias[bpos];
    }
}


void kClearUnit(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = CalculateBlocks(size);
    kClearUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kClearUnit_kernel");
}

// Initializes hidden or output unit with biases of 2 incoming units
__global__ void
LAUNCH_BOUNDS()
kClearDualSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]              = pBias1[bpos] + pBias2[bpos];
    }
}

void kClearDualSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kClearDualSourceUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kClearDualSourceUnit_kernel");
}



// Initializes hidden or output unit with biases of 3 incoming units
__global__ void
LAUNCH_BOUNDS()
kClearTripleSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]              = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kClearTripleSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kClearTripleSourceUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kClearTripleSource_kernel");
}

// Initializes hidden or output unit with biases of 4 incoming units
__global__ void
LAUNCH_BOUNDS()
kClearQuadSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]              = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kClearQuadSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kClearQuadSourceUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kClearQuadSource_kernel");
}
__global__ void
LAUNCH_BOUNDS()
kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            uint64_t pos2               = offset + pSparseIndex[start];
            pUnit[pos2]                 = 1.0f;
            start                      += cData._warpSize;
        }
    }
}

void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{
    uint32_t last                       = position + batch;
    uint32_t count                      = last - position;
    uint32_t blocks                     = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaError_t status                  = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat));
    RTERROR(status, "kLoadSparseInputUnit failed");
    kLoadSparseInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
    LAUNCHERROR("kLoadSparseInputUnit_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            pUnit[pos2]                 = data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint32_t last                       = position + batch;
    uint32_t count                      = last - position;
    uint32_t blocks                     = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    cudaError_t status                  = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat));
    RTERROR(status, "kLoadSparseAnalogInputUnit failed");    
    kLoadSparseAnalogInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
    LAUNCHERROR("kLoadSparseAnalogInputUnit_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            NNFloat value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = cData._denoising_q;
            start                      += cData._warpSize;
        }
    }
}


void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pRandom)
{
    uint32_t last                       = position + batch;
    uint32_t count                      = last - position;
    uint32_t blocks                     = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    
    
    //printf("KLSPDU %u %u %u %u %lu %lu %lu %lu %lu\n", position, batch, stride, blocks, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pRandom);

    cudaError_t status                  = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat));
    RTERROR(status, "kLoadSparseDenoisedInputUnit failed");
    kLoadSparseDenoisedInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pRandom);
    LAUNCHERROR("kLoadSparseDenoisedInputUnit_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            NNFloat value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = cData._denoising_q * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T *pSparseData, NNFloat* pRandom)
{
    uint32_t last                       = position + batch;
    uint32_t count                      = last - position;
    uint32_t blocks                     = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaError_t status                  = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat));
    RTERROR(status, "kLoadSparseAnalogDenoisedInputUnit failed");
    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pRandom);
    LAUNCHERROR("kLoadSparseAnalogDenoisedInputUnit_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = pData[soffset];
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (NNFloat)pData[soffset] * (NNFloat)(1.0 / 256.0);
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos          = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (NNFloat)pData[soffset] * (NNFloat)(1.0 / 128.0);
    }
}

template<typename T> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadInputUnit_kernel");
}

template<> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadNormalizedInputUnit_kernel");
}

template<> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadNormalizedInputUnit_kernel");
}

// Adds bias from single incoming unit
__global__ void
LAUNCH_BOUNDS()
kAddBias_kernel(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t size)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias[bpos];
    }
}


void kAddBias(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size               = stride * batch;
    uint32_t blocks             = CalculateBlocks(size);
    kAddBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kAddBias_kernel");
}


// Adds biases of 2 incoming units to hidden or output unit
__global__ void
LAUNCH_BOUNDS()
kAddDualBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos];
    }
}

void kAddDualBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddDualBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kAddDualBias_kernel");
}

// Adds biases of 3 incoming units to hidden or output unit
__global__ void
LAUNCH_BOUNDS()
kAddTripleBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kAddTripleBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddTripleBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kAddTripleBias_kernel");
}

// Adds biases of 4 incoming units to hidden or output unit
__global__ void
LAUNCH_BOUNDS()
kAddQuadBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kAddQuadBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size               = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks             = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddQuadBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kAddQuadBias_kernel");
}

#if (__CUDA_ARCH__ >= 500)
static const uint32_t MAXSPARSE = SM_5X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_5X_MAXSPARSEANALOG;
#else
static const uint32_t MAXSPARSE = SM_3X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_3X_MAXSPARSEANALOG;
#endif


__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSE];                         // Shared set of offsets to non-zero weights

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseIndex[start] * stride;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            unit               += pWeight[offset + opos];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}


void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pUnit, beta);
    LAUNCHERROR("kCalculateSparseZ_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ T sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseIndex[start] * stride;
        sValue[pos]             = pSparseData[start];
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            unit               += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseIndex[start] * stride;
        sValue[pos]             = (NNFloat)pSparseData[start] * (NNFloat)(1.0 / 256.0);
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            unit               += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseIndex[start] * stride;
        sValue[pos]             = (NNFloat)pSparseData[start] * (NNFloat)(1.0 / 128.0);
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            unit               += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<typename T> void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseZ_kernel");
}

template<> void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseZ_kernel");
}

template<> void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseZ_kernel");
}

__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSE];                         // Shared set of offsets to non-zero weights

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        NNFloat value           = pRandom[start];
        sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[start] * stride;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            int32_t offset      = sOffset[i];
            if (offset != cData._maxUint32_t)
                unit           += pWeight[offset + opos] * cData._denoising_q;  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseDenoisedZ_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ T sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        NNFloat value           = pRandom[start];
        sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : pSparseIndex[start] * stride;
        sValue[pos]             = pSparseData[start] * cData._denoising_q;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            int32_t offset      = sOffset[i];
            if (offset != cData._maxUint32_t)
                unit           += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ int32_t sOffset[MAXSPARSEANALOG];                    // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        NNFloat value           = pRandom[start];
        sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : pSparseIndex[start] * stride;
        sValue[pos]             = (NNFloat)pSparseData[start] * (NNFloat)(1.0 / 256.0) * cData._denoising_q;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            int32_t offset      = sOffset[i];
            if (offset != cData._maxUint32_t)
                unit           += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        NNFloat value           = pRandom[start];
        sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : pSparseIndex[start] * stride;
        sValue[pos]             = (NNFloat)pSparseData[start] * (NNFloat)(1.0 / 128.0) * cData._denoising_q;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pUnit                      += blockIdx.x * stride;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < stride)
    {        
        // Read all non-zero inputs
        NNFloat unit            = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset      = sOffset[i];
            if (offset != cData._maxUint32_t)
                unit           += pWeight[offset + opos] * sValue[i];  
        }
        
        // Write output
        pUnit[opos]             = unit;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<typename T> void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}

template<> void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}

template<> void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads            = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    // Determine batch position
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    // Add indices to sparse transposed activation matrix
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseTransposedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
    LAUNCHERROR("kCalculateSparseTransposedMatrix_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    // Determine batch position
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    // Add indices to sparse transposed activation matrix
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            NNFloat rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
            }
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseTransposedDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
    LAUNCHERROR("kCalculateSparseTransposedDenoisedMatrix_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData)
{
    // Determine batch position
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    // Add indices to sparse transposed activation matrix
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            T value                         = pSparseData[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = value;
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData)
{
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseTransposedAnalogMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateSparseTransposedAnalogMatrix_kernel");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData)
{
    // Determine batch position
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    // Add indices to sparse transposed activation matrix
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            NNFloat rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value                     = pSparseData[start];
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData)
{
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseTransposedAnalogDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateSparseTransposedAnalogDenoisedMatrix_kernel");
}


__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pDelta, NNFloat* pWeightGradient)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSE];                         // Shared set of offsets to non-zero weights

    // Read transposed sparse indices into shared memory so they're only read once
    sOpos                       = blockDim.x; 
    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseTransposedIndex[start] * n;
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    alpha                      *= cData._denoising_q;
    pWeightGradient            += blockIdx.x * n;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < n)
    {                
        // Read all non-zero inputs, accumulate in 64-bit FP to maintain deterministic results
        NNFloat oldgradient     = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : beta * pWeightGradient[opos];
        int64_t sum             = 0;
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            sum                += llrintf(ERRORSCALEF * pDelta[offset + opos]);  
        }
        
        // Write output
        NNFloat fsum            = alpha * (NNFloat)((double)sum * ONEOVERERRORSCALE);
        pWeightGradient[opos]   = oldgradient + fsum;            
            
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}


void kCalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    uint32_t threads            = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseTransposedWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedWeightGradient_kernel");
}

template <typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ T sValue[MAXSPARSEANALOG];

    // Read transposed sparse indices and data into shared memory so they're only read once
    sOpos                       = blockDim.x; 
    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    alpha                      *= cData._denoising_q;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseTransposedIndex[start] * n;
        sValue[pos]             = pSparseTransposedData[start];
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pWeightGradient            += blockIdx.x * n;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < n)
    {        
        // Read all non-zero inputs, accumulate in 64-bit FP to maintain deterministic results
        NNFloat oldgradient     = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : beta * pWeightGradient[opos];
        int64_t sum             = 0;
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            T value             = sValue[i];
            sum                += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);  
        }
        
        // Write output
        NNFloat fsum            = alpha * (NNFloat)((double)sum * ONEOVERERRORSCALE);
        pWeightGradient[opos]   = oldgradient + fsum;        
        
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template <>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, char* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read transposed sparse indices and data into shared memory so they're only read once
    sOpos                       = blockDim.x; 
    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    alpha                      *= cData._denoising_q;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseTransposedIndex[start] * n;
        sValue[pos]             = (NNFloat)pSparseTransposedData[start] * (NNFloat)(1.0 / 128.0);
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pWeightGradient            += blockIdx.x * n;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < n)
    {        
        // Read all non-zero inputs, accumulate in 64-bit FP to maintain deterministic results 
        NNFloat oldgradient     = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : beta * pWeightGradient[opos];
        int64_t sum             = 0;
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            NNFloat value       = sValue[i];
            sum                += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);  
        }
        
        // Write output
        NNFloat fsum            = alpha * (NNFloat)((double)sum * ONEOVERERRORSCALE);
        pWeightGradient[opos]   = oldgradient + fsum;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template <>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, unsigned char* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
__shared__ uint32_t sOpos;                                      // Shared output position
__shared__ uint32_t sOffset[MAXSPARSEANALOG];                   // Shared set of offsets to non-zero weights
__shared__ NNFloat sValue[MAXSPARSEANALOG];

    // Read transposed sparse indices and data into shared memory so they're only read once
    sOpos                       = blockDim.x; 
    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    uint32_t inputs             = end - start;
    uint32_t pos                = threadIdx.x;
    alpha                      *= cData._denoising_q;
    start                      += threadIdx.x;
    while (start < end)
    {
        sOffset[pos]            = pSparseTransposedIndex[start] * n;
        sValue[pos]             = (NNFloat)pSparseTransposedData[start] * (NNFloat)(1.0 / 256.0);
        pos                    += blockDim.x;
        start                  += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    // Cycle through all output positions
    pWeightGradient            += blockIdx.x * n;
    uint32_t opos               = threadIdx.x;
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    while (opos < n)
    {        
        // Read all non-zero inputs, accumulate in 64-bit FP to maintain deterministic results
        NNFloat oldgradient     = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : beta * pWeightGradient[opos];
        int64_t sum             = 0;
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset     = sOffset[i];
            NNFloat value       = sValue[i];
            sum                += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);  
        }
        
        // Write output
        NNFloat fsum            = alpha * (NNFloat)((double)sum * ONEOVERERRORSCALE);
        pWeightGradient[opos]   = oldgradient + fsum;
    
        // Advance to next set of outputs
        if (tgx == 0)
        {
            opos                = atomicAdd(&sOpos, cData._warpSize);
        }
        opos                    = __shfl(opos, 0);
        opos                   += tgx;
    }
}

template<typename T> 
void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    uint32_t threads            = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);    
    kCalculateSparseTransposedAnalogWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedAnalogWeightGradient_kernel");
}

template<> 
void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, char* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    uint32_t threads            = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseTransposedAnalogWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedAnalogWeightGradient_kernel");
}

template<> 
void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, unsigned char* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    uint32_t threads            = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseTransposedAnalogWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedAnalogWeightGradient_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kUpdateBiases_kernel(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = (NNFloat)0.0;
        pDelta                 += pos;
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        pBias[pos]             -= alpha * sum;
    }
}

void kUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    LAUNCHERROR("kUpdateBiases_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateRegularizationError_kernel(NNFloat* pWeight, uint64_t size)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
        NNFloat w               = pWeight[pos];
        error                   = w * w;   
    }

    // Reduce error across threads
    error                      += __shfl(error, threadIdx.x ^ 1);
    error                      += __shfl(error, threadIdx.x ^ 2);
    error                      += __shfl(error, threadIdx.x ^ 4);
    error                      += __shfl(error, threadIdx.x ^ 8);
    error                      += __shfl(error, threadIdx.x ^ 16);

    if ((threadIdx.x & cData._warpMask) == 0)
    {
        atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error)));
    }  
}

// Calculates raw weight decay/regularization error
NNFloat kCalculateRegularizationError(NNFloat lambda, NNFloat* pWeight, uint64_t size)
{
    uint32_t blocks         = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    kCalculateRegularizationError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pWeight, size);
    LAUNCHERROR("kCalculateRegularizationError_kernel");
    getGpu()._pbAccumulator->Download(); 
    //printf("Reg %llu %f\n", size, lambda * 0.5f * (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
    return (NNFloat)(lambda * 0.5f * (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

// Instantiates allowable templated functions so we can hide the implementations here
// instead of in the header file because we're mixing CUDA and C++ and that's
// a migraine headache in the making otherwise.
void KernelsTempFunction()
{

    kInitSort<NNFloat, NNFloat>(1, NULL, NULL);
    kInitSort<uint32_t, NNFloat>(1, NULL, NULL);
    kInitSort<NNFloat, uint32_t>(1, NULL, NULL);
    kInitSort<uint32_t, uint32_t>(1, NULL, NULL);
    kSort<NNFloat, NNFloat>(1, NULL, NULL, NULL, NULL, NULL, 0);
    kSort<NNFloat, uint32_t>(1, NULL, NULL, NULL, NULL, NULL, 0);
    kSort<uint32_t, NNFloat>(1, NULL, NULL, NULL, NULL, NULL, 0);
    kSort<uint32_t, uint32_t>(1, NULL, NULL, NULL, NULL, NULL, 0);
    
    kLoadSparseAnalogDenoisedInputUnit<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogDenoisedInputUnit<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    
    kLoadSparseAnalogInputUnit<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kLoadSparseAnalogInputUnit<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);    

    kCalculateSparseAnalogZ<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogZ<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0); 
    
    kCalculateSparseAnalogDenoisedZ<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);
    kCalculateSparseAnalogDenoisedZ<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, (NNFloat)0.0);    
   
    kCalculateSparseTransposedAnalogMatrix<NNFloat>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<double>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<unsigned char>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<char>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<uint32_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<uint64_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<int32_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogMatrix<int64_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    
    kCalculateSparseTransposedAnalogDenoisedMatrix<NNFloat>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<double>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<unsigned char>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<char>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<uint32_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<uint64_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<int32_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogDenoisedMatrix<int64_t>(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);    
    
    kCalculateSparseTransposedAnalogWeightGradient<NNFloat>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<double>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<unsigned char>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<char>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<uint32_t>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<uint64_t>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<int32_t>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseTransposedAnalogWeightGradient<int64_t>(0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL);    
    
    kLoadInputUnit<NNFloat>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<double>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<unsigned char>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<char>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<uint32_t>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<uint64_t>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<int32_t>(0, 0, 0, NULL, NULL);
    kLoadInputUnit<int64_t>(0, 0, 0, NULL, NULL); 
}


__global__ void
LAUNCH_BOUNDS()
kSGDUpdateWeights_kernel(NNFloat lambda, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat g               = pWeightGradient[pos];
        NNFloat w               = pWeight[pos];
        pWeight[pos]            = w + g - lambda * w;
    }
}

void kSGDUpdateWeights(NNFloat lambda, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint32_t blocks             = CalculateBlocks(size);
    kSGDUpdateWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(lambda, size, pWeightGradient, pWeight);
    LAUNCHERROR("kMomentumUpdateWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kSGDUpdateBiases_kernel(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = 0.0f;
        pDelta                 += pos;

        // Calculate bias gradient
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }

        // Update velocity and bias
        NNFloat bias            = pBias[pos];
        pBias[pos]              = bias + alpha * sum;
    }
}

void kSGDUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kSGDUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    LAUNCHERROR("kSGDUpdateBiases_kernel");
}


__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateWeights_kernel(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat g               = pWeightGradient[pos];
        NNFloat w               = pWeight[pos];
        NNFloat v               = pWeightVelocity[pos];
        v                       = mu * v + g - lambda * w;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + v;
    }
}

void kMomentumUpdateWeights(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint32_t blocks             = CalculateBlocks(size);
    kMomentumUpdateWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(lambda, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    LAUNCHERROR("kMomentumUpdateWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateBiases_kernel(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = 0.0f;
        pDelta                 += pos;

        // Calculate bias gradient
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }

        // Update velocity and bias
        NNFloat v               = pBiasVelocity[pos];
        v                       = mu * v - alpha * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             += v;
    }
}

void kMomentumUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kMomentumUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    LAUNCHERROR("kMomentumUpdateBiases_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateWeights_kernel(NNFloat alpha, NNFloat lambda, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat g               = pWeightGradient[pos];
        NNFloat w               = pWeight[pos];
        NNFloat v               = pWeightVelocity[pos];
        g                      -= lambda * w;
        v                      += g * g;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateWeights(NNFloat alpha, NNFloat lambda, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    unsigned long blocks        = CalculateBlocks(size);
    kAdaGradUpdateWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, lambda, size, pWeightVelocity, pWeightGradient, pWeight);
    LAUNCHERROR("kAdaGradUpdateWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateBiases_kernel(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = 0.0f;
        pDelta                 += pos;

        // Calculate bias gradient
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }

        // Update velocity and bias
        NNFloat v               = pBiasVelocity[pos];
        v                      += sum * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kAdaGradUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, batch, width, pDelta, pBiasVelocity, pBias);
    LAUNCHERROR("kAdaGradUpdateBiases_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kNesterovUpdateWeights_kernel(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat g               = pWeightGradient[pos];
        NNFloat w               = pWeight[pos];
        NNFloat vOld            = pWeightVelocity[pos];
        NNFloat vNew            = mu * vOld + g - lambda * w;
        pWeightVelocity[pos]    = vNew;
        w                       = w + vNew + mu * (vNew - vOld);
        pWeight[pos]            = w;      
    }
}

void kNesterovUpdateWeights(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint32_t blocks             = CalculateBlocks(size);
    kNesterovUpdateWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(lambda, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    LAUNCHERROR("kNesterovUpdateWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kNesterovUpdateBiases_kernel(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = 0.0f;
        pDelta                 += pos;

        // Calculate bias gradient
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }

        // Update velocity and bias
        NNFloat vOld            = pBiasVelocity[pos];
        NNFloat vNew            = mu * vOld - alpha * sum;
        pBiasVelocity[pos]      = vNew;
        pBias[pos]             += vNew + mu * (vNew - vOld);
    }
}

void kNesterovUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kNesterovUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    LAUNCHERROR("kNesterovUpdateBiases_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kNesterovShiftWeights_kernel(NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = pWeight[pos];
        NNFloat v               = pWeightVelocity[pos];
        pWeight[pos]            = w + mu * v;
    }
}

void kNesterovShiftWeights(NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeight)
{
    uint32_t blocks             = CalculateBlocks(size);
    kNesterovShiftWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(mu, size, pWeightVelocity, pWeight);
    LAUNCHERROR("kNesterovShiftWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kNesterovShiftBiases_kernel(NNFloat mu, uint32_t width, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat b               = pBias[pos];
        NNFloat v               = pBiasVelocity[pos];
        pBias[pos]              = b + mu * v;
    }
}

void kNesterovShiftBiases(NNFloat mu, uint32_t width, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kNesterovShiftBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(mu, width, pBiasVelocity, pBias);
    LAUNCHERROR("kNesterovShiftBiases_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kRMSPropUpdateWeights_kernel(NNFloat alpha, NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint64_t pos  = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat g               = pWeightGradient[pos];
        NNFloat w               = pWeight[pos];
        NNFloat v               = pWeightVelocity[pos];
        g                      -= lambda * w;
        v                       = mu * v + (1.0f - mu) * g * g;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight)
{
    uint32_t blocks             = CalculateBlocks(size);
    kRMSPropUpdateWeights_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, lambda, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    LAUNCHERROR("kRMSPropUpdateWeights_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kRMSPropUpdateBiases_kernel(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        NNFloat sum             = 0.0f;
        pDelta                 += pos;

        // Calculate bias gradient
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }

        // Update velocity and bias
        NNFloat v               = pBiasVelocity[pos];
        v                       = mu * v + (1.0f - mu) * sum * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias)
{
    uint32_t blocks             = CalculateBlocks(width);
    kRMSPropUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    LAUNCHERROR("kRMSPropUpdateBiases_kernel");
}

#include "bitonic.h"
__global__ void
LAUNCH_BOUNDS()
kCalculateTopK_kernel(NNFloat* pOutputBuffer, NNFloat* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile NNFloat sKey[160 * 4];
__shared__ volatile uint32_t sValue[160 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
        
    
    if (pos < batch)
    {
        NNFloat *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
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
        if (wpos < width)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutput[wpos];
            v1                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k2                      = pOutput[wpos];
            v2                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
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
        while (rpos < width)
        {
            // Read block of data
            unsigned wpos           = rpos + tgx;
            NNFloat key             = -MAX_VALUE;
            uint32_t value          = wpos;
            if (wpos < width)
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
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                // Shift members in shared memory to beginning
                bufferSize         -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 128];
                    psValue[tgx]    = psValue[tgx + 128];
                }
            }

            // Advance to next block of data
            rpos                    += cData._warpSize;
        }

        // Do final sort if buffer has any remaining data
        if ((bufferSize > 0) || (width < 128))
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
            if (tgx < bufferSize - cData._warpSize)
            {
                k5                   = psKey[tgx + cData._warpSize];
                v5                   = psValue[tgx + cData._warpSize];
            }
            if (tgx < bufferSize - 2 * cData._warpSize)
            {
                k6                   = psKey[tgx + 2 * cData._warpSize];
                v6                   = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx < bufferSize - 3 * cData._warpSize)
            {
                k7                   = psKey[tgx + 3 * cData._warpSize];
                v7                   = psValue[tgx + 3 * cData._warpSize];
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
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k1;
            pValue[wpos]             = v1;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k2;
            pValue[wpos]             = v2;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k3;
            pValue[wpos]             = v3;
        }
    }
}

void kCalculateTopK(NNFloat* pOutput, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t blocks                 = (batch + 3) / 4;
    kCalculateTopK_kernel<<<blocks, 128>>>(pOutput, pKey, pValue, batch, width, k);
    LAUNCHERROR("kCalculateTopK_kernel");
}


__global__ void
LAUNCH_BOUNDS()
kCalculateTopK_kernel(NNFloat* pOutputKey, NNFloat* pOutputValue, NNFloat* pKeyBuffer, NNFloat* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile NNFloat sKey[160 * 4];
__shared__ volatile NNFloat sValue[160 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
        
    
    if (pos < batch)
    {
        pOutputKey                 += pos * width;
        pOutputValue               += pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile NNFloat* psKey     = &sKey[160 * offset];
        volatile NNFloat* psValue   = &sValue[160 * offset];

        // Initialize values to 
        NNFloat k0                  = -MAX_VALUE;
        NNFloat k1                  = -MAX_VALUE;
        NNFloat k2                  = -MAX_VALUE;
        NNFloat k3                  = -MAX_VALUE;
        NNFloat k4                  = -MAX_VALUE;
        NNFloat k5                  = -MAX_VALUE;
        NNFloat k6                  = -MAX_VALUE;
        NNFloat k7                  = -MAX_VALUE;
        NNFloat v0                  = 0.0f;
        NNFloat v1                  = 0.0f;
        NNFloat v2                  = 0.0f;
        NNFloat v3                  = 0.0f;
        NNFloat v4                  = 0.0f;
        NNFloat v5                  = 0.0f;
        NNFloat v6                  = 0.0f;
        NNFloat v7                  = 0.0f;

        // Read first 128 elements into registers
        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutputKey[wpos];
            v0                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutputKey[wpos];
            v1                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k2                      = pOutputKey[wpos];
            v2                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k3                      = pOutputKey[wpos];
            v3                      = pOutputValue[wpos];
        }
     
        // Run through remainder of data
        NNFloat minValue            = -MAX_VALUE;
        uint32_t rpos               = 128;
        uint32_t bufferSize         = 0;
        NNFloat key1, key2;
        NNFloat value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            // Read block of data
            unsigned wpos           = rpos + tgx;
            NNFloat key             = -MAX_VALUE;
            NNFloat value           = 0.0f;
            if (wpos < width)
            {
                key                 = pOutputKey[wpos];
                value               = pOutputValue[wpos];              
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
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                // Shift members in shared memory to beginning
                bufferSize         -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 128];
                    psValue[tgx]    = psValue[tgx + 128];
                }
            }

            // Advance to next block of data
            rpos                   += cData._warpSize;
        }

        // Do final sort if buffer has any remaining data
        if ((bufferSize > 0) || (width < 128))
        {
            // Store sentinel values in registers
            k4                      = -MAX_VALUE;
            k5                      = -MAX_VALUE;
            k6                      = -MAX_VALUE;
            k7                      = -MAX_VALUE;
            v4                      = 0;
            v5                      = 0;
            v6                      = 0;
            v7                      = 0;

            // Load last block of unsorted data into registers
            if (tgx < bufferSize)
            {
                k4                  = psKey[tgx];
                v4                  = psValue[tgx];
            }
            if (tgx < bufferSize - cData._warpSize)
            {
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
            }
            if (tgx < bufferSize - 2 * cData._warpSize)
            {
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx < bufferSize - 3 * cData._warpSize)
            {
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        // Copy results to index and value pointers
        NNFloat* pKey               = pKeyBuffer + pos * k;
        NNFloat* pValue             = pValueBuffer + pos * k;                
        wpos                        = tgx;
        if (wpos < k)
        {
            pKey[wpos]              = k0;
            pValue[wpos]            = v0;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k1;
            pValue[wpos]            = v1;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k2;
            pValue[wpos]            = v2;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k3;
            pValue[wpos]            = v3;
        }
    }
}

void kCalculateTopK(NNFloat* pOutputKey, NNFloat* pOutputValue, NNFloat *pKey, NNFloat* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t blocks                 = (batch + 3) / 4;
    kCalculateTopK_kernel<<<blocks, 128>>>(pOutputKey, pOutputValue, pKey, pValue, batch, width, k);
    LAUNCHERROR("kCalculateTopK_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateTopK_kernel(NNFloat* pOutputKey, uint32_t* pOutputValue, NNFloat* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile NNFloat sKey[160 * 4];
__shared__ volatile uint32_t sValue[160 * 4];
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                        = threadIdx.x & cData._warpMask;        
    
    if (pos < batch)
    {
        pOutputKey                     += pos * width;
        pOutputValue                   += pos * width;
        uint32_t offset                 = threadIdx.x >> cData._warpBits;
        volatile NNFloat* psKey         = &sKey[160 * offset];
        volatile uint32_t* psValue      = &sValue[160 * offset];

        // Initialize values to 
        NNFloat k0                      = -MAX_VALUE;
        NNFloat k1                      = -MAX_VALUE;
        NNFloat k2                      = -MAX_VALUE;
        NNFloat k3                      = -MAX_VALUE;
        NNFloat k4                      = -MAX_VALUE;
        NNFloat k5                      = -MAX_VALUE;
        NNFloat k6                      = -MAX_VALUE;
        NNFloat k7                      = -MAX_VALUE;
        uint32_t v0                     = 0;
        uint32_t v1                     = 0;
        uint32_t v2                     = 0;
        uint32_t v3                     = 0;
        uint32_t v4                     = 0;
        uint32_t v5                     = 0;
        uint32_t v6                     = 0;
        uint32_t v7                     = 0;

        // Read first 128 elements into registers
        uint32_t wpos                   = tgx;
        if (wpos < width)
        {
            k0                              = pOutputKey[wpos];
            v0                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k1                              = pOutputKey[wpos];
            v1                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k2                              = pOutputKey[wpos];
            v2                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k3                              = pOutputKey[wpos];
            v3                              = pOutputValue[wpos];
        }
     
        // Run through remainder of data
        NNFloat minValue                    = -MAX_VALUE;
        uint32_t rpos                       = 128;
        uint32_t bufferSize                 = 0;
        NNFloat key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            // Read block of data
            unsigned wpos                   = rpos + tgx;
            NNFloat key                     = -MAX_VALUE;
            NNFloat value                   = 0.0f;
            if (wpos < width)
            {
                key                         = pOutputKey[wpos];
                value                       = pOutputValue[wpos];              
            }
            
            // Add values > minValue to shared memory buffer
            uint32_t count                  = __ballot(key > minValue);
            if (key > minValue)
            {
                uint32_t mask               = 0xffffffff >> (32 - tgx);
                uint32_t offset             = __popc(count & mask);
                offset                     += bufferSize;
                psKey[offset]               = key;
                psValue[offset]             = value;
            }
            bufferSize                     += __popc(count);

            // Check if buffer is full
            if (bufferSize >= 128)
            {
                // Sort 256 elements
                k4                          = psKey[tgx];
                v4                          = psValue[tgx];
                k5                          = psKey[tgx + cData._warpSize];
                v5                          = psValue[tgx + cData._warpSize];
                k6                          = psKey[tgx + 2 * cData._warpSize];
                v6                          = psValue[tgx + 2 * cData._warpSize];
                k7                          = psKey[tgx + 3 * cData._warpSize];
                v7                          = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                // Shift members in shared memory to beginning
                bufferSize                 -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]              = psKey[tgx + 128];
                    psValue[tgx]            = psValue[tgx + 128];
                }
            }

            // Advance to next block of data
            rpos                           += cData._warpSize;
        }

        // Do final sort if buffer has any remaining data
        if ((bufferSize > 0) || (width < 128))
        {
            // Store sentinel values in registers
            k4                              = -MAX_VALUE;
            k5                              = -MAX_VALUE;
            k6                              = -MAX_VALUE;
            k7                              = -MAX_VALUE;
            v4                              = 0;
            v5                              = 0;
            v6                              = 0;
            v7                              = 0;

            // Load last block of unsorted data into registers
            if (tgx < bufferSize)
            {
                k4                          = psKey[tgx];
                v4                          = psValue[tgx];
            }
            if (tgx < bufferSize - cData._warpSize)
            {
                k5                          = psKey[tgx + cData._warpSize];
                v5                          = psValue[tgx + cData._warpSize];
            }
            if (tgx < bufferSize - 2 * cData._warpSize)
            {
                k6                          = psKey[tgx + 2 * cData._warpSize];
                v6                          = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx < bufferSize - 3 * cData._warpSize)
            {
                k7                          = psKey[tgx + 3 * cData._warpSize];
                v7                          = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        // Copy results to index and value pointers
        NNFloat* pKey                       = pKeyBuffer + pos * k;
        uint32_t* pValue                    = pValueBuffer + pos * k;                
        wpos                                = tgx;
        if (wpos < k)
        {
            pKey[wpos]                      = k0;
            pValue[wpos]                    = v0;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k1;
            pValue[wpos]                    = v1;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k2;
            pValue[wpos]                    = v2;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k3;
            pValue[wpos]                    = v3;
        }
    }
}


void kCalculateTopK(NNFloat* pOutputKey, uint32_t* pOutputValue, NNFloat *pKey, uint32_t * pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t blocks                         = (batch + 3) / 4;
    kCalculateTopK_kernel<<<blocks, 128>>>(pOutputKey, pOutputValue, pKey, pValue, batch, width, k);
    LAUNCHERROR("kCalculateTopK_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeWeights_kernel(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        NNFloat r2                          = 0.0f;
        NNFloat* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        NNFloat* p                          = pWeight;
        
        // Calculate squared weight vector length
        while (p < pEnd)
        {
            NNFloat x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        // Normalize if necessary
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            p                               = pWeight;
            while (p < pEnd)
            {
                *p                         *= norm;
                p                          += outputStride;
            }             
        }
    }

}

void kNormalizeWeights(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight)
{
    uint32_t blocks                         = (outputStride + 127) / 128;
    kNormalizeWeights_kernel<<<blocks, 128>>>(norm, outputStride, inputStride, pWeight); 
    LAUNCHERROR("kNormalizeWeights_kernel");   
}


__global__ void
LAUNCH_BOUNDS()
kCalculateWeightMagnitudes_kernel(uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        NNFloat r2                          = 0.0f;
        NNFloat* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        NNFloat* p                          = pWeight;
        
        // Calculate squared weight vector length
        while (p < pEnd)
        {
            NNFloat x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        // Output to accumulator
        pMagnitude[pos]                     = r2;
    }

}

void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude)
{
    uint32_t blocks                         = (outputStride + 127) / 128;
    kCalculateWeightMagnitudes_kernel<<<blocks, 128>>>(outputStride, inputStride, pWeight, pMagnitude); 
    LAUNCHERROR("kCalculateWeightMagnitudes_kernel");   
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeWeightMagnitudes_kernel(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        NNFloat r2                          = pMagnitude[pos];
        NNFloat* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        NNFloat* p                          = pWeight;
        
        // Normalize if necessary
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            p                               = pWeight;
            while (p < pEnd)
            {
                *p                         *= norm;
                p                          += outputStride;
            }             
        }
    }

}

void kNormalizeWeightMagnitudes(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude)
{
    uint32_t blocks                         = (outputStride + 127) / 128;
    kNormalizeWeightMagnitudes_kernel<<<blocks, 128>>>(norm, outputStride, inputStride, pWeight, pMagnitude); 
    LAUNCHERROR("kNormalizeWeightMagnitudes_kernel");   
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDropout_kernel(NNFloat* pUnit, NNFloat* pRandom, NNFloat p, NNFloat scale)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    NNFloat r                               = pRandom[pos];
    if (r < p)
        pUnit[pos]                          = (NNFloat)0.0;
    NNFloat o                               = pUnit[pos];
    o                                       = (r < p) ? (NNFloat)0.0 : o;
    pUnit[pos]                              = scale * o;
}

void kCalculateDropout(NNFloat* pUnit, NNFloat* pRandom, uint32_t batch, uint32_t stride, NNFloat p)
{
    curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
    unsigned long blocks                = CalculateBlocks(batch * stride);
    NNFloat scale                       = (NNFloat)1.0 / ((NNFloat)1.0 - p);
    kCalculateDropout_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pRandom, p, scale);
    LAUNCHERROR("kCalculateDropout_kernel");
}

#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"
template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue)
{
    uint32_t itemStride                     = ((items + 511) >> 9) << 9;
    size_t tempBytes;
    cub::DoubleBuffer<KeyType> d_keys(pbKey->_pDevData, pbKey->_pDevData + itemStride);
    cub::DoubleBuffer<ValueType> d_values(pbValue->_pDevData, pbValue->_pDevData + itemStride);
    cub::DeviceRadixSort::SortPairs(NULL, tempBytes, d_keys, d_values, items);
    return tempBytes;
}

template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes)
{
    cub::DoubleBuffer<KeyType>  d_keys(pKey0, pKey1);
    cub::DoubleBuffer<ValueType> d_values(pValue0, pValue1);
    cub::DeviceRadixSort::SortPairs(pTemp, tempBytes, d_keys, d_values, items);
    return true;   
}
__global__ void
LAUNCH_BOUNDS()
kAddBuffers_kernel(NNFloat* pDst, NNFloat* pSrc, uint64_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos)                      += *(pSrc + pos);
}

void kAddBuffers(NNFloat* pDst, NNFloat* pSrc, uint64_t size)
{
    uint32_t blocks                         = CalculateBlocks(size);
    kAddBuffers_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pDst, pSrc, size);
    LAUNCHERROR("kAddBuffers_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kAddBuffers2D_kernel(NNFloat* pDst, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset                        = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos                       = blockIdx.x * dpitch + yOffset;
        uint64_t spos                       = blockIdx.x * spitch + yOffset;
        pDst[dpos]                         += pSrc[spos];
    }
}

void kAddBuffers2D(NNFloat* pDst, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height)
{
    dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);       
    kAddBuffers2D_kernel<<<grid, getGpu()._threadsPerBlock>>>(pDst, dpitch, pSrc, spitch, width);
    LAUNCHERROR("kAddBuffers2D_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCopy2D_kernel(NNFloat* pDst, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset                        = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos                       = blockIdx.x * dpitch + yOffset;
        uint64_t spos                       = blockIdx.x * spitch + yOffset;
        pDst[dpos]                          = pSrc[spos];
    }
}

void kCopy2D(NNFloat* pDst, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height)
{
    dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);       
    kCopy2D_kernel<<<grid, getGpu()._threadsPerBlock>>>(pDst, dpitch, pSrc, spitch, width);
    LAUNCHERROR("kCopy2D_kernel");
}


