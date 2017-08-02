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

void SetKDeltaGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKDeltaGpuData copy to cData failed");
}

void GetKDeltaGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyFromSymbol: GetKDeltaGpuData copy From cData failed");
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * a * ((NNFloat)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * a * ((NNFloat)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * a * ((NNFloat)1.0 - a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((NNFloat)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((NNFloat)1.0 - a * a);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((NNFloat)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * (a > (NNFloat)0.0);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * (a > (NNFloat)0.0);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * (a > (NNFloat)0.0);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = a - t;      
    }
}

template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSigmoidOutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateTanhOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateTanhOutputDelta_kernel");
            break;

        case Linear:
            kCalculateLinearOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateReluOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateReluOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            kCalculateLeakyReluOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, slope);
            LAUNCHERROR("kCalculateLeakyReluOutputDelta_kernel");
            break;

       case SoftMax:
            kCalculateSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSoftMaxOutputDelta_kernel");
            break;                
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    __shared__ NNFloat sDelta0;
    sDelta0                     = (NNFloat)0;
    __syncthreads();

    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    pDelta                     += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    NNFloat* pDelta0            = pDelta;
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = pData[0];
    pData                      += pos;

    // Calculate loss
    while (pos < stride)
    {
        NNFloat negativeDP      = *pUnit;   
        NNFloat loss            = max((NNFloat)0.0, margin - positiveDP + negativeDP);
        NNFloat delta           = (NNFloat)0.0;
        if (loss > (NNFloat)0.0)
        {
            delta               = (NNFloat)1.0;
            atomicAdd(&sDelta0, (NNFloat)1.0);
        }
        *pDelta                 = delta;
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }

    // Output delta0
    __syncthreads();
    if (threadIdx.x == 0)
        *pDelta0                = sDelta0; 
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    __shared__ NNFloat sDelta0;
    sDelta0                     = (NNFloat)0;
    __syncthreads();

    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    pDelta                     += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    NNFloat* pDelta0            = pDelta;
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = (NNFloat)pData[0] * (NNFloat)(1.0 / 256.0);
    pData                      += pos;

    // Calculate loss
    while (pos < stride)
    {
        NNFloat negativeDP      = *pUnit;   
        NNFloat loss            = max((NNFloat)0.0, margin - positiveDP + negativeDP);
        NNFloat delta           = (NNFloat)0.0;
        if (loss > (NNFloat)0.0)
        {
            delta               = (NNFloat)1.0;
            atomicAdd(&sDelta0, (NNFloat)1.0);
        }
        *pDelta                 = delta;
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }

    // Output delta0
    __syncthreads();
    if (threadIdx.x == 0)
        *pDelta0                = sDelta0; 
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    __shared__ NNFloat sDelta0;
    sDelta0                     = (NNFloat)0;
    __syncthreads();

    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    pDelta                     += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    NNFloat* pDelta0            = pDelta;
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = (NNFloat)pData[0] * (NNFloat)(1.0 / 128.0);
    pData                      += pos;

    // Calculate loss
    while (pos < stride)
    {
        NNFloat negativeDP      = *pUnit;   
        NNFloat loss            = max((NNFloat)0.0, margin - positiveDP + negativeDP);
        NNFloat delta           = (NNFloat)0.0;
        if (loss > (NNFloat)0.0)
        {
            delta               = (NNFloat)1.0;
            atomicAdd(&sDelta0, (NNFloat)1.0);
        }
        *pDelta                 = delta;
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }

    // Output delta0
    __syncthreads();
    if (threadIdx.x == 0)
        *pDelta0                = sDelta0; 
}


template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    unsigned long threads = max(32, min(stride, getGpu()._threadsPerBlock));
    kCalculateHingeOutputDelta_kernel<<<batch, threads>>>(position, batch, stride,  pUnit, pDelta, pData);
    LAUNCHERROR("kCalculateHingeOutputDelta_kernel");    
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = cData._deltaBoost_zero * a * a * ((NNFloat)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = cData._deltaBoost_one * (a - (NNFloat)1.0) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = a * ((NNFloat)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = (a - (NNFloat)1.0) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = a - (NNFloat)1.0;   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawReluOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];         
        pDelta[pos]             = a * (a > (NNFloat)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLeakyReluOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = a * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = (a - (NNFloat)1.0) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat t               = (NNFloat)1.0 / (end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = a - t;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero, NNFloat slope)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    }
    
    switch (activation)
    {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateSparseNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateSparseNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawReluOutputDelta_kernel");
            }
            kCalculateSparseNonZeroReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroReluOutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLeakyReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLeakyReluOutputDelta_kernel");
            }
            kCalculateSparseNonZeroLeakyReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, slope);
            LAUNCHERROR("kCalculateSparseNonZeroLeakyReluOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxOutputDelta_kernel");
            break;                        
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = cData._deltaBoost_one * (a - t) * a * (t - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = cData._deltaBoost_one * (a - t) * a * (t - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = cData._deltaBoost_one * (a - t) * a * (t - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = a - t;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = a - t;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = a - t;   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLeakyReluOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = (a - t) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
            pos1               += cData._warpSize;
        }
    }
}




template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = a - t; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = a - t; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = a - t; 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    }    
    
    switch (activation)
    {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawReluOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroReluOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLeakyReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLeakyReluOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroLeakyReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData, slope);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroLeakyReluOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel");
            break;         
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
        case SoftMax:
            kCalculateSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = cData._deltaBoost_zero * a;      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = cData._deltaBoost_one * (a - (NNFloat)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    }

    switch (activation)
    {
        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxOutputDelta_kernel");
            break;    

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonzeroSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = cData._deltaBoost_one * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
 
    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    } 
    
    switch (activation)
    {
        case SoftMax:
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonzeroSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}



template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        NNFloat output          = (NNFloat)0.0;
        if ((t == (NNFloat)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        NNFloat output          = (NNFloat)0.0;
        if ((t == (NNFloat)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        NNFloat output          = (NNFloat)0.0;
        if ((t > (NNFloat)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = output;
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        NNFloat output          = (NNFloat)0.0;
        if ((t > (NNFloat)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        NNFloat output          = (NNFloat)0.0;
        if ((t > (NNFloat)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        NNFloat output          = (NNFloat)0.0;
        if ((t > (NNFloat)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = output;
    }
}

template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            break;

        case SoftMax:
            kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            break;                    
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        NNFloat output          = (NNFloat)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = cData._SMCE_zeroScale * a;
        pDelta[pos]             = output;   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = cData._SMCE_oneScale * (a - (NNFloat)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        NNFloat output          = (NNFloat)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = cData._SMCE_zeroScale * a;
        pDelta[pos]             = output;   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat t               = (NNFloat)1.0 / (NNFloat)(end - pos1);
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = cData._SMCE_oneScale * (a - t);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}




void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    }
    
    switch (activation)
    {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroScaleMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            break;            
            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
          NNFloat a               = pUnit[pos];
          NNFloat output          = (NNFloat)0.0;
          if (a > cData._SMCE_zeroTarget)
          {
              output              = cData._SMCE_zeroScale * a;
          }
          pDelta[pos]             = output;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              NNFloat a           = pUnit[pos2];
              T t                 = pSparseData[pos1];
              NNFloat output      = (NNFloat)0.0;
              if (a < cData._SMCE_oneTarget)
              {
                  output          = cData._SMCE_oneScale * t * (a - (NNFloat)1.0);
              }
              pDelta[pos2]        = output;
              pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    switch (activation)
    {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            cout << "unsupported activation for this cost function" << endl;
            getGpu().Shutdown();
            exit(-1);
            break;
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * a * ((NNFloat)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((NNFloat)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0;      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * (a > (NNFloat)0.0);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * a * ((NNFloat)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((NNFloat)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * (a > (NNFloat)0.0);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * a * ((NNFloat)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((NNFloat)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * (a > (NNFloat)0.0);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = ((a - t) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));
    }
}

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateTanhL1OutputDelta_kernel");
            break;

        case Linear:
            kCalculateLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateReluL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateReluL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateLeakyReluL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, slope);
            LAUNCHERROR("kCalculateLeakyReluL1OutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL1OutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = ((a > (NNFloat)0.0) ? (NNFloat)1.0 : (NNFloat)-1.0) * a * ((NNFloat)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = ((a - (NNFloat)1.0) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * a * ((NNFloat)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL1OutputDelta_kernel(uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = (a > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((NNFloat)1.0 - a * a);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = ((a - (NNFloat)1.0) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((NNFloat)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL1OutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];  
        pDelta[pos]             = (a > (NNFloat)0.0) ? (NNFloat)1.0 : (NNFloat)-1.0;           
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = (a - (NNFloat)1.0) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0;    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawReluL1OutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = (a > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * (a > (NNFloat)0.0);           
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = ((a - (NNFloat)1.0) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * (a > (NNFloat)0.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLeakyReluL1OutputDelta_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = (a > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0)); 
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRawLeakyReluL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = ((a - (NNFloat)1.0) > (NNFloat)0.0 ? (NNFloat)1.0 : (NNFloat)-1.0) * ((a > (NNFloat)0.0) + slope * (a <= (NNFloat)0.0));   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero, NNFloat slope)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    // Clear entire delta if ignoring zero outputs
    if (bSparseIgnoreZero)
    {
        cudaMemset(pDelta, 0, size * sizeof(NNFloat));
    }
    
    switch (activation)
    {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroTanhL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroTanhL1OutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroLinearL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawReluL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawReluL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroReluL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroReluL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLeakyReluL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLeakyReluL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroRawLeakyReluL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, slope);
            LAUNCHERROR("kCalculateSparseNonZeroRawLeakyReluL1OutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparsenessPenalty_kernel(uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat p, NNFloat beta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate sum of activations
    if (pos < stride)
    {
        NNFloat pi              = (NNFloat)0.0;
        for (int i = 0; i < batch; i++)
        {
            pi                 += pUnit[pos];
            pos                += stride;
        }

        // Calculate sparseness penalty
        pi                     /= (NNFloat)batch;
        pi                      = max(MIN_ACTIVATION, min(MAX_ACTIVATION, pi));
        NNFloat penalty         = beta * (-p / pi + ((NNFloat)1.0 - p) / ((NNFloat)1.0 - pi));
        
        // Apply sparseness penalty to deltas
        pos                     = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = 0; i < batch; i++)
        {
            pDelta[pos]        += penalty;
            pos                += stride;
        }
    }
}



// Calculates and applies sparseness penalty to hidden layers
void kCalculateSparsenessPenalty(uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat p, NNFloat beta)
{
    dim3 grid1(CalculateBlocks(stride));
    kCalculateSparsenessPenalty_kernel<<<grid1, getGpu()._threadsPerBlock>>>(batch, stride, pUnit, pDelta, p, beta);
    LAUNCHERROR("kCalculateSparsenessPenalty_kernel");
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat scale, NNFloat oneOverScale)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        NNFloat d               = pDelta[pos];
        x                      *= oneOverScale;
        pDelta[pos]             = scale * x * ((NNFloat)1.0 - x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateTanhHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat scale, NNFloat oneOverScale)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        NNFloat d               = pDelta[pos];
        x                      *= oneOverScale;
        pDelta[pos]             = scale * ((NNFloat)1.0 - x * x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateReluHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        if (x <= (NNFloat)0.0)
            pDelta[pos]         = (NNFloat)0.0;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateLeakyReluHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        if (x <= (NNFloat)0.0)
        {
            pDelta[pos]         *= slope;
        }
    }
}

void kCalculateHadamardProduct(Activation activation, uint64_t size, NNFloat scale, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
{
    uint32_t blocks             = CalculateBlocks(size);
    NNFloat oneOverScale        = (NNFloat)1.0 / scale;        

    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, scale, oneOverScale);
            LAUNCHERROR("kCalculateSigmoidHadamardProduct_kernel");
            break;   

        case Tanh:
            kCalculateTanhHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, scale, oneOverScale);
            LAUNCHERROR("kCalculateTanhHadamardProduct_kernel");
            break;    

        case Linear:
            // Derivative of linear output is 1, no need to call any kernel here
            break;

        case RectifiedLinear:
            kCalculateReluHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateReluHadamardProduct_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateLeakyReluHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, slope);
            LAUNCHERROR("kCalculateLeakyReluHadamardProduct_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltas_kernel(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta)
{
    uint32_t dpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    pDelta                                 += dpos * stride;
    if (dpos < batch)
    { 
        // Calculate vector length
        uint32_t pos                        = tgx;
        NNFloat r2                          = (NNFloat)0.0;
        while (pos < stride)
        {
            NNFloat x                       = pDelta[pos];
            r2                             += x * x;
            pos                            += cData._warpSize;
        }
        
        // Reduce sum
        r2                                 += __shfl(r2, tgx ^ 1);
        r2                                 += __shfl(r2, tgx ^ 2);
        r2                                 += __shfl(r2, tgx ^ 4);
        r2                                 += __shfl(r2, tgx ^ 8);
        r2                                 += __shfl(r2, tgx ^ 16);
    
        // Normalalize vector if too large
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            pos                             = tgx;
            while (pos < stride)
            {
                pDelta[pos]                *= norm;
                pos                        += cData._warpSize;
            }
        }    
        
   }  
}

void kNormalizeDeltas(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta)
{
    uint32_t blocks                         = (batch + 3) / 4;
    kNormalizeDeltas_kernel<<<blocks, 128>>>(norm, batch, stride, pDelta);
    LAUNCHERROR("kNormalizeDeltas_kernel");
}


__global__ void
LAUNCH_BOUNDS()
kCalculateDeltaMagnitudes_kernel(uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude)
{
    uint32_t dpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    pDelta                                 += dpos * stride;
    if (dpos < batch)
    { 
        // Calculate vector length
        uint32_t pos                        = tgx;
        NNFloat r2                          = (NNFloat)0.0;
        while (pos < stride)
        {
            NNFloat x                       = pDelta[pos];
            r2                             += x * x;
            pos                            += cData._warpSize;
        }
        
        // Reduce sum
        r2                                 += __shfl(r2, tgx ^ 1);
        r2                                 += __shfl(r2, tgx ^ 2);
        r2                                 += __shfl(r2, tgx ^ 4);
        r2                                 += __shfl(r2, tgx ^ 8);
        r2                                 += __shfl(r2, tgx ^ 16);
    
        // Output result
        if (tgx == 0)
            pMagnitude[dpos]                = r2;
   }  
}

void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude)
{
    uint32_t blocks                         = (batch + 3) / 4;
    kCalculateDeltaMagnitudes_kernel<<<blocks, 128>>>(batch, stride, pDelta, pMagnitude);
    LAUNCHERROR("kCalculateDeltaMagnitudes_kernel");        
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltaMagnitudes_kernel(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude)
{
    uint32_t dpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    pDelta                                 += dpos * stride;
    if (dpos < batch)
    {    
        // Normalalize vector if too large
        NNFloat r2                          = pMagnitude[dpos];
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            uint32_t pos                    = tgx;
            while (pos < stride)
            {
                pDelta[pos]                *= norm;
                pos                        += cData._warpSize;
            }
        }    
        
   }  
}

void kNormalizeDeltaMagnitudes(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude)
{
    uint32_t blocks                         = (batch + 3) / 4;
    kNormalizeDeltaMagnitudes_kernel<<<blocks, 128>>>(norm, batch, stride, pDelta, pMagnitude);
    LAUNCHERROR("kNormalizeDeltaMagnitudes_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateMaxoutDelta_kernel(NNFloat* pSrc, NNFloat* pSrcDelta, size_t size, NNFloat beta, NNFloat* pDst, NNFloat* pDstDelta)
{
    uint64_t pos                        = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat s = pSrc[pos];
        NNFloat sdelta = pSrcDelta[pos];
        NNFloat d = pDst[pos];
        NNFloat delta                   = (s == d) ? sdelta : (NNFloat)0;
        
        if (beta == (NNFloat)0)
            pDstDelta[pos]              = delta;
        else if (delta != (NNFloat)0.0)
            pDstDelta[pos]              = beta * pDstDelta[pos] + delta;
    }
}


void kCalculateMaxoutDelta(NNFloat* pSrc, NNFloat* pSrcDelta, size_t size, NNFloat beta, NNFloat* pDst, NNFloat* pDstDelta)
{
    unsigned long blocks                    = CalculateBlocks(size);
    kCalculateMaxoutDelta_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pSrc, pSrcDelta, size, beta, pDst, pDstDelta);
    LAUNCHERROR("kCalculateMaxoutDelta_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateCosineDelta_kernel(NNFloat* pDPDelta, NNFloat* pDP, NNFloat* pA, NNFloat* pB, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride)
{
    // Preincrement pointers
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride;
    pDP                    += blockIdx.x * stride;
    pA                     += blockIdx.x * stride;
    pB                     += blockIdx.x * stride;    
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    NNFloat dp              = *pDP;
    NNFloat dpDelta         = *pDPDelta;
    NNFloat a               = *pA;
    NNFloat b               = *pB;
    NNFloat ab              = a * b;
    NNFloat a2              = a * a;
    NNFloat b2              = b * b;
    
    // Calculate deltas
    while (pos < inputStride)
    {
        NNFloat ai          = *p0Vector;
        NNFloat bi          = *pVector;

        NNFloat delta0      = dpDelta * ((bi / ab) - (ai * dp / a2));
        NNFloat delta       = dpDelta * ((ai / ab) - (bi * dp / b2));
        if (beta0 == (NNFloat)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (NNFloat)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateCosineDelta(NNFloat* pDPDeltaIn, NNFloat* pDPIn, NNFloat* pA, NNFloat* pB, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride)
{
    unsigned long blocks = batch;
    unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
    kCalculateCosineDelta_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pDPDeltaIn, pDPIn, pA, pB, p0Vector, pVector, batch, stride, pDelta0, beta0, pDelta, beta, inputStride);

    LAUNCHERROR("kCalculateCosineDelta_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDotProductDelta_kernel(NNFloat* pDPDelta, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride)
{
    // Preincrement pointers
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride; 
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    NNFloat dpDelta         = *pDPDelta;
    
    // Calculate deltas
    while (pos < inputStride)
    {
        NNFloat ai          = *p0Vector;
        NNFloat bi          = *pVector;
        NNFloat delta0      = dpDelta * bi;
        NNFloat delta       = dpDelta * ai;
        if (beta0 == (NNFloat)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (NNFloat)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateDotProductDelta(NNFloat* pDPDelta, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride)
{
    unsigned long blocks = batch;
    unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
    kCalculateDotProductDelta_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pDPDelta, p0Vector, pVector, batch, stride, pDelta0, beta0, pDelta, beta, inputStride);   
    LAUNCHERROR("kCalculateDotProductDelta_kernel");
}   

// Instantiates allowable templated functions so we can hide the implementations here
// instead of in the header file because we're mixing CUDA and C++ and that's
// a migraine headache in the making otherwise.
void KDeltaTempFunction()
{   
    kCalculateCrossEntropyOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateCrossEntropyOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateScaledMarginalCrossEntropyOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateL1OutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateL1OutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);

    kCalculateOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    kCalculateOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL, 0);
    
    kCalculateHingeOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    
    kCalculateHingeOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);    

    kCalculateScaledMarginalCrossEntropyOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<long>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateSparseAnalogOutputDelta<NNFloat>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<double>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<unsigned char>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<char>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<uint32_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<uint64_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<int32_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<int64_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);
    kCalculateSparseAnalogOutputDelta<long>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false, 0);

    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<double>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<char>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<long>(Sigmoid, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL, false);
}



