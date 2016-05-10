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
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

void GetKDeltaGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy From cData failed");
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

template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
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
            
       case SoftMax:
            kCalculateSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSoftMaxOutputDelta_kernel");
            break;                
    }
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

void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case Sigmoid:
            kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            kCalculateSparseNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            kCalculateSparseNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            kCalculateSparseNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateSparseRawReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawReluOutputDelta_kernel");
            kCalculateSparseNonZeroReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroReluOutputDelta_kernel");
            break;
            
        case SoftMax:
            kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
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
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case Sigmoid:
            kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateSparseRawReluOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawReluOutputDelta_kernel");
            kCalculateSparseAnalogNonZeroReluOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroReluOutputDelta_kernel");
            break;
            
        case SoftMax:
            kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
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

void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case SoftMax:
            kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            kCalculateSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxOutputDelta_kernel");
            break;    

        case Sigmoid:
            kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
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
void kCalculateSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case SoftMax:
        case Sigmoid:
            kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
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




void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case Sigmoid:
            kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroScaleMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
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

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
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
            LAUNCHERROR("kCalculateL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateReluL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateReluL1OutputDelta_kernel");
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

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));
    
    switch (activation)
    {
        case Sigmoid:
            kCalculateSparseRawSigmoidL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateRawSigmoidL1OutputDelta_kernel");
            kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateNonZeroSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateSparseRawTanhL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateRawTanhL1OutputDelta_kernel");
            kCalculateSparseNonZeroTanhL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateNonZeroTanhL1OutputDelta_kernel");
            break;

        case Linear:
            kCalculateSparseRawLinearL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateRawLinearL1OutputDelta_kernel");
            kCalculateSparseNonZeroLinearL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateNonZeroLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateSparseRawReluL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateRawReluL1OutputDelta_kernel");
            kCalculateSparseNonZeroReluL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex);
            LAUNCHERROR("kCalculateNonZeroL1OutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparsenessPenalty_kernel(uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
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
        NNFloat penalty         = cData._sparsenessPenalty_beta * (-cData._sparsenessPenalty_p / pi + ((NNFloat)1.0 - cData._sparsenessPenalty_p) / ((NNFloat)1.0 - pi));
        
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
void kCalculateSparsenessPenalty(uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    dim3 grid1(CalculateBlocks(stride));
    kCalculateSparsenessPenalty_kernel<<<grid1, getGpu()._threadsPerBlock>>>(batch, stride, pUnit, pDelta);
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

void kCalculateHadamardProduct(Activation activation, uint64_t size, NNFloat scale, NNFloat* pUnit, NNFloat* pDelta)
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

    kCalculateL1OutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateL1OutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateScaledMarginalCrossEntropyOutputDelta<NNFloat>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<double>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<unsigned char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<char>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<uint64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int32_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<int64_t>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyOutputDelta<long>(Sigmoid, 0, 0, 0, NULL, NULL, NULL);

    kCalculateSparseAnalogOutputDelta<NNFloat>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<double>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<unsigned char>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<char>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<uint32_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<uint64_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<int32_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<int64_t>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogOutputDelta<long>(Linear, 0, 0, 0, NULL,  NULL, NULL, NULL, NULL, NULL);
}



