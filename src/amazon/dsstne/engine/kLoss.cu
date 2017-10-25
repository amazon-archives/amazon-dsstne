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

void SetKLossGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

void GetKLossGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyFromSymbol: SetKernelsGpuData copy From cData failed");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL1Error_kernel(NNFloat* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        error                   = fabsf(a);     
    }
    
    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - (NNFloat)1.0) - fabsf(a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - (NNFloat)1.0);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}



NNFloat kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);    
        kCalculateSparseOnlyNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseOnlyNonZeroL1Error_kernel");    
    }
    else
    {
        uint64_t size               = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks             = CalculateBlocks(size);    
        kCalculateSparseRawL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawL1Error_kernel");
        blocks                      = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseNonZeroL1Error_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t) - fabsf(a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t) - fabsf(a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += fabsf(a - t) - fabsf(a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<typename T>
NNFloat kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks         = CalculateBlocks(batch * getGpu()._warpSize);    
        kCalculateSparseAnalogOnlyNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogOnlyNonZeroL1Error_kernel");   
    }
    else
    {
        uint64_t size           = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks         = CalculateBlocks(size);    
        kCalculateSparseRawL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawL1Error_kernel");
        blocks                  = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogNonZeroL1Error_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL2Error_kernel(NNFloat* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        error                   = (NNFloat)0.5 * a * a;     
    }
    
    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - (NNFloat)1.0) * (a - (NNFloat)1.0));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - (NNFloat)1.0) * (a - (NNFloat)1.0) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}


NNFloat kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks         = CalculateBlocks(batch * getGpu()._warpSize);    
        kCalculateSparseOnlyNonZeroL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseOnlyNonZeroL2Error_kernel");    
    }
    else
    {
        uint64_t size           = batch * stride;
        uint32_t blocks         = CalculateBlocks(size);    
        kCalculateSparseRawL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawL2Error_kernel");
        blocks                  = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseNonZeroL2Error_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += (NNFloat)0.5 * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}


template<typename T>
NNFloat kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks         = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogOnlyNonZeroL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogOnlyNonZeroL2Error_kernel");    
    }
    else
    {
        uint64_t size           = batch * stride;
        uint32_t blocks         = CalculateBlocks(size);    
        kCalculateSparseRawL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawL2Error_kernel");
        blocks                  = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogNonZeroL2Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogNonZeroL2Error_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawCrossEntropyError_kernel(NNFloat* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        error                   = -log(max(MIN_ERROR, (NNFloat)1.0 - a));     
    }

    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += -log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += -log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (NNFloat)1.0 - a));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

NNFloat kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks         = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseOnlyNonZeroCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseOnlyNonZeroCrossEntropyError_kernel");    
    }
    else
    {    
        uint64_t size           = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks         = CalculateBlocks(size);
        kCalculateSparseRawCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawCrossEntropyError_kernel");
        blocks                  = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseNonZeroCrossEntropyError_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    //printf("Error is %f\n",  (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);

    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat t               = (NNFloat)1.0 / (NNFloat)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            error              += -t * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

NNFloat kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseMultinomialCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
    LAUNCHERROR("kCalculateSparseMultinomialCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download(); 
    //printf("Error is %f\n",  (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);

    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}



template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += -t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += -t * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            error              += -t * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}


template<typename T>
NNFloat kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseAnalogMultinomialCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
    LAUNCHERROR("kCalculateSparseAnalogMultinomialCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download(); 
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawScaledMarginalCrossEntropyError_kernel(NNFloat* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
        NNFloat a               = pUnit[pos];
        if (a > cData._SMCE_zeroTarget)
            error               = -cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));     
    }
    
    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            if (a > cData._SMCE_zeroTarget)
                error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));   
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

NNFloat kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks         = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel");   
    }
    else
    {
        uint64_t size           = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks         = CalculateBlocks(size);
        kCalculateSparseRawScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawScaledMarginalCrossEntropyError_kernel");
        blocks                  = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
        LAUNCHERROR("kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel");
    }    
    getGpu()._pbAccumulator->Download(); 
    //printf("Error is %f\n",  (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel(NNFloat* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < size)
    {
          NNFloat a               = pUnit[pos];
          if (a > cData._SMCE_zeroTarget)
          {
              error               = -cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));
          }
    }

    REDUCEERROR(error)
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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

              if (a > cData._SMCE_zeroTarget)
              {
                  error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));
              }

              if (a < cData._SMCE_oneTarget)
              {
                  error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
              }
              pos1               += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
NNFloat kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    if (!bSparseIgnoreZero)
    {
        uint64_t size               = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks             = CalculateBlocks(size);
        kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, size);
        LAUNCHERROR("kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel");
    }
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
    LAUNCHERROR("kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat t               = (NNFloat)1.0f / (NNFloat)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

NNFloat kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex);
    LAUNCHERROR("kCalculateSparseMultinomialScaledMarginalCrossEntropyError_kernel");    
    getGpu()._pbAccumulator->Download(); 
    //printf("Error is %f\n",  (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}




template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, unsigned char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            NNFloat t           = pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    NNFloat error               = (NNFloat)0.0;
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
            NNFloat t           = pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            if (a < cData._SMCE_oneTarget)
                error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

    REDUCEERROR(error)
}

template<typename T>
NNFloat kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    uint32_t blocks             = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
    LAUNCHERROR("kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel");    
    getGpu()._pbAccumulator->Download(); 
    //printf("Error is %f\n",  (double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = fabsf(a - t);        
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        error                   = fabsf(a - t);        
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        error                   = fabsf(a - t);        
    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateL1Error_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateL1Error_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateL2Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = (NNFloat)0.5 * (a - t) * (a - t);         
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL2Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        error                   = (NNFloat)0.5 * (a - t) * (a - t);         

    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL2Error_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        error                   = (NNFloat)0.5 * (a - t) * (a - t);         

    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateL2Error_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateL2Error_kernel");    
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE); 
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = pData[0];
    pData                      += pos;

    // Calculate loss
    NNFloat loss                = (NNFloat)0.0;
    while (pos < stride)
    {
        NNFloat negativeDP      = *pUnit;   
        loss                   += max((NNFloat)0.0, margin - positiveDP + negativeDP);
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }
    
    REDUCEERROR(loss)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = (NNFloat)pData[0] * (NNFloat)(1.0 / 256.0);
    pData                      += pos;

    // Calculate loss
    NNFloat loss                = (NNFloat)0.0;
    while (pos < stride)
    {
        NNFloat negativeDP      = pUnit[pos];   
        loss                   += max((NNFloat)0.0, margin - positiveDP + negativeDP);
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }
    
    REDUCEERROR(loss)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    // Increment pointers and fetch margin and positive example
    uint32_t pos                = threadIdx.x + 1;
    pUnit                      += blockIdx.x * stride;
    NNFloat positiveDP          = pUnit[0];
    pUnit                      += pos;
    pData                      += blockIdx.x * stride;
    NNFloat margin              = (NNFloat)pData[0] * (NNFloat)(1.0 / 128.0);
    pData                      += pos;

    // Calculate loss
    NNFloat loss                = (NNFloat)0.0;
    while (pos < stride)
    {
        NNFloat negativeDP      = pUnit[pos];   
        loss                   += max((NNFloat)0.0, margin - positiveDP + negativeDP);
        pos                    += blockDim.x;
        pUnit                  += blockDim.x;
        pData                  += blockDim.x;      
    }
    
    REDUCEERROR(loss)
}

template<typename T> NNFloat kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    unsigned long threads = max(32, min(stride, 128));
    kCalculateHingeError_kernel<<<batch, threads>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateHingeError_kernel");    
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE); 
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = -t * log(max(MIN_ERROR, a)) - ( (NNFloat)1.0 - t) * log(max(MIN_ERROR, (NNFloat)1.0 - a));     
        //printf("%d %llu %f %f %f\n", position, pos, a, t, error);
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        error                   = -t * log(max(MIN_ERROR, a)) - ( (NNFloat)1.0 - t) * log(max(MIN_ERROR, (NNFloat)1.0 - a));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        error                   = -t * log(max(MIN_ERROR, a)) - ( (NNFloat)1.0 - t) * log(max(MIN_ERROR, (NNFloat)1.0 - a));     
        //printf("%d %llu %f %f %f\n", position, pos, a, t, error);
    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = -t * log(max(MIN_ERROR, a));  
        //printf("%d %llu %f %f %f\n", position, pos, a, t, error);
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        error                   = -t * log(max(MIN_ERROR, a));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        error                   = -t * log(max(MIN_ERROR, a));     
        //printf("%d %llu %f %f %f\n", position, pos, a, t, error);
    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateMultinomialCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateMultinomialCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) || 
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ( (NNFloat)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        if (((t == (NNFloat)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((NNFloat)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        if (((t == (NNFloat)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (NNFloat)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((NNFloat)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (NNFloat)1.0 - a));  
        //printf("%d %llu %f %f %f\n", position, pos, a, t, error);
    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}



template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        if ((t != (NNFloat)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    NNFloat error               = (NNFloat)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        if ((t != (NNFloat)0.0) && (a < cData._SMCE_oneTarget))
            error               = -t * cData._SMCE_oneScale * log(max(MIN_ERROR, a));  
    }

    REDUCEERROR(error)
}

template<typename T> NNFloat kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateMultinomialScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kCalculateMultinomialScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (NNFloat)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


// Instantiates allowable templated functions so we can hide the implementations here
// instead of in the header file because we're mixing CUDA and C++ and that's
// a migraine headache in the making otherwise.
void kLossTempFunction()
{
    kCalculateL1Error<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<double>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<char>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<uint64_t>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateL1Error<int64_t>(0, 0, 0, NULL, NULL);

    kCalculateL2Error<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<double>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<char>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<uint64_t>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateL2Error<int64_t>(0, 0, 0, NULL, NULL);
    
    kCalculateHingeError<NNFloat>(0, 0, 0, NULL, NULL);    
    kCalculateHingeError<double>(0, 0, 0, NULL, NULL);    
    kCalculateHingeError<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateHingeError<char>(0, 0, 0, NULL, NULL);      
    kCalculateHingeError<uint32_t>(0, 0, 0, NULL, NULL); 
    kCalculateHingeError<uint64_t>(0, 0, 0, NULL, NULL); 
    kCalculateHingeError<int32_t>(0, 0, 0, NULL, NULL); 
    kCalculateHingeError<int64_t>(0, 0, 0, NULL, NULL); 
    
    kCalculateCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<double>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<char>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL);

    kCalculateScaledMarginalCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<double>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<char>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL);    
    kCalculateScaledMarginalCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateScaledMarginalCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL);  

    kCalculateMultinomialCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<double>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<char>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL);

    kCalculateMultinomialScaledMarginalCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<double>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<char>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL);    
    kCalculateMultinomialScaledMarginalCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL);
    kCalculateMultinomialScaledMarginalCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL);  

    kCalculateSparseAnalogL1Error<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL1Error<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);

    kCalculateSparseAnalogL2Error<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseAnalogL2Error<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);

    kCalculateSparseAnalogMultinomialCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);

    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);    
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);
    kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL);  

    kCalculateSparseDataScaledMarginalCrossEntropyError<NNFloat>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<double>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<unsigned char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<char>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<uint32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<uint64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<int32_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<int64_t>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
    kCalculateSparseDataScaledMarginalCrossEntropyError<long>(0, 0, 0, NULL, NULL, NULL, NULL, NULL, false);
}


