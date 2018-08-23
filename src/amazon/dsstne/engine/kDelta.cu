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
kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
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
kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
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
kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
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
kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
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

template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
            kCalculateRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateRELUOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            kCalculateLRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, slope);
            LAUNCHERROR("kCalculateLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            kCalculateELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, alpha);
            LAUNCHERROR("kCalculateELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            kCalculateSELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, alpha, lambda);
            LAUNCHERROR("kCalculateSELUOutputDelta_kernel");
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
kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
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
kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
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
kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
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
kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
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
kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
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
kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
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
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
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
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
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
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
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
kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
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
kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
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
kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
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
kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) + (a < (NNFloat)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t) * ((a >= (NNFloat)0.0) * lambda + (a < (NNFloat)0.0) * (lambda * alpha * exp(a)));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
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
kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
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
kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
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

template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateIndexedSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSigmoidOutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateIndexedTanhOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedTanhOutputDelta_kernel");
            break;

        case Linear:
            kCalculateIndexedLinearOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedRELUOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, slope);
            LAUNCHERROR("kCalculateIndexedLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            kCalculateIndexedELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, alpha);
            LAUNCHERROR("kCalculateIndexedELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSELUOutputDelta_kernel");
            break;

       case SoftMax:
            kCalculateIndexedSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSoftMaxOutputDelta_kernel");
            break;                
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = pData[pos];
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = (NNFloat)pData[pos] * (NNFloat)(1.0 / 256.0);
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = (NNFloat)pData[pos] * (NNFloat)(1.0 / 128.0);
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}


template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    unsigned long threads = max(32, min(stride, getGpu()._threadsPerBlock));
    kCalculateHingeOutputDelta_kernel<<<batch, threads>>>(position, batch, stride,  pUnit, pDelta, pData);
    LAUNCHERROR("kCalculateHingeOutputDelta_kernel");    
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = pData[pos];
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = (NNFloat)pData[pos] * (NNFloat)(1.0 / 256.0);
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dOffset            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        NNFloat a               = pUnit[pos];
        NNFloat t               = (NNFloat)pData[pos] * (NNFloat)(1.0 / 128.0);
        pDelta[pos]             = (a < (NNFloat)0.0) ? -t : (NNFloat)0.0;
        pos                    += blockDim.x;
    }
}


template<typename T> void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    unsigned long threads = max(32, min(stride, getGpu()._threadsPerBlock));
    kCalculateIndexedHingeOutputDelta_kernel<<<batch, threads>>>(position, batch, stride,  pUnit, pDelta, pIndex, pData);
    LAUNCHERROR("kCalculateHingeOutputDelta_kernel");    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * a * a * ((NNFloat)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = w * a * ((NNFloat)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];         
        pDelta[pos]             = w * a * (a > (NNFloat)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = (NNFloat)1.0;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0 / (NNFloat)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateSparseNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateSparseNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUOutputDelta_kernel");
            }
            kCalculateSparseNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroRELUOutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUOutputDelta_kernel");
            }
            kCalculateSparseNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, slope);
            LAUNCHERROR("kCalculateSparseNonZeroLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUOutputDelta_kernel");
            }
            kCalculateSparseNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha);
            LAUNCHERROR("kCalculateSparseNonZeroELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha, lambda);
            LAUNCHERROR("kCalculateSparseNonZeroSELUOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxOutputDelta_kernel");
            break;                        
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0 / (NNFloat)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kIndexedCalculateSparseNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, slope);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel");
            break;                        
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, slope);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, alpha);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, alpha, lambda);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel");
            break;         
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * a * ((NNFloat)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t *pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((NNFloat)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * (a > (NNFloat)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat *pSparseWeight, unsigned char* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat *pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroSigmoidSparseOutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel");
            break;
            
        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, slope);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, alpha);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel");
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

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        pDelta[uOffset + pos]   = (a - t);      
    }
}

template<typename T> 
void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
        case SoftMax:
            kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * a;       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxOutputDelta_kernel");
            break;    

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonzeroSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (NNFloat)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel");
            break;    

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonzeroSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateSparseAnalogNonzeroSigmoidCrossEntropyOutputDelta_kernel");
            break;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat t           = (NNFloat)pSparseData[pos1] * (NNFloat)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel");
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseAnalogNonzeroSigmoidCrossEntropyOutputDelta_kernel");
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
            LAUNCHERROR("kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            break;                    
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
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

template<typename T> void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            break;                    
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._SMCE_zeroScale;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        NNFloat output          = (NNFloat)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = w * a;
        pDelta[pos]             = output;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._SMCE_oneScale * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (NNFloat)1.0);
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
kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._SMCE_oneScale * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0 / (NNFloat)(end - pos1));
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroScaleMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            break;            
            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._SMCE_oneScale * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (NNFloat)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = (pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0 / (NNFloat)(end - pos1);
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            NNFloat output      = (NNFloat)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = cData._SMCE_oneScale * (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero)
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
                kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroScaleMarginalCrossEntropyOutputDelta_kernel");
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel");
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
kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
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
void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
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
            kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel");
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
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
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
        pDelta[uOffset + pos]   = sgn(a - t) * ((NNFloat)1.0 - a * a);      
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
        pDelta[uOffset + pos]   = sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
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
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
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
        pDelta[uOffset + pos]   = sgn(a- t) * ((NNFloat)1.0 - a * a);      
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
        pDelta[uOffset + pos]   = sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
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
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
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
        pDelta[uOffset + pos]   = sgn(a - t) * ((NNFloat)1.0 - a * a);      
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
        pDelta[uOffset + pos]   = sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = (cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x) * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
            kCalculateRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData);
            LAUNCHERROR("kCalculateRELUL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, slope);
            LAUNCHERROR("kCalculateLRELUL1OutputDelta_kernel");
            break;
            
        case ExponentialLinear:
            kCalculateELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, alpha);
            LAUNCHERROR("kCalculateELUL1OutputDelta_kernel");
            break;            

        case ScaledExponentialLinear:
            kCalculateSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, alpha, lambda);
            LAUNCHERROR("kCalculateSELUL1OutputDelta_kernel");
            break;   
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((NNFloat)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a- t) * ((NNFloat)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 256.0);
        pDelta[uOffset + pos]   = sgn(a - t) * a * ((NNFloat)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((NNFloat)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * (a > (NNFloat)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dOffset        = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x] * stride;
        NNFloat a               = pUnit[uOffset + pos];
        NNFloat t               = (NNFloat)pData[dOffset + pos] * NNFloat(1.0 / 128.0);
        pDelta[uOffset + pos]   = sgn(a - t) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateIndexedSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            kCalculateIndexedTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedTanhL1OutputDelta_kernel");
            break;

        case Linear:
            kCalculateIndexedLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData);
            LAUNCHERROR("kCalculateIndexedRELUL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, slope);
            LAUNCHERROR("kCalculateIndexedLRELUL1OutputDelta_kernel");
            break;
            
        case ExponentialLinear:
            kCalculateIndexedELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, alpha);
            LAUNCHERROR("kCalculateIndexedELUL1OutputDelta_kernel");
            break;            

        case ScaledExponentialLinear:
            kCalculateIndexedSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSELUL1OutputDelta_kernel");
            break;   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w *  sgn(a) * a * ((NNFloat)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * a * ((NNFloat)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];    
        pDelta[pos]             = w * sgn(a) * ((NNFloat)1.0 - a * a);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((NNFloat)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];  
        pDelta[pos]             = w * sgn(a);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * (a > (NNFloat)0.0);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * (a > (NNFloat)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));    
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUL1OutputDelta_kernel(uint32_t position, NNFloat* pSparseWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        NNFloat w               = cData._deltaBoost_zero;
        if (pSparseWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pSparseWeight[dpos];
        }
        NNFloat a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroTanhL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroTanhL1OutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroLinearL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroRELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateSparseNonZeroRELUL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, slope);
            LAUNCHERROR("kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel");
            break;
            
        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha);
            LAUNCHERROR("kCalculateSparseNonZeroELUL1OutputDelta_kernel");
            break;
            
        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUL1OutputDelta_kernel");
            }
            kCalculateSparseNonZeroSELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha, lambda);
            LAUNCHERROR("kCalculateSparseNonZeroSELUL1OutputDelta_kernel");
            break;            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * a * ((NNFloat)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((NNFloat)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * (a > (NNFloat)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((a > (NNFloat)0.0) * lambda + (a <= (NNFloat)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, NNFloat slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        NNFloat w               = cData._deltaBoost_one * ((pSparseWeight != NULL) ? pSparseWeight[dpos] : (NNFloat)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            NNFloat a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (NNFloat)1.0) * ((a > (NNFloat)0.0) + (a <= (NNFloat)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda)
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
                kCalculateSparseRawSigmoidL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawSigmoidL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel");
            break;
        
        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawTanhL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel");
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawLinearL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta);
                LAUNCHERROR("kCalculateSparseRawRELUL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, slope);
                LAUNCHERROR("kCalculateSparseRawLRELUL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, slope);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel");
            break;
            
        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha);
                LAUNCHERROR("kCalculateSparseRawELUL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel");
            break;
            
        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL1OutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pSparseWeight, stride, size, pUnit, pDelta, alpha, lambda);
                LAUNCHERROR("kCalculateSparseRawSELUL1OutputDelta_kernel");
            }
            kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseWeight, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel");
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
kCalculateSigmoidHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        NNFloat d               = pDelta[pos];
        pDelta[pos]             = x * ((NNFloat)1.0 - x) * d;
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
kCalculateRELUHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
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
kCalculateLRELUHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
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

__global__ void
LAUNCH_BOUNDS()
kCalculateELUHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat alpha)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];
        if (x <= (NNFloat)0.0)
            pDelta[pos]        *= (x + alpha);            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSELUHadamardProduct_kernel(uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x               = pUnit[pos];        
        NNFloat delta           = pDelta[pos];
        if (x > (NNFloat)0.0)
        {
            delta              *= lambda;
        }
        else
        {
            delta              *= (x + lambda * alpha);
        }
        pDelta[pos]             = delta;
    }
}

void kCalculateHadamardProduct(Activation activation, uint64_t size, NNFloat scale, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    uint32_t blocks             = CalculateBlocks(size);
    NNFloat oneOverScale        = (NNFloat)1.0 / scale;        

    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
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
            kCalculateRELUHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            LAUNCHERROR("kCalculateRELUHadamardProduct_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, slope);
            LAUNCHERROR("kCalculateLRELUHadamardProduct_kernel");
            break;
            
        case ExponentialLinear:
            kCalculateELUHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, alpha);
            LAUNCHERROR("kCalculateELUHadamardProduct_kernel");
            break;

        case ScaledExponentialLinear:
            kCalculateSELUHadamardProduct_kernel<<<blocks, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta, alpha, lambda);
            LAUNCHERROR("kCalculateSELUHadamardProduct_kernel");
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
        REDUCE(r2)
    
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
        REDUCE(r2)
    
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

#define EXPLICITLY_INSTANTIATE_KERNELS(T)                                                                                                                                                                  \
template void kCalculateL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, T*, NNFloat, NNFloat, NNFloat);                                                                     \
template void kCalculateIndexedL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, T*, NNFloat, NNFloat, NNFloat);                                                   \
template void kCalculateCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, T*);                                                                                      \
template void kCalculateIndexedCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, T*);                                                                    \
template void kCalculateScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, T*);                                                                        \
template void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, T*);                                                      \
template void kCalculateOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, T*, NNFloat, NNFloat, NNFloat);                                                                       \
template void kCalculateIndexedOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, T*, NNFloat, NNFloat, NNFloat);                                                     \
template void kCalculateHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, T*);                                                                                             \
template void kCalculateIndexedHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, T*);                                                                           \
template void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint64_t*, uint64_t*, uint32_t*, T*, bool);                       \
template void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*, NNFloat*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);     \
template void kCalculateSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*,  NNFloat*, uint64_t*, uint64_t*, uint32_t*, NNFloat*, T*, bool, NNFloat, NNFloat, NNFloat);                   \
template void kCalculateIndexedSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, NNFloat*,  NNFloat*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, NNFloat*, T*, bool, NNFloat, NNFloat, NNFloat); \
/**/

EXPLICITLY_INSTANTIATE_KERNELS(NNFloat)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)
