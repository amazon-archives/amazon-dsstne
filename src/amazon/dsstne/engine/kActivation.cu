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

__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_i   = (int*) address;
    int old             = *address_as_i, assumed;
    do 
    {
        assumed         = old;
        old             = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } 
    while (assumed != old);
    return __int_as_float(old);
}

void SetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

void GetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy From cData failed");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidActivation_kernel(NNFloat* pData, uint64_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat a               = 1.0f / (1.0f + exp(-pData[pos]));
        pData[pos]              = a;
    }
}


void kCalculateSigmoidActivation(NNFloat* pData, uint64_t size)
{
    uint32_t blocks             = CalculateBlocks(size);
    kCalculateSigmoidActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateSigmoidActivation_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateTanhActivation_kernel(NNFloat* pData, uint64_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos]              = tanh(pData[pos]);
}

void kCalculateTanhActivation(NNFloat* pData, uint64_t size)
{
    uint32_t blocks             = CalculateBlocks(size);
    kCalculateTanhActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateTanhActivation_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateReluActivation_kernel(NNFloat* pData, uint64_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos]              = max(0.0f, pData[pos]);
}

void kCalculateReluActivation(NNFloat* pData, uint64_t size)
{
    uint32_t blocks             = CalculateBlocks(size);
    kCalculateReluActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateReluActivation_kernel");
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxActivation_kernel(NNFloat* pData, uint32_t stride)
{
    __shared__ unsigned long long int sAccumulator;
    __shared__ NNFloat sMaxValue;

    if (threadIdx.x == 0)
    {
        sAccumulator            = 0;
        sMaxValue               = (NNFloat)-99999999.0f;
    }
    __syncthreads();
    

    // Move data pointer to proper row, calculate activations, and sum them up as well as find maxmum output
    pData                      += blockIdx.x * stride;
    uint32_t pos                = threadIdx.x;
    NNFloat maxValue            = (NNFloat)-9999999999.0;
    
    // Calculate max value to improve numerical stability (Theano does this so I'll assume it's a good idea)
    while (pos < stride)
    {
        NNFloat z               = pData[pos];
        maxValue                = max(z, maxValue);
        pos                    += blockDim.x;
    }
    
    // Reduce maxValue within and between warps
    uint32_t tgx                = threadIdx.x & cData._warpMask;    
    maxValue                    = max(maxValue, __shfl(maxValue, tgx ^ 1));
    maxValue                    = max(maxValue, __shfl(maxValue, tgx ^ 2));
    maxValue                    = max(maxValue, __shfl(maxValue, tgx ^ 4));
    maxValue                    = max(maxValue, __shfl(maxValue, tgx ^ 8));
    maxValue                    = max(maxValue, __shfl(maxValue, tgx ^ 16));

    // Convert to 64-bit int to work around GPU instruction set deficiency
    if (tgx == 0) 
        atomicMax(&sMaxValue, maxValue);
    __syncthreads();        
    maxValue                    = sMaxValue;       

    // Calculate sum
    pos                         = threadIdx.x;
    NNFloat sum                 = (NNFloat)0.0;
    while (pos < stride)
    {
        NNFloat z               = pData[pos];
        sum                    += exp(z - maxValue);
        pos                    += blockDim.x;
    }    
         
    // Reduce sums within and between warps
    sum                        += __shfl(sum, tgx ^ 1);
    sum                        += __shfl(sum, tgx ^ 2);
    sum                        += __shfl(sum, tgx ^ 4);
    sum                        += __shfl(sum, tgx ^ 8);
    sum                        += __shfl(sum, tgx ^ 16);
    unsigned long long int lsum = llitoulli(llrintf(ERRORSCALEF * sum));
    if (tgx == 0) 
        atomicAdd(&sAccumulator, lsum);
    __syncthreads();               
    NNFloat norm                = (NNFloat)1.0 / (NNFloat)((double)sAccumulator * ONEOVERERRORSCALE);
    

    // Normalize output by dividing by sum of activations
    pos                         = threadIdx.x;
    while (pos < stride)
    {
        NNFloat z               = pData[pos];
        NNFloat a               = exp(z - maxValue);
        pData[pos]              = min((NNFloat)1.0, a * norm);
        pos                    += blockDim.x;
    }    

}
void kCalculateSoftMaxActivation(NNFloat* pData, uint32_t batch, uint32_t stride)
{
    uint32_t warps              = getGpu()._threadsPerBlock / getGpu()._warpSize;
    kCalculateSoftMaxActivation_kernel<<<batch, getGpu()._threadsPerBlock>>>(pData, stride);
    LAUNCHERROR("kCalculateSoftMaxActivation_kernel");
}


