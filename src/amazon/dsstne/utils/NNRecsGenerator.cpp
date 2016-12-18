/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>

#include "NNRecsGenerator.h"
#include "GpuTypes.h"
#include "Utils.h"
#include "Filters.h"

using namespace std;

const string NNRecsGenerator::DEFAULT_LAYER_RECS_GEN_LABEL = "Output";
const string NNRecsGenerator::DEFAULT_SCORE_PRECISION = "4.3f";

// Multiplicative of xK. 
// We sorting TOPK_SCALAR times of xK mainly in concern with multi GPU sorting
// sorting the topK from xK* #GPUs has some glitches because xK*#GPUs are usually small
// sorting the topK from xK* #GPUs * TOPK_SCALAR is OK though 
const unsigned int NNRecsGenerator::TOPK_SCALAR = 5;

/**
We should allocate and deallocate the GPU memory once to save time on allocating and deallocating the
GPU Memory
The GPU buffer is created for sorting
and the parametes are batch size
how many recs do you need to Sort
The Filered location  which is used as Buffer of the Recs Generated to sort
*/
NNRecsGenerator::NNRecsGenerator(unsigned int xBatchSize,
                                 unsigned int xK,
                                 unsigned int xOutputBufferSize,
                                 const string &layer,
                                 const string &precision)
  : pbKey(new GpuBuffer<NNFloat>(xBatchSize * xK * TOPK_SCALAR, true)),
    pbUIValue(new GpuBuffer<unsigned int>(xBatchSize * xK * TOPK_SCALAR, true)),
    pFilteredOutput(new GpuBuffer<NNFloat>(xOutputBufferSize, true)),
    recsGenLayerLabel(layer),
    scorePrecision(precision)
{
}

void NNRecsGenerator::generateRecs(const NNNetwork *xNetwork,
                                   unsigned int xK,
                                   const FilterConfig *xFilterSet,
                                   const vector<string> &xCustomerIndex,
                                   const vector<string> &xFeatureIndex)
{
    timeval t0;
    gettimeofday(&t0, NULL);
    int lBatch    = xNetwork->GetBatch();
    int lExamples = xNetwork->GetExamples();
    int lPosition = xNetwork->GetPosition();
    if (lPosition + lBatch > lExamples)
    {
        lBatch = lExamples - lPosition;
    }

    // Variables for multi GPU memory copy and sorting 
    bool bMultiGPU                           = (getGpu()._numprocs > 1);
    unique_ptr<GpuBuffer<NNFloat>> pbMultiKey;
    unique_ptr<GpuBuffer<unsigned int>> pbMultiUIValue;
    unique_ptr<GpuBuffer<unsigned int>> pbUIValueCache;
    NNFloat* pMultiKey                       = NULL;
    unsigned int* pMultiUIValue              = NULL;
    unsigned int* pUIValueCache              = NULL;
 
    cudaIpcMemHandle_t keyMemHandle;
    cudaIpcMemHandle_t valMemHandle;
    const NNFloat* dOutput         = xNetwork->GetUnitBuffer(recsGenLayerLabel);
    const NNLayer* pLayer          = xNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    tie(lx, ly, lz, lw)            = pLayer->GetDimensions();
    int lOutputStride              = lx * ly * lz * lw;
    unsigned int llx, lly, llz, llw;
    tie(llx, lly, llz, llw)        = pLayer->GetLocalDimensions();
   
    // Local Stride is how many FEATUREs actually in one GPU
    int lLocalOutputStride         = llx * lly * llz * llw;
    unsigned int outputBufferSize  = lLocalOutputStride * lBatch;
    if (!bMultiGPU)
    {
        outputBufferSize = xNetwork->GetBufferSize(recsGenLayerLabel);
    }

    unique_ptr<float[]> hOutputBuffer(new float[outputBufferSize]);
   
    // Get P2P handles to multi-gpu data on node 0
    if (bMultiGPU)
    {
        if (getGpu()._id == 0)
        {
            const size_t bufferSize = getGpu()._numprocs * lBatch * xK * TOPK_SCALAR;

            pbMultiKey.reset(new GpuBuffer<NNFloat>(bufferSize, true));
            pbMultiUIValue.reset(new GpuBuffer<unsigned int>(bufferSize, true));

            pMultiKey                   = pbMultiKey->_pDevData;
            pMultiUIValue               = pbMultiUIValue->_pDevData;
            cudaError_t status          = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");
            status                      = cudaIpcGetMemHandle(&valMemHandle, pMultiUIValue);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiUIValue");
        }

        MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Open up pointers to GPU 0 memory buffers
        if (getGpu()._id != 0)
        {
            cudaError_t status          = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");
            status                      = cudaIpcOpenMemHandle((void**)&pMultiUIValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
        }

    }
    cudaMemcpy(hOutputBuffer.get(), dOutput, outputBufferSize * sizeof(NNFloat), cudaMemcpyDeviceToHost);
    // Iterate through all the filters and apply filters for each sample in the lBatch

    // We dont need the memory buffer if we have only one filter as copying the memory effects performance  
    // TODO need to add a better time wrapper to measure the time duration of a  function call
    timeval timeStart;
    gettimeofday(&timeStart, NULL);
    for (int j = 0; j < lBatch; j++)
    {
        int sampleIndex = lPosition + j;

        // offset is the starting FEATUREs in this GPU to the first one in global FEATURE Index 
        int offset = getGpu()._id * lLocalOutputStride;
        xFilterSet->applySamplesFilter(hOutputBuffer.get() + j * lLocalOutputStride, sampleIndex, offset, lLocalOutputStride);
    }

    timeval timeEnd;
    pFilteredOutput->Upload(hOutputBuffer.get());
    // TODO: Add Node Filter support for multi GPU 
    // Each GPU sorting its top xK * TOPK_SCALAR
    kCalculateTopK(pFilteredOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, lLocalOutputStride, xK * TOPK_SCALAR);

    if (bMultiGPU)
    {
        // P2P gather Top K data from all processes
        uint32_t offset             = xK * TOPK_SCALAR * getGpu()._id;
        uint32_t kstride            = xK * TOPK_SCALAR * getGpu()._numprocs;
        // Copy values to buffer which is on GPU 0
        cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, xK * TOPK_SCALAR * sizeof(NNFloat), xK * TOPK_SCALAR * sizeof(NNFloat), lBatch, cudaMemcpyDefault);
        // Copy indices ( local to each GPU ) to buffer which is on GPU0 
        cudaMemcpy2D(pMultiUIValue + offset, kstride * sizeof(unsigned int), pbUIValue->_pDevData, xK * TOPK_SCALAR * sizeof(unsigned int), xK * TOPK_SCALAR * sizeof(unsigned int), lBatch, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        // Do final Top K calculation on process 0 to produce the top K out of the top numprocs * K
        if (getGpu()._id == 0)
        {
            // Sort for values for xK * TOPK_SCALAR, pbUIValue are positions on pbMultiKey
            kCalculateTopK(pbMultiKey->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);

            // Buffer for local Indices
            pbUIValueCache.reset(new GpuBuffer<unsigned int>(getGpu()._numprocs * lBatch * xK * TOPK_SCALAR, true));

            // pbUIValueCache for local index of FEATUREs in its GPU
            kCalculateTopK(pbMultiKey->_pDevData, pbMultiUIValue->_pDevData, pbKey->_pDevData, pbUIValueCache->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);
        }
    }

    if (getGpu()._id == 0)
    {
        const char *fileName = xFilterSet->getOutputFileName().c_str();
        gettimeofday(&timeEnd, NULL);
        cout << "Time Elapsed for Filtering and selecting Top " << xK << " recs: " << elapsed_time(timeEnd, timeStart) << endl;
        cout << "Writing to " << fileName << endl;
        FILE *fp = fopen(fileName, "a");
        pbKey->Download();
        pbUIValue->Download();
        NNFloat* pKey                   = pbKey->_pSysData;
        unsigned int* pIndex            = pbUIValue->_pSysData;

        if (bMultiGPU)
        {
            pbUIValueCache->Download();
            pUIValueCache               = pbUIValueCache->_pSysData;
        }

        string strFormat = "%s,%" + scorePrecision + ":";
        for (int j = 0; j < lBatch; j++)
        {
            fprintf(fp, "%s%c", xCustomerIndex[lPosition + j].c_str(), '\t');
            for (int x = 0; x < xK; ++x)
            {
                const size_t bufferPos = j * xK * TOPK_SCALAR + x;

                // Single GPU case, FEATURE index is global
                int finalIndex = pIndex[bufferPos];
                float value = pKey[bufferPos];
                if (bMultiGPU)
                {
                    // Multi GPU case. Need to do two level look up
                    // which GPU this index comes from
                    int gpuId = finalIndex / (xK * TOPK_SCALAR);
                    // Local index within one GPU
                    int localIndex = pUIValueCache[bufferPos];
                    int globalIndex = gpuId * lLocalOutputStride + localIndex; 
                    if (globalIndex < xFeatureIndex.size())
                    {
                        fprintf(fp, strFormat.c_str(), xFeatureIndex[globalIndex].c_str(), value);
                    } 
                }
                else if (finalIndex < xFeatureIndex.size())
                {
                    fprintf(fp, strFormat.c_str(), xFeatureIndex[finalIndex].c_str(), value);
                }
            }

            fprintf(fp, "\n");
        }
        fclose(fp);
        gettimeofday(&timeEnd, NULL);
        cout << "Time Elapsed for Writing to file: " << elapsed_time(timeEnd, timeStart) << endl;
    }

    // Delete multi-GPU data and P2P handles if multi-GPU
    if (bMultiGPU)
    {
        if (getGpu()._id != 0)
        {
            cudaError_t status              = cudaIpcCloseMemHandle(pMultiKey);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
            status                          = cudaIpcCloseMemHandle(pMultiUIValue);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
        }
    } 
}
