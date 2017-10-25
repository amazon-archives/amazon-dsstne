/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include <values.h>
#include <sys/time.h>



static const Mode mode = Mode::Training;
//static const MODE mode = Prediction;
//static const MODE mode = Validation;

int main(int argc, char** argv)
{

    // Initialize GPU network
    getGpu().Startup(argc, argv); 
    getGpu().SetRandomSeed(12345ull);

    // Create Neural network
    int batch               = 1024;
    int total_epochs        = 60;
    int training_epochs     = 20;
    int epochs              = 0;
    float alpha             = 0.025f;
    float lambda            = 0.0001f;
    float lambda1           = 0.0f;
    float mu                = 0.5f;
    float mu1               = 0.0f;
    NNNetwork* pNetwork;

    // Load training data
    vector <NNDataSetBase*> vDataSet;
    if (mode == Mode::Training)
        vDataSet = LoadNetCDF("../../data/data_training.nc");
    else
        vDataSet = LoadNetCDF("../../data/data_test.nc");

#if 0        
    vector<tuple<uint64_t, uint64_t> > vMemory = vDataSet[0]->getMemoryUsage();
    uint64_t cpuMemory, gpuMemory;
    tie(cpuMemory, gpuMemory)   = vMemory[0];
    cout << "CPUMem: " << cpuMemory << " GPUMem: " << gpuMemory << endl;
    exit(-1);
#endif    
    // Create neural network
    if (argc < 2)
    {
        if (mode != Prediction)
            pNetwork = LoadNeuralNetworkJSON("config.json", batch, vDataSet);
        else
            pNetwork = LoadNeuralNetworkNetCDF("network.nc", batch);
    }
    else
        pNetwork = LoadNeuralNetworkNetCDF(argv[1], batch);
 
    // Dump memory usage
    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << endl;
    cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << endl;
    pNetwork->LoadDataSets(vDataSet);
    pNetwork->SetCheckpoint("check", 1);

    // Train, validate or predict based on operating mode
    if (mode == Mode::Validation)
    {
        pNetwork->SetTrainingMode(Nesterov);
        pNetwork->Validate();
    }
    else if (mode == Mode::Training)
    {
        pNetwork->SetTrainingMode(Nesterov);
        while (epochs < total_epochs)
        {
            //float margin        = (float)phase * 0.01f;
            //pNetwork->SetSMCE(1.0f - margin, margin, 30.0f, 1.0f); 
            pNetwork->Train(training_epochs, alpha, lambda, lambda1, mu, mu1);
            alpha              *= 0.8f;
            epochs             += training_epochs;
        }
        
        // Save final Neural network
        pNetwork->SaveNetCDF("network.nc");
    }
    else
    {

        // Determine output layer dimensions for top K calculations
        bool bFilterPast        = true; 
        const NNLayer* pLayer   = pNetwork->GetLayer("Output");
        uint32_t Nx, Ny, Nz, Nw;
        tie(Nx, Ny, Nz, Nw)     = pLayer->GetLocalDimensions();
        const uint32_t STRIDE   = Nx * Ny * Nz * Nw; 
        
        // Calculate Precision and recall
        unsigned int K                      = 100;
        
        // Find input dataset
        size_t inputIndex                   = 0;
        while ((inputIndex < vDataSet.size()) && (vDataSet[inputIndex]->_name != "input"))
            inputIndex++;
        if (inputIndex == vDataSet.size())
        {
            printf("Unable to find input dataset, exiting.\n");
            exit(-1);
        }
        
        // Find output dataset
        size_t outputIndex                  = 0;
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))
            outputIndex++;
        if (outputIndex == vDataSet.size())
        {
            printf("Unable to find output dataset, exiting.\n");
            exit(-1);
        }
       
        vector<NNFloat> vPrecision(K);
        vector<NNFloat> vRecall(K);
        vector<NNFloat> vNDCG(K);
        vector<uint32_t> vDataPoints(batch);
        GpuBuffer<NNFloat>* pbTarget        = new GpuBuffer<NNFloat>(batch * STRIDE, true);
        GpuBuffer<NNFloat>* pbOutput        = new GpuBuffer<NNFloat>(batch * STRIDE, true);
        NNDataSet<NNFloat>* pInputDataSet   = (NNDataSet<NNFloat>*)vDataSet[inputIndex];
        NNDataSet<NNFloat>* pOutputDataSet  = (NNDataSet<NNFloat>*)vDataSet[outputIndex];
        GpuBuffer<NNFloat>* pbKey           = new GpuBuffer<NNFloat>(batch * K, true);
        GpuBuffer<unsigned int>* pbUIValue  = new GpuBuffer<unsigned int>(batch * K, true);
        GpuBuffer<NNFloat>* pbFValue        = new GpuBuffer<NNFloat>(batch * K, true);
        NNFloat* pOutputValue               = pbOutput->_pSysData;
        bool bMultiGPU                      = (getGpu()._numprocs > 1);     
        GpuBuffer<NNFloat>* pbMultiKey      = NULL;
        GpuBuffer<NNFloat>* pbMultiFValue   = NULL;
        NNFloat* pMultiKey                  = NULL;
        NNFloat* pMultiFValue               = NULL;  
        cudaIpcMemHandle_t keyMemHandle;
        cudaIpcMemHandle_t valMemHandle;
        
        // Get P2P handles to multi-gpu data on node 0
        if (bMultiGPU)
        {
            if (getGpu()._id == 0)
            {   
                pbMultiKey                  = new GpuBuffer<NNFloat>(getGpu()._numprocs * batch * K, true);
                pbMultiFValue               = new GpuBuffer<NNFloat>(getGpu()._numprocs * batch * K, true);
                pMultiKey                   = pbMultiKey->_pDevData;
                pMultiFValue                = pbMultiFValue->_pDevData;
                cudaError_t status          = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");
                status                      = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");                      
            }
            MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            
            // Open up pointers to GPU 0 memory buffers
            if (getGpu()._id != 0)
            {
                cudaError_t status          = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");        
                status                      = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");    
            }  
        }
        
        for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
        {
            pNetwork->SetPosition(pos);
            pNetwork->PredictBatch();    
            unsigned int batch              = pNetwork->GetBatch();
            if (pos + batch > pNetwork->GetExamples())
                batch                       = pNetwork->GetExamples() - pos;
            NNFloat* pTarget                = pbTarget->_pSysData;
            memset(pTarget, 0, STRIDE * batch * sizeof(NNFloat));            
            const NNFloat* pOutputKey       = pNetwork->GetUnitBuffer("Output");
            NNFloat* pOut                   = pOutputValue;
            cudaError_t status              = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(NNFloat), cudaMemcpyDeviceToHost);
            RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
            
            for (int i = 0; i < batch; i++)
            {
                int j                       = pos + i;
                vDataPoints[i]              = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];
                
                for (size_t k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; k++)
                {
                    pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;
                }
                
                if (bFilterPast)
                {
                    for (size_t k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; k++)
                    {
                        pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;
                    }
                }
                pTarget                    += STRIDE;
                pOut                       += STRIDE;
            }
            pbTarget->Upload();
            pbOutput->Upload();
            kCalculateTopK(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);
            pbKey->Download();
            pbFValue->Download();
            
            // Do second pass to grab top K from the top n * K outputs if running multi-GPU
            if (bMultiGPU)
            {
            
                // Grab total datapoint counts
                MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
                
                // P2P gather Top K data from all processes
                uint32_t offset             = K * getGpu()._id;
                uint32_t kstride            = K * getGpu()._numprocs;
                cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);
                
                cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(NNFloat), pbFValue->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);                
                             
                // Do final Top K calculation on process 0 to produce the top K out of the top numprocs * K
                if (getGpu()._id == 0)
                {
                    kCalculateTopK(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);
                }
            }

            // Do P/R calculation entirely on process 0
            if (getGpu()._id == 0)
            {
                pbKey->Download();
                pbFValue->Download();
                NNFloat* pKey                   = pbKey->_pSysData;
                NNFloat* pValue                 = pbFValue->_pSysData;
                for (int i = 0; i < batch; i++)
                {
                    NNFloat p                   = vDataPoints[i];
                    NNFloat tp                  = 0.0f;
                    NNFloat fp                  = 0.0f;
                    NNFloat idcg                = 0.0f;
                    for (NNFloat pp = 0.0f; pp < p; pp++)
                    {
                        idcg                   += 1.0f / log2(pp + 2.0f);
                    }
                    NNFloat dcg                 = 0.0f;
                    for (int j = 0; j < K; j++)
                    {
                        //printf("%7d %3d %12.6f %12.6f\n", i, j, pKey[j], pValue[j]);
                        if (pValue[j] == 1.0f)
                        {
                            tp++;
                            dcg                += 1.0f / log2((float)(j + 2));
                        }
                        else
                            fp++;
                        vPrecision[j]          += tp / (tp + fp);
                        vRecall[j]             += tp / p;
                        vNDCG[j]               += dcg / idcg;
                    }
                    pKey                       += K;
                    pValue                     += K;
                }
            }
        }

        // Delete temporary data
        delete pbKey;
        delete pbFValue;
        delete pbUIValue;
        delete pbTarget;
        delete pbOutput;
        
        // Delete multi-GPU data and P2P handles if multi-GPU
        if (bMultiGPU)
        {
            if (getGpu()._id != 0)
            {
                cudaError_t status              = cudaIpcCloseMemHandle(pMultiKey);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
                status                          = cudaIpcCloseMemHandle(pMultiFValue);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");                
            }
            delete pbMultiKey;
            delete pbMultiFValue;
        }

        // Report results from process 0
        if (getGpu()._id == 0)
        {
            for (int i = 0; i < K; i++)
                printf("%d,%6.4f,%6.4f,%6.4f\n", i + 1, vPrecision[i] / pNetwork->GetExamples(), vRecall[i] / pNetwork->GetExamples(), vNDCG[i] / pNetwork->GetExamples()); 
        }

    }

    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    if (getGpu()._id == 0)
    {
        cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << endl;
        cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << endl;
    }
    
    // Save Neural network
    if (mode == Mode::Training)
        pNetwork->SaveNetCDF("network.nc");
    delete pNetwork;

    // Delete datasets
    for (auto p : vDataSet)
        delete p;

    getGpu().Shutdown();
    return 0;
}
