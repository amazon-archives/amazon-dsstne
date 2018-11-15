/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __DSSTNECALCULATE_H__
#define __DSSTNECALCULATE_H__

/**
 * Do a prediction calculation and return the top K results for a neural network.
 * @param pNetwork - an NNNetwork*
 * @param vDataSet - a vector<NNDataSetBase*>
 * @param cdl - a CDL instance
 * @param K - an unsigned integer that specifies the number K of results
 * @return a PyObject* that references a list of lists wherein the inner lists contains three floats
 */
static PyObject* dsstnecalculate_PredictTopK(NNNetwork* pNetwork, vector<NNDataSetBase*>& vDataSet, CDL& cdl, unsigned int K) {
    // Determine output layer dimensions for top K calculations
    bool bFilterPast        = false;    //true;
    const NNLayer* pLayer   = pNetwork->GetLayer("Output");
    uint32_t Nx, Ny, Nz, Nw;
    tie(Nx, Ny, Nz, Nw)     = pLayer->GetLocalDimensions();
    const uint32_t STRIDE   = Nx * Ny * Nz * Nw;

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

    // Renamed from 'batch' to 'cdlBatch' so that it doesn't hide 'batch' inside of the 'for' loop below
    int cdlBatch = cdl._batch;

    vector<NNFloat> vPrecision(K);
    vector<NNFloat> vRecall(K);
    vector<NNFloat> vNDCG(K);
    vector<uint32_t> vDataPoints(cdlBatch);
    GpuBuffer<NNFloat>* pbTarget        = new GpuBuffer<NNFloat>(cdlBatch * STRIDE, true);
    GpuBuffer<NNFloat>* pbOutput        = new GpuBuffer<NNFloat>(cdlBatch * STRIDE, true);
    NNDataSet<NNFloat>* pInputDataSet   = (NNDataSet<NNFloat>*)vDataSet[inputIndex];
    NNDataSet<NNFloat>* pOutputDataSet  = (NNDataSet<NNFloat>*)vDataSet[outputIndex];
    GpuBuffer<NNFloat>* pbKey           = new GpuBuffer<NNFloat>(cdlBatch * K, true);
    GpuBuffer<unsigned int>* pbUIValue  = new GpuBuffer<unsigned int>(cdlBatch * K, true);
    GpuBuffer<NNFloat>* pbFValue        = new GpuBuffer<NNFloat>(cdlBatch * K, true);
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
            pbMultiKey                  = new GpuBuffer<NNFloat>(getGpu()._numprocs * cdlBatch * K, true);
            pbMultiFValue               = new GpuBuffer<NNFloat>(getGpu()._numprocs * cdlBatch * K, true);
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
            
        for (unsigned int i = 0; i < batch; i++)
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
            for (unsigned int i = 0; i < batch; i++)
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
                for (unsigned int j = 0; j < K; j++)
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

    // Return the results from process 0 as a Python list of lists, where each inner list reports one of the top K results.
    PyObject* python_list = PyList_New(K);
    if (getGpu()._id == 0)
      {
        for (unsigned int i = 0; i < K; i++)
          {
            // Building a Python list via [...] isn't strictly necessary.
            PyList_SetItem(python_list, i, Py_BuildValue("[fff]",
                                                         vPrecision[i] / pNetwork->GetExamples(),
                                                         vRecall[i] / pNetwork->GetExamples(),
                                                         vNDCG[i] / pNetwork->GetExamples()));
          }
      }
    return python_list;
}

/**
 * Transpose a NumPy 2D matrix to create a contiguous matrix that is returned through the output NumPy matrix.
 * @param ASINWeightArray - the output NumPy 2D matrix that is modified by this Transpose function
 * @param pEmbeddingArray - the input NumPy 2D matrix
 * @throws exception upon failure to parse arguments or if the matrices are incompatible as determined by dsstnecalculate_Transpose
 */
static PyObject* dsstnecalculate_Transpose(PyArrayObject* vASINWeight, PyArrayObject* vEmbedding) {
    // The matrices must be single-precision floating point.
    CheckNumPyArray(vASINWeight);
    CheckNumPyArray(vEmbedding);
    // The destination matrix must be a rank 2 matrix.
    if (PyArray_NDIM(vASINWeight) != 2) {
        std::string message = "Normalize received incorrect vASINWeight matrix dimensionality; expected = 2  received = "
          + std::to_string(PyArray_NDIM(vASINWeight));
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    // The source matrix must be a rank 2 matrix.
    if (PyArray_NDIM(vEmbedding) != 2) {
        std::string message = "Normalize received incorrect vEmbedding matrix dimensionality; expected = 2  received = "
          + std::to_string(PyArray_NDIM(vEmbedding));
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    // The destination matrix dimensions must not exceed the source matrix dimensions.
    if (PyArray_DIM(vASINWeight, 0) > PyArray_DIM(vEmbedding, 1) || PyArray_DIM(vASINWeight, 1) > PyArray_DIM(vEmbedding, 0)) {
        std::string message = "Normalize received vASINWeight dimensions that exceed vEmbedding dimensions; vASINWeight dimensions = ("
          + std::to_string(PyArray_DIM(vASINWeight, 0)) + ", " + std::to_string(PyArray_DIM(vASINWeight, 1)) + ")  vEmbedding dimensions = ("
          + std::to_string(PyArray_DIM(vEmbedding, 0)) + ", " + std::to_string(PyArray_DIM(vEmbedding, 1)) + ")";
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    NNFloat* pASINWeight = reinterpret_cast<NNFloat*>(PyArray_DATA(vASINWeight));
    NNFloat* pEmbedding = reinterpret_cast<NNFloat*>(PyArray_DATA(vEmbedding));
    // The dimensions of vASINWeight specify the 'for' loop termination conditions
    npy_intp STRIDE = PyArray_DIM(vASINWeight, 0);
    npy_intp EmbeddingExamples = PyArray_DIM(vASINWeight, 1);
    for (int i = 0; i < EmbeddingExamples; i++)
    {
        for (int j = 0; j < STRIDE; j++)
        {
            pASINWeight[j * EmbeddingExamples + i] = pEmbedding[i * STRIDE + j];
        }
    }
    Py_RETURN_NONE;
}

/**
 * Calculate the MRR for a neural network.
 * @param pNetwork - an NNNetwork*
 * @param vDataSet - a vector<NNDataSetBase*>
 * @param pOutputLayer - an NNLayer* that references the output layer
 * @param outputIndex - an unsigned integer that represents the index to the output data set
 * @param NxOutput - an unsigned integer that represents the leading dimension of the output layer
 * @return a PyObject* that references the MRR float
 */
static PyObject* dsstnecalculate_CalculateMRR(NNNetwork* pNetwork, vector<NNDataSetBase*>& vDataSet, NNLayer* pOutputLayer, uint32_t outputIndex, uint32_t NxOutput) {
    
    // Get the output data set and calculate MRR
    NNDataSet<uint32_t>* pOutputDataSet  = (NNDataSet<uint32_t>*)vDataSet[outputIndex];
    NNFloat MRR = (NNFloat)0.0;
    vector<NNFloat> vOutput(NxOutput * pNetwork->GetBatch());
    //vector<NNFloat> vHidden(NxHidden * pNetwork->GetBatch());
    for (size_t pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
    {
        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();    
        unsigned int batch              = pNetwork->GetBatch();
        if (pos + batch > pNetwork->GetExamples())
            batch                       = pNetwork->GetExamples() - pos;
        pOutputLayer->GetUnits(vOutput);
      //  pHiddenLayer->GetUnits(vHidden);
        
        // Don't try to APIify this part, this should be in python for the python case
        //  
        for (size_t i = 0; i < batch; i++)
        {
            size_t offset = i * NxOutput;
            NNFloat asinValue = -99999.0;
            for (size_t j = pOutputDataSet->_vSparseStart[i + pos]; j < pOutputDataSet->_vSparseEnd[i + pos]; j++)
                asinValue = std::max(asinValue, vOutput[offset + pOutputDataSet->_vSparseIndex[j]]);
            size_t asinCount = 1;
            NNFloat maxValue = -50000.0;
            NNFloat minValue = 50000.0;
            for (size_t j = 0; j < NxOutput; j++)
            {
                if (maxValue < vOutput[offset + j])
                    maxValue = vOutput[offset + j];
                if (minValue > vOutput[offset + j])
                    minValue = vOutput[offset + j];                
                if (vOutput[offset + j] > asinValue)
                    asinCount++;
            }
#if 0            
            printf("%6lu %16.8f %16.8f %16.8f %lu %u\n", pos + i, asinValue, minValue, maxValue, asinCount, pOutputDataSet->_vSparseIndex[pOutputDataSet->_vSparseStart[i + pos]]);
            
            {
                NNFloat maxValue = -50000.0;
                NNFloat minValue = 50000.0; 
                size_t ioffset = i * NxHidden;
                NNFloat asinValue = -99999999.0f;
                NNFloat asinValue1 = -99999999.0f;                
                size_t asin = pOutputDataSet->_vSparseIndex[pOutputDataSet->_vSparseStart[i + pos]];
                for (size_t j = 0; j < NxOutput; j++)
                {
                   
                    size_t offset = j * NxHidden;
                    float dp = 0.0f;
                    float dp1 = 0.0f;
                    for (size_t k = 0; k < NxHidden; k++)
                    {
                        dp += vHidden[ioffset + k] * vEmbedding[offset + k];
                        dp1 += vHidden[ioffset + k] * vASINWeight[j + k * NxOutput];
                    }
                    if (j == asin)
                    {
                        asinValue = dp;
                        asinValue1 = dp1;
                    }
                    if (dp < minValue)
                        minValue = dp;
                    if (dp > maxValue)
                        maxValue = dp;
                }
                printf("%lu %16.8f %16.8f | %16.8f %16.8f %lu\n", pos + i, asinValue, asinValue1, minValue, maxValue, asin);
            }
#endif            
            
            
            MRR += (1.0 / (NNFloat)asinCount);
        }
    }
    MRR /= (NNFloat)(pNetwork->GetExamples());
    return Py_BuildValue("f", MRR);
}
#endif
