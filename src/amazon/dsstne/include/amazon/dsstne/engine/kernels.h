/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef KERNELS_H
#define KERNELS_H

// GPU kernel utilities
uint32_t CalculateBlocks(uint64_t size);
template<typename T> __device__ T sgn(T x) { return (x > 0) - (x < 0); }

// GPU Data copy calls
void SetKernelsGpuData();
void GetKernelsGpuData();
void SetKLossGpuData();
void GetKLossGpuData();
void SetKActivationGpuData();
void GetKActivationGpuData();
void SetKDeltaGpuData();
void GetKDeltaGpuData();

// Miscellaneous kernels
void kScaleAndBias(NNFloat* pData, uint64_t size, NNFloat scale, NNFloat bias);
void kAddBias(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch);
void kAddDualBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch);
void kAddTripleBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch);
void kAddQuadBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch);
void kClearUnit(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch);
void kClearDualSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch);
void kClearTripleSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch);
void kClearQuadSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch);
void kUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias);
void kCalculateTopK(NNFloat* pOutputKey, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);
void kCalculateTopK(NNFloat* pOutputKey, NNFloat* pOutputValue, NNFloat *pKey, NNFloat* pValue, uint32_t batch, uint32_t width, uint32_t k);
void kCalculateTopK(NNFloat* pOutputKey, uint32_t* pOutputValue, NNFloat *pKey, uint32_t * pValue, uint32_t batch, uint32_t width, uint32_t k);
void kCalculateKSparse(NNFloat* pUnit, uint32_t batch, uint32_t stride, uint32_t kSparse);
void kAddScaleBuffers(NNFloat* pDest, NNFloat* pSrc, NNFloat scale, uint64_t size);
void kAddBuffers(NNFloat* pDest, NNFloat* pSrc, uint64_t size, cudaStream_t stream = 0);
void kAddBuffers2D(NNFloat* pDest, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);
void kCopy2D(NNFloat* pDest, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

// Sorting kernels
template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);
template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);

// Data load kernels
template<typename T> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom);
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom);
template<typename T> void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom);
template<typename T> void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom);

// Sparse forward propagation kernels
void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pUnit, NNFloat beta);
void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta);
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);
void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);

// Sparse backpropagation kernels
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
void kCalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pDelta, NNFloat* pWeightGradient);
template<typename T> void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);




template<typename T> void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
template<typename T> void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
template<typename T> void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData);
void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient);

// Error calculation functions, also non-templated to keep CUDA code in .cu files
template<typename T> NNFloat kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);
template<typename T> NNFloat kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData);

// Sparse Error Functions
NNFloat kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
NNFloat kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
NNFloat kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
NNFloat kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
NNFloat kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight);
template<typename T> NNFloat kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> NNFloat kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> NNFloat kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData);
template<typename T> NNFloat kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> NNFloat kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

// Regularization error functions
NNFloat kCalculateRegularizationError(NNFloat lambda, NNFloat lambda1, NNFloat* pWeight, uint64_t size);

// Normalization functions
void kNormalizeWeights(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight);
void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude);
void kNormalizeWeightMagnitudes(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude);
void kNormalizeDeltas(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta);
void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude);
void kNormalizeDeltaMagnitudes(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude);

// Dropout kernels
void kCalculateScaledBiasedDropout(NNFloat* pUnit, NNFloat* pRandom, uint32_t batch, uint32_t stride, NNFloat p, NNFloat target, NNFloat a, NNFloat b);
void kCalculateDropout(NNFloat* pUnit, NNFloat* pRandom, uint32_t batch, uint32_t stride, NNFloat p, NNFloat target);

// Delta functions for dense output layers
template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData);
template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData);
template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData);

// Delta functions for sparse output layers
void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero);
void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
void kCalculateSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
void kCalculateIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat scope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat scope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

// Delta functions for sparse output layers with analog output
template<typename T> void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);
template<typename T> void kCalculateIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pSparseWeight, T* pSparseData, bool bSparseIgnoreZero, NNFloat slope, NNFloat alpha, NNFloat lambda);



// Sparseness penalty kernel for sparse hidden layers
void kCalculateSparsenessPenalty(uint32_t batch,  uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat p, NNFloat beta);

// Hadamard product for computing deltas of hidden layers
void kCalculateHadamardProduct(Activation activation, uint64_t size, NNFloat scale, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda);

// Activation functions
void kCalculateSigmoidActivation(NNFloat* pData, uint64_t size);
void kCalculateTanhActivation(NNFloat* pData, uint64_t size);
void kCalculateRELUActivation(NNFloat* pData, uint64_t size);
void kCalculateELUActivation(NNFloat* pData, uint64_t size, NNFloat alpha);
void kCalculateSELUActivation(NNFloat* pData, uint64_t size, NNFloat alpha, NNFloat lambda);
void kCalculateLRELUActivation(NNFloat* pData, uint64_t size, NNFloat slope);
void kCalculateSoftMaxActivation(NNFloat* pData, uint32_t batch, uint32_t stride);

// SGD/Momentum/AdaGrad/Nesterov weight update kernels
void kSGDUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight);
void kSGDUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias);
void kMomentumUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kMomentumUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kAdaGradUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kAdaGradUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kNesterovShiftWeights(NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeight);
void kNesterovShiftBiases(NNFloat mu, uint32_t width, NNFloat* pBiasVelocity, NNFloat* pBias);
void kNesterovUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kNesterovUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kRMSPropUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kRMSPropUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kAdaDeltaUpdateWeights(NNFloat lambda, NNFloat lambda1, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeightGradientVelocity, NNFloat* pWeight);
void kAdaDeltaUpdateBiases(NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBiasGradientVelocity, NNFloat* pBias);
void kAdamUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1, NNFloat t, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeightGradientVelocity, NNFloat* pWeight);
void kAdamUpdateBiases(NNFloat alpha, NNFloat mu, NNFloat mu1, NNFloat t, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBiasGradientVelocity, NNFloat* pBias);

// Pooling Functions
void kCalculateMaxout(NNFloat* pSrc, size_t size, NNFloat* pDst);
void kCalculateCosine(NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDPOut, NNFloat* pAOut, NNFloat* pBOut, uint32_t outStride);                        
void kCalculateDotProduct(NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDPOut, uint32_t outStride);                        


// Pooling deltas
void kCalculateMaxoutDelta(NNFloat* pSrc, NNFloat* pSrcDelta, size_t size, NNFloat beta, NNFloat* pDst, NNFloat* pDstDelta);
void kCalculateDotProductDelta(NNFloat* pDPDelta, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride);
void kCalculateCosineDelta(NNFloat* pDPDelta, NNFloat* pDP, NNFloat* pA, NNFloat* pB, NNFloat* p0Vector, NNFloat* pVector, uint32_t batch, uint32_t stride, NNFloat* pDelta0, NNFloat beta0, NNFloat* pDelta, NNFloat beta, uint32_t inputStride);



// CUDA macros and routines
#ifdef __NVCC__
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

// Handle arbitrary API churn from new and improved thread within thread model
#if (CUDA_VERSION >= 9000)
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
#define SHFL(x, lane) __shfl((x), (lane))
#define BALLOT(predicate) __ballot(predicate)
#define ANY(predicate) __any(predicate)
#endif // CUDA_VERSION >= 9000


#define REDUCEERROR(error) \
    if (ANY(error != (NNFloat)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    } 


#define REDUCE(a) \
    if (ANY((a) != (NNFloat)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    } 


#endif // __NVCC__

#endif // KERNELS_H
