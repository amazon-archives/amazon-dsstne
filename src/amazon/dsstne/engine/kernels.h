/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

// GPU kernel utilities
uint32_t CalculateBlocks(uint64_t size);

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
void kAddBuffers(NNFloat* pDest, NNFloat* pSrc, uint64_t size);
void kAddBuffers2D(NNFloat* pDest, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height);
void kCopy2D(NNFloat* pDest, uint32_t dpitch, NNFloat* pSrc, uint32_t spitch, uint32_t width, uint32_t height);

// Sorting kernels
template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);
template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);

// Data load kernels
template<typename T> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom);
template<typename T> void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom);


// Sparse forward propagation kernels
void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pUnit, NNFloat beta);
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);
template<typename T> void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta);

// Sparse backpropagation kernels
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex);
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex);
void kCalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pDelta, NNFloat* pWeightGradient);
template<typename T> void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData);
template<typename T> void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData);
template<typename T> void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, T* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient);

// Error calculation functions, also non-templated to keep CUDA code in .cu files
template<typename T> NNFloat kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);
template<typename T> NNFloat kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData);

// Sparse Error Functions
NNFloat kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
NNFloat kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
NNFloat kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
NNFloat kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
NNFloat kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
NNFloat kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
template<typename T> NNFloat kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> NNFloat kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);

// Regularization error functions
NNFloat kCalculateRegularizationError(NNFloat lambda, NNFloat* pWeight, uint64_t size);

// Normalization functions
void kNormalizeWeights(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight);
void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude);
void kNormalizeWeightMagnitudes(NNFloat norm, uint32_t outputStride, uint32_t inputStride, NNFloat* pWeight, NNFloat* pMagnitude);
void kNormalizeDeltas(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta);
void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude);
void kNormalizeDeltaMagnitudes(NNFloat norm, uint32_t batch, uint32_t stride, NNFloat* pDelta, NNFloat* pMagnitude);

// Dropout kernels
void kCalculateDropout(NNFloat* pUnit, NNFloat* pRandom, uint32_t batch, uint32_t stride, NNFloat p);

// Delta functions for dense output layers
template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);
template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData);

// Delta functions for sparse output layers
void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex);
template<typename T> void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);
template<typename T> void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData);

// Delta functions for sparse output layers with analog output
template<typename T> void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit,  NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData);


// Sparseness penalty kernel for sparse hidden layers
void kCalculateSparsenessPenalty(uint32_t batch,  uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);

// Hadamard product for computing deltas of hidden layers
void kCalculateHadamardProduct(Activation activation, uint64_t size, NNFloat scale, NNFloat* pUnit, NNFloat* pDelta);

// Activation functions
void kCalculateSigmoidActivation(NNFloat* pData, uint64_t size);
void kCalculateTanhActivation(NNFloat* pData, uint64_t size);
void kCalculateReluActivation(NNFloat* pData, uint64_t size);
void kCalculateSoftMaxActivation(NNFloat* pData, uint32_t batch, uint32_t stride);

// SGD/Momentum/AdaGrad/Nesterov weight update kernels
void kSGDUpdateWeights(NNFloat lambda, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight);
void kSGDUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias);
void kMomentumUpdateWeights(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kMomentumUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kAdaGradUpdateWeights(NNFloat alpha, NNFloat lambda, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kAdaGradUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kNesterovShiftWeights(NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeight);
void kNesterovShiftBiases(NNFloat mu, uint32_t width, NNFloat* pBiasVelocity, NNFloat* pBias);
void kNesterovUpdateWeights(NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kNesterovUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);
void kRMSPropUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat mu, uint64_t size, NNFloat* pWeightVelocity, NNFloat* pWeightGradient, NNFloat* pWeight);
void kRMSPropUpdateBiases(NNFloat alpha, NNFloat mu, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBiasVelocity, NNFloat* pBias);

// Pooling Functions

