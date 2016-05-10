/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNWEIGHT_H

class NNWeight {
    friend class NNNetwork;
    friend class NNLayer;
    friend NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, uint32_t batch);
    const NNLayer&          _inputLayer;            // Source of activations
    const NNLayer&          _outputLayer;           // Output destination/Delta sources
    const bool              _bShared;               // Are we using another set of shard weights
    const bool              _bTransposed;           // Use tranpose of shared weights?
    bool                    _bLocked;               // Prevent updates to this set of weights
    NNWeight*               _pSharedWeight;         // Pointer to shared weight
    uint32_t                _sharingCount;          // Number of layers utilizing weight matrix
    uint32_t                _updateCount;           // Indicates number of times a shared weight matrix has been updated
    uint64_t                _width;                 // Number of output units in outgoing FC layer or width of convolution for Convolutional layer
    uint64_t                _height;                // Number of input units in incoming FC layer or height of convolution for 2D/3D Convolutional layer
    uint64_t                _length;                // Length of 3D convolution for Convolutional layer
    uint64_t                _size;                  // Total size of weights
    NNFloat                 _norm;                  // Maximum allowable weight vector length (default 0, unconstrained)
    vector<NNFloat>         _vWeight;               // CPU weight array
    vector<NNFloat>         _vBias;                 // CPU bias array
    GpuBuffer<NNFloat>*     _pbWeight;              // GPU weight array 
    GpuBuffer<NNFloat>*     _pbBias;                // GPU bias array
    GpuBuffer<NNFloat>*     _pbWeightGradient;      // Accumulated gradient per batch
    GpuBuffer<NNFloat>*     _pbWeightVelocity;      // Velocity used for momentum and RMSProp
    GpuBuffer<NNFloat>*     _pbBiasVelocity;        // Velocity used for momentum and RMSProp
    NNWeight(NNLayer& inputLayer, NNLayer& outputLayer, bool bShared = false, bool bTransposed = false, bool bLocked = false, NNFloat maxNorm = 0.0f);
    ~NNWeight();
    void ClearSharedGradient();
    void ClearGradient();
    NNFloat CalculateRegularizationError(NNFloat lambda);
    void ClearVelocity();
    void Randomize();
    void Lock();
    void Unlock();
    void Dump(string fname, NNFloat* pBuffer);
    void RefreshState(TrainingMode trainingMode);
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat mu);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index, NNFloat* pWeight = NULL, NNFloat* pBias = NULL);

public:
    bool CopyWeights(NNWeight* pWeight);
    bool SetNorm(NNFloat norm);
};


struct NNWeightDescriptor
{
    string                  _inputLayer;
    string                  _outputLayer;
    uint64_t                _width;
    uint64_t                _height;
    uint64_t                _length;
    vector<NNFloat>         _vWeight;
    vector<NNFloat>         _vBias;
    bool                    _bShared;
    bool                    _bTransposed;
    bool                    _bLocked;
    NNFloat                  _norm;
    string                  _sourceInputLayer;     // _sourceInputLayer and _sourceOutputLayer collectively
    string                  _sourceOutputLayer;    // specify which weight matrix will be shared here

    NNWeightDescriptor();
};

bool LoadNNWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNWeightDescriptor& wd);
ostream& operator<< (ostream& out, NNWeightDescriptor& d);
uint32_t MPI_Bcast_NNWeightDescriptor(NNWeightDescriptor& d);
#define NNWEIGHT_H
#endif
