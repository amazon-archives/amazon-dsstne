/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNWEIGHT_H
#define NNWEIGHT_H

#include <memory>

class NNWeight {
public:
    enum Transform
    {
        Convolution,
        Linear
    };
    static std::pair<NNWeight::Transform, string> _sTransformPair[];
    static std::map<NNWeight::Transform, string> _sTransformMap;

private:
    friend class NNNetwork;
    friend class NNLayer;
    friend NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, uint32_t batch);

    NNLayer&                        _inputLayer;                // Source of activations
    NNLayer&                        _outputLayer;               // Output destination/Delta sources
    const bool                      _bShared;                   // Are we using another set of shard weights
    const bool                      _bTransposed;               // Use tranpose of shared weights?
    Transform                       _transform;                 // Transform type
    bool                            _bLocked;                   // Prevent updates to this set of weights
    NNWeight*                       _pSharedWeight;             // Pointer to shared weight
    uint32_t                        _sharingCount;              // Number of layers utilizing weight matrix
    uint32_t                        _updateCount;               // Indicates number of times a shared weight matrix has been updated
    uint32_t                        _dimensionality;            // Dimensionality for convolution (3 to 5)
    uint64_t                        _width;                     // Number of output units in outgoing FC layer or width of convolution for Convolutional layer
    uint64_t                        _height;                    // Number of input units in incoming FC layer or height of convolution for 2D/3D Convolutional layer
    uint64_t                        _length;                    // Number of input neurons in 2D convolution or length of a 3D convolution
    uint64_t                        _depth;                     // Number of output neurons in a 2D convolution or number of input neurons for a 3D convolution
    uint64_t                        _breadth;                   // Number of output neurons in a 3D convolution
    uint32_t                        _widthStride;               // X stride for all convolutions
    uint32_t                        _heightStride;              // Y stride for 2D/3D convolutions
    uint32_t                        _lengthStride;              // Z stride for 3D convolution
    uint64_t                        _size;                      // Total size of weights
    uint64_t                        _biasSize;                  // Total size of biases
    uint64_t                        _localSize;                 // Local size of weights
    uint64_t                        _localBiasSize;             // Local size of biases
    NNFloat                         _norm;                      // Maximum allowable weight vector length (default 0, unconstrained)
    cudnnTensorDescriptor_t         _convBiasTensor;            // Tensor describing weight biases
    cudnnFilterDescriptor_t         _convFilterDesc;            // CUDNN convolution filter (specifies kernel)
    cudnnConvolutionDescriptor_t    _convDesc;                  // CUDNN convolution stride and dimensions
    int                             _convPad[3];                // Padding for convolution    
    cudnnConvolutionFwdAlgo_t       _convFWAlgo;                // CUDNN convolution forward propagation algorithm
    cudnnConvolutionBwdFilterAlgo_t _convBWWeightAlgo;          // CUDNN convolution weight gradient backpropagation algorithm
    cudnnConvolutionBwdDataAlgo_t   _convBWDeltaAlgo;           // CUDNN convolution delta backpropagation algorithm
    vector<NNFloat>                 _vWeight;                   // CPU weight array
    vector<NNFloat>                 _vBias;                     // CPU bias array
    unique_ptr<GpuBuffer<NNFloat>> _pbWeight;                  // GPU weight array
    unique_ptr<GpuBuffer<NNFloat>> _pbBias;                    // GPU bias array
    unique_ptr<GpuBuffer<NNFloat>> _pbWeightGradient;          // Accumulated gradient per batch
    unique_ptr<GpuBuffer<NNFloat>> _pbBiasGradient;            // GPU bias gradient only used for cuDNN convolutional layers because stupid cuDNN
    unique_ptr<GpuBuffer<NNFloat>> _pbWeightVelocity;          // Velocity used for momentum and RMSProp
    unique_ptr<GpuBuffer<NNFloat>> _pbBiasVelocity;            // Velocity used for momentum and RMSProp
    unique_ptr<GpuBuffer<NNFloat>> _pbWeightGradientVelocity;  // Gradient velocity used for AdaDelta and Adam
    unique_ptr<GpuBuffer<NNFloat>> _pbBiasGradientVelocity;    // Gradient velocity used for AdaDelta and Adam
    NNWeight(NNLayer& inputLayer, NNLayer& outputLayer, bool bShared = false, bool bTransposed = false, bool bLocked = false, NNFloat maxNorm = 0.0f);
    ~NNWeight();
    void ClearSharedGradient();
    void ClearGradient();
    NNFloat CalculateRegularizationError(NNFloat lambda, NNFloat lambda1);
    void ClearVelocity();
    void Randomize();
    void Lock();
    void Unlock();
    void Dump(string fname, NNFloat* pBuffer);
    void RefreshState(NNNetwork* pNetwork, TrainingMode trainingMode);
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1, NNFloat t);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index, NNFloat* pWeight = NULL, NNFloat* pBias = NULL);
    NNFloat* GetWeightBuffer() { return _pbWeight ? _pbWeight->_pDevData : NULL; }
    NNFloat* GetWeightGradientBuffer() { return _pbWeightGradient ? _pbWeightGradient->_pDevData : NULL; }
    uint64_t GetBufferSize() { return _localSize; }
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
    uint64_t                _depth;
    uint64_t                _breadth;
    vector<NNFloat>         _vWeight;
    vector<NNFloat>         _vBias;
    bool                    _bShared;
    bool                    _bTransposed;
    bool                    _bLocked;
    NNFloat                 _norm;
    string                  _sourceInputLayer;     // _sourceInputLayer and _sourceOutputLayer collectively
    string                  _sourceOutputLayer;    // specify which weight matrix will be shared here

    NNWeightDescriptor();
};

bool LoadNNWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNWeightDescriptor& wd);
ostream& operator<< (ostream& out, NNWeightDescriptor& d);
uint32_t MPI_Bcast_NNWeightDescriptor(NNWeightDescriptor& d);
#endif
