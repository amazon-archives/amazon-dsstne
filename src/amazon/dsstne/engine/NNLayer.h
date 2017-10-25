/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNLAYER_H
#define NNLAYER_H
#ifndef __NVCC__
#include <memory>

class NNLayerDescriptor;
class NNLayer {
public:
    friend class NNNetwork;
    friend class NNWeight;
    friend NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, int batch);
    enum Kind 
    {
        Input,
        Hidden,
        Output,
        Target,
    };

    static std::pair<NNLayer::Kind, string> _sKindPair[];
    static std::map<NNLayer::Kind, string> _sKindMap;
    
    enum Type 
    {
        FullyConnected,
        Convolutional,
        Pooling
    };

    static std::pair<NNLayer::Type, string> _sTypePair[];
    static std::map<NNLayer::Type, string> _sTypeMap;

    enum Attributes 
    {
        None                = 0x0,
        Sparse              = 0x1,
        Denoising           = 0x2,
        BatchNormalization  = 0x4,
    };

    static std::pair<NNLayer::Attributes, string> _sAttributesPair[];
    static std::map<NNLayer::Attributes, string> _sAttributesMap;

    enum Parallelization {
        Data,
        Model,
        Serial,
    };

    static std::pair<NNLayer::Parallelization, string> _sParallelizationPair[];
    static std::map<NNLayer::Parallelization, string> _sParallelizationMap;


private:
    const string                _name;                      // Name of layer
    const Kind                  _kind;                      // Input, Hidden, Output, Pooling, or Target
    const Type                  _type;                      // FullyConnected or Convolutional
    PoolingFunction             _poolingFunction;           // Pooling function for pooling layers
    string                      _dataSet;                   // Name of data set for input and output layers
    NNDataSetBase*              _pDataSet;                  // Data set pointer for input and output layers
    vector<string>              _vSource;                   // Source layers/data sets
    vector<string>              _vSkip;                     // Skip layer sources    
    uint32_t                    _Nx;                        // Unit X size (or image or voxel width)
    uint32_t                    _Ny;                        // Image or voxel height (or 1)
    uint32_t                    _Nz;                        // Number of image neurons or voxel depth (or 1)
    uint32_t                    _Nw;                        // Number of voxel neurons (or 1)
    uint32_t                    _stride;                    // Total unit size
    uint32_t                    _localStride;               // Stride for local activation/delta
    uint32_t                    _maxLocalStride;            // Largest of all strides across all processes
    uint32_t                    _batch;                     // Mini-batch size
    uint32_t                    _localBatch;                // Data parallel batch size
    uint32_t                    _deltaUpdateCount;          // Counter to indicate how many delta updates have been performed during backpropagation
    uint32_t                    _unitUpdateCount;           // Counter to indicate how many unit updates have been performed during forward propagation
    uint32_t                    _dimensions;                // Input data dimensions
    uint32_t                    _minX;                      // Beginning of X units for model parallel execution
    uint32_t                    _maxX;                      // End of X units for model parallel execution    
    WeightInitialization        _weightInit;                // Weight initialization scheme
    NNFloat                     _weightInitScale;           // Weight Initialization scaling factor
    NNFloat                     _biasInit;                  // Bias initialization value
    NNFloat                     _RELUSlope;                 // Leaky RELU slope parameter
    NNFloat                     _ELUAlpha;                  // Alpha parameter for ELU and SELU activations
    NNFloat                     _SELULambda;                // Lambda parameter for SELU activations    
    bool                        _bBatchNormalization;       // Perform batch normalization
    const uint32_t              _kernelX;                   // kernel X size
    const uint32_t              _kernelY;                   // kernel Y size
    const uint32_t              _kernelZ;                   // kernel Z size
    const uint32_t              _kernelStrideX;             // kernel X stride
    const uint32_t              _kernelStrideY;             // kernel Y stride
    const uint32_t              _kernelStrideZ;             // kernel Z stride    
    const uint32_t              _kernelPaddingX;            // kernel X padding
    const uint32_t              _kernelPaddingY;            // kernel Y padding
    const uint32_t              _kernelPaddingZ;            // kernel Z padding        
    const uint32_t              _kernelDimensions;          // Number of components to kernel and kernel stride
    const Activation            _activation;                // Activation function
    const NNFloat               _pDropout;                  // Dropout probability
    const bool                  _bSparse;                   // Is layer sparse
    bool                        _bFastSparse;               // Is layer capable of using fast sparse kernels
    NNFloat                     _sparsenessPenalty_p;       // Layer-specific sparseness target
    NNFloat                     _sparsenessPenalty_beta;    // Layer-specific sparseness penalty weight
    const bool                  _bDenoising;                // Is layer utilizing Denoising?       
    NNFloat                     _weightNorm;                // Maximum weight vector length
    NNFloat                     _deltaNorm;                 // Maximum delta vector length                         
    Parallelization             _parallelization;           // Should layer be Data or Model parallelized?
    bool                        _bTransposeParallelization; // Should we tranpose parallelization?
    bool                        _bDirty;                    // Indicates layer state needs to be update
    cudnnTensorDescriptor_t     _tensorDescriptor;          // Tensor descriptor for this layer
    cudnnTensorDescriptor_t     _oddBatchTensorDescriptor;  // Tensor descriptor for end of epoch batches or weird inference calls because cuDNN
    uint32_t                    _oddBatch;                  // Batch size of odd batch tensor descriptor 
    cudnnPoolingDescriptor_t    _poolingDescriptor;         // Pooling descriptor for most pooling layers
    cudnnLRNDescriptor_t        _LRNDescriptor;             // Local response normalization descriptor for LRN layers 
    vector<NNLayer*>            _vIncomingLayer;            // source layer(s)
    vector<NNWeight*>           _vIncomingWeight;           // Incoming weights
    vector<NNLayer*>            _vOutgoingLayer;            // Destination layers
    vector<NNWeight*>           _vOutgoingWeight;           // Outoing weights
    vector<NNLayer*>            _vIncomingLargerLayer;      // source layer(s)
    vector<NNWeight*>           _vIncomingLargerWeight;     // Incoming weights
    vector<NNLayer*>            _vOutgoingLargerLayer;      // Destination larger layers
    vector<NNWeight*>           _vOutgoingLargerWeight;     // Destination larger weights
    vector<NNLayer*>            _vIncomingSkip;             // List of incoming skip layers
    vector<NNLayer*>            _vOutgoingSkip;             // List of outgoing skip layer targets
    vector<NNFloat>             _vUnit;                     // Layer's units
    vector<NNFloat>             _vDelta;                    // Layer's deltas
    vector<NNFloat>             _vBuffer1;                  // Scratch buffer for various layer types
    vector<NNFloat>             _vBuffer2;                  // Scratch buffer for various layer types
    unique_ptr<GpuBuffer<NNFloat>> _pbUnit;                 // GPU memory for unit activations
    unique_ptr<GpuBuffer<NNFloat>> _pbDelta;                // GPU memory for unit deltas
    unique_ptr<GpuBuffer<NNFloat>> _pbDropout;              // Dropout random values if active
    unique_ptr<GpuBuffer<NNFloat>> _pbBuffer1;              // Scratch buffer 1 if active
    unique_ptr<GpuBuffer<NNFloat>> _pbBuffer2;              // Scratch buffer 2 if active   
    int32_t                     _priority;                  // Mutable priority for calculating propagation ordering
    NNLayer(NNLayerDescriptor& l, uint32_t batch);
    ~NNLayer();
    void Allocate(bool validate);
    void Deallocate();
    void SetBatch(uint32_t batch);
    void RefreshParallelization();
    void RefreshState(NNNetwork* pNetwork, bool validate);
    void LoadPredictionBatch(uint32_t position, uint32_t batch);
    void LoadTrainingBatch(uint32_t position, uint32_t batch);
    void LoadValidationBatch(uint32_t position, uint32_t batch);
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);
    void ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining);    
    void ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining);
    void ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining);
    void CalculateActivation(uint32_t batch);
    void CalculateDropout(uint32_t batch);
    NNFloat CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);
    void BackPropagate(uint32_t position, uint32_t batch, NNFloat alpha);
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch, NNFloat alpha);    
    void BackPropagateConvolutional(uint32_t position, uint32_t batch, NNFloat alpha);
    void BackPropagatePooling(uint32_t position, uint32_t batch, NNFloat alpha);        
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);
    void GenerateDenoisingData();
    void Reduce(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride, uint32_t updateCount);
    void Gather(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride);
    void ClearUpdates();
    void Dump(string fname, NNFloat* pData);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);
    NNFloat* GetUnitBuffer() { return _pbUnit ? _pbUnit->_pDevData : NULL; }
    NNFloat* GetDeltaBuffer() { return _pbDelta ? _pbDelta->_pDevData : NULL; }
    uint64_t GetBufferSize() { return _batch * _stride; }
    cudnnTensorDescriptor_t getTensorDescriptor(uint32_t batch);

public:
    tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetDimensions() const;
    tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;
    tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;
    tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;
    //NNFloat GetPDropout();
    //NNFloat GetWeightNorm();
    //NNFloat GetDeltaNorm();
    //bool SetWeightNorm(NNFloat norm);
    //bool SetDeltaNorm(NNFloat norm);
};


ostream& operator<< (ostream& out, NNLayer::Kind& k);
ostream& operator<< (ostream& out, NNLayer::Type& t);
ostream& operator<< (ostream& out, NNLayer::Parallelization& p);
ostream& operator<< (ostream& out, NNLayer::Attributes& a);

struct NNLayerDescriptor
{
    string                  _name;                      // Name of layer
    NNLayer::Kind           _kind;                      // Input, Hidden, Pooling, or Output
    NNLayer::Type           _type;                      // FullyConnected, Convolutional, or Pooling
    PoolingFunction         _poolingFunction;           // Pooling function for pooling layers
    string                  _dataSet;                   // Name of dataset for input and output layers
    vector<string>          _vSource;                   // Source layers/data sets
    vector<string>          _vSkip;                     // Skip layer sources
    uint32_t                _Nx;                        // Unit X size (or image or voxel width)
    uint32_t                _Ny;                        // Image or voxel height (or 1)
    uint32_t                _Nz;                        // Number of image neurons or voxel depth (or 1)
    uint32_t                _Nw;                        // Number of voxel neurons (or 1)
    uint32_t                _dimensions;                // Convolution unit or input data dimensions
    bool                    _bDimensionsProvided;       // Have all dimensions been determined?
    WeightInitialization    _weightInit;                // Weight initialization scheme
    NNFloat                 _weightInitScale;           // Weight Initialization scaling factor
    NNFloat                 _biasInit;                  // Bias initialization value
    uint32_t                _kernelX;                   // kernel X size
    uint32_t                _kernelY;                   // kernel Y size
    uint32_t                _kernelZ;                   // kernel Z size
    uint32_t                _kernelStrideX;             // kernel X stride
    uint32_t                _kernelStrideY;             // kernel Y stride
    uint32_t                _kernelStrideZ;             // kernel Z stride
    uint32_t                _kernelPaddingX;            // kernel X padding
    uint32_t                _kernelPaddingY;            // kernel Y padding
    uint32_t                _kernelPaddingZ;            // kernel Z padding     
    uint32_t                _kernelDimensions;          // Number of components to kernel and kernel stride
    NNFloat                 _weightNorm;                // Maximum weight vector length
    NNFloat                 _deltaNorm;                 // Maximum delta vector length
    NNFloat                 _pDropout;                  // Dropout probability
    Activation              _activation;                // Activation function
    NNFloat                 _sparsenessPenalty_p;       // Layer-specific sparseness target
    NNFloat                 _sparsenessPenalty_beta;    // Layer-specific sparseness penalty weight    
    uint32_t                _attributes;                // Specific layer properties
    NNFloat                 _RELUSlope;                 // Leaky RELU slope parameter
    NNFloat                 _ELUAlpha;                  // Alpha parameter for ELU and SELU activations
    NNFloat                 _SELULambda;                // Lambda parameter for SELU activations
    NNLayerDescriptor();
};

bool LoadNNLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNLayerDescriptor& ld);
ostream& operator<< (ostream& out, NNLayerDescriptor& d);
uint32_t MPI_Bcast_NNLayerDescriptor(NNLayerDescriptor& d);
#endif
#endif
