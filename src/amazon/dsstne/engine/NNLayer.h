/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNLAYER_H
#ifndef __NVCC__
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
        Pooling,
        Target,
    };

    static std::pair<NNLayer::Kind, string> _sKindPair[];
    static std::map<NNLayer::Kind, string> _sKindMap;
    
    enum Type 
    {
        FullyConnected,
        Convolutional
    };

    static std::pair<NNLayer::Type, string> _sTypePair[];
    static std::map<NNLayer::Type, string> _sTypeMap;

    enum Attributes 
    {
        None                = 0x0,
        Sparse              = 0x1,
        Denoising           = 0x2,
    };

    static std::pair<NNLayer::Attributes, string> _sAttributesPair[];
    static std::map<NNLayer::Attributes, string> _sAttributesMap;

    enum Parallelization {
        Data,
        Model,
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
    const uint32_t              _Nx;                        // Unit X size
    const uint32_t              _Ny;                        // Unit Y size
    const uint32_t              _Nz;                        // Unit Z size
    uint32_t                    _stride;                    // Total unit size
    uint32_t                    _localStride;               // Stride for local activation/delta
    uint32_t                    _maxLocalStride;            // Largest of all strides across all processes    
    uint32_t                    _batch;                     // Mini-batch size
    uint32_t                    _localBatch;                // Data parallel batch size
    uint32_t                    _deltaUpdateCount;          // Counter to indicate how many delta updates have been performed during backpropagation
    uint32_t                    _unitUpdateCount;           // Counter to indicate how many unit updates have been performed during forward propagation
    const uint32_t              _dimensions;                // Input data dimensions
    uint32_t                    _minX;                      // Beginning of X units for model parallel execution
    uint32_t                    _maxX;                      // End of X units for model parallel execution    
    WeightInitialization        _weightInit;                // Weight initialization scheme
    NNFloat                     _weightInitScale;           // Weight Initialization scaling factor
    NNFloat                     _biasInit;                  // Bias initialization value
    const uint32_t              _kernelX;                   // kernel X size
    const uint32_t              _kernelY;                   // kernel Y size
    const uint32_t              _kernelZ;                   // kernel Z size
    const uint32_t              _kernelStrideX;             // kernel X stride
    const uint32_t              _kernelStrideY;             // kernel Y stride
    const uint32_t              _kernelStrideZ;             // kernel Z stride
    const Activation            _activation;                // Activation function
    const NNFloat               _pDropout;                  // Dropout probability
    const bool                  _bSparse;                   // Is layer sparse
    bool                        _bFastSparse;               // Is layer capable of using fast sparse kernels
    const bool                  _bDenoising;                // Is layer utilizing Denoising?       
    NNFloat                     _weightNorm;                // Maximum weight vector length
    NNFloat                     _deltaNorm;                 // Maximum delta vector length                         
    Parallelization             _parallelization;           // Should layer be Data or Model parallelized?
    bool                        _bDirty;                    // Indicates layer state needs to be update
    vector<NNLayer*>            _vIncomingLayer;            // source layer(s)
    vector<NNWeight*>           _vIncomingWeight;           // Incoming weights
    vector<NNLayer*>            _vOutgoingLayer;            // Destination layers
    vector<NNWeight*>           _vOutgoingWeight;           // Outoing weights
    vector<NNLayer*>            _vIncomingLargerLayer;      // source layer(s)
    vector<NNWeight*>           _vIncomingLargerWeight;     // Incoming weights
    vector<NNLayer*>            _vOutgoingLargerLayer;      // Destination larger layers
    vector<NNWeight*>           _vOutgoingLargerWeight;     // Destination larger weights
    vector<NNFloat>             _vUnit;                     // Layer's units
    vector<NNFloat>             _vDelta;                    // Layer's deltas
    GpuBuffer<NNFloat>*         _pbUnit;                    // GPU memory for unit activations
    GpuBuffer<NNFloat>*         _pbDelta;                   // GPU memory for unit deltas  
    GpuBuffer<NNFloat>*         _pbDropout;                 // Dropout random values if active
    int32_t                     _priority;                  // Mutable priority for calculating propagation ordering
    NNLayer(NNLayerDescriptor& l, uint32_t batch);
    ~NNLayer();
    void Allocate(bool validate);
    void Deallocate();
    void SetBatch(uint32_t batch);
    void RefreshState(bool validate);
    void LoadPredictionBatch(uint32_t position, uint32_t batch);
    void LoadTrainingBatch(uint32_t position, uint32_t batch);
    void LoadValidationBatch(uint32_t position, uint32_t batch);
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);
    void CalculateActivation(uint32_t batch);
    void CalculateDropout(uint32_t batch);
    NNFloat CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);
    void BackPropagate(uint32_t position, uint32_t batch, NNFloat alpha);
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

public:
    tuple<uint32_t, uint32_t, uint32_t> GetDimensions();
    tuple<uint32_t, uint32_t, uint32_t> GetLocalDimensions();
    tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions();
    tuple<uint32_t, uint32_t, uint32_t> GetKernelStride();
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
    NNLayer::Kind           _kind;                      // Input, Hidden, Pooling, Target, or Output
    NNLayer::Type           _type;                      // FullyConnected or Convolutional
    PoolingFunction         _poolingFunction;           // Pooling function for pooling layers
    string                  _dataSet;                   // Name of dataset for input and output layers
    vector<string>          _vSource;                   // Source layers/data sets
    uint32_t                _Nx;                        // Convolution unit or input data X size
    uint32_t                _Ny;                        // Convolution unit or input data Y size
    uint32_t                _Nz;                        // Convolution unit or input data Z size
    uint32_t                _dimensions;                // Convolution unit or input data dimensions
    WeightInitialization    _weightInit;                // Weight initialization scheme
    NNFloat                 _weightInitScale;           // Weight Initialization scaling factor
    NNFloat                 _biasInit;                  // Bias initialization value
    uint32_t                _kernelX;                   // kernel X size
    uint32_t                _kernelY;                   // kernel Y size
    uint32_t                _kernelZ;                   // kernel Z size
    uint32_t                _kernelStrideX;             // kernel X stride
    uint32_t                _kernelStrideY;             // kernel Y stride
    uint32_t                _kernelStrideZ;             // kernel Z stride
    NNFloat                 _weightNorm;                // Maximum weight vector length
    NNFloat                 _deltaNorm;                 // Maximum delta vector length
    NNFloat                 _pDropout;                  // Dropout probability
    Activation              _activation;                // Activation function
    uint32_t                _attributes;                // Specific layer properties
    NNLayerDescriptor();
};

bool LoadNNLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNLayerDescriptor& ld);
ostream& operator<< (ostream& out, NNLayerDescriptor& d);
uint32_t MPI_Bcast_NNLayerDescriptor(NNLayerDescriptor& d);
#endif
#define NNLAYER_H
#endif
