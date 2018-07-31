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
    const uint32_t              _attributes;                // Attributes of the layer
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
    uint32_t                    _strideBN;                  // stride of Batch Norm Mean and Variance
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
    cudnnTensorDescriptor_t     _scaleBiasMeanVarDescBN;
    cudnnTensorDescriptor_t     _tensorDescriptorBN;        // Tensor descriptor for the BatchNormalization for this layer
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
    unique_ptr<GpuBuffer<NNFloat>> _pbDxBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbScaleDiffBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbBiasDiffBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbUnitBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbScaleBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbBiasBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbRunningMeanBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbRunningVarianceBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbSaveMeanBN;
    unique_ptr<GpuBuffer<NNFloat>> _pbSaveInvVarianceBN;
    uint32_t                    _bnCalls;
    NNFloat                     _bnLearningRate;
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
    void BackPropagate(uint32_t position, uint32_t batch);
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);
    void BackPropagatePooling(uint32_t position, uint32_t batch);
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
    cudnnTensorDescriptor_t getTensorDescriptorBN(uint32_t batch);

public:
    /**
     * Returns the name of this layer.
     */
    const string& GetName() const;

    /**
     * Returns the dataset name of this layer.
     * The dataset name of the layer is defined during
     * the creation of the network (with NNLayerDescriptor::_dataSet)
     * and is persisted into the network NetCDF file. It is used
     * to attribute a NNDataSet to NNLayer in NNNetwork::LoadDataSets
     */
    const string& GetDataSetName() const;

    NNLayer::Kind GetKind() const;

    /**
     * Returns the attributes of the layer. The attributes can be masked with NNLayer::Attributes
     * to check whether the layer has a particular type of attribute.
     */
    uint32_t GetAttributes() const;

    /**
     * Returns a pointer to the data of this layer. NNNetwork::LoadDataSets(vector<NNDataSetBase*>& vData)
     * associates datasets to the layer.
     */
    NNDataSetBase* GetDataSet() const;

    /**
     * Returns the number of dimensions (e.g. 1-d, 2-d, 3-d) of the layer.
     */
    uint32_t GetNumDimensions() const;

    /**
     * Returns the dimensions of the layer (e.g 128 x 1 x 1 for a 1-d, 128 wide layer).
     */
    tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetDimensions() const;

    /**
     * When running model parallel, returns the process-local dimensions of the layer.
     * The layer is sharded across processes, hence each process gets 1/n^th of the layer.
     * Currently only shards on dimX (see NNLayer::NNLayer()).
     */
    tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;
    tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;
    tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;
    bool GetUnits(vector<NNFloat>& vUnit);
    bool GetUnits(NNFloat* pUnit);
    bool SetUnits(const vector<NNFloat>& vUnit);
    bool GetDeltas(vector<NNFloat>& vUnit);
    bool GetDeltas(NNFloat *pUnit);
    bool SetDeltas(const vector<NNFloat>& vUnit);

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
    vector<NNFloat>         _vScaleBN;
    vector<NNFloat>         _vBiasBN;
    vector<NNFloat>         _vRunningMeanBN;
    vector<NNFloat>         _vRunningVarianceBN;
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
