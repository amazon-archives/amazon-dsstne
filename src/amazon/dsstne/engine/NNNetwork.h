/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNNETWORK_H
#ifndef __NVCC__
struct NNNetworkDescriptor;


class NNNetwork {
public:
    friend class NNLayer;
    friend class NNWeight;
    friend void GpuContext::SetNeuralNetwork(NNNetwork* pNetwork);
    enum Kind {
        FeedForward,
        AutoEncoder,
    };

    static std::pair<NNNetwork::Kind, string> _sKindPair[];
    static std::map<NNNetwork::Kind, string> _sKindMap;

    enum Mode {
        Prediction = 0,
        Training = 1,
        Validation = 2,
        Unspecified = 3
    };
    
private:
    friend NNNetwork* LoadNeuralNetworkJSON(const string& fname, const uint32_t batch, const vector<NNDataSetBase*>& vDataSet);
    friend NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, const uint32_t batch);
    friend NNNetwork* ImportAutoEncoder(const string& fname, uint32_t batch);
    string                      _name;                      // ASCII name for network
    uint32_t                    _batch;                     // Overall batch size
    uint32_t                    _position;                  // Current position
    bool                        _bExamplesFound;            // Has examples count been found yet?
    bool                        _bAllDataLoaded;            // True if either all or no datasets loaded (training(all) versus prediction(none))
    uint32_t                    _examples;                  // Total examples in current dataset(s)
    const Kind                  _kind;
    ErrorFunction               _errorFunction;             // Error function for output layer(s)
    TrainingMode                _trainingMode;              // Specified training mode
    Mode                        _mode;                      // Operational mode (training or prediction)
    uint32_t                    _epochs;                    // Total number of training epochs
    uint32_t                    _indices;                   // Total number of indices in all input and output data

    // Local response normalization settings
    NNFloat                     _LRN_k;                     // LRN offset
    uint32_t                    _LRN_n;                     // LRN spread
    NNFloat                     _LRN_alpha;                 // LRN scaling
    NNFloat                     _LRN_beta;                  // LRN exponent

    // Maxout parameters
    uint32_t                    _maxout_k;                  // Maxout neighborhood

    // Sparseness penalty parameters for sparse hidden layers
    bool                        _bSparsenessPenalty;        // Specifies whether to use autoencoder sparseness penalty or not
    NNFloat                     _sparsenessPenalty_p;       // Target sparseness probability for autoencoder
    NNFloat                     _sparsenessPenalty_beta;    // Target sparse penalty weight

    // Denoising parameters
    bool                        _bDenoising;                // Specifies whether to use denoising on autoencoders or not
    NNFloat                     _denoising_p;               // Probability of denoising autoencoder inputs (for sparse layers, only denoise on non-zero values) 

    // Delta Boost parameters
    NNFloat                     _deltaBoost_one;            // Adjusts scaling of nonzero-valued outputs
    NNFloat                     _deltaBoost_zero;           // Adjusts scaling of zero-valued outputs

    // Scaled Marginal Cross Entropy parameters
    NNFloat                     _SMCE_oneTarget;            // Relaxed target for non-zero target values (Default 0.9)
    NNFloat                     _SMCE_zeroTarget;           // Relaxed target for zero target values (Default 0.1)
    NNFloat                     _SMCE_oneScale;             // Scaling factor for non-zero target values (Default 1.0)
    NNFloat                     _SMCE_zeroScale;            // Scaling factor for zero target values (Default 1.0)

    // Index shuffling parameters
    bool                        _bShuffleIndices;           // Flag to shuffle training data (or not)
    uint32_t                    _shuffleIndices;            // Number of indices assigned
    uint32_t*                   _pShuffleIndex;             // Shuffle index
    GpuBuffer<uint32_t>*        _pbShuffleIndex;            // Shuffle buffer for processes other than 0  
    GpuSort<uint32_t, uint32_t>* _pShuffleIndexSort;        // Pointer to Device/GPU sort

    // Checkpoint information
    string                      _checkpoint_name;           // Name of checkpoint file
    int32_t                     _checkpoint_interval;       // Number of epochs between training checkpoints
    int32_t                     _checkpoint_epochs;         // Number of epochs since last checkpoint written

    // Network data
    vector<NNLayer*>            _vLayer;                    // List of all layers in network
    vector<NNLayer*>            _vInputLayer;               // List of all input layers for loading minibatches of data
    vector<NNLayer*>            _vOutputLayer;              // List of all output layers for calculating error
    vector<NNWeight*>           _vWeight;                   // List of weight arrays between layers
    vector<NNWeight*>           _vSharedWeight;             // List of weight arrays that are shared between 2 or more layers
    vector<NNDataSetBase*>      _vData;                     // List of all data sets loaded into neural network
    vector<NNLayer*>            _vFPOrder;                  // Layers re-ordered for consistent forward propagation
    vector<NNLayer*>            _vBPOrder;                  // Layers re-ordered for consistent backpropagation
    map<string, NNLayer*>       _mLayer;                    // Maps layer names to layers
    bool                        _bDirty;                    // Flag signalling network has been changed
    bool                        _bClearVelocity;            // Clear training velocity with each training call?

    // Work buffer for merging multiGPU computations (weight and delta normalization)
    size_t                      _scratchBufferSize;         // Current scratch buffer size
    GpuBuffer<NNFloat>*         _pbScratchBuffer;           // Pointer to scratch buffer


    // P2P model-parallelization buffers
    uint32_t                    _maxStride;                 // Maximum stride of all scattered/gathered network layers
    uint32_t                    _sendIndex;                 // Current send buffer index
    uint32_t                    _receiveIndex;              // Current receiver buffer index
    GpuBuffer<NNFloat>*         _pbP2PBuffer[2];            // Peer buffer for sending/calculating peer data
    NNFloat*                    _pPeerBuffer[2];            // Peer for receiving/reducing peer data
    NNFloat*                    _pCPUBuffer;                // System memory work buffer for MPI copies


public:

    ~NNNetwork();
    void ClearDataSets();
    void LoadDataSets(vector<NNDataSetBase*>& vData);
    void Randomize();
    bool Validate();
    float Train(uint32_t epochs = 1, NNFloat alpha = 0.1f, NNFloat lambda = 0.001f, NNFloat mu = 0.1f);
    void PredictBatch(uint32_t layers = 0);
    void CalculateTopK(const string& layer, uint32_t k, GpuBuffer<NNFloat>* pbKey, GpuBuffer<uint32_t>* pbValue);
    void SaveBatch(string fname);
    void DumpBatch(FILE* fp);
    void SaveLayer(const string& fname, const string& layer);
    void DumpLayer(FILE* fp, const string& layer);
    void SaveWeights(const string& fname, const string& inputLayer, const string& outputLayer);
    bool LockWeights(const string& inputLayer, const string& outputLayer);
    bool UnlockWeights(const string& inputLayer, const string& outputLayer); 
    uint32_t GetExamples();
    void SetBatch(uint32_t batch);
    unsigned int GetBatch();
    void SetPosition(uint32_t position);
    uint32_t GetPosition() { return _position; }
    void SetTrainingMode(TrainingMode mode);
    void SetShuffleIndices(bool bShuffleIndices);
    void SetCPUValidate(bool bValidate);
    void SetClearVelocity(bool bClear) { _bClearVelocity = bClear; };
    bool SaveNetCDF(const string& fname);

    // Getters
    NNFloat* GetUnitBuffer(const string& layer);
    NNFloat* GetDeltaBuffer(const string& layer);
    NNFloat* GetWeightBuffer(const string& inputLayer, const string& outputLayer);
    NNWeight* GetWeight(const string& inputLayer, const string& outputLayer);
    uint64_t GetBufferSize(const string& layer);
    NNLayer* GetLayer(const string &layer);
    vector<string> GetLayers();
    string GetName();
    tuple<NNFloat, uint32_t, NNFloat, NNFloat> GetLRN();                                // Returns k, n, alpha, beta
    tuple<uint32_t> GetMaxout();                                                        // Returns k
    tuple<NNFloat, NNFloat> GetSparsenessPenalty();                                     // Returns p, beta
    tuple<NNFloat> GetDenoising();                                                      // Returns p
    tuple<NNFloat, NNFloat> GetDeltaBoost();                                            // Returns one, zero,
    tuple<NNFloat, NNFloat, NNFloat, NNFloat> GetSMCE();                                // Returns oneTarget, zeroTarget, oneScale, zeroScale
    tuple<bool> GetShuffleIndices();                                                    // Returns ShuffleIndices boolean
    tuple<string, int32_t> GetCheckPoint();                                             // Returns Checkpoint name and interval
    NNFloat* GetScratchBuffer(size_t size = 0);                                         // Gets current scratch buffer, resizing if too small
    NNFloat* GetP2PSendBuffer();                                                        // Returns current local send buffer
    NNFloat* GetP2PReceiveBuffer();                                                     // Returns current local receive buffer
    NNFloat* GetP2PCPUBuffer();                                                         // Returns system memory work buffer    
    NNFloat* GetPeerBuffer();                                                           // Returns current adjacent peer receive buffer
    NNFloat* GetPeerBackBuffer();                                                       // Returns current adjacent peer send buffer
    bool P2P_Bcast(void* pBuffer, size_t size);                                         // Broadcasts data from process 0 to all other
    bool P2P_Allreduce(NNFloat* pBuffer, size_t size);                                  // Reduces buffer across all processes


    // Setters
    bool SetLRN(uint32_t k = 2, uint32_t n = 5, NNFloat alpha = (NNFloat)0.0001, NNFloat beta = (NNFloat)0.75);
    bool SetMaxout(uint32_t k = 2);
    bool SetSparsenessPenalty(NNFloat p = 0.0f, NNFloat beta = 0.0f);
    bool SetDenoising(NNFloat p = 0.0f);
    bool SetDeltaBoost(NNFloat one = 1.0f, NNFloat zero = 1.0f);
    bool SetSMCE(NNFloat oneTarget = 0.9f, NNFloat zeroTarget = 0.1f, NNFloat oneScale = 1.0f, NNFloat zeroScale = 1.0f);
    bool SetCheckpoint(string name, int32_t interval);

private:
    void CalculatePropagationOrder();
    void AllocatePeerBuffers();
    void DeallocatePeerBuffers();
    void SwapPeerBuffers();
    void LoadBatch();
    void PredictTrainingBatch(uint32_t layers = 0);
    void PredictValidationBatch(uint32_t layers = 0);
    void RefreshShuffleBuffers();
    void ShuffleIndices();
    tuple<NNFloat, NNFloat> CalculateError(NNFloat lambda);
    void ClearUpdates();
    void BackPropagate(NNFloat alpha);
    void UpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat mu);
    NNNetwork(NNNetworkDescriptor& nd, uint32_t batch = DefaultBatch);
    void RefreshState();
    void Shuffle();
};

ostream& operator<< (ostream& out, NNNetwork::Kind& k);


struct NNNetworkDescriptor
{
    string                      _name;                      // Optional name for neural network
    NNNetwork::Kind             _kind;                      // Either AutoEncoder or FeedForward (default)
    ErrorFunction               _errorFunction;             // Error function for training
    vector<NNLayerDescriptor>   _vLayerDescriptor;          // Vector containing neural network layers
    vector<NNWeightDescriptor>  _vWeightDescriptor;         // Vector containing preloaded weight data
    bool                        _bShuffleIndices;           // Flag to signal whether to shuffle training data or not
    uint32_t                    _maxout_k;                  // Size of Maxout (default 2)
    NNFloat                     _LRN_k;                     // Local Response Normalization offset (default 2)
    uint32_t                    _LRN_n;                     // Local Response Normalization spread (default 5)
    NNFloat                     _LRN_alpha;                 // Local Response Normalization scaling (default 0.0001)
    NNFloat                     _LRN_beta;                  // Local Response Normalization exponent (default 0.75)
    bool                        _bSparsenessPenalty;        // Specifies whether to use sparseness penalty on hidden layers or not
    NNFloat                     _sparsenessPenalty_p;       // Target sparseness probability for hidden layers
    NNFloat                     _sparsenessPenalty_beta;    // Sparseness penalty weight 
    bool                        _bDenoising;                // Specifies whether to use denoising on input layers
    NNFloat                     _denoising_p;               // Probability of denoising inputs (for sparse layers, only denoise on non-zero values) 
    NNFloat                     _deltaBoost_one;            // Adjusts scaling of nonzero-valued outputs
    NNFloat                     _deltaBoost_zero;           // Adjusts scaling of zero-valued outputs
    NNFloat                     _SMCE_oneTarget;            // Relaxed target for non-zero target values (Default 0.9)
    NNFloat                     _SMCE_zeroTarget;           // Relaxed target for zero target values (Default 0.1)
    NNFloat                     _SMCE_oneScale;             // Scaling factor for non-zero target values (Default 1.0)
    NNFloat                     _SMCE_zeroScale;            // Scaling factor for zero target values (Default 1.0)
    string                      _checkpoint_name;           // Checkpoint file name
    int32_t                     _checkpoint_interval;       // Number of epochs between checkpoints
    int32_t                     _checkpoint_epochs;         // Number of epochs since last checkpoint
    NNNetworkDescriptor();
};

ostream& operator<< (ostream& out, NNNetworkDescriptor& d);
NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, const uint32_t batch = DefaultBatch);
NNNetwork* LoadNeuralNetworkJSON(const string &fname, const uint32_t batch = DefaultBatch, const vector<NNDataSetBase*>& vDataSet = vector<NNDataSetBase*>());
bool SaveNeuralNetworkJSON(const NNNetwork& net, const string& fname);
bool SaveNeuralNetworkNetCDF(const NNNetwork& net, const string& jname);
NNNetwork* ImportAutoEncoder(const string& fname, uint32_t batch = DefaultBatch);
#endif // __NVCC__
#define NNNETWORK_H
#endif
