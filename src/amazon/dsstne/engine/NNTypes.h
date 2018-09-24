/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNTYPES_H
#define NNTYPES_H
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#ifndef __NVCC__
#include <tuple>
#include <json/json.h>
#endif
#include <cmath>
#include <memory>

class NNDataSetBase;
class NNLayer;
class NNNetwork;
class NNWeight;

// Activates step by step CPU validation
#define VALIDATION
#ifdef VALIDATION
extern "C"
{
    #include <cblas.h>
}
#endif


static const float NN_VERSION       = 0.9f;
static const float MIN_ERROR        = 1.0e-12f;
static const float MIN_ACTIVATION   = 0.000001f;
static const float MAX_ACTIVATION   = 0.999999f;
static const float MAX_VALUE        = 999999999999999.0f;

template <typename T> struct GpuBuffer;

enum 
{
    DefaultBatch    = 512
};

enum Mode {
    Prediction = 0,
    Training = 1,
    Validation = 2,
    Unspecified = 3
};

enum TrainingMode 
{
    SGD = 0,
    Momentum = 1,
    AdaGrad = 2,
    Nesterov = 3,
    RMSProp = 4,
    AdaDelta = 5,
    Adam = 6,
};

ostream& operator<< (ostream& out, const TrainingMode& e);

enum ErrorFunction 
{
    L1,
    L2,
    CrossEntropy,
    ScaledMarginalCrossEntropy,
    DataScaledMarginalCrossEntropy,
    Hinge,
    L2Hinge,
};

ostream& operator<< (ostream& out, const ErrorFunction& e);

enum Activation {
    Sigmoid,
    Tanh,
    RectifiedLinear,
    Linear,
    ParametricRectifiedLinear,
    SoftPlus,
    SoftSign,
    SoftMax,
    RELUMax,
    LinearMax,
    ExponentialLinear,
    LeakyRectifiedLinear,
    ScaledExponentialLinear,
};

ostream& operator<< (ostream& out, const Activation& a);

enum WeightInitialization
{
    Xavier,
    CaffeXavier,
    Gaussian,
    Uniform,
    UnitBall,
    Constant,
    SELU, 
};
    
ostream& operator<< (ostream& out, const WeightInitialization& w);
    
enum PoolingFunction {
    None,
    Max,
    Average,
    LRN,
    Maxout,
    DotProduct,
    Cosine,
    Stochastic,
    LCN,
    GlobalTemporal,
};

ostream& operator<< (ostream& out, const PoolingFunction& p);

#include "kernels.h"
#include "GpuSort.h"
#include "NNEnum.h"
#include "NNWeight.h"
#include "NNLayer.h"
#include "NNNetwork.h"


int MPI_Bcast_string(string& s);

struct NNDataSetDimensions
{
    uint32_t _dimensions;
    uint32_t _width;
    uint32_t _height;
    uint32_t _length;
};

struct NNDataSetBase {

    string                          _name;                          // Dataset name
    NNDataSetEnums::DataType        _dataType;                      // Dataset type (see above enum)
    uint32_t                        _attributes;                    // Dataset characteristics (see NNDataSetEnum::Attributes in NNEnum.h)
    uint32_t                        _examples;                      // Number of examples
    uint32_t                        _uniqueExamples;                // Number of unique examples for indexed data (set to examples if unindexed)
    uint32_t                        _localExamples;                 // Number of local examples when data sharded
    uint32_t                        _dimensions;                    // Dimensionality of data set
    uint32_t                        _width;                         // Dataset x dimension
    uint32_t                        _height;                        // Dataset y dimension
    uint32_t                        _length;                        // Dataset z dimension
    uint32_t                        _stride;                        // Stride between examples
    NNDataSetEnums::Sharding        _sharding;                      // Sharding of dataset for parallel execution
    uint32_t                        _minX;                          // Beginning of local X sharding for model parallel execution 
    uint32_t                        _maxX;                          // End of local X sharding for model parallel execution
    uint64_t                        _sparseDataSize;                // Total sparse datapoints
    NNFloat                         _sparseDensity;                 // Overall sparse density (0.0 - 1.0)
    vector<uint64_t>                _vSparseStart;                  // Vector of sparse datapoint starts per example
    unique_ptr<GpuBuffer<uint64_t>> _pbSparseStart;                 // GPU copy of _vSparseStart
    vector<uint64_t>                _vSparseEnd;                    // Vector of sparse datapoint ends per example
    unique_ptr<GpuBuffer<uint64_t>> _pbSparseEnd;                   // GPU copy of _vSparseEnd
    vector<uint32_t>                _vSparseIndex;                  // Vector of sparse indices
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseIndex;                 // GPU copy of _vSparseIndex
    vector<NNFloat>                 _vSparseWeight;                 // Per example sparse index weights
    unique_ptr<GpuBuffer<NNFloat>>  _pbSparseWeight;                // GPU copy of _vSparseWeight
    vector<uint32_t>                _vIndex;                        // Indexed data array
    unique_ptr<GpuBuffer<uint32_t>> _pbIndex;                       // GPU copy of _vIndex;
    unique_ptr<GpuBuffer<NNFloat>>  _pbDenoisingRandom;             // Denoising randoms
    
    // Transposed sparse lookup for sparse backpropagation
    vector<uint64_t>                _vSparseDatapointCount;
    vector<uint32_t>                _vSparseMaxDatapointCount;
    vector<uint32_t>                _vSparseMultiDatapointCount;
    vector<uint32_t>                _vSparseTransposedStart;
    uint64_t                        _sparseTransposedIndices;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedStart;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedEnd;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedIndex;
    unique_ptr<GpuBuffer<NNFloat>>  _pbSparseTransposedData;    

    // States
    bool                            _bDenoising;
    bool                            _bDirty;
    bool                            _bStreaming;
    bool                            _bIndexed;
    uint32_t                        _batch;

    NNDataSetBase();
    NNDataSetDimensions GetDimensions();
    uint32_t GetExamples() { return _examples; };
    uint32_t GetUniqueExamples() { return _uniqueExamples; };

    virtual bool SaveNetCDF(const string& fname) = 0;
    virtual bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n) = 0;
    virtual ~NNDataSetBase() = 0;
    virtual void RefreshState(uint32_t batch) = 0;
    virtual bool Shard(NNDataSetEnums::Sharding sharding) = 0;
    virtual bool UnShard() = 0;
    virtual bool SetStreaming(bool flag) = 0;
    virtual bool GetStreaming() = 0;
    virtual vector<tuple<uint64_t, uint64_t> > getMemoryUsage() = 0;
    virtual bool CalculateSparseDatapointCounts() = 0;
    virtual bool GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient) = 0;
    virtual bool SetDenoising(bool flag) = 0;
    virtual bool GenerateDenoisingData() = 0;
    virtual bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;    
    virtual float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda) = 0;
    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda) = 0;
    virtual bool CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda) = 0;
    virtual bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;

 protected:
    NNDataSetBase(NNDataSetEnums::DataType dataType, uint32_t examples, uint32_t uniqueExamples, const NNLayer &layer);
};

ostream& operator<< (ostream& out, NNDataSetEnums::Attributes& a);
ostream& operator<< (ostream& out, NNDataSetEnums::Kind& k);
ostream& operator<< (ostream& out, NNDataSetEnums::DataType& t);
ostream& operator<< (ostream& out, NNDataSetEnums::Sharding& s);

template<typename T> class NNDataSet : public NNDataSetBase {
public:
    friend class NNetwork;
    friend class NNLayer;
    friend vector<NNDataSetBase*> LoadNetCDF(const string& fname);
    friend bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataSet);

private:
    // Type-specific data
    vector<T>                   _vData;
    unique_ptr<GpuBuffer<T>>    _pbData;
    vector<T>                   _vSparseData;
    unique_ptr<GpuBuffer<T>>    _pbSparseData;

    // Force constructor private
    NNDataSet(const string& fname, uint32_t n);
    bool Rename(const string& name);
    bool SaveNetCDF(const string& fname);
    bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n);
    void RefreshState(uint32_t batch) {} 
    bool Shard(NNDataSetEnums::Sharding sharding);
    bool UnShard();
    vector<tuple<uint64_t, uint64_t> > getMemoryUsage();
    bool CalculateSparseDatapointCounts();
    bool GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient);     
    bool SetStreaming(bool flag);
    bool GetStreaming();  
    bool SetDenoising(bool flag);
    bool GenerateDenoisingData();
    bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda);
    bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);    
    bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda);
    bool CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda);
    bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);    

public:
    /**
     * Creates dense dataset for the provided layer
     * with space allocated for the specified number of examples
     */
    NNDataSet(uint32_t examples, const NNLayer &layer);

    /**
     * Creates dense indexed dataset for the provided layer
     * with space allocated for the specified number of examples
     * and index data space allocated for the specified number of uniqueExamples.
     */
    NNDataSet(uint32_t examples, uint32_t uniqueExamples, const NNLayer &layer);

    /**
     * Creates sparse dataset for the layer with space allocated
     * for the specified number of examples. The isWeighted parameter
     * can be set to true to create a sparse weighted dataset.
     */
    NNDataSet(uint32_t examples, NNFloat sparseDensity, const NNLayer &layer, bool isWeighted);

    /**
     * Creates a sparse indexed dataset for the layer with space
     * allocated for the specified number of examples and index
     * (sparse) data space allocated for the specified number of
     * uniqueExamples and sparseDataSize (length of sparseData).
     * The isWeighted parameter can be set to true to create a
     * sparse indexed weighted dataset.
     */
    NNDataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize, const NNLayer &layer, bool isWeighted);

    ~NNDataSet();
    void Shuffle();
    T GetDataPoint(uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);
    bool SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);
    uint32_t GetSparseDataPoints(uint32_t n);
    uint32_t GetSparseIndex(uint32_t n, uint32_t i);
    bool SetSparseIndex(uint32_t n, uint32_t i, uint32_t v);
    T GetSparseDataPoint(uint32_t n, uint32_t i);
    bool SetSparseDataPoint(uint32_t n, uint32_t i, T v);
};

template<typename T> bool NNDataSet<T>::LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Indexed)  
        kLoadIndexedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);        
    else
        kLoadInputUnit(position, batch, stride, pUnit, _pbData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) 
{
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
    if (_attributes & NNDataSetEnums::Boolean)
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kLoadIndexedSparseInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight);            
        else
            kLoadSparseInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kLoadIndexedSparseAnalogInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData);
        else
            kLoadSparseAnalogInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) 
{
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;   
    if (_attributes & NNDataSetEnums::Boolean)
    {     
        if (_attributes & NNDataSetEnums::Indexed)
            kLoadIndexedSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData);        
        else
            kLoadSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kLoadIndexedSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
        else
            kLoadSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta) 
{
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL; 
    if (_attributes & NNDataSetEnums::Boolean)
    {       
        if (_attributes & NNDataSetEnums::Indexed)        
            kCalculateIndexedSparseZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, pUnit, beta);
        else
            kCalculateSparseZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, pUnit, beta);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)  
            kCalculateIndexedSparseAnalogZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta) 
{
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
    if (_attributes & NNDataSetEnums::Boolean)
    {
        if (_attributes & NNDataSetEnums::Indexed)          
            kCalculateIndexedSparseDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)   
            kCalculateIndexedSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer)
{
    // Rebuild sparse data table if dataset changed
    if (_bDirty || (batch != _batch))
    {        
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    // Initialize transposed sparse offsets
    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);
    
    // Call appropriate matrix generation kernel
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
    NNFloat* pSparseTransposedData = ((_attributes & NNDataSetEnums::Weighted) || !(_attributes & NNDataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : NULL;    
    if (_attributes & NNDataSetEnums::Boolean)
    {        
        if (_attributes & NNDataSetEnums::Indexed)   
            kCalculateIndexedSparseTransposedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)   
            kCalculateIndexedSparseTransposedAnalogMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData); 
        else
            kCalculateSparseTransposedAnalogMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData); 
    }
    
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer)
{

    // Rebuild sparse data table if dataset changed
    if (_bDirty || (batch != _batch))
    {        
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    // Initialize transposed sparse offsets
    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);
    
    // Call appropriate matrix generation kernel    
    NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
    NNFloat* pSparseTransposedData = ((_attributes & NNDataSetEnums::Weighted) || !(_attributes & NNDataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : NULL;
    if (_attributes & NNDataSetEnums::Boolean)
    {
        if (_attributes & NNDataSetEnums::Indexed) 
            kCalculateIndexedSparseTransposedDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed) 
            kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);  
        else
            kCalculateSparseTransposedAnalogDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);  
    }
    
#if 0    
    vector<uint32_t> vSparseTransposedStart(53120);        
    vector<uint32_t> vSparseTransposedEnd(53120);
    _pbSparseTransposedStart->Download(&vSparseTransposedStart[0]);
    _pbSparseTransposedEnd->Download(&vSparseTransposedEnd[0]);
    for (uint32_t i = 0; i < 53120; i++)
    printf("%6u %9u %9u %9u %9u\n", i, vSparseTransposedStart[i], vSparseTransposedEnd[i], vSparseTransposedEnd[i] - vSparseTransposedStart[i], (uint32_t)_vSparseDatapointCount[i]);
    exit(-1);
#endif       
    return true;
}


template<typename T> bool NNDataSet<T>::CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient)
{    
    if ((_attributes & NNDataSetEnums::Boolean) && !(_attributes & NNDataSetEnums::Weighted))
        kCalculateSparseTransposedWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pDelta, pWeightGradient);
    else
        kCalculateSparseTransposedAnalogWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData, pDelta, pWeightGradient);               
    return true;
}

template<typename T> float NNDataSet<T>::CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        if (_attributes & NNDataSetEnums::Boolean)
        {
           if (_attributes & NNDataSetEnums::Indexed) 
           {
                return kCalculateIndexedSparseL1Error(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pSparseWeight,
                       bSparseIgnoreZero);
           }
           else
           {
                return kCalculateSparseL1Error(position, batch, stride, pUnit, 
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pSparseWeight,
                       bSparseIgnoreZero);
           }
        }
        else
        {
           if (_attributes & NNDataSetEnums::Indexed)
           {
                return kCalculateIndexedSparseAnalogL1Error(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                        pSparseWeight,
                       _pbSparseData->_pDevData,
                       bSparseIgnoreZero);
           }
           else
           {
                return kCalculateSparseAnalogL1Error(position, batch, stride, pUnit, 
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                        pSparseWeight,
                       _pbSparseData->_pDevData,
                       bSparseIgnoreZero);
           }
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            return kCalculateIndexedL1Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateL1Error(position, batch, stride, pUnit, _pbData->_pDevData);
    }
}

template<typename T> float NNDataSet<T>::CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL; 
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;        
        if (_attributes & NNDataSetEnums::Boolean)
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2Error(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pSparseWeight,
                       bSparseIgnoreZero);                
            }
            else
            {
                return kCalculateSparseL2Error(position, batch, stride, pUnit, 
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                        pSparseWeight,
                       bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogL2Error(position, batch, stride, pUnit, 
                           _pbIndex->_pDevData,
                           _pbSparseStart->_pDevData, 
                           _pbSparseEnd->_pDevData, 
                           _pbSparseIndex->_pDevData,
                           pSparseWeight,
                           _pbSparseData->_pDevData,
                           bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseAnalogL2Error(position, batch, stride, pUnit, 
                           _pbSparseStart->_pDevData, 
                           _pbSparseEnd->_pDevData, 
                           _pbSparseIndex->_pDevData,
                           pSparseWeight,
                           _pbSparseData->_pDevData,
                           bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)        
            return kCalculateIndexedL2Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateL2Error(position, batch, stride, pUnit, _pbData->_pDevData);            
    }
}




template<typename T> float NNDataSet<T>::CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL; 
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;        
        if (_attributes & NNDataSetEnums::Boolean)
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2HingeError(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pSparseWeight,
                       bSparseIgnoreZero);                
            }
            else
            {
                return kCalculateSparseL2HingeError(position, batch, stride, pUnit, 
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                        pSparseWeight,
                       bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogL2HingeError(position, batch, stride, pUnit, 
                           _pbIndex->_pDevData,
                           _pbSparseStart->_pDevData, 
                           _pbSparseEnd->_pDevData, 
                           _pbSparseIndex->_pDevData,
                           pSparseWeight,
                           _pbSparseData->_pDevData,
                           bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseAnalogL2HingeError(position, batch, stride, pUnit, 
                           _pbSparseStart->_pDevData, 
                           _pbSparseEnd->_pDevData, 
                           _pbSparseIndex->_pDevData,
                           pSparseWeight,
                           _pbSparseData->_pDevData,
                           bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)        
            return kCalculateIndexedL2HingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateL2HingeError(position, batch, stride, pUnit, _pbData->_pDevData);            
    }
}

template<typename T> float NNDataSet<T>::CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;        
        if (_attributes & NNDataSetEnums::Indexed)   
        {
            return kCalculateIndexedSparseCrossEntropyError(position, batch, stride, pUnit,
                   _pbIndex->_pDevData,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   pSparseWeight,
                   bSparseIgnoreZero);             
        }
        else
        {
            return kCalculateSparseCrossEntropyError(position, batch, stride, pUnit,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   pSparseWeight,
                   bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)   
            return kCalculateIndexedCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else    
            return kCalculateCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
    }
}

template<typename T> float NNDataSet<T>::CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;         
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;  
        if (_attributes & NNDataSetEnums::Indexed)         
        {
            return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                   _pbIndex->_pDevData,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   pSparseWeight,
                   bSparseIgnoreZero);
        }
    else
        {
            return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, 
                    _pbSparseEnd->_pDevData, 
                    _pbSparseIndex->_pDevData,
                    pSparseWeight,
                    bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)  
            return kCalculateIndexedScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
    }
}

template<typename T> float NNDataSet<T>::CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {    
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;  
        if (_attributes & NNDataSetEnums::Boolean)
        {  
            if (_attributes & NNDataSetEnums::Indexed)         
            {            
                return kCalculateIndexedSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight);
            }
            else
            {            
                return kCalculateSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight);
            }                
        }
        else
        {
            if (_attributes & NNDataSetEnums::Indexed)         
            {   
                return kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        _pbSparseData->_pDevData);                
            }
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)    
            return kCalculateIndexedMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
    }
}

template<typename T> float NNDataSet<T>::CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)   
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL; 
        if (_attributes & NNDataSetEnums::Boolean)
        {           
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight);
            }
            else
            {
                return kCalculateSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight);                
            }  
        }
        else
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {            
                return kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            return kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            return kCalculateMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
    }
}

template<typename T> float NNDataSet<T>::CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        // Scale by 1 if data is Boolean
        if (_attributes & NNDataSetEnums::Boolean)
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit, 
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pSparseWeight,
                        bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & NNDataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                                _pbIndex->_pDevData,
                                _pbSparseStart->_pDevData,
                                _pbSparseEnd->_pDevData,
                                _pbSparseIndex->_pDevData,
                                _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                                _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
        }
    }
    else
    {
        // BUG BUG BUG unacceptable behavior, this should be caught at startup
        cout << "unsupported data format of this cost function" << endl;
        getGpu().Shutdown();
        exit(-1);
    }
}

template<typename T> float NNDataSet<T>::CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & NNDataSetEnums::Indexed)
        return kCalculateIndexedHingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
    else
        return kCalculateHingeError(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> bool NNDataSet<T>::CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;        
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
        else
            kCalculateSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, slope, alpha, lambda);
        else
            kCalculateL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, slope, alpha, lambda);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;        
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero);
        else
            kCalculateSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero);

    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            kCalculateCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero);
        else
            kCalculateSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero);
    }
    else
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData);
        else
            kCalculateScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    if (_attributes & NNDataSetEnums::Sparse) {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;        
        if (_attributes & NNDataSetEnums::Boolean) 
        {
            if (_attributes & NNDataSetEnums::Indexed)
                kCalculateIndexedSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & NNDataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, slope, alpha, lambda);
        else
            kCalculateOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, slope, alpha, lambda);
    }
    return true;
}


template<typename T> bool NNDataSet<T>::CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    if (_attributes & NNDataSetEnums::Sparse) {
        NNFloat* pSparseWeight = (_attributes & NNDataSetEnums::Weighted) ? _pbSparseWeight->_pDevData : NULL;
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;        
        if (_attributes & NNDataSetEnums::Boolean) 
        {
            if (_attributes & NNDataSetEnums::Indexed)
                kCalculateIndexedSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & NNDataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pSparseWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & NNDataSetEnums::Indexed)
            kCalculateIndexedL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, slope, alpha, lambda);
        else
            kCalculateL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, slope, alpha, lambda);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation,
                uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & NNDataSetEnums::SparseIgnoreZero;
        if (_attributes & NNDataSetEnums::Indexed)
        {
            kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                             _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
        else
        {
            kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                            _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
    } 
    else 
    {
        cout << "unsupported data format of this cost function" << endl;
        getGpu().Shutdown();
        exit(-1);
    }
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & NNDataSetEnums::Indexed)
        kCalculateIndexedHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData);
    else
        kCalculateHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    return true;
}

vector<NNDataSetBase*> LoadNetCDF(const string& fname);
bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataset);
vector<NNDataSetBase*> LoadImageData(const string& fname);
vector<NNDataSetBase*> LoadCSVData(const string& fname);
vector<NNDataSetBase*> LoadJSONData(const string& fname);
vector<NNDataSetBase*> LoadAudioData(const string& name);

#endif
