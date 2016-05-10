/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNTYPES_H
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
#include <sys/time.h>
#include <math.h>
#include <values.h>

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


static const float NN_VERSION       = 0.8f;
static const float MIN_ERROR        = 1.0e-12f;
static const float MIN_ACTIVATION   = 0.000001f;
static const float MAX_ACTIVATION   = 0.999999f;
static const float MAX_VALUE        = 999999999999999.0f;

template <typename T> struct GpuBuffer;

enum 
{
    DefaultBatch    = 512
};

enum TrainingMode 
{
    SGD = 0,
    Momentum = 1,
    AdaGrad = 2,
    Nesterov = 3,
    RMSProp = 4,
    AdaDelta = 5,
};

ostream& operator<< (ostream& out, const TrainingMode& e);

enum ErrorFunction 
{
    L1,
    L2,
    CrossEntropy,
    ScaledMarginalCrossEntropy,
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
    ReluMax,
    LinearMax,
};

ostream& operator<< (ostream& out, const Activation& a);

enum WeightInitialization
{
    Xavier,
    CaffeXavier,
    Gaussian,
    Uniform,
    UnitBall,
    Constant 
};
    
ostream& operator<< (ostream& out, const WeightInitialization& w);
    
enum PoolingFunction {
    None,
    Max,
    Stochastic,
    LocalContrastNormalization,
    LocalResponseNormalization,
    GlobalTemporal,
};

ostream& operator<< (ostream& out, const PoolingFunction& p);

#include "kernels.h"
#include "GpuSort.h"
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
    enum Attributes
    {
        Sparse = 1,         // Sparse dataset
        Boolean = 2,        // All datapoints are 0/1
        Unused = 4,         // Reserved, was multinomial, but that might be implicit
        Recurrent = 8,      // Data has a time dimension
        Mutable = 16,       // Data can be modified by running network backwards
    };

    enum Kind
    {
        Numeric = 0,
        Image = 1,
        Audio = 2
    };

    enum Sharding
    {
        None = 0,
        Model = 1,
        Data = 2,
    };

    enum DataType
    {
        UInt = 0,
        Int = 1,
        LLInt = 2,
        ULLInt = 3,
        Float = 4,
        Double = 5,
        RGBA8 = 6,
        RGBA16 = 7,
        UChar = 8,
        Char = 9
    };

    string                  _name;                          // Dataset name
    NNDataSetBase::DataType _dataType;                      // Dataset type (see above enum)
    uint32_t                _attributes;                    // Dataset characteristic (see above enum)
    uint32_t                _examples;                      // Number of examples
    uint32_t                _dimensions;                    // Dimensionality of data set
    uint32_t                _width;                         // Dataset x dimension
    uint32_t                _height;                        // Dataset y dimension
    uint32_t                _length;                        // Dataset z dimension
    uint32_t                _stride;                        // Stride between examples
    NNDataSetBase::Sharding _sharding;                      // Sharding of dataset for parallel execution
    uint32_t                _minX;                          // Beginning of local X sharding for model parallel execution 
    uint32_t                _maxX;                          // End of local X sharding for model parallel execution
    uint64_t                _sparseDataSize;                // Total sparse datapoints
    uint32_t                _maxSparseDatapoints;           // Maximum observed sparse datapoints per example
    vector<uint64_t>        _vSparseStart;                  // Vector of sparse datapoint starts per example
    GpuBuffer<uint64_t>*    _pbSparseStart;                 // GPU copy of _vSparseStart
    vector<uint64_t>        _vSparseEnd;                    // Vector of sparse datapoint ends per example
    GpuBuffer<uint64_t>*    _pbSparseEnd;                   // GPU copy of _vSparseEnd
    vector<uint32_t>        _vSparseIndex;                  // Vector of sparse indices
    GpuBuffer<uint32_t>*    _pbSparseIndex;                 // GPU copy of _vSparseIndex
    GpuBuffer<NNFloat>*     _pbDenoisingRandom;             // Denoising randoms 
    
    // Transposed sparse lookup for sparse backpropagation
    vector<uint64_t>        _vSparseDatapointCount;
    vector<uint32_t>        _vSparseTransposedStart;
    uint32_t                _sparseTransposedIndices;
    GpuBuffer<uint32_t>*    _pbSparseTransposedStart;
    GpuBuffer<uint32_t>*    _pbSparseTransposedEnd;
    GpuBuffer<uint32_t>*    _pbSparseTransposedIndex;

    // States
    bool                    _bDenoising;
    bool                    _bDirty;
    uint32_t                _batch;
    
      


    NNDataSetBase();
    NNDataSetDimensions GetDimensions();
    uint32_t GetExamples() { return _examples; };

    virtual bool SaveNetCDF(const string& fname) = 0;
    virtual bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n) = 0;
    virtual ~NNDataSetBase() = 0;
    virtual void RefreshState(uint32_t batch) = 0;
    virtual bool Shard(NNDataSetBase::Sharding sharding) = 0;
    virtual bool UnShard() = 0;
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
    virtual float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;  
};

ostream& operator<< (ostream& out, NNDataSetBase::Attributes& a);
ostream& operator<< (ostream& out, NNDataSetBase::Kind& k);
ostream& operator<< (ostream& out, NNDataSetBase::DataType& t);
ostream& operator<< (ostream& out, NNDataSetBase::Sharding& s);



template<typename T> class NNDataSet : public NNDataSetBase {
public:
    friend class NNetwork;
    friend class NNLayer;
    friend vector<NNDataSetBase*> LoadNetCDF(const string& fname);
    friend bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataSet);

private:

    vector<T>               _vData;
    GpuBuffer<T>*           _pbData;
    vector<T>               _vSparseData;
    GpuBuffer<T>*           _pbSparseData;
    GpuBuffer<T>*           _pbSparseTransposedData;


    // Force constructor private
    NNDataSet(const string& fname, uint32_t n);
    bool Rename(const string& name);
    bool SaveNetCDF(const string& fname);
    bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n);
    void RefreshState(uint32_t batch) {}    
    bool Shard(NNDataSetBase::Sharding sharding);
    bool UnShard();
    vector<tuple<uint64_t, uint64_t> > getMemoryUsage();
    bool CalculateSparseDatapointCounts();
    bool GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient);     
    bool SetDenoising(bool flag);
    bool GenerateDenoisingData();
    bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);    
    bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);

public:

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
    kLoadInputUnit(position, batch, stride, pUnit, _pbData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) 
{
    if (_attributes & Boolean)
        kLoadSparseInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData);
    else
        kLoadSparseAnalogInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) 
{
    if (_attributes & Boolean)
        kLoadSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbDenoisingRandom->_pDevData);
    else
        kLoadSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta) 
{
    if (_attributes & Boolean)
        kCalculateSparseZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pUnit, beta);
    else
        kCalculateSparseAnalogZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData, pUnit, beta);
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta) 
{
    if (_attributes & Boolean)
        kCalculateSparseDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
    else
        kCalculateSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
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
    if (_attributes & Boolean)
        kCalculateSparseTransposedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData);
    else
        kCalculateSparseTransposedAnalogMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData); 
        
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
    if (_attributes & Boolean)
        kCalculateSparseTransposedDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData);
    else
        kCalculateSparseTransposedAnalogDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData);  
    
    
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
    if (_attributes & Boolean)
        kCalculateSparseTransposedWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pDelta, pWeightGradient);
    else
        kCalculateSparseTransposedAnalogWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData, pDelta, pWeightGradient);               
    return true;
}

template<typename T> float NNDataSet<T>::CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)
    {
        if (_attributes & Boolean)
           return kCalculateSparseL1Error(position, batch, stride, pUnit, 
                  _pbSparseStart->_pDevData, 
                  _pbSparseEnd->_pDevData, 
                  _pbSparseIndex->_pDevData);
        else
           return kCalculateSparseAnalogL1Error(position, batch, stride, pUnit, 
                  _pbSparseStart->_pDevData, 
                  _pbSparseEnd->_pDevData, 
                  _pbSparseIndex->_pDevData,
                  _pbSparseData->_pDevData);  
    }
    else    
        return kCalculateL1Error(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> float NNDataSet<T>::CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)
    {
        if (_attributes & Boolean)
            return kCalculateSparseL2Error(position, batch, stride, pUnit, 
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData);
        else
            return kCalculateSparseAnalogL2Error(position, batch, stride, pUnit, 
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   _pbSparseData->_pDevData);    
    }
    else
        return kCalculateL2Error(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> float NNDataSet<T>::CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)
        return kCalculateSparseCrossEntropyError(position, batch, stride, pUnit,
               _pbSparseStart->_pDevData, 
               _pbSparseEnd->_pDevData, 
               _pbSparseIndex->_pDevData);
    else
        return kCalculateCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> float NNDataSet<T>::CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)   
        return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
               _pbSparseStart->_pDevData, 
               _pbSparseEnd->_pDevData, 
               _pbSparseIndex->_pDevData);
    else
        return kCalculateScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> float NNDataSet<T>::CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)
    {    
        if (_attributes & Boolean)
        {
            return kCalculateSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData);
        }
        else
            return kCalculateSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   _pbSparseData->_pDevData);
    }
    else
        return kCalculateMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> float NNDataSet<T>::CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit)
{
    if (_attributes & Sparse)   
    {
        if (_attributes & Boolean)
            return kCalculateSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData);
        else
            return kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   _pbSparseData->_pDevData);
    }
    else
        return kCalculateMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData);
}

template<typename T> bool NNDataSet<T>::CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & Sparse)
        kCalculateSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData);       
    else
        kCalculateL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & Sparse)
        kCalculateSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData);             
    else
        kCalculateCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & Sparse)
        kCalculateSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData);
    else
        kCalculateScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    return true;
}

template<typename T> bool NNDataSet<T>::CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta)
{
    if (_attributes & Sparse) {
      if (_attributes & Boolean) {
        kCalculateSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData);       
      } else {
        kCalculateSparseAnalogOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, _pbSparseData->_pDevData);
      }
    } else {
        kCalculateOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData);
    }
    return true;
}



vector<NNDataSetBase*> LoadNetCDF(const string& fname);
bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataset);
vector<NNDataSetBase*> LoadImageData(const string& fname);
vector<NNDataSetBase*> LoadCSVData(const string& fname);
vector<NNDataSetBase*> LoadJSONData(const string& fname);
vector<NNDataSetBase*> LoadAudioData(const string& name);

#define NNTYPES_H
#endif
