/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <stdexcept>
#include <sstream>

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"

/**
 * Explicit template class instantiation.
 * Allows the template function definitions
 * to be in this cpp file rather than the header.
 * Instantiating the same ones as in:
 * kernels.cu#EXPLICITLY_INSTANTIATE_KERNELS
 */
template class NNDataSet<NNFloat>;
template class NNDataSet<double>;
template class NNDataSet<unsigned char>;
template class NNDataSet<char>;
template class NNDataSet<uint32_t>;
template class NNDataSet<uint64_t>;
template class NNDataSet<int32_t>;
template class NNDataSet<int64_t>;

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

static std::map<TrainingMode, string> sTrainingModeMap = {
    {TrainingMode::SGD,      "SGD"},
    {TrainingMode::Momentum, "Momentum"},
    {TrainingMode::AdaGrad,  "AdaGrad"},
    {TrainingMode::Nesterov, "Nesterov"},
    {TrainingMode::RMSProp,  "RMSProp"},
    {TrainingMode::AdaDelta, "AdaDelta"},
    {TrainingMode::Adam,     "Adam"}
};

ostream& operator<< (ostream& out, const TrainingMode& e)
{
    out << sTrainingModeMap[e];
    return out;
}


static std::map<ErrorFunction, string> sErrorFunctionMap = {
    {ErrorFunction::L1,                             "L1"},
    {ErrorFunction::L2,                             "L2"},
    {ErrorFunction::CrossEntropy,                   "CrossEntropy"},
    {ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy"},
    {ErrorFunction::Hinge,                          "Hinge"},
    {ErrorFunction::L2Hinge,                        "L2Hinge"},
};

ostream& operator<< (ostream& out, const ErrorFunction& e)
{
    out << sErrorFunctionMap[e];
    return out;
}


static std::map<Activation, string> sActivationMap = {
    {Activation::Sigmoid,                              "Sigmoid"},
    {Activation::Tanh,                                 "Tanh"},
    {Activation::Linear,                               "Linear"},
    {Activation::ParametricRectifiedLinear,            "ParametricRectifiedLinear"},
    {Activation::SoftSign,                             "SoftSign"},
    {Activation::SoftPlus,                             "SoftPlus"},
    {Activation::SoftMax,                              "SoftMax"},
    {Activation::RELUMax,                              "RELUMax"},
    {Activation::LinearMax,                            "LinearMax"},
    {Activation::RectifiedLinear,                      "RectifiedLinear"},
    {Activation::LeakyRectifiedLinear,                 "LeakyRectifiedLinear"},
    {Activation::ExponentialLinear,                    "ExponentialLinear"},
    {Activation::ScaledExponentialLinear,              "ScaledExponentialLinear"}
};

ostream& operator<< (ostream& out, const Activation& a)
{
    out << sActivationMap[a];
    return out;
}


static std::map<WeightInitialization, string> sWeightInitializationMap = {
    {WeightInitialization::Xavier,           "Xavier"},
    {WeightInitialization::CaffeXavier,      "CaffeXavier"},
    {WeightInitialization::Gaussian,         "Gaussian"},
    {WeightInitialization::Uniform,          "Uniform"},
    {WeightInitialization::UnitBall,         "UnitBall"},
    {WeightInitialization::Constant,         "Constant"},
    {WeightInitialization::SELU,             "SELU"}
};

ostream& operator<< (ostream& out, const WeightInitialization& w)
{
    out << sWeightInitializationMap[w];
    return out;
}


static std::map<PoolingFunction, string> sPoolingFunctionMap = {
    {PoolingFunction::None,                       "None"},
    {PoolingFunction::Max,                        "Max"},
    {PoolingFunction::Average,                    "Average"},
    {PoolingFunction::Maxout,                     "Maxout"},
    {PoolingFunction::DotProduct,                 "DotProduct"},
    {PoolingFunction::Cosine,                     "Cosine"},
    {PoolingFunction::Stochastic,                 "Stochastic"},
    {PoolingFunction::LCN,                        "LocalContrastNormalization"},
    {PoolingFunction::LRN,                        "LocalResponseNormalization"},
    {PoolingFunction::GlobalTemporal,             "GlobalTemporal"}
};

ostream& operator<< (ostream& out, const PoolingFunction& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}


static std::map<NNDataSetEnums::Kind, string> sKindMap = {
    {NNDataSetEnums::Numeric, "Numeric"},
    {NNDataSetEnums::Image,   "Image"},
    {NNDataSetEnums::Audio,   "Audio"}
};

ostream& operator<< (ostream& out, NNDataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}


static std::map<NNDataSetEnums::Attributes, string> sAttributesMap = {
    {NNDataSetEnums::Sparse,                       "Sparse"},
    {NNDataSetEnums::Boolean,                      "Boolean"},
    {NNDataSetEnums::Compressed,                   "Compressed"},
    {NNDataSetEnums::Recurrent,                    "Recurrent"},
    {NNDataSetEnums::Mutable,                      "Mutable"},
    {NNDataSetEnums::Attributes::SparseIgnoreZero, "SparseIgnoreZero"},
    {NNDataSetEnums::Attributes::Indexed,          "Indexed"},
    {NNDataSetEnums::Attributes::Weighted,         "Weighted"},
};

ostream& operator<< (ostream& out, NNDataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}

static std::map<NNDataSetEnums::Sharding, string> sShardingMap = {
    {NNDataSetEnums::None,  "None"},
    {NNDataSetEnums::Model, "Model"},
    {NNDataSetEnums::Data,  "Data"}
};

ostream& operator<< (ostream& out, NNDataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}


static std::map<NNDataSetEnums::DataType, string> sDataTypeMap = {
    {NNDataSetEnums::UInt,   "UInt"},
    {NNDataSetEnums::Int,    "Int"},
    {NNDataSetEnums::LLInt,  "LLInt"},
    {NNDataSetEnums::ULLInt, "ULLInt"},
    {NNDataSetEnums::Float,  "Float"},
    {NNDataSetEnums::Double, "Double"},
    {NNDataSetEnums::RGB8,   "RGB8"},
    {NNDataSetEnums::RGB16,  "RGB16"},
    {NNDataSetEnums::UChar,  "UChar"},
    {NNDataSetEnums::Char,   "Char"}
};

ostream& operator<< (ostream& out, NNDataSetEnums::DataType& t)
{
    out << sDataTypeMap[t];
    return out;
}

static MPI_Datatype getMPIDataType(NNDataSetEnums::DataType datatype)
{
    MPI_Datatype mpiType;
    switch (datatype)
    {
        case NNDataSetEnums::UInt:
            mpiType             = MPI_UINT32_T;
            break;

        case NNDataSetEnums::Int:
            mpiType             = MPI_INT32_T;
            break;

        case NNDataSetEnums::ULLInt:
            mpiType             = MPI_UINT64_T;
            break;

        case NNDataSetEnums::LLInt:
            mpiType             = MPI_INT64_T;
            break;

        case NNDataSetEnums::Float:
            mpiType             = MPI_FLOAT;
            break;

        case NNDataSetEnums::Double:
            mpiType             = MPI_DOUBLE;
            break;
    }
    return mpiType;
}

static NcType getNetCDFDataType(NNDataSetEnums::DataType datatype)
{
    switch (datatype)
    {
        case NNDataSetEnums::UInt:
            return ncUint;

        case NNDataSetEnums::Int:
            return ncInt;

        case NNDataSetEnums::ULLInt:
            return ncUint64;

        case NNDataSetEnums::LLInt:
            return ncInt64;

        case NNDataSetEnums::Float:
            return ncFloat;

        case NNDataSetEnums::Double:
            return ncDouble;
    }
}

static inline bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int MPI_Bcast_string(string& s)
{
    int length                          = s.size();
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char buff[length + 1];
    strcpy(buff, s.c_str());
    int result                          = MPI_Bcast(&buff, length, MPI_CHAR, 0, MPI_COMM_WORLD);
    buff[length]                        = '\0';
    s                                   = buff;
    return result;
}

NNDataSetDimensions::NNDataSetDimensions() :
    NNDataSetDimensions(1, 1, 1)
{}

NNDataSetDimensions::NNDataSetDimensions(uint32_t width, uint32_t height, uint32_t length) :
    _width(width),
    _height(height),
    _length(length),
    _dimensions(0)
{
    if (width > 1)
    {
        ++_dimensions;
    }
    if (height > 1)
    {
        ++_dimensions;
    }
    if (length > 1)
    {
        ++_dimensions;
    }
}

template<typename T> NNDataSetBase* createNNDataSet(const NNDataSetDescriptor &descriptor)
{
    using NNDataSetEnums::Attributes;

    uint32_t attributes = descriptor._attributes;
    if (!NNDataSetDescriptor::isSupported(attributes))
    {
        stringstream msg;
        msg << "Unsupported attributes " << attributes << " for dataset " << descriptor._name;
       std::runtime_error(msg.str());
    }

    NNDataSetBase *dataset;
    if (attributes & Attributes::Sparse)
    {
        /* sparse data */
        dataset = new NNDataSet<T>(descriptor._examples, descriptor._sparseDensity, descriptor._dim, false,
                                   descriptor._name);
    } else
    {
        /* dense data */
        dataset = new NNDataSet<T>(descriptor._examples, descriptor._dim, descriptor._name);
    }
    return dataset;
}

NNDataSetBase* createNNDataSet(const NNDataSetDescriptor &descriptor)
{
    NNDataSetBase *dataset;
    using NNDataSetEnums::DataType;

    switch (descriptor._dataType) {
        case DataType::UInt:
            dataset = createNNDataSet<uint32_t>(descriptor);
            break;
        case DataType::Int:
            dataset = createNNDataSet<int>(descriptor);
            break;
        case DataType::Float:
            dataset = createNNDataSet<float>(descriptor);
            break;
        case DataType::Double:
            dataset = createNNDataSet<double>(descriptor);
            break;
        case DataType::Char:
            dataset = createNNDataSet<char>(descriptor);
            break;
        case DataType::UChar:
        case DataType::RGB8:
            dataset = createNNDataSet<uint8_t>(descriptor);
            break;
        default:
            stringstream msg;
            msg << "Unsupported data type: " << descriptor._dataType
                << ". DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8";
            std::runtime_error(msg.str());
    }
    return dataset;
}

NNDataSetBase::NNDataSetBase() :
_name(""),
_attributes(NNDataSetEnums::None),
_examples(0),
_uniqueExamples(0),
_dimensions(0),
_width(0),
_height(0),
_length(0),
_stride(0),
_sharding(NNDataSetEnums::Sharding::None),
_minX(0),
_maxX(0),
_sparseDataSize(0),
_sparseTransposedIndices(0),
_sparseDensity(0),
_bDenoising(false),
_pbSparseStart(),
_pbSparseEnd(),
_pbSparseIndex(),
_pbIndex(),
_pbSparseTransposedStart(),
_pbSparseTransposedEnd(),
_pbSparseTransposedIndex(),
_pbSparseTransposedData(),
_batch(0),
_pbDenoisingRandom(),
_bStreaming(false),
_bIndexed(false),
_bDirty(true)
{
}

NNDataSetBase::NNDataSetBase(const string &name, NNDataSetEnums::DataType dataType, uint32_t examples,
                             uint32_t uniqueExamples, const NNDataSetDimensions &datasetDim) :
    _name(name),
    _dataType(dataType),
    _attributes(NNDataSetEnums::None),
    _examples(examples),
    _uniqueExamples(uniqueExamples),
    _localExamples(examples),
    _dimensions(datasetDim._dimensions),
    _width(datasetDim._width),
    _height(datasetDim._height),
    _length(datasetDim._length),
    _stride(0),
    _sharding(NNDataSetEnums::Sharding::None),
    _minX(0),
    _maxX(0),
    _sparseDataSize(0),
    _sparseTransposedIndices(0),
    _sparseDensity(0),
    _bDenoising(false),
    _pbSparseStart(),
    _pbSparseEnd(),
    _pbSparseIndex(),
    _pbIndex(),
    _pbSparseTransposedStart(),
    _pbSparseTransposedEnd(),
    _pbSparseTransposedIndex(),
    _pbSparseTransposedData(),
    _batch(0),
    _pbDenoisingRandom(),
    _bStreaming(false),
    _bIndexed(false),
    _bDirty(true)
{
}

NNDataSetBase::~NNDataSetBase() {}

NNDataSetDimensions NNDataSetBase::GetDimensions()
{
    return NNDataSetDimensions(_width, _height, _length);
}

template<typename T> vector<tuple<uint64_t, uint64_t> > NNDataSet<T>::getMemoryUsage()
{
    // Calculate per-process memory usage
    uint64_t cpuMemory                          = 0;
    uint64_t gpuMemory                          = 0;
    if (_attributes & NNDataSetEnums::Sparse)
    {
        // Sparse start and end consumption
        cpuMemory                              += _uniqueExamples * 2 * sizeof(uint64_t);
        gpuMemory                              += _uniqueExamples * 2 * sizeof(uint64_t);
        cpuMemory                              += _vSparseIndex.size() * sizeof(uint32_t);
        gpuMemory                              += _vSparseIndex.size() * sizeof(uint32_t);
        if (!(_attributes & NNDataSetEnums::Boolean))
        {
            cpuMemory                          += _vSparseData.size() * sizeof(T);
            gpuMemory                          += _vSparseData.size() * sizeof(T);
        }
    }
    else
    {
        cpuMemory                              += _vData.size() * sizeof(T);
        gpuMemory                              += _vData.size() * sizeof(T);
    }

    // Add local index structure
    if (_bIndexed)
    {
        cpuMemory                              += _examples * sizeof(uint32_t);
        gpuMemory                              += _examples * sizeof(uint32_t);
    }

    // Gather and return memory usage per process
    vector<tuple<uint64_t, uint64_t> > vResult(getGpu()._numprocs);
    vResult[getGpu()._id]                       = make_tuple(cpuMemory, gpuMemory);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vResult.data(), sizeof(tuple<uint64_t, uint64_t>), MPI_BYTE, MPI_COMM_WORLD);
    return vResult;
}

/* dense data */
template<typename T> NNDataSet<T>::NNDataSet(uint32_t examples, const NNDataSetDimensions &dim, const string &name) :
    NNDataSetBase(name, NNDataSetEnums::getDataType<T>(), examples, examples, dim)
{
    // sparse density for dense data is 100%
    _sparseDensity = 1.0f;
     _stride = _width * _height * _length;
    _vData.resize(_stride * _examples);
    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
}

/* dense indexed data */
template<typename T> NNDataSet<T>::NNDataSet(uint32_t examples, uint32_t uniqueExamples, const NNDataSetDimensions &dim,
                                             const string &name) :
    NNDataSetBase(name, NNDataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    // sparse density for dense data is 100%
    _sparseDensity = 1.0f;
    _stride = _width * _height * _length;
    _attributes = NNDataSetEnums::Attributes::Indexed;
    _bIndexed = true;

    /*
     * _vData holds the index data (unique data points)
     * _vIndex holds the index numbers to the data in _vData of each example
     */
    _vData.resize(_stride * _uniqueExamples);
    _vIndex.resize(_examples, 0);
    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
    _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
}

/* sparse and sparse weighted data */
template<typename T> NNDataSet<T>::NNDataSet(uint32_t examples, NNFloat sparseDensity, const NNDataSetDimensions &dim,
                                             bool isWeighted, const string &name) :
    /**
     * 1. stride for sparse data needs to be calculated per example,
     * stride = (sparseEnd - sparseStart), hence no need to set _stride
     *
     * 2. because we do not have the actual data points (examples) up front,
     * we need to allocate enough space for each sparse example, take sparseDensity
     * as the maximum number of non-zero coordinates for any example in this dataset.
     */
    NNDataSet(examples, examples,
              (size_t) (((double) (dim._width * dim._height * dim._length * examples)) * sparseDensity), dim, false,
              isWeighted, name)
{
    /*
     *  reset the attribute to be sparse only since we call
     *  the constructor for sparse + indexed NNDataSet
     */
    _attributes = NNDataSetEnums::Attributes::Sparse;
    if(isWeighted) {
        _attributes |= NNDataSetEnums::Attributes::Weighted;
    }
}

/* sparse indexed and sparse indexed weighted data */
template<typename T> NNDataSet<T>::NNDataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize,
                                             const NNDataSetDimensions &dim, bool isIndexed, bool isWeighted, const string &name) :
    NNDataSetBase(name, NNDataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    _attributes = NNDataSetEnums::Attributes::Sparse;
    _sparseDataSize = sparseDataSize;

    /**
     * _vSparseStart, _vSparseEnd, _vSparseData, _vSparseIndex holds the
     * unique data points. _vIndex holds the index numbers to the data for
     * each example. For indexed sparse data, we are expected to know the
     * unique data points up front, so we do not require the caller to pass
     * a sparseDensity since we can calculate it as:
     * sparseDensity = sparseDataSize / (uniqueExamples * x * y * z)
     * this is done in NNDataSet::CalculateSparseDatapointCounts()
     */
    _vSparseStart.resize(_uniqueExamples, 0);
    _vSparseEnd.resize(_uniqueExamples, 0);
    _vSparseData.resize(_sparseDataSize);
    _vSparseIndex.resize(_sparseDataSize, 0);

    /**
     * Need to initialize sparse start and end so that NNDataSet::Shard works properly.
     * Otherwise, NNDataSet::Shard will resize the dataset to zero, since it'll iterates
     * through the sparse data and shards each example by features modulo numProcs.
     */
    size_t sparseStride = (_sparseDataSize + _uniqueExamples - 1) / _uniqueExamples;

    _vSparseStart[0] = 0;
    _vSparseEnd[0] = sparseStride;
    for(uint32_t i =1; i < _uniqueExamples; ++i)
    {
        _vSparseStart[i] = _vSparseEnd[i-1];
        _vSparseEnd[i] = _vSparseStart[i] + sparseStride;
    }

    // initialize gpu buffers
    _pbSparseStart.reset(new GpuBuffer<uint64_t>(_vSparseStart.size(), false, _bStreaming));
    _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_vSparseEnd.size(), false, _bStreaming));
    _pbSparseData.reset(new GpuBuffer<T>(_vSparseData.size(), false, _bStreaming));
    _pbSparseIndex.reset(new GpuBuffer<uint32_t>(_vSparseIndex.size(), false, _bStreaming));

    if(isIndexed) {
        _attributes |=  NNDataSetEnums::Attributes::Indexed;
        _bIndexed = true;
        _vIndex.resize(_examples, 0);
        _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
    }

    if (isWeighted)
    {
        _attributes |= NNDataSetEnums::Attributes::Weighted;
        _vDataWeight.resize(_examples);
        _pbDataWeight.reset(new GpuBuffer<NNFloat>(_vDataWeight.size(), false, _bStreaming));
    }
}

template<typename T> void NNDataSet<T>::LoadDenseData(const void *srcData)
{
    const T* srcDataTyped = static_cast<const T*>(srcData);

    if (_attributes & NNDataSetEnums::Attributes::Sparse)
    {
        throw std::runtime_error("Cannot set dense data on a sparse NNDataSet");
    } else
    {
         copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());
         _pbData->Upload(_vData.data());
    }
}

template<typename T> void NNDataSet<T>::LoadSparseData(const uint64_t *srcSparseStart, const uint64_t *srcSparseEnd,
                                                       const void *srcSparseData, const uint32_t *srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & NNDataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        copy(srcSparseStart, srcSparseStart + _uniqueExamples, _vSparseStart.data());
        copy(srcSparseEnd, srcSparseEnd + _uniqueExamples, _vSparseEnd.data());
        copy(srcSparseDataTyped, srcSparseDataTyped + dataLength, _vSparseData.data());
        copy(srcSparseIndex, srcSparseIndex + dataLength, _vSparseIndex.data());

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    } else
    {
        throw std::runtime_error("Cannot set sparse data on a non sparse NNDataSet");
    }
}

template<typename T> void NNDataSet<T>::LoadSparseData(const long *srcSparseStart, const long *srcSparseEnd,
                                                       const void *srcSparseData, const long *srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & NNDataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        for (uint32_t i = 0; i < _uniqueExamples; ++i)
        {
            _vSparseStart[i] = (uint64_t) srcSparseStart[i];
            _vSparseEnd[i] = (uint64_t) srcSparseEnd[i];
        }
        for (uint64_t i = 0; i < dataLength; ++i)
        {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = (uint32_t) srcSparseIndex[i];
        }

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    } else
    {
        throw std::runtime_error("Cannot set sparse data on a non sparse NNDataSet");
    }
}

template<typename T> void NNDataSet<T>::LoadIndexedData(const uint32_t *srcIndexedData)
{
    if (_attributes & NNDataSetEnums::Attributes::Indexed)
    {
        copy(srcIndexedData, srcIndexedData + _vIndex.size(), _vIndex.data());
        _pbIndex->Upload(_vIndex.data());
    } else
    {
        throw std::runtime_error("Cannot set indexed data on a non indexed NNDataSet");
    }
}

template<typename T> void NNDataSet<T>::LoadDataWeight(const NNFloat *srcWeightData)
{
    if (_attributes & NNDataSetEnums::Attributes::Weighted)
    {
        copy(srcWeightData, srcWeightData + _vDataWeight.size(), _vDataWeight.data());
        _pbDataWeight->Upload(_vDataWeight.data());
    } else
    {
        throw std::runtime_error("Cannot set weight data on a non weighted NNDataSet");
    }
}

template<typename T> T NNDataSet<T>::GetDataPoint(uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
    // Illegal to call on sparse data set
    if (_attributes & NNDataSetEnums::Sparse)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetDataPoint: attempt to read non-sparse data from sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetDataPoint: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Test bounds of x, y, and z
    if ((x >= _width) || (y >= _height) || (z >= _length))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetDataPoint: illegal datapoint coordinates (%u, %u, %u).\n", x, y, z);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    return _vData[(n * _stride) + x + _width * (y + z * _height)];
}

template<typename T> bool NNDataSet<T>::SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
    // Illegal to call on sparse data set
    if (_attributes & NNDataSetEnums::Sparse)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetDataPoint: attempt to read non-sparse data from sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetDataPoint: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Test bounds of x, y, and z
    if ((x >= _width) || (y >= _height) || (z >= _length))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetDataPoint: illegal datapoint coordinates (%u, %u, %u).\n", x, y, z);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    _vData[(n * _stride) + x + _width * (y + z * _height)]  = v;
}

template<typename T> uint64_t NNDataSet<T>::GetSparseDataPoints(uint32_t n)
{
    // Illegal to call on non-sparse data set
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseDataPoints: attempt to read sparse data from non-sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseDataPoints: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    return _vSparseEnd[n] - _vSparseStart[n];
}

template<typename T> uint32_t NNDataSet<T>::GetSparseIndex(uint32_t n, uint32_t i)
{
    // Illegal to call on non-sparse data set
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseIndex: attempt to read sparse data from non-sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseIndex: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Make sure index is within bounds
    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseIndex: Sparse index %u out of range (0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    return _vSparseIndex[_vSparseStart[n] + i];
}

template<typename T> bool NNDataSet<T>::SetSparseIndex(uint32_t n, uint32_t i, uint32_t v)
{
    // Illegal to call on non-sparse data set
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseDataIndex: attempt to read sparse data from non-sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseDataIndex: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Make sure index is within bounds
    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseIndex: Sparse index %u out of range (0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    _vSparseIndex[_vSparseStart[n] + i]         = v;
    _bDirty                                     = true;
    return true;
}

template<typename T> T NNDataSet<T>::GetSparseDataPoint(uint32_t n, uint32_t i)
{
    // Illegal to call on non-sparse data set
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseDataPoint: attempt to read sparse data from non-sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseDataPoint: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Make sure index is within bounds
    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseDataPoint: Sparse index %u out of range (0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    return _vSparseData[_vSparseStart[n] + i];
}

template<typename T> bool NNDataSet<T>::SetSparseDataPoint(uint32_t n, uint32_t i, T v)
{
    // Illegal to call on non-sparse data set
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseDataPoint: attempt to read sparse data from non-sparse data set.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Make sure example is within bounds
    if (n >= _examples)
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseDataPoint: illegal example index.\n");
        }
        getGpu().Shutdown();
        exit(-1);
    }

    // Remap n if indexed
    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    // Make sure index is within bounds
    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetSparseDataPoint: Sparse index %u out of range (0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(-1);
    }

    _vSparseData[_vSparseStart[n] + i]         = v;
    _bDirty                                    = true;
    return true;
}

template<typename T> NNDataSet<T>::NNDataSet(const string& fname, uint32_t n) :
_pbData(),
_pbSparseData()
{
    // Read File entirely with process 0
    bool bResult                                = true;
    if (getGpu()._id == 0)
    {
        bool bOpened                            = false;
        try
        {
            // Work around poor exception throwing design here
            NcFile nfc(fname.c_str(), NcFile::read);
            bOpened                             = true;

            string nstring                      = to_string(n);
            string vname                        = "name" + nstring;
            NcGroupAtt nameAtt                  = nfc.getAtt(vname);
            if (nameAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No dataset name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(_name);
            cout << "NNDataSet<T>::NNDataSet: Name of data set: " << _name << endl;


            vname                               = "dataType" + nstring;
            NcGroupAtt dataTypeAtt              = nfc.getAtt(vname);
            if (dataTypeAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No datatype supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            int dataType;
            dataTypeAtt.getValues(&dataType);
            _dataType                           = (NNDataSetEnums::DataType)dataType;

            vname                               = "attributes" + nstring;
            NcGroupAtt attributesAtt            = nfc.getAtt(vname);
            if (attributesAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            attributesAtt.getValues(&_attributes);
            if (_attributes != 0)
            {
                int tempAtt                     = _attributes;
                cout << "NNDataSet<T>::NNDataSet: Attributes:";
                while (tempAtt != 0)
                {
                    NNDataSetEnums::Attributes a = (NNDataSetEnums::Attributes)(1 << (ffs(tempAtt) - 1));
                    cout << " " << a;
                    tempAtt                    ^= 1 << (ffs(tempAtt) - 1);
                }
                cout << endl;
            }

            vname                               = "examplesDim" + nstring;
            NcDim examplesDim                   = nfc.getDim(vname);
            if (examplesDim.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No examples count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            _examples                           = examplesDim.getSize();

            // Check for nonzero examples count
            if (_examples == 0)
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: Zero-valued Examples count in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            // Grab unique examples count if present
            vname                               = "uniqueExamplesDim" + nstring;
            NcDim uniqueExamplesDim                   = nfc.getDim(vname);
            if (uniqueExamplesDim.isNull())
            {
                _uniqueExamples                 = _examples;
            }
            else
            {
                _uniqueExamples                 = uniqueExamplesDim.getSize();
            }

            vname                               = "dimensions" + nstring;
            NcGroupAtt dimensionsAtt            = nfc.getAtt(vname);
            if (dimensionsAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No dimension count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dimensionsAtt.getValues(&_dimensions);

            // Check for valid dimensions count
            if ((_dimensions < 1) || (_dimensions > 3))
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: Invalid dimension count (" + to_string(_dimensions) + ") supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            vname                               = "width" + nstring;
            NcGroupAtt widthAtt                 = nfc.getAtt(vname);
            if (widthAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: No datapoint width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&_width);

            if (_dimensions > 1)
            {
                vname                           = "height" + nstring;
                NcGroupAtt heightAtt            = nfc.getAtt(vname);
                if (heightAtt.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No datapoint height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                heightAtt.getValues(&_height);
            }
            else
                _height                         = 1;

            if (_dimensions > 2)
            {
                vname                           = "length" + nstring;
                NcGroupAtt lengthAtt            = nfc.getAtt(vname);
                if (lengthAtt.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No datapoint length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                lengthAtt.getValues(&_length);
            }
            else
                _length                         = 1;
            cout << "NNDataSet<T>::NNDataSet: " << _dimensions << "-dimensional data comprised of (" << _width << ", " << _height << ", " << _length << ") datapoints." << endl;

            // Make sure all dimensions are at least 1
            if ((_width == 0) || (_height == 0) || (_length == 0))
            {
                throw NcException("NcException", "NNDataSet::NNDataSet: Invalid dataset dimensions in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            // Read sparse data (type is irrelevant here)
            if (_attributes & NNDataSetEnums::Sparse)
            {
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                vname                           = "sparseDataDim" + nstring;
                NcDim sparseDataDim             = nfc.getDim(vname);
                if (sparseDataDim.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No sparse data dimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                _sparseDataSize                 = sparseDataDim.getSize();

                // Check for at least one datapoint
                if (_sparseDataSize == 0)
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: Sparse data set with no actual data in NetCDF input file " + fname, __FILE__, __LINE__);
                }

                _vSparseIndex.resize(_sparseDataSize);
                cout << "NNDataSet<T>::NNDataSet: " << _sparseDataSize << " total datapoints." << endl;
                vname                           = "sparseStart" + nstring;
                NcVar sparseStartVar            = nfc.getVar(vname);
                if (sparseStartVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No sparse offset start supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname                           = "sparseEnd" + nstring;
                NcVar sparseEndVar              = nfc.getVar(vname);
                if (sparseEndVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No sparse data end supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname                           = "sparseIndex" + nstring;
                NcVar sparseIndexVar            = nfc.getVar(vname);
                if (sparseIndexVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No sparse data indices supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }

                // Read data into CPU memory (account for old datasets using 32-bit indices)
                NcType vStartType               = sparseStartVar.getType();
                if (vStartType == ncUint)
                {
                    vector<uint32_t> vTempSparseStart(_uniqueExamples);
                    sparseStartVar.getVar((uint32_t*)vTempSparseStart.data());
                    copy(vTempSparseStart.begin(), vTempSparseStart.end(), _vSparseStart.begin());
                }
                else
                    sparseStartVar.getVar((uint64_t*)_vSparseStart.data());

                NcType vEndType                 = sparseEndVar.getType();
                if (vEndType == ncUint)
                {
                    vector<uint32_t> vTempSparseEnd(_uniqueExamples);
                    sparseEndVar.getVar((uint32_t*)vTempSparseEnd.data());
                    copy(vTempSparseEnd.begin(), vTempSparseEnd.end(), _vSparseEnd.begin());
                }
                else
                    sparseEndVar.getVar((uint64_t*)_vSparseEnd.data());
                sparseIndexVar.getVar((uint32_t*)_vSparseIndex.data());

                // If not Boolean, then read templated point values
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    vname                       = "sparseData" + nstring;
                    NcVar sparseDataVar         = nfc.getVar(vname);
                    if (sparseDataVar.isNull())
                    {
                        throw NcException("NcException", "NNDataSet::NNDataSet: No sparse data located in NetCDF input file " + fname, __FILE__, __LINE__);
                    }
                    _vSparseData.resize(sparseDataDim.getSize());
                    sparseDataVar.getVar(_vSparseData.data());
                }
            }
            else
            {
                // Non-sparse data
                _stride                         = _width * _height * _length;
                vname                           = "dataDim" + nstring;
                NcDim dataDim                   = nfc.getDim(vname);
                if (dataDim.isNull())
                {
                        throw NcException("NcException", "NNDataSet::NNDataSet: No data dimensions located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname                           = "data" + nstring;
                NcVar dataVar                   = nfc.getVar(vname);

                if (_attributes & NNDataSetEnums::Boolean)
                {

                    // Read compressed boolean data then expand it
                    uint64_t size               = (uint64_t)_width * (uint64_t)_height * (uint64_t)_length;
                    _vData.resize(dataDim.getSize() * size);
                    memset(_vData.data(), 0, _vData.size() * sizeof(T));
                    vector<T> vData(dataDim.getSize());
                    dataVar.getVar(vData.data());
                    for (int i = 0; i < dataDim.getSize(); i++)
                        _vData[i * size + vData[i]] = (T)1.0;
                }
                else
                {
                    _vData.resize(dataDim.getSize());
                    dataVar.getVar(_vData.data());
                }
            }

            // Read data weights if present
            if (_attributes & NNDataSetEnums::Weighted)
            {
                vname                       = "dataWeight" + nstring;
                NcVar DataWeightVar         = nfc.getVar(vname);
                if (DataWeightVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No data weights located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                _vDataWeight.resize(_examples);
                DataWeightVar.getVar(_vDataWeight.data());
            }

            // Read index if indexed
            if (_attributes & NNDataSetEnums::Indexed)
            {
                vname                       = "index" + nstring;
                NcVar indexVar              = nfc.getVar(vname);
                if (indexVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: No indexed data located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
               _vIndex.resize(_examples);
               indexVar.getVar(_vIndex.data());
            }

            cout << "NNDataSet<T>::NNDataSet: " << _examples << " examples." << endl;
            cout << "NNDataSet<T>::NNDataSet: " << _uniqueExamples << " unique examples." << endl;
        }
        catch (NcException& e)
        {

            if (!bOpened)
            {
                cout << "Exception: NNDataSet::NNDataSet: Error opening NetCDF input file " << fname << endl;
            }
            else
            {
                cout << "Exception: " << e.what() << endl;
            }
            bResult                             = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    // Receive data attributes from master process
    MPI_Bcast_string(_name);
    MPI_Bcast(&_dataType, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_examples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_uniqueExamples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_width, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_height, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_sparseDataSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Create unsharded local fragments
    if (getGpu()._id != 0)
    {
        _vData.resize(0);
        _vSparseStart.resize(_uniqueExamples, 0);
        _vSparseEnd.resize(_uniqueExamples, 0);
        _vSparseIndex.resize(0);
        _vSparseData.resize(0);
    }

    // Broadcast indices if indexed
    if (_attributes & NNDataSetEnums::Indexed)
    {
        _vIndex.resize(_examples);
        MPI_Bcast(_vIndex.data(), _examples, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    // Broadcast sparse weights if presented
    if (_attributes & NNDataSetEnums::Weighted)
    {
        _vDataWeight.resize(_examples);
        MPI_Bcast(_vDataWeight.data(), _examples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Generate sparse data lookup tables if data is sparse
    if (_attributes & NNDataSetEnums::Sparse)
    {
        CalculateSparseDatapointCounts();
    }
}

template<typename T> bool NNDataSet<T>::Rename(const string& name)
{
    _name                                       = name;
    return true;
}

// Counts the number of each type of sparse datapoint for generating transposed matrices during backpropagation
template<typename T> bool NNDataSet<T>::CalculateSparseDatapointCounts()
{
    if (_attributes & NNDataSetEnums::Sparse)
    {
        // Calculate individual counts for each datapoint
        uint64_t N                              = _width * _height * _length;
        _vSparseDatapointCount.resize(N);
        _vSparseMaxDatapointCount.resize(N);
        _vSparseMultiDatapointCount.resize(N);
        std::fill(_vSparseDatapointCount.begin(), _vSparseDatapointCount.end(), 0);
        std::fill(_vSparseMaxDatapointCount.begin(), _vSparseMaxDatapointCount.end(), 0);
        std::fill(_vSparseMultiDatapointCount.begin(), _vSparseMultiDatapointCount.end(), 0);

        // Count max sparse datapoints, accounting for duplicates
        vector<uint32_t> vCount(N, 0);
        vector<uint32_t> vExampleCount(_uniqueExamples, 0);
        if (_attributes & NNDataSetEnums::Indexed)
        {
            for (size_t i = 0; i < _examples; i++)
            {
                vExampleCount[_vIndex[i]]++;
            }
        }
        else
        {
            std::fill(vExampleCount.begin(), vExampleCount.end(), 1);
        }

        for (size_t i = 0; i < _uniqueExamples; i++)
        {
            uint64_t count                      = _vSparseEnd[i] - _vSparseStart[i];
            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
            {
                vCount[_vSparseIndex[j]]++;
            }

            bool bMulti = false;
            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
            {
                uint32_t x                      = _vSparseIndex[j];

                if (vCount[x] > 0)
                {
                    _vSparseMaxDatapointCount[x] = std::max(_vSparseMaxDatapointCount[x], vCount[x]);
                    if (vCount[x] > 1)
                        _vSparseMultiDatapointCount[x] += vExampleCount[i];
                    _vSparseDatapointCount[x]  += vExampleCount[i] * vCount[x];
                    vCount[x]                   = 0;
                }

            }
        }

        // Scale up maximum points
        size_t sz = 0;
        size_t batch = 2048;
        size_t active = 0;
        for (size_t i = 0; i < N; i++)
        {
            //cout << i << ": " << _vSparseDatapointCount[i] << " " << _vSparseMaxDatapointCount[i] << " "<< _vSparseMultiDatapointCount[i] << endl;
            size_t size1 = _vSparseDatapointCount[i];
            size1 = std::min(batch, size1);
            active += (_vSparseDatapointCount[i] > 0);
            if (_vSparseMaxDatapointCount[i] > 1)
            {
                size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
                size1 = std::max(size1, size2);
            }
            sz += size1;
        }

        // Calculate sparse density
        _sparseDensity = (double_t)_sparseDataSize / (double_t)(_uniqueExamples * N);
        return true;
    }
    else
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::CalculateSparseDatapointCounts: Attempt to calculate sparse datapoint counts on non-sparse dataset %s.\n", _name.c_str());
        }
        return false;
    }
}

template<typename T> bool NNDataSet<T>::GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer)
{

    if (_bDirty)
    {
        CalculateSparseDatapointCounts();
        _bDirty                             = false;
    }

    uint64_t NData                          = _width * _height * _length;
    uint32_t Nx, Ny, Nz, Nw;
    tie(Nx, Ny, Nz, Nw)                     = pLayer->GetLocalDimensions();
    uint64_t NLayer                         = Nx * Ny * Nz * Nw;
    uint64_t N                              = max(NData, NLayer);
    _vSparseTransposedStart.resize(N);
    if (!_pbSparseTransposedStart)
        _pbSparseTransposedStart.reset(new GpuBuffer<uint32_t>(N));
    if (!_pbSparseTransposedEnd)
        _pbSparseTransposedEnd.reset(new GpuBuffer<uint32_t>(N));

    // Set batch and calculate new sparse matrix data
    _batch                                  = batch;
    uint32_t offset                         = 0;
    for (size_t i = 0; i < _vSparseDatapointCount.size(); i++)
    {
        _vSparseTransposedStart[i]          = offset;
        size_t size1 = _vSparseDatapointCount[i];
        size1 = std::min((size_t)batch, size1);
        if (_vSparseMaxDatapointCount[i] > 1)
        {
            size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
            size1 = std::max(size1, size2);
        }
        offset                             += size1;
        offset                              = ((offset + 31) >> 5) << 5;
    }
    _pbSparseTransposedStart->Upload(_vSparseTransposedStart.data());

    if (offset > _sparseTransposedIndices)
    {
        _sparseTransposedIndices            = offset;
        printf("NNDataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient index matrix %s.\n", _sparseTransposedIndices * sizeof(uint32_t), _name.c_str());
        _pbSparseTransposedIndex.reset(new GpuBuffer<uint32_t>(_sparseTransposedIndices));
        if (!(_attributes & NNDataSetEnums::Boolean) || (_attributes & NNDataSetEnums::Weighted))
        {
            printf("NNDataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient value matrix %s.\n", _sparseTransposedIndices * sizeof(NNFloat), _name.c_str());
            _pbSparseTransposedData.reset(new GpuBuffer<NNFloat>(_sparseTransposedIndices));
        }
    }
    return true;
}

template<typename T> bool NNDataSet<T>::SetDenoising(bool flag)
{
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::SetDenoising: Attempt to set denoising on non-sparse data set.\n");
        }
        return false;
    }
    else if (!flag && _bDenoising)
    {
        _pbDenoisingRandom.reset();
        _bDenoising                             = false;
    }
    else if (flag && !_bDenoising)
    {
        _pbDenoisingRandom.reset(new GpuBuffer<NNFloat>((uint64_t)_vSparseIndex.size()));
    }
    return true;
}

template<typename T> bool NNDataSet<T>::SetStreaming(bool flag)
{
    // Check for streaming capability, warn on each GPU that doesn't support it
    if (!getGpu()._bUnifiedMemory)
    {
        printf("NNDataSet::SetStreaming: Streaming datasets not supported on GPU %d\n", getGpu()._id);
    }

    // Set dirty if streaming state has changed
    if (flag != _bStreaming)
    {
        _bStreaming = flag & getGpu()._bUnifiedMemory;
        _bDirty     = true;
    }

    return true;
}

template<typename T> bool NNDataSet<T>::GetStreaming()
{
    return _bStreaming;
}

template<typename T> bool NNDataSet<T>::GenerateDenoisingData()
{
    if (!(_attributes & NNDataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GenerateDenoisingData: Attempt to generate denoising randoms on non-sparse data set.\n");
        }
        return false;
    }
    curandGenerateUniform(getGpu()._RNG, _pbDenoisingRandom->_pDevData, _vSparseIndex.size());
    return true;
}

template<typename T> bool NNDataSet<T>::UnShard()
{
    if (_sharding == NNDataSetEnums::Model)
    {
        if (_attributes & NNDataSetEnums::Sparse)
        {
            // Download all current data from all GPUs
            _pbSparseStart->Download(_vSparseStart.data());
            _pbSparseEnd->Download(_vSparseEnd.data());
            _pbSparseIndex->Download(_vSparseIndex.data());
            _pbSparseStart.reset();
            _pbSparseEnd.reset();
            _pbSparseIndex.reset();
            if (!(_attributes & NNDataSetEnums::Boolean))
            {
                _pbSparseData->Download(_vSparseData.data());
                _pbSparseData.reset();
            }

            // Subtract local index offset
            int32_t xmin                        = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
            int32_t xmax                        = ((size_t)_width * ((size_t)getGpu()._id + 1)) / (size_t)getGpu()._numprocs;
            for (auto& index : _vSparseIndex)
                index                          -= xmin;

            // Gather total sparse counts
            vector<uint32_t> vSparseCount(_uniqueExamples);
            for (uint32_t i = 0; i < _uniqueExamples; i++)
            {
                vSparseCount[i]                 = _vSparseEnd[i] - _vSparseStart[i];
            }
            uint64_t datapoints                 = _vSparseIndex.size();
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : &datapoints, &datapoints, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vSparseCount.data(), vSparseCount.data(), _uniqueExamples, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

            // Unshard
            if (getGpu()._id == 0)
            {
                vector<uint64_t> vTempSparseStart(_uniqueExamples);
                vector<uint64_t> vTempSparseEnd(_uniqueExamples);
                vector<uint32_t> vTempSparseIndex(datapoints);
                vector<T> vTempSparseData;
                if (!(_attributes & NNDataSetEnums::Boolean))
                    vTempSparseData.resize(datapoints);
                vTempSparseStart[0]             = 0;
                uint64_t start                  = 0;

                // Initialize counts and generate local shard
                for (int i = 0; i < _uniqueExamples; i++)
                {
                    vTempSparseStart[i]         = start;
                    vTempSparseEnd[i]           = start;
                    for (uint64_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
                    {
                        vTempSparseIndex[vTempSparseEnd[i]]
                                                = _vSparseIndex[vTempSparseEnd[i]];
                        if (!(_attributes & NNDataSetEnums::Boolean))
                        {
                            vTempSparseData[vTempSparseEnd[i]]
                                                = _vSparseData[vTempSparseEnd[i]];
                        }
                        vTempSparseEnd[i]++;
                    }
                    start                      += vSparseCount[i];
                }

                // Gather remaining shards
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;
                    MPI_Recv(vSparseCount.data(), _uniqueExamples, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    vector<uint32_t> vPeerSparseIndex(size);
                    MPI_Recv(&vPeerSparseIndex, size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);
                    vector<T> vPeerSparseData;
                    if (!(_attributes & NNDataSetEnums::Boolean))
                    {
                        vPeerSparseData.resize(size);
                        MPI_Recv(vPeerSparseData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    }

                    // Merge local data
                    for (uint32_t i = 0; i < _uniqueExamples; i++)
                    {
                        uint64_t start          = 0;
                        for (int j = 0; j < vSparseCount[i]; j++)
                        {
                            vTempSparseIndex[vTempSparseEnd[i]]
                                                = vPeerSparseIndex[start];
                            if (!(_attributes & NNDataSetEnums::Boolean))
                            {
                                vTempSparseData[vTempSparseEnd[i]]
                                                = vPeerSparseData[start];
                            }
                            vTempSparseEnd[i]++;
                            start++;
                        }
                    }
                }
                _vSparseStart                   = vTempSparseStart;
                _vSparseEnd                     = vTempSparseEnd;
                _vSparseIndex                   = vTempSparseIndex;
                if (!(_attributes & NNDataSetEnums::Boolean))
                    _vSparseData                = vTempSparseData;

                // Reallocate GPU data
                _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vSparseIndex.size(), false, _bStreaming));
                _pbSparseStart->Upload(_vSparseStart.data());
                _pbSparseEnd->Upload(_vSparseEnd.data());
                _pbSparseIndex->Upload(_vSparseIndex.data());
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    _pbSparseData.reset(new GpuBuffer<T>((uint64_t)_vSparseData.size(), false, _bStreaming));
                    _pbSparseData->Upload(_vSparseData.data());
                }
            }
            else
            {
                // Send all data to master
                uint64_t size                   = _vSparseIndex.size();
                MPI_Send(vSparseCount.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    MPI_Send(_vSparseData.data(), size, getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            // Download all current data from all GPUs
            _pbData->Download(_vData.data());
            _pbData.reset();

            // Unshard
            if (getGpu()._id == 0)
            {
                vector<T> vTempData(_vData);
                _vData.resize(_uniqueExamples * _width);

                // Copy Local Shard
                uint32_t xmax                   = _width / getGpu()._numprocs;
                for (uint64_t i = 0; i < _uniqueExamples; i++)
                    for (uint64_t j = 0; j < xmax; j++)
                        _vData[i * _width + j]  = vTempData[i * xmax + j];


                // Receive data from remainder of processes
                for (int i = 1; i < getGpu()._numprocs; i++)
                {
                    int xmin                    = (i * _width) / getGpu()._numprocs;
                    xmax                        = ((i + 1) * _width) / getGpu()._numprocs;
                    int slice                   = xmax - xmin;
                    int size                    = _uniqueExamples * slice;
                    vTempData.resize(size);
                    MPI_Status status;
                    MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    for (int j = 0; j < _uniqueExamples; j++)
                        for (int k = 0; k < slice; k++)
                            _vData[j * _width + xmin + k]
                                                = vTempData[j * slice + k];
                }

                // Reallocate GPU data
                _pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());

            }
            else
            {
                // Send all data to master
                MPI_Send(_vData.data(), _vData.size(), getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
            }

        }
    }
    else if (_sharding == NNDataSetEnums::Data)
    {

    }
    _sharding = NNDataSetEnums::Sharding::None;

    // Allocate/Reallocate Indices
    if (_attributes & NNDataSetEnums::Indexed)
    {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }

    // Allocate/Reallocate Sparse Weights
    if (_attributes & NNDataSetEnums::Weighted)
    {
        _pbDataWeight.reset(new GpuBuffer<NNFloat>((uint64_t)_vDataWeight.size(), false, _bStreaming));
        _pbDataWeight->Upload(_vDataWeight.data());
    }

    return true;
}

template<typename T> bool NNDataSet<T>::Shard(NNDataSetEnums::Sharding sharding)
{
    // Skip if already sharded
    if (sharding == _sharding)
        return true;

    // Merge previously sharded data to process 0, undoing any existing sharding
    UnShard();

    // Shard data out to all processes
    if (sharding == NNDataSetEnums::Model)
    {
        _sharding                                       = NNDataSetEnums::Model;
        _minX                                           = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
        _maxX                                           = ((size_t)_width * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;
        if (_attributes & NNDataSetEnums::Sparse)
        {
            if (getGpu()._id == 0)
            {
                // Send sharded data to other processes
                printf("NNDataSet<T>::Shard: Model Sharding sparse dataset %s across all GPUs.\n", _name.c_str());
                for (size_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint32_t xmin                       = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax                       = ((size_t)_width * (i + 1)) / (size_t)getGpu()._numprocs;
                    vector<uint64_t> vLocalSparseStart(_uniqueExamples);
                    vector<uint64_t> vLocalSparseEnd(_uniqueExamples);
                    vector<uint32_t> vLocalSparseIndex;
                    vector<T> vLocalSparseData;
                    for (int j = 0; j < _uniqueExamples; j++)
                    {
                        vLocalSparseStart[j]            = vLocalSparseIndex.size();
                        for (uint64_t k = _vSparseStart[j]; k < _vSparseEnd[j]; k++)
                        {
                            if ((_vSparseIndex[k] >= xmin) && (_vSparseIndex[k] < xmax))
                            {
                                vLocalSparseIndex.push_back(_vSparseIndex[k] - xmin);
                                if (!(_attributes & NNDataSetEnums::Boolean))
                                {
                                    vLocalSparseData.push_back(_vSparseData[k]);
                                }
                            }
                        }
                        vLocalSparseEnd[j]              = vLocalSparseIndex.size();
                    }

                    // Broadcast index data to appropriate process
                    uint64_t size                       = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
                    if (!(_attributes & NNDataSetEnums::Boolean))
                    {
                        MPI_Datatype mpiType        = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                // Finally derive local shard
                vector<uint64_t> vTempSparseStart       = _vSparseStart;
                vector<uint64_t> vTempSparseEnd         = _vSparseEnd;
                vector<uint32_t> vTempSparseIndex       = _vSparseIndex;
                vector<T> vTempSparseData               = _vSparseData;
                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                for (uint32_t j = 0; j < _uniqueExamples; j++)
                {
                    _vSparseStart[j]                = _vSparseIndex.size();
                    for (uint64_t k = vTempSparseStart[j]; k < vTempSparseEnd[j]; k++)
                    {
                        if ((vTempSparseIndex[k] >= _minX) && (vTempSparseIndex[k] < _maxX))
                        {
                            _vSparseIndex.push_back(vTempSparseIndex[k]);
                            if (!(_attributes & NNDataSetEnums::Boolean))
                            {
                                _vSparseData.push_back(vTempSparseData[k]);
                            }
                        }
                    }
                    _vSparseEnd[j]                      = _vSparseIndex.size();
                }
            }
            else
            {
                // Receive sharded data from master process
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                _vSparseIndex.resize(size);
                MPI_Recv(_vSparseStart.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &status);
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    MPI_Datatype mpiType                = getMPIDataType(_dataType);
                    _vSparseData.resize(size);
                    MPI_Recv(_vSparseData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
                }
            }

            // Allocate GPU buffers and upload
            _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
            _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
            _pbSparseIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vSparseIndex.size(), false, _bStreaming));
            _pbSparseStart->Upload(_vSparseStart.data());
            _pbSparseEnd->Upload(_vSparseEnd.data());
            //for (int i = 0; i < 100; i++)
            //    printf("%6d %12d %12d\n", i, _vSparseStart[i], _vSparseEnd[i]);
            //exit(-1);
            _pbSparseIndex->Upload(_vSparseIndex.data());
            if (!(_attributes & NNDataSetEnums::Boolean))
            {
                _pbSparseData.reset(new GpuBuffer<T>((uint64_t)_vSparseData.size(), false, _bStreaming));
                _pbSparseData->Upload(_vSparseData.data());
            }
        }
        else
        {
            // Non-sparse data
            if (getGpu()._id == 0)
            {
                // Send sharded data to other processes
                printf("NNDataSet<T>::Shard: Model Sharding dataset %s across all GPUs.\n", _name.c_str());
                for (size_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint32_t xmin                       = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax                       = ((size_t)_width * (size_t)(i + 1)) / (size_t)getGpu()._numprocs;
                    uint32_t slice                      = xmax - xmin;
                    vector<T> vLocalData(_uniqueExamples * slice);
                    for (size_t j = 0; j < _uniqueExamples; j++)
                    {
                        for (size_t k = 0; k < slice; k++)
                            vLocalData[j * slice + k]
                                                        = _vData[j * _width + xmin + k];
                    }


                    // Broadcast index data to appropriate process
                    size_t size                         = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType                = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), _uniqueExamples * slice, mpiType, i, 0, MPI_COMM_WORLD);
                }

                // Finally derive local shard
                vector<T> vTempData                     = _vData;
                uint64_t xmax                           = _width / getGpu()._numprocs;
                _vData.resize(_uniqueExamples * xmax);
                for (uint64_t j = 0; j < _uniqueExamples; j++)
                {
                    for (uint64_t k = 0; k < xmax; k++)
                    {
                        _vData[j * xmax + k]            = vTempData[j * _width + k];
                    }
                }
            }
            else
            {
                // Receive sharded data from master process
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vData.resize(size);
                MPI_Datatype mpiType                    = getMPIDataType(_dataType);
                MPI_Recv(_vData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
            }


            // Allocate space then upload data to GPU memory
            _pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
            _pbData->Upload(_vData.data());
        }
    }
    else if (sharding == NNDataSetEnums::Data)
    {
        // Interleave examples
        _sharding                                       = NNDataSetEnums::Data;
        size_t segment                                  = _uniqueExamples / getGpu()._numprocs;
        size_t remainder                                = _uniqueExamples % getGpu()._numprocs;
        _localExamples                                  = segment + (remainder > getGpu()._id);

        if (_attributes & NNDataSetEnums::Sparse)
        {
            if (getGpu()._id == 0)
            {
                // Send sharded data to other processes
                printf("NNDataSet<T>::Shard: Data Sharding sparse dataset %s across all GPUs.\n", _name.c_str());
                for (size_t i = 1; i < getGpu()._numprocs; i++)
                {
                    size_t localExamples                = segment + (remainder > i);
                    vector<uint64_t> vLocalSparseStart(localExamples);
                    vector<uint64_t> vLocalSparseEnd(localExamples);
                    vector<uint32_t> vLocalSparseIndex;
                    vector<T> vLocalSparseData;
                    size_t position                     = i;
                    for (size_t j = position; j < _uniqueExamples; j+= getGpu()._numprocs)
                    {
                        vLocalSparseStart[j]            = vLocalSparseIndex.size();
                        for (size_t k = _vSparseStart[j]; k < _vSparseEnd[j]; k++)
                        {
                            vLocalSparseIndex.push_back(_vSparseIndex[k]);
                            if (!(_attributes & NNDataSetEnums::Boolean))
                            {
                                vLocalSparseData.push_back(_vSparseData[k]);
                            }
                        }
                        vLocalSparseEnd[j]              = vLocalSparseIndex.size();
                    }

                    // Broadcast index data to appropriate process
                    uint64_t size                       = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), localExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), localExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
                    if (!(_attributes & NNDataSetEnums::Boolean))
                    {
                        MPI_Datatype mpiType        = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                // Finally derive local shard
                vector<uint64_t> vTempSparseStart       = _vSparseStart;
                vector<uint64_t> vTempSparseEnd         = _vSparseEnd;
                vector<uint32_t> vTempSparseIndex       = _vSparseIndex;
                vector<T> vTempSparseData               = _vSparseData;
                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_localExamples);
                _vSparseEnd.resize(_localExamples);
                for (uint32_t j = 0; j < _uniqueExamples; j++)
                {
                    _vSparseStart[j]                = _vSparseIndex.size();
                    for (uint64_t k = vTempSparseStart[j]; k < vTempSparseEnd[j]; k++)
                    {
                        if ((vTempSparseIndex[k] >= _minX) && (vTempSparseIndex[k] < _maxX))
                        {
                            _vSparseIndex.push_back(vTempSparseIndex[k]);
                            if (!(_attributes & NNDataSetEnums::Boolean))
                            {
                                _vSparseData.push_back(vTempSparseData[k]);
                            }
                        }
                    }
                    _vSparseEnd[j]                      = _vSparseIndex.size();
                }



            }
            else
            {
                // Receive sharded data from master process

            }
        }
        else
        {
            if (getGpu()._id == 0)
            {
                // Send sharded data to other processes
                printf("NNDataSet<T>::Shard: Data Sharding dataset %s across all GPUs.\n", _name.c_str());
                for (size_t i = 1; i < getGpu()._numprocs; i++)
                {
                    size_t localExamples                = segment + (remainder > i);
                    vector<T> vLocalData(localExamples * _stride);
                    T* pData                            = vLocalData.data();
                    size_t position                     = i;
                    for (size_t j = position; j < _uniqueExamples; j+= getGpu()._numprocs)
                    {
                        memcpy(pData, &_vData[j * _stride], _stride * sizeof(T));
                        pData                          += _stride;
                    }

                    // Broadcast index data to appropriate process
                    size_t size                         = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType                = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), localExamples * _stride, mpiType, i, 0, MPI_COMM_WORLD);
                }

                // Finally, derive local segment
                vector<T> vLocalData(_localExamples * _stride);
                T* pData                                = vLocalData.data();
                size_t position                         = 0;
                for (size_t j = position; j < _uniqueExamples; j+= getGpu()._numprocs)
                {
                    memcpy(pData, &_vData[j * _stride], _stride * sizeof(T));
                    pData                              += _stride;
                }
                _vData.resize(_localExamples * _stride);
                _vData                                  = vLocalData;
            }
            else
            {
                // Receive sharded data from master process
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vData.resize(size);
                MPI_Datatype mpiType                = getMPIDataType(_dataType);
                MPI_Recv(_vData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
            }

            // Allocate space then upload data to GPU memory
            _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
            _pbData->Upload(_vData.data());
        }
    }

    // Allocate/Reallocate indices
    if (_attributes & NNDataSetEnums::Indexed)
    {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }
    return true;
}

// Saves data set to NetCDF file
template<typename T> bool NNDataSet<T>::SaveNetCDF(const string& fname)
{
    bool bResult                            = true;

    // Unshard data back to process 0 if necessary
    NNDataSetEnums::Sharding oldSharding    = _sharding;
    UnShard();

    // Now save data entirely from process 0
    if (getGpu()._id == 0)
    {
        bool bOpened                        = false;
        try
        {
            NcFile nfc(fname, NcFile::replace);
            bOpened                         = true;

            NcGroupAtt datasetsAtt          = nfc.putAtt("datasets", ncUint, 1);
            if (datasetsAtt.isNull())
            {
                throw NcException("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }

            bool bResult                    = WriteNetCDF(nfc, fname, 0);
            if (!bResult)
                throw NcException("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << endl;
            }
            else
            {
                cout << e.what() << endl;
            }
            bResult                         = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    // Restore original sharding
    Shard(oldSharding);

    return bResult;
}


// Saves data set to nth component of NetCDF file
template<typename T> bool NNDataSet<T>::WriteNetCDF(NcFile& nfc, const string& fname, const uint32_t n)
{
    bool bResult                            = true;
    try {
        if (getGpu()._id == 0)
        {
            string nstring                  = to_string(n);
            string vname                    = "name" + nstring;
            NcGroupAtt nameAtt              = nfc.putAtt(vname, _name);
            if (nameAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset name to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname                           = "attributes" + nstring;
            NcGroupAtt attributesAtt        = nfc.putAtt(vname, ncUint, _attributes);
            if (attributesAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset attributes to NetCDF file " + fname, __FILE__, __LINE__);
            }

            // Stubbed to numeric for now, will eventually be numeric, image, or audio descriptor
            vname                           = "kind" + nstring;
            NcGroupAtt kindAtt              = nfc.putAtt(vname, ncUint, 0);
            if (kindAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset kind to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname                           = "datatype" + nstring;
            NcGroupAtt datatypeAtt          = nfc.putAtt(vname, ncUint, _dataType);
            if (datatypeAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset type to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname                           = "dimensions" + nstring;
            NcGroupAtt dimensionsAtt        = nfc.putAtt(vname, ncUint, _dimensions);
            if (dimensionsAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset dimensions to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname                           = "width" + nstring;
            NcGroupAtt widthAtt             = nfc.putAtt(vname, ncUint, _width);
            if (widthAtt.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset width to NetCDF file " + fname, __FILE__, __LINE__);
            }

            if (_dimensions > 1)
            {
                vname                       = "height" + nstring;
                NcGroupAtt heightAtt        = nfc.putAtt(vname, ncUint, _height);
                if (heightAtt.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset height to NetCDF file " + fname, __FILE__, __LINE__);
                }

                if (_dimensions > 2)
                {
                    vname                   = "length" + nstring;
                    NcGroupAtt lengthAtt    = nfc.putAtt(vname, ncUint, _length);
                    if (lengthAtt.isNull())
                    {
                        throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset length to NetCDF file " + fname, __FILE__, __LINE__);
                    }
                }
            }

            vname                           = "uniqueExamplesDim" + nstring;
            NcDim uniqueExamplesDim         = nfc.addDim(vname, (size_t)_uniqueExamples);
            if (uniqueExamplesDim.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset unique example count to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname                           = "examplesDim" + nstring;
            NcDim examplesDim               = nfc.addDim(vname, (size_t)_examples);
            if (examplesDim.isNull())
            {
                throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset example count to NetCDF file " + fname, __FILE__, __LINE__);
            }


            if (_attributes & NNDataSetEnums::Sparse)
            {
                vname                       = "sparseDataDim" + nstring;
                NcDim sparseDataDim         = nfc.addDim(vname, _vSparseIndex.size());
                if (sparseDataDim.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset sparse datapoint count to NetCDF file " + fname, __FILE__, __LINE__);
                }

                vname                       = "sparseStart" + nstring;
                NcVar sparseStartVar        = nfc.addVar(vname, "uint", uniqueExamplesDim.getName());
                if (sparseStartVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset sparse start variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseStartVar.putVar(_vSparseStart.data());

                vname                       = "sparseEnd" + nstring;
                NcVar sparseEndVar          = nfc.addVar(vname, "uint", uniqueExamplesDim.getName());
                if (sparseEndVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset sparse end variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseEndVar.putVar(_vSparseEnd.data());

                vname                       = "sparseIndex" + nstring;
                NcVar sparseIndexVar        = nfc.addVar(vname, "uint64", sparseDataDim.getName());
                if (sparseIndexVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset sparse index variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseIndexVar.putVar(_vSparseIndex.data());

                // Write analog sparse values if present
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    vname                       = "sparseData" + nstring;
                    NcType sparseType           = getNetCDFDataType(_dataType);
                    NcVar sparseDataVar         = nfc.addVar(vname, sparseType.getName(), sparseDataDim.getName());
                    if (sparseDataVar.isNull())
                    {
                        throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to write dataset sparse data variable to NetCDF file " + fname, __FILE__, __LINE__);
                    }
                    sparseDataVar.putVar(_vSparseData.data());
                }
            }
            else
            {

            }

            // Writes weights if present
            if (_attributes & NNDataSetEnums::Weighted)
            {
                vname                       = "dataWeight" + nstring;
                NcVar DataWeightVar         = nfc.addVar(vname, "float", uniqueExamplesDim.getName());
                if (DataWeightVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::NNDataSet: Failed to write data weights to NetCDF file " + fname, __FILE__, __LINE__);
                }
                DataWeightVar.putVar(_vDataWeight.data());
            }

            // Save indices if indexed
            if (_attributes & NNDataSetEnums::Indexed)
            {
                vname                       = "index" + nstring;
                NcVar indexVar              = nfc.addVar(vname, "uint32", examplesDim.getName());
                if (indexVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to create dataset index variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                indexVar.putVar(_vIndex.data());
            }
        }
    }
    catch (NcException& e)
    {
        cout << e.what() << endl;
        bResult                             = false;
    }
    return bResult;
}

template<typename T> NNDataSet<T>::~NNDataSet()
{
}

bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataSet)
{
    bool bResult                            = true;

    // Unshard data back to process 0 if necessary
    vector<NNDataSetEnums::Sharding> vSharding(vDataSet.size());
    for (uint32_t i = 0; i < vDataSet.size(); i++)
    {
        vSharding[i]                        = vDataSet[i]->_sharding;
        vDataSet[i]->UnShard();
    }

    // Now save data entirely from process 0
    if (getGpu()._id == 0)
    {
        bool bOpened                        = false;
        try
        {
            NcFile nfc(fname, NcFile::replace);
            bOpened                         = true;


            NcGroupAtt datasetsAtt          = nfc.putAtt("datasets", ncUint, (unsigned int)vDataSet.size());
            if (datasetsAtt.isNull())
            {
                throw NcException("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }
            for (uint32_t i = 0; i < vDataSet.size(); i++)
            {
                bool bResult                = vDataSet[i]->WriteNetCDF(nfc, fname, i);
                if (!bResult)
                    throw NcException("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
            }
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << endl;
            }
            else
            {
                cout << e.what() << endl;
            }
            bResult                         = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    // Restore original sharding
    for (uint32_t i = 0; i < vDataSet.size(); i++)
    {
        vDataSet[i]->Shard(vSharding[i]);
    }

    return bResult;
}

vector<NNDataSetBase*> LoadNetCDF(const string& fname)
{
    vector<NNDataSetBase*> vDataSet;
    vector<NNDataSetEnums::DataType> vDataType;
    bool bResult                                = true;

    if (getGpu()._id == 0)
    {
        bool bOpened                            = false;
        try
        {
            NcFile rnc(fname.c_str(), NcFile::read);
            bOpened                             = true;

            // Determine # of data sets
            NcGroupAtt dataSetsAtt              = rnc.getAtt("datasets");
            if (dataSetsAtt.isNull())
            {
                throw NcException("NcException", "LoadNetCDF: No datasets count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t datasets;
            dataSetsAtt.getValues(&datasets);

            for (uint32_t i = 0; i < datasets; i++)
            {
                string nstring                  = std::to_string(i);
                string vname                    = "dataType" + nstring;
                NcGroupAtt dataTypeAtt          = rnc.getAtt(vname);
                if (dataTypeAtt.isNull())
                {
                      throw NcException("NcException", "LoadNetCDF: No " + vname + " attribute located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t dataType;
                dataTypeAtt.getValues(&dataType);
                switch (dataType)
                {
                    case NNDataSetEnums::UInt:
                    case NNDataSetEnums::Int:
                    case NNDataSetEnums::LLInt:
                    case NNDataSetEnums::ULLInt:
                    case NNDataSetEnums::Float:
                    case NNDataSetEnums::Double:
                    case NNDataSetEnums::RGB8:
                    case NNDataSetEnums::RGB16:
                    case NNDataSetEnums::UChar:
                    case NNDataSetEnums::Char:
                        vDataType.push_back((NNDataSetEnums::DataType)dataType);
                        break;

                    default:
                        printf("LoadNetCDF: Invalid data type in binary input file %s.\n", fname.c_str());
                }
            }
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "NcException: LoadNetCDF: Error opening NetCDF input file " << fname << endl;
            }
            else
            {
                cout << "Exception: " << e.what() << endl;
            }
            bResult                         = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    uint32_t size                           = vDataType.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    vDataType.resize(size);
    MPI_Bcast(vDataType.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);


    // Read data sets into vDataSet
    for (int i = 0; i < vDataType.size(); i++)
    {
        NNDataSetBase* pDataSet             = NULL;
        if (getGpu()._id == 0)
            cout << "LoadNetCDF: Loading " << vDataType[i] << " data set" << endl;
        switch (vDataType[i])
        {
            case NNDataSetEnums::UInt:
                pDataSet                    = new NNDataSet<uint32_t>(fname, i);
                break;

            case NNDataSetEnums::Int:
                pDataSet                    = new NNDataSet<long>(fname, i);
                break;

            case NNDataSetEnums::Float:
                pDataSet                    = new NNDataSet<float>(fname, i);
                break;

            case NNDataSetEnums::Double:
                pDataSet                    = new NNDataSet<double>(fname, i);
                break;

            case NNDataSetEnums::Char:
                pDataSet                    = new NNDataSet<char>(fname, i);
                break;

            case NNDataSetEnums::UChar:
            case NNDataSetEnums::RGB8:
                pDataSet                    = new NNDataSet<uint8_t>(fname, i);
                break;

            default:
                printf("LoadNetCDF: invalid dataset type in binary input file %s.\n", fname.c_str());
                getGpu().Shutdown();
                exit(-1);
        }
        vDataSet.push_back(pDataSet);
    }

    return vDataSet;
}
vector<NNDataSetBase*> LoadImageData(const string& fname) {}
vector<NNDataSetBase*> LoadCSVData(const string& fname) {}
vector<NNDataSetBase*> LoadJSONData(const string& fname) {}
vector<NNDataSetBase*> LoadAudioData(const string& name) {}
