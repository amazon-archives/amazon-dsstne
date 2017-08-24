/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;


static std::pair<TrainingMode, string> sTrainingModePair[] =
{
    std::pair<TrainingMode, string>(TrainingMode::SGD,      "SGD"),
    std::pair<TrainingMode, string>(TrainingMode::Momentum, "Momentum"),
    std::pair<TrainingMode, string>(TrainingMode::AdaGrad,  "AdaGrad"),
    std::pair<TrainingMode, string>(TrainingMode::Nesterov, "Nesterov"),
    std::pair<TrainingMode, string>(TrainingMode::RMSProp,  "RMSProp"),
    std::pair<TrainingMode, string>(TrainingMode::AdaDelta, "AdaDelta"),  
    std::pair<TrainingMode, string>(TrainingMode::Adam,     "Adam"),  
};

static std::map<TrainingMode, string> sTrainingModeMap =
std::map<TrainingMode, string>(sTrainingModePair, sTrainingModePair + sizeof(sTrainingModePair) / sizeof(sTrainingModePair[0]));

ostream& operator<< (ostream& out, const TrainingMode& e)
{
    out << sTrainingModeMap[e];
    return out;
}

static std::pair<ErrorFunction, string> sErrorFunctionPair[] =
{
    std::pair<ErrorFunction, string>(ErrorFunction::L1,                             "L1"),
    std::pair<ErrorFunction, string>(ErrorFunction::L2,                             "L2"),
    std::pair<ErrorFunction, string>(ErrorFunction::CrossEntropy,                   "CrossEntropy"),
    std::pair<ErrorFunction, string>(ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy"),
    std::pair<ErrorFunction, string>(ErrorFunction::Hinge,                          "Hinge"),
};

static std::map<ErrorFunction, string> sErrorFunctionMap =
std::map<ErrorFunction, string>(sErrorFunctionPair, sErrorFunctionPair + sizeof(sErrorFunctionPair) / sizeof(sErrorFunctionPair[0]));

ostream& operator<< (ostream& out, const ErrorFunction& e)
{
    out << sErrorFunctionMap[e];
    return out;
}



static std::pair<Activation, string> sActivationPair[] =
{
    std::pair<Activation, string>(Activation::Sigmoid,                              "Sigmoid"),
    std::pair<Activation, string>(Activation::Tanh,                                 "Tanh"),
    std::pair<Activation, string>(Activation::Linear,                               "Linear"),
    std::pair<Activation, string>(Activation::ParametricRectifiedLinear,            "ParametricRectifiedLinear"),
    std::pair<Activation, string>(Activation::SoftSign,                             "SoftSign"),
    std::pair<Activation, string>(Activation::SoftPlus,                             "SoftPlus"),
    std::pair<Activation, string>(Activation::SoftMax,                              "SoftMax"),
    std::pair<Activation, string>(Activation::RELUMax,                              "RELUMax"),
    std::pair<Activation, string>(Activation::LinearMax,                            "LinearMax"),
    std::pair<Activation, string>(Activation::RectifiedLinear,                      "RectifiedLinear"),
    std::pair<Activation, string>(Activation::LeakyRectifiedLinear,                 "LeakyRectifiedLinear"),        
    std::pair<Activation, string>(Activation::ExponentialLinear,                    "ExponentialLinear"),
    std::pair<Activation, string>(Activation::ScaledExponentialLinear,              "ScaledExponentialLinear"),        
};

static std::map<Activation, string> sActivationMap =
std::map<Activation, string>(sActivationPair, sActivationPair + sizeof(sActivationPair) / sizeof(sActivationPair[0]));


ostream& operator<< (ostream& out, const Activation& a)
{
    out << sActivationMap[a];
    return out;
}

static std::pair<WeightInitialization, string> sWeightInitializationPair[] =
{
    std::pair<WeightInitialization, string>(WeightInitialization::Xavier,           "Xavier"),
    std::pair<WeightInitialization, string>(WeightInitialization::CaffeXavier,      "CaffeXavier"),
    std::pair<WeightInitialization, string>(WeightInitialization::Gaussian,         "Gaussian"),
    std::pair<WeightInitialization, string>(WeightInitialization::Uniform,          "Uniform"),
    std::pair<WeightInitialization, string>(WeightInitialization::UnitBall,         "UnitBall"),
    std::pair<WeightInitialization, string>(WeightInitialization::Constant,         "Constant"),
    std::pair<WeightInitialization, string>(WeightInitialization::SELU,             "SELU"),    
};
static std::map<WeightInitialization, string> sWeightInitializationMap =
std::map<WeightInitialization, string>(sWeightInitializationPair, sWeightInitializationPair + sizeof(sWeightInitializationPair) / 
sizeof(sWeightInitializationPair[0]));

ostream& operator<< (ostream& out, const WeightInitialization& w)
{
    out << sWeightInitializationMap[w];
    return out;
}

static std::pair<PoolingFunction, string> sPoolingFunctionPair[] =
{
    std::pair<PoolingFunction, string>(PoolingFunction::None,                       "None"),
    std::pair<PoolingFunction, string>(PoolingFunction::Max,                        "Max"),
    std::pair<PoolingFunction, string>(PoolingFunction::Average,                    "Average"),
    std::pair<PoolingFunction, string>(PoolingFunction::Maxout,                     "Maxout"),
    std::pair<PoolingFunction, string>(PoolingFunction::DotProduct,                 "DotProduct"),    
    std::pair<PoolingFunction, string>(PoolingFunction::Cosine,                     "Cosine"), 
    std::pair<PoolingFunction, string>(PoolingFunction::Stochastic,                 "Stochastic"),
    std::pair<PoolingFunction, string>(PoolingFunction::LCN,                        "LocalContrastNormalization"),
    std::pair<PoolingFunction, string>(PoolingFunction::LRN,                        "LocalResponseNormalization"),
    std::pair<PoolingFunction, string>(PoolingFunction::GlobalTemporal,             "GlobalTemporal"),
};

static std::map<PoolingFunction, string> sPoolingFunctionMap =
std::map<PoolingFunction, string>(sPoolingFunctionPair, sPoolingFunctionPair + sizeof(sPoolingFunctionPair) / sizeof(sPoolingFunctionPair[0]));


ostream& operator<< (ostream& out, const PoolingFunction& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}

ostream& operator<< (ostream& out, PoolingFunction& p);


static std::pair<NNDataSetEnums::Kind, string> sKindPair[] =
{
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Numeric, "Numeric"),
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Image,   "Image"),
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Audio,   "Audio")
};

static std::map<NNDataSetEnums::Kind, string> sKindMap =
std::map<NNDataSetEnums::Kind, string>(sKindPair, sKindPair + sizeof(sKindPair) / sizeof(sKindPair[0]));

ostream& operator<< (ostream& out, NNDataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}



static std::pair<NNDataSetEnums::Attributes, string> sAttributesPair[] =
{
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Sparse,                       "Sparse"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Boolean,                      "Boolean"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Compressed,                   "Compressed"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Recurrent,                    "Recurrent"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Mutable,                      "Mutable"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Attributes::SparseIgnoreZero, "SparseIgnoreZero"),
};

static std::map<NNDataSetEnums::Attributes, string> sAttributesMap =
std::map<NNDataSetEnums::Attributes, string>(sAttributesPair, sAttributesPair + sizeof(sAttributesPair) / sizeof(sAttributesPair[0]));


ostream& operator<< (ostream& out, NNDataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}


static std::pair<NNDataSetEnums::Sharding, string> sShardingPair[] =
{
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::None,  "None"),
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::Model, "Model"),
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::Data,  "Data")
};

static std::map<NNDataSetEnums::Sharding, string> sShardingMap =
std::map<NNDataSetEnums::Sharding, string>(sShardingPair, sShardingPair + sizeof(sShardingPair) / sizeof(sShardingPair[0]));

ostream& operator<< (ostream& out, NNDataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}

static std::pair<NNDataSetEnums::DataType, string> sDataTypePair[] =
{
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::UInt,   "UInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Int,    "Int"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::LLInt,  "LLInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::ULLInt, "ULLInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Float,  "Float"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Double, "Double"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::RGB8,  "RGB8"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::RGB16, "RGB16"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::UChar,  "UChar"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Char,   "Char"),
};

static std::map<NNDataSetEnums::DataType, string> sDataTypeMap =
std::map<NNDataSetEnums::DataType, string>(sDataTypePair, sDataTypePair + sizeof(sDataTypePair) / sizeof(sDataTypePair[0]));


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

NNDataSetBase::NNDataSetBase() :
_name(""),
_attributes(0),
_examples(0),
_dimensions(0),
_width(0),
_height(0),
_length(0),
_stride(0),
_sharding(NNDataSetEnums::None),
_minX(0),
_maxX(0),
_sparseDataSize(0),
_sparseTransposedIndices(0),
_maxSparseDatapoints(0),
_sparseDensity(0),
_bDenoising(false),
_pbSparseStart(),
_pbSparseEnd(),
_pbSparseIndex(),
_pbSparseTransposedStart(),
_pbSparseTransposedEnd(),
_pbSparseTransposedIndex(),
_batch(0),
_pbDenoisingRandom(),
_bStreaming(false),
_bDirty(true)
{


}

NNDataSetBase::~NNDataSetBase() {}

NNDataSetDimensions NNDataSetBase::GetDimensions()
{
    NNDataSetDimensions dim;
    dim._dimensions                             = _dimensions;
    dim._width                                  = _width;
    dim._height                                 = _height;
    dim._length                                 = _length;
    return dim;
}

template<typename T> vector<tuple<uint64_t, uint64_t> > NNDataSet<T>::getMemoryUsage()
{
    // Calculate per-process memory usage
    uint64_t cpuMemory                          = 0;
    uint64_t gpuMemory                          = 0;
    if (_attributes & NNDataSetEnums::Sparse)
    {
        cpuMemory                              += _examples * 2 * sizeof(uint64_t);
        gpuMemory                              += _examples * 2 * sizeof(uint64_t);
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
    
    // Gather and return memory usage per process
    vector<tuple<uint64_t, uint64_t> > vResult(getGpu()._numprocs);
    vResult[getGpu()._id]                       = make_tuple(cpuMemory, gpuMemory);   
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vResult.data(), sizeof(tuple<uint64_t, uint64_t>), MPI_BYTE, MPI_COMM_WORLD);
    return vResult;  
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

template<typename T> uint32_t NNDataSet<T>::GetSparseDataPoints(uint32_t n)
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

    // Make sure index is within bounds
    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        if (getGpu()._id == 0)
        {
            printf("NNDataSet::GetSparseIndex: Sparse index %u out of range (0, %u).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
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


    return _vSparseData[_vSparseStart[n] + i];
}

template<typename T> bool NNDataSet<T>::SetSparseDataPoint(uint32_t n, uint32_t i, T v)
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

    _vSparseData[_vSparseStart[n] + i]         = v;
    _bDirty                                    = true;
    return true;
}

template<typename T> NNDataSet<T>::NNDataSet(const string& fname, uint32_t n) :
_pbData(),
_pbSparseData(),
_pbSparseTransposedData()
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
                _vSparseStart.resize(examplesDim.getSize());
                _vSparseEnd.resize(examplesDim.getSize());
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
                    vector<uint32_t> vTempSparseStart(examplesDim.getSize());
                    sparseStartVar.getVar((uint32_t*)vTempSparseStart.data());
                    copy(vTempSparseStart.begin(), vTempSparseStart.end(), _vSparseStart.begin());
                }
                else
                    sparseStartVar.getVar((uint64_t*)_vSparseStart.data());
                    
                NcType vEndType                 = sparseEndVar.getType();    
                if (vEndType == ncUint)
                {
                    vector<uint32_t> vTempSparseEnd(examplesDim.getSize());
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
                        throw NcException("NcException", "NNDataSet::NNDataSet: No data dimensons located in NetCDF input file " + fname, __FILE__, __LINE__);
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
            cout << "NNDataSet<T>::NNDataSet: " << examplesDim.getSize() << " examples." << endl;
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
    MPI_Bcast(&_dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_width, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_height, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_sparseDataSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    
    
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
        std::fill(_vSparseDatapointCount.begin(), _vSparseDatapointCount.end(), 0);    
        for (auto x : _vSparseIndex)
        {
            // Check for boundary violation and stop before it corrupts CPU memory
            if (x >= _width)
            {
                if (getGpu()._id == 0)
                {
                    printf("NNDataSet::CalculateSparseDatapointCounts: Out of range index (%u) in sparse dataset %s.\n", x, _name.c_str());
                }
                getGpu().Shutdown();
                exit(-1);
            }
            _vSparseDatapointCount[x]++;
        }
        
        // Locate example with the highest datapoint count to test eligibility for forward SparseCalculateZ kernel
        _maxSparseDatapoints                    = 0;
        for (size_t i = 0; i < _vSparseStart.size(); i++)
        {
            uint64_t count                      = _vSparseEnd[i] - _vSparseStart[i];
            if (count > _maxSparseDatapoints) 
            {
                _maxSparseDatapoints            = count;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &_maxSparseDatapoints, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

        // Print warning message if too many datapoints for sparse kernels
        uint32_t maxSparse      = (_attributes & NNDataSetEnums::Boolean) ? getGpu()._maxSparse : getGpu()._maxSparseAnalog;
        if (_maxSparseDatapoints > maxSparse)
        {
            if (getGpu()._id == 0)
            {
                printf("NNDataSet::CalculateSparseDatapointCounts: Maximum sparse datapoints (%u) per example in dataset %s too large for fast sparse kernels.\n", _maxSparseDatapoints, _name.c_str());
            }
        }
        
        // Calculate sparse density
        _sparseDensity = (double_t)_sparseDataSize / (double_t)(_examples * N);
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
        
        offset                             += (batch < _vSparseDatapointCount[i]) ? batch : _vSparseDatapointCount[i];
        offset                              = ((offset + 31) >> 5) << 5;
    }
    _pbSparseTransposedStart->Upload(_vSparseTransposedStart.data());
        
    if (offset > _sparseTransposedIndices)
    {
        _sparseTransposedIndices            = offset;
        _pbSparseTransposedIndex.reset(new GpuBuffer<uint32_t>(_sparseTransposedIndices));
        if (!(_attributes & NNDataSetEnums::Boolean))
        {
            _pbSparseTransposedData.reset(new GpuBuffer<T>(_sparseTransposedIndices));
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
    if (getGpu()._bUnifiedMemory)
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
            vector<uint32_t> vSparseCount(_examples);
            for (uint32_t i = 0; i < _examples; i++)
            {
                vSparseCount[i]                 = _vSparseEnd[i] - _vSparseStart[i];
            }
            uint64_t datapoints                 = _vSparseIndex.size();
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : &datapoints, &datapoints, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vSparseCount.data(), vSparseCount.data(), _examples, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
            
            // Unshard
            if (getGpu()._id == 0)
            {
                vector<uint64_t> vTempSparseStart(_examples);
                vector<uint64_t> vTempSparseEnd(_examples);
                vector<uint32_t> vTempSparseIndex(datapoints);
                vector<T> vTempSparseData;
                if (!(_attributes & NNDataSetEnums::Boolean))
                    vTempSparseData.resize(datapoints);
                vTempSparseStart[0]             = 0;
                uint64_t start                  = 0;
                
                // Initialize counts and generate local shard
                for (int i = 0; i < _examples; i++)
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
                    MPI_Recv(vSparseCount.data(), _examples, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);
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
                    for (uint32_t i = 0; i < _examples; i++)
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
                _pbSparseStart.reset(new GpuBuffer<uint64_t>(_examples, false, _bStreaming));
                _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_examples, false, _bStreaming));
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
                MPI_Send(vSparseCount.data(), _examples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
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
                _vData.resize(_examples * _width);
                
                // Copy Local Shard                
                uint32_t xmax                   = _width / getGpu()._numprocs;
                for (uint64_t i = 0; i < _examples; i++)
                    for (uint64_t j = 0; j < xmax; j++)
                        _vData[i * _width + j]  = vTempData[i * xmax + j];


                // Receive data from remainder of processes
                for (int i = 1; i < getGpu()._numprocs; i++)
                {
                    int xmin                    = (i * _width) / getGpu()._numprocs;
                    xmax                        = ((i + 1) * _width) / getGpu()._numprocs;
                    int slice                   = xmax - xmin;
                    int size                    = _examples * slice;
                    vTempData.resize(size);
                    MPI_Status status;
                    MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    for (int j = 0; j < _examples; j++)
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
    _sharding = NNDataSetEnums::None;


    // Allocate/deallocate 

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
        _sharding                               = NNDataSetEnums::Model;
        _minX                                   = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
        _maxX                                   = ((size_t)_width * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;    
        if (_attributes & NNDataSetEnums::Sparse)
        {
            if (getGpu()._id == 0)
            {
                // Send sharded data to other processes
                printf("NNDataSet<T>::Shard: Model Sharding dataset %s across all GPUs.\n", _name.c_str());
                for (size_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint32_t xmin               = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax               = ((size_t)_width * (i + 1)) / (size_t)getGpu()._numprocs;
                    vector<uint64_t> vLocalSparseStart(_examples);
                    vector<uint64_t> vLocalSparseEnd(_examples);
                    vector<uint32_t> vLocalSparseIndex;
                    vector<T> vLocalSparseData;
                    for (int j = 0; j < _examples; j++)
                    {
                        vLocalSparseStart[j]    = vLocalSparseIndex.size();
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
                        vLocalSparseEnd[j]          = vLocalSparseIndex.size(); 
                    }

                    // Broadcast index data to appropriate process
                    uint64_t size                   = vLocalSparseIndex.size();

                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), _examples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), _examples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
                    if (!(_attributes & NNDataSetEnums::Boolean))
                    {
                        MPI_Datatype mpiType        = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                // Finally derive local shard
                vector<uint64_t> vTempSparseStart   = _vSparseStart;
                vector<uint64_t> vTempSparseEnd     = _vSparseEnd;
                vector<uint32_t> vTempSparseIndex   = _vSparseIndex;
                vector<T> vTempSparseData           = _vSparseData;
                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_examples);
                _vSparseEnd.resize(_examples);
                for (uint32_t j = 0; j < _examples; j++)
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
                    _vSparseEnd[j]                  = _vSparseIndex.size(); 
                }
            }
            else
            {
                // Receive sharded data from master process
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vSparseStart.resize(_examples);
                _vSparseEnd.resize(_examples);
                _vSparseIndex.resize(size);
                MPI_Recv(_vSparseStart.data(), _examples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseEnd.data(), _examples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &status); 
                if (!(_attributes & NNDataSetEnums::Boolean))
                {
                    MPI_Datatype mpiType            = getMPIDataType(_dataType);
                    _vSparseData.resize(size);
                    MPI_Recv(_vSparseData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
                }
            }

            // Allocate GPU buffers and upload
            _pbSparseStart.reset(new GpuBuffer<uint64_t>(_examples, false, _bStreaming));
            _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_examples, false, _bStreaming));
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
                    uint32_t xmin                   = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax                   = ((size_t)_width * (size_t)(i + 1)) / (size_t)getGpu()._numprocs;
                    uint32_t slice                  = xmax - xmin;
                    vector<T> vLocalData(_examples * slice);
                    for (uint64_t j = 0; j < _examples; j++)
                    {
                        for (uint64_t k = 0; k < slice; k++)
                            vLocalData[j * slice + k]   
                                                    = _vData[j * _width + xmin + k];
                    }


                    // Broadcast index data to appropriate process
                    uint64_t size                   = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType            = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), _examples * slice, mpiType, i, 0, MPI_COMM_WORLD);
                }

                // Finally derive local shard
                vector<T> vTempData                 = _vData;
                uint64_t xmax                       = _width / getGpu()._numprocs;
                _vData.resize(_examples * xmax);
                for (uint64_t j = 0; j < _examples; j++)
                {
                    for (uint64_t k = 0; k < xmax; k++)
                    {
                        _vData[j * xmax + k]        = vTempData[j * _width + k];
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
                MPI_Datatype mpiType                = getMPIDataType(_dataType);
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
        _sharding                                   = NNDataSetEnums::Data;
        uint32_t segment                            = _examples / getGpu()._numprocs;
        uint32_t remainder                          = _examples % getGpu()._numprocs;  
        _localExamples                              = segment + (remainder > getGpu()._id);         
        if (getGpu()._id == 0)
        {
            // Send sharded data to other processes
            printf("NNDataSet<T>::Shard: Data Sharding dataset %s across all GPUs.\n", _name.c_str());
            for (size_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint32_t localExamples          = segment + (remainder > i);
                vector<T> vLocalData(localExamples * _stride);
                T* pData                        = vLocalData.data();
                uint32_t position               = i;
                for (size_t j = position; j < _examples; j+= getGpu()._numprocs)
                {
                    memcpy(pData, &_vData[j * _stride], _stride * sizeof(T));
                    pData                      += _stride;
                }
                
                // Broadcast index data to appropriate process
                uint64_t size                   = vLocalData.size();
                MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                MPI_Datatype mpiType            = getMPIDataType(_dataType);
                MPI_Send(vLocalData.data(), localExamples * _stride, mpiType, i, 0, MPI_COMM_WORLD);
            }
            
            // Finally, derive local segment
            _vData.resize(_localExamples * _stride);
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
        _pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
        _pbData->Upload(_vData.data()); 
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
                NcVar sparseStartVar        = nfc.addVar(vname, "uint", examplesDim.getName());
                if (sparseStartVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to create dataset sparse start variable NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseStartVar.putVar(_vSparseStart.data());
                
                vname                       = "sparseEnd" + nstring;
                NcVar sparseEndVar          = nfc.addVar(vname, "uint", examplesDim.getName());
                if (sparseEndVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to create dataset sparse end variable NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseEndVar.putVar(_vSparseEnd.data());
 
                vname                       = "sparseIndex" + nstring;
                NcVar sparseIndexVar        = nfc.addVar(vname, "uint64", sparseDataDim.getName());
                if (sparseIndexVar.isNull())
                {
                    throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to create dataset sparse index variable NetCDF file " + fname, __FILE__, __LINE__);
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
                        throw NcException("NcException", "NNDataSet::WriteNetCDF: Failed to create dataset sparse data variable NetCDF file " + fname, __FILE__, __LINE__);
                    }               
                    sparseDataVar.putVar(_vSparseData.data());              
                }            
            }
            else
            {
            
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
