/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "NNTypes.h"
#include "kernels.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace netCDF;
using namespace netCDF::exceptions;

void DumpP(const char *name, NNFloat *p, int stride) {
    cout << name << ":  ";
    vector<NNFloat> data(stride);
    cudaMemcpy(data.data(), p, stride*sizeof(NNFloat), cudaMemcpyDefault);
    for (auto i : data) {
        cout << i << ", ";
    }
    cout << endl;
}


// NNLayer functions

NNLayer::NNLayer(NNLayerDescriptor& d, uint32_t batch) :
_name(d._name),
_kind(d._kind),
_type(d._type),
_attributes(d._attributes),
_poolingFunction(d._poolingFunction),
_dataSet(d._dataSet),
_pDataSet(NULL),
_vSource(d._vSource),
_vSkip(d._vSkip),
_pbUnit(),
_pbDelta(),
_pbDropout(),
_pbDeltaBN(),
_pbScaleGradientBN(),
_pbScaleGradientVelocityBN(),
_pbBiasGradientBN(),
_pbBiasGradientVelocityBN(),
_pbUnitBN(),
_pbScaleBN(),
_pbBiasBN(),
_pbRunningMeanBN(),
_pbRunningVarianceBN(),
_pbSaveMeanBN(),
_pbSaveInvVarianceBN(),
_Nx(d._Nx),
_Ny(d._Ny),
_Nz(d._Nz),
_Nw(d._Nw),
_strideBN(0),
_dimensions(d._dimensions),
_weightInit(d._weightInit),
_weightInitScale(d._weightInitScale),
_biasInit(d._biasInit),
_kernelX(d._kernelX),
_kernelY(d._kernelY),
_kernelZ(d._kernelZ),
_kernelStrideX(d._kernelStrideX),
_kernelStrideY(d._kernelStrideY),
_kernelStrideZ(d._kernelStrideZ),
_kernelPaddingX(d._kernelPaddingX),
_kernelPaddingY(d._kernelPaddingY),
_kernelPaddingZ(d._kernelPaddingZ),
_kernelDimensions(d._kernelDimensions),
_weightNorm(d._weightNorm),
_deltaNorm(d._deltaNorm),
_pDropout(d._pDropout),
_activation(d._activation),
_oddBatch(0),
_bSparse(d._attributes & NNLayer::Attributes::Sparse),
_sparsenessPenalty_p(d._sparsenessPenalty_p),
_sparsenessPenalty_beta(d._sparsenessPenalty_beta),
_bDenoising(d._attributes & NNLayer::Attributes::Denoising),
_bFastSparse(false),
_bDirty(true),
_bnCalls(0),
_priority(-1),
_deltaUpdateCount(0),
_unitUpdateCount(0),
_batch(batch),
_localBatch(batch),
_RELUSlope(d._RELUSlope),
_ELUAlpha(d._ELUAlpha),
_SELULambda(d._SELULambda),
_bBatchNormalization(d._attributes & NNLayer::Attributes::BatchNormalization)
{
    _stride                         = _Nx * _Ny * _Nz * _Nw;
    if (_type == FullyConnected)
        _parallelization            = Model;
    else
        _parallelization            = Data;

    // Model parallel settings
    _minX                           = ((size_t)_Nx * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
    _maxX                           = ((size_t)_Nx * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;
    _localStride                    = (_maxX - _minX) * _Ny * _Nz * _Nw;
    _maxLocalStride                 = (((size_t)_Nx + getGpu()._numprocs - 1) / (size_t)getGpu()._numprocs) * _Ny * _Nz * _Nw;
    
    // Allocate cuDNN tensor data if convolutional or pooling layer
    if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus   = cudnnCreateTensorDescriptor(&_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _tensordescriptor");
        cudnnStatus                 = cudnnCreateTensorDescriptor(&_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _oddBatchTensordescriptor");        
    }
    
    if (_bBatchNormalization)
    {
        cudaError_t status;
        cudnnStatus_t cudnnStatus   = cudnnCreateTensorDescriptor(&_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _scaleBiasMeanVarDescBN");
        cudnnStatus                 = cudnnCreateTensorDescriptor(&_tensorDescriptorBN);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _tensordescriptorBN");

        if (_type == NNLayer::Type::Convolutional)
            _strideBN   = _Nz;
        else
            _strideBN   = _localStride;

        // Allocate all of the device memory for Batch Normalization
        // need to have this here rather than Allocate(), because that memory gets wiped on Refresh and other paths
        // but this Bn needs to stay, since it can be filled in via Load
        _pbScaleGradientBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbBiasGradientBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbScaleBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbBiasBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbRunningMeanBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbRunningVarianceBN.reset(new GpuBuffer<NNFloat>(_strideBN));

        _pbSaveMeanBN.reset(new GpuBuffer<NNFloat>(_strideBN));
        _pbSaveInvVarianceBN.reset(new GpuBuffer<NNFloat>(_strideBN));

        if (getGpu()._id == 0)
        {
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN scale diff for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN bias diff for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN scale for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN bias for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN running mean for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN running variance for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN saving mean for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
            printf("NNLayer::NNLayer: Allocating %" PRIu64 " bytes of BN saving InvVariance for layer %s\n", _strideBN * sizeof(NNFloat), _name.c_str());        
        }

        if (d._vScaleBN.size() != 0)
        {
            status = cudaMemcpy(_pbScaleBN->_pDevData, d._vScaleBN.data(), _strideBN * sizeof(NNFloat), cudaMemcpyHostToDevice);
        } else {
            vector<NNFloat> ones(_strideBN);
            for (int i=0; i<_strideBN; ++i)
                ones[i] = 1;
            status = cudaMemcpy(_pbScaleBN->_pDevData, ones.data(), _strideBN * sizeof(NNFloat), cudaMemcpyHostToDevice);
        }
        RTERROR(status, "NNLayer::NNLayer: cudaMemcpy failed on  _pbScaleBN");        
        if (d._vBiasBN.size() != 0)
        {
            status = cudaMemcpy(_pbBiasBN->_pDevData, d._vBiasBN.data(), _strideBN * sizeof(NNFloat), cudaMemcpyHostToDevice);
        } else {
            status = cudaMemset(_pbBiasBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        }
        RTERROR(status, "NNLayer::NNLayer: cudaMemcpy failed on  _pbBiasBN");        
        if (d._vRunningMeanBN.size() != 0)
        {
            status = cudaMemcpy(_pbRunningMeanBN->_pDevData, d._vRunningMeanBN.data(), _strideBN * sizeof(NNFloat), cudaMemcpyHostToDevice);
        } else {
            status = cudaMemset(_pbRunningMeanBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        }
        RTERROR(status, "NNLayer::NNLayer: cudaMemcpy failed on  _pbRunningMeanBN");        
        if (d._vRunningVarianceBN.size() != 0)
        {
            status = cudaMemcpy(_pbRunningVarianceBN->_pDevData, d._vRunningVarianceBN.data(), _strideBN * sizeof(NNFloat), cudaMemcpyHostToDevice);
        } else {
            status = cudaMemset(_pbRunningVarianceBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        }
        RTERROR(status, "NNLayer::NNLayer: cudaMemcpy failed on  _pbRunningVarianceBN");        

        // clear the others that are used by cuDNN
        status = cudaMemset(_pbScaleGradientBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        RTERROR(status, "NNLayer::NNLayer: cudaMemset failed on  _pbScaleGradientBN");        
        status = cudaMemset(_pbBiasGradientBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        RTERROR(status, "NNLayer::NNLayer: cudaMemset failed on  _pbBiasGradientBN");        
        status = cudaMemset(_pbSaveMeanBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        RTERROR(status, "NNLayer::NNLayer: cudaMemset failed on  _pbSaveMeanBN");        
        status = cudaMemset(_pbSaveInvVarianceBN->_pDevData, 0, _strideBN * sizeof(NNFloat));
        RTERROR(status, "NNLayer::NNLayer: cudaMemset failed on  _pbSaveInvVarianceBN");        
    }

    if (_type == NNLayer::Type::Pooling)
    {
        // Allocate cuDNN pooling descriptor for pooling layers that need them
        cudnnStatus_t cudnnStatus = cudnnCreatePoolingDescriptor(&_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create pooling descriptor");
        vector<int> vKernel(3);
        vector<int> vKernelPadding(3);
        vector<int> vKernelStride(3);
        vKernel[0]                  = _kernelX;
        vKernel[1]                  = _kernelY;
        vKernel[2]                  = _kernelZ;
        vKernelPadding[0]           = _kernelPaddingX;
        vKernelPadding[1]           = _kernelPaddingY;
        vKernelPadding[2]           = _kernelPaddingZ;
        vKernelStride[0]            = _kernelStrideX;
        vKernelStride[1]            = _kernelStrideY;
        vKernelStride[2]            = _kernelStrideZ;
        
        switch (_poolingFunction)
        {
            case PoolingFunction::Max:
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_PROPAGATE_NAN,
                                           _kernelDimensions,
                                           vKernel.data(),
                                           vKernelPadding.data(),
                                           vKernelStride.data());
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to set max pooling descriptor");
                break;

            case PoolingFunction::Average:
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_PROPAGATE_NAN,
                                           _kernelDimensions,
                                           vKernel.data(),
                                           vKernelPadding.data(),
                                           vKernelStride.data());
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to set average pooling descriptor");
                break;
                
            case PoolingFunction::LRN:
                cudnnStatus         = cudnnCreateLRNDescriptor(&_LRNDescriptor);
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create LRN descriptor");
                break;
        }
        
        // Special handling for non-cuDNN pooling layers     
    }
}

NNLayer::~NNLayer()
{
    Deallocate();
    // Deallocate cuDNN tensor data if convolutional or pooling layer
    if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        // Delete tensor descriptors
        cudnnStatus_t cudnnStatus       = cudnnDestroyTensorDescriptor(_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _tensorDescriptor");        
        cudnnStatus                     = cudnnDestroyTensorDescriptor(_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _oddBatchTensorDescriptor");  
    }

    if (_bBatchNormalization)
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyTensorDescriptor(_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _scaleBiasMeanVarDescBN");        
        cudnnStatus                     = cudnnDestroyTensorDescriptor(_tensorDescriptorBN);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _tensorDescriptorBN");        
        _pbScaleBN.reset();
        _pbBiasBN.reset();
        _pbScaleGradientBN.reset();
        _pbBiasGradientBN.reset();
        _pbRunningMeanBN.reset();
        _pbRunningVarianceBN.reset();
        _pbSaveMeanBN.reset();
        _pbSaveInvVarianceBN.reset();
    }
    
    // Delete pooling layer-specific stuff
    if (_type == NNLayer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyPoolingDescriptor(_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to destroy _poolingDescriptor");
        
        if (_poolingFunction == PoolingFunction::LRN)
        {
            // Delete LRN descriptor
            cudnnStatus_t cudnnStatus   = cudnnDestroyLRNDescriptor(_LRNDescriptor);
            CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _LRNDescriptor");
        }
    }
}

void NNLayer::Deallocate()
{
    if (getGpu()._id == 0)
        printf("NNLayer::Deallocate: Deallocating all data for layer %s\n", _name.c_str());

    _pbUnit.reset();
    _pbUnitBN.reset();
    _pbDelta.reset();
    _pbDeltaBN.reset();
    _pbDropout.reset();
    _pbBuffer1.reset();
    _pbBuffer2.reset();
    _pbScaleVelocityBN.reset();
    _pbScaleGradientVelocityBN.reset();
    _pbBiasVelocityBN.reset();
    _pbBiasGradientVelocityBN.reset();
}

bool NNLayer::GetUnits(vector<NNFloat>& vUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        // Resize output vector if necessary
        if (vUnit.size() < _stride)
        {
            vUnit.resize(_stride);
        }
    
        // Get unit data
        _pbUnit->Download(vUnit.data());
    }
    else
    {
        printf("NNLayer::GetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool NNLayer::GetUnits(NNFloat* pUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        // Check that pUnit is a valid pointer
        if (pUnit == NULL)
        {
            printf("NNLayer::GetUnits: Download pointer invalid.\n");
            bValid = false;
        }
        else
        {
            // Get unit data
            _pbUnit->Download(pUnit);
        }
    }
    else
    {
        printf("NNLayer::GetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool NNLayer::GetDeltas(vector<NNFloat>& vDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        // Resize output vector if necessary
        if (vDelta.size() < _stride)
        {
            vDelta.resize(_stride);
        }
    
        // Get delta data
        _pbDelta->Download(vDelta.data());
    }
    else
    {
        printf("NNLayer::GetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool NNLayer::GetDeltas(NNFloat* pDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        // Check that pDelta is a valid pointer
        if (pDelta == NULL)
        {
            printf("NNLayer::GetDeltas: Download pointer invalid.\n");
            bValid = false;
        }
        else
        {
            // Get unit data
            _pbDelta->Download(pDelta);
        }
    }
    else
    {
        printf("NNLayer::GetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool NNLayer::SetUnits(const vector<NNFloat>& vUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        // Resize output vector if necessary
        if (vUnit.size() < _stride)
        {
            printf("NNLayer::SetUnits: Input unit data too small to set all units.\n");
            bValid = false;
        }
    
        // Set unit data
        _pbUnit->Upload(vUnit.data());
    }
    else
    {
        printf("NNLayer::SetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}


bool NNLayer::SetDeltas(const vector<NNFloat>& vDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        // Resize output vector if necessary
        if (vDelta.size() < _stride)
        {
            printf("NNLayer::SetDeltas: Input delta data too small to set all deltas.\n");
            bValid = false;
        }
    
        // Set delta data
        _pbDelta->Upload(vDelta.data());
    }
    else
    {
        printf("NNLayer::SetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

cudnnTensorDescriptor_t NNLayer::getTensorDescriptor(uint32_t batch)
{
    // Return usual tensor descriptor if regular batch
    if (batch == _batch)
    {
        return _tensorDescriptor;
    }
    
    // Return odd batch for ends of epochs or resize
    // here for a one-shot
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus;
        vector<int> vDimensions(5, 1);
        vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                                             
                cudnnStatus         = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "NNLayer::getTensorDescriptor: Unable to set oddBatchTensorDescriptor");
        _oddBatch = batch;
    }

    return _oddBatchTensorDescriptor;
}

const string& NNLayer::GetName() const {
  return _name;
}

const string& NNLayer::GetDataSetName() const {
    return _dataSet;
}
NNLayer::Kind NNLayer::GetKind() const {
  return _kind;
}

NNLayer::Type NNLayer::GetType() const {
  return _type;
}

uint32_t NNLayer::GetAttributes() const {
  return _attributes;
}

NNDataSetBase* NNLayer::GetDataSet() const {
  return _pDataSet;
}

uint32_t NNLayer::GetNumDimensions() const {
  return _dimensions;
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t> NNLayer::GetDimensions() const
{
    return make_tuple(_Nx, _Ny, _Nz, _Nw);
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t> NNLayer::GetLocalDimensions() const
{
    return make_tuple(_maxX - _minX, _Ny, _Nz, _Nw);
}

tuple<uint32_t, uint32_t, uint32_t> NNLayer::GetKernelDimensions() const
{
    return make_tuple(_kernelX, _kernelY, _kernelZ);
}

tuple<uint32_t, uint32_t, uint32_t> NNLayer::GetKernelStride() const
{
    return make_tuple(_kernelStrideX, _kernelStrideY, _kernelStrideZ);
}


static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    vector<int> vDim(16);
    vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");    
    cout << "Tensor:   " << ndims << " dimensions" << endl;
    cout << "DataType: " << dataType << endl;
    for (int i = 0; i < ndims; i++)
        cout << i << " " << vDim[i] << " " << vStride[i] << endl;
    cout << endl;
    
}

void NNLayer::Allocate(bool validate)
{
    Deallocate();
    uint64_t size                   = (uint64_t)_maxLocalStride * (uint64_t)_localBatch; 
    
    // Special handlers for specific layer types (should break this out to multiple classes)    
    if ((_type == NNLayer::Type::Pooling) && (_poolingFunction == PoolingFunction::Cosine))
    {
        _vBuffer1.resize(size);
        _pbBuffer1.reset(new GpuBuffer<NNFloat>(size));
        if (getGpu()._id == 0)
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of auxilliary buffer 1 data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
        _vBuffer2.resize(size);
        _pbBuffer2.reset(new GpuBuffer<NNFloat>(size));
        if (getGpu()._id == 0)
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of auxilliary buffer 2 data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
    }
        
    // Set tensor descriptor if pooling or convolutional layer
    else if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus;
        vector<int> vDimensions(5, 1);
        vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _localBatch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                           
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "NNLayer::Allocate: Unable to set tensor descriptor");
        DumpTensor(_tensorDescriptor);
    }

    // Allocate hidden unit data for hidden and output layers and for non-sparse input layers
    if (!_bSparse || !_bFastSparse || (_kind != Input)
        || (_bSparse && (_kind == Input) && validate) // only for validation
    )
    {
        _vUnit.resize(size);
        _pbUnit.reset(new GpuBuffer<NNFloat>(size));
        if (getGpu()._id == 0)
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of unit data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
    }

    // Allocate delta data for non-input layers
    if (_kind != Input)
    {
        _vDelta.resize(size);
        _pbDelta.reset(new GpuBuffer<NNFloat>(size));
        if (getGpu()._id == 0)       
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of delta data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
        
        if (_bBatchNormalization)
        {
            _pbUnitBN.reset(new GpuBuffer<NNFloat>(size));
            _pbDeltaBN.reset(new GpuBuffer<NNFloat>(size));            
        }        
        
    }
    
    // Allocate dropout data if active
    if (_pDropout > (NNFloat)0.0)
    {
        _pbDropout.reset(new GpuBuffer<NNFloat>(size));
        if (getGpu()._id == 0)        
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of dropout data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
    } 
    _bDirty                         = false;
}

void NNLayer::SetBatch(uint32_t batch)
{
    if (batch != _batch)
    {
        _batch                      = batch;
        if (_parallelization == NNLayer::Parallelization::Data)
            _localBatch             = batch / getGpu()._numprocs;
        else
            _localBatch             = batch;
        _bDirty                     = true;
    }
}

void NNLayer::RefreshParallelization()
{
    uint32_t convolutionalInputs = 0;
    uint32_t fullyConnectedInputs = 0;
    uint32_t poolingInputs = 0;
    uint32_t convolutionalOutputs = 0;
    uint32_t fullyConnectedOutputs = 0;
    uint32_t poolingOutputs = 0;    
    
    // Count number of inputs and outputs of each type
    for (auto l : _vIncomingLayer)
    {
        switch (l->_type)
        {
            case NNLayer::Type::Pooling:
                poolingInputs++;
                break;
            
            case NNLayer::Type::FullyConnected:
                fullyConnectedInputs++;
                break;
                
            case NNLayer::Type::Convolutional:
                convolutionalInputs++;
                break;
        }
    }
    
    for (auto l : _vOutgoingLayer)
    {
        switch (l->_type)
        {
            case NNLayer::Type::Pooling:
                poolingOutputs++;
                break;
                
            case NNLayer::Type::FullyConnected:
                fullyConnectedOutputs++;
                break;
                
            case NNLayer::Type::Convolutional:
                convolutionalOutputs++;
                break;
        }
    }
    
    switch (_kind)
    {
        // Input layer parallelization based on outputs
        case NNLayer::Kind::Input:
            if (convolutionalOutputs > 0)
                _parallelization = NNLayer::Parallelization::Data;
            else
                _parallelization = NNLayer::Parallelization::Model;
            break;
    
        // Output layer parallelization based on inputs
        case NNLayer::Kind::Output:
            if (convolutionalInputs > 0)
                _parallelization = NNLayer::Parallelization::Data;
            else
                _parallelization = NNLayer::Parallelization::Model;
            break;
        
        // Hidden Layer based on convolution, pooling, or fully-connected type, with possible transition post-activation
        // and post-delta calculation if outputs are of the other type
        case NNLayer::Hidden:
            // Fully connected layers are always model-parallel with possible incoming transposition
            if (_type == NNLayer::Type::FullyConnected)
            {    
                _parallelization = NNLayer::Parallelization::Model;
                if (convolutionalOutputs > 0)
                    _bTransposeParallelization = true;
            }
            
            // Pooling layer based on inputs, with possible transition post-activation
            // and post-delta calculation if outputs are of the other type
            else if (_type == NNLayer::Type::Pooling)
            {
                if (convolutionalInputs > 0)
                {
                    _parallelization = NNLayer::Parallelization::Data;
                    if (fullyConnectedOutputs > 0)
                        _bTransposeParallelization = true;
                }
                else
                {
                    _parallelization = NNLayer::Parallelization::Model;
                    if (convolutionalOutputs > 0)
                        _bTransposeParallelization = true;                
                }
            }
            
            // Otherwise a convolution layer, data-parallel with possible incoming transposition
            else
            {
                _parallelization = NNLayer::Parallelization::Data;                
                 if (fullyConnectedOutputs > 0)
                    _bTransposeParallelization = true;
            }
            break;
    }
}

void NNLayer::RefreshState(NNNetwork* pNetwork, TrainingMode trainingMode, bool validate)
{
    if (_bDirty)
    {
        // First test for fast sparse kernel compatibility if sparse input layer
        _bFastSparse                = false;
        if ((_kind == Input) && (_pDataSet != NULL) && (_bSparse))
        {
            if (_pDataSet->_sparseDensity > (NNFloat)0.1)
            {
                 if (getGpu()._id == 0)
                    printf("NNLayer::RefreshState: Sparse density per (%.2f) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_sparseDensity, _name.c_str());                 
            }
            else
            {
                _bFastSparse        = true;
            }
        }
        
        // Determine parallelization strategy
        if (getGpu()._numprocs > 1)
            RefreshParallelization();

        Allocate(validate);
        
        if (_bBatchNormalization)
        {
            if (trainingMode != TrainingMode::SGD)
            {
                if (!_pbScaleVelocityBN)
                    _pbScaleVelocityBN.reset(new GpuBuffer<NNFloat>(_localStride));
                if (!_pbBiasVelocityBN)
                    _pbBiasVelocityBN.reset(new GpuBuffer<NNFloat>(_localStride));

                // Add additional buffers for AdaDelta and Adam
                if ((trainingMode == TrainingMode::AdaDelta) || (trainingMode == TrainingMode::Adam))
                {
                    if (!_pbScaleGradientVelocityBN)
                        _pbScaleGradientVelocityBN.reset(new GpuBuffer<NNFloat>(_localStride));
                    if (!_pbBiasGradientVelocityBN)
                        _pbBiasGradientVelocityBN.reset(new GpuBuffer<NNFloat>(_localStride));
                }
                else
                {
                    _pbScaleGradientVelocityBN.reset();
                    _pbScaleGradientVelocityBN.reset();
                }
            }
            else
            {
                _pbScaleVelocityBN.reset();
                _pbBiasVelocityBN.reset();
                _pbScaleGradientVelocityBN.reset();
                _pbBiasGradientVelocityBN.reset();
            }
        } 

        // Shard data set if necessary
        if ((_kind != Hidden) && (_pDataSet != NULL))
        {
            if (_parallelization == NNLayer::Parallelization::Model)
            {
                _pDataSet->Shard(NNDataSetEnums::Model);
            }
            else if (_parallelization == NNLayer::Parallelization::Data)
            {
                _pDataSet->Shard(NNDataSetEnums::Data);
            }
        }
        _bDirty                     = false;
    }

    // Turn on/off denoising if active for input layers
    if ((_kind == Input) && _pDataSet)
        _pDataSet->SetDenoising(_bDenoising);
        
    // Set up LRN descriptor if active
    if ((_type == NNLayer::Type::Pooling) && (_poolingFunction == PoolingFunction::LRN))
    {
        cudnnStatus_t status = cudnnSetLRNDescriptor(_LRNDescriptor,
                                                    pNetwork->_LRN_n,
                                                    pNetwork->_LRN_alpha,
                                                    pNetwork->_LRN_beta,
                                                    pNetwork->_LRN_k);
        CUDNNERROR(status, "NNLayer::RefreshState: unable to set LRN descriptor");
    }
}

void NNLayer::ClearUpdates()
{
    _unitUpdateCount                = 0;
    _deltaUpdateCount               = 0;
    _bnCalls                        = 0;
}

void NNLayer::LoadPredictionBatch(uint32_t position, uint32_t batch)
{

    if (_kind == Input)
    {
        if (!_bSparse)
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        else if (!_bFastSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void NNLayer::LoadTrainingBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            if (_bFastSparse)
            {
                if (_bDenoising)
                {
                    _pDataSet->CalculateSparseTransposedDenoisedMatrix(position, batch, this);
                }
                else
                {
                    _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
                }
            }
            else
            {
                if (_bDenoising)
                {
                    _pDataSet->LoadSparseDenoisedInputUnit(position, batch, _localStride, _pbUnit->_pDevData);    
                }
                else
                {
                    _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);  
                }               
            }
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            
            // Apply dropout if active
            if (_pDropout > (NNFloat)0.0)
                CalculateDropout(batch);    
        }
    }
}

void NNLayer::LoadValidationBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void NNLayer::GenerateDenoisingData()
{
    if (_pDataSet)
        _pDataSet->GenerateDenoisingData();
}

void NNLayer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining)
{
    
    // Will switch to class-based decision shortly once working
    switch (_type)
    {
        case FullyConnected:
            ForwardPropagateFullyConnected(position, batch, bTraining);
            break;
            
        case Convolutional:
            ForwardPropagateConvolutional(position, batch, bTraining);
            break;
            
        case Pooling:
            ForwardPropagatePooling(position, batch, bTraining);
            break;                        
        
    }
}
    
    
void NNLayer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{    
    // Single GPU is the simplest case
    if (getGpu()._numprocs == 1)
    {
        if (_kind != Input)
        {         
            // Initialize units to bias values
            switch (_vIncomingLayer.size())
            {
                case 0: // Only skip layers
                    cudaMemset(GetIncomingUnitBuffer(), 0, _stride * batch * sizeof(NNFloat));
                    break;
                    
                case 1:
                    kClearUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _stride, batch);
                    break; 
                    
                case 2:
                    kClearDualSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                   
                    
                case 3:
                    kClearTripleSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[2]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;      

                case 4:
                    kClearQuadSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[3]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                  
                    
                default:
                    if (getGpu()._id == 0)
                        printf("NNLayer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());          
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }
        
        
            const NNFloat sgemm_beta                = (NNFloat)1.0;
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                // Special case sparse input layers with sparse matrix * matrix kernel
                if (_vIncomingLayer[i]->_bFastSparse)
                {
                    NNFloat* pWeight                = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    if (bTraining && _vIncomingLayer[i]->_bDenoising)
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);
                }
                else      
                {
                    const NNFloat sgemm_alpha       = (NNFloat)1.0;
                    cublasStatus_t cstatus;
                    NNFloat* pA                     = _vIncomingLayer[i]->GetUnitBuffer();
                    NNFloat* pB                     = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    NNFloat* pC                     = GetIncomingUnitBuffer();
                    int m                           = batch;
                    int n                           = _localStride;
                    int k                           = _vIncomingLayer[i]->_stride;
                    int lda                         = _vIncomingWeight[i]->_bTransposed ? k : n;
                    int ldb                         = k;
                    int ldc                         = n;

                    cstatus                         =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    _vIncomingWeight[i]->_bTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    n,
                                                    m,
                                                    k,
                                                    &sgemm_alpha,
                                                    pB,
                                                    lda,
                                                    pA,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                    // Make sure matrix multiply succeeded
                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
            }

            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
            }
            
            // Perform batch normalization if active
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
           
            // Calculate activation
            CalculateActivation(batch);
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);             
       
#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif              
        }       
    }
    else // Multi-GPU
    {
        if (_kind != Input)
        {              
            // Calculate activations from incoming larger layers locally, then reduce the result
            // to the appropriate process, picking up contribution from each process
            if (_vIncomingLargerLayer.size() > 0)
            {
                NNFloat sgemm_beta                  = (NNFloat)0.0;
                for (uint32_t i = 0; i < _vIncomingLargerLayer.size(); i++)
                {
                    NNLayer* pInputLayer            = _vIncomingLargerLayer[i];
                    NNFloat* pWeight                = _vIncomingLargerWeight[i]->_bShared ? 
                                                      _vIncomingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingLargerWeight[i]->_pbWeight->_pDevData;                                           

                    if (pInputLayer->_bFastSparse)
                    {
                        if (bTraining && pInputLayer->_bDenoising)
                            pInputLayer->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                        else
                            pInputLayer->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                    }
                    else
                    {
                
                        // Calculate local SGEMM
                        const NNFloat sgemm_alpha   = (NNFloat)1.0;

                        NNFloat* pA                 = pWeight;
                        NNFloat* pB                 = pInputLayer->GetUnitBuffer();
                        NNFloat* pC                 = getGpu()._pNetwork->GetP2PSendBuffer();
                        int m                       = _stride;
                        int n                       = batch;
                        int k                       = pInputLayer->_localStride;
                        int lda                     = _stride;
                        int ldb                     = pInputLayer->_localStride;
                        int ldc                     = _stride;

                        cublasStatus_t cstatus      =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    m,
                                                    n,
                                                    k,
                                                    &sgemm_alpha,
                                                    pA,
                                                    lda,
                                                    pB,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                        // Make sure matrix multiply succeeded
                        if (cstatus != CUBLAS_STATUS_SUCCESS)
                        {
                            if (getGpu()._id == 0)
                                printf("NNLayer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            getGpu().Shutdown();
                            exit(-1);
                        }                                     
                    }
                    
                    // Accumulate subsequent calculations if active
                    sgemm_beta                      = (NNFloat)1.0;
                }
                //printf("FP %s IL UC %d\n", _name.c_str(), _unitUpdateCount);

                // Reduce output
                Reduce(batch, _stride, GetIncomingUnitBuffer(), _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }
            
            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _localStride);
            }            
                   
            // Add biases and calculate activations
            switch (_vIncomingLayer.size())
            {
                case 0: // Only skip layers
                    break;
                
                case 1:
                    kAddBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _localStride, batch);
                    break; 
                        
                case 2:
                    kAddDualBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, _localStride, batch);
                    break;                   
                        
                case 3:
                    kAddTripleBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[2]->_pbBias->_pDevData, _localStride, batch);
                    break;      

                case 4:
                    kAddQuadBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[3]->_pbBias->_pDevData, _localStride, batch);
                    break;                  
                        
                default:
                    if (getGpu()._id == 0)
                        printf("NNLayer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }    
            
            // Perform batch normalization if active
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
                                      
            // Calculate activation
            CalculateActivation(batch);   
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);  
        }
        
#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif                                      
        // Circulate activations to outgoing larger layers
        if (_vOutgoingLargerLayer.size() > 0)
        {  
        
            if (_bFastSparse)
            {
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    NNLayer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    NNFloat* pWeight            = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                    const NNFloat sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                    
                    if (bTraining && _bDenoising)
                        _pDataSet->CalculateSparseDenoisedZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _pDataSet->CalculateSparseZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);
                }
            }
            else
            {
        
                // Gather inputs to this layer
                Gather(batch, _stride, GetUnitBuffer(), _localStride);

                // Calculate contributions to all outgoing X(L)
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    NNLayer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    NNWeight* pWeight           = _vOutgoingLargerWeight[i];     
                    NNWeight* pSrcWeight        = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                    NNFloat* pA                 = pSrcWeight->_pbWeight->_pDevData;
                    NNFloat* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    NNFloat* pC                 = pOutputLayer->GetIncomingUnitBuffer();
                    
                    int m                       = pOutputLayer->_localStride;
                    int n                       = batch;
                    int k                       = _stride;
                    int lda                     = pOutputLayer->_localStride;
                    int ldb                     = _stride;
                    int ldc                     = pOutputLayer->_localStride;
                    const NNFloat sgemm_alpha   = 1.0;
                    const NNFloat sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
            
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::ForwardPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                        
                    // Increment unit update count
                   // printf("FP %s OL UC %d\n", _name.c_str(), pOutputLayer->_unitUpdateCount);
                    pOutputLayer->_unitUpdateCount++;
                }
            }
        }
    }
    
#if 0
    // REMOVE
    _pbUnit->Download(_vUnit.data());
    MPI_Barrier(MPI_COMM_WORLD);
    if (getGpu()._id == 0)
        cout << _name << " ";
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            for (auto f : _vUnit)
                printf("%8.4f ", f);
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cout << endl;
    exit(-1);
#endif    
}


void NNLayer::ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining)
{ 
    if (_kind != NNLayer::Kind::Input)
    {
        // Single GPU is the simplest case
        if (getGpu()._numprocs == 1)
        {
            NNFloat alpha                   = (NNFloat)1.0;
            NNFloat beta                    = (NNFloat)0.0;            
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                NNLayer* pLayer             = _vIncomingLayer[i];
                NNWeight* pWeight           = _vIncomingWeight[i]->_bShared ? 
                                              _vIncomingWeight[i]->_pSharedWeight : 
                                              _vIncomingWeight[i];

                cudnnStatus_t cudnnStatus   = cudnnConvolutionForward(getGpu()._cuDNNHandle,
                                                                      &alpha,
                                                                      pLayer->getTensorDescriptor(batch),
                                                                      pLayer->GetUnitBuffer(),
                                                                      pWeight->_convFilterDesc,
                                                                      pWeight->_pbWeight->_pDevData,
                                                                      pWeight->_convDesc,
                                                                      pWeight->_convFWAlgo,
                                                                      getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                      getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                      &beta,
                                                                      getTensorDescriptor(batch),
                                                                      GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnConvolutionForward Failed");
                                                                                 
                // All weights have their own biases, so don't used those from shared weight if so
                cudnnStatus                 = cudnnAddTensor(getGpu()._cuDNNHandle,
                                                             &alpha,
                                                             _vIncomingWeight[i]->_convBiasTensor,
                                                             _vIncomingWeight[i]->_pbBias->_pDevData,
                                                             &alpha,
                                                             getTensorDescriptor(batch),
                                                             GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnAddTensor Failed");
                beta                        = 1.0f;            
            }
            
            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
            }
            
            // Perform batch normalization if active
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, 1, 1);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining)
                {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_SPATIAL,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_SPATIAL,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),   // output
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardInference Failed");
                }
            }
           
            // Calculate activation
            CalculateActivation(batch);
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);             
        }       
    }
}

void NNLayer::ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining)
{ 
    if (_kind != NNLayer::Kind::Input)
    {
        NNFloat alpha                           = (NNFloat)1.0;
        NNFloat beta                            = (NNFloat)0.0;
        for (int i = 0; i < _vIncomingLayer.size(); i++)
        {
            NNLayer* pLayer                     = _vIncomingLayer[i];
            cudnnStatus_t cudnnStatus;
            switch (_poolingFunction)
            {
                case PoolingFunction::Max:
                case PoolingFunction::Average:             
                    cudnnStatus                 = cudnnPoolingForward(getGpu()._cuDNNHandle,
                                                                  _poolingDescriptor,
                                                                  &alpha,
                                                                  pLayer->getTensorDescriptor(batch),
                                                                  pLayer->GetUnitBuffer(),
                                                                  &beta,
                                                                  getTensorDescriptor(batch),
                                                                  GetIncomingUnitBuffer());
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagatePooling: cudnnPoolingForward Failed");
                    break;

                case PoolingFunction::LRN:
                    cudnnStatus                 = cudnnLRNCrossChannelForward(getGpu()._cuDNNHandle,
                                                                          _LRNDescriptor,
                                                                          CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                                          &alpha,
                                                                          pLayer->getTensorDescriptor(batch),
                                                                          pLayer->GetUnitBuffer(),
                                                                          &beta,
                                                                          getTensorDescriptor(batch),
                                                                          GetIncomingUnitBuffer());
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagatePooling: cudnnLRNCrossChannelForward Failed");                                                                              
                    break;

                case PoolingFunction::Cosine:
                    if (i >= 1)
                    {
                        NNLayer* p0Layer        = _vIncomingLayer[0];
                        uint32_t offset         = i - 1;
                        kCalculateCosine(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride, 
                                    GetIncomingUnitBuffer() + offset, 
                                    _pbBuffer1->_pDevData + offset, 
                                    _pbBuffer2->_pDevData + offset, 
                                    _localStride);
                    }
                    break;
                    
                case PoolingFunction::DotProduct:
                    if (i >= 1)
                    {
                        NNLayer* p0Layer        = _vIncomingLayer[0];
                        uint32_t offset         = i - 1;
                        kCalculateDotProduct(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride, 
                                    GetIncomingUnitBuffer() + offset, 
                                    _localStride);
                    }
                    break;
                    
                case PoolingFunction::Maxout:
                    // Will special case 4 or fewer sources into one pass, this will be for remainder of >4 sources
                    if (beta != (NNFloat)0.0)
                    {
                        kCalculateMaxout(pLayer->GetUnitBuffer(), batch * _localStride, GetIncomingUnitBuffer());
                    }
                    else
                    {
                        cudaError_t status     = cudaMemcpy(GetIncomingUnitBuffer(), pLayer->GetUnitBuffer(), batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
                        RTERROR(status, "NNLayer::ForwardPropagate: Error calling cudaMemcpy for maxout pooling.");
                    }
                    break;
                    
            }
            beta                            = (NNFloat)1.0;
        }

        // Copy data from incoming skip layers
        for (auto l : _vIncomingSkip)
        {
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
        }        
    }
}

void NNLayer::CalculateActivation(uint32_t batch)
{
    uint64_t size                   = (uint64_t)batch * (uint64_t)_localStride;
    switch (_activation)
    {
        case Sigmoid:
            kCalculateSigmoidActivation(GetUnitBuffer(), size);
            break;

        case Tanh:
            kCalculateTanhActivation(GetUnitBuffer(), size);
            break;

        case RectifiedLinear:
            kCalculateRELUActivation(GetUnitBuffer(), size);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUActivation(GetUnitBuffer(), size, _RELUSlope);
            break;
            
        case ExponentialLinear:
            kCalculateELUActivation(GetUnitBuffer(), size, _ELUAlpha);        
            break;
            
        case ScaledExponentialLinear:
            kCalculateSELUActivation(GetUnitBuffer(), size, _ELUAlpha, _SELULambda);        
            break;            

        case SoftMax:
            kCalculateSoftMaxActivation(GetUnitBuffer(), batch, _localStride);
            break;

        // Stub for no activation needed
        case Linear:
            break;
    }
}

void NNLayer::CalculateDropout(uint32_t batch)
{
    // Perform different dropouts depending on activation
    NNFloat lambda              = (_activation == ScaledExponentialLinear) ? _SELULambda : (NNFloat)1.0;
    NNFloat alpha               = -lambda * _ELUAlpha;
    NNFloat q                   = (NNFloat)1.0 - _pDropout;
    NNFloat a                   = (NNFloat)1.0 / sqrt(q + alpha * alpha * _pDropout * q);
    NNFloat b                   = -a * _pDropout * alpha;
    NNFloat target              = (_activation == Sigmoid) ? (NNFloat)0.5 : (NNFloat)0.0;


    
    switch (_activation)
    {
        case ExponentialLinear:
        case ScaledExponentialLinear:
            kCalculateScaledBiasedDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, alpha, a, b);
            break;
            
        default:
            kCalculateDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, target);
            break;
    }
}

NNFloat NNLayer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("NNLayer::CalculateError: Attempt to calculate error on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            return _pDataSet->CalculateL1Error(position, batch, _localStride, GetUnitBuffer());

        case L2:
            return _pDataSet->CalculateL2Error(position, batch, _localStride, GetUnitBuffer());
            
        case L2Hinge:
            return _pDataSet->CalculateL2HingeError(position, batch, _localStride, GetUnitBuffer());            

        case Hinge:
            return _pDataSet->CalculateHingeError(position, batch, _localStride, GetUnitBuffer());              

        case CrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else
                return _pDataSet->CalculateCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case ScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else        
                return _pDataSet->CalculateScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case DataScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
            {
                cout << "unsupported combination of activation with cost function" << endl;
                getGpu().Shutdown();
                exit(-1);
            }
            else
            {
                return _pDataSet->CalculateDataScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            }
    }
    
    return (NNFloat)0.0;
}

void NNLayer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("NNLayer::CalculateOutputDelta: Attempt to calculate output delta on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            _pDataSet->CalculateL1OutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case CrossEntropy:
            _pDataSet->CalculateCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case ScaledMarginalCrossEntropy:
            _pDataSet->CalculateScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case L2:
            _pDataSet->CalculateOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;
            
        case L2Hinge:
            _pDataSet->CalculateL2HingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;            

        case Hinge:
            _pDataSet->CalculateHingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;            

        case DataScaledMarginalCrossEntropy:
            _pDataSet->CalculateDataScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        default:
            cout << "Unsupported cost function" << endl;
            exit(2);
    }
    
    // Batch Norm?
    
    
    // Normalize deltas if desired
    if (_deltaNorm > (NNFloat)0.0)
    {
        if (getGpu()._numprocs == 1)
            kNormalizeDeltas(_deltaNorm, batch, _localStride, GetDeltaBuffer());
        else
        {
            NNFloat* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(batch);
            kCalculateDeltaMagnitudes(batch, _localStride, GetDeltaBuffer(), pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
            kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetDeltaBuffer(), pMagnitude);            
        }
    }
}


void NNLayer::BackPropagate(uint32_t position, uint32_t batch)
{
    
    // Will switch to class-based decision shortly once working
    switch (_type)
    {
        case FullyConnected:
            BackPropagateFullyConnected(position, batch);
            break;
            
        case Convolutional:
            BackPropagateConvolutional(position, batch);
            break;
            
        case Pooling:
            BackPropagatePooling(position, batch);
            break;                        
        
    }
}

void NNLayer::BackPropagateConvolutional(uint32_t position, uint32_t batch)
{
    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            // Normalize deltas if desired
            if (_deltaNorm > (NNFloat)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }
            
            // Calculate batch normalization gradients
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, 1, 1);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_SPATIAL,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,    // x desc
                        GetIncomingUnitBuffer(),   // x
                        _tensorDescriptorBN,    // dy dec
                        GetIncomingDeltaBuffer(),    // dy
                        _tensorDescriptorBN,    // dy dec
                        GetDeltaBuffer(),     // dx - output
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer:BackPropagateConvolutional cudnnBatchNormalizationBackward Failed");
            }
        }


        // Cycle through incoming layers and process gradient and delta contributions
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];

            NNWeight* pWeight                   = _vIncomingWeight[i];     
            NNWeight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
            NNFloat gradient_alpha              = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);            

            // Skip update if weights are locked
            cudnnStatus_t cudnnStatus;
            if (!pWeight->_bLocked)
            {
                NNFloat beta                    = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardFilter(getGpu()._cuDNNHandle,
                                                                                 &gradient_alpha,
                                                                                 pInputLayer->getTensorDescriptor(batch),
                                                                                 pInputLayer->GetUnitBuffer(),
                                                                                 getTensorDescriptor(batch),
                                                                                 GetDeltaBuffer(),
                                                                                 pSrcWeight->_convDesc,
                                                                                 pSrcWeight->_convBWWeightAlgo,
                                                                                 getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                                 getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                                 &beta,
                                                                                 pSrcWeight->_convFilterDesc,
                                                                                 pSrcWeight->_pbWeightGradient->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: cudnnConvolutionBackwardFilter Failed"); 
                
                // Biases are unshared, so overwrite any old gradient data
                beta                            = (NNFloat)0.0;
                cudnnStatus                     = cudnnConvolutionBackwardBias(getGpu()._cuDNNHandle,
                                                                           &gradient_alpha,
                                                                           getTensorDescriptor(batch),
                                                                           GetDeltaBuffer(),
                                                                           &beta,
                                                                           pWeight->_convBiasTensor,
                                                                           pWeight->_pbBiasGradient->_pDevData);                
                

                // Increment update count
                pSrcWeight->_updateCount++;
            }
     
            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                NNFloat delta_alpha             = (NNFloat)1.0;                
                NNFloat beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardData(getGpu()._cuDNNHandle,
                                                                               &delta_alpha,
                                                                               pSrcWeight->_convFilterDesc,
                                                                               pSrcWeight->_pbWeight->_pDevData,
                                                                               getTensorDescriptor(batch),
                                                                               GetDeltaBuffer(),
                                                                               pSrcWeight->_convDesc, 
                                                                               pSrcWeight->_convBWDeltaAlgo,
                                                                               getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                               getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                               &beta,
                                                                               pInputLayer->getTensorDescriptor(batch),
                                                                               pInputLayer->GetIncomingDeltaBuffer());
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: cudnnConvolutionBackwardData Failed");

                // Increment update count
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        // Copy deltas to incoming skip layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

void NNLayer::BackPropagatePooling(uint32_t position, uint32_t batch)
{
    // Special case single GPU
    {
        // Cycle through incoming layers and process gradient and delta contributions
        NNFloat pooling_alpha                   = (NNFloat)1.0;
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];

            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                cudnnStatus_t cudnnStatus;
                NNFloat beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                switch (_poolingFunction)
                {
                    case Max:
                    case Average:
                        cudnnStatus             = cudnnPoolingBackward(getGpu()._cuDNNHandle,
                                                                       _poolingDescriptor,
                                                                       &pooling_alpha,
                                                                       getTensorDescriptor(batch),
                                                                       GetUnitBuffer(),
                                                                       getTensorDescriptor(batch),
                                                                       GetDeltaBuffer(),
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->GetUnitBuffer(),
                                                                       &beta,
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->GetIncomingDeltaBuffer());                                                                         
                        CUDNNERROR(cudnnStatus, "NNLayer::BackPropagatePooling: cudnnPoolingBackward Failed");

                        // Increment update count
                        pInputLayer->_deltaUpdateCount++;                           
                        break;

                    case LRN:
                        cudnnStatus             = cudnnLRNCrossChannelBackward(getGpu()._cuDNNHandle,
                                                                                _LRNDescriptor,
                                                                                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                                                &pooling_alpha,
                                                                                getTensorDescriptor(batch),
                                                                                GetUnitBuffer(),
                                                                                getTensorDescriptor(batch),
                                                                                GetDeltaBuffer(),
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->GetUnitBuffer(),
                                                                                &beta,
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->GetIncomingDeltaBuffer());                      
                        CUDNNERROR(cudnnStatus, "NNLayer::BackPropagatePooling: cudnnLRNCrossChannelBackward Failed");

                        // Increment update count
                        pInputLayer->_deltaUpdateCount++;   
                        break;
                        
                    case Maxout:
                        kCalculateMaxoutDelta(GetUnitBuffer(), GetDeltaBuffer(), batch * _localStride, beta, pInputLayer->GetUnitBuffer(), pInputLayer->GetIncomingDeltaBuffer());
                        // Increment update count
                        pInputLayer->_deltaUpdateCount++;                         
                        break;

                    case Cosine:
                        if (i != 0)
                        {
                            NNLayer* p0Layer    = _vIncomingLayer[0];
                            NNFloat beta0       = (p0Layer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;                                        
                            uint32_t offset     = i - 1;
                            NNFloat* pDPIn      = GetUnitBuffer() + offset;
                            NNFloat* pDPDeltaIn = GetDeltaBuffer() + offset;                            
                            NNFloat* pAIn       = _pbBuffer1->_pDevData + offset;
                            NNFloat* pBIn       = _pbBuffer2->_pDevData + offset;
                            kCalculateCosineDelta(pDPDeltaIn, pDPIn, pAIn, pBIn, 
                            p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride, 
                            p0Layer->GetIncomingDeltaBuffer(), beta0, 
                            pInputLayer->GetIncomingDeltaBuffer(), beta, 
                            pInputLayer->_localStride);

                            // Increment update count
                            p0Layer->_deltaUpdateCount++;
                            pInputLayer->_deltaUpdateCount++; 
                        }                            
                        break;
                        
                    case DotProduct:
                        if (i != 0)
                        {
                            NNLayer* p0Layer    = _vIncomingLayer[0];
                            NNFloat beta0       = (p0Layer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;                                                 
                            uint32_t offset     = i - 1;
                            NNFloat* pDPDeltaIn = GetDeltaBuffer() + offset;
                            kCalculateDotProductDelta(pDPDeltaIn, p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride, 
                            p0Layer->GetIncomingDeltaBuffer(), beta0, 
                            pInputLayer->GetIncomingDeltaBuffer(), beta, 
                            pInputLayer->_localStride);

                            // Increment update count
                            p0Layer->_deltaUpdateCount++;
                            pInputLayer->_deltaUpdateCount++; 
                        }                            
                        break;                        

                }
            }
        }    
        
        // Copy deltas to incoming layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

// Calculates all contributions to Delta(t-1) or (Delta(t) * W(t-1->t)^T) which is the product of a [batch][stride] and [stride][outgoing stride] matrix
// And for efficiency purposes, the local contribution to dW(t-1->t), which is x(t-1)^T * Delta(t)
void NNLayer::BackPropagateFullyConnected(uint32_t position, uint32_t batch)
{    
    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            // Normalize deltas if desired (Norms must be reduced across all GPUs)
            if (_deltaNorm > (NNFloat)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }
            
            // Calculate batch normalization gradients
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_PER_ACTIVATION,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,    // x desc
                        GetIncomingUnitBuffer(),   // x
                        _tensorDescriptorBN,    // dy dec
                        GetIncomingDeltaBuffer(),  // dy
                        _tensorDescriptorBN,    // dy dec
                        GetDeltaBuffer(),   // dx - output
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }

#if 0
        if (_kind == Hidden)
        {
            string fname = "delta_" + _name;
            Dump(fname, _pbDelta->_pDevData);
        }
#endif 
        
        // Cycle through incoming layers and process gradient and delta contributions
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            NNWeight* pWeight                   = _vIncomingWeight[i];     
            NNWeight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            // Skip update if weights are locked
            if (!pWeight->_bLocked)
            {
                // Calculate weight gradients
                NNFloat* pDelta                 = GetDeltaBuffer();
                NNFloat* pUnit                  = pInputLayer->GetUnitBuffer();
                NNFloat* pA                     = pWeight->_bTransposed ? pDelta                    : pUnit;
                NNFloat* pB                     = pWeight->_bTransposed ? pUnit                     : pDelta;
                int m                           = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int n                           = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int ldb                         = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int ldc                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;

                // Update weights
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                
                if ((pInputLayer->_kind == NNLayer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else
                {
                    cstatus                 = cublasSgemm(getGpu()._cuBLASHandle, 
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              m,
                                              n,
                                              k,
                                              &sgemm_alpha,
                                              pB,
                                              lda,
                                              pA,
                                              ldb,
                                              &sgemm_beta,
                                              pC,
                                              ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
                
                // Increment update count
                pSrcWeight->_updateCount++;
            }
     
            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                NNFloat sgemm_alpha         = (NNFloat)1.0;
                NNFloat sgemm_beta          = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                int m                       = pInputLayer->_localStride;
                int n                       = batch;  
                
                
                NNFloat* pA                 = GetDeltaBuffer();
                NNFloat* pB                 = pWeight->_bShared ? 
                                              pSrcWeight->_pbWeight->_pDevData :
                                              pWeight->_pbWeight->_pDevData;

                NNFloat* pC                 = pInputLayer->GetIncomingDeltaBuffer();
                int k                       = _localStride;
                int lda                     = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb                     = k;
                int ldc                     = pInputLayer->_localStride;
                
                //printf("Delta between %s and %s %16.8f %16.8f\n", pInputLayer->_name.c_str(), _name.c_str(), sgemm_alpha, sgemm_beta);             
                
                //printf("%s: %d %d %d | %d %d %d\n", _name.c_str(), m, n, k, lda, ldb, ldc);
                cstatus                     = cublasSgemm(getGpu()._cuBLASHandle, 
                                            pWeight->_bTransposed ? CUBLAS_OP_N : CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            k,
                                            &sgemm_alpha,
                                            pB,
                                            lda,
                                            pA,
                                            ldb,
                                            &sgemm_beta,
                                            pC,
                                            ldc);   

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }
                
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        // Copy deltas to incoming layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
    else    // Multi-GPU case
    {
        // Process outgoing larger layers by gathering additional contributions to delta(L) and scattering X(L) to contribute
        // to dW(L->L+1)
        if (_vOutgoingLargerLayer.size() > 0)
        {
            // Gather X(L) on all GPUs to calculate contributions to all dW(L->L+1)
            Gather(batch, _stride, GetUnitBuffer(), _localStride);

            // Calculate contribution to weight gradients of each outgoing layer
            for (int i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                NNLayer* pOutputLayer           = _vOutgoingLargerLayer[i];
                NNWeight* pWeight               = _vOutgoingLargerWeight[i];     
                NNWeight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                // Calculate weight gradient contribution
                NNFloat* pA                     = pOutputLayer->GetDeltaBuffer();
                NNFloat* pB                     = getGpu()._pNetwork->GetP2PSendBuffer();
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = pOutputLayer->_localStride;
                int n                           = _stride;
                int k                           = batch;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = _stride;
                int ldc                         = pOutputLayer->_localStride;

                // Update weight gradients
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;               
                
                cublasStatus_t cstatus          = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }
                //printf("BP %s OL UW %d\n", _name.c_str(), pSrcWeight->_updateCount);

                // Increment update count
                pSrcWeight->_updateCount++;
            }  

            // Calculate contributions to Delta(L)            
            NNFloat sgemm_beta                  = (NNFloat)0.0;              
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                NNLayer* pOutputLayer           = _vOutgoingLargerLayer[i];
                const NNFloat sgemm_alpha       = (NNFloat)1.0;
                NNFloat* pA                     = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                NNFloat* pB                     = pOutputLayer->GetDeltaBuffer();
                NNFloat* pC                     = getGpu()._pNetwork->GetP2PSendBuffer();
                int m                           = _stride;
                int n                           = batch;
                int k                           = pOutputLayer->_localStride;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = pOutputLayer->_localStride;
                int ldc                         = _stride;

                cublasStatus_t cstatus          =
                                                cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);  

                // Make sure matrix multiply succeeded
                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                    getGpu().Shutdown();
                    exit(-1);
                }
#if 0
                NNFloat* pD = pOutputLayer->_vDelta.data();
                NNFloat* pW = _vOutgoingWeight[i]->_vWeight.data();
                
                pOutputLayer->_pbDelta->Download(pD);
                _vOutgoingLargerWeight[i]->_pbWeight->Download(pW);
                pW += pOutputLayer->_localStride;
                NNFloat sum = 0.0f;
                for (int j = 0; j < pOutputLayer->_localStride; j++)
                {
                    sum += (*pD) * (*pW);
                    pD++;
                    pW++;
                }
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                if (getGpu()._id == 0)
                    printf("ZAG %16.12f\n", sum);
                MPI_Barrier(MPI_COMM_WORLD);  
#endif
                        
                // Add subsequent layers
                sgemm_beta                      = (NNFloat)1.0;
            }

            //printf("BP %s OL UD %d\n", _name.c_str(), _deltaUpdateCount);

            // Reduce Delta(L)
            Reduce(batch, _stride, GetIncomingDeltaBuffer(), _localStride, _deltaUpdateCount);
            _deltaUpdateCount++;
        }


        
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);                
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            // Normalize deltas if desired (Norms must be reduced across all GPUs)
            if (_deltaNorm > (NNFloat)0.0)
            {            
                NNFloat* pMagnitude             = getGpu()._pNetwork->GetScratchBuffer(batch);
                kCalculateDeltaMagnitudes(batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
            }
            
            // Calculate batch normalization gradients if active
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_PER_ACTIVATION,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,    // x desc
                        GetIncomingUnitBuffer(),   // x
                        _tensorDescriptorBN,    // dy dec
                        GetIncomingDeltaBuffer(),  // dy
                        _tensorDescriptorBN,    // dy dec
                        GetDeltaBuffer(),   // dx - output
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }

        // Copy deltas to incoming layers that skip into this layer
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }          

        // Gather delta(L) to contribute to delta and dW of incoming larger layers
        if (_vIncomingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, GetDeltaBuffer(), _localStride);   
                 
            for (int i = 0; i < _vIncomingLargerLayer.size(); i++)
            {
                NNLayer* pInputLayer            = _vIncomingLargerLayer[i];
                NNWeight* pWeight               = _vIncomingLargerWeight[i];     
                NNWeight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                // Calculate weight gradient contribution
                NNFloat* pA                     = getGpu()._pNetwork->GetP2PSendBuffer();
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = _stride;
                int n                           = pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = _stride;
                int ldb                         = pInputLayer->_localStride;
                int ldc                         = _stride;

                // Update weight gradients
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                
                // Use sparse kernels if possible
                if ((pInputLayer->_kind == NNLayer::Kind::Input) && pInputLayer->_bFastSparse)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pA, pC);
                }
                else
                { 
                    NNFloat* pB                 = pInputLayer->GetUnitBuffer();          
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }

                //printf("BP %s IL UW %d\n", _name.c_str(), pSrcWeight->_updateCount);
                
                // Increment update count
                pSrcWeight->_updateCount++;
               
                // Calculate delta contribution if not input layer
                if (pInputLayer->_kind != Input)
                {
                    sgemm_alpha                 = (NNFloat)1.0;
                    sgemm_beta                  = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                    pA                          = pSrcWeight->_pbWeight->_pDevData;
                    NNFloat* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    pC                          = pInputLayer->GetIncomingDeltaBuffer();
                    m                           = pInputLayer->_localStride;
                    n                           = batch;
                    k                           = _stride;                           
                    lda                         = _stride;
                    ldb                         = _stride;
                    ldc                         = pInputLayer->_localStride;
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }

                    //printf("BP %s IL UD %d\n", _name.c_str(), pInputLayer->_deltaUpdateCount);
                    pInputLayer->_deltaUpdateCount++;
                }
            }
        }
    }
    
    
#if 0
    // Dump weight gradient
    NNWeight* pWeight                       = _vIncomingWeight[0];
    vector<NNFloat> vLocalWeightGradient(pWeight->_size);
    pWeight->_pbWeightGradient->Download(vLocalWeightGradient.data());
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            uint32_t count = 0;
            while (count < pWeight->_size)
            {
                for (int j = 0; j < pWeight->_outputLayer._stride; j++)
                {
                    printf("%8.4f ", vLocalWeightGradient[count++]);
                }
                printf("\n");
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }   
    if (getGpu()._id == 0)
        cout << endl;
    //getGpu().Shutdown();
    //exit(-1);
#endif   
}

void NNLayer::UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1, NNFloat t)
{
    if (_bBatchNormalization)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                
            case Momentum:
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case AdaGrad:
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;     

            case Adam:
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;   
        }
    }
}

// Reduces contributions from all GPUs to local component of X(L) or Delta(L)
void NNLayer::Reduce(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride, uint32_t updateCount)
{

    // Only valid for multi-GPU execution
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                             = getGpu()._numprocs - 1;
        NNFloat* pSendBuffer                        = getGpu()._pNetwork->GetP2PSendBuffer();

        if (getGpu()._bP2P)
        {

#if 0    
    {
    MPI_Barrier(MPI_COMM_WORLD);
    vector<NNFloat> vData(batch * stride);
    cudaMemcpy(vData.data(), getGpu()._pNetwork->GetP2PSendBuffer(), batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
    for (size_t n = 0; n < getGpu()._numprocs; n++)
    {
        if (n == getGpu()._id)
        {
            for (size_t i = 0; i < batch; i++)
            {
                printf("%3lu: ", i);
                {
                    for (size_t j = 0; j < stride; j++)
                    {
                        vData[i * stride + j] = getGpu()._id + 1;                        
                        printf("%8.6f ", vData[i * stride + j]);
                    }
                    printf("\n");
                }
            }
            printf("\n");            
        }
        fflush(stdout);        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cudaMemcpy(getGpu()._pNetwork->GetP2PSendBuffer(), vData.data(), batch * stride * sizeof(NNFloat), cudaMemcpyDefault);    
    MPI_Barrier(MPI_COMM_WORLD);
    }
#endif 

            // Initialize ring structures
            vector<NNFloat*> vpReceiveBuffer(getGpu()._vP2PRings.size());
            vector<NNFloat*> vpPeerBuffer(getGpu()._vP2PRings.size());
            vector<NNFloat*> vpSendBuffer(getGpu()._vP2PRings.size());
            vector<uint32_t> vBatch(getGpu()._vP2PRings.size());
            vector<uint32_t> vMinX(getGpu()._vP2PRings.size());
            vector<uint32_t> vMaxX(getGpu()._vP2PRings.size());
            vector<uint32_t> vSpan(getGpu()._vP2PRings.size());     
            vector<uint32_t> vPos(getGpu()._vP2PRings.size());
            
            for (size_t i = 0; i < getGpu()._vP2PRings.size(); i++)
            {
                uint32_t start = batch * getGpu()._vP2PRings[i].offset / getGpu()._totalP2PRank;
                uint32_t end = batch * (getGpu()._vP2PRings[i].offset + getGpu()._vP2PRings[i].rank) / getGpu()._totalP2PRank;
                vBatch[i] = end - start;
                vpReceiveBuffer[i] = getGpu()._pNetwork->GetP2PReceiveBuffer() + start * stride;
                vpPeerBuffer[i] = getGpu()._pNetwork->GetPeerBuffer(i) + start * stride;
                vpSendBuffer[i] = getGpu()._pNetwork->GetP2PSendBuffer() + start * stride;
                vPos[i] = (getGpu()._vP2PRings[i].position + 1) % getGpu()._numprocs;
                int process = getGpu()._vP2PRings[i].v[vPos[i]];
                vMinX[i] = (stride * process) / getGpu()._numprocs;
                vMaxX[i] = (stride * (process + 1)) / getGpu()._numprocs;
                vSpan[i] = vMaxX[i] - vMinX[i];
               // cout << "LR: " << getGpu()._id << " " << i << " | " << batch << " " << start << " " << end << " | " << process << " " << vMinX[i] << " " << vMaxX[i] << " " << vSpan[i] << endl; 
            }
            //MPI_Barrier(MPI_COMM_WORLD);
            //getGpu().Shutdown();
            //exit(-1);
                    
            // Send segments around the adding local contributions from each process
            for (uint32_t i = 0; i < stages; i++)
            {
                for (int j = 0; j < getGpu()._vP2PRings.size(); j++)
                {
                    kCopy2D(vpPeerBuffer[j] + vMinX[j], stride, vpSendBuffer[j] + vMinX[j], stride, vSpan[j], vBatch[j], getGpu()._vP2PRings[j].stream);
                }
                cudaDeviceSynchronize();       
                MPI_Barrier(MPI_COMM_WORLD);
        
                // Move to next position and add just arrived contribution
                for (int j = 0; j < getGpu()._vP2PRings.size(); j++)
                {
                    vPos[j]                                 = (vPos[j] + 1) % getGpu()._numprocs;
                    int process                             = getGpu()._vP2PRings[j].v[vPos[j]];
                    vMinX[j]                                = (stride * process) / getGpu()._numprocs;
                    vMaxX[j]                                = (stride * (process + 1)) / getGpu()._numprocs;
                    vSpan[j]                                = vMaxX[j] - vMinX[j];
                    kAddBuffers2D(vpSendBuffer[j] + vMinX[j], stride, vpReceiveBuffer[j] + vMinX[j], stride, vSpan[j], vBatch[j], getGpu()._vP2PRings[j].stream);
                }
            }
            cudaDeviceSynchronize();
        }
        else
        {
            // Download to system memory and use MPI to perform reduction
            NNFloat* pCPUBuffer                     = getGpu()._pNetwork->GetP2PCPUBuffer();
            cudaError_t status                      = cudaMemcpy(pCPUBuffer, pSendBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Reduce: cudaMemcpy download failed " + getGpu()._id );
            MPI_Allreduce(MPI_IN_PLACE, pCPUBuffer, batch * stride, MPI_NNFLOAT, MPI_SUM, MPI_COMM_WORLD);

            // Upload back to GPU memory
            status = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Reduce: cudaMemcpy upload failed" + getGpu()._id );           
        }

        // Copy data out to pBuffer
        uint32_t minX                               = (stride * getGpu()._id) / getGpu()._numprocs;
        uint32_t maxX                               = (stride * (getGpu()._id + 1)) / getGpu()._numprocs;
        uint32_t span                               = maxX - minX;        
        if (updateCount > 0) 
        {
            kAddBuffers2D(pBuffer, localStride, getGpu()._pNetwork->GetP2PSendBuffer() + minX, stride, span, batch);
        }
        else 
        {
            kCopy2D(pBuffer, localStride, getGpu()._pNetwork->GetP2PSendBuffer() + minX, stride, span, batch);
        }
        
#if 0        
        {
            MPI_Barrier(MPI_COMM_WORLD);
            vector<NNFloat> vData(batch * localStride);
            cudaMemcpy(vData.data(), pBuffer, batch * localStride * sizeof(NNFloat), cudaMemcpyDefault);
            for (size_t n = 0; n < getGpu()._numprocs; n++)
            {
                if (n == getGpu()._id)
                {
                    printf("RD %d | %d %d %d\n", getGpu()._id, minX, maxX, span);
                    for (size_t i = 0; i < batch; i++)
                    {
                        printf("%3lu: ", i);
                        {
                            for (size_t j = 0; j < localStride; j++)
                            {
                                printf("%8.6f ", vData[i * localStride + j]);
                            }
                            printf("\n");
                        }
                    }
                    printf("\n");
                }
                fflush(stdout);            
                MPI_Barrier(MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            getGpu().Shutdown();
            exit(-1);
        }
#endif         


#if 0    
    {
    MPI_Barrier(MPI_COMM_WORLD);
    vector<NNFloat> vData(batch * stride);
    cudaMemcpy(vData.data(), getGpu()._pNetwork->GetP2PSendBuffer(), batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
    for (size_t n = 0; n < getGpu()._numprocs; n++)
    {
        if (n == getGpu()._id)
        {
            for (size_t i = 0; i < batch; i++)
            {
                printf("%3lu: ", i);
                {
                    for (size_t j = 0; j < stride; j++)
                    {                      
                        printf("%8.6f ", vData[i * stride + j]);
                    }
                    printf("\n");
                }
            }
            printf("\n");            
        }
        fflush(stdout);        
        MPI_Barrier(MPI_COMM_WORLD);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
            getGpu().Shutdown();
            exit(-1);    
    }
#endif 

        
        
    }
}

// Copies all local components of X(L) or Delta(L) to all other GPUs
void NNLayer::Gather(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride)
{
    // Only valid for multi-GPU execution
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                                 = getGpu()._numprocs - 1;
        uint64_t process                                = getGpu()._id;
        NNFloat* pSendBuffer                            = getGpu()._pNetwork->GetP2PSendBuffer();
        uint32_t minX                                   = (stride * process) / getGpu()._numprocs;
        uint32_t maxX                                   = (stride * (process + 1)) / getGpu()._numprocs;
        uint32_t span                                   = maxX - minX;

#if 0        
        {
            MPI_Barrier(MPI_COMM_WORLD);
            vector<NNFloat> vData(batch * localStride);
            cudaMemcpy(vData.data(), pBuffer, batch * localStride * sizeof(NNFloat), cudaMemcpyDefault);
            for (size_t n = 0; n < getGpu()._numprocs; n++)
            {
                if (n == getGpu()._id)
                {
                    for (size_t i = 0; i < batch; i++)
                    {
                        printf("%3lu: ", i);
                        {
                            for (size_t j = 0; j < localStride; j++)
                            {
                                vData[i * localStride + j] = getGpu()._id + 1;
                                printf("%8.6f ", vData[i * localStride + j]);
                            }
                            printf("\n");
                        }
                    }
                    printf("\n");
                    fflush(stdout);            
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            cudaMemcpy(pBuffer, vData.data(), batch * localStride * sizeof(NNFloat), cudaMemcpyDefault);
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif        
        
        if (getGpu()._bP2P)
        {
            // Initialize ring structures
            vector<NNFloat*> vpPeerBuffer(getGpu()._vP2PRings.size());
            vector<NNFloat*> vpSendBuffer(getGpu()._vP2PRings.size());
            vector<uint32_t> vBatch(getGpu()._vP2PRings.size());
            vector<uint32_t> vMinX(getGpu()._vP2PRings.size());
            vector<uint32_t> vMaxX(getGpu()._vP2PRings.size());
            vector<uint32_t> vSpan(getGpu()._vP2PRings.size());   
            vector<uint32_t> vPos(getGpu()._vP2PRings.size());            
            
            for (size_t i = 0; i < getGpu()._vP2PRings.size(); i++)
            {
                uint32_t start = batch * getGpu()._vP2PRings[i].offset / getGpu()._totalP2PRank;
                uint32_t end = batch * (getGpu()._vP2PRings[i].offset + getGpu()._vP2PRings[i].rank) / getGpu()._totalP2PRank;
                vBatch[i] = end - start;
                vpPeerBuffer[i] = getGpu()._pNetwork->GetPeerBackBuffer(i) + start * stride;
                vpSendBuffer[i] = getGpu()._pNetwork->GetP2PSendBuffer() + start * stride;
                vPos[i] = getGpu()._vP2PRings[i].position;
                vMinX[i] = minX;
                vMaxX[i] = maxX;
                vSpan[i] = span;
            }

            // Insure Send Buffer is idle before commencing gather
            cudaDeviceSynchronize();  
            MPI_Barrier(MPI_COMM_WORLD);

            // Copy local segment to send buffer
            kCopy2D(pSendBuffer + minX, stride, pBuffer, localStride, span, batch); 
                       
            // Send segments around the adding local contributions from each process
            for (uint32_t i = 0; i < stages; i++)
            {
                for (int j = 0; j < getGpu()._vP2PRings.size(); j++)
                {
                    //printf("KCopy2D %2d %2d: %4d %4d %4d %4d\n", getGpu()._id, j, vMinX[j], vMaxX[j], vSpan[j], vBatch[j]);
                    kCopy2D(vpPeerBuffer[j] + vMinX[j], stride, vpSendBuffer[j] + vMinX[j], stride, vSpan[j], vBatch[j], getGpu()._vP2PRings[j].stream);
                }
                cudaDeviceSynchronize();       
                MPI_Barrier(MPI_COMM_WORLD);
        
                // Move to next position and add just arrived contribution
                for (int j = 0; j < getGpu()._vP2PRings.size(); j++)
                {
                    vPos[j]                                 = (vPos[j] + 1) % getGpu()._numprocs;                    
                    int process                             = getGpu()._vP2PRings[j].v[vPos[j]];
                    vMinX[j]                                = (stride * process) / getGpu()._numprocs;
                    vMaxX[j]                                = (stride * (process + 1)) / getGpu()._numprocs;
                    vSpan[j]                                = vMaxX[j] - vMinX[j];
                }
            }            
        }
        else
        {
            NNFloat* pCPUBuffer                        = getGpu()._pNetwork->GetP2PCPUBuffer();

            // Download local segment to system memory
            cudaError_t status                         = cudaMemcpy2D(pCPUBuffer + minX, stride * sizeof(NNFloat), pBuffer, localStride * sizeof(NNFloat), localStride * sizeof(NNFloat), batch, cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Gather: cudaMemcpy download failed");


            // use MPI_Bcast to scatter to all other processes
            for (uint32_t i = 0; i < getGpu()._numprocs; i++)
            {
                uint32_t minX                          = (stride * i) / getGpu()._numprocs;
                uint32_t maxX                          = (stride * (i + 1)) / getGpu()._numprocs;
                uint32_t span                          = maxX - minX;
                MPI_Datatype spanType;
                MPI_Type_vector(batch, span, stride, MPI_NNFLOAT, &spanType);
                MPI_Type_commit(&spanType);
                MPI_Bcast(pCPUBuffer + minX, 1, spanType, i, MPI_COMM_WORLD);
                MPI_Type_free(&spanType);
            }
 
            // Upload gathered buffer back to GPU memory
            status                                     = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Gather: cudaMemcpy upload failed");
        }
    }
#if 0    
    {
    MPI_Barrier(MPI_COMM_WORLD);
    vector<NNFloat> vData(batch * stride);
    cudaMemcpy(vData.data(), getGpu()._pNetwork->GetP2PSendBuffer(), batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
    for (size_t n = 0; n < getGpu()._numprocs; n++)
    {
        if (n == getGpu()._id)
        {
            for (size_t i = 0; i < batch; i++)
            {
                printf("%3lu: ", i);
                {
                    for (size_t j = 0; j < stride; j++)
                    {
                        printf("%8.6f ", vData[i * stride + j]);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            fflush(stdout);            
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    getGpu().Shutdown();
    exit(-1);
    }
#endif    
}

// Dumps unit or delta data to file
void NNLayer::Dump(string fname, NNFloat* pBuffer)
{   
    vector<NNFloat> vData(_batch * _stride);
    if (getGpu()._numprocs == 1) 
    {
        cudaMemcpy(vData.data(), pBuffer, _batch * _stride * sizeof(NNFloat), cudaMemcpyDefault);
    } 
    else 
    {
        if (getGpu()._id == 0)
        {
            NNFloat* pData              = vData.data();       
            cudaMemcpy2D(pData, _stride * sizeof(NNFloat), pBuffer, _localStride * sizeof(NNFloat), _localStride * sizeof(NNFloat), _batch, cudaMemcpyDefault);
            pData                      += _localStride;
            for (uint32_t i = 1; i < getGpu()._numprocs; i++)
            {                        
                uint64_t size;
                MPI_Status status;                
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                vector<NNFloat> vTemp(size);
                MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                uint64_t lstride    = size / _batch;
                NNFloat* pSrc = vTemp.data();
                NNFloat* pDst = pData;
                for (uint32_t j = 0; j < _batch; j++)
                {
                    memcpy(pDst, pSrc, lstride * sizeof(NNFloat));
                    pSrc               += lstride;
                    pDst               += _stride;
                }                          
                pData                  += lstride;
            }
        }
        else
        {
            uint64_t size               = _batch * _localStride;
            vector<NNFloat> vLocalData(size);
            cudaMemcpy(vLocalData.data(), pBuffer, size * sizeof(NNFloat), cudaMemcpyDefault);
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(vLocalData.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }
    }

    // Dump data to file from process 0
    if (getGpu()._id == 0)
    {
        FILE* fp                    = fopen(fname.c_str(), "w");
        NNFloat* pData              = vData.data();
        for (int i = 0; i < _batch; i++)
        {
            fprintf(fp, "%4d ", i);
            for (int j = 0; j < _stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}


std::pair<NNLayer::Kind, string> NNLayer::_sKindPair[] =
{
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Input,      "Input"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Hidden,     "Hidden"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Output,     "Output"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Target,     "Target"),    
};

std::map<NNLayer::Kind, string> NNLayer::_sKindMap =
std::map<NNLayer::Kind, string>(_sKindPair, _sKindPair + sizeof(_sKindPair) / sizeof(_sKindPair[0]));


std::pair<NNLayer::Type, string> NNLayer::_sTypePair[] =
{
    std::pair<NNLayer::Type, string>(NNLayer::Type::FullyConnected, "FullyConnected"),
    std::pair<NNLayer::Type, string>(NNLayer::Type::Convolutional,  "Convolutional"),
    std::pair<NNLayer::Type, string>(NNLayer::Type::Pooling,        "Pooling"),    
};

std::map<NNLayer::Type, string> NNLayer::_sTypeMap =
std::map<NNLayer::Type, string>(_sTypePair, _sTypePair + sizeof(_sTypePair) / sizeof(_sTypePair[0]));

std::pair<NNLayer::Attributes, string> NNLayer::_sAttributesPair[] =
{
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::None,               "None"),
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::Sparse,             "Sparse"),
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::Denoising,          "Denoising"),
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::BatchNormalization, "BatchNormalization"),
};

std::map<NNLayer::Attributes, string> NNLayer::_sAttributesMap =
std::map<NNLayer::Attributes, string>(_sAttributesPair, _sAttributesPair + sizeof(_sAttributesPair) / sizeof(_sAttributesPair[0]));

std::pair<NNLayer::Parallelization, string> NNLayer::_sParallelizationPair[] =
{
    
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Data,     "Data"),
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Model,    "Model"),
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Serial,   "Serial"),
};

std::map<NNLayer::Parallelization, string> NNLayer::_sParallelizationMap =
std::map<NNLayer::Parallelization, string>(_sParallelizationPair, _sParallelizationPair + sizeof(_sParallelizationPair) / sizeof(_sParallelizationPair[0]));


ostream& operator<< (ostream& out, NNLayer::Kind& k)
{
    out << NNLayer::_sKindMap[k];
    return out;
}
ostream& operator<< (ostream& out, NNLayer::Type& t)
{
    out << NNLayer::_sTypeMap[t];
    return out;
}

ostream& operator<< (ostream& out, NNLayer::Parallelization& p)
{
    out << NNLayer::_sParallelizationMap[p];
    return out;
}

ostream& operator<< (ostream& out, NNLayer::Attributes& a)
{
    out << NNLayer::_sAttributesMap[a];
    return out;
}




NNLayerDescriptor::NNLayerDescriptor() :
_kind(NNLayer::Kind::Hidden),
_type(NNLayer::Type::FullyConnected),
_poolingFunction(None),
_Nx(1),
_Ny(1),
_Nz(1),
_Nw(1),
_dimensions(1),
_bDimensionsProvided(true),
_weightInit(Xavier),
_weightInitScale((NNFloat)1.0),
_biasInit((NNFloat)0.0),
_kernelX(1),
_kernelY(1),
_kernelZ(1),
_kernelStrideX(1),
_kernelStrideY(1),
_kernelStrideZ(1),
_kernelPaddingX(0),
_kernelPaddingY(0),
_kernelPaddingZ(0),
_kernelDimensions(1),
_weightNorm((NNFloat)0.0),
_deltaNorm((NNFloat)0.0),
_pDropout((NNFloat)0.0),
_activation(Activation::Sigmoid),
_sparsenessPenalty_p((NNFloat)0.0),
_sparsenessPenalty_beta((NNFloat)0.0),
_RELUSlope(NAN),
_ELUAlpha(NAN),
_SELULambda(NAN),
_attributes(NNLayer::Attributes::None)
{

}

bool LoadNNLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNLayerDescriptor& ld)
{
    bool bResult                                = true; 

    if (getGpu()._id == 0)
    {
        try {
            string lstring                      = "layer" + std::to_string(index) + "_";
            NcGroupAtt nameAtt                  = nc.getAtt(lstring + "name");
            if (nameAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(ld._name);

            NcGroupAtt kindAtt                  = nc.getAtt(lstring + "kind");
            if (kindAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kind supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kindAtt.getValues(&ld._kind);

            NcGroupAtt typeAtt                  = nc.getAtt(lstring + "type");
            if (typeAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No type supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            typeAtt.getValues(&ld._type);
            
            NcGroupAtt poolingFunctionAtt       = nc.getAtt(lstring + "poolingfunction");
            if (poolingFunctionAtt.isNull())
            {
                if (ld._type == NNLayer::Type::Pooling)
                    throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No pooling function supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._poolingFunction             = None;
            }
            else
                poolingFunctionAtt.getValues(&ld._poolingFunction);

            NcGroupAtt dataSetAtt               = nc.getAtt(lstring + "dataSet");
            if (dataSetAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No dataSet supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dataSetAtt.getValues(ld._dataSet);

            NcGroupAtt NxAtt                    = nc.getAtt(lstring + "Nx");
            if (NxAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No Nx supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NxAtt.getValues(&ld._Nx);

            NcGroupAtt NyAtt                    = nc.getAtt(lstring + "Ny");
            if (NyAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No Ny supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NyAtt.getValues(&ld._Ny);

            NcGroupAtt NzAtt                    = nc.getAtt(lstring + "Nz");
            if (NzAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No Nz supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NzAtt.getValues(&ld._Nz);

            NcGroupAtt NwAtt                    = nc.getAtt(lstring + "Nw");
            if (NwAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No Nw supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NwAtt.getValues(&ld._Nw);

            NcGroupAtt dimensionsAtt            = nc.getAtt(lstring + "dimensions");
            if (dimensionsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No dimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dimensionsAtt.getValues(&ld._dimensions);

            NcGroupAtt kernelXAtt               = nc.getAtt(lstring + "kernelX");
            if (kernelXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelXAtt.getValues(&ld._kernelX);

            NcGroupAtt kernelYAtt               = nc.getAtt(lstring + "kernelY");
            if (kernelYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelYAtt.getValues(&ld._kernelY);

            NcGroupAtt kernelZAtt               = nc.getAtt(lstring + "kernelZ");
            if (kernelZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelZAtt.getValues(&ld._kernelZ);

            NcGroupAtt kernelStrideXAtt         = nc.getAtt(lstring + "kernelStrideX");
            if (kernelStrideXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelStrideX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideXAtt.getValues(&ld._kernelStrideX);

            NcGroupAtt kernelStrideYAtt         = nc.getAtt(lstring + "kernelStrideY");
            if (kernelStrideYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelStrideY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideYAtt.getValues(&ld._kernelStrideY);

            NcGroupAtt kernelStrideZAtt         = nc.getAtt(lstring + "kernelStrideZ");
            if (kernelStrideZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelStrideZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideZAtt.getValues(&ld._kernelStrideZ);


            NcGroupAtt kernelPaddingXAtt        = nc.getAtt(lstring + "kernelPaddingX");
            if (kernelPaddingXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelPaddingX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingXAtt.getValues(&ld._kernelPaddingX);

            NcGroupAtt kernelPaddingYAtt        = nc.getAtt(lstring + "kernelPaddingY");
            if (kernelPaddingYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelPaddingY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingYAtt.getValues(&ld._kernelPaddingY);

            NcGroupAtt kernelPaddingZAtt        = nc.getAtt(lstring + "kernelPaddingZ");
            if (kernelPaddingZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelPaddingZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingZAtt.getValues(&ld._kernelPaddingZ);          

            NcGroupAtt kernelDimensionsAtt      = nc.getAtt(lstring + "kernelDimensions");
            if (kernelDimensionsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No kernelDimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelDimensionsAtt.getValues(&ld._kernelDimensions);
            
            NcGroupAtt weightInitAtt            = nc.getAtt(lstring + "weightInit");
            if (weightInitAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No weightInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInit                  = Xavier;
            }
            else
                weightInitAtt.getValues(&ld._weightInit);      
            
            NcGroupAtt weightInitScaleAtt       = nc.getAtt(lstring + "weightInitScale");
            if (weightInitScaleAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No weightInitScale supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInitScale             = (NNFloat)1.0;
            }
            else
                weightInitScaleAtt.getValues(&ld._weightInitScale);   
                
            NcGroupAtt biasInitAtt              = nc.getAtt(lstring + "biasInit");
            if (biasInitAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No biasInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._biasInit                    = (NNFloat)0.0;
            }
            else
                biasInitAtt.getValues(&ld._biasInit);       
                      
            NcGroupAtt weightNormAtt            = nc.getAtt(lstring + "weightNorm");
            if (weightNormAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No weightNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightNorm                  = (NNFloat)0.0;
            }
            else
                weightNormAtt.getValues(&ld._weightNorm);
            
            NcGroupAtt deltaNormAtt             = nc.getAtt(lstring + "deltaNorm");
            if (deltaNormAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No deltaNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._deltaNorm                   = (NNFloat)0.0;
            }
            else
                deltaNormAtt.getValues(&ld._deltaNorm);
                
            NcGroupAtt pDropoutAtt              = nc.getAtt(lstring + "pDropout");
            if (pDropoutAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No pDropout supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            else
                pDropoutAtt.getValues(&ld._pDropout);

            NcGroupAtt activationAtt            = nc.getAtt(lstring + "activation");
            if (activationAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No activation supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            activationAtt.getValues(&ld._activation);

            // Version 0.85, RELU slope for leaky RELUs
            NcGroupAtt RELUSlopeAtt             = nc.getAtt(lstring + "RELUSlope");
            if (RELUSlopeAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No RELUSlope supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            RELUSlopeAtt.getValues(&(ld._RELUSlope));

            
            // Version 0.85, ELU alpha for ELUs
            NcGroupAtt ELUAlphaAtt              = nc.getAtt(lstring + "ELUAlpha");
            if (ELUAlphaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No ELUAlpha supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            ELUAlphaAtt.getValues(&(ld._ELUAlpha));
            
            // Version 0.85, SELU lambda for SELUs
            NcGroupAtt SELULambdaAtt            = nc.getAtt(lstring + "SELULambda");
            if (SELULambdaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No SELULambda supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SELULambdaAtt.getValues(&(ld._SELULambda)); 
            
            NcGroupAtt sparsenessPenalty_pAtt   = nc.getAtt("sparsenessPenalty_p");   
            if (sparsenessPenalty_pAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No sparsenessPenalty_p supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            else
            {
                sparsenessPenalty_pAtt.getValues(&(ld._sparsenessPenalty_p));
            }

            NcGroupAtt sparsenessPenalty_betaAtt= nc.getAtt("sparsenessPenalty_beta");
            if (sparsenessPenalty_betaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No sparsenessPenalty_beta supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._sparsenessPenalty_p = (NNFloat)0.0;
            }
            else
            {
                sparsenessPenalty_betaAtt.getValues(&(ld._sparsenessPenalty_beta));
            }

            NcGroupAtt attributesAtt            = nc.getAtt(lstring + "attributes");
            if (attributesAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            attributesAtt.getValues(&ld._attributes);

            // did this layer have BatchNorm data?
            if (ld._attributes & NNLayer::Attributes::BatchNormalization)
            {
                NcDim bnDim                 = nc.getDim(lstring + "bnDim");
                NcVar scaleBNVar            = nc.getVar(lstring + "scaleBN");
                NcVar biasBNVar             = nc.getVar(lstring + "biasBN");
                NcVar runningMeanBNVar      = nc.getVar(lstring + "runningMeanBN");
                NcVar runningVarianceBNVar  = nc.getVar(lstring + "runningVarianceBN");

                ld._vScaleBN.resize(bnDim.getSize());
                ld._vBiasBN.resize(bnDim.getSize());
                ld._vRunningMeanBN.resize(bnDim.getSize());
                ld._vRunningVarianceBN.resize(bnDim.getSize());

                scaleBNVar.getVar(ld._vScaleBN.data());
                biasBNVar.getVar(ld._vBiasBN.data());
                runningMeanBNVar.getVar(ld._vRunningMeanBN.data());
                runningVarianceBNVar.getVar(ld._vRunningVarianceBN.data());
            }

            // Read sources
            uint32_t sources                    = 0;
            NcGroupAtt sourcesAtt               = nc.getAtt(lstring + "sources");
            if (sourcesAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No sources supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            sourcesAtt.getValues(&sources);

            for (uint32_t i = 0; i < sources; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt sourceAtt            = nc.getAtt(lstring + "source" + nstring);
                if (sourcesAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No source attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string source;
                sourceAtt.getValues(source);
                ld._vSource.push_back(source);        
            }   
            
            // Read skips
            uint32_t skips                      = 0;
            NcGroupAtt skipsAtt                 = nc.getAtt(lstring + "skips");
            if (skipsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No skips supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            skipsAtt.getValues(&skips);

            for (uint32_t i = 0; i < skips; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt skipAtt              = nc.getAtt(lstring + "skip" + nstring);
                if (skipAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "NNLayer::NNLayer: No skip attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string skip;
                skipAtt.getValues(skip);
                ld._vSkip.push_back(skip);        
            }                    
        }
        catch (NcException& e)
        {
            cout << "Exception: " << e.what() << endl;
            bResult                             = false;
        }
    }
   
    return bResult;
}

ostream& operator<< (ostream& out, NNLayerDescriptor& d)
{
    out << "Name:                  " << d._name << endl;
    out << "Kind:                  " << d._kind << endl;
    out << "Type:                  " << d._type << endl;
    if (d._type != NNLayer::Type::Pooling)
        out << "Pooling Function:      " << d._poolingFunction << endl;
    out << "Nx:                    " << d._Nx << endl;
    out << "Ny:                    " << d._Ny << endl;
    out << "Nz:                    " << d._Nz << endl;
    out << "Nw:                    " << d._Nw << endl;
    if (d._type != NNLayer::Type::FullyConnected)
    {
        out << "kernelX:               " << d._kernelX << endl;
        out << "kernelY:               " << d._kernelY << endl;
        out << "kernelZ:               " << d._kernelZ << endl;
        out << "kernelStrideX:         " << d._kernelStrideX << endl;
        out << "kernelStrideY:         " << d._kernelStrideY << endl;
        out << "kernelStrideZ:         " << d._kernelStrideZ << endl;
        out << "kernelPaddingX:        " << d._kernelPaddingX << endl;
        out << "kernelPaddingY:        " << d._kernelPaddingY << endl;
        out << "kernelPaddingZ:        " << d._kernelPaddingZ << endl;
        out << "kernelDimensions:      " << d._kernelDimensions << endl;
    }
    if (d._type != NNLayer::Type::Pooling)
    {
        out << "pDropout:              " << d._pDropout << endl;
        out << "weightInit:            " << d._weightInit << endl;
        out << "weightInitScale:       " << d._weightInitScale << endl;
        out << "biasInit:              " << d._biasInit << endl;
        out << "weightNorm:            " << d._weightNorm << endl;
        out << "deltaNorm:             " << d._deltaNorm << endl;
        out << "activation:            " << d._activation << endl;
        out << "RELUSlope:             " << d._RELUSlope << endl;
        out << "ELUAlpha:              " << d._ELUAlpha << endl;
        out << "SELULambda:            " << d._SELULambda << endl; 
        out << "Sparse:                " << ((d._attributes & NNLayer::Attributes::Sparse) != 0) << endl;
        out << "batchNormalization:    " << ((d._attributes & NNLayer::Attributes::BatchNormalization) != 0) << endl;
        if (d._type == NNLayer::Type::FullyConnected)
        {
            if (d._sparsenessPenalty_p > (NNFloat)0.0)
                out << "sparsenessPenalty_p    " << d._sparsenessPenalty_p << endl;
            if (d._sparsenessPenalty_beta > (NNFloat)0.0)
                out << "sparsenessPenalty_beta " << d._sparsenessPenalty_beta << endl;
        }
        if (d._kind != NNLayer::Kind::Hidden)
            out << "DataSet:               " << d._dataSet << endl;
    }
    for (size_t i = 0 ; i < d._vSource.size(); i++)
    {
        out << "source " << setw(3) << i << ":            " << d._vSource[i] << endl;
    }
    for (size_t i = 0 ; i < d._vSkip.size(); i++)
    {
        out << "skip " << setw(3) << i << ":            " << d._vSkip[i] << endl;
    }     
    return out;
}

uint32_t MPI_Bcast_NNLayerDescriptor(NNLayerDescriptor& d)
{
    MPI_Bcast_string(d._name);
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._type, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._poolingFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&d._Nx, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Ny, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nw, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDimensionsProvided, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._pDropout, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInit, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInitScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._biasInit, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._activation, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&d._attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._RELUSlope, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._ELUAlpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SELULambda, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast_string(d._dataSet);
    size_t size                         = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);
    size                                = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);        
    return 0;
}

bool NNLayer::WriteNetCDF(NcFile& nc, uint32_t index)
{
    bool bResult                        = true;
    if (getGpu()._id == 0)
    {
        string lstring                  = "layer" + std::to_string(index) + "_";
        nc.putAtt(lstring + "name", _name);
        nc.putAtt(lstring + "kind", ncUint, _kind);
        nc.putAtt(lstring + "type", ncUint, _type);
        nc.putAtt(lstring + "poolingfunction", ncUint, _poolingFunction);
        nc.putAtt(lstring + "dataSet", _dataSet);
        nc.putAtt(lstring + "Nx", ncUint, _Nx);
        nc.putAtt(lstring + "Ny", ncUint, _Ny);
        nc.putAtt(lstring + "Nz", ncUint, _Nz);
        nc.putAtt(lstring + "Nw", ncUint, _Nw);
        nc.putAtt(lstring + "dimensions", ncUint, _dimensions);
        nc.putAtt(lstring + "kernelX", ncUint, _kernelX);
        nc.putAtt(lstring + "kernelY", ncUint, _kernelY);
        nc.putAtt(lstring + "kernelZ", ncUint, _kernelZ);
        nc.putAtt(lstring + "kernelDimensions", ncUint, _kernelDimensions);
        nc.putAtt(lstring + "kernelStrideX", ncUint, _kernelStrideX);
        nc.putAtt(lstring + "kernelStrideY", ncUint, _kernelStrideY);
        nc.putAtt(lstring + "kernelStrideZ", ncUint, _kernelStrideZ);
        nc.putAtt(lstring + "kernelPaddingX", ncUint, _kernelPaddingX);
        nc.putAtt(lstring + "kernelPaddingY", ncUint, _kernelPaddingY);
        nc.putAtt(lstring + "kernelPaddingZ", ncUint, _kernelPaddingZ);
        nc.putAtt(lstring + "pDropout", ncFloat, _pDropout);
        nc.putAtt(lstring + "weightInit", ncUint, _weightInit);
        nc.putAtt(lstring + "weightInitScale", ncFloat, _weightInitScale);
        nc.putAtt(lstring + "biasInit", ncFloat, _biasInit);
        nc.putAtt(lstring + "weightNorm", ncFloat, _weightNorm);
        nc.putAtt(lstring + "deltaNorm", ncFloat, _deltaNorm);
        nc.putAtt(lstring + "activation", ncUint, _activation);
        nc.putAtt(lstring + "sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
        nc.putAtt(lstring + "sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);
        nc.putAtt(lstring + "RELUSlope", ncFloat, _RELUSlope);
        nc.putAtt(lstring + "ELUAlpha", ncFloat, _ELUAlpha);
        nc.putAtt(lstring + "SELULambda", ncFloat, _SELULambda);
                
        uint32_t attributes             = 0;
        if (_bSparse)
            attributes                 |= NNLayer::Attributes::Sparse;
        if (_bDenoising)
            attributes                 |= NNLayer::Attributes::Denoising;
        if (_bBatchNormalization)
            attributes                 |= NNLayer::Attributes::BatchNormalization;
        nc.putAtt(lstring + "attributes", ncUint, attributes);
        nc.putAtt(lstring + "sources", ncUint, (uint32_t)_vSource.size());
        for (size_t i = 0; i < _vSource.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "source" + nstring, _vSource[i]);
        }
        nc.putAtt(lstring + "skips", ncUint, (uint32_t)_vSkip.size());        
        for (size_t i = 0; i < _vSkip.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
        }

        // append the BatchNorm data, if needed
        if (_bBatchNormalization)
        {
            vector<NNFloat>  bndata(_strideBN);
            size_t bytes = _strideBN * sizeof(NNFloat);
            NcDim bnDim   = nc.addDim(lstring + "bnDim", _strideBN);

            cudaMemcpy(bndata.data(), _pbScaleBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar scaleVar  = nc.addVar(lstring + "scaleBN", "float", bnDim.getName());
            scaleVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbBiasBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar biasVar  = nc.addVar(lstring + "biasBN", "float", bnDim.getName());
            biasVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningMeanBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningMeanVar  = nc.addVar(lstring + "runningMeanBN", "float", bnDim.getName());
            runningMeanVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningVarianceBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningVarianceVar  = nc.addVar(lstring + "runningVarianceBN", "float", bnDim.getName());
            runningVarianceVar.putVar(bndata.data());
        }
    }
    else
        bResult                     = false;

    return bResult;
}
