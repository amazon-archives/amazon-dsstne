/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace netCDF;
using namespace netCDF::exceptions;

NNWeightDescriptor::NNWeightDescriptor() :
_width(1),
_height(1),
_length(1),
_bShared(false),
_bTransposed(false),
_bLocked(false),
_norm((NNFloat)0.0)
{

}


bool LoadNNWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNWeightDescriptor& wd)
{
    bool bResult                                = true; 

    if (getGpu()._id == 0)
    {
        string wstring                          = "weight" + std::to_string(index) + "_";
        try {
            NcGroupAtt inputLayerAtt            = nc.getAtt(wstring + "inputLayer");
            if (inputLayerAtt.isNull())
            {
                throw NcException("NcException", "No input layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            inputLayerAtt.getValues(wd._inputLayer);  

            NcGroupAtt outputLayerAtt           = nc.getAtt(wstring + "outputLayer");
            if (outputLayerAtt.isNull())
            {
                throw NcException("NcException", "No output layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            outputLayerAtt.getValues(wd._outputLayer);

            NcGroupAtt normAtt                  = nc.getAtt(wstring + "norm");
            if (normAtt.isNull())
            {
                //throw NcException("NcException", "No norm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                wd._norm                        = (NNFloat)0.0;
            }
            else
                normAtt.getValues(&wd._norm);

            NcGroupAtt bSharedAtt               = nc.getAtt(wstring + "bShared");
            if (bSharedAtt.isNull())
            {
                throw NcException("NcException", "No bShared supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bShared;
            bSharedAtt.getValues(&bShared);
            wd._bShared                         = (bShared != 0);

            // Read shared weight attributes if _bShared is true
            if (wd._bShared)
            {
                NcGroupAtt sourceInputLayerAtt  = nc.getAtt(wstring + "sourceInputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw NcException("NcException", "No sourceInputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceInputLayerAtt.getValues(wd._sourceInputLayer);
                NcGroupAtt sourceOutputLayerAtt = nc.getAtt(wstring + "sourceOutputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw NcException("NcException", "No sourceOutputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceOutputLayerAtt.getValues(wd._sourceOutputLayer);
                NcGroupAtt bTransposedAtt       = nc.getAtt(wstring + "bTransposed");
                if (bTransposedAtt.isNull())
                {
                    throw NcException("NcException", "No bTransposed for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t bTransposed;
                bTransposedAtt.getValues(&bTransposed);
                wd._bTransposed                 = (bTransposed != 0);
            }

            NcGroupAtt bLockedAtt               = nc.getAtt(wstring + "bLocked");
            if (bLockedAtt.isNull())
            {
                wd._bLocked                     = false;
            }
            else
            {
                uint32_t bLocked;
                bLockedAtt.getValues(&bLocked);
                wd._bLocked                     = (bLocked != 0);
            }

            NcGroupAtt widthAtt                 = nc.getAtt(wstring + "width");
            if (widthAtt.isNull())
            {
                throw NcException("NcException", "No width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&wd._width);

            NcGroupAtt heightAtt                = nc.getAtt(wstring + "height");
            if (heightAtt.isNull())
            {
                throw NcException("NcException", "No height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            heightAtt.getValues(&wd._height);

            NcGroupAtt lengthAtt               = nc.getAtt(wstring + "length");
            if (lengthAtt.isNull())
            {
                throw NcException("NcException", "No length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            lengthAtt.getValues(&wd._length);

            // Read biases
            NcDim biasDim                       = nc.getDim(wstring + "biasDim");
            NcVar biasVar                       = nc.getVar(wstring + "bias");  
            wd._vBias.resize(biasDim.getSize()); 
            biasVar.getVar(wd._vBias.data());         

            if (!wd._bShared)
            {
                NcDim weightDim                 = nc.getDim(wstring + "weightDim");
                NcVar weightVar                 = nc.getVar(wstring + "weights");
                wd._vWeight.resize(weightDim.getSize()); 
                weightVar.getVar(wd._vWeight.data());
            }
#if 0
            printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
            for (int i = 0; i < 20; i++)
                printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
        }
        catch (NcException& e)
        {
            cout << "NNWeightDescriptor::NNWeightDescriptor: Exception: " << e.what() << endl;
            bResult                             = false;
        }

    }

    return bResult;
}

uint32_t MPI_Bcast_NNWeightDescriptor(NNWeightDescriptor& d)
{
    MPI_Bcast_string(d._inputLayer);
    MPI_Bcast_string(d._outputLayer);
    MPI_Bcast(&d._bShared, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bTransposed, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bLocked, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast_string(d._sourceInputLayer);
    MPI_Bcast_string(d._sourceOutputLayer);
    MPI_Bcast(&d._width, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._height, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._length, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    uint64_t weights                        = d._vWeight.size(); 
    MPI_Bcast(&weights, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    d._vWeight.resize(weights);
    MPI_Bcast(d._vWeight.data(), weights, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uint64_t biases                         = d._vBias.size();
    MPI_Bcast(&biases, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    d._vBias.resize(biases);
    MPI_Bcast(d._vBias.data(), biases, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return 0;
}

ostream& operator<< (ostream& out, NNWeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        out << "Input Layer:        " << d._inputLayer << endl;
        out << "Output Layer:       " << d._outputLayer << endl;
        out << "bShared:            " << std::boolalpha << d._bShared << endl;
        out << "bTransposed:        " << std::boolalpha << d._bTransposed << endl;
        if (d._bShared)
        {
            out << "sourceInputLayer:   " << d._sourceInputLayer << endl;
            out << "sourceOutputLayer:  " << d._sourceOutputLayer << endl;
        }
        out << "bLocked:            " << std::boolalpha << d._bLocked << endl;
        out << "norm:               " << d._norm << endl;
    }
    return out;
}


// An X vector is X[B][S] where B is that batch size and S is the stride of the layer
// X(L+1) is therefore X[OS] where OS is the outgoing stride from applying the weights
// to B discrete X vectors of size S
//
// Therefore, a weight matrix satisfies X(L+1) = X * W(L) will be W[S][OS]
NNWeight::NNWeight(NNLayer& inputLayer, NNLayer& outputLayer, bool bShared, bool bTransposed, bool bLocked, NNFloat norm) :
_inputLayer(inputLayer),
_outputLayer(outputLayer),
_sharingCount(1),
_updateCount(0),
_bShared(bShared),
_bTransposed(bTransposed),
_bLocked(bLocked),
_norm(norm),
_pSharedWeight(NULL),
_pbWeight(NULL),
_pbBias(NULL),
_pbWeightGradient(NULL),
_pbWeightVelocity(NULL),
_pbBiasVelocity(NULL)
{
    // Add to input and output layer lists
    inputLayer._vOutgoingLayer.push_back(&outputLayer);
    outputLayer._vIncomingLayer.push_back(&inputLayer);
    inputLayer._vOutgoingWeight.push_back(this);
    outputLayer._vIncomingWeight.push_back(this);
    
    // Add to Incoming Large or Outgoing larger
    uint32_t outgoingSize   = outputLayer._stride * 3;
    uint32_t incomingSize   = inputLayer._stride * 2;

    if (outgoingSize > incomingSize)
    {
        inputLayer._vOutgoingLargerLayer.push_back(&outputLayer);
        inputLayer._vOutgoingLargerWeight.push_back(this);
        _width                  = outputLayer._localStride;    
        _height                 = inputLayer._Nx;
    }
    else
    {
        outputLayer._vIncomingLargerLayer.push_back(&inputLayer);
        outputLayer._vIncomingLargerWeight.push_back(this);
        _width                  = outputLayer._Nx;
        _height                 = inputLayer._localStride;
    }
    _length                     = 1;
    _size                       = _width * _height * _length;
    


    if (!_bShared)
    {
        if (getGpu()._id == 0)
            printf("NNWeight::NNWeight: Allocating %" PRIu64 " bytes (%" PRIu64 ", %" PRIu64 ") for weights between %s and %s\n", _size * sizeof(float), _width, _height, inputLayer._name.c_str(), outputLayer._name.c_str());
        _vWeight.resize(_size);
        _pbWeight           = new GpuBuffer<NNFloat>((uint64_t)_size);
        _pbWeightGradient   = new GpuBuffer<NNFloat>((uint64_t)_size);        
    }

    _vBias.resize(outputLayer._localStride);
    _pbBias                 = new GpuBuffer<NNFloat>((uint64_t)outputLayer._localStride);
}

NNWeight::~NNWeight()
{
    if (!_bShared)
    {
        delete _pbWeight;
        delete _pbWeightVelocity;
        delete _pbWeightGradient;
    }
    delete _pbBias;
    delete _pbBiasVelocity;
}

void NNWeight::ClearVelocity()
{
    cudaMemset(_pbWeightVelocity->_pDevData, 0, _size * sizeof(NNFloat));
    cudaMemset(_pbBiasVelocity->_pDevData, 0, _outputLayer._localStride * sizeof(NNFloat));
}

void NNWeight::ClearGradient()
{
    cudaMemset(_pbWeightGradient->_pDevData, 0, _size * sizeof(NNFloat));
}

void NNWeight::Randomize()
{
    if (!_bShared)
    {
        NNFloat scale, bias;        
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            // Initialize weights to range from _weightInitScale * (-sqrt(3 / n_output) to sqrt(3 / n_output))
            // ala the adaptation of Gloriot and Bengio in Caffe
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);
            break;
            
        case Xavier:
            // Initialize weights to range from _weightInitScale * (-sqrt(6 / (n_output+n_input)) and sqrt(6 / (n_output+n_input)))
            // ala Gloriot and Bengio
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias                = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);
            break;
     
        case Uniform:
            // Initialize weights uniformly from -_weightInitScale to +_weightInitScale
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = 2.0f * _outputLayer._weightInitScale;
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);  
            break;
            
        case Gaussian:
            // Initialize weights to N(0, _weightInitScale)
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _size, 0.0f, _outputLayer._weightInitScale);
            break;        
            
        case UnitBall:      
            // Initialize weights uniformly from 0 to _weightInitScale  
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale;              
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, 0.0f);     
            break;
          
        case Constant:
            // Initialize all weights to _weightInitScale
            cudaMemset(_pbWeight->_pDevData, 0, _size * sizeof(NNFloat));
            kScaleAndBias(_pbWeight->_pDevData, _size, (NNFloat)0.0, _outputLayer._weightInitScale); 
            break;
        };
    }
        
    // Initialize Biases
    cudaMemset(_pbBias->_pDevData, 0, _outputLayer._localStride * sizeof(NNFloat));
    kScaleAndBias(_pbBias->_pDevData, _outputLayer._localStride, (NNFloat)0.0, -_outputLayer._biasInit); 
}

void NNWeight::Lock()
{
    _bLocked                = true;
}

void NNWeight::Unlock()
{
    _bLocked                = false;
}

void NNWeight::RefreshState(TrainingMode mode)
{
    if (mode != TrainingMode::SGD)
    {
        if (!_pbWeightVelocity)
            _pbWeightVelocity   = new GpuBuffer<NNFloat>(_size);
        if (!_pbBiasVelocity)
            _pbBiasVelocity     = new GpuBuffer<NNFloat>(_outputLayer._localStride);
    }
    else
    {
        delete _pbWeightVelocity;
        delete _pbBiasVelocity;
        _pbWeightVelocity       = NULL;
        _pbBiasVelocity         = NULL;
    }
}

float NNWeight::CalculateRegularizationError(NNFloat lambda)
{
    // Error on a shared set of weights is only calculated from its original source
    if (_bShared)
        return 0;
    else
        return kCalculateRegularizationError(lambda, _pbWeight->_pDevData, _size);
}

// Calculates Unit(l)^T * Delta(l + 1), the product of a [stride][batch] and [batch][outgoing stride] matrix
// and then updates weight values utilizing the current training mode
void NNWeight::UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat mu)
{
    cublasStatus_t cstatus;

    // Skip update if weights are locked
    if (_bLocked)
        return; 

    // Update weights if the original holder or unshared in general
    if (!_bShared)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(alpha * lambda, _size, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                
            case Momentum:
                kMomentumUpdateWeights(alpha * lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case AdaGrad:
                kAdaGradUpdateWeights(alpha, alpha * lambda, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(alpha * lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(alpha, alpha * lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
        }
    }  

    // Biases are unshared so always update them
    switch (trainingMode)
    {
       case SGD:
            kSGDUpdateBiases(-alpha / (NNFloat)batch, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBias->_pDevData);
            break;

       case Momentum:
            kMomentumUpdateBiases(alpha / (NNFloat)batch, mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
            break;
                
       case AdaGrad:
            kAdaGradUpdateBiases(alpha / (NNFloat)batch, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
            break;
                
       case Nesterov:
            kNesterovUpdateBiases(alpha / (NNFloat)batch, mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
            break;
                
       case RMSProp:
            kRMSPropUpdateBiases(alpha / (NNFloat)batch, mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
            break;
      
    }
#if 0
        if (_width < 1024)
        {
            _pbBias->Download(&_vBias[0]);
            for (int i = 0; i < _width; i++)
                printf("%3d %16.8f\n", i, _vBias[i]);
        }
#endif
          
    // and only do so after all updates have been applied    
    if ((_norm > (NNFloat)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        else
        {
            NNFloat* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);       
        }
    }
}

bool NNWeight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, NNFloat* pWeight, NNFloat* pBias)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        string wstring          = "weight" + std::to_string(index) + "_";
        nc.putAtt(wstring + "inputLayer", _inputLayer._name);
        nc.putAtt(wstring + "outputLayer", _outputLayer._name);
        nc.putAtt(wstring + "width", ncUint64, (unsigned long long int)_outputLayer._stride);  
        nc.putAtt(wstring + "height", ncUint64, (unsigned long long int)_inputLayer._stride);  
        nc.putAtt(wstring + "length", ncUint64, (unsigned long long int)_length);
        nc.putAtt(wstring + "bShared", ncUint, (uint32_t)_bShared);
        nc.putAtt(wstring + "bLocked", ncUint, (uint32_t)_bLocked);
        nc.putAtt(wstring + "norm", ncFloat, _norm);
        NcDim biasDim           = nc.addDim(wstring + "biasDim", _outputLayer._stride);
        NcVar biasVar           = nc.addVar(wstring + "bias", ncFloat, biasDim);
        if (pBias == NULL)
            pBias               = _vBias.data();
        biasVar.putVar(pBias);  
        if (_bShared)
        {
            nc.putAtt(wstring + "bTransposed", ncUint, (uint32_t)_bTransposed);
            nc.putAtt(wstring + "sourceInputLayer", _pSharedWeight->_inputLayer._name);
            nc.putAtt(wstring + "sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {

#if 0
        printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
        for (int i = 0; i < 20; i++)
            printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
            NcDim weightDim     = nc.addDim(wstring + "weightDim", (unsigned long long int)_outputLayer._stride * (unsigned long long int)_inputLayer._stride);
            NcVar weightVar     = nc.addVar(wstring + "weights", ncFloat, weightDim);
            if (!pWeight)
                pWeight         = _vWeight.data();
            weightVar.putVar(pWeight);
        }
    }

    return bResult;
}

bool NNWeight::CopyWeights(NNWeight* pWeight)
{
    bool bValid                 = true;

    // Check for valid weight pointer
    if (!pWeight)
    {
        if (getGpu()._id == 0)
            printf("NNWeight::CopyWeights: Invalid weight pointer.\n");
        bValid                  = false;
    }
    else if ((pWeight->_width != _width) || (pWeight->_height != _height) || (pWeight->_length != _length))
    {
        if (getGpu()._id == 0)
        {
            printf("NNWeight::CopyWeights: Mismatched weight dimensions (%" PRIu64 " x %" PRIu64 " x %" PRIu64") versus (%" PRIu64 " x %" PRIu64 " x %" PRIu64 ").\n", _width, _height, _length,
            pWeight->_width, pWeight->_height, pWeight->_length);
        }
        bValid                  = false;        
    }
    else
    {
        _vWeight                = pWeight->_vWeight;
        _vBias                  = pWeight->_vBias;
        _pbWeight->Upload(&_vWeight[0]);
        _pbBias->Upload(&_vBias[0]);
    }
    return bValid;
}

void NNWeight::Dump(string fname, NNFloat* pBuffer)
{
    // Create vector to hold entire weight matrix and resize it as such on process 0
    vector<NNFloat> vWeight;
    if (getGpu()._id == 0)
        vWeight.resize(_outputLayer._stride * _inputLayer._stride);

    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        cudaMemcpy(vWeight.data(), pBuffer, _outputLayer._stride * _inputLayer._stride * sizeof(NNFloat), cudaMemcpyDefault);
    }
    else
    {
        // Cannibalize system weight vector to hold buffer data
        uint32_t outgoingSize       = _outputLayer._stride * 3;               
        uint32_t incomingSize       = _inputLayer._stride * 2; 
        if (outgoingSize > incomingSize)
            cudaMemcpy(_vWeight.data(), pBuffer, _outputLayer._localStride * _inputLayer._stride * sizeof(NNFloat), cudaMemcpyDefault);
        else      
            cudaMemcpy(_vWeight.data(), pBuffer, _outputLayer._stride * _inputLayer._localStride * sizeof(NNFloat), cudaMemcpyDefault);

        // Reduce weight data into GPU 0
        if (getGpu()._id == 0)
        {
            NNFloat* pWeight            = vWeight.data();                    
            if (outgoingSize > incomingSize)
            {
                cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(NNFloat), _vWeight.data(), _outputLayer._localStride * sizeof(NNFloat), _outputLayer._localStride * sizeof(NNFloat), _inputLayer._stride, cudaMemcpyDefault);
                pWeight                += _outputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {                        
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    vector<NNFloat> vTemp(size);
                    MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    uint64_t lstride    = size / _inputLayer._stride;
                    NNFloat* pSrcWeight = vTemp.data();
                    NNFloat* pDstWeight = pWeight;
                    for (uint32_t j = 0; j < _inputLayer._stride; j++)
                    {
                        memcpy(pDstWeight, pSrcWeight, lstride * sizeof(NNFloat));
                        pSrcWeight     += lstride;
                        pDstWeight     += _outputLayer._stride;
                    }                          
                    pWeight            += lstride;
                }
            }
            else
            {
                cudaMemcpy(pWeight, _vWeight.data(), _outputLayer._stride * _inputLayer._localStride * sizeof(NNFloat), cudaMemcpyDefault);
                pWeight                += _outputLayer._stride * _inputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    pWeight            += size;
                }                        
            }
        }              
        else
        {
            uint64_t size               = _vWeight.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }

    }

    // Write file
    if (getGpu()._id == 0)
    {
        FILE* fp                        = fopen(fname.c_str(), "w");
        NNFloat* pData                  = vWeight.data();
        for (int i = 0; i < _inputLayer._stride; i++)
        {
            for (int j = 0; j < _outputLayer._stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}
