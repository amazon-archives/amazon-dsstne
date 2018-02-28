/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"
#include "Utils.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <queue>
#include <set>
#include <cfloat>

using namespace netCDF;
using namespace netCDF::exceptions;

NNNetworkDescriptor::NNNetworkDescriptor() :
_kind(NNNetwork::Kind::FeedForward),
_errorFunction(ErrorFunction::CrossEntropy),
_bShuffleIndices(true),
_maxout_k(2),
_LRN_k(2),
_LRN_n(5),
_LRN_alpha((NNFloat)0.0001),
_LRN_beta((NNFloat)0.75),
_RELUSlope((NNFloat)1.0),
_ELUAlpha((NNFloat)1),
_SELULambda((NNFloat)1.050701),
_bSparsenessPenalty(false),
_sparsenessPenalty_p((NNFloat)0.0),
_sparsenessPenalty_beta((NNFloat)0.0),
_bDenoising(false),
_denoising_p((NNFloat)0.0),
_deltaBoost_one((NNFloat)1.0),
_deltaBoost_zero((NNFloat)1.0),
_SMCE_oneTarget((NNFloat)0.9),
_SMCE_zeroTarget((NNFloat)0.1),
_SMCE_oneScale((NNFloat)1.0),
_SMCE_zeroScale((NNFloat)1.0),
_name(""),
_checkpoint_name("checkpoint"),
_checkpoint_interval(0),
_checkpoint_epochs(0),
_bConvLayersCalculated(false)
{

}

ostream& operator<< (ostream& out, NNNetworkDescriptor& d)
{
    out << "Name:                    " << d._name << endl;
    out << "Kind:                    " << d._kind << endl;
    out << "bShuffleIndices          " << std::boolalpha << d._bShuffleIndices << endl;
    out << "Error Function:          " << d._errorFunction << endl;
    out << "MaxOut_k:                " << d._maxout_k << endl;
    out << "LRN_k:                   " << d._LRN_k << endl;
    out << "LRN_n:                   " << d._LRN_n << endl;
    out << "LRN_beta:                " << d._LRN_beta << endl;
    out << "LRN_alpha:               " << d._LRN_alpha << endl;
    out << "bSparsenessPenalty:      " << std::boolalpha << d._bSparsenessPenalty << endl;
    out << "sparsenessPenalty_beta:  " << d._sparsenessPenalty_beta << endl;
    out << "sparsenessPenalty_p:     " << d._sparsenessPenalty_p << endl;
    out << "bDenoising:              " << std::boolalpha << d._bDenoising << endl;
    out << "denoising_p:             " << d._denoising_p << endl;
    out << "deltaBoost_one:          " << d._deltaBoost_one << endl;
    out << "deltaBoost_zero:         " << d._deltaBoost_zero << endl;
    out << "SMCE_oneTarget:          " << d._SMCE_oneTarget << endl;
    out << "SMCE_zeroTarget:         " << d._SMCE_zeroTarget << endl;
    out << "SMCE_oneScale:           " << d._SMCE_oneScale << endl;
    out << "SMCE_zeroScale:          " << d._SMCE_zeroScale << endl;
    out << "checkpoint_name:         " << d._checkpoint_name << endl;
    out << "checkpoint_interval:     " << d._checkpoint_interval << endl;

    // Dump layers
    out << endl << "Layers:" << endl;
    for (uint32_t i = 0; i < d._vLayerDescriptor.size(); i++)
    {
        out << "Layer " << i << endl << d._vLayerDescriptor[i];
    }

    // Dump Weights
    out << endl << "Weights:" << endl;
    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        out << "Weight " << i << endl << d._vWeightDescriptor[i];
    }
    return out;
}

bool ValidateNetworkDescriptor(NNNetworkDescriptor& d)
{
    return true;
}

tuple<NNFloat, uint32_t, NNFloat, NNFloat> NNNetwork::GetLRN() const
{
    return make_tuple(_LRN_k, _LRN_n, _LRN_alpha, _LRN_beta);
}

tuple<uint32_t> NNNetwork::GetMaxout() const
{
    return make_tuple(_maxout_k);
}

tuple<NNFloat, NNFloat> NNNetwork::GetSparsenessPenalty() const
{
    return make_tuple(_sparsenessPenalty_p, _sparsenessPenalty_beta);
}

tuple<NNFloat> NNNetwork::GetDenoising() const
{
    return make_tuple(_denoising_p);
}

tuple<NNFloat, NNFloat> NNNetwork::GetDeltaBoost() const
{
    return make_tuple(_deltaBoost_one, _deltaBoost_zero);
}

tuple<NNFloat, NNFloat, NNFloat, NNFloat> NNNetwork::GetSMCE() const
{
    return make_tuple(_SMCE_oneTarget, _SMCE_zeroTarget, _SMCE_oneScale, _SMCE_zeroScale);
}

tuple<bool> NNNetwork::GetShuffleIndices() const
{
    return make_tuple(_bShuffleIndices);
}

tuple<string, int32_t> NNNetwork::GetCheckPoint() const
{
    return make_tuple(_checkpoint_name, _checkpoint_interval);
}

const NNLayer* NNNetwork::GetLayer(const string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetLayerDimensions: Unknown layer %s.\n", layer.c_str());
        }

        return NULL;
    }

    return itr->second;
}

NNFloat* NNNetwork::GetScratchBuffer(size_t size)
{
    // Increase size if requested
    if (size > _scratchBufferSize)
    {
        _pbScratchBuffer.reset(new GpuBuffer<NNFloat>(size));
        _scratchBufferSize                  = size;

    }
    return _pbScratchBuffer->_pDevData;
}

void NNNetwork::SetCUDNNWorkspace(size_t size)
{
    if (size > _maxCUDNNWorkspaceSize)
    {
        _maxCUDNNWorkspaceSize = size;
    }
}

NNFloat* NNNetwork::GetP2PSendBuffer()
{
    return _pbP2PBuffer[_sendIndex]->_pDevData;
}

NNFloat* NNNetwork::GetP2PReceiveBuffer()
{
    return _pbP2PBuffer[_receiveIndex]->_pDevData;
}

NNFloat* NNNetwork::GetP2PCPUBuffer()
{
    return _pCPUBuffer.get();
}

NNFloat* NNNetwork::GetPeerBuffer()
{
    return _pPeerBuffer[_receiveIndex];
}

NNFloat* NNNetwork::GetPeerBackBuffer()
{
    return _pPeerBuffer[_sendIndex];
}

bool NNNetwork::SetLRN(uint32_t k, uint32_t n, NNFloat alpha, NNFloat beta)
{
    _LRN_k                  = k;
    _LRN_n                  = n;
    _LRN_alpha              = alpha;
    _LRN_beta               = beta;
    _bDirty                 = true;

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetLRN: k set to %u, n set to %u, alpha set to %f, beta set to %f.\n", k, n, alpha, beta);

    return true;
}

bool NNNetwork::SetMaxout(uint32_t k)
{
    if (k != _maxout_k)
    {
        _maxout_k           = k;
        _bDirty             = true;
    }

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetMaxout: k set to %u\n.", k);

    return true;
}

bool NNNetwork::SetSparsenessPenalty(NNFloat p, NNFloat beta)
{
    // Validate target sparseness
    if ((p < (NNFloat)0.0) || (p > (NNFloat)1.0))
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetSparsenessPenalty: Target sparseness must be >=0 and <=1.\n");
        return false;
    }

    _sparsenessPenalty_p    = p;
    _sparsenessPenalty_beta = beta;
    _bSparsenessPenalty     = (_sparsenessPenalty_beta > (NNFloat)0.0);
    _bDirty                 = true;

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetSparsenessPenalty: p set to %f, beta set to %f.\n", p, beta);

    return true;
}

bool NNNetwork::SetDenoising(NNFloat p)
{
    // Validate denoising probability
    if ((p < (NNFloat)0.0) || (p >= (NNFloat)1.0))
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetDenoising: Denoising probability must be >=0 and <1.\n");
        return false;
    }

    if (_denoising_p != p)
    {
        _denoising_p        = p;
        _bDenoising         = (_denoising_p > (NNFloat)0.0);
        _bDirty             = true;
    }

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetDenoising: p set to %f.\n", p);

    return true;
}

bool NNNetwork::SetDeltaBoost(NNFloat one, NNFloat zero)
{
    // Validate parameters
    if (one < (NNFloat)0.0)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetDeltaBoost: Illegal value for one (%f).\n", one);
        return false;
    }
    else if (zero < (NNFloat)0.0)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetDeltaBoost: Illegal value for zero (%f).\n", zero);
        return false;
    }

    // Set parameters and signal success
    _deltaBoost_one         = one;
    _deltaBoost_zero        = zero;
    _bDirty                 = true;

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetDeltaBoost: one set to %f, zero set to %f.\n", one, zero);

    return true;
}
bool NNNetwork::SetSMCE(NNFloat oneTarget, NNFloat zeroTarget, NNFloat oneScale, NNFloat zeroScale)
{
    // Validate parameters
    if ((oneTarget < (NNFloat)0.0) || (oneTarget > (NNFloat)1.0))
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetSMCE: Illegal value for oneTarget (%f).\n", oneTarget);
        return false;
    }
    else if ((zeroTarget < (NNFloat)0.0) || (zeroTarget > (NNFloat)1.0))
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetSMCE: Illegal value for zeroTarget (%f).\n", zeroTarget);
        return false;
    }
    else if (oneScale < (NNFloat)0.0)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetSMCE: Illegal value for oneScale (%f).\n", oneScale);
        return false;
    }
    else if (zeroScale < (NNFloat)0.0)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetSMCE: Illegal value for zeroScale (%f).\n", zeroScale);
        return false;
    }

    // Set parameters and signal success
    _SMCE_oneTarget         = oneTarget;
    _SMCE_zeroTarget        = zeroTarget;
    _SMCE_oneScale          = oneScale;
    _SMCE_zeroScale         = zeroScale;
    _bDirty                 = true;

    // Report new settings
    if (getGpu()._id == 0)
        printf("NNNetwork::SetSMCE: oneTarget set to %f, zeroTarget set to %f, oneScale set to %f, zeroScale set to %f.\n", oneTarget, zeroTarget, oneScale, zeroScale);

    return true;
}

bool NNNetwork::SetCheckpoint(string name, int32_t interval)
{
    _checkpoint_name        = name;
    _checkpoint_interval    = interval;

    // Report new settings
    if (getGpu()._id == 0) {
        printf("NNNetwork::SetCheckPoint: filename set to %s, interval set to %d epochs.\n", name.c_str(), interval);
    }
    return true;
}

NNNetwork::NNNetwork(NNNetworkDescriptor& d, uint32_t batch) :
_name(d._name),
_kind(d._kind),
_mode(Prediction),
_trainingMode(SGD),
_batch(batch),
_localBatch(batch),
_position(0),
_localPosition(0),
_bShuffleIndices(d._bShuffleIndices),
_shuffleIndices(0),
_pShuffleIndex(NULL),
_pShuffleIndexSort(),
_pbShuffleIndex(),
_bExamplesFound(false),
_bAllDataLoaded(true),
_examples(0),
_errorFunction(d._errorFunction),
_LRN_k(d._LRN_k),
_LRN_n(d._LRN_n),
_LRN_alpha(d._LRN_alpha),
_LRN_beta(d._LRN_beta),
_maxout_k(d._maxout_k),
_bSparsenessPenalty(d._bSparsenessPenalty),
_sparsenessPenalty_beta(d._sparsenessPenalty_beta),
_sparsenessPenalty_p(d._sparsenessPenalty_p),
_bDenoising(d._bDenoising),
_denoising_p(d._denoising_p),
_deltaBoost_one(d._deltaBoost_one),
_deltaBoost_zero(d._deltaBoost_zero),
_SMCE_oneTarget(d._SMCE_oneTarget),
_SMCE_zeroTarget(d._SMCE_zeroTarget),
_SMCE_oneScale(d._SMCE_oneScale),
_SMCE_zeroScale(d._SMCE_zeroScale),
_checkpoint_name(d._checkpoint_name),
_checkpoint_interval(d._checkpoint_interval),
_checkpoint_epochs(0),
_epochs(0),
_batches(0),
_bClearVelocity(true),
_bDirty(true),
_maxStride(0),
_scratchBufferSize(0),
_pbScratchBuffer(),
_pPeerBuffer{NULL, NULL},
_pbP2PBuffer(),
_pCPUBuffer(),
_sendIndex(0),
_receiveIndex(1),
_CUDNNWorkspaceSize(0),
_maxCUDNNWorkspaceSize(0),
_pbCUDNNWorkspace(),
_verbose(false)
{

    // Allocate layers
    for (auto l: d._vLayerDescriptor)
    {
        _vLayer.push_back(new NNLayer(l, batch));
        _mLayer[_vLayer.back()->_name]          = _vLayer.back();

        // Single out and save input and output layers to simplify
        // training and prediction
        if (_vLayer.back()->_kind == NNLayer::Kind::Input)
        {
            _vInputLayer.push_back(_vLayer.back());
        }
        else if (_vLayer.back()->_kind == NNLayer::Kind::Output)
        {
             _vOutputLayer.push_back(_vLayer.back());
        }
    }

    if (getGpu()._id == 0)
    {
        cout << "NNNetwork::NNNetwork: " << _vInputLayer.size() <<" input layer" << (_vInputLayer.size() > 1 ? "s" : "") << endl;
        cout << "NNNetwork::NNNetwork: " << _vOutputLayer.size() << " output layer" << (_vOutputLayer.size() > 1 ? "s" : "") << endl;
    }

    // Add skip layers and pooling connections
    for (auto l : _vLayer)
    {

        // Handle skip connections
        for (auto s : l->_vSkip)
        {
            NNLayer* pLayer = _mLayer[s];

            // Make sure dimensions match
            if (pLayer->_stride != l->_stride)
            {
                if (getGpu()._id == 0)
                    printf("NNNetwork::NNNetwork: Layer dimensions do not match for skip connection between layer %s and %s.\n",
                            l->_name.c_str(), pLayer->_name.c_str());
                getGpu().Shutdown();
                exit(-1);
            }

            l->_vIncomingSkip.push_back(pLayer);
            pLayer->_vOutgoingSkip.push_back(l);
        }

        // Add pooling connections
        if (l->_type == NNLayer::Type::Pooling)
        {
            for (auto s: l->_vSource)
            {
                NNLayer* pLayer = _mLayer[s];
                l->_vIncomingLayer.push_back(pLayer);
                pLayer->_vOutgoingLayer.push_back(l);
                
                // Validate dot product and maxout layers sources are all the same size
                if ((l->_poolingFunction == PoolingFunction::Maxout) || (l->_poolingFunction == PoolingFunction::Maxout))
                {
                    if (pLayer->_stride != l->_vIncomingLayer[0]->_stride)
                    {
                        if (getGpu()._id == 0)
                        {
                            cout << "NNNetwork::NNetwork: All source layer dimensions must match for " << l->_poolingFunction << " layer " << l->_name << endl;
                        }
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }

                // BUG Multi-GPU
            }
        }
    }

    // Allocate weights between layers
    for (auto wd: d._vWeightDescriptor)
    {
        NNLayer* pInputLayer                    = _mLayer[wd._inputLayer];
        NNLayer* pOutputLayer                   = _mLayer[wd._outputLayer];
        NNWeight* pWeight                       = new NNWeight(*pInputLayer, *pOutputLayer, wd._bShared, wd._bTransposed, wd._bLocked, wd._norm);
        _vWeight.push_back(pWeight);

        // Initialize weight values if they aren't provided.  In the case of
        // shared weights, only the biases are set
        if ((wd._vWeight.size() == 0) || (wd._vBias.size() == 0))
        {
            pWeight->Randomize();
        }

        // Copy weights if unshared and values are supplied (sharded across input layer if model parallel multi-GPU)
        if (!wd._bShared && (wd._vWeight.size() != 0))
        {
            if (getGpu()._numprocs > 1)
            {
                NNFloat* pDst                   = pWeight->_vWeight.data();
                uint32_t outgoingSize           = pOutputLayer->_stride * 3;
                uint32_t incomingSize           = pInputLayer->_stride * 2;

                if (outgoingSize > incomingSize)
                {
                    NNFloat* pSrc               = wd._vWeight.data() + pOutputLayer->_minX;
                    for (size_t i = 0; i < pInputLayer->_stride; i++)
                    {
                        memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(NNFloat));
                        pSrc                   += pOutputLayer->_stride;
                        pDst                   += pOutputLayer->_localStride;
                    }
                }
                else
                {
                    NNFloat* pSrc               = wd._vWeight.data() + pInputLayer->_minX * pOutputLayer->_stride;
                    memcpy(pDst, pSrc, pInputLayer->_localStride * pOutputLayer->_stride * sizeof(NNFloat));
                }

            }
            else
            {
                pWeight->_vWeight               = wd._vWeight;
            }
            pWeight->_pbWeight->Upload(pWeight->_vWeight.data());
        }

        // Copy biases if present (sharded across output layer if multi-GPU)
        if (wd._vBias.size() != 0)
        {
            if (getGpu()._numprocs > 1)
            {
                NNFloat* pSrc                   = wd._vBias.data() + pOutputLayer->_minX;
                NNFloat* pDst                   = pWeight->_vBias.data();
                memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(NNFloat));
            }
            else
            {
                pWeight->_vBias                     = wd._vBias;
            }
            pWeight->_pbBias->Upload(pWeight->_vBias.data());
        }
    }

    // Now locate sources for all shared weights using second
    // pass so this is order-independent
    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        NNWeightDescriptor &wd                  = d._vWeightDescriptor[i];
        if (wd._bShared)
        {
            NNWeight* pWeight                   = _vWeight[i];
            string inputLayer                   = wd._sourceInputLayer;
            string outputLayer                  = wd._sourceOutputLayer;
            bool bFound                         = false;
            for (int j = 0; j < _vWeight.size(); j++)
            {
                if (!(_vWeight[j]->_bShared) &&
                     (_vWeight[j]->_inputLayer._name == inputLayer) &&
                     (_vWeight[j]->_outputLayer._name == outputLayer))
                {
                    // If transposed, check that this is a 2D weight matrix rather than a 3D tensor
                    if (wd._bTransposed)
                    {
                        if (wd._length > 1)
                        {
                            if (getGpu()._id == 0)
                                printf("NNNetwork::NNNetwork: Can't transpose 3D weight matrix for shared weights between layers %s and %s\n",
                                _vWeight[i]->_inputLayer._name.c_str(), _vWeight[i]->_outputLayer._name.c_str());
                            getGpu().Shutdown();
                            exit(-1);
                        }

                        if ((_vWeight[i]->_width != _vWeight[j]->_height) || (_vWeight[i]->_height != _vWeight[j]->_width))
                        {
                            if (getGpu()._id == 0)
                                printf("NNNetwork::NNNetwork: Transposed dimensions for shared weights between layers %s and %s do not match\n",
                                _vWeight[i]->_inputLayer._name.c_str(), _vWeight[i]->_outputLayer._name.c_str());
                            getGpu().Shutdown();
                            exit(-1);
                        }
                    }
                    else if ((_vWeight[i]->_width != _vWeight[j]->_width) ||
                             (_vWeight[i]->_height != _vWeight[j]->_height) ||
                             (_vWeight[i]->_length != _vWeight[j]->_length))
                    {
                             if (getGpu()._id == 0)
                                printf("NNNetwork::NNNetwork: Dimensions for shared weights between layers %s and %s do not match\n",
                                _vWeight[i]->_inputLayer._name.c_str(), _vWeight[i]->_outputLayer._name.c_str());
                            getGpu().Shutdown();
                            exit(-1);
                    }

                    _vWeight[i]->_pSharedWeight = _vWeight[j];
                    if (_vWeight[j]->_sharingCount == 1)
                        _vSharedWeight.push_back(_vWeight[j]);
                    _vWeight[j]->_sharingCount++;
                    bFound                      = true;
                    break;
                }
            }

            // Complain and exit if source for shared weights wasn't located
            if (!bFound)
            {
                if (getGpu()._id == 0)
                    printf("NNNetwork::NNNetwork: Unable to locate shared weights for connection between layers %s and %s.\n",
                    _vWeight[i]->_inputLayer._name.c_str(), _vWeight[i]->_outputLayer._name.c_str());
                getGpu().Shutdown();
                exit(-1);
            }
        }
    }

    CalculatePropagationOrder();
}

void NNNetwork::Randomize()
{
    for (auto pw : _vWeight)
        pw->Randomize();
}

void NNNetwork::SetBatch(uint32_t batch)
{
    // Validate batch size
    if (batch % getGpu()._numprocs)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::SetBatch: Batch size must be a multiple of process count.\n");
        return;
    }

    if (batch != _batch)
    {
        _batch                  = batch;
        for (auto pL: _vLayer)
        {
            pL->SetBatch(batch);
        }

        _bDirty                 = true;
        if (getGpu()._id == 0)
            printf("NNNetwork::SetBatch: Batch size set to %d.\n", _batch);
    }
}

uint32_t NNNetwork::GetBatch() const
{
    return _batch;
}

uint32_t NNNetwork::GetExamples() const
{
    return _examples;
}

void NNNetwork::SetShuffleIndices(bool bShuffleIndices)
{
    if (_bShuffleIndices != bShuffleIndices)
    {
        _bShuffleIndices        = bShuffleIndices;
        _bDirty                 = true;
    }

    if (getGpu()._id == 0)
        printf("NNNetwork::SetShuffleIndices: Index shuffling is now %s\n", (_bShuffleIndices ? "on" : "off"));
}

uint32_t NNNetwork::GetPosition() const
{
    return _position;
}

void NNNetwork::SetPosition(uint32_t position)
{
    if (_bExamplesFound)
    {
        if (position < _examples)
            _position           = position;
        else if (getGpu()._id == 0)
            printf("NNNetwork::SetPosition: Invalid position setting: %u, maximum %u\n", position, _examples);
    }
    else if (getGpu()._id == 0)
    {
        printf("NNNetwork::SetPosition: Illegal attempt to set position without examples count information.\n");
    }
}

bool NNNetwork::LockWeights(const string& inputLayer, const string& outputLayer)
{
    NNLayer* pInputLayer        = _mLayer[inputLayer];
    NNLayer* pOutputLayer       = _mLayer[outputLayer];

    if (pInputLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::LockWeights: Unable to find input layer %s.\n", inputLayer.c_str());
        return false;
    }

    if (pOutputLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::LockWeights: Unable to find input layer %s.\n", outputLayer.c_str());
        return false ;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++)
    {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name))
        {
            _vWeight[i]->Lock();
            return true;
        }
    }

    if (getGpu()._id == 0)
        printf("NNNetwork::LockWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    return false;
}

bool NNNetwork::UnlockWeights(const string& inputLayer, const string& outputLayer)
{
    NNLayer* pInputLayer        = _mLayer[inputLayer];
    NNLayer* pOutputLayer       = _mLayer[outputLayer];

    if (pInputLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::UnlockWeights: Unable to find input layer %s.\n", inputLayer.c_str());
        return false;
    }

    if (pOutputLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::UnlockWeights: Unable to find input layer %s.\n", outputLayer.c_str());
        return false ;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++)
    {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name))
        {
            _vWeight[i]->Unlock();
            return true;
        }
    }

    if (getGpu()._id == 0)
        printf("NNNetwork::UnlockWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    return false;
}

void NNNetwork::SetTrainingMode(TrainingMode mode)
{
    if (_trainingMode != mode)
    {
        _trainingMode                       = mode;
        _bDirty                             = true;
    }

    if (getGpu()._id == 0)
        cout<<"NNNetwork::SetTrainingMode: Optimizer is now " <<_trainingMode<<endl;

}

void NNNetwork::RefreshShuffleBuffers()
{
    // Shuffle buffers are sticky once training has been triggered to prevent malloc thrashing
    if (_bAllDataLoaded)
    {
        if (_bShuffleIndices && (_mode == Training))
        {
            if (_shuffleIndices != _examples)
            {
                // Delete old shuffle buffer data
                if (getGpu()._id == 0)
                {
                    _pShuffleIndexSort.reset();
                }
                else
                {
                    _pbShuffleIndex.reset();
                }


                _shuffleIndices             = _examples;


                if (getGpu()._id == 0)
                {
                    _pShuffleIndexSort.reset(new GpuSort<uint32_t, uint32_t>(_shuffleIndices));
                    _pShuffleIndex              = _pShuffleIndexSort->GetValuePointer();

                    // Create and upload indices, need to use larger
                    // than needed vector because of stride alignment
                    // and double-buffering
                    uint32_t stride             = ((_shuffleIndices + 511) >> 9) << 9;
                    vector<uint32_t> vIndex(stride * 2);
                    for (uint32_t i = 0; i < _examples; i++)
                    {
                        vIndex[i]               = i;
                    }
                    _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());
               }
               else
               {
                    _pbShuffleIndex.reset(new GpuBuffer<uint32_t>(_shuffleIndices));
                    _pShuffleIndex              = _pbShuffleIndex->_pDevData;
               }
            }
        }
    }
}

void NNNetwork::ShuffleIndices()
{
    // Sort on process 0
    if (getGpu()._id == 0)
    {
        // Initialize shuffle indices (yes I don't strictly need to do this
        // but if you think about the importance of deterministic execution
        // you'll eventually figure out why this is the right thing to do(tm))
        uint32_t stride                     = ((_shuffleIndices + 511) >> 9) << 9;
        vector<uint32_t> vIndex(stride * 2);
        for (uint32_t i = 0; i < _examples; i++)
        {
            vIndex[i]                       = i;
        }
         _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());

        // Generate random keys
        curandGenerate(getGpu()._RNG, _pShuffleIndexSort->GetKeyPointer(), _shuffleIndices);

        // Sort by keys
        _pShuffleIndexSort->Sort();
    }

    // Broadcast results to all other processes
    if (getGpu()._numprocs > 1)
    {
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        // Broadcast from P2P Send buffer
        P2P_Bcast(_pShuffleIndex, _examples * sizeof(uint32_t));
    }
}

void NNNetwork::RefreshState()
{
    // If any datasets have been loaded, then *all* data sets
    // must be loaded
    if (!_bAllDataLoaded)
    {
        _bAllDataLoaded                         = true;

        // Always need data sets (even if empty) to predict or train
        for (auto l: _vInputLayer)
        {
            if (l->_pDataSet == NULL)
            {
                if (getGpu()._id == 0)
                    cout << "NNNetwork::RefreshState: Missing data set " << l->_dataSet << " for input layer " << l->_name << endl;
                _bAllDataLoaded                 = false;
            }
        }

        // If validation or training, need data sets for all output layers
        if (_mode != Prediction)
        {
            for (auto l: _vOutputLayer)
            {
                if (l->_pDataSet == NULL)
                {
                    if (getGpu()._id == 0)
                        cout << "NNNetwork::RefreshState: Missing data set " << l->_dataSet << " for output layer " << l->_name << endl;
                    _bAllDataLoaded                 = false;
                }
            }
        }
    }

    if (_bDirty)
    {
        // Reallocate layers if batch size doesn't match
        for (auto l: _vLayer)
        {
            if (l->_bDirty)
            {
                l->RefreshState(this, _mode == Validation);
            }
        }

        // Add weight gradients and velocity if needed
        for (auto w: _vWeight)
        {
            w->RefreshState(this, _trainingMode);
        }

        // Allocate index shuffle buffers
        RefreshShuffleBuffers();
    }

    // Allocate P2P Data if active (Refresh layer state first as above or this could fail)
    if (getGpu()._numprocs > 1)
    {
        DeallocatePeerBuffers();
        AllocatePeerBuffers();
    }

    // Increase size of CUDNN workspace if necessary
    if (_maxCUDNNWorkspaceSize > _CUDNNWorkspaceSize)
    {
        if (getGpu()._id == 0)
            cout << "NNNetwork::RefreshState: Setting cuDNN workspace size to " << _maxCUDNNWorkspaceSize << " bytes." << endl;
        _CUDNNWorkspaceSize                     = _maxCUDNNWorkspaceSize;
        _pbCUDNNWorkspace.reset(new GpuBuffer<uint8_t>(_CUDNNWorkspaceSize));
    }

    // Update GPU state if dirty or GPU context is set for
    // different network
    if (_bDirty || (getGpu()._pNetwork != this))
    {
        getGpu().SetNeuralNetwork(this);
    }

    _bDirty                                     = false;
}

// Clears all active data sets from neural network
void NNNetwork::ClearDataSets()
{
    _examples                                   = 0;
    _bExamplesFound                             = false;
    for (auto l: _vInputLayer)
         l->_pDataSet                           = NULL;
    for (auto l: _vOutputLayer)
         l->_pDataSet                           = NULL;
}


void NNNetwork::LoadDataSets(vector<NNDataSetBase*>& vData)
{
    // Search for a data set to match each input layer
    _bAllDataLoaded                             = false;
    for (auto l: _vInputLayer)
    {
        for (auto d: vData)
        {
            if (l->_dataSet.compare(d->_name) == 0)
            {
                // Check if dimensionality of data matches
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        printf("NNNetwork::LoadDataSets: Dimensionality mismatch %uD input layer %s versus %uD data set %s\n",
                        l->_dimensions, l->_name.c_str(), d->_dimensions, d->_name.c_str());
                    }
                }

                // Check if input layer and data set dimensions match
                if ((l->_Nx < d->_width) ||
                    (l->_Ny < d->_height) ||
                    (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        printf("NNNetwork::LoadDataSets: Data element mismatch (%u, %u, %u) input layer %s versus (%u, %u, %u) data set %s\n",
                        l->_Nx, l->_Ny, l->_Nz, l->_name.c_str(),
                        d->_width, d->_height, d->_length, d->_name.c_str());
                    }
                    break;
                }

                // Determine examples count for network if this is the first recognized data set
                if (!_bExamplesFound)
                {
                    _examples                   = d->_examples;
                    _bExamplesFound             = true;
                }

                // Check for valid examples count
                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                        printf("NNNetwork::LoadDataSets: Mismatched examples count (%u vs %u) in dataset %s\n", _examples, d->_examples, d->_name.c_str());
                    break;
                }

                // Signal successful location of matching data set
                l->_pDataSet                    = d;
                l->_bDirty                      = true;
                if (getGpu()._id == 0)
                    printf("NNNetwork::LoadDataSets: Found data set %s for input layer %s\n", d->_name.c_str(), l->_name.c_str());
                break;
            }
        }
    }

    // Search for a data set to match each output layer
    for (auto l: _vOutputLayer)
    {
        for (auto d: vData)
        {
            if (l->_dataSet.compare(d->_name) == 0)
            {
                // Check if dimensionality of data matches
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        printf("NNNetwork::LoadDataSets: Dimensionality mismatch %uD output layer %s versus %uD data set %s\n",
                        l->_dimensions, l->_name.c_str(), d->_dimensions, d->_name.c_str());
                    }
                }

                // Check if output layer and data set dimensions match
                if ((l->_Nx < d->_width) ||
                    (l->_Ny < d->_height) ||
                    (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        printf("NNNetwork::LoadDataSets: Data element mismatch (%u, %u, %u) output layer %s versus (%u, %u, %u) data set %s\n",
                        l->_Nx, l->_Ny, l->_Nz, l->_name.c_str(),
                        d->_width, d->_height, d->_length, d->_name.c_str());
                    }
                    break;
                }

                // Determine examples count for network if this is the first recognized data set
                if (!_bExamplesFound)
                {
                    _examples                   = d->_examples;
                    _bExamplesFound             = true;
                }

                // Check for valid examples count
                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                        printf("NNNetwork::LoadDataSets: Mismatched examples count (%u vs %u) in dataset %s\n", _examples, d->_examples, d->_name.c_str());
                    break;
                }

                // Signal successful location of matching data set
                l->_pDataSet                    = d;
                l->_bDirty                      = true;
                if (getGpu()._id == 0)
                    printf("NNNetwork::LoadDataSets: Found data set %s for output layer %s\n", d->_name.c_str(), l->_name.c_str());
                break;
            }
        }
    }
    _bDirty                                     = true;
}


void NNNetwork::LoadBatch()
{
    // Refresh state if needed
    if (_bDirty)
        RefreshState();

    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;

    // Now load input
    for (auto l: _vInputLayer)
    {
        switch (_mode)
        {
            case Prediction:
                l->LoadPredictionBatch(_position, batch);
                break;

            case Training:
                l->LoadTrainingBatch(_position, batch);
                break;

            case Validation:
                l->LoadValidationBatch(_position, batch);
                break;

            default:
              cout << "unsupported mode in LoadBatch" << endl;
              exit(1);
        }
    }
}

void NNNetwork::SaveWeights(const string& fname, const string& inputLayer, const string& outputLayer)
{
    bool bResult            = true;
    if (getGpu()._id == 0)
    {
        NNLayer* pInputLayer        = _mLayer[inputLayer];
        NNLayer* pOutputLayer       = _mLayer[outputLayer];

        if (pInputLayer == NULL)
        {
            printf("NNNetwork::SaveWeights: Unable to find input layer %s.\n", inputLayer.c_str());
            bResult                 = false;
            goto exit;
        }

        if (pOutputLayer == NULL)
        {
            printf("NNNetwork::SaveWeights: Unable to find input layer %s.\n", outputLayer.c_str());
            bResult                 = false;
            goto exit;
        }

        for (auto w: _vWeight)
        {
            if ((w->_inputLayer._name == pInputLayer->_name) && (w->_outputLayer._name == pOutputLayer->_name))
            {
                FILE* fp = fopen(fname.c_str(), "w");
                if (fp == NULL)
                {
                    printf("NNNetwork::SaveWeights: Failed to open output file %s.\n", fname.c_str());
                    bResult         = false;
                    goto exit;
                }

                w->_pbWeight->Download(w->_vWeight.data());
                w->_pbBias->Download(w->_vBias.data());
                fprintf(fp, "%" PRIu64 ",%" PRIu64 "\n", w->_width, w->_height);
                for (int j = 0; j < w->_height; j++)
                {
                    for (int k = 0; k < w->_width; k++)
                    {
                        fprintf(fp, "%12.8f", w->_vWeight[j * w->_width + k]);
                        if (k != w->_width - 1)
                            fprintf(fp, ",");
                        else
                            fprintf(fp, "\n");
                    }
                }
                for (int k = 0; k < w->_width; k++)
                {
                    fprintf(fp, "%12.8f", w->_vBias[k]);
                    if (k != w->_width - 1)
                        fprintf(fp, ",");
                    else
                        fprintf(fp, "\n");
                }
                fclose(fp);
                bResult             = true;
                goto exit;
            }
        }

        printf("NNNetwork::SaveWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
        bResult                     = false;
    }

    // Check for success
exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }
}


void NNNetwork::SaveLayer(const string& fname, const string& layer)
{
    bool bResult            = true;
    if (getGpu()._id == 0)
    {
        NNLayer* pLayer     = _mLayer[layer];
        if (pLayer == NULL)
        {
            if (getGpu()._id == 0)
               printf("NNNetwork::SaveLayer: Attempt to save nonexistent layer %s.\n", layer.c_str());
            bResult         = false;
            goto exit;
        }

        FILE* fp = fopen(fname.c_str(), "w");
        if (fp == NULL)
        {
            if (getGpu()._id == 0)
                printf("NNNetwork::SaveLayer: Failed to open output file %s.\n", fname.c_str());
            bResult         = false;
            goto exit;
        }
        DumpLayer(fp, layer);
        fclose(fp);
    }

    // Check for success
exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }
}

void NNNetwork::DumpLayer(FILE* fp, const string& layer)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        // Check for valid layer
        NNLayer* pLayer     = _mLayer[layer];
        if (pLayer == NULL)
        {
            printf("NNNetwork::SaveLayer: Attempt to dump nonexistent layer %s.\n", layer.c_str());
            bResult         = false;
            goto exit;
        }

        uint64_t batch      = pLayer->_batch;
        if (batch + _position > _examples)
        {
            batch           = _examples - _position;
        }
        uint32_t stride     = pLayer->_localStride;
        uint64_t size       = _batch * stride;
        vector<float> vData(size);
        pLayer->_pbUnit->Download(vData.data());
        for (uint32_t j = 0; j < batch; j++)
        {
            for (uint32_t k = 0; k < stride; k++)
            {
                fprintf(fp, "%f", vData[j * stride + k]);
                if (k < (stride -1))
                    fprintf(fp, ",");
                else
                    fprintf(fp, "\n");
            }
        }
    }


    // Check for success
exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }
}

void NNNetwork::SaveBatch(string fname)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        FILE* fp = fopen(fname.c_str(), "w");
        if (fp == NULL)
        {
            if (getGpu()._id == 0)
                printf("NNNetwork::SaveBatch: Failed to open output file %s.\n", fname.c_str());
            bResult         = false;
            goto exit;
        }

        DumpBatch(fp);
        fclose(fp);
    }

    // Check for success
exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }
}

void NNNetwork::DumpBatch(FILE* fp)
{
    if (getGpu()._id == 0)
    {
        for (int i = 0; i < _vOutputLayer.size(); i++)
        {
            uint32_t stride     = _vOutputLayer[i]->_localStride;
            uint32_t batch      = _vOutputLayer[i]->_batch;
            if (batch + _position > _examples)
                batch           = _examples - _position;
            uint64_t size       = (uint64_t)batch * (uint64_t)stride;
            vector<NNFloat> vData(size);
            _vOutputLayer[i]->_pbUnit->Download(vData.data());
            for (uint32_t j = 0; j < batch; j++)
            {
                for (uint32_t k = 0; k < stride; k++)
                {
                    fprintf(fp, "%f", vData[j * stride + k]);
                    if (k < (stride -1))
                        fprintf(fp, ",");
                    else
                        fprintf(fp, "\n");
                }
            }
        }
    }
}

void NNNetwork::PredictBatch(uint32_t layers)
{
    // Calculate max layers
    uint32_t maxLayers = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::PredictBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
        return;
    }

    // Check if already in prediction mode
    if (_mode != Prediction)
    {
        _mode                                               = Prediction;
        _bDirty                                             = true;
    }

    // Refresh state if needed
    if (_bDirty)
    {
        RefreshState();

        // Make sure all data is present
        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                cout << "NNNetwork::PredictBatch: Attempt to predict with neural network " << _name << " without providing data sets" << endl;
                cout << "for all input and output layers." << endl;
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }



    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;


    // Load batch from current position
    ClearUpdates();
    LoadBatch();

    // Now run forward through all layers
    for (auto l: _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}

void NNNetwork::PredictTrainingBatch(uint32_t layers)
{
    // Calculate max layers
    uint32_t maxLayers                      = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::PredictTrainingBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
        return;
    }

    // Refresh state if needed
    if (_bDirty)
    {
        RefreshState();

        // Make sure all data is present
        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                cout << "NNNetwork::PredictTrainingBatch: Attempt to predict with neural network " << _name << " without providing data sets" << endl;
                cout << "for all input and output layers." << endl;
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }



    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;

    //printf("Predicting at %d, %d examples\n", _position, batch);

    // Load batch from current position
    LoadBatch();

    // Now run forward through all layers
    for (auto l: _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, true);
    }

}


void NNNetwork::PredictValidationBatch(uint32_t layers)
{
    // Calculate max layers
    uint32_t maxLayers = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::PredictvalidationBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
        return;
    }

    // Check if already in prediction mode
    if (_mode != Validation)
    {
        _mode                                               = Prediction;
        _bDirty                                             = true;
    }

    // Refresh state if needed
    if (_bDirty)
    {
        RefreshState();

        // Make sure all data is present
        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                cout << "NNNetwork::PredictValidationBatch: Attempt to predict with neural network " << _name << " without providing data sets" << endl;
                cout << "for all input and output layers." << endl;
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }



    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;


    // Load batch from current position
    LoadBatch();

    // Now run forward through all layers
    ClearUpdates();
    for (auto l: _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}

NNFloat NNNetwork::Train(uint32_t epochs, NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1)
{
    // Check if already in training mode
    if (_mode != Training)
    {
        _mode                                               = Training;
        _bDirty                                             = true;
    }

    // Refresh state if needed
    if (_bDirty)
    {
        RefreshState();

        // Make sure all data is present
        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                cout << "NNNetwork::Train: Attempt to train neural network " << _name << " without providing data sets" << endl;
                cout << "for all input and output layers." << endl;
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }

    // Clear weight velocity vectors if appropriate
    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (uint32_t i = 0; i < _vWeight.size(); i++)
            _vWeight[i]->ClearVelocity();
        _batches = 0;
    }

    NNFloat total_error_training                            = (NNFloat)0.0;
    NNFloat total_error_regularization                      = (NNFloat)0.0;
    NNFloat average_error_training                          = (NNFloat)FLT_MAX;
    NNFloat average_error_regularization                    = (NNFloat)0.0;
    NNFloat moving_average                                  = (NNFloat)0.0;
    uint32_t brake_steps                                    = 0;
    uint32_t init_steps                                     = 100;

    for (uint32_t epoch = 0; epoch < epochs; epoch++)
    {
        timeval t0;
        gettimeofday(&t0, NULL);
        total_error_training                                = (NNFloat)0.0;
        total_error_regularization                          = (NNFloat)0.0;

        // Generate denoising randoms if denoising is active
        if (_bDenoising)
        {
            for (auto l: _vInputLayer)
            {
                if (l->_bDenoising)
                    l->GenerateDenoisingData();
            }
        }

        // Shuffle indices if shuffle is turned on
        if (_bShuffleIndices)
        {
            ShuffleIndices();
        }

        for (uint32_t pos = 0; pos < GetExamples(); pos += GetBatch())
        {
            SetPosition(pos);
            ClearUpdates();
            PredictTrainingBatch();
            NNFloat error_training, error_regularization, error;
            tie(error_training, error_regularization)       = CalculateError(lambda, lambda1);
            uint32_t minibatch                              = GetBatch();
            if (_examples - pos < minibatch)
                minibatch                                   = _examples - pos;
            total_error_training                           += error_training;
            total_error_regularization                     += error_regularization * minibatch;
            if (_verbose  && getGpu()._id == 0) {
                printf("NNNetwork::Train: Minibatch@%u, average error %f, (%f training, %f regularization), alpha %f\n", pos, error_training / minibatch + error_regularization, error_training / minibatch, error_regularization, alpha);
            }

            // Adjust step size if network is diverging (you should probably reduce the step size instead but let's
            // assume you're pretty much asleep at the wheel and the network has to fend for itself).
            NNFloat step_alpha                              = alpha;
            moving_average                                  = 0.9 * moving_average + 0.1 * error_training;
            if (init_steps == 0)
            {
                if (error_training > 2.0 * moving_average)
                {
                    brake_steps = 25;
                    if (getGpu()._id == 0)
                        printf("NNNetwork::Train: Detected network divergence, attempting recovery.\n");
                }
            }
            else
                init_steps--;

            // Reduce step size while braking is active
            if (brake_steps > 0)
            {
                step_alpha                                 *= (NNFloat)0.1;
                brake_steps--;
            }

            // Calculate Gradients then update weights
            if (brake_steps < 24)
            {
                BackPropagate(alpha);
                UpdateWeights(step_alpha, lambda, lambda1, mu, mu1);
            }

#if 0
            static const int WSIZE = 32;
            if (getGpu()._id == 0)
            {
                vector<NNFloat> vGrad(WSIZE);
                for (auto w : _vWeight)
                {
                    cudaMemcpy(vGrad.data(), w->_pbWeight->_pDevData, WSIZE * sizeof(NNFloat), cudaMemcpyDefault);
                    printf("WG %s %s\n", w->_inputLayer._name.c_str(), w->_outputLayer._name.c_str());
                    for (int i = 0; i < WSIZE; i++)
                        printf("%10.6f ", vGrad[i]);
                    printf("\n");
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
           //getGpu().Shutdown();
           //exit(-1);
        }
        timeval t1;
        gettimeofday(&t1, NULL);
        average_error_training                              = total_error_training / GetExamples();
        average_error_regularization                        = total_error_regularization / GetExamples();
        if (getGpu()._id == 0)
            printf("NNNetwork::Train: Epoch %d, average error %f, average training error %f, average regularization error %f, elapsed time %fs\n", ++_epochs,
                    average_error_training + average_error_regularization,
                    average_error_training, average_error_regularization,
                    elapsed_time(t1, t0));

        // Check for checkpoint
        if (_checkpoint_interval > 0)
        {
            _checkpoint_epochs++;
            if (_checkpoint_epochs >= _checkpoint_interval)
            {
                string filename = _checkpoint_name + to_string(_epochs) + ".nc";
                if (getGpu()._id == 0)
                    printf("NNNetwork::Train: saving checkpoint %s\n", filename.c_str());

                SaveNetCDF(filename);
                _checkpoint_epochs                          = 0;
            }
        }
    }

    return average_error_training + average_error_regularization;
}

void NNNetwork::ClearUpdates()
{
    for (auto w: _vWeight)
    {
        w->_updateCount                     = 0;
    }

    for (auto l: _vLayer)
        l->ClearUpdates();
}

tuple<NNFloat, NNFloat> NNNetwork::CalculateError(NNFloat lambda, NNFloat lambda1)
{
    NNFloat error_training                  = (NNFloat)0.0;
    NNFloat error_regularization            = (NNFloat)0.0;

    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;

    // Calculate loss error
    for (auto l: _vOutputLayer)
    {
        error_training                     += l->CalculateError(_position, batch, _errorFunction);
    }

    // Calculate regularization error
    if (lambda > (NNFloat)0.0)
    {
        for (auto w: _vWeight)
        {
            error_regularization           += w->CalculateRegularizationError(lambda, lambda1);
        }
    }

    // Reduce results if running multi-GPU
    if (getGpu()._numprocs > 1)
    {
        double derror_training              = error_training;
        double derror_regularization        = error_regularization;
        MPI_Allreduce(MPI_IN_PLACE, &derror_training, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &derror_regularization, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error_training                      = derror_training;
        error_regularization                = derror_regularization;
    }
    return make_tuple(error_training, error_regularization);
}

// Calculates deltas and gradients working backwards (output deltas are always 100% local)
void NNNetwork::BackPropagate(NNFloat alpha)
{
    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;

    for (auto l: _vBPOrder)
    {
        switch (l->_kind)
        {
            case NNLayer::Kind::Output:
                l->CalculateOutputDelta(_position, batch, _errorFunction);
                l->BackPropagate(_position, batch, alpha);
                break;

            case NNLayer::Kind::Hidden:
                l->BackPropagate(_position, batch, alpha);
                break;
        }
    }
}

void NNNetwork::UpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1)
{
    // Calculate batch size
    uint32_t batch                          = _batch;
    if (_position + batch > _examples)
        batch                               = _examples - _position;
        
    // Increment batch count (do this before weight update in case using Adam optimizer)
    _batches++;        

    for (int64_t i = _vWeight.size() - 1; i >= 0; i--)
    {
        _vWeight[i]->UpdateWeights(_trainingMode, batch, alpha, lambda, lambda1, mu, mu1, _batches);
    }

    // Update batch normalization parameters
}

// Returns top k outputs of any layer (where k <= 128) along with their indices
void NNNetwork::CalculateTopK(const string& layer, uint32_t k, GpuBuffer<NNFloat>* pbKey, GpuBuffer<unsigned int>* pbValue)
{
    // Find desired layer
    NNLayer* pLayer         = _mLayer[layer];
    if (pLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::CalculateTopK: Unknown layer %s.\n", layer.c_str());
        return;
    }
    else if (k > 128)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::CalculateTopK: Can only calculate 128 or fewer elements.\n");
        return;
    }
    else if (k > pLayer->_Nx * pLayer->_Ny * pLayer->_Nz)
    {
        if (getGpu()._id == 0)
            printf("NNNetwork::CalculateTopK: Layer has fewer elements than k (%u vs %u).\n", k, pLayer->_Nx * pLayer->_Ny * pLayer->_Nz);
        return;
    }

    // Call kernel and return
    uint32_t batch          = _batch;
    if (_position + batch > _examples)
        batch               = _examples - _position;
    kCalculateTopK(pLayer->_pbUnit->_pDevData, pbKey->_pDevData, pbValue->_pDevData, batch, pLayer->_localStride, k);

    return;
}

bool NNNetwork::SaveNetCDF(const string& fname)
{
    bool bResult                            = true;

    // Unshard weights and biases to local copy
    vector< vector<NNFloat> > vvWeight;
    vector< vector<NNFloat> > vvBias;
    for (auto w : _vWeight)
    {
        // Download weights to local copy on process 0
        vector<NNFloat> vWeight;
        vector<NNFloat> vBias;

        // BUG need to account for multi-GPU conv layers and biases
        if (!w->_bShared)
        {
            w->_pbWeight->Download(w->_vWeight.data());

            if (getGpu()._numprocs == 1)
            {
                vWeight                         = w->_vWeight;
            }
            else
            {
                uint32_t outgoingSize           = w->_outputLayer._stride * 3;
                uint32_t incomingSize           = w->_inputLayer._stride * 2;
                if (getGpu()._id == 0)
                {
                    vWeight.resize(w->_outputLayer._stride * w->_inputLayer._stride);
                    NNFloat* pWeight            = vWeight.data();
                    if (outgoingSize > incomingSize)
                    {
                        cudaMemcpy2D(pWeight, w->_outputLayer._stride * sizeof(NNFloat), w->_vWeight.data(), w->_outputLayer._localStride * sizeof(NNFloat), w->_outputLayer._localStride * sizeof(NNFloat), w->_inputLayer._stride, cudaMemcpyDefault);
                        pWeight                += w->_outputLayer._localStride;
                        for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                        {
                            uint64_t size;
                            MPI_Status status;
                            MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                            vector<NNFloat> vTemp(size);
                            MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                            uint64_t lstride    = size / w->_inputLayer._stride;
                            NNFloat* pSrcWeight = vTemp.data();
                            NNFloat* pDstWeight = pWeight;
                            for (uint32_t j = 0; j < w->_inputLayer._stride; j++)
                            {
                                memcpy(pDstWeight, pSrcWeight, lstride * sizeof(NNFloat));
                                pSrcWeight     += lstride;
                                pDstWeight     += w->_outputLayer._stride;
                            }
                            pWeight            += lstride;
                        }
                    }
                    else
                    {
                        cudaMemcpy(pWeight, w->_vWeight.data(), w->_outputLayer._stride * w->_inputLayer._localStride * sizeof(NNFloat), cudaMemcpyDefault);
                        pWeight                += w->_outputLayer._stride * w->_inputLayer._localStride;
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
                    uint64_t size       = w->_vWeight.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(w->_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                }
            }
        }

        // Download biases to local copy on process 0
        w->_pbBias->Download(w->_vBias.data());
        if (getGpu()._id == 0)
        {
            vBias                   = w->_vBias;
            vBias.resize(w->_outputLayer._stride);
            uint64_t offset         = w->_vBias.size();
            for (size_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(vBias.data() + offset, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                offset             += size;
            }
        }
        else
        {
            uint64_t size           = w->_vBias.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(w->_vBias.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // Add to growing weight and bias lists
        vvWeight.push_back(vWeight);
        vvBias.push_back(vBias);
    }

    // Open output file
    if (getGpu()._id == 0)
    {
        try
        {
            NcFile nc(fname, NcFile::replace);

            // Write descriptive values
            nc.putAtt("version", ncFloat, NN_VERSION);
            nc.putAtt("name", _name);
            nc.putAtt("kind", ncUint, _kind);
            nc.putAtt("errorFunction", ncUint, _errorFunction);
            nc.putAtt("maxout_k", ncInt, _maxout_k);
            nc.putAtt("LRN_k", ncFloat, _LRN_k);
            nc.putAtt("LRN_n", ncInt, _LRN_n);
            nc.putAtt("LRN_alpha", ncFloat, _LRN_alpha);
            nc.putAtt("LRN_beta", ncFloat, _LRN_beta);
            nc.putAtt("bSparsenessPenalty", ncUint, (uint32_t)_bSparsenessPenalty);
            nc.putAtt("sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
            nc.putAtt("sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);
            nc.putAtt("bDenoising", ncUint, (uint32_t)_bDenoising);
            nc.putAtt("denoising_p", ncFloat, _denoising_p);
            nc.putAtt("deltaBoost_one", ncFloat, _deltaBoost_one);
            nc.putAtt("deltaBoost_zero", ncFloat, _deltaBoost_zero);
            nc.putAtt("SMCE_oneScale", ncFloat, _SMCE_oneScale);
            nc.putAtt("SMCE_zeroScale", ncFloat, _SMCE_zeroScale);
            nc.putAtt("SMCE_oneTarget", ncFloat, _SMCE_oneTarget);
            nc.putAtt("SMCE_zeroTarget", ncFloat, _SMCE_zeroTarget);
            nc.putAtt("ShuffleIndices", ncUint, (uint32_t)_bShuffleIndices);
            nc.putAtt("checkpoint_name", _checkpoint_name);
            nc.putAtt("checkpoint_interval", ncInt, _checkpoint_interval);
            nc.putAtt("checkpoint_epochs", ncInt, _checkpoint_epochs);

            // Write Layers
            nc.putAtt("layers", ncUint, (uint32_t)_vLayer.size());
            for (uint32_t i = 0; i < _vLayer.size(); i++)
                _vLayer[i]->WriteNetCDF(nc, i);

            // Write weights
            nc.putAtt("weights", ncUint, (uint32_t)_vWeight.size());
            for (uint32_t i = 0; i < _vWeight.size(); i++)
                _vWeight[i]->WriteNetCDF(nc, i, vvWeight[i].data(), vvBias[i].data());
        }
        catch (NcException& e)
        {
            printf("NNNetwork::SaveNetCDF Error opening binary output file %s to save neural network %s.\n", fname.c_str(), _name.c_str());
            bResult             = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    return bResult;
}

vector<string> NNNetwork::GetLayers() const
{
    vector<string> vResult;
    for (auto l: _vLayer)
    {
        vResult.push_back(l->_name);
    }

    return vResult;
}


NNFloat* NNNetwork::GetUnitBuffer(const string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetUnitBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return NULL;
    }

    return itr->second->GetUnitBuffer();
}

NNFloat* NNNetwork::GetDeltaBuffer(const string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return NULL;
    }

    return itr->second->GetDeltaBuffer();
}

uint64_t NNNetwork::GetBufferSize(const string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return 0;
    }

    return itr->second->GetBufferSize();
}

const NNWeight* NNNetwork::GetWeight(const string& inputLayer, const string& outputLayer) const
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }

        return NULL;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetWeight: Unknown output layer %s.\n", outputLayer.c_str());
        }

        return NULL;
    }

    const NNLayer *pInputLayer = inputLayerItr->second;
    const NNLayer *pOutputLayer = outputLayerItr->second;

    // Search for matching set of weights
    for (auto p: _vWeight)
    {
        if ((&(p->_inputLayer) == pInputLayer) && (&(p->_outputLayer) == pOutputLayer))
        {
            return p;
        }
    }

    // Report failure to find weights connecting layers
    if (getGpu()._id == 0)
    {
        printf("NNNetwork::GetWeight: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return NULL;
}

NNFloat* NNNetwork::GetWeightBuffer(const string& inputLayer, const string& outputLayer) const
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }

        return NULL;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("NNNetwork::GetWeightBuffer: Unknown output layer %s.\n", outputLayer.c_str());
        }
        return NULL;
    }

    const NNLayer *pInputLayer = inputLayerItr->second;
    const NNLayer *pOutputLayer = outputLayerItr->second;

    // Search for matching set of weights
    for (auto p: _vWeight)
    {
        if ((&(p->_inputLayer) == pInputLayer) && (&(p->_outputLayer) == pOutputLayer))
        {
            return p->_vWeight.data();
        }
    }

    // Report failure to find weights connecting layers
    if (getGpu()._id == 0)
    {
        printf("NNNetwork::GetWeightBuffer: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return NULL;
}

NNNetwork::~NNNetwork()
{
    // Delete P2P data
    DeallocatePeerBuffers();

    // Delete weights
    for (uint32_t i = 0; i < _vWeight.size(); i++)
        delete _vWeight[i];

    // Delete layers
    for (uint32_t i = 0; i < _vLayer.size(); i++)
        delete _vLayer[i];
}

uint32_t CalculateConvolutionDimensions(uint32_t width, uint32_t filter, uint32_t stride)
{
    if (width <= filter)
        return 1;
    else if (stride == 1)
        return width;
    else
        return (width - filter) / stride + 1;
}

// Automagically calculates convolution and pooling layer dimensions from inputs
void CalculateDerivedLayerDimensions(NNNetworkDescriptor& d)
{
    map <NNLayerDescriptor*, bool> mbDimensionsCalculated;                      // Holds flag indicated whether dimensions are known
    map <string, NNLayerDescriptor*> mLayer;                                    // Holds flag indicated whether dimensions are known

    // Input, Output, and fully connected layers are determined from spec
    for (size_t i = 0; i < d._vLayerDescriptor.size(); i++)
    {
        NNLayerDescriptor* pL               = &(d._vLayerDescriptor[i]);
        bool bFlag                          = true;
        if  ((pL->_kind == NNLayer::Kind::Hidden) &&
            ((pL->_type == NNLayer::Type::Pooling) || (pL->_type == NNLayer::Type::Convolutional)))
            bFlag = false;
        mbDimensionsCalculated[pL]          = bFlag;
        mLayer[pL->_name]                   = pL;
    }

    // Loop repeatedly until all network layers determined
    bool bFinished;
    do {
        bFinished = true;

        for (size_t i = 0; i < d._vLayerDescriptor.size(); i++)
        {
            NNLayerDescriptor* pL               = &(d._vLayerDescriptor[i]);
            bool bPooling                       = pL->_type == NNLayer::Type::Pooling;
            bool bLRN                           = bPooling && (pL->_poolingFunction == PoolingFunction::LRN);
            bool bDotProduct                    = bPooling && (pL->_poolingFunction == PoolingFunction::DotProduct);

            if (!mbDimensionsCalculated[pL])
            {
                bool bAllInputsCalculated   = true;
                for (auto s : pL->_vSource)
                {
                    NNLayerDescriptor* pS = mLayer[s];
                    bAllInputsCalculated  &= mbDimensionsCalculated[pS];
                }

                // Skip if not all source dimensions are known
                if (!bAllInputsCalculated)
                {
                    bFinished               = false;
                    continue;
                }

                // Check for consistent sizing from all inputs
                bool bSized = false;
                NNLayerDescriptor* pL0      = mLayer[pL->_vSource[0]];
                uint32_t N                  = pL->_Nx;
                uint32_t oldNx              = bDotProduct ? pL0->_Nx : 1;
                uint32_t oldNy              = bDotProduct ? pL0->_Ny : 1;
                uint32_t oldNz              = bDotProduct ? pL0->_Nz : 1;
                uint32_t nx                 = bDotProduct ? pL->_vSource.size() - 1 : 1;
                uint32_t ny                 = 1;
                uint32_t nz                 = 1;
                uint32_t nw                 = 1;
                for (auto s : pL->_vSource)
                {
                    NNLayerDescriptor* pS = mLayer[s];
                    //printf("L: %s S: %s %u %u %u %u\n", pL->_name.c_str(), pS->_name.c_str(), pS->_Nx, pS->_Ny, pS->_Nz, pS->_Nw);
                    
                    if (bDotProduct)
                    {   
                        if ((oldNx != pS->_Nx) || (oldNy != pS->_Ny) || (oldNz != pS->_Nz))
                        {
                            if (getGpu()._id == 0)
                                printf("NNNetwork::CalculateDerivedLayerDimensions: Inconsistent incoming data size for dot product layer %s\n", pL->_name.c_str());
                            getGpu().Shutdown();
                            exit(-1);                                               
                        }
                    }
                    else
                    {
                        if (!bLRN)
                        {
                            nx                      = CalculateConvolutionDimensions(pS->_Nx, pL->_kernelX, pL->_kernelStrideX);
                            ny                      = CalculateConvolutionDimensions(pS->_Ny, pL->_kernelY, pL->_kernelStrideY);
                            nz                      = CalculateConvolutionDimensions(pS->_Nz, pL->_kernelZ, pL->_kernelStrideZ);
                            nw                      = pS->_Nw;
                            if (bPooling)
                                pL->_dimensions     = pS->_dimensions;
                        }
                        else
                        {
                            nx                      = pS->_Nx;
                            ny                      = pS->_Ny;
                            nz                      = pS->_Nz;
                            nw                      = pS->_Nw;
                            pL->_dimensions         = pS->_dimensions;
                        }

                        // Calculate padding
                        switch (pL->_kernelDimensions)
                        {
                            case 3:
                                if (pS->_Nz < pL->_kernelZ)
                                {
                                    pL->_kernelPaddingZ = (pL->_kernelZ - pS->_Nz + 1) / 2;
                                }
                                else if (pL->_kernelStrideZ == 1)
                                {
                                    pL->_kernelPaddingZ = pL->_kernelZ / 2;
                                }

                            case 2:
                                if (pS->_Ny < pL->_kernelY)
                                {
                                    pL->_kernelPaddingY = (pL->_kernelY - pS->_Ny + 1) / 2;
                                }
                                else if (pL->_kernelStrideY == 1)
                                {
                                    pL->_kernelPaddingY = pL->_kernelY / 2;
                                }

                            case 1:
                                if (pS->_Nx < pL->_kernelX)
                                {
                                    pL->_kernelPaddingX = (pL->_kernelX - pS->_Nx + 1) / 2;
                                }
                                else if (pL->_kernelStrideX == 1)
                                {
                                    pL->_kernelPaddingX = pL->_kernelX / 2;
                                }
                        }

                        // Check for consistency
                        if (bSized)
                        {
                            if ((nx != oldNx) || (ny != oldNy) || (nz != oldNz))
                            {
                                if (getGpu()._id == 0)
                                    printf("NNNetwork::CalculateDerivedLayerDimensions: Inconsistent incoming data size for convolution layer %s\n", pL->_name.c_str());
                                getGpu().Shutdown();
                                exit(-1);
                            }
                        }
                        bSized                      = true;
                        oldNx                       = nx;
                        oldNy                       = ny;
                        oldNz                       = nz;
                        mbDimensionsCalculated[pL]  = true;
                    }
                }
                pL->_Nx                         = nx;
                pL->_Ny                         = ny;
                pL->_Nz                         = nz;
                pL->_Nw                         = nw;
                if (!bPooling)
                {
                    switch (pL->_kernelDimensions)
                    {
                        case 1:
                            pL->_Ny             = N;
                            pL->_dimensions     = 2;
                            break;

                        case 2:
                            pL->_Nz             = N;
                            pL->_dimensions     = 3;
                            break;

                        case 3:
                            pL->_Nw             = N;
                            pL->_dimensions     = 4;
                            break;
                    }
                }

                //printf("L %s: %d | %d %d %d %d\n", pL->_name.c_str(), pL->_dimensions, pL->_Nx, pL->_Ny, pL->_Nz, pL->_Nw);

            }
        }
    }
    while (!bFinished);
    //exit(-1);
}

void NNNetwork::CalculatePropagationOrder()
{
    // Comparator for layer priorites
    struct CompareLayer {
        bool operator()(NNLayer* l1 , NNLayer* l2)
        {
           return (l1->_priority < l2->_priority);
        }
    };

    // Initialize FP priorities
    for (auto p : _vLayer)
    {
        p->_priority            = (p->_kind == NNLayer::Kind::Input) ? 0 : -1;
    }


    // Create FP queue
    priority_queue<NNLayer*, vector<NNLayer*>, CompareLayer> pqueue;
    for (auto p : _vInputLayer)
    {
        pqueue.push(p);
    }


    // Walk forward through network, setting priority of each layer to distance from furthest input
    while (!pqueue.empty())
    {
        NNLayer* pLayer         = pqueue.top();
        pqueue.pop();

        int32_t priority       = pLayer->_priority + 1;
        for (auto p : pLayer->_vOutgoingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority    = priority;
                pqueue.push(p);
            }
        }

        // Handle skip layers
        for (auto p : pLayer->_vOutgoingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority    = priority;
                pqueue.push(p);
            }
        }
    }

    // Create forward propagation list
    _vFPOrder.resize(0);
    for (auto p : _vLayer)
    {
        _vFPOrder.push_back(p);
    }
    sort(_vFPOrder.begin(), _vFPOrder.end(), CompareLayer());
    //for (auto p : _vFPOrder)
    //    cout << p->_name << endl;

    // Initialize BP priorities
    for (auto p : _vLayer)
    {
        p->_priority            = (p->_kind == NNLayer::Kind::Output) ? 0 : -1;
    }

    // Create BP queue
    for (auto p : _vOutputLayer)
    {
        pqueue.push(p);
    }

    // Walk backwards through network, setting priority of each layer to distance from furthest output
    while (!pqueue.empty())
    {
        NNLayer* pLayer         = pqueue.top();
        pqueue.pop();
        int32_t priority       = pLayer->_priority + 1;
        for (auto p : pLayer->_vIncomingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority    = priority;
                pqueue.push(p);
            }
        }

        // Handle skip layers
        for (auto p : pLayer->_vIncomingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority    = priority;
                pqueue.push(p);
            }
        }
    }

    // Create backpropagation list
    _vBPOrder.resize(0);
    for (auto p : _vLayer)
    {
        _vBPOrder.push_back(p);
    }

    sort(_vBPOrder.begin(), _vBPOrder.end(), CompareLayer());
    //for (auto p : _vBPOrder)
    //    cout << p->_name << endl;
    //exit(-1);
}

// Validates network gradients numerically
bool NNNetwork::Validate()
{
    // below parameters are used only for numerical gradient validation (non centered formula),
    // that is why neural network will be tested only in SGD mode
    bool result                 = true;
    const NNFloat delta         = (NNFloat)0.001;
    const NNFloat alpha         = (NNFloat)1.0;
    const NNFloat lambda        = (NNFloat)0.0; // regularization parameter (no need for bias test)
    const NNFloat lambda1       = (NNFloat)0.0; // regularization parameter (no need for bias test)
    const NNFloat mu            = (NNFloat)0.0; // no momentum
    const NNFloat mu1           = (NNFloat)0.0; // no momentum

    // There are couple of issues with numerical gradient validation:
    // The deeper network the higher numerical error in the cost function evaluation;
    // Current implementation uses non centered derivative formula and it is not the best choice (TODO).
    // Because of these two reasons threshold tuning becomes empirical,
    // and the goal is choose value as small as possible so that probability of unit test failure
    // in case when gradient implementation is correct should be zero
    // and probability of detection of incorrect gradient implementation should be as high as possible.
    // On deeper networks threshold 10 gives false positive failures even though gradient implementation is correct. So I set to 20
    const NNFloat epsilon = delta * 20.f;

    // Only runs in single-processor mode
    if (getGpu()._numprocs > 1)
    {
        cout << "NNNetwork::Validate: Do not call this method from a multi-process run, just don't, mmkay?" << endl;
        return false;
    }

    // Check if already in validate mode
    if (_mode != Validation)
    {
        _mode                                               = Validation;
        _bDirty                                             = true;
    }

    // Refresh state if needed
    if (_bDirty)
    {
        RefreshState();

        // Make sure all data is present
        if (!_bAllDataLoaded)
        {
            cout << "NNNetwork::Validate: Attempt to train neural network " << _name << " without providing data sets" << endl;
            cout << "for all input and output layers." << endl;
            getGpu().Shutdown();
            exit(-1);
        }
    }

    // Clear weight velocity vectors if appropriate
    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (uint32_t i = 0; i < _vWeight.size(); i++)
            _vWeight[i]->ClearVelocity();
    }

    // Shuffle indices if shuffle is turned on
    if (_bShuffleIndices)
    {
        ShuffleIndices();
    }

    cout << "Validating network weights and biases with epsilon error threshold of " << epsilon << endl;

    // Calculate initial Loss
    SetPosition(0);
    ClearUpdates();
    PredictValidationBatch();
    NNFloat initialErrorTraining, initialErrorRegularization, initialError;
    tie(initialErrorTraining, initialErrorRegularization) = CalculateError(lambda, lambda1);
    initialError                                          = initialErrorTraining + initialErrorRegularization;
    cout << "initialErrorTraining " << initialErrorTraining << "; initialErrorRegularization " << initialErrorRegularization << endl;

    // Calculate initial gradients
    BackPropagate(alpha);

    // Save initial weights bias and weights gradients
    vector< vector<NNFloat>> vWeightGradient;
    for (int id =0; id < _vWeight.size(); id++)
    {
        NNWeight* w = _vWeight[id];

        vWeightGradient.push_back(vector<NNFloat>(w->_vWeight.size()));
        w->_pbWeight->Download(w->_vWeight.data());
        w->_pbBias->Download(w->_vBias.data());
        w->_pbWeightGradient->Download(vWeightGradient.back().data());
    }

    // get gradients for bias (bias gradient is not stored, so get it throgh UpdateWeights)
    vector< vector<NNFloat>> vBiasGradient;
    UpdateWeights(alpha, lambda, lambda1, mu, mu1);

    for (int id = 0; id < _vWeight.size(); id++)
    {
        NNWeight* w = _vWeight[id];
        vBiasGradient.push_back(vector<NNFloat>(w->_vBias.size()));
        vector<NNFloat>& bias = vBiasGradient[id];

        // Display information about current weight set
        cout << "Validating weights between layer " << w->_inputLayer._name << " and " << w->_outputLayer._name << endl;

        w->_pbWeight->Upload(w->_vWeight.data()); // restore weights
        // get bias gradient (TODO instead of this hack it is better to have explicit bias gradient)
        vector<NNFloat> bias_g(w->_pbBias->_length);
        w->_pbBias->Download(bias_g.data());
        for (int b = 0; b < bias_g.size(); b++)
        {
          bias[b] = bias_g[b] - w->_vBias[b];
        }
        w->_pbBias->Upload(w->_vBias.data()); // restore bias
    }

    // Now tweak each weight and bias individually to determine change in loss function
    for (int id = 0; id < _vWeight.size(); id++)
    {
        NNWeight* w = _vWeight[id];

        // Display information about current weight set
        cout << "Validating weights between layer " << w->_inputLayer._name << " and " << w->_outputLayer._name << endl;

        cout << "Tweak weights" << endl;
        for (size_t i = 0; i < w->_vWeight.size(); i++)
        {
            NNFloat oldWeight                               = w->_vWeight[i];
            // weight gradient is normalized by -1 / (pSrcWeight->_sharingCount * _batch), so delta is normalized the same way
            w->_vWeight[i]                                 += delta / (_batch * w->_sharingCount);
            w->_pbWeight->Upload(w->_vWeight.data());
            PredictValidationBatch();
            w->_vWeight[i]                                  = oldWeight;
            NNFloat errorTraining, errorRegularization, error;
            tie(errorTraining, errorRegularization)       = CalculateError(lambda, lambda1);
            error                                           = errorTraining + errorRegularization;
            NNFloat dEdW                                    = (error - initialError) / delta;
            NNFloat weightGradient = vWeightGradient[id][i];
            cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                            "; dEdW " << dEdW << "; weightGradient " << weightGradient <<  endl;
            if (fabs(dEdW + weightGradient) > epsilon)
            {
                cout << error << " " << initialError << endl;
                cout << "Failed Weight " << i << " exceeds error threshold: " << dEdW << " vs " << weightGradient << endl;
                result = false;
            }
        }
        w->_pbWeight->Upload(w->_vWeight.data()); // restore original weights

        cout << "Tweak biases" << endl;
        for (size_t i = 0; i < w->_vBias.size(); i++)
        {
            NNFloat oldBias                               = w->_vBias[i];
            // bias gradient is normalized by 1 / (_batch), so delta is normalized the same way
            w->_vBias[i]                                 += delta / (_batch);
            w->_pbBias->Upload(w->_vBias.data());
            PredictValidationBatch();
            w->_vBias[i]                                  = oldBias;
            NNFloat errorTraining, errorRegularization, error;
            tie(errorTraining, errorRegularization)       = CalculateError(lambda, lambda1);
            error                                           = errorTraining + errorRegularization;
            NNFloat dEdb                                    = (error - initialError) / delta;
            NNFloat biasGradient = vBiasGradient[id][i];
            cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                            "; dEdb " << dEdb << "; biasGradient " << biasGradient <<  endl;
            if (fabs(dEdb + biasGradient) > epsilon)
            {
                cout << error << " " << initialError << endl;
                cout << "Failed Bias " << i << " exceeds error threshold: " << dEdb << " vs " << biasGradient << endl;
                result = false;
            }
        }
        w->_pbBias->Upload(w->_vBias.data()); // restore original bias
    }

    return result;
}

void NNNetwork::DeallocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        // make sure buffers aren't in use
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        // Release peer data
        for (size_t i = 0; i < 2; i++)
        {
            if (_pPeerBuffer[i] != NULL)
            {
                cudaError_t status          = cudaIpcCloseMemHandle(_pPeerBuffer[i]);
                RTERROR(status, "NNNetwork::DeallocatePeerBuffers: Error closing IpcMemHandle");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Release local data
        for (size_t i = 0; i < 2; i++)
        {
            _pbP2PBuffer[i].reset();
        }

        // Release MPI work buffer
        _pCPUBuffer.reset();
    }

}

void NNNetwork::AllocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        // Calculate maximum layer stride
        _maxStride                          = 0;
        for (auto w : _vWeight)
        {
            uint32_t stride                 = (w->_outputLayer._stride * 2) > (w->_inputLayer._stride * 2) ? w->_inputLayer._stride : w->_outputLayer._stride;
            if (stride > _maxStride)
            {
                _maxStride                  = stride;
            }
        }
        // Calculate maximum memory
        uint64_t maxMemory                  = _maxStride * _batch;
        if (maxMemory < _examples)
        {
            maxMemory                       = _examples;
        }

        // Allocate local data
        for (size_t i = 0; i < 2; i++)
        {
            _pbP2PBuffer[i].reset(new GpuBuffer<NNFloat>(maxMemory));
        }

        // Gather P2P data
        if (getGpu()._bP2P)
        {
            cudaIpcMemHandle_t* pMemHandle      = new cudaIpcMemHandle_t[2 * getGpu()._numprocs];
            size_t pos                          = getGpu()._id * 2;
            cudaError_t status                  = cudaIpcGetMemHandle(&(pMemHandle[pos]), _pbP2PBuffer[0]->_pDevData);
            RTERROR(status, "NNNetwork::AllocatePeerBuffers: Error getting first P2P IPCMemHandle");
            status                              = cudaIpcGetMemHandle(&(pMemHandle[pos + 1]), _pbP2PBuffer[1]->_pDevData);
            RTERROR(status, "NNNetwork::AllocatePeerBuffers: Error getting second P2P IPCMemHandle");
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pMemHandle, 2 * sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
            unsigned int peer                   = 2 * ((getGpu()._id + getGpu()._numprocs - 1) % getGpu()._numprocs);
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[0]), pMemHandle[peer], cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "NNNetwork::AllocatePeerBuffers: Unable to open first peer IPCMemHandle");
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[1]), pMemHandle[peer + 1], cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "NNNetwork::AllocatePeerBuffers: Unable to open second peer IPCMemHandle");
        }
        else
        {
            _pCPUBuffer.reset(new NNFloat[maxMemory]);
        }
    }
}

// Flips send and receive P2P buffers
void NNNetwork::SwapPeerBuffers()
{
    _sendIndex                              = 1 - _sendIndex;
    _receiveIndex                           = 1 - _receiveIndex;
}

std::pair<NNNetwork::Kind, string> NNNetwork::_sKindPair[] =
{
    std::pair<NNNetwork::Kind, string>(NNNetwork::Kind::FeedForward, "FeedForward"),
    std::pair<NNNetwork::Kind, string>(NNNetwork::Kind::AutoEncoder, "AutoEncoder")
};

std::map<NNNetwork::Kind, string> NNNetwork::_sKindMap =
std::map<NNNetwork::Kind, string>(_sKindPair, NNNetwork::_sKindPair + sizeof(NNNetwork::_sKindPair) / sizeof(NNNetwork::_sKindPair[0]));

ostream& operator<< (ostream& out, NNNetwork::Kind& k)
{
    out << NNNetwork::_sKindMap[k];
    return out;
}

uint32_t MPI_Bcast_NNNetworkDescriptor(NNNetworkDescriptor& d)
{

    // Broadcast network name
    MPI_Bcast_string(d._name);

    // Broadcast network (inefficient, but simple and only done once)
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._errorFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._maxout_k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_k, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bSparsenessPenalty, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDenoising, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._denoising_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaBoost_one, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaBoost_zero, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_oneScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_zeroScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_oneTarget, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_zeroTarget, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._checkpoint_interval, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._checkpoint_epochs, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast_string(d._checkpoint_name);
    MPI_Bcast(&d._bShuffleIndices, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);



    // Broadcast layers
    uint32_t layers                             = d._vLayerDescriptor.size();
    MPI_Bcast(&layers, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vLayerDescriptor.resize(layers);
    for (uint32_t i = 0; i < layers; i++)
    {
        MPI_Bcast_NNLayerDescriptor(d._vLayerDescriptor[i]);
    }

    // Broadcast weights if present
    uint32_t weights                            = d._vWeightDescriptor.size();
    MPI_Bcast(&weights, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vWeightDescriptor.resize(weights);

    for (uint32_t i = 0; i < weights; i++)
    {
        MPI_Bcast_NNWeightDescriptor(d._vWeightDescriptor[i]);
    }

    return 0;
}

NNNetwork* LoadNeuralNetworkJSON(const string& fname, const uint32_t batch, const vector<NNDataSetBase*>& vDataSet)
{
    NNNetwork* pNetwork                             = NULL;
    NNNetworkDescriptor nd;
    Json::Value index;
    Json::Reader reader;
    bool bValid                                     = true;
    bool bWeightsSupplied                           = false;
    string wfname;

    if (getGpu()._id == 0)
    {
        std::ifstream stream(fname, std::ifstream::binary);
        bool parsedSuccess                          = reader.parse(stream, index, false);

        if (!parsedSuccess)
        {
            // Report failures and their locations
            // in the document.
            printf("LoadNeuralNetworkJSON: Failed to parse JSON file: %s, error: %s\n", fname.c_str(), reader.getFormattedErrorMessages().c_str());
            bValid                                  = false;
        }
        else
        {
            // Iterate through network in a case-insensitive manner
            NNFloat version                         = NN_VERSION;
            set<string> sLayer;
            for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
            {
                // Extract JSON object key/value pair
                string name                         = itr.name();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                Json::Value key                     = itr.key();
                Json::Value value                   = *itr;
                string vstring                      = value.isString() ? value.asString() : "";
                std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

                // Read version if present
                if (name.compare("version") == 0)
                {
                    version                         = value.asFloat();
                    if (version < 0.6999)
                    {
                        printf("LoadNeuralNetworkJSON: version %f (must be at least 0.7)\n", version);
                        bValid                      = false;
                        goto exit;
                    }
                }

                // Read name if present
                else if (name.compare("name") == 0)
                {
                    nd._name                        = value.asString();
                }

                // Read kind if present
                else if (name.compare("kind") == 0)
                {
                    if (vstring.compare("feedforward") == 0)
                        nd._kind                    = NNNetwork::Kind::FeedForward;
                    else if (vstring.compare("autoencoder") == 0)
                        nd._kind                    = NNNetwork::Kind::AutoEncoder;
                    else
                    {
                        printf("LoadNeuralNetworkJSON: Invalid network kind: %s\n", value.asString().c_str());
                        bValid                      = false;
                        goto exit;
                    }
                }

                // Read weights data if present
                else if (name.compare("weightsdata") == 0)
                {
                    bWeightsSupplied                = true;
                    wfname                          = value.asString();
                }

                // Read LRN parameters if present
                else if ((name.compare("lrn") == 0) || (name.compare("localresponsenormalization") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("k") == 0)
                            nd._LRN_k               = pvalue.asFloat();
                        else if (pname.compare("n") == 0)
                            nd._LRN_n               = pvalue.asInt();
                        else if (pname.compare("alpha") == 0)
                            nd._LRN_alpha           = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._LRN_beta            = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid LocalResponseNormalization parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read Maxout parameters if present
                else if (name.compare("maxout") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("k") == 0)
                            nd._maxout_k            = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid MaxOut parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read Sparseness parameters if present
                else if (name.compare("sparsenesspenalty") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("p") == 0)
                            nd._sparsenessPenalty_p = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._sparsenessPenalty_beta  = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid SparsenessPenalty parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read denoising parameters if present
                else if (name.compare("denoising") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("p") == 0)
                        {
                            nd._denoising_p         = pvalue.asFloat();
                        }
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid Denoising parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read Delta Boost parameters if present
                else if (name.compare("deltaboost") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("one") == 0)
                            nd._deltaBoost_one      = pvalue.asFloat();
                        else if (pname.compare("zero") == 0)
                            nd._deltaBoost_zero     = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid DeltaBoost parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read ScaledMarginalCrossEntropy parameters if present
                else if ((name.compare("scaledmarginalcrossentropy") == 0) || 
                         (name.compare("datascaledmarginalcrossentropy") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname                = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey            = pitr.key();
                        Json::Value pvalue          = *pitr;
                        if (pname.compare("onescale") == 0)
                            nd._SMCE_oneScale       = pvalue.asFloat();
                        else if (pname.compare("zeroscale") == 0)
                            nd._SMCE_zeroScale      = pvalue.asFloat();
                        else if (pname.compare("onetarget") == 0)
                            nd._SMCE_oneTarget      = pvalue.asFloat();
                        else if (pname.compare("zerotarget") == 0)
                            nd._SMCE_zeroTarget     = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            printf("LoadNeuralNetworkJSON: Invalid ScaledMarginalCrossentropy parameter: %s\n", name.c_str());
                            bValid                      = false;
                            goto exit;
                        }
                    }
                }

                // Read SchuffleIndices parameter if present
                else if (name.compare("shuffleindices") == 0)
                {
                    nd._bShuffleIndices             = value.asBool();
                }

                // Read default ELU parameters
                else if ((name.compare("reluslope") == 0) || (name.compare("slope") == 0))
                {
                    nd._RELUSlope                   = value.asFloat();
                }
                else if (name.compare("elualpha") == 0)
                {
                    nd._ELUAlpha                    = value.asFloat();                    
                }
                else if (name.compare("selulambda") == 0)
                {
                    nd._SELULambda                  = value.asFloat();
                }
                
                // Read error function
                else if (name.compare("errorfunction") == 0)
                {
                    if (vstring.compare("l1") == 0)
                        nd._errorFunction           = ErrorFunction::L1;
                    else if (vstring.compare("l2") == 0)
                        nd._errorFunction           = ErrorFunction::L2;
                    else if (vstring.compare("hinge") == 0)
                        nd._errorFunction           = ErrorFunction::Hinge;
                    else if ((vstring.compare("crossentropy") == 0) || (vstring.compare("cross entropy") == 0))
                        nd._errorFunction           = ErrorFunction::CrossEntropy;
                    else if (vstring.compare("scaledmarginalcrossentropy") == 0)
                        nd._errorFunction           = ErrorFunction::ScaledMarginalCrossEntropy;
                    else if (vstring.compare("datascaledmarginalcrossentropy") == 0)
                        nd._errorFunction           = ErrorFunction::DataScaledMarginalCrossEntropy;
                    else
                    {
                        printf("LoadNeuralNetworkJSON: Invalid error function: %s\n", value.asString().c_str());
                        bValid                      = false;
                        goto exit;
                    }
                }

                // Read layer(s)
                else if (name.compare("layers") == 0)
                {
                    uint32_t size                   = value.isArray() ? value.size() : 1;
                    for (uint32_t i = 0; i < size; i++)
                    {
                        vector<NNWeightDescriptor> vSharedWeight;
                        NNLayerDescriptor ldl;
                        bool bSource                = false;
                        Json::Value layer           = value.isArray() ? value[i] : value;
                        bool bAutoSize              = false;

                        // Determine default layer kind and type
                        if (i == 0)
                            ldl._kind               = NNLayer::Kind::Input;
                        else if (i == size - 1)
                            ldl._kind               = NNLayer::Kind::Output;
                        else
                            ldl._kind               = NNLayer::Kind::Hidden;
                        ldl._type                   = NNLayer::Type::FullyConnected;


                        // Search for supplied layer kind and type because we need to know this before parsing
                        // the remainder of supplied keys
                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            string lname            = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey        = litr.key();
                            Json::Value lvalue      = *litr;

                            // Read kind if present (default: Hidden)
                            if (lname.compare("kind") == 0)
                            {
                                string s            = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("input") == 0)
                                    ldl._kind       = NNLayer::Kind::Input;
                                else if (s.compare("hidden") == 0)
                                    ldl._kind       = NNLayer::Kind::Hidden;
                                else if (s.compare("target") == 0)
                                    ldl._kind       = NNLayer::Kind::Target;
                                else if (s.compare("output") == 0)
                                    ldl._kind       = NNLayer::Kind::Output;
                                else
                                {
                                    printf("LoadNeuralNetworkJSON: Invalid layer kind: %s\n", lvalue.asString().c_str());
                                    bValid          = false;
                                    goto exit;
                                }
                            }

                            // Read type if present (default: FullyConnected)
                            else if (lname.compare("type") == 0)
                            {
                                string s        = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("fullyconnected") == 0)
                                    ldl._type   = NNLayer::Type::FullyConnected;
                                else if (s.compare("convolutional") == 0)
                                    ldl._type   = NNLayer::Type::Convolutional;
                                else if (s.compare("pooling") == 0)
                                    ldl._type = NNLayer::Type::Pooling;
                                else
                                {
                                    printf("LoadNeuralNetworkJSON: Invalid layer type: %s\n", lvalue.asString().c_str());
                                    bValid      = false;
                                    goto exit;
                                }
                            }
                        }

                        // FullyConnected non-pooling Layers have default dimensions, others must be supplied or calculated
                        if ((ldl._type == NNLayer::Type::Pooling) || (ldl._type == NNLayer::Type::Convolutional))
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        // Determine default layer name
                        switch (ldl._kind)
                        {
                            case NNLayer::Kind::Input:
                                ldl._name           = "Input" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case NNLayer::Kind::Hidden:
                                ldl._name           = "Hidden" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case NNLayer::Kind::Output:
                                ldl._name           = "Output" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case NNLayer::Kind::Target:
                                ldl._name           = "Target" + to_string(nd._vLayerDescriptor.size());
                                break;
                        }

                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            string lname            = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey        = litr.key();
                            Json::Value lvalue      = *litr;

                            // Skip what we already know
                            if ((lname.compare("kind") == 0) || (lname.compare("type") == 0))
                            {
                                continue;
                            }

                            // Read name if present
                            if (lname.compare("name") == 0)
                            {
                                ldl._name           = lvalue.asString();
                                if (sLayer.find(ldl._name) != sLayer.end())
                                {
                                    printf("LoadNeuralNetworkJSON: Duplicate layer name detected: %s\n", ldl._name.c_str());
                                    bValid          = false;
                                    goto exit;
                                }
                                sLayer.insert(ldl._name);
                                continue;
                            }

                            if (lname.compare("sparse") == 0)
                            {
                                if (lvalue.asBool())
                                    ldl._attributes|= NNLayer::Attributes::Sparse;
                                continue;
                            }
                            else if (lname.compare("n") == 0)
                            {
                                if (lvalue.isArray())
                                {
                                    if (lvalue.size() < 5)
                                    {
                                        ldl._dimensions     = lvalue.size();
                                        switch (lvalue.size())
                                        {
                                            case 4:
                                                ldl._Nw = lvalue[3].asInt();
                                            case 3:
                                                ldl._Nz = lvalue[2].asInt();
                                            case 2:
                                                ldl._Ny = lvalue[1].asInt();
                                            case 1:
                                                ldl._Nx = lvalue[0].asInt();
                                        }

                                    }
                                    else
                                    {
                                        printf("LoadNeuralNetworkJSON: >4 dimensions detected in layer: %s\n", ldl._name.c_str());
                                        bValid          = false;
                                        goto exit;
                                    }

                                }
                                else if (lvalue.isString())
                                {
                                    string nstring          = lvalue.asString();
                                    std::transform(nstring.begin(), nstring.end(), nstring.begin(), ::tolower);
                                    if ((ldl._kind != NNLayer::Kind::Hidden) && (nstring.compare("auto") == 0))
                                        bAutoSize       = true;
                                    else if (nstring.compare("auto") == 0)
                                    {
                                        printf("LoadNeuralNetworkJSON: Illegal attempt to use auto for hidden layer: %s\n", ldl._name.c_str());
                                        bValid          = false;
                                        goto exit;
                                    }
                                }
                                else
                                {
                                    ldl._Nx             = lvalue.asInt();
                                    ldl._dimensions     = 1;
                                }
                                continue;
                            }
                            else if (lname.compare("pdropout") == 0)
                            {
                                ldl._pDropout           = lvalue.asFloat();
                                continue;
                            }


                            // Read types common present in everything but input layers
                            if (ldl._kind != NNLayer::Kind::Input)
                            {
                                // Read source(s) if present
                                if (lname.compare("source") == 0)
                                {
                                    uint32_t size       = lvalue.isArray() ? lvalue.size() : 1;

                                    // MaxPooling and LRN layers can only have one source
#if 0
                                    if ((ldl._type == NNLayer::Type::Pooling) && (size > 1))
                                    {
                                            printf("LoadNeuralNetworkJSON: Pooling layer %s has multiple sources\n", ldl._name.c_str());
                                            bValid                  = false;
                                            goto exit;
                                    }
#endif

                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSource.push_back(src.asString());
                                        bSource         = true;             // Signal existence of at least one source
                                    }
                                    continue;
                                }

                                else if ((lname.compare("kernel") == 0) || (lname.compare("kernelstride") == 0))
                                {
                                    uint32_t x                  = 1;
                                    uint32_t y                  = 1;
                                    uint32_t z                  = 1;
                                    uint32_t dimensions         = 1;
                                    if (lvalue.isArray())
                                    {
                                        if (lvalue.size() < 4)
                                        {
                                            dimensions          = lvalue.size();
                                            switch (lvalue.size())
                                            {
                                                case 3:
                                                    z           = lvalue[2].asInt();
                                                case 2:
                                                    y           = lvalue[1].asInt();
                                                case 1:
                                                    x           = lvalue[0].asInt();
                                            }
                                        }
                                        else
                                        {
                                            bValid              = false;
                                            goto exit;
                                        }
                                    }
                                    else
                                    {
                                        x                       = lvalue.asInt();
                                    }

                                    // Copy values to kernel or kernel stride
                                    if (lname.compare("kernel") == 0)
                                    {
                                        ldl._kernelX            = x;
                                        ldl._kernelY            = y;
                                        ldl._kernelZ            = z;
                                        ldl._kernelDimensions   = dimensions;
                                    }
                                    else
                                    {
                                        ldl._kernelStrideX      = x;
                                        ldl._kernelStrideY      = y;
                                        ldl._kernelStrideZ      = z;
                                    }
                                    continue;
                                }
                            }




                            // Hidden layer-specific features
                            if (ldl._kind == NNLayer::Kind::Hidden)
                            {
                                // Layer-specific sparse penalty
                                if (lname.compare("sparsenesspenalty") == 0)
                                {
                                    for (Json::ValueIterator pitr = lvalue.begin(); pitr != lvalue.end() ; pitr++)
                                    {
                                        string pname                = pitr.name();
                                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                                        Json::Value pkey            = pitr.key();
                                        Json::Value pvalue          = *pitr;
                                        if (pname.compare("p") == 0)
                                            ldl._sparsenessPenalty_p = pvalue.asFloat();
                                        else if (pname.compare("beta") == 0)
                                            ldl._sparsenessPenalty_beta  = pvalue.asFloat();
                                        else
                                        {
                                            printf("LoadNeuralNetworkJSON: Invalid sparseness penalty parameter for hidden layer %s\n", ldl._name.c_str());
                                            bValid                  = false;
                                            goto exit;
                                        }
                                    }
                                    continue;
                                }

                                // Pooling layer-specific features
                                if (ldl._type == NNLayer::Type::Pooling)
                                {
                                    if (lname.compare("function") == 0)
                                    {
                                        string s          = lvalue.asString();
                                        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                        if (s.compare("max") == 0)
                                            ldl._poolingFunction = PoolingFunction::Max;
                                        else if (s.compare("maxout") == 0)
                                            ldl._poolingFunction = PoolingFunction::Maxout;
                                        else if (s.compare("dotproduct") == 0)
                                            ldl._poolingFunction = PoolingFunction::DotProduct;
                                        else if (s.compare("cosine") == 0)
                                            ldl._poolingFunction = PoolingFunction::Cosine;
                                        else if (s.compare("average") == 0)
                                            ldl._poolingFunction = PoolingFunction::Average;
                                        else if ((s.compare("lrn") == 0) || (s.compare("localresponsenormalization") == 0))
                                            ldl._poolingFunction = PoolingFunction::LRN;
                                        else
                                        {
                                            printf("LoadNeuralNetworkJSON: Invalid pooling function (%s) for pooling layer %s\n", lvalue.asString().c_str(), ldl._name.c_str());
                                            bValid                  = false;
                                            goto exit;
                                        }
                                        continue;
                                    }
                                }
                            }

                            // Output layer-specific features
                            if (ldl._kind == NNLayer::Kind::Output)
                            {

                            }

                            // Hidden and output layer-specific features
                            if ((ldl._kind == NNLayer::Kind::Hidden) || (ldl._kind == NNLayer::Kind::Output))
                            {
                                // Read skip(s) if present
                                if (lname.compare("skip") == 0)
                                {
                                    uint32_t size       = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSkip.push_back(src.asString());
                                    }
                                    continue;
                                }

                                // Read activation if present
                                else if (lname.compare("activation") == 0)
                                {
                                    string s        = lvalue.asString();
                                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                    if (s.compare("sigmoid") == 0)
                                        ldl._activation = Activation::Sigmoid;
                                    else if (s.compare("tanh") == 0)
                                        ldl._activation = Activation::Tanh;
                                    else if (s.compare("linear") == 0)
                                        ldl._activation = Activation::Linear;
                                    else if ((s.compare("relu") == 0) || (s.compare("rectifiedlinear") == 0))
                                        ldl._activation = Activation::RectifiedLinear;
                                    else if ((s.compare("lrelu") == 0) || (s.compare("leakyrectifiedlinear") == 0))
                                        ldl._activation = Activation::LeakyRectifiedLinear;
                                    else if ((s.compare("elu") == 0) || (s.compare("exponentiallinear") == 0))
                                        ldl._activation = Activation::ExponentialLinear;
                                    else if ((s.compare("selu") == 0) || (s.compare("scaledexponentiallinear") == 0))
                                        ldl._activation = Activation::ScaledExponentialLinear;                                        
                                    else if (s.compare("softplus") == 0)
                                        ldl._activation = Activation::SoftPlus;
                                    else if (s.compare("softsign") == 0)
                                        ldl._activation = Activation::SoftSign;
                                    else if (s.compare("softmax") == 0)
                                        ldl._activation = Activation::SoftMax;
                                    else if (s.compare("relumax") == 0)
                                        ldl._activation = Activation::RELUMax;
                                    else if (s.compare("linearmax") == 0)
                                        ldl._activation = Activation::LinearMax;
                                    else
                                    {
                                        printf("LoadNeuralNetworkJSON: Invalid layer activation: %s\n", lvalue.asString().c_str());
                                        bValid          = false;
                                        goto exit;
                                    }
                                    continue;
                                }
                                
                                // Read layer-specific ELU parameters
                                else if ((lname.compare("reluslope") == 0) || (lname.compare("slope") == 0))
                                {
                                    ldl._RELUSlope              = lvalue.asFloat();
                                    continue;
                                }
                                else if (lname.compare("elualpha") == 0)
                                {
                                    ldl._ELUAlpha               = lvalue.asFloat();
                                    continue;                    
                                }
                                else if (lname.compare("selulambda") == 0)
                                {
                                    ldl._SELULambda             = lvalue.asFloat();
                                    continue;
                                }
                                
                                // Weight normalization
                                else if (lname.compare("weightnorm") == 0)
                                {
                                    ldl._weightNorm             = lvalue.asFloat();
                                    continue;
                                }

                                // Read delta normalization cap if active
                                else if (lname.compare("deltanorm") == 0)
                                {
                                    ldl._deltaNorm              = lvalue.asFloat();
                                    continue;
                                }

                                // Read weight initialization scheme
                                else if (lname.compare("weightinit") == 0)
                                {
                                    for (int i = 0; i < lvalue.size(); i++)
                                    {
                                        for (Json::ValueIterator witr = lvalue.begin(); witr != lvalue.end() ; witr++)
                                        {
                                            string wname                = witr.name();
                                            std::transform(wname.begin(), wname.end(), wname.begin(), ::tolower);
                                            Json::Value wkey            = witr.key();
                                            Json::Value wvalue          = *witr;

                                            if (wname.compare("scheme") == 0)
                                            {
                                                string scheme           = wvalue.asString();
                                                std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::tolower);
                                                if (scheme.compare("xavier") == 0)
                                                    ldl._weightInit     = Xavier;
                                                else if (scheme.compare("caffexavier") == 0)
                                                    ldl._weightInit     = CaffeXavier;
                                                else if (scheme.compare("gaussian") == 0)
                                                    ldl._weightInit     = Gaussian;
                                                else if (scheme.compare("uniform") == 0)
                                                    ldl._weightInit     = Uniform;
                                                else if (scheme.compare("unitball") == 0)
                                                    ldl._weightInit     = UnitBall;
                                                else if (scheme.compare("constant") == 0)
                                                    ldl._weightInit     = Constant;
                                                else if (scheme.compare("selu") == 0)
                                                    ldl._weightInit     = SELU;                                                    
                                                else
                                                {
                                                    printf("LoadNeuralNetworkJSON: Invalid weight initialization scheme: %s\n", scheme.c_str());
                                                    bValid          = false;
                                                    goto exit;
                                                }
                                            }
                                            else if (wname.compare("scale") == 0)
                                            {
                                               ldl._weightInitScale     = wvalue.asFloat();
                                            }
                                            else if (wname.compare("bias") == 0)
                                            {
                                               ldl._biasInit            = wvalue.asFloat();
                                            }
                                            else
                                            {
                                                printf("LoadNeuralNetworkJSON: Invalid weight initialization field: %s\n", wname.c_str());
                                                bValid                  = false;
                                                goto exit;
                                            }
                                        }
                                    }
                                    continue;
                                }

                                // Read shared weight entry
                                else if (lname.compare("sharedweights") == 0)
                                {
                                    uint32_t size                       = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t i = 0; i < size; i++)
                                    {
                                        NNWeightDescriptor nd;
                                        Json::Value share   = lvalue.isArray() ? lvalue[i] : lvalue;
                                        for (Json::ValueIterator sitr = share.begin(); sitr != share.end() ; sitr++)
                                        {
                                            string sname                = sitr.name();
                                            std::transform(sname.begin(), sname.end(), sname.begin(), ::tolower);
                                            Json::Value skey            = sitr.key();
                                            Json::Value svalue          = *sitr;

                                            if (sname.compare("sourceinputlayer") == 0)
                                            {
                                                nd._sourceInputLayer    = svalue.asString();
                                            }
                                            else if (sname.compare("sourceoutputlayer") == 0)
                                            {
                                                nd._sourceOutputLayer   = svalue.asString();
                                            }
                                            else if (sname.compare("inputlayer") == 0)
                                            {
                                                nd._inputLayer          = svalue.asString();
                                            }
                                            else if (sname.compare("transposed") == 0)
                                            {
                                                nd._bTransposed         = svalue.asBool();
                                            }
                                            else
                                            {
                                                printf("LoadNeuralNetworkJSON: Invalid shared weight field: %s\n", sname.c_str());
                                                bValid                  = false;
                                                goto exit;
                                            }
                                        }
                                        nd._bShared                     = true;
                                        vSharedWeight.push_back(nd);
                                    }
                                    continue;
                                }
                            }


                            // Input and output layer-specific features
                            if ((ldl._kind == NNLayer::Kind::Input) || (ldl._kind == NNLayer::Kind::Output))
                            {
                                if (lname.compare("dataset") == 0)
                                {
                                    ldl._dataSet                        = lvalue.asString();
                                    continue;
                                }

                            }

                            // If we reach here, we didn't recognize the field
                            printf("LoadNeuralNetworkJSON: Unknown neural network layer field: %s\n", lname.c_str());
                            bValid                                      = false;
                            goto exit;
                        }
                        
                        // Automagically determine dimensions of input or output units
                        if (bAutoSize)
                        {
                            bool bFound                                 = false;
                            for (auto p : vDataSet)
                            {
                                if (p->_name.compare(ldl._dataSet) == 0)
                                {
                                    ldl._Nx                             = p->_width;
                                    ldl._Ny                             = p->_height;
                                    ldl._Nz                             = p->_length;
                                    ldl._dimensions                     = p->_dimensions;
                                    bFound                              = true;
                                }
                            }
                            if (!bFound)
                            {
                                printf("LoadNeuralNetworkJSON: Unable to find data set %s to determine dimensions for layer: %s\n", ldl._dataSet.c_str(), ldl._name.c_str());
                                bValid                                  = false;
                                goto exit;
                            }
                        }

                        // Add default source to hidden and output layers if none supplied
                        if (!bSource && (ldl._kind != NNLayer::Kind::Input))
                        {
                            ldl._vSource.push_back(nd._vLayerDescriptor.back()._name);
                        }
                                               
                        // Automagically compute dimensions of dot-product pooling layer (harmless BUG maybe?  Resolve)
                        if ((ldl._type == NNLayer::Type::Pooling) && 
                            (ldl._poolingFunction == PoolingFunction::DotProduct) || (ldl._poolingFunction == PoolingFunction::Cosine))
                        {
                            // Make sure dot product has 2 or more sources
                            if (ldl._vSource.size() < 2)
                            {
                                printf("LoadNeuralNetworkJSON: Dot product layer %s must have 2 or more sources\n", ldl._name.c_str());
                                bValid                                  = false;
                                goto exit;                            
                            }
                            ldl._Nx                                     = ldl._vSource.size() - 1;
                            ldl._Ny                                     = 1;
                            ldl._Nz                                     = 1;
                            ldl._dimensions                             = 1;
                        }                       

                        // Add weight descriptors to non-pooling layers
                        if (ldl._type != NNLayer::Type::Pooling)
                        {

                            uint32_t sharedWeightsFound         = 0;
                            for (uint32_t i = 0; i < ldl._vSource.size(); i++)
                            {
                                NNWeightDescriptor wd;
                                wd._inputLayer                  = ldl._vSource[i];
                                wd._outputLayer                 = ldl._name;
                                wd._norm                        = ldl._weightNorm;

                                // Search for shared weights
                                for (uint32_t j = 0; j < vSharedWeight.size(); j++)
                                {
                                    // Copy shared bits if match is located
                                    if (vSharedWeight[j]._inputLayer == wd._inputLayer)
                                    {
                                        wd._bShared             = true;
                                        wd._bTransposed         = vSharedWeight[j]._bTransposed;
                                        wd._sourceInputLayer    = vSharedWeight[j]._sourceInputLayer;
                                        wd._sourceOutputLayer   = vSharedWeight[j]._sourceOutputLayer;
                                        sharedWeightsFound++;
                                        break;
                                    }
                                }
                                nd._vWeightDescriptor.push_back(wd);
                            }

                            // Guarantee all shared weights were found
                            if (sharedWeightsFound < vSharedWeight.size())
                            {
                                printf("LoadNeuralNetworkJSON: Unable to locate all shared weights\n");
                                bValid                          = false;
                                goto exit;
                            }
                        }

                        // Determine if full layer dimensions have been provided or they need to
                        // be calculated from all sources
                        if (ldl._dimensions < ldl._kernelDimensions)
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        nd._vLayerDescriptor.push_back(ldl);
                    }
                }

                else
                {
                    printf("LoadNeuralNetworkJSON: Unknown neural network field: %s\n", name.c_str());
                    bValid                      = false;
                    goto exit;
                }
            }
        }

        // Calculate booleans
        if (nd._sparsenessPenalty_beta > (NNFloat)0.0)
            nd._bSparsenessPenalty                      = true;

        // Turn on denoising if active
        if (nd._denoising_p > (NNFloat)0.0)
        {
            nd._bDenoising                              = true;
            for (size_t i = 0; i <  nd._vLayerDescriptor.size(); i++)
            {
                if ((nd._vLayerDescriptor[i]._kind == NNLayer::Kind::Input) && ((nd._vLayerDescriptor[i]._attributes & NNLayer::Attributes::Sparse) != 0))
                {
                    nd._vLayerDescriptor[i]._attributes |= NNLayer::Attributes::Denoising;
                }
            }
        }
    }
    
    // Grab default network values for unspecified layer attributes
    for (size_t i = 0; i <  nd._vLayerDescriptor.size(); i++)
    {
        if (isnan(nd._vLayerDescriptor[i]._RELUSlope))
            nd._vLayerDescriptor[i]._RELUSlope          = nd._RELUSlope;
        if (isnan(nd._vLayerDescriptor[i]._ELUAlpha))
            nd._vLayerDescriptor[i]._ELUAlpha           = nd._ELUAlpha;
        if (isnan(nd._vLayerDescriptor[i]._SELULambda))
            nd._vLayerDescriptor[i]._SELULambda         = nd._SELULambda;
    }

    // Calculate dimensions for unspecified convolution and pooling layers
    CalculateDerivedLayerDimensions(nd);

    // Check for success, shut down upon failure
exit:
    MPI_Bcast(&bValid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bValid)
    {
        getGpu().Shutdown();
        exit(-1);
    }


    MPI_Bcast_NNNetworkDescriptor(nd);

    // Enumerate network
    if (getGpu()._id == 0)
    {
        cout << "LoadNeuralNetworkJSON: Enumerating network:" << endl;
        cout << nd << endl;
    }

    // Now create network;
    pNetwork                                    = new NNNetwork(nd, batch);
    pNetwork->RefreshState();
    return pNetwork;
}

NNNetwork* LoadNeuralNetworkNetCDF(const string& fname, const uint32_t batch)
{
    NNNetwork* pNetwork                         = NULL;
    NNNetworkDescriptor nd;

    // Load network data into GPU 0
    bool bResult                                = true;
    NNFloat version                             = (NNFloat)0.0;
    uint32_t layers                             = 0;
    uint32_t weights                            = 0;

    MPI_Bcast_string(nd._name);

    // Turn off calculation of convolution layer dimensions in NNNetwork constructor
    nd._bConvLayersCalculated                   = true;

    if (getGpu()._id == 0)
    {
        bool bOpened                            = false;
        try
        {
            // Work around stupid unformative exception throwing here with a bool
            NcFile nc(fname, NcFile::read);
            bOpened                             = true;

            // Read network attributes
            NcGroupAtt versionAtt               = nc.getAtt("version");
            if (versionAtt.isNull())
            {
                throw NcException("NcException", "NNNetwork::NNetwork: No version supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            versionAtt.getValues(&version);

            NcGroupAtt nameAtt                  = nc.getAtt("name");
            if (nameAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(nd._name);

            NcGroupAtt kindAtt                  = nc.getAtt("kind");
            if (nameAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No kind supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kindAtt.getValues(&(nd._kind));

            NcGroupAtt errorFunctionAtt         = nc.getAtt("errorFunction");
            if (errorFunctionAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No error function supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            errorFunctionAtt.getValues(&(nd._errorFunction));


            NcGroupAtt maxout_kAtt              = nc.getAtt("maxout_k");
            if (maxout_kAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No maxout_k supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            maxout_kAtt.getValues(&(nd._maxout_k));

            NcGroupAtt LRN_kAtt                 = nc.getAtt("LRN_k");
            if (LRN_kAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No LRN_k supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            LRN_kAtt.getValues(&(nd._LRN_k));

            NcGroupAtt LRN_nAtt                 = nc.getAtt("LRN_n");
            if (LRN_nAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No LRN_n supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            LRN_nAtt.getValues(&(nd._LRN_n));

            NcGroupAtt LRN_alphaAtt             = nc.getAtt("LRN_alpha");
            if (LRN_alphaAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No LRN_alpha supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            LRN_alphaAtt.getValues(&(nd._LRN_alpha));

            NcGroupAtt LRN_betaAtt              = nc.getAtt("LRN_beta");
            if (LRN_betaAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No LRN_beta supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            LRN_betaAtt.getValues(&(nd._LRN_beta));

            NcGroupAtt bSparsenessPenaltyAtt    = nc.getAtt("bSparsenessPenalty");
            if (bSparsenessPenaltyAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No bSparsenessPenalty supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bSparsenessPenalty;
            bSparsenessPenaltyAtt.getValues(&bSparsenessPenalty);
            nd._bSparsenessPenalty              = (bSparsenessPenalty != 0);

            NcGroupAtt sparsenessPenalty_pAtt   = nc.getAtt("sparsenessPenalty_p");
            if (sparsenessPenalty_pAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No sparsenessPenalty_p supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            sparsenessPenalty_pAtt.getValues(&(nd._sparsenessPenalty_p));

            NcGroupAtt sparsenessPenalty_betaAtt= nc.getAtt("sparsenessPenalty_beta");
            if (sparsenessPenalty_betaAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No sparsenessPenalty_beta supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            sparsenessPenalty_betaAtt.getValues(&(nd._sparsenessPenalty_beta));

            NcGroupAtt bDenoisingAtt            = nc.getAtt("bDenoising");
            if (bDenoisingAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No bDenoising supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bDenoising;
            bDenoisingAtt.getValues(&bDenoising);
            nd._bDenoising                      = (bDenoising != 0);

            NcGroupAtt denoising_pAtt           = nc.getAtt("denoising_p");
            if (denoising_pAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No denoising_p supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            denoising_pAtt.getValues(&(nd._denoising_p));


            // Read DeltaBoost parameters
            NcGroupAtt deltaBoost_oneAtt        = nc.getAtt("deltaBoost_one");
            if (deltaBoost_oneAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No deltaBoost_one supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            deltaBoost_oneAtt.getValues(&(nd._deltaBoost_one));

            NcGroupAtt deltaBoost_zeroAtt       = nc.getAtt("deltaBoost_zero");
            if (deltaBoost_zeroAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No deltaBoost_zero supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            deltaBoost_zeroAtt.getValues(&(nd._deltaBoost_zero));

            // Read Scaled Marginal CrossEntropy parameters
            NcGroupAtt SMCE_oneScaleAtt         = nc.getAtt("SMCE_oneScale");
            if (SMCE_oneScaleAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No SMCE_oneScale supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SMCE_oneScaleAtt.getValues(&(nd._SMCE_oneScale));

            NcGroupAtt SMCE_zeroScaleAtt        = nc.getAtt("SMCE_zeroScale");
            if (SMCE_zeroScaleAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No SMCE_zeroScale supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SMCE_zeroScaleAtt.getValues(&(nd._SMCE_zeroScale));

            NcGroupAtt SMCE_oneTargetAtt        = nc.getAtt("SMCE_oneTarget");
            if (SMCE_oneTargetAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No SMCE_oneTarget supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SMCE_oneTargetAtt.getValues(&(nd._SMCE_oneTarget));

            NcGroupAtt SMCE_zeroTargetAtt       = nc.getAtt("SMCE_zeroTarget");
            if (SMCE_zeroTargetAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No SMCE_zeroTarget supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SMCE_zeroTargetAtt.getValues(&(nd._SMCE_zeroTarget));

            NcGroupAtt checkpoint_nameAtt       = nc.getAtt("checkpoint_name");
            if (checkpoint_nameAtt.isNull())
            {
                //throw NcException("NcException", "NNetwork::NNetwork: No checkpoint_name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                // Use default value from constructor
            }
            else
                checkpoint_nameAtt.getValues(nd._checkpoint_name);

            NcGroupAtt checkpoint_intervalAtt   = nc.getAtt("checkpoint_interval");
            if (checkpoint_intervalAtt.isNull())
            {
                //throw NcException("NcException", "NNetwork::NNetwork: No checkpoint_interval supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                // Use default value from constructor
            }
            else
                checkpoint_intervalAtt.getValues(&(nd._checkpoint_interval));

            NcGroupAtt checkpoint_epochsAtt     = nc.getAtt("checkpoint_epochs");
            if (checkpoint_epochsAtt.isNull())
            {
                //throw NcException("NcException", "NNetwork::NNetwork: No checkpoint_epochs supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                // Use default value from constructor
            }
            else
                checkpoint_epochsAtt.getValues(&(nd._checkpoint_epochs));


            NcGroupAtt shuffleIndicesAtt        = nc.getAtt("ShuffleIndices");
            if (shuffleIndicesAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No shuffleIndices supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bShuffleIndices;
            shuffleIndicesAtt.getValues(&bShuffleIndices);
            nd._bShuffleIndices                 = (bShuffleIndices != 0);

            // Read network layer count
            NcGroupAtt layersAtt                = nc.getAtt("layers");
            if (layersAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No layers supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            layersAtt.getValues(&layers);

            // Get layer descriptors
            for (uint32_t i = 0; i < layers; i++)
            {
                NNLayerDescriptor ld;
                if (!LoadNNLayerDescriptorNetCDF(fname, nc, i, ld))
                {
                    throw NcException("NcException", "NNetwork::NNetwork: Error reading layer data in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                nd._vLayerDescriptor.push_back(ld);
            }

            // Read network weight count
            NcGroupAtt weightsAtt               = nc.getAtt("weights");
            if (weightsAtt.isNull())
            {
                throw NcException("NcException", "NNetwork::NNetwork: No weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            weightsAtt.getValues(&weights);

            // Get weight descriptors and data
            for (uint32_t i = 0; i < weights; i++)
            {
                NNWeightDescriptor wd;
                if (!LoadNNWeightDescriptorNetCDF(fname, nc, i, wd))
                {
                    throw NcException("NcException", "NNetwork::NNetwork: Error reading weight data in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                nd._vWeightDescriptor.push_back(wd);
            }


            //cout << nd << endl;

        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "Exception: NNetWork::NNetwork: Error opening NetCDF input file " << fname << endl;
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

    // Finally build network from descriptor and return
    MPI_Bcast_NNNetworkDescriptor(nd);

    // Enumerate network
    if (getGpu()._id == 0)
    {
        cout << "LoadNeuralNetworkJSON: Enumerating network:" << endl;
        cout << nd << endl;
    }

    // Create network
    pNetwork                                    = new NNNetwork(nd, batch);
    pNetwork->RefreshState();
    return pNetwork;
}

bool NNNetwork::P2P_Bcast(void* pBuffer, size_t size)
{
    cudaError_t status;

    // Skip if in single GPU mode
    if (getGpu()._numprocs > 1)
    {
        if (getGpu()._bP2P)
        {
            // Special case 2 GPUs as a single copy
            if (getGpu()._numprocs == 2)
            {
                if (getGpu()._id == 0)
                {
                    status                                  = cudaMemcpy(GetPeerBackBuffer(), pBuffer, size, cudaMemcpyDefault);
                    RTERROR(status, "NNNetwork::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                }
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else // Scatter data to all other CPUs in numprocs chunks for numprocs * 2 - 2 stages
            {
                // Copy source data to P2P send buffer if on process 0
                if (getGpu()._id == 0)
                {
                    status                                  = cudaMemcpy(GetP2PSendBuffer(), pBuffer, size, cudaMemcpyDefault);
                    RTERROR(status, "NNNetwork::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                }

                uint32_t stages                             = 2 * getGpu()._numprocs - 2;
                uint32_t distance                           = (getGpu()._numprocs - getGpu()._id) % getGpu()._numprocs;
                uint64_t segment                            = 0;
                for (uint32_t i = 0; i < stages; i++)
                {
                    // Send chunk if active
                    if ((getGpu()._id != 1) &&  (i >= distance) && (segment < getGpu()._numprocs))
                    {
                        size_t start                        = (size * segment) / getGpu()._numprocs;
                        size_t end                          = (size * (segment + 1)) / getGpu()._numprocs;
                        status                              = cudaMemcpy((char*)GetPeerBackBuffer() + start, (char*)GetP2PSendBuffer() + start, end - start, cudaMemcpyDefault);
                        RTERROR(status, "NNNetwork::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                        segment++;
                    }

                    // Wait for all copies to complete
                    cudaDeviceSynchronize();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }

            // Grab Data from P2P receive buffer if on process 1 or higher
            if (getGpu()._id > 0)
            {
                status                                      = cudaMemcpy(pBuffer, GetP2PSendBuffer(), size, cudaMemcpyDefault);
                RTERROR(status, "NNNetwork::P2P_Bcast: Failure to copy source data from P2P sendbuffer");
            }
        }
        else
        {
            cudaMemcpy(_pCPUBuffer.get(), pBuffer, size, cudaMemcpyDefault);
            MPI_Bcast(_pCPUBuffer.get(), size, MPI_BYTE, 0, MPI_COMM_WORLD);
            cudaMemcpy(pBuffer, _pCPUBuffer.get(), size, cudaMemcpyDefault);
        }
    }

    return true;
}

bool NNNetwork::P2P_Allreduce(NNFloat* pBuffer, size_t size)
{
    // Skip if in single GPU mode
    if (getGpu()._numprocs > 1)
    {
        if (getGpu()._bP2P)
        {
            // Special case 2 GPUs
            if (getGpu()._numprocs == 2)
            {
                cudaMemcpy(GetPeerBuffer(), pBuffer, size * sizeof(NNFloat), cudaMemcpyDefault);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                kAddBuffers(pBuffer, GetP2PReceiveBuffer(), size);
            }
            else
            {
                uint32_t stages                         = getGpu()._numprocs - 1;
                uint64_t segment                        = getGpu()._id;
                uint64_t start                          = (size * segment) / getGpu()._numprocs;
                uint64_t end                            = (size * (segment + 1)) / getGpu()._numprocs;

                // Send segments around the adding local contributions from each process
                for (uint32_t i = 0; i < stages; i++)
                {

                    if (i == 0)
                        cudaMemcpy(GetPeerBuffer(), pBuffer + start, (end - start) * sizeof(NNFloat), cudaMemcpyDefault);
                    else
                        cudaMemcpy(GetPeerBuffer(), GetP2PSendBuffer(), (end - start) * sizeof(NNFloat), cudaMemcpyDefault);

                    // Wait for completion
                    cudaDeviceSynchronize();
                    MPI_Barrier(MPI_COMM_WORLD);
                    SwapPeerBuffers();
                    segment                             = (segment + 1) % getGpu()._numprocs;
                    start                               = (size * segment) / getGpu()._numprocs;
                    end                                 = (size * (segment + 1)) / getGpu()._numprocs;
                    kAddBuffers(GetP2PSendBuffer(), pBuffer + start, end - start);
                }

                // Circulate segments a second time and copy out results
                cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (end - start) * sizeof(NNFloat), cudaMemcpyDefault);
                for (uint32_t i = 0; i < stages; i++)
                {
                    cudaMemcpy(GetPeerBuffer(), GetP2PSendBuffer(), (end - start) * sizeof(NNFloat), cudaMemcpyDefault);

                    // Wait for completion
                    cudaDeviceSynchronize();
                    MPI_Barrier(MPI_COMM_WORLD);
                    SwapPeerBuffers();
                    segment                             = (segment + 1) % getGpu()._numprocs;
                    start                               = (size * segment) / getGpu()._numprocs;
                    end                                 = (size * (segment + 1)) / getGpu()._numprocs;
                    cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (end - start) * sizeof(NNFloat), cudaMemcpyDefault);
                }
            }
        }
        else
        {
            // Not meant to be efficient, you should invest in P2P GPU hardware and servers,
            // but instead present for those who enjoy dancing bears.  This code will let you
            // run a bajillion GPUs over MPI, not very efficiently, but you could have
            // thousands of GPUs if you so desired.
            cudaMemcpy(_pCPUBuffer.get(), pBuffer, size * sizeof(NNFloat), cudaMemcpyDefault);
            MPI_Allreduce(MPI_IN_PLACE, _pCPUBuffer.get(), size, MPI_NNFLOAT, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpy(pBuffer, _pCPUBuffer.get(), size * sizeof(NNFloat), cudaMemcpyDefault);
        }
    }
    return true;
}
