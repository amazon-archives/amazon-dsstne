/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __GPUSORT_H__
#define __GPUSORT_H__

#include <memory>

template<typename KeyType, typename ValueType> class GpuSort
{
private:
    unsigned int                    _items;
    unsigned int                    _itemStride;
    unique_ptr<GpuBuffer<KeyType>>  _pbKey;
    KeyType*                        _pKey0;
    KeyType*                        _pKey1;
    unique_ptr<GpuBuffer<ValueType>> _pbValue;
    ValueType*                      _pValue0;
    ValueType*                      _pValue1;
    size_t                          _tempBytes;
    unique_ptr<GpuBuffer<char>>     _pbTemp;
    KeyType*                        _pKey;
    ValueType*                      _pValue;


    
public:
    GpuSort(unsigned int items) :
    _items(items),
    _itemStride(((items + 511) >> 9) << 9),
    _pbKey(new GpuBuffer<KeyType>(_itemStride * 2)),
    _pKey0(_pbKey->_pDevData),
    _pKey1(_pbKey->_pDevData + _itemStride),
    _pbValue(new GpuBuffer<ValueType>(_itemStride * 2)),
    _pValue0(_pbValue->_pDevData),
    _pValue1(_pbValue->_pDevData + _itemStride),
    _tempBytes(kInitSort(_items, _pbValue.get(), _pbKey.get())),
    _pbTemp(new GpuBuffer<char>(_tempBytes))
    {
        _pKey                       = _pKey0;
        _pValue                     = _pValue0;      
    }

    ~GpuSort()
    {
    }
    bool Sort() { return kSort(_items, _pKey0, _pKey1, _pValue0, _pValue1, _pbTemp->_pDevData, _tempBytes); }
    GpuBuffer<KeyType>* GetKeyBuffer() { return _pbKey.get(); }
    GpuBuffer<ValueType>* GetValueBuffer() { return _pbValue.get(); }
    KeyType* GetKeyPointer() { return _pKey;}
    ValueType* GetValuePointer() { return _pValue; }
};
#endif
