/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */
#include <cppunit/extensions/HelperMacros.h>

#include "amazon/dsstne/engine/GpuTypes.h"

class TestGpuBuffer: public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestGpuBuffer);

    CPPUNIT_TEST(testResize);

    CPPUNIT_TEST_SUITE_END();

 public:

    void testResize()
    {
        size_t length = 1024;
        // create managed memory to make testing easier
        GpuBuffer<uint32_t> buff(length, false, true);

        for(size_t i = 0; i < length ; ++i)
        {
            buff._pDevData[i] = i;
        }

        // check that the buffer contains what we put
        for(uint32_t i =0; i < length; ++i){
            CPPUNIT_ASSERT_EQUAL(i, buff._pDevData[i]);
        }

        // shouldn't have resized (same length) check that buffer still contains the same data
        buff.Resize(length);
        for (uint32_t i = 0; i < length; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(i, buff._pDevData[i]);
        }

        // shouldn't have resized (smaller length) check that buffer still contains the same data
        buff.Resize(length - 1);
        for (uint32_t i = 0; i < length; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(i, buff._pDevData[i]);
        }

        // should resize (length larger) check that not all the data is the same
        buff.Resize(length + 1);
        bool isSame = true;
        for(uint32_t i=0; i< length; ++i){
            isSame &= (buff._pDevData[i] == i);
        }
        CPPUNIT_ASSERT(!isSame);
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGpuBuffer);
