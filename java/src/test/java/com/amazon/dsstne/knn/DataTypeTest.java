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

package com.amazon.dsstne.knn;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class DataTypeTest {

    @Test
    public void testDataTypeFromString() {
        assertEquals(DataType.FP16, DataType.fromString("fp16"));
        assertEquals(DataType.FP32, DataType.fromString("fp32"));
    }

    /**
     * Ordinal of {@link DataType} should not change since we pass it to C++ via JNI.
     * If it needs to change then also need to change in the brazil package AstdlKnnCoreJNI.
     */
    @Test
    public void testDataTypeFromString_ordinal() {
        assertEquals(0, DataType.FP32.ordinal());
        assertEquals(1, DataType.FP16.ordinal());
    }
}