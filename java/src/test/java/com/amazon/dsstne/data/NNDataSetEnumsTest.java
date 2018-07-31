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

package com.amazon.dsstne.data;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.amazon.dsstne.NNDataSetEnums.DataType;
import com.amazon.dsstne.NNDataSetEnums.Kind;
import com.amazon.dsstne.NNDataSetEnums.Sharding;

/**
 * Test that the ordinals of the enums match
 * the ones in C++ NNDataSetEnums. If the order
 * of the enums are changed then these tests should fail.
 */
public class NNDataSetEnumsTest {

    @Test
    public void testKind() {
        assertEquals(0, Kind.Numeric.ordinal());
        assertEquals(1, Kind.Image.ordinal());
        assertEquals(2, Kind.Audio.ordinal());
    }

    @Test
    public void testSharding() {
        assertEquals(0, Sharding.None.ordinal());
        assertEquals(1, Sharding.Model.ordinal());
        assertEquals(2, Sharding.Data.ordinal());
    }

    @Test
    public void testDataType() {
        assertEquals(0, DataType.UInt.ordinal());
        assertEquals(1, DataType.Int.ordinal());
        assertEquals(2, DataType.LLInt.ordinal());
        assertEquals(3, DataType.ULLInt.ordinal());
        assertEquals(4, DataType.Float.ordinal());
        assertEquals(5, DataType.Double.ordinal());
        assertEquals(6, DataType.RGB8.ordinal());
        assertEquals(7, DataType.RGB16.ordinal());
        assertEquals(8, DataType.UChar.ordinal());
        assertEquals(9, DataType.Char.ordinal());

    }
}