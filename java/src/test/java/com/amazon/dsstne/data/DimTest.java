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

import com.amazon.dsstne.Dim;

/**
 * @author kiuk
 */
public class DimTest {

    @Test
    public void testDim() {
        Dim oneD = Dim._1d(128, 32);
        Dim twoD = Dim._2d(128, 64, 32);
        Dim threeD = Dim._3d(128, 64, 32, 16);

        assertEquals(1, oneD.dimensions);
        assertEquals(2, twoD.dimensions);
        assertEquals(3, threeD.dimensions);

        assertEquals(128, oneD.stride);
        assertEquals(128 * 64, twoD.stride);
        assertEquals(128 * 64 * 32, threeD.stride);
    }
}

