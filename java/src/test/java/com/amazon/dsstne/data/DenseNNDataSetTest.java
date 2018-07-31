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

import java.nio.IntBuffer;

import org.junit.Test;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSetEnums.DataType;

public class DenseNNDataSetTest {

    @Test
    public void testAddInt() {
        DenseNNDataSet ds = new DenseNNDataSet(Dim._1d(4, 3), DataType.Int);
        ds.add(2, new int[] {2, 2, 2, 2});
        ds.add(1, new int[] {1, 1, 1, 1});

        IntBuffer buf = ds.getData().asIntBuffer();
        for (int i = 0; i < 4; i++) {
            assertEquals(0, buf.get(i));
        }

        for (int i = 0; i < 4; i++) {
            assertEquals(0, buf.get(1));
        }

        for (int i = 0; i < 4; i++) {
            assertEquals(0, buf.get(2));
        }
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testAddWeighted() {
        DenseNNDataSet ds = new DenseNNDataSet(Dim._1d(4, 3), DataType.Int);
        ds.addWeighted(0, new int[0], new float[0]);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testAddSparse() {
        DenseNNDataSet ds = new DenseNNDataSet(Dim._1d(4, 3), DataType.Int);
        ds.addSparse(0, new long[0], new int[0]);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testAddSparseWeighted() {
        DenseNNDataSet ds = new DenseNNDataSet(Dim._1d(4, 3), DataType.Int);
        ds.addSparseWeighted(0, new long[0], new float[0], new int[0]);
    }
}