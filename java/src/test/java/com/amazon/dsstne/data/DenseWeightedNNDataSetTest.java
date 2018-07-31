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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.FloatBuffer;

import org.junit.Test;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSetEnums.DataType;

/**
 * @author kiuk
 */
public class DenseWeightedNNDataSetTest {

    @Test
    public void testAddFloat() {
        DenseWeightedNNDataSet ds = new DenseWeightedNNDataSet(Dim._1d(4, 2), DataType.Float);
        ds.add(0, new float[] {1.2f, 2.3f, 3.4f, 4.5f});
        ds.addWeighted(1, new float[] {1.1f, 2.2f, 3.3f, 4.4f}, new float[] {4.4f, 3.3f, 2.2f, 1.1f});

        FloatBuffer buf = ds.getData().asFloatBuffer();
        float[] temp = new float[ds.getStride()];
        buf.get(temp);
        assertArrayEquals(new float[] {1.2f, 2.3f, 3.4f, 4.5f}, temp, 0f);

        buf.get(temp);
        assertArrayEquals(new float[] {1.1f, 2.2f, 3.3f, 4.4f}, temp, 0f);

        for (int i = 0; i < ds.getStride(); i++) {
            assertEquals(1f, ds.getWeights()[i], 0f);
        }

        assertEquals(4.4f, ds.getWeights()[ds.getStride() + 0], 0f);
        assertEquals(3.3f, ds.getWeights()[ds.getStride() + 1], 0f);
        assertEquals(2.2f, ds.getWeights()[ds.getStride() + 2], 0f);
        assertEquals(1.1f, ds.getWeights()[ds.getStride() + 3], 0f);
    }

}