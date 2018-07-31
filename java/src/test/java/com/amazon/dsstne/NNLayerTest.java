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

package com.amazon.dsstne;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.amazon.dsstne.NNLayer.Kind;

/**
 * @author kiuk
 */
public class NNLayerTest {

    @Test
    public void testKindOrdinal() {
        assertEquals(0, Kind.Input.ordinal());
        assertEquals(1, Kind.Hidden.ordinal());
        assertEquals(2, Kind.Output.ordinal());
        assertEquals(3, Kind.Target.ordinal());
    }

    @Test
    public void testAttributeMaskValues() {
        assertEquals(0x0, NNLayer.Attribute.None);
        assertEquals(0x1, NNLayer.Attribute.Sparse);
        assertEquals(0x2, NNLayer.Attribute.Denoising);
        assertEquals(0x4, NNLayer.Attribute.BatchNormalization);
    }

    @Test
    public void testCreateNNLayer() {
        NNLayer inputLayer = new NNLayer("input-layer", "input-layer-data", 0, 1, 1, 128, 1, 1);
        NNLayer hiddenLayer = new NNLayer("hidden-layer", "", 1, 1, 1, 128, 1, 1);
        NNLayer outputLayer = new NNLayer("output-layer", "output-layer-data", 2, 1, 1, 128, 1, 1);
        NNLayer targetLayer = new NNLayer("target-layer", "", 3, 1, 1, 128, 1, 1);

        assertEquals(Kind.Input, inputLayer.getKind());
        assertEquals(Kind.Hidden, hiddenLayer.getKind());
        assertEquals(Kind.Output, outputLayer.getKind());
        assertEquals(Kind.Target, targetLayer.getKind());
    }
}
