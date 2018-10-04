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

import org.junit.Assert;
import org.junit.Test;

import com.amazon.dsstne.NNLayer.Attribute;
import com.amazon.dsstne.NNLayer.Kind;

public class TopKOutputTest {

    private String layerName = "test-layer";
    private String datasetName = "test-dataset";
    private int x = 128;
    private int y = 1;
    private int z = 1;
    private int batchSize = 128;
    private NNLayer layer = new NNLayer(layerName, datasetName, Kind.Input.ordinal(), Attribute.None, 1, x, y, z);

    @Test
    public void testNamesAreSet() {
        NetworkConfig config = NetworkConfig.with().batchSize(batchSize).build();
        TopKOutput output = TopKOutput.create(config, layer);
        Assert.assertEquals(datasetName, output.getName());
        Assert.assertEquals(layerName, output.getLayerName());
    }

    @Test
    public void testOutputAllUnitBuffer() {
        NetworkConfig config = NetworkConfig.with().batchSize(batchSize).build();
        TopKOutput output = TopKOutput.create(config, layer);
        Assert.assertEquals(x, output.getDim().x);
        Assert.assertEquals(y, output.getDim().y);
        Assert.assertEquals(z, output.getDim().z);
        Assert.assertEquals(1, output.getDim().dimensions);
        Assert.assertEquals(batchSize, output.getDim().examples);
    }

    @Test
    public void testOutputTopK() {
        int k = 100;
        NetworkConfig config = NetworkConfig.with().batchSize(batchSize).k(100).build();
        TopKOutput output = TopKOutput.create(config, layer);
        Assert.assertEquals(k, output.getDim().x);
        Assert.assertEquals(y, output.getDim().y);
        Assert.assertEquals(z, output.getDim().z);
        Assert.assertEquals(1, output.getDim().dimensions);
        Assert.assertEquals(batchSize, output.getDim().examples);
    }

}