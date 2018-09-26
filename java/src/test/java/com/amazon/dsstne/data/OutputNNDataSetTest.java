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

import org.junit.Assert;
import org.junit.Test;

import com.amazon.dsstne.NNLayer;
import com.amazon.dsstne.NNLayer.Attribute;
import com.amazon.dsstne.NNLayer.Kind;
import com.amazon.dsstne.NetworkConfig;

public class OutputNNDataSetTest {

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
        OutputNNDataSet outputDataset = OutputNNDataSet.create(config, layer);
        Assert.assertEquals(datasetName, outputDataset.getName());
        Assert.assertEquals(layerName, outputDataset.getLayerName());
    }

    @Test
    public void testOutputAllUnitBuffer() {
        NetworkConfig config = NetworkConfig.with().batchSize(batchSize).build();
        OutputNNDataSet outputDataset = OutputNNDataSet.create(config, layer);
        Assert.assertEquals(x, outputDataset.getDim().x);
        Assert.assertEquals(y, outputDataset.getDim().y);
        Assert.assertEquals(z, outputDataset.getDim().z);
        Assert.assertEquals(1, outputDataset.getDim().dimensions);
        Assert.assertEquals(batchSize, outputDataset.getDim().examples);
    }

    @Test
    public void testOutputTopK() {
        int k = 100;
        NetworkConfig config = NetworkConfig.with().batchSize(batchSize).k(100).build();
        OutputNNDataSet outputDataset = OutputNNDataSet.create(config, layer);
        Assert.assertEquals(k, outputDataset.getDim().x);
        Assert.assertEquals(y, outputDataset.getDim().y);
        Assert.assertEquals(z, outputDataset.getDim().z);
        Assert.assertEquals(1, outputDataset.getDim().dimensions);
        Assert.assertEquals(batchSize, outputDataset.getDim().examples);
    }

}