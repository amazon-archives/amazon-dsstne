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

import lombok.Getter;
import lombok.Setter;

/**
 * Data set to hold outputs of the predictions from a network.
 * Output data is comprised of two datasets: indexes and scores.
 * The indexes hold the indexes of the top-k results.
 * The scores hold the output values of the top-k results.
 */
@Getter
public class TopKOutput {

    private final Dim dim;

    @Setter
    private String name = "";

    /**
     * Name of the output layer this dataset is for.
     */
    @Setter
    private String layerName = "";

    private final float[] scores;
    private final long[] indexes;

    /* package private */ TopKOutput(final Dim dim) {
        this.dim = dim;
        this.scores = new float[dim.x * dim.y * dim.z * dim.examples];
        this.indexes = new long[dim.x * dim.y * dim.z * dim.examples];
    }

    public static TopKOutput create(final NetworkConfig config, final NNLayer outputLayer) {
        int k = config.getK();
        int batchSize = config.getBatchSize();
        Dim outputLayerDim = outputLayer.getDim();

        TopKOutput outputDataset;
        if (config.getK() == NetworkConfig.ALL) {
            outputDataset = new TopKOutput(new Dim(outputLayerDim, batchSize));
        } else {
            if(outputLayerDim.dimensions > 1) {
                throw new IllegalArgumentException("Top k outputs only supported on 1-D outputs");
            }
            outputDataset = new TopKOutput(Dim._1d(k, batchSize));
        }
        outputDataset.setName(outputLayer.getDatasetName());
        outputDataset.setLayerName(outputLayer.getName());
        return outputDataset;
    }
}

