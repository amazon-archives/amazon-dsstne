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

import java.util.Arrays;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSetEnums.DataType;

/**
 * Dense dataset with additional weight information
 * associated with each element of the data.
 */
public class DenseWeightedNNDataSet extends DenseNNDataSet {

    private final float[] weights;

    public DenseWeightedNNDataSet(final Dim dim, final DataType dataType) {
        super(dim, dataType);
        this.weights = new float[dim.stride * dim.examples];
    }

    private void putWeightOne(final int index) {
        Arrays.fill(this.weights, index * getStride(), index + getStride(), 1f);
    }

    private void putWeight(final int index, final float[] weights) {
        System.arraycopy(weights, 0, this.weights, index * getStride(), getStride());
    }

    @Override
    public float[] getWeights() {
        return weights;
    }

    @Override
    public void add(final int index, final char[] data) {
        super.add(index, data);
        putWeightOne(index);
    }

    @Override
    public void add(final int index, final int[] data) {
        super.add(index, data);
        putWeightOne(index);
    }

    @Override
    public void add(final int index, final float[] data) {
        super.add(index, data);
        putWeightOne(index);
    }

    @Override
    public void add(final int index, final double[] data) {
        super.add(index, data);
        putWeightOne(index);
    }

    @Override
    public void addWeighted(final int index, final char[] data, final float[] weights) {
        super.add(index, data);
        putWeight(index, weights);
    }

    @Override
    public void addWeighted(final int index, final int[] data, final float[] weights) {
        super.add(index, data);
        putWeight(index, weights);
    }

    @Override
    public void addWeighted(final int index, final float[] data, final float[] weights) {
        super.add(index, data);
        putWeight(index, weights);
    }

    @Override
    public void addWeighted(final int index, final double[] data, final float[] weights) {
        super.add(index, data);
        putWeight(index, weights);
    }
}
