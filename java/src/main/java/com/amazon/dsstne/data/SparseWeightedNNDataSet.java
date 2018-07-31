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
 * Sparse dataset that additionally has weights that are associated with
 * each element of the data.
 */
public class SparseWeightedNNDataSet extends SparseNNDataSet {

    private final float[] weights;

    public SparseWeightedNNDataSet(final Dim dim, final DataType dataType, final double sparseDensity) {
        super(dim, dataType, sparseDensity);
        this.weights = new float[getStride() * dim.examples];
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
    public void addSparse(final int index, final long[] sparseIndex, final char[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeightOne(index);
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final int[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeightOne(index);
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final float[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeightOne(index);
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final double[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeightOne(index);
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final char[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeight(index, weights);
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final int[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeight(index, weights);
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final float[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeight(index, weights);
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final double[] data) {
        super.addSparse(index, sparseIndex, data);
        putWeight(index, weights);
    }
}
