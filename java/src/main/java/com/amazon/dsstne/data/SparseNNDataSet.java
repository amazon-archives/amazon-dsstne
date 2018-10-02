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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSet;
import com.amazon.dsstne.NNDataSetEnums.Attribute;
import com.amazon.dsstne.NNDataSetEnums.DataType;

import lombok.Getter;

/**
 * Sparse numeric datasets.
 */
public class SparseNNDataSet extends NNDataSet {

    @Getter
    private final double sparseDensity;

    private final int stride;

    private final long[] sparseStart;
    private final long[] sparseEnd;
    private final long[] sparseIndex;

    public SparseNNDataSet(final Dim dim, final DataType dataType, final double sparseDensity) {
        super(dim, dataType);

        if (sparseDensity < 0.0 || sparseDensity > 1.0) {
            throw new IllegalArgumentException("sparseDensity should be between 0.0 and 1.0");
        }
        this.sparseDensity = sparseDensity;
        this.stride = (int) (dim.stride * sparseDensity);

        this.sparseStart = new long[dim.examples];
        this.sparseEnd = new long[dim.examples];

        this.sparseIndex = new long[stride * dim.examples];
        this.data = ByteBuffer.allocateDirect(stride * dim.examples * DataType.sizeof(dataType));
        this.data.order(ByteOrder.nativeOrder());
    }

    @Override
    public int getStride() {
        return stride;
    }

    @Override
    public int getAttribute() {
        return Attribute.Sparse;
    }

    @Override
    public float[] getWeights() {
        return EMPTY_FLOAT_ARRAY;
    }

    @Override
    public long[] getSparseStart() {
        return sparseStart;
    }

    @Override
    public long[] getSparseEnd() {
        return sparseEnd;
    }

    @Override
    public long[] getSparseIndex() {
        return sparseIndex;
    }

    @Override
    public void add(final int index, final char[] data) {
        throw new UnsupportedOperationException("add not supported for sparse dataset, use addSparse");
    }

    @Override
    public void add(final int index, final int[] data) {
        throw new UnsupportedOperationException("add not supported for sparse dataset, use addSparse");
    }

    @Override
    public void add(final int index, final float[] data) {
        throw new UnsupportedOperationException("add not supported for sparse dataset, use addSparse");
    }

    @Override
    public void add(final int index, final double[] data) {
        throw new UnsupportedOperationException("add not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addWeighted(final int index, final char[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addWeighted(final int index, final int[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addWeighted(final int index, final float[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addWeighted(final int index, final double[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final char[] data) {
        checkLength(sparseIndex.length, data.length);
        setPosition(index);
        this.data.asCharBuffer().put(data);
        putSparseInfo(index, sparseIndex);
    }

    private void checkLength(final int sparseIndexLength, final int sparseDataLength) {
        if (sparseDataLength != sparseDataLength) {
            throw new IllegalArgumentException(
                "sparseIndex length (" + sparseIndexLength + ") != sparseDataLength (" + sparseDataLength + ")");
        }
    }

    protected void putSparseInfo(final int index, final long[] sparseIndex) {
        int offset;
        if(index == 0) {
            offset = 0;
        } else {
            //FIXME change sparseStart[] and sparseEnd[] to int[] since java only allows int addressing
            offset = (int) this.sparseEnd[index - 1];
        }

        System.arraycopy(sparseIndex, 0, this.sparseIndex, offset, sparseIndex.length);
        this.sparseStart[index] = offset;
        this.sparseEnd[index] = offset + sparseIndex.length;
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final int[] data) {
        checkLength(sparseIndex.length, data.length);
        setPosition(index);
        this.data.asIntBuffer().put(data);
        putSparseInfo(index, sparseIndex);
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final float[] data) {
        checkLength(sparseIndex.length, data.length);
        setPosition(index);
        this.data.asFloatBuffer().put(data);
        putSparseInfo(index, sparseIndex);
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final double[] data) {
        checkLength(sparseIndex.length, data.length);
        setPosition(index);
        this.data.asDoubleBuffer().put(data);
        putSparseInfo(index, sparseIndex);
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final char[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final int[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final float[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final double[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse");
    }
}
