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
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSet;
import com.amazon.dsstne.NNDataSetEnums.DataType;

/**
 * Dense numeric dataset.
 */
public class DenseNNDataSet extends NNDataSet {

    public DenseNNDataSet(final Dim dim, final DataType dataType) {
        super(dim, dataType);
        this.data = ByteBuffer.allocateDirect(getStride() * dim.examples * DataType.sizeof(dataType));
        this.data.order(ByteOrder.nativeOrder());
    }

    @Override
    public int getStride() {
        return getDim().stride;
    }

    @Override
    public long[] getSparseStart() {
        return EMPTY_LONG_ARRAY;
    }

    @Override
    public long[] getSparseEnd() {
        return EMPTY_LONG_ARRAY;
    }

    @Override
    public long[] getSparseIndex() {
        return EMPTY_LONG_ARRAY;
    }

    @Override
    public float[] getWeights() {
        return EMPTY_FLOAT_ARRAY;
    }

    @Override
    public void add(final int index, final char[] data) {
        this.data.position(index * getStride());
        this.data.asCharBuffer().put(data, 0, getStride());
    }

    @Override
    public void add(final int index, final int[] data) {
        IntBuffer buffView = this.data.asIntBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    @Override
    public void add(final int index, final float[] data) {
        FloatBuffer buffView = this.data.asFloatBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    @Override
    public void add(final int index, final double[] data) {
        DoubleBuffer buffView = this.data.asDoubleBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    @Override
    public void addWeighted(final int index, final char[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addWeighted(final int index, final int[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addWeighted(final int index, final float[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addWeighted(final int index, final double[] data, final float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final char[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final int[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final float[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    @Override
    public void addSparse(final int index, final long[] sparseIndex, final double[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final char[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights, final int[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final float[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    @Override
    public void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final double[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }
}
