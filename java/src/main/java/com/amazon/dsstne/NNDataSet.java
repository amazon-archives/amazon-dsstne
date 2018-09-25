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

import java.nio.ByteBuffer;

import com.amazon.dsstne.NNDataSetEnums.Attribute;
import com.amazon.dsstne.NNDataSetEnums.DataType;
import com.amazon.dsstne.NNDataSetEnums.Sharding;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

/**
 * Java counter part to the original (C++) version of NNDataSet.
 * Represents the data in the layer of the network. The pointers
 * to the data buffers reside completely on the C++ side, we only pass
 * the attributes, in this sense this class more closely resembles the
 * descriptor classes in dsstne (NNLayerDescriptor, NNNetworkDescriptor).
 */
@Getter
@RequiredArgsConstructor
public abstract class NNDataSet {

    protected static final long[] EMPTY_LONG_ARRAY = new long[0];
    protected static final float[] EMPTY_FLOAT_ARRAY = new float[0];

    /**
     * Name of this dataset.
     */
    @Setter
    private String name = "";

    /**
     * Name of the layer for which this dataset feeds data into.
     * This field is optional if the dataset object is used outside
     * the context of a {@link NNNetwork}.
     */
    @Setter
    private String layerName = "";

    private final Dim dim;

    private final DataType dataType;

    protected ByteBuffer data;

    @Setter
    private Sharding sharding = Sharding.None;

    /**
     * Stride between examples. For dense, this is equal to dim_x * dim_y * dim_z,
     * for sparse it is dim_x * dim_y * dim_z * sparseDensity.
     */
    public abstract int getStride();

    public int getDimensions() {
        return dim.dimensions;
    }

    public int getDimX() {
        return dim.x;
    }

    public int getDimY() {
        return dim.y;
    }

    public int getDimZ() {
        return dim.z;
    }

    public int getExamples() {
        return dim.examples;
    }

    public int getAttribute() {
        return Attribute.None;
    }

    public ByteBuffer getData() {
        setPosition(0);
        return data;
    }

    public abstract long[] getSparseStart();

    public abstract long[] getSparseEnd();

    public abstract long[] getSparseIndex();

    public abstract float[] getWeights();

    public int getDataTypeOrdinal() {
        return getDataType().ordinal();
    }

    protected void setPosition(int index) {
        this.data.position(index * getStride() * DataType.sizeof(getDataType()));
    }

    public boolean isSparse() {
        return (getAttribute() & Attribute.Sparse) != 0;
    }

    public boolean isBoolean() {
        return (getAttribute() & Attribute.Boolean) != 0;
    }

    public boolean isWeighted() {
        return (getAttribute() & Attribute.Weighted) != 0;
    }

    public boolean isIndexed() {
        return (getAttribute() & Attribute.Indexed) != 0;
    }

    public abstract void add(final int index, final char[] data);

    public abstract void add(final int index, final int[] data);

    public abstract void add(final int index, final float[] data);

    public abstract void add(final int index, final double[] data);

    public abstract void addWeighted(final int index, final char[] data, final float[] weights);

    public abstract void addWeighted(final int index, final int[] data, final float[] weights);

    public abstract void addWeighted(final int index, final float[] data, final float[] weights);

    public abstract void addWeighted(final int index, final double[] data, final float[] weights);

    public abstract void addSparse(final int index, final long[] sparseIndex, final char[] data);

    public abstract void addSparse(final int index, final long[] sparseIndex, final int[] data);

    public abstract void addSparse(final int index, final long[] sparseIndex, final float[] data);

    public abstract void addSparse(final int index, final long[] sparseIndex, final double[] data);

    public abstract void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final char[] data);

    public abstract void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final int[] data);

    public abstract void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final float[] data);

    public abstract void addSparseWeighted(final int index, final long[] sparseIndex, final float[] weights,
        final double[] data);

}
