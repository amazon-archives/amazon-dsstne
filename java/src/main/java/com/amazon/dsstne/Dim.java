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

/**
 * Describes a dimension (typically of either {@link NNDataSet} or {@link NNLayer}).
 */
public class Dim {
    /**
     * 1 for 1-d, 2 for 2-d, 3 for 3-d.
     */
    public final int dimensions;

    /**
     * Length of the first dimension.
     */
    public final int x;

    /**
     * Length of the second dimension. (1 for 1-d data).
     */
    public final int y;

    /**
     * Length of the third dimension. (1 for 1-d and 2-d data).
     */
    public final int z;

    /**
     * Number of examples.
     */
    public final int examples;

    /**
     * Stride between examples.
     */
    public final int stride;

    public Dim(final int dimensions, final int x, final int y, final int z, final int examples) {
        this.dimensions = dimensions;
        this.x = x;
        this.y = y;
        this.z = z;
        this.examples = examples;
        this.stride = x * y * z;
    }

    /**
     * Creates a 1-d dimension.
     */
    public static Dim _1d(final int x, final int examples) {
        return new Dim(1, x, 1, 1, examples);
    }

    /**
     * Creates a 2-d dimension.
     */
    public static Dim _2d(final int x, final int y, final int examples) {
        return new Dim(2, x, y, 1, examples);
    }

    /**
     * Creates a 2-d dimension.
     */
    public static Dim _3d(final int x, final int y, final int z, final int examples) {
        return new Dim(3, x, y, z, examples);
    }
}
