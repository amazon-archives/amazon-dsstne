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

import lombok.Value;

/**
 * Represents a neural network layer. This is a data-only object around
 * DSSTNE's NNLayer with no functionality other than providing information
 * about a layer.
 *
 * @author kiuk
 */
@Value
public class NNLayer {

    /**
     * Same as C++ NNLayer::Kind. Enums ordinals/values must be in sync.
     */
    public enum Kind {
        /*
         * DO NOT CHANGE ORDER
         * THE ENUM POSITION NEEDS TO MATCH
         * THE ENUM VALUES OF C++ version of NNDataSetEnums::Kind
         */
        Input, Hidden, Output, Target;

    }

    /**
     * Same as C++ NNLayer::Attribute. Since attribute is a mask we represent
     * them as int rather than enums.
     */
    public static final class Attribute {
        /*
         * DO NOT CHANGE MASK VALUES
         * UNLESS THEY ALSO CHANGE IN NNLayer::Attribute
         */
        public static final int None = 0x0;
        public static final int Sparse = 0x1;
        public static final int Denoising = 0x2;
        public static final int BatchNormalization = 0x4;
    }

    /**
     * Name of this layer.
     */
    private final String name;

    /**
     * Name of the dataset for this layer.
     */
    private final String datasetName;

    /**
     * {@link Kind} of this layer.
     */
    private final Kind kind;

    /**
     * Attributes of this layer.
     */
    private final int attributes;

    /**
     * Number of dimensions of this layer. (e.g. 3-d is represented in (x,y,z))
     */
    private final int dimensions;

    /**
     * Size of dimension X. Zero if dimension < 1.
     */
    private final int dimX;

    /**
     * Size of dimension Y. Zero if dimension < 2.
     */
    private final int dimY;

    /**
     * Size of dimension Z. Zero if dimension < 3.
     */
    private final int dimZ;

    /**
     * Creates a layer. Use takes int as kind (instead of the enum {@link Kind}) to facilitate creating
     * this class from JNI. Note that this will break if the ordinal of the {@link Kind} enum in Java and C++
     * do not match.
     */
    public NNLayer(final String name, final String datasetName, final int kind, final int attributes,
        final int dimensions, final int dimX,
        final int dimY, final int dimZ) {
        this.name = name;
        this.datasetName = datasetName;
        this.kind = Kind.values()[kind];
        this.attributes = attributes;
        this.dimensions = dimensions;
        this.dimX = dimX;
        this.dimY = dimY;
        this.dimZ = dimZ;
    }

    public Dim getDim() {
        return new Dim(dimensions, dimX, dimY, dimZ, 0);
    }
}
