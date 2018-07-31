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
 * Same as src/amazon/dsstne/engine/NNDataSetEnums.h. This class exists
 * so that we can pass attribute information of the layers in the network between
 * C++ and Java. The values here must be in sync with the ones in the NNDataSetEnums.h.
 */
public class NNDataSetEnums {

    public static final class Attribute {
        public static final int None = 0x0;
        public static final int Sparse = 0x1;
        public static final int Boolean = 0x2;
        public static final int Compressed = 0x4;
        public static final int Recurrent = 0x8;
        public static final int Mutable = 0xF;
        public static final int SparseIgnoreZero = 0x20;
        public static final int Indexed = 0x40;
        public static final int Weighted = 0x80;
    }

    /**
     * Corresponds to NNDataSetEnums::Kind.
     */
    public enum Kind {
        /*
         * DO NOT CHANGE ORDER
         * THE ENUM POSITION NEEDS TO MATCH
         * THE ENUM VALUES OF C++ version of NNDataSetEnums::Kind
         */
        Numeric, // 0
        Image, // 1
        Audio; // 2

    }

    /**
     * Corresponds to NNDataSetEnums::Sharding.
     */
    public enum Sharding {
        /*
         * DO NOT CHANGE ORDER
         * THE ENUM POSITION NEEDS TO MATCH
         * THE ENUM VALUES OF C++ version of NNDataSetEnums::Sharding
         */
        None, // 0
        Model, // 1
        Data; // 2
    }

    /**
     * Corresponds to NNDataSetEnums::DataType.
     */
    public enum DataType {
        /*
         * DO NOT CHANGE ORDER
         * THE ENUM POSITION NEEDS TO MATCH
         * THE ENUM VALUES OF C++ version of NNDataSetEnums::Sharding
         */
        UInt, // 0
        Int, // 1
        LLInt, // 2
        ULLInt, // 3
        Float, // 4
        Double, // 5
        RGB8, // 6
        RGB16, // 7
        UChar, // 8
        Char; // 9

        /**
         * The size of the data type in bytes.
         */
        public static int sizeof(final DataType dataType) {
            switch (dataType) {
            case Int:
                return Integer.SIZE / 8;
            case LLInt:
                return Long.SIZE / 8;
            case Float:
                return java.lang.Float.SIZE / 8;
            case Double:
                return java.lang.Double.SIZE / 8;
            case Char:
                return Character.SIZE / 8;
            default:
                throw new IllegalArgumentException(dataType + " not supported in java binding");
            }
        }
    }
}
