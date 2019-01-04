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

package com.amazon.dsstne.knn;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class Vector {

    /**
     * Statically creates a builder instance for Vector.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for instances of Vector.
     */
    public static class Builder {

        private int index;

        /**
         * Sets the value of the field "index" to be used for the constructed object.
         *
         * @param index The value of the "index" field.
         *
         * @return This builder.
         */
        public Builder withIndex(int index) {
            this.index = index;
            return this;
        }

        private List<Float> coordinates;

        /**
         * Sets the value of the field "coordinates" to be used for the constructed object.
         *
         * @param coordinates The value of the "coordinates" field.
         *
         * @return This builder.
         */
        public Builder withCoordinates(List<Float> coordinates) {
            this.coordinates = coordinates;
            return this;
        }

        /**
         * Sets the fields of the given instances to the corresponding values recorded when calling the "with*" methods.
         *
         * @param instance The instance to be populated.
         */
        protected void populate(Vector instance) {
            instance.setIndex(this.index);
            instance.setCoordinates(this.coordinates);
        }

        /**
         * Builds an instance of Vector.
         * <p>
         * The built object has its fields set to the values given when calling the "with*" methods of this builder.
         * </p>
         */
        public Vector build() {
            Vector instance = new Vector();

            populate(instance);

            return instance;
        }
    }

    ;

    private int index;
    private List<Float> coordinates;

    public int getIndex() {
        return this.index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public List<Float> getCoordinates() {
        return this.coordinates;
    }

    public void setCoordinates(List<Float> coordinates) {
        this.coordinates = coordinates;
    }

    private static final int classNameHashCode =
        internalHashCodeCompute("com.amazon.dsstne.knn.Vector");

    /**
     * HashCode implementation for Vector
     * based on java.util.Arrays.hashCode
     */
    @Override
    public int hashCode() {
        return internalHashCodeCompute(
            classNameHashCode,
            getIndex(),
            getCoordinates());
    }

    private static int internalHashCodeCompute(Object... objects) {
        return Arrays.hashCode(objects);
    }

    /**
     * Equals implementation for Vector
     * based on instanceof and Object.equals().
     */
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof Vector)) {
            return false;
        }

        Vector that = (Vector) other;

        return
            Objects.equals(getIndex(), that.getIndex())
                && Objects.equals(getCoordinates(), that.getCoordinates());
    }

    @Override
    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append("Vector(");

        ret.append("index=");
        ret.append(String.valueOf(index));
        ret.append(", ");

        ret.append("coordinates=");
        ret.append(String.valueOf(coordinates));
        ret.append(")");

        return ret.toString();
    }
}
