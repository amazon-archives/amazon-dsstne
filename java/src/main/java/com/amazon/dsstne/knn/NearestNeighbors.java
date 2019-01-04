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

public class NearestNeighbors {

  /**
   * Statically creates a builder instance for NearestNeighbors.
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Fluent builder for instances of NearestNeighbors.
   */
  public static class Builder {

    private int index;
    /**
     * Sets the value of the field "index" to be used for the constructed object.
     * @param index
     *   The value of the "index" field.
     * @return
     *   This builder.
     */
    public Builder withIndex(int index) {
      this.index = index;
      return this;
    }

    private List<Neighbor> neighbors;
    /**
     * Sets the value of the field "neighbors" to be used for the constructed object.
     * @param neighbors
     *   The value of the "neighbors" field.
     * @return
     *   This builder.
     */
    public Builder withNeighbors(List<Neighbor> neighbors) {
      this.neighbors = neighbors;
      return this;
    }

    /**
     * Sets the fields of the given instances to the corresponding values recorded when calling the "with*" methods.
     * @param instance
     *   The instance to be populated.
     */
    protected void populate(NearestNeighbors instance) {
      instance.setIndex(this.index);
      instance.setNeighbors(this.neighbors);
    }

    /**
     * Builds an instance of NearestNeighbors.
     * <p>
     * The built object has its fields set to the values given when calling the "with*" methods of this builder.
     * </p>
     */
    public NearestNeighbors build() {
      NearestNeighbors instance = new NearestNeighbors();

      populate(instance);

      return instance;
    }
  };

  private int index;
  private List<Neighbor> neighbors;

  public int getIndex() {
    return this.index;
  }

  public void setIndex(int index) {
    this.index = index;
  }

  public List<Neighbor> getNeighbors() {
    return this.neighbors;
  }

  public void setNeighbors(List<Neighbor> neighbors) {
    this.neighbors = neighbors;
  }

  private static final int classNameHashCode =
      internalHashCodeCompute("com.amazon.dsstne.knn.NearestNeighbors");

  /**
   * HashCode implementation for NearestNeighbors
   * based on java.util.Arrays.hashCode
   */
  @Override
  public int hashCode() {
    return internalHashCodeCompute(
        classNameHashCode,
        getIndex(),
        getNeighbors());
  }

  private static int internalHashCodeCompute(Object... objects) {
    return Arrays.hashCode(objects);
  }

  /**
   * Equals implementation for NearestNeighbors
   * based on instanceof and Object.equals().
   */
  @Override
  public boolean equals(final Object other) {
    if (!(other instanceof NearestNeighbors)) {
      return false;
    }

    NearestNeighbors that = (NearestNeighbors) other;

    return
        Objects.equals(getIndex(), that.getIndex())
        && Objects.equals(getNeighbors(), that.getNeighbors());
  }
  @Override
  public String toString() {
    StringBuilder ret = new StringBuilder();
    ret.append("NearestNeighbors(");

    ret.append("index=");
    ret.append(String.valueOf(index));
    ret.append(", ");

    ret.append("neighbors=");
    ret.append(String.valueOf(neighbors));
    ret.append(")");

    return ret.toString();
  }

}
