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
import java.util.Objects;

public class Neighbor {

  /**
   * Statically creates a builder instance for Neighbor.
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Fluent builder for instances of Neighbor.
   */
  public static class Builder {

    private String id;
    /**
     * Sets the value of the field "id" to be used for the constructed object.
     * @param id
     *   The value of the "id" field.
     * @return
     *   This builder.
     */
    public Builder withId(String id) {
      this.id = id;
      return this;
    }

    private float score;
    /**
     * Sets the value of the field "score" to be used for the constructed object.
     * @param score
     *   The value of the "score" field.
     * @return
     *   This builder.
     */
    public Builder withScore(float score) {
      this.score = score;
      return this;
    }

    /**
     * Sets the fields of the given instances to the corresponding values recorded when calling the "with*" methods.
     * @param instance
     *   The instance to be populated.
     */
    protected void populate(Neighbor instance) {
      instance.setId(this.id);
      instance.setScore(this.score);
    }

    /**
     * Builds an instance of Neighbor.
     * <p>
     * The built object has its fields set to the values given when calling the "with*" methods of this builder.
     * </p>
     */
    public Neighbor build() {
      Neighbor instance = new Neighbor();

      populate(instance);

      return instance;
    }
  };

  private String id;
  private float score;

  public String getId() {
    return this.id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public float getScore() {
    return this.score;
  }

  public void setScore(float score) {
    this.score = score;
  }

  private static final int classNameHashCode =
      internalHashCodeCompute("com.amazon.dsstne.knn.Neighbor");

  /**
   * HashCode implementation for Neighbor
   * based on java.util.Arrays.hashCode
   */
  @Override
  public int hashCode() {
    return internalHashCodeCompute(
        classNameHashCode,
        getId(),
        getScore());
  }

  private static int internalHashCodeCompute(Object... objects) {
    return Arrays.hashCode(objects);
  }

  /**
   * Equals implementation for Neighbor
   * based on instanceof and Object.equals().
   */
  @Override
  public boolean equals(final Object other) {
    if (!(other instanceof Neighbor)) {
      return false;
    }

    Neighbor that = (Neighbor) other;

    return
        Objects.equals(getId(), that.getId())
        && Objects.equals(getScore(), that.getScore());
  }
  @Override
  public String toString() {
    StringBuilder ret = new StringBuilder();
    ret.append("Neighbor(");

    ret.append("id=");
    ret.append(String.valueOf(id));
    ret.append(", ");

    ret.append("score=");
    ret.append(String.valueOf(score));
    ret.append(")");

    return ret.toString();
  }

}
