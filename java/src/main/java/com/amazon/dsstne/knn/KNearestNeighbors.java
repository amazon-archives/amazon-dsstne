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

import java.util.List;

/**
 * @author kiuk
 */
public interface KNearestNeighbors {

    /**
     * Finds and returns the k-nearest-neighbors for the input vector.
     */
    NearestNeighbors findKnn(final int k, final Vector inputVector);

    /**
     * Batch call to {@link #findKnn(int, Vector)}.
     */
    List<NearestNeighbors> findKnnBatch(final int k, final List<Vector> inputVectors);

    /**
     * Maximum k supported by this knn.
     */
    int getMaxK();

    /**
     * The length of the features in the matrix. The input vector should have the same length.
     */
    int getFeatureSize();

}
