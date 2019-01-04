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

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * Provides common functionality to the {@link KNearestNeighbors} implementation.
 * One should prefer to extend from this class rather than implementing {@link KNearestNeighbors} directly.
 */
@RequiredArgsConstructor
@Getter(AccessLevel.PROTECTED)
public abstract class AbstractKNearestNeighbors implements KNearestNeighbors {

    private final String label;
    private final int maxK;
    private final int featureSize;
    private final int batchSize;

    protected void validateFeatureSize(final Vector inputVector) {
        int size = inputVector.getCoordinates().size();
        if (size != featureSize) {
            throw new IllegalArgumentException("feature size: " + size + " should equal: " + featureSize);
        }
    }

    protected void validateFeatureSize(final List<Vector> inputBatch) {
        inputBatch.forEach(this::validateFeatureSize);
    }

    protected void validateK(final int k) {
        if (k <= 0 || k > maxK) {
            throw new IllegalArgumentException("k should be > 0 and < " + maxK + ". Given: " + k);
        }
    }

    @Override
    public int getMaxK() {
        return maxK;
    }

    @Override
    public int getFeatureSize() {
        return featureSize;
    }
}
