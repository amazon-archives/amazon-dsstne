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

import java.util.ArrayList;
import java.util.List;

/**
 * Simply delegates the findKnn call to the underlying KnnCuda implementation.
 * NOT thread-safe. Intended to use for testing since this method does not batch properly.
 *
 * @author kiuk
 */
public class UnbatchedKNearestNeighbors extends AbstractKNearestNeighbors {

    private final KNearestNeighborsCuda knnCuda;
    private final float[] inputs; // input vectors
    private final float[] scores; // output scores
    private final String[] ids; // output ids (keys)

    UnbatchedKNearestNeighbors(final KNearestNeighborsCuda knnCuda) {
        this(null, knnCuda);
    }

    public UnbatchedKNearestNeighbors(final String label, final KNearestNeighborsCuda knnCuda) {
        super(label, knnCuda.getMaxK(), knnCuda.getFeatureSize(), knnCuda.getBatchSize());
        int batchSize = getBatchSize();
        int featureSize = getFeatureSize();
        int maxK = getMaxK();

        this.knnCuda = knnCuda;
        this.inputs = new float[batchSize * featureSize];
        this.scores = new float[batchSize * maxK];
        this.ids = new String[batchSize * maxK];
    }

    @Override
    public NearestNeighbors findKnn(final int k, final Vector inputVector) {
        validateK(k);
        validateFeatureSize(inputVector);

        List<Float> coordinates = inputVector.getCoordinates();

        for (int i = 0; i < coordinates.size(); i++) {
            inputs[i] = coordinates.get(i);
        }

        knnCuda.findKnn(inputs, scores, ids);

        List<Neighbor> nearestNeighbors = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            float score = scores[i];
            String id = ids[i];
            Neighbor neighbor = Neighbor.builder().withId(id).withScore(score).build();
            nearestNeighbors.add(neighbor);
        }

        return NearestNeighbors.builder().withIndex(inputVector.getIndex()).withNeighbors(nearestNeighbors).build();
    }

    @Override
    public List<NearestNeighbors> findKnnBatch(final int k, final List<Vector> inputBatch) {
        List<NearestNeighbors> results = new ArrayList<>(inputBatch.size());
        for (Vector vector : inputBatch) {
            results.add(findKnn(k, vector));
        }
        return results;
    }
}
