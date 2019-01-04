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

import lombok.Value;

/**
 * Result of {@link KNearestNeighborsCuda#findKnn(int, float[], int)} .
 * The ids and scores are aligned by index. That is, (ids[i], scores[i]) defines the
 * neighbor's id and score. The scores are sorted in ascending order (closest neighbors first).
 *
 * @author kiuk
 */
@Value
public class KnnResult {
    private final String[] keys;
    private final float[] scores;
    private final int k;

    public KnnResult(final String[] keys, final float[] scores, final int k) {
        int slen = scores.length;
        int ilen = keys.length;
        if (slen != ilen) {
            throw new IllegalArgumentException(
                    "scores and indexes have different lengths (scores: " + slen + ", indexes: " + ilen + ")");
        }
        if (scores.length % k != 0) {
            throw new IllegalArgumentException("k: " + k + " must divide the length of data: " + scores.length);
        }

        this.keys = keys;
        this.scores = scores;
        this.k = k;
    }

    /**
     * Returns the batch size (the number of input vectors in the findKnn batch).
     */
    public int getBatchSize() {
        return scores.length / k;
    }

    /**
     * Id of the i^th neighbor in the row^th row in the batch.
     */
    public String getKeyAt(final int row, final int i) {
        return keys[row * k + i];
    }

    /**
     * Score of the i^th neighbor in the row^th row in the batch.
     */
    public float getScoreAt(final int row, final int i) {
        return scores[row * k + i];
    }
}
