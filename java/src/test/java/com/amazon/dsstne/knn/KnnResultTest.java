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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class KnnResultTest {

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidResults() {
        new KnnResult(new String[1], new float[2], 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidK() {
        new KnnResult(new String[3], new float[3], 2);
    }

    @Test
    public void testGetK() {
        KnnResult result = new KnnResult(new String[6], new float[6], 3);
        assertEquals(3, result.getK());
    }

    @Test
    public void testGetBatchSize() {
        KnnResult result = new KnnResult(new String[32], new float[32], 8);
        assertEquals(32 / 8, result.getBatchSize());
    }
}