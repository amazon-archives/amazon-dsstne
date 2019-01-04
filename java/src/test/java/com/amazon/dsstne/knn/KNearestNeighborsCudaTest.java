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

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

public class KNearestNeighborsCudaTest {

    /**
     * Prints the matrix defined in the <code>data</code>.
     */
    static void printMatrix(final float[] data, final int rows, final int cols) {
        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();
        for (int j = 0; j < cols; j++) {
            System.out.format("%9d|\t", j);
        }
        System.out.println();

        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.format("%10.3f\t", data[i * cols + j]);
            }
            System.out.println();
        }
        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();
    }

    /**
     * Prints the matrix defined in the <code>data</code>.
     */
    static void printMatrix(final String[] data, final int rows, final int cols) {
        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();
        for (int j = 0; j < cols; j++) {
            System.out.format("%9d|\t", j);
        }
        System.out.println();

        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.format("%10s\t", data[i * cols + j]);
            }
            System.out.println();
        }
        for (int j = 0; j < cols; j++) {
            System.out.print("----------\t");
        }
        System.out.println();
    }

    @Ignore
    @Test
    public void testDelimiter() throws IOException {
        int maxK = 8; // currently we support up to k = 128
        int batchSize = 16; // ideally multiples of 32
        int featureSize = 256; // number of columns (e.g. features) in the dataset
        Map<File, Integer> fileToDevice = new HashMap<>();
        fileToDevice.put(new File("tst/amazon/cuda/algorithms/data/data-delim.txt"), 0);

        char keyValueDelim = ':';
        char vectorDelim = ',';

        KNearestNeighborsCuda knnCuda =
                new KNearestNeighborsCuda(maxK, batchSize, featureSize, DataType.FP32, fileToDevice, keyValueDelim,
                        vectorDelim);
        knnCuda.init();

        float[] batchInputs = new float[batchSize * featureSize]; // input vectors
        float[] batchScores = new float[batchSize * maxK]; // output scores
        final String[] batchIds = new String[batchSize * maxK]; // output ids (keys)

        knnCuda.findKnn(batchInputs, batchScores, batchIds);

        printMatrix(batchScores, batchSize, maxK);
        printMatrix(batchIds, batchSize, maxK);

        knnCuda.close();
    }

    @Test
    public void testToChar() {
        assertEquals('\t', KNearestNeighborsCuda.toChar("\t"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testToChar_zero_length() {
        KNearestNeighborsCuda.toChar("");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testToChar_length_greater_than_one() {
        KNearestNeighborsCuda.toChar("\t ");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDataTypeFromString_NoSuchEnum() {
        DataType.fromString("fp64");
    }
}
