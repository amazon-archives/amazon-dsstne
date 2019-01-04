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

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Finds k-nearest-neighbors. Internally calls a CUDA implementation.
 */
@Slf4j
@RequiredArgsConstructor
public class KNearestNeighborsCuda implements Closeable {
    private static final long NULLPTR = 0;
    private static final String LIBNAME = "dsstne_knn_java";
    private static final char DEFAULT_KEYVAL_DELIM = '\t';
    private static final char DEFAULT_VEC_DELIM = ' ';
    private static final DataType DEFAULT_DATA_TYPE = DataType.FP32;

    @Getter
    private final int maxK;
    @Getter
    private final int batchSize;
    @Getter
    private final int featureSize;
    @Getter
    private final DataType dataType;
    private final Map<File, Integer> fileDeviceMapping;
    private final char keyValueDelim;
    private final char vectorDelim;

    private volatile long ptr = NULLPTR;

    /**
     * Loads the data files onto the GPU. One device per file starting from device 0 incrementing by 1.
     */
    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize,
        final List<File> dataFiles) {
        this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, dataFiles);
    }

    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize, final DataType dataType,
        final List<File> dataFiles) {
        this(maxK, batchSize, featureSize, dataType, toMapInIndexOrder(dataFiles), DEFAULT_KEYVAL_DELIM,
            DEFAULT_VEC_DELIM);
    }

    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize, final DataType dataType,
        final List<File> dataFiles, final String keyValueDelim, final String vectorDelim) {
        this(maxK, batchSize, featureSize, dataType, toMapInIndexOrder(dataFiles), keyValueDelim, vectorDelim);
    }

    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize, final int device,
        final File dataFile) {
        this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, Collections.singletonMap(dataFile, device),
            DEFAULT_KEYVAL_DELIM, DEFAULT_VEC_DELIM);
    }

    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize, final DataType dataType,
        final Map<File, Integer> fileDeviceMapping, final String keyValueDelim, final String vectorDelim) {
        this(maxK, batchSize, featureSize, dataType, fileDeviceMapping, toChar(keyValueDelim), toChar(vectorDelim));
    }

    public KNearestNeighborsCuda(final int maxK, final int batchSize, final int featureSize,
        final Map<File, Integer> fileDeviceMapping, final char keyValueDelim, final char vectorDelim) {
        this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, fileDeviceMapping, keyValueDelim, vectorDelim);
    }

    static char toChar(final String in) {
        if (in.length() != 1) {
            throw new IllegalArgumentException("String: " + in + " is not length 1, cannot convert to char");
        } else {
            return in.charAt(0);
        }
    }

    static Map<File, Integer> toMapInIndexOrder(final List<File> dataFiles) {
        Map<File, Integer> mapping = new LinkedHashMap<>();
        if (dataFiles != null) {
            for (int i = 0; i < dataFiles.size(); i++) {
                mapping.put(dataFiles.get(i), i);
            }
        }
        return mapping;
    }

    public void init() {
        log.info("Loading library: {}", LIBNAME);
        System.loadLibrary(LIBNAME);
        log.info("Loaded library: {}", LIBNAME);

        String[] files = new String[fileDeviceMapping.size()];
        int[] devices = new int[fileDeviceMapping.size()];
        int uniqueDevices = uniqueDevices();

        log.info("Initializing with maxK: {}, batchSize: {}, devices: {}", maxK, batchSize, uniqueDevices);
        ptr = initialize(maxK, batchSize, uniqueDevices, dataType.ordinal());

        int i = 0;
        for (Entry<File, Integer> entry : fileDeviceMapping.entrySet()) {
            files[i] = entry.getKey().getAbsolutePath();
            devices[i] = entry.getValue();
            i++;
        }

        try {
            log.info("Loading data onto devices. {}", fileDeviceMapping);
            load(files, devices, keyValueDelim, vectorDelim, ptr);
        } catch (IOException e) {
            throw new RuntimeException("Error loading files: " + fileDeviceMapping, e);
        }
    }

    private int uniqueDevices() {
        return new HashSet<>(fileDeviceMapping.values()).size();
    }

    /**
     * Shortcut to findKnn(MAX_K, inputVectors, FEATURE_SIZE, scores, indexes).
     */
    public void findKnn(final float[] inputVectors, final float[] scores, final String[] keys) {
        findKnn(maxK, inputVectors, featureSize, scores, keys);
    }

    /**
     * @see #findKnn(int, float[], int, int, float[], String[])
     */
    public void findKnn(final float[] inputVectors, final int activeBatchSize, final float[] scores,
        final String[] keys) {
        findKnn(maxK, inputVectors, activeBatchSize, featureSize, scores, keys);
    }

    /**
     * @see #findKnn(int, float[], int, int, float[], String[])
     */
    public void findKnn(final int k, final float[] inputVectors, final int width, final float[] scores,
        final String[] keys) {
        findKnn(k, inputVectors, batchSize, width, scores, keys);
    }

    /**
     * Returns the k-nearest neighbors of each vector. Vectors are represented as a flattened 2-D array
     * where the <code>width</code> is the length of each vector. The k-nearest neighbors will be written to
     * the given scores and indexes arrays which are assumed to have size = batchSize * maxK. If k is less
     * than maxK, then only the first k elements of scores and indexes are valid. The <code>activeBatchSize</code>
     * indicates how many rows from the <code>inputVectors</code> to compute knn for. For example if this class
     * has been configured with batchSize = 32, but only 16/32 vectors are set in inputVectors, then passing
     * activeBatchSize = 16 hints to the underlying CUDA algorithm that only the first 16 input rows are actually
     * set. No guarantees are given that the underlying CUDA algorithm will actually take this hint and perform
     * optimizations. One should assume this parameter as a "best effort" optimization.
     */
    public void findKnn(final int k, final float[] inputVectors, final int activeBatchSize, final int width,
        final float[] scores, final String[] keys) {
        validateArguments(k, inputVectors, activeBatchSize, width);

        int outputLength = batchSize * maxK;
        if (scores.length != outputLength) {
            throw new IllegalArgumentException(
                "output scores array must be of size: " + outputLength + " (batchSize x maxK)");
        }
        if (keys.length != outputLength) {
            throw new IllegalArgumentException(
                "output keys array must be of size: " + outputLength + " (batchSize x maxK)");
        }

        findKnn(k, inputVectors, batchSize, width, scores, keys, ptr);
    }

    /**
     * Shortcut to findKnn(MAX_K, inputVectors).
     */
    public KnnResult findKnn(final float[] inputVectors) {
        return findKnn(maxK, inputVectors);
    }

    /**
     * Same as {@link #findKnn(float[], float[], String[])}, except that the method returns
     * the results (instead of using output parameters). Note that since the results are returned
     * this method will create a new result object on each invocation. For low latency applications
     * where memory allocations should be minimized, prefer finding knn by passing an output parameter.
     */
    public KnnResult findKnn(final int k, final float[] inputVectors) {
        validateArguments(k, inputVectors, batchSize, featureSize);
        return findKnn(k, inputVectors, batchSize, featureSize, ptr);
    }

    private void validateArguments(final int k, final float[] inputVectors, final int activeBatchSize,
        final int width) {
        if (width < 1) {
            throw new IllegalArgumentException("dimension of the input vector should be at least one");
        }
        if (k > maxK) {
            throw new IllegalArgumentException("k = " + k + " is greater than maxK = " + maxK);
        }
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        if (inputVectors.length % width != 0) {
            throw new IllegalArgumentException(
                "width: " + width + " does not divide the vectors: " + inputVectors.length);
        }
        int actualBatchSize = inputVectors.length / width;
        if (actualBatchSize != batchSize) {
            throw new IllegalArgumentException(actualBatchSize + " is not equal to configured batchSize: " + batchSize);
        }
        if (inputVectors.length == 0) {
            throw new IllegalArgumentException("input vector contain at least one vector");
        }

        if (activeBatchSize > batchSize) {
            throw new IllegalArgumentException(
                "active batch size must be less than or equal to batchSize: " + batchSize);
        }

        if (activeBatchSize > actualBatchSize) {
            throw new IllegalArgumentException(
                "active batch size must be less than or equal to actual batch size: " + actualBatchSize
            );
        }
    }

    @Override
    public void close() throws IOException {
        if (ptr != 0) {
            shutdown(ptr);
            ptr = 0;
        }
    }

    /**
     * Initializes and allocates CPU/GPU memory and returns a pointer to the structure
     * that maintains the pointers to the allocated memory. numGPUs specifies on how many
     * GPUs we pre-allocate memory.
     */
    private static native long initialize(final int maxK, final int maxBatchSize, final int numGPUs,
        final int dataType);

    /**
     * Load the data (e.g. asin embeddings) contained in the file into the specified GPU.
     * The <code>filenames</code> and <code>devices</code> should be aligned. For example,
     * if <code>filenames = ["file1", "file2", "file3"]</code> and <code>devices = [0,0,1]</code>,
     * then <code>file1,file2</code> are loaded into GPU <code>0</code>
     * and <code>file2</code> is loaded to GPU <code>1</code>.
     */
    private static native void load(final String[] filenames, final int[] devices, final char keyValDelim,
        final char vectorDelim, final long ptr) throws IOException;

    /**
     * Frees the memory.
     */
    private static native void shutdown(final long ptr);

    /**
     * Finds the k-nearest-neighbors to the given <code>input</code> batch.
     * The <code>size</code> is the number if active (set) rows in <code>input</code> (< input.length).
     * The <code>width</code> is the number of columns in the input batch.
     */
    private static native void findKnn(final int k, final float[] input, final int size, final int width,
        final float[] scores,
        final String[] keys, final long ptr);

    /**
     * Finds the k-nearest-neighbors just like {@link #findKnn(int, float[], int, int, float[], String[], long)},
     * but this method returns the result object rather than taking an output parameter.
     * Because this method creates a new result object each time it pays the cost of object/array allocation.
     * Prefer to use {@link #findKnn(int, float[], int, int, float[], String[], long)} for low latency applications.
     */
    private static native KnnResult findKnn(final int k, final float[] input, final int size, final int width,
        final long ptr);
}
