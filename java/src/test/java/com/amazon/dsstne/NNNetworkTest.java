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

package com.amazon.dsstne;

import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import com.amazon.dsstne.NNDataSetEnums.DataType;
import com.amazon.dsstne.data.DenseNNDataSet;
import com.amazon.dsstne.data.OutputNNDataSet;
import com.amazon.dsstne.data.SparseNNDataSet;

public class NNNetworkTest {
    private static final String DIR_SUFFIX = "src/test/java/com/amazon/dsstne/test-data/";
    private final String networkFile = DIR_SUFFIX + "movielens_20m_autoencoder.nc";
    private final String inputFile = DIR_SUFFIX + "movielens_input_1024.txt";
    private final String indexFile = DIR_SUFFIX + "index.txt";
    private final String expectedOutputFile = DIR_SUFFIX + "expected_output_1024.txt";

    private final int k = 16;
    private final int batchSize = 256;
    private final double sparseDensity = 0.1d;

    private NetworkConfig config;
    private NNNetwork network;
    private Map<String, Long> index;
    private Map<Long, String> rIndex;
    private List<String[]> expectedKeys;
    private List<float[]> expectedScores;

    private final float testDelta = 0.01f;

    @Before
    public void setup() throws IOException {
        config = NetworkConfig.with().networkFilePath(networkFile).batchSize(batchSize).k(k).build();
        network = Dsstne.load(config);
        System.out.println("Loaded network: \n" + network);

        index = new HashMap<>();
        rIndex = new HashMap<>();
        parseIndex(indexFile, index, rIndex);

        expectedKeys = new ArrayList<>();
        expectedScores = new ArrayList<>();
        readExpectedOutput(expectedKeys, expectedScores);
    }

    @After
    public void teardown() {
        network.close();
        index.clear();
        rIndex.clear();
    }

    @Test
    public void testPredictSparse() throws IOException {
        NNDataSet[] inputDatasets = new NNDataSet[network.getInputLayers().length];
        for (int i = 0; i < network.getInputLayers().length; ++i) {
            NNLayer inputLayer = network.getInputLayers()[i];
            inputDatasets[i] =
                new SparseNNDataSet(new Dim(inputLayer.getDim(), batchSize), DataType.Int, sparseDensity);
            inputDatasets[i].setName(inputLayer.getDatasetName());
            inputDatasets[i].setLayerName(inputLayer.getName());
        }

        network.load(inputDatasets);
        System.out.println("Loaded " + inputDatasets.length + " input datasets to network");

        SparseDataProvider dp = new SparseDataProvider(inputFile, index);
        assertPredictions(inputDatasets, dp);
    }

    private void assertPredictions(final NNDataSet[] inputDatasets, final DataProvider dp) throws IOException {
        for (int batch = 0; dp.getBatch(inputDatasets[0]); ++batch) {
            long start = System.currentTimeMillis();
            OutputNNDataSet[] outputDatasets = network.predict(inputDatasets);
            long end = System.currentTimeMillis();
            System.out.println("==== Batch # " + (batch + 1) + " Predict. Took " + (end - start) + " ms ====");
            OutputNNDataSet output = outputDatasets[0];
            float[] scores = output.getScores();
            long[] indexes = output.getIndexes();

            for (int i = 0; i < batchSize; ++i) {
                int pos = batch * batchSize + i;
                String[] expectedKey = expectedKeys.get(pos);
                float[] expectedScr = expectedScores.get(pos);

                String failMessage = null;
                for (int j = 0; j < k; ++j) {
                    String actualKey = rIndex.get(indexes[i * k + j]);
                    float actualScore = scores[i * k + j];
                    if (!expectedKey[j].equals(actualKey) || Math.abs(expectedScr[j] - actualScore) > testDelta) {
                        failMessage =
                            String.format("Unequal key or score at input %d, k=%d. Expected: %s,%5.3f Actual: %s,%5.3f",
                                pos + 1, j + 1, expectedKey[j], expectedScr[j], actualKey, actualScore);
                    }
                }

                if (failMessage != null) {
                    System.out.println("== Actual ==");
                    System.out.print((pos + 1) + "\t");
                    for (int j = 0; j < k; ++j) {
                        String actualKey = rIndex.get(indexes[i * k + j]);
                        float actualScore = scores[i * k + j];
                        System.out.print(String.format("%s,%1.3f:", actualKey, actualScore));
                    }
                    System.out.println();
                    System.out.println("== Expected ==");
                    System.out.print((pos + 1) + "\t");
                    for (int j = 0; j < k; ++j) {
                        System.out.print(String.format("%s,%1.3f:", expectedKey[j], expectedScr[j]));
                    }
                    System.out.println();
                    fail(failMessage);
                }
            }
        }
    }

    @Ignore
    @Test
    public void testPredictDense() throws IOException {
        NNDataSet[] inputDatasets = new NNDataSet[network.getInputLayers().length];
        for (int i = 0; i < network.getInputLayers().length; ++i) {
            NNLayer inputLayer = network.getInputLayers()[i];
            inputDatasets[i] =
                new DenseNNDataSet(new Dim(inputLayer.getDim(), batchSize), DataType.Int);
            inputDatasets[i].setName(inputLayer.getDatasetName());
            inputDatasets[i].setLayerName(inputLayer.getName());
        }

        network.load(inputDatasets);
        System.out.println("Loaded " + inputDatasets.length + " input datasets to network");

        DenseDataProvider dp = new DenseDataProvider(inputDatasets[0].getStride(), inputFile, index);

        assertPredictions(inputDatasets, dp);
    }

    private void readExpectedOutput(List<String[]> allIndexes, List<float[]> allScores) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(expectedOutputFile))) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] kv = line.split("\t");
                int pos = Integer.parseInt(kv[0]);
                String[] vs = kv[1].split(":");
                String[] idx = new String[vs.length];
                float[] scores = new float[vs.length];
                for (int i = 0; i < vs.length; ++i) {
                    String v = vs[i];
                    idx[i] = v.split(",")[0];
                    scores[i] = Float.parseFloat(v.split(",")[1]);
                }
                // pos is 1 indexed
                allScores.add(pos - 1, scores);
                allIndexes.add(pos - 1, idx);
            }
        }
    }

    private void parseIndex(final String indexFile, final Map<String, Long> index,
        final Map<Long, String> rIndex)
        throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(indexFile))) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] split = line.split("\t");
                String key = split[0];
                long idx = Long.parseLong(split[1]);
                if (index.put(key, idx) != null) {
                    throw new IllegalArgumentException("Duplicate key found: " + key);
                }
                if (rIndex.put(idx, key) != null) {
                    throw new IllegalArgumentException("Duplicate index found: " + idx);
                }
            }
        }
    }

    static abstract class DataProvider implements Closeable {
        final Map<String, Long> indexMap;
        final File inputFile;
        final BufferedReader input;

        DataProvider(final String inputFile, final Map<String, Long> indexMap) throws IOException {
            this.indexMap = indexMap;
            this.inputFile = new File(inputFile);
            this.input = new BufferedReader(new FileReader(inputFile));
        }

        boolean getBatch(final NNDataSet dataset) throws IOException {
            int numExamples = dataset.getExamples();

            boolean eof = false;
            for (int i = 0; i < numExamples; ++i) {
                String line = input.readLine();

                if (line == null && i == 0) {
                    eof = true;
                    break;
                } else if (line != null) {
                    String split[] = line.split("\t")[1].split(":");
                    int[] data = new int[split.length];
                    long[] index = new long[split.length];

                    for (int j = 0; j < split.length; ++j) {
                        String key = split[j].split(",")[0];
                        Long idx = indexMap.get(key);

                        if (idx == null) {
                            throw new RuntimeException("No index found for key: " + key);
                        }

                        data[j] = 1;
                        index[j] = idx;
                    }
                    addExample(dataset, i, index, data);
                }

            }
            return !eof;
        }

        protected abstract void addExample(NNDataSet dataset, int idx, final long[] index, final int[] data);

        @Override
        public void close() throws IOException {
            input.close();
        }
    }

    static class DenseDataProvider extends DataProvider {
        private final int[] dataBuffer;

        DenseDataProvider(final int stride, final String inputFile, final Map<String, Long> indexMap)
            throws IOException {
            super(inputFile, indexMap);
            this.dataBuffer = new int[stride];
        }

        @Override
        protected void addExample(final NNDataSet dataset, final int idx, final long[] index, final int[] data) {
            Arrays.fill(dataBuffer, 0); // clear the buffer
            for (int i = 0; i < index.length; ++i) {
                dataBuffer[(int) index[i]] = data[i];
            }
            dataset.add(idx, dataBuffer);
        }
    }

    static class SparseDataProvider extends DataProvider {
        SparseDataProvider(final String inputFile, final Map<String, Long> indexMap) throws IOException {
            super(inputFile, indexMap);
        }

        @Override
        protected void addExample(final NNDataSet dataset, final int idx, final long[] index, final int[] data) {
            dataset.addSparse(idx, index, data);
        }

    }
}
