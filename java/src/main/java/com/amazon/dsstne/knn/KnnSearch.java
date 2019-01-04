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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.amazon.dsstne.knn.DataUtil.Row;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.common.base.Stopwatch;
import lombok.extern.slf4j.Slf4j;

/**
 * CLI to run knn search on given data and input files.
 *
 * @author kiuk
 */

@Slf4j
public class KnnSearch implements Closeable {

    private static final int SEC_IN_MS = 1000;

    //CHECKSTYLE:OFF

    @Parameter(names = "--help", help = true)
    private boolean help;

    @Parameter(names = "--data-files", variableArity = true, description = "data vector file(s)", required = true)
    private List<String> dataFileNames = new ArrayList<>();

    @Parameter(names = "--input-files", variableArity = true, description = "input vector file(s)", required = true)
    private List<String> inputFileNames = new ArrayList<>();

    @Parameter(names = "--output-file", description = "file to write knn results to. stdout if not specified")
    private String outputFileName;

    @Parameter(names = "--k", description = "k")
    private int k = 100;

    @Parameter(names = "--batch-size", description = "batch size")
    private int batchSize = 128;

    @Parameter(names = "--data-type", description = "loads the data as fp16 or fp32")
    private String dataType = "fp32";

    @Parameter(names = "--key-val-delim", description = "delimiter for separating key-vector on each row")
    private String keyValDelim = "\t";

    @Parameter(names = "--vec-delim", description = "delimiter for separating vector elements on each row")
    private String vectorDelim = " ";

    @Parameter(names = "--id-score-sep", description = "separator for id:score in the output")
    private String idScoreSep = ":";

    @Parameter(names = "--score-precision", description = "number of decimal points for output score's precision")
    private int scorePrecision = 3;

    @Parameter(names = "--output-keys-only", description = "do not output scores")
    private boolean outputKeysOnly;

    @Parameter(names = "--report-interval", description = "print performance metrics after n batches")
    private int reportInterval = 1000;

    //CHECKSTYLE:ON

    private int featureSize;
    private String scoreFormat;
    private ExecutorService writerThread;
    private ExecutorService searchThread;

    private BufferedReader reader;
    private PrintWriter writer;

    private KNearestNeighborsCuda knnCuda;

    /**
     * Constructor.
     */
    private KnnSearch(final String[] args) throws IOException {
        JCommander jc = new JCommander(this);
        jc.setProgramName("knn-search");

        try {
            jc.parse(args);

            if (help) {
                jc.usage();
                System.exit(0);
            }
        } catch (Exception e) {
            log.error("Error running command", e);
            jc.usage();
            System.exit(1);
        }

        String inputFile = inputFileNames.get(0);
        this.featureSize = DataUtil.findFeatureSize(new File(inputFile), keyValDelim, vectorDelim);
        log.info("Auto determined feature size = {} from file {}", featureSize, inputFile);

        this.scoreFormat = "%." + scorePrecision + "f";
        this.writerThread = Executors.newSingleThreadScheduledExecutor();
        this.searchThread = Executors.newSingleThreadScheduledExecutor();
        this.reader = createReader();
        this.writer = createWriter();

        this.knnCuda = new KNearestNeighborsCuda(k, batchSize, featureSize, DataType.fromString(this.dataType),
                toFile(dataFileNames), keyValDelim, vectorDelim);
        knnCuda.init();
    }

    private void run() throws IOException {
        log.info("Starting search. Reporting metrics every {} batches", reportInterval);

        Stopwatch timer = Stopwatch.createStarted();
        long totalBatchTime = 0;

        long batchNum = 0;
        String line = reader.readLine();
        while (line != null) {
            Stopwatch batchTimer = Stopwatch.createStarted();

            String[] inputRowIds = new String[batchSize];
            float[] inputVectors = new float[batchSize * featureSize];

            /* Main thread: read input batch */
            int i = 0;
            do {
                Row row = DataUtil.parseLine(line, keyValDelim, vectorDelim);
                inputRowIds[i] = row.key;
                float[] vector = row.vector;
                System.arraycopy(vector, 0, inputVectors, i * featureSize, featureSize);
                i++;
                line = reader.readLine();
            } while (i < batchSize && line != null);

            // batch may not be full if the batchSize does not divide number of input rows
            final int activeBatchSize = i;

            /* Search thread: invoke kernel */
            searchThread.submit(() -> {
                KnnResult result = knnCuda.findKnn(inputVectors);

            /* Write thread: write output */
                writerThread.submit(() -> {
                    for (int j = 0; j < activeBatchSize; j++) {
                        String inputRowId = inputRowIds[j];
                        writer.print(inputRowId);
                        writer.print(keyValDelim);

                        float score = result.getScoreAt(j, 0);
                        String key = result.getKeyAt(j, 0);
                        writer.print(key);
                        if (!outputKeysOnly) {
                            writer.print(idScoreSep);
                            writer.format(scoreFormat, score);
                        }

                        for (int m = 1; m < k; m++) {
                            score = result.getScoreAt(j, m);
                            key = result.getKeyAt(j, m);

                            writer.print(vectorDelim);
                            writer.print(key);
                            if (!outputKeysOnly) {
                                writer.print(idScoreSep);
                                writer.format(scoreFormat, score);
                            }
                        }
                        writer.println();
                    }
                });
            });

            long elapsedBatch = batchTimer.elapsed(TimeUnit.MILLISECONDS);
            long elapsedTotal = timer.elapsed(TimeUnit.SECONDS);
            totalBatchTime += elapsedBatch;

            ++batchNum;

            if (batchNum % reportInterval == 0) {
                log.info(String.format("Processed %7d batches in %4ds. Elapsed %7ds. TPS %7d", batchNum,
                        totalBatchTime / SEC_IN_MS, elapsedTotal,
                        (batchNum * batchSize) / timer.elapsed(TimeUnit.SECONDS)));
                totalBatchTime = 0;
            }
        }

        try {
        /*
         * order matters here! need to wait for the search thread to finish, then writer thread,
         * otherwise we may end up with partially written output.
         */
            searchThread.shutdown();
            searchThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            writerThread.shutdown();
            writerThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        long totalTime = timer.elapsed(TimeUnit.SECONDS);

        log.info("Done processing {} batches in {} s", batchNum, totalTime);
    }

    @Override
    public void close() throws IOException {
        reader.close();
        writer.close();
        knnCuda.close();
    }

    private BufferedReader createReader() throws FileNotFoundException {
        List<InputStream> fis = new ArrayList<>();
        for (String fileName : inputFileNames) {
            fis.add(new FileInputStream(fileName));
        }
        InputStream is = new SequenceInputStream(Collections.enumeration(fis));
        return new BufferedReader(new InputStreamReader(is, Charset.forName("UTF-8")));
    }

    private PrintWriter createWriter() throws IOException {
        OutputStream os;
        if (outputFileName == null) {
            os = System.out;
        } else {
            File outputFile = new File(outputFileName);
            outputFile.getParentFile().mkdirs();
            outputFile.createNewFile();
            os = new FileOutputStream(outputFileName);
        }
        return new PrintWriter(new BufferedWriter(new OutputStreamWriter(os, Charset.forName("UTF-8"))));
    }

    private List<File> toFile(final List<String> fileNames) {
        List<File> files = new ArrayList<>();
        for (String fileName : fileNames) {
            files.add(new File(fileName));
        }
        return files;
    }

    /**
     * Runs knn search.
     */
    public static void main(final String[] args) {
        try (KnnSearch knnSearch = new KnnSearch(args)) {
            knnSearch.run();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
