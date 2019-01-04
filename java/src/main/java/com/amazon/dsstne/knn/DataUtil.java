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
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

import lombok.RequiredArgsConstructor;

/**
 * Utility methods to deal with parsing and reading data.
 */
public class DataUtil {

    /**
     * Parses the data files using the provided key-value delimiter and vector delimiter to
     * determine the feature size. Further ensures that all files have the same feature size.
     * Otherwise, throws {@link IllegalArgumentException}
     */
    public static int findFeatureSize(final Collection<File> dataFiles, final String keyValDelim,
            final String vectorDelim) {
        if (dataFiles.size() < 1) {
            throw new IllegalArgumentException("data files is empty, must pass at least one file");
        }

        Map<File, Integer> dataToFeatureSize = new LinkedHashMap<>();

        for (File dataFile : dataFiles) {
            int featureSize = findFeatureSize(dataFile, keyValDelim, vectorDelim);
            dataToFeatureSize.put(dataFile, featureSize);
        }

        int featureSize = 0;
        for (Integer fs : dataToFeatureSize.values()) {
            if (featureSize == 0) {
                featureSize = fs;
            } else {
                if (featureSize != fs) {
                    throw new IllegalArgumentException(
                            "Feature sizes are different in data files: " + dataToFeatureSize);
                }
            }
        }
        return featureSize;
    }

    /**
     * Parses the data file using the provided key-value delimiter and vector delimiter to
     * determine the feature size.
     */
    public static int findFeatureSize(final File dataFile, final String keyValDelim, final String vectorDelim) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(dataFile), Charset.forName("UTF-8")))) {
            String line = reader.readLine();
            if (line == null) {
                throw new IllegalArgumentException("file: " + dataFile + " is empty");
            }
            return findFeatureSize(line, keyValDelim, vectorDelim);
        } catch (IOException e) {
            // should never happen
            throw new RuntimeException(e);
        }
    }

    /**
     * Parses the given line using the provided key-value delimiter and vector delimiter and
     * returns the feature size (the number of elements in the vector). The line is expected to be in:
     * key[key-value-delimiter]0.1[vector-delimiter]0.2[vector-delimiter]...
     */
    public static int findFeatureSize(final String line, final String keyValDelim, final String vectorDelim) {
        Row row = parseLine(line, keyValDelim, vectorDelim);
        return row.vector.length;
    }

    /**
     * Parses the given line using the provided key-value and vector delimiters and returns
     * (key, float[]) pair.
     */
    public static Row parseLine(final String line, final String keyValDelim, final String vectorDelim) {
        String[] keyValue = line.split(keyValDelim);
        if (keyValue.length != 2) {
            throw new IllegalArgumentException("Malformed key-value pair in line: " + line);
        }

        String[] vectorLiteral = keyValue[1].split(vectorDelim);
        if (vectorLiteral.length < 1) {
            throw new IllegalArgumentException("Malformed vector in line: " + line);
        }

        float[] vector = new float[vectorLiteral.length];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = Float.parseFloat(vectorLiteral[i]);
        }

        return new Row(keyValue[0], vector);
    }

    /**
     * A pair holding the key and the vector (float[]).
     */
    @RequiredArgsConstructor
    public static class Row {
        //CHECKSTYLE:OFF
        public final String key;
        public final float[] vector;
        //CHECKSTYLE:ON`
    }
}
