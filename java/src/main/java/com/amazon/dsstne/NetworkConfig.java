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

import java.nio.file.Paths;
import java.util.Map;

import lombok.Builder;
import lombok.Data;
import lombok.Singular;

/**
 * Configurations to {@link NNNetwork}.
 *
 * @author kiuk
 */
@Data
@Builder(builderMethodName = "with")
public class NetworkConfig {
    private static final char EXTENSION_SEPARATOR = '.';

    /**
     * Location of the network netcdf file.
     */
    private String networkFilePath;

    /**
     * Name of the dataset (model) in the model netcdf file.
     * Default: name of the network netcdf file (not including the .nc suffix)
     */
    private String networkName;

    /**
     * Input batch size for the prediction (feed-forward)
     * Default: 32
     */
    @Builder.Default
    private int batchSize = 32;

    /**
     * Number of predictions to generate per input.
     * Default: 100
     */
    @Builder.Default
    private int k = 100;

    /**
     * Specifications of input data mapped to each input layer by layer name.
     */
    @Singular
    private Map<String, NNDataSet> inputDataSets;

    @Singular
    private Map<String, NNDataSet> outputDataSets;

    public String getNetworkName() {
        if (this.networkName == null || this.networkName.isEmpty()) {
            // default to the name of the network file
            String fileName = Paths.get(networkFilePath).getFileName().toString();
            int index = fileName.lastIndexOf(EXTENSION_SEPARATOR);
            return fileName.substring(0, index);
        } else {
            return this.networkName;
        }
    }
}
