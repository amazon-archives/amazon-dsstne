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

import java.util.List;

import com.amazon.dsstne.NNLayer.Kind;
import com.amazon.dsstne.data.OutputNNDataSet;

/**
 * Entry point to obtaining {@link NNNetwork}. Defines all the native methods.
 */
public class Dsstne {
    public static final long NULLPTR = 0x0;

    static {
        System.loadLibrary("dsstneJava");
    }

    public static NNNetwork load(final NetworkConfig config) {
        long ptr = load(config.getNetworkFilePath(), config.getBatchSize());
        if (ptr == NULLPTR) {
            throw new RuntimeException("Failed to load network from config: " + config);
        }

        List<NNLayer> inputLayers = get_layers(ptr, Kind.Input.ordinal());
        List<NNLayer> outputLayers = get_layers(ptr, Kind.Output.ordinal());

        if (inputLayers.isEmpty()) {
            throw new RuntimeException("No input layers found in: " + config);
        }
        if (outputLayers.isEmpty()) {
            throw new RuntimeException("No output layers found in: " + config);
        }

        return new NNNetwork(config, ptr, inputLayers, outputLayers);
    }

    /**
     * Loads the network from the netcdf file. Returns a pointer to a context data structure
     * that is used to access the network.
     */
    private static native long load(final String networkFilePath, final int batchSize);

    public static native void loadDatasets(final long ptr, NNDataSet[] datasets);

    /**
     * Shuts down this model and GPU context and releases all resources. Once shutdown, the init method
     * must be called again to start up the context.
     */
    static native void shutdown(final long ptr);

    /**
     * Returns the metadata (e.g. dimensions, name, etc) of the layer (one entry per layer)
     * of the specified {@link NNLayer.Kind}.
     */
    private static native List<NNLayer> get_layers(final long ptr, final int kind);

    static native void predict(final long ptr, final int k, final NNDataSet[] inputs, final OutputNNDataSet[] outputs);
}
