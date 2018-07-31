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

import java.io.Closeable;
import java.util.List;

import com.amazon.dsstne.data.OutputNNDataSet;

import lombok.Getter;

/**
 * The neural network.
 */
public class NNNetwork implements Closeable {
    @Getter
    private final NetworkConfig config;
    @Getter
    private final NNLayer[] inputLayers;
    @Getter
    private final NNLayer[] outputLayers;

    private volatile long ptr;

    public NNNetwork(final NetworkConfig config, final long ptr, final List<NNLayer> inputLayers,
        final List<NNLayer> outputLayers) {
        if (outputLayers.size() > 1) {
            /* TODO support more than one output layer
             * (need to overload predict method and map output configs (e.g. k) to each output layer)
             */
            throw new RuntimeException("Only one output layer is supported at the moment. Got " + outputLayers.size());
        }

        this.config = config;
        this.ptr = ptr;
        this.inputLayers = inputLayers.toArray(new NNLayer[0]);
        this.outputLayers = outputLayers.toArray(new NNLayer[0]);
    }

    public void load(NNDataSet[] datasets) {

    }

    public void predict(final NNDataSet input, final OutputNNDataSet output) {
        if (inputLayers.length != 1 || outputLayers.length != 1) {
            throw new UnsupportedOperationException(
                "method can only valid with networks with single input/output layer");
        }

        predict(new NNDataSet[] {input}, new OutputNNDataSet[] {output});
    }

    public OutputNNDataSet[] predict(final NNDataSet[] inputs) {
        OutputNNDataSet[] outputs = new OutputNNDataSet[outputLayers.length];
        for (int i = 0; i < outputLayers.length; i++) {
            NNLayer outputLayer = outputLayers[i];
            outputs[i] = new OutputNNDataSet(
                new Dim(outputLayer.getDimensions(), outputLayer.getDimX(), outputLayer.getDimY(),
                    outputLayer.getDimZ(),
                    config.getBatchSize()));
        }
        predict(inputs, outputs);
        return outputs;
    }

    public void predict(final NNDataSet[] inputs, final OutputNNDataSet[] outputs) {
        checkArguments(inputs, outputs);
        Dsstne.predict(ptr, config.getK(), inputs, outputs);
    }

    /**
     * Checks that the number of data matches the number of layers and that the dimensions match.
     * Data and layers are matched in index order.
     */
    private void checkArguments(final NNDataSet[] inputs, final OutputNNDataSet[] outputs) {
        if (inputs.length != inputLayers.length) {
            throw new IllegalArgumentException("Number of input data and input layers do not match");
        }

        for (int i = 0; i < inputs.length; i++) {
            Dim dataDim = inputs[i].getDim();
            NNLayer inputLayer = inputLayers[i];

            if (dataDim.dimensions != inputLayer.getDimensions()) {
                throw new IllegalArgumentException(
                    "Num dimension mismatch between layer " + inputLayer.getName() + " and data " + i);
            }

            if (dataDim.x != inputLayer.getDimX()
                || dataDim.y != inputLayer.getDimY()
                || dataDim.z != inputLayer.getDimZ()) {
                throw new IllegalArgumentException(
                    "Dimension mismatch between input layer " + inputLayer.getName() + " and input data " + i);
            }

            if (dataDim.examples != config.getBatchSize()) {
                throw new IllegalArgumentException(
                    "Examples in input data " + i + " does not match the batch size of the network");
            }
        }

        // check output
        if (outputs.length != outputLayers.length) {
            throw new IllegalArgumentException("Number of output data and output layers do not match");
        }

        for (int i = 0; i < outputs.length; i++) {
            NNLayer outputLayer = outputLayers[i];
            Dim dataDim = outputs[i].getDim();

            // FIXME what does it mean to do top-K on dimensions 2 and 3?
            if (dataDim.x != config.getK()
                || dataDim.y != outputLayer.getDimY()
                || dataDim.z != outputLayer.getDimZ()) {
                throw new IllegalArgumentException(
                    "Dimension mismatch between output layer " + outputLayer.getName() + " and output data " + i);
            }

            if (dataDim.examples != config.getBatchSize()) {
                throw new IllegalArgumentException(
                    "Examples in output data " + i + " does not match the batch size of the network");
            }
        }
    }

    @Override
    public void close() {
        Dsstne.shutdown(this.ptr);
    }
}

