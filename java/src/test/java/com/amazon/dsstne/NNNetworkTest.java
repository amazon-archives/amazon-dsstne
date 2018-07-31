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

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

import org.junit.Test;

import com.amazon.dsstne.NNDataSetEnums.DataType;
import com.amazon.dsstne.NetworkConfig.NetworkConfigBuilder;
import com.amazon.dsstne.data.DenseNNDataSet;
import com.amazon.dsstne.data.OutputNNDataSet;

public class NNNetworkTest {

    @Test
    public void test() throws IOException {
        NetworkConfigBuilder builder =
            NetworkConfig.with().networkFilePath("/home/kiuk/tmp/gl.nc").k(26752).batchSize(1);
        NetworkConfig config = builder.build();

        System.out.println("Loading " + config);

        try (NNNetwork network = Dsstne.load(config)) {
            for (NNLayer layer : network.getInputLayers()) {
                System.out.println("Found input layer: " + layer);
            }

            for (NNLayer layer : network.getOutputLayers()) {
                System.out.println("Found output layer: " + layer);
            }

            NNDataSet input = new DenseNNDataSet(Dim._1d(26752, 1), DataType.Int);

            long start = System.currentTimeMillis();
            int[] nint = new int[26752];
            Arrays.fill(nint, 1);
            System.out.println("fill: " + (System.currentTimeMillis() - start) + "ms");

            start = System.currentTimeMillis();
            IntBuffer ibuf = input.getData().asIntBuffer();
            for (int i = 0; i < input.getDim().x; i++) {
                input.getData().asIntBuffer().put(i, 1);
            }
            System.out.println("input: " + (System.currentTimeMillis() - start) + "ms");

            start = System.currentTimeMillis();
            input.getData().asIntBuffer().put(nint, 0, nint.length);
            System.out.println("input put batch: " + (System.currentTimeMillis() - start) + "ms");

            OutputNNDataSet output = new OutputNNDataSet(Dim._1d(26752, 1));
            network.predict(input, output);

            FloatBuffer buff = output.getScoresData().asFloatBuffer();
            System.out.println("=========== OUTPUT BUFFER =============");
            //      while(buff.position() < buff.limit()) {
            //        System.out.print(buff.get()+ ",");
            //      }
        }
    }
}
