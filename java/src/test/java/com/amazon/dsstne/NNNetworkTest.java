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

import org.junit.Ignore;
import org.junit.Test;

import com.amazon.dsstne.NNDataSetEnums.DataType;
import com.amazon.dsstne.data.DenseNNDataSet;
import com.amazon.dsstne.data.OutputNNDataSet;

public class NNNetworkTest {

    @Ignore
    @Test
    public void test() throws IOException {
        int k = 4;
        int batchSize = 32;

        NetworkConfig config =
            NetworkConfig.with().networkFilePath("/home/kiuk/tmp/gl.nc").batchSize(batchSize).k(k).build();

        System.out.println("Loading " + config);

        NNNetwork network = Dsstne.load(config);

        System.out.println("Loaded network: \n" + network);

        NNDataSet[] inputDatasets = new NNDataSet[network.getInputLayers().length];
        for(int i=0; i<network.getInputLayers().length; ++i) {
            NNLayer inputLayer = network.getInputLayers()[i];
            String layerName = inputLayer.getName();
            String datasetName = inputLayer.getDatasetName();
            inputDatasets[i] = new DenseNNDataSet(new Dim(inputLayer.getDim(), batchSize), DataType.Int);
            inputDatasets[i].setName(datasetName);
            inputDatasets[i].setLayerName(layerName);
        }

        network.load(inputDatasets);
        OutputNNDataSet[] outputDatasets = network.predict(inputDatasets);
        OutputNNDataSet output = outputDatasets[0];
        float[] scores = output.getScores();
        long[] indexes = output.getIndexes();

        int K = (k == -1) ? 10 : k;
        for(int i=0; i<batchSize; ++i) {
            for(int j=0; j< K; ++j) {
                System.out.print(String.format("%d:%5.4f,", indexes[i * K + j], scores[i * K + j]));
            }
            System.out.println();
        }
//        for (int i = 0; i < outputDatasets.length; ++i) {
//            outputDatasets[i].getIndexes()
//        }
        network.close();
    }
}
