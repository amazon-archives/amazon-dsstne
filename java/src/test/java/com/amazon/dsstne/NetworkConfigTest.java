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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * @author kiuk
 */
public class NetworkConfigTest {

    @Test
    public void testGetNetworkName() {
        NetworkConfig config =
            NetworkConfig.with().networkFilePath("/tmp/test-model.nc").networkName("my-model").build();
        assertEquals("my-model", config.getNetworkName());
    }

    @Test
    public void testDefaultGetNetworkName() {
        NetworkConfig config = NetworkConfig.with().networkFilePath("/tmp/test-model.nc").build();
        assertEquals("test-model", config.getNetworkName());
    }

}