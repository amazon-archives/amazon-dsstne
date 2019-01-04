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

/**
 * Data type on the GPU. Loads and operates in either fp32 (single precision) or fp16 (half precision).
 */
public enum DataType {
    /* DO NOT CHANGE THE ORDER, IT IS PASSED TO C++ VIA JNI */
    FP32, FP16;

    /**
     * Case insensitive {@link #valueOf(String)}.
     */
    public static DataType fromString(final String dt) {
        return DataType.valueOf(dt.toUpperCase());
    }
}
