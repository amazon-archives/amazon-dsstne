/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef BITONIC_H


// 256 element register resident bitonic merge sort 
// To use, define:
// unsigned int otgx;
// type k0, k1, k2, k3, k4, k5, k6, k7;
// type v0, v1, v2, v3, v4, v5, v6, v7;
#define BITONICWARPEXCHANGE_256(mask) \
    key1                                = k0; \
    value1                              = v0; \
    otgx                                = tgx ^ mask; \
    key2                                = __shfl(k0, otgx); \
    value2                              = __shfl(v0, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0                                  = flag ? key1 : key2; \
    v0                                  = flag ? value1 : value2; \
    key1                                = k1; \
    value1                              = v1; \
    key2                                = __shfl(k1, otgx); \
    value2                              = __shfl(v1, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1                                  = flag ? key1 : key2; \
    v1                                  = flag ? value1 : value2; \
    key1                                = k2; \
    value1                              = v2; \
    key2                                = __shfl(k2, otgx); \
    value2                              = __shfl(v2, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2                                  = flag ? key1 : key2; \
    v2                                  = flag ? value1 : value2; \
    key1                                = k3; \
    value1                              = v3; \
    key2                                = __shfl(k3, otgx); \
    value2                              = __shfl(v3, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3                                  = flag ? key1 : key2; \
    v3                                  = flag ? value1 : value2; \
    key1                                = k4; \
    value1                              = v4; \
    key2                                = __shfl(k4, otgx); \
    value2                              = __shfl(v4, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k4                                  = flag ? key1 : key2; \
    v4                                  = flag ? value1 : value2; \
    key1                                = k5; \
    value1                              = v5; \
    key2                                = __shfl(k5, otgx); \
    value2                              = __shfl(v5, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k5                                  = flag ? key1 : key2; \
    v5                                  = flag ? value1 : value2; \
    key1                                = k6; \
    value1                              = v6; \
    key2                                = __shfl(k6, otgx); \
    value2                              = __shfl(v6, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k6                                  = flag ? key1 : key2; \
    v6                                  = flag ? value1 : value2; \
    key1                                = k7; \
    value1                              = v7; \
    key2                                = __shfl(k7, otgx); \
    value2                              = __shfl(v7, otgx); \
    flag                                = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k7                                  = flag ? key1 : key2; \
    v7                                  = flag ? value1 : value2;

#define BITONICSORT32_256() \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(3) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(7) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(15) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(31) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) 



#define BITONICMERGE64_256() \
    otgx                                = 31 - tgx; \
    key1                                = k0; \
    value1                              = v0; \
    key2                                = __shfl(k1, otgx); \
    value2                              = __shfl(v1, otgx); \
    flag                                = (key1 > key2); \
    k0                                  = flag ? key1 : key2; \
    v0                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k1                                  = __shfl(key1, otgx); \
    v1                                  = __shfl(value1, otgx); \
    key1                                = k2; \
    value1                              = v2; \
    key2                                = __shfl(k3, otgx); \
    value2                              = __shfl(v3, otgx); \
    flag                                = (key1 > key2); \
    k2                                  = flag ? key1 : key2; \
    v2                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k3                                  = __shfl(key1, otgx); \
    v3                                  = __shfl(value1, otgx); \
    key1                                = k4; \
    value1                              = v4; \
    key2                                = __shfl(k5, otgx); \
    value2                              = __shfl(v5, otgx); \
    flag                                = (key1 > key2); \
    k4                                  = flag ? key1 : key2; \
    v4                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k5                                  = __shfl(key1, otgx); \
    v5                                  = __shfl(value1, otgx); \
    key1                                = k6; \
    value1                              = v6; \
    key2                                = __shfl(k7, otgx); \
    value2                              = __shfl(v7, otgx); \
    flag                                = (key1 > key2); \
    k6                                  = flag ? key1 : key2; \
    v6                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k7                                  = __shfl(key1, otgx); \
    v7                                  = __shfl(value1, otgx);
    
#define BITONICSORT64_256() \
    BITONICSORT32_256() \
    BITONICMERGE64_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#define BITONICMERGE128_256() \
    otgx                                = 31 - tgx; \
    key1                                = k0; \
    value1                              = v0; \
    key2                                = __shfl(k3, otgx); \
    value2                              = __shfl(v3, otgx); \
    flag                                = (key1 > key2); \
    k0                                  = flag ? key1 : key2; \
    v0                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k3                                  = __shfl(key1, otgx); \
    v3                                  = __shfl(value1, otgx); \
    key1                                = k1; \
    value1                              = v1; \
    key2                                = __shfl(k2, otgx); \
    value2                              = __shfl(v2, otgx); \
    flag                                = (key1 > key2); \
    k1                                  = flag ? key1 : key2; \
    v1                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k2                                  = __shfl(key1, otgx); \
    v2                                  = __shfl(value1, otgx); \
    key1                                = k4; \
    value1                              = v4; \
    key2                                = __shfl(k7, otgx); \
    value2                              = __shfl(v7, otgx); \
    flag                                = (key1 > key2); \
    k4                                  = flag ? key1 : key2; \
    v4                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k7                                  = __shfl(key1, otgx); \
    v7                                  = __shfl(value1, otgx); \
    key1                                = k5; \
    value1                              = v5; \
    key2                                = __shfl(k6, otgx); \
    value2                              = __shfl(v6, otgx); \
    flag                                = (key1 > key2); \
    k5                                  = flag ? key1 : key2; \
    v5                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k6                                  = __shfl(key1, otgx); \
    v6                                  = __shfl(value1, otgx);

#define BITONICMERGE256_256() \
    otgx                                = 31 - tgx; \
    key1                                = k0; \
    value1                              = v0; \
    key2                                = __shfl(k7, otgx); \
    value2                              = __shfl(v7, otgx); \
    flag                                = (key1 > key2); \
    k0                                  = flag ? key1 : key2; \
    v0                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k7                                  = __shfl(key1, otgx); \
    v7                                  = __shfl(value1, otgx); \
    key1                                = k1; \
    value1                              = v1; \
    key2                                = __shfl(k6, otgx); \
    value2                              = __shfl(v6, otgx); \
    flag                                = (key1 > key2); \
    k1                                  = flag ? key1 : key2; \
    v1                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k6                                  = __shfl(key1, otgx); \
    v6                                  = __shfl(value1, otgx); \
    key1                                = k2; \
    value1                              = v2; \
    key2                                = __shfl(k5, otgx); \
    value2                              = __shfl(v5, otgx); \
    flag                                = (key1 > key2); \
    k2                                  = flag ? key1 : key2; \
    v2                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k5                                  = __shfl(key1, otgx); \
    v5                                  = __shfl(value1, otgx); \
    key1                                = k3; \
    value1                              = v3; \
    key2                                = __shfl(k4, otgx); \
    value2                              = __shfl(v4, otgx); \
    flag                                = (key1 > key2); \
    k3                                  = flag ? key1 : key2; \
    v3                                  = flag ? value1 : value2; \
    key1                                = flag ? key2 : key1; \
    value1                              = flag ? value2 : value1; \
    k4                                  = __shfl(key1, otgx); \
    v4                                  = __shfl(value1, otgx);

#define BITONICEXCHANGE32_256() \
    if (k0 < k1) \
    { \
        key1                            = k0; \
        value1                          = v0; \
        k0                              = k1; \
        v0                              = v1; \
        k1                              = key1; \
        v1                              = value1; \
    } \
    if (k2 < k3) \
    { \
        key1                            = k2; \
        value1                          = v2; \
        k2                              = k3; \
        v2                              = v3; \
        k3                              = key1; \
        v3                              = value1; \
    } \
    if (k4 < k5) \
    { \
        key1                            = k4; \
        value1                          = v4; \
        k4                              = k5; \
        v4                              = v5; \
        k5                              = key1; \
        v5                              = value1; \
    } \
    if (k6 < k7) \
    { \
        key1                            = k6; \
        value1                          = v6; \
        k6                              = k7; \
        v6                              = v7; \
        k7                              = key1; \
        v7                              = value1; \
    }

#define BITONICEXCHANGE64_256() \
    if (k0 < k2) \
    { \
        key1                            = k0; \
        value1                          = v0; \
        k0                              = k2; \
        v0                              = v2; \
        k2                              = key1; \
        v2                              = value1; \
    } \
    if (k1 < k3) \
    { \
        key1                            = k1; \
        value1                          = v1; \
        k1                              = k3; \
        v1                              = v3; \
        k3                              = key1; \
        v3                              = value1; \
    } \
    if (k4 < k6) \
    { \
        key1                            = k4; \
        value1                          = v4; \
        k4                              = k6; \
        v4                              = v6; \
        k6                              = key1; \
        v6                              = value1; \
    } \
    if (k5 < k7) \
    { \
        key1                            = k5; \
        value1                          = v5; \
        k5                              = k7; \
        v5                              = v7; \
        k7                              = key1; \
        v7                              = value1; \
    }


#define BITONICSORT128_256() \
    BITONICSORT64_256() \
    BITONICMERGE128_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#define BITONICSORT256_256() \
    BITONICSORT128_256() \
    BITONICMERGE256_256() \
    BITONICEXCHANGE64_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#endif // BITONIC_H
