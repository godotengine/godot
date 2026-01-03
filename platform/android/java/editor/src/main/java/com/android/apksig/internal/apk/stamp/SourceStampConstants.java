/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.apksig.internal.apk.stamp;

/** Constants used for source stamp signing and verification. */
public class SourceStampConstants {
    private SourceStampConstants() {}

    public static final int V1_SOURCE_STAMP_BLOCK_ID = 0x2b09189e;
    public static final int V2_SOURCE_STAMP_BLOCK_ID = 0x6dff800d;
    public static final String SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME = "stamp-cert-sha256";
    public static final int PROOF_OF_ROTATION_ATTR_ID = 0x9d6303f7;
    /**
     * The source stamp timestamp attribute value is an 8-byte little-endian encoded long
     * representing the epoch time in seconds when the stamp block was signed. The first 8 bytes
     * of the attribute value buffer will be used to read the timestamp, and any additional buffer
     * space will be ignored.
     */
    public static final int STAMP_TIME_ATTR_ID = 0xe43c5946;
}
