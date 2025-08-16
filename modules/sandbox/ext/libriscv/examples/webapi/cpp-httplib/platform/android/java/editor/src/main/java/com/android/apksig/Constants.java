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

package com.android.apksig;

import com.android.apksig.internal.apk.stamp.SourceStampConstants;
import com.android.apksig.internal.apk.v1.V1SchemeConstants;
import com.android.apksig.internal.apk.v2.V2SchemeConstants;
import com.android.apksig.internal.apk.v3.V3SchemeConstants;

/**
 * Exports internally defined constants to allow clients to reference these values without relying
 * on internal code.
 */
public class Constants {
    private Constants() {}

    public static final int VERSION_SOURCE_STAMP = 0;
    public static final int VERSION_JAR_SIGNATURE_SCHEME = 1;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V2 = 2;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V3 = 3;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V31 = 31;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V4 = 4;

    /**
     * The maximum number of signers supported by the v1 and v2 APK Signature Schemes.
     */
    public static final int MAX_APK_SIGNERS = 10;

    /**
     * The default page alignment for native library files in bytes.
     */
    public static final short LIBRARY_PAGE_ALIGNMENT_BYTES = 16384;

    public static final String MANIFEST_ENTRY_NAME = V1SchemeConstants.MANIFEST_ENTRY_NAME;

    public static final int APK_SIGNATURE_SCHEME_V2_BLOCK_ID =
            V2SchemeConstants.APK_SIGNATURE_SCHEME_V2_BLOCK_ID;

    public static final int APK_SIGNATURE_SCHEME_V3_BLOCK_ID =
            V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
    public static final int APK_SIGNATURE_SCHEME_V31_BLOCK_ID =
            V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID;
    public static final int PROOF_OF_ROTATION_ATTR_ID = V3SchemeConstants.PROOF_OF_ROTATION_ATTR_ID;

    public static final int V1_SOURCE_STAMP_BLOCK_ID =
            SourceStampConstants.V1_SOURCE_STAMP_BLOCK_ID;
    public static final int V2_SOURCE_STAMP_BLOCK_ID =
            SourceStampConstants.V2_SOURCE_STAMP_BLOCK_ID;

    public static final String OID_RSA_ENCRYPTION = "1.2.840.113549.1.1.1";
}
