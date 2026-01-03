/*
 * Copyright (C) 2018 The Android Open Source Project
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

package com.android.apksig.internal.apk;

import java.nio.ByteBuffer;

/**
 * APK Signature Scheme block and additional information relevant to verifying the signatures
 * contained in the block against the file.
 */
public class SignatureInfo {
    /** Contents of APK Signature Scheme block. */
    public final ByteBuffer signatureBlock;

    /** Position of the APK Signing Block in the file. */
    public final long apkSigningBlockOffset;

    /** Position of the ZIP Central Directory in the file. */
    public final long centralDirOffset;

    /** Position of the ZIP End of Central Directory (EoCD) in the file. */
    public final long eocdOffset;

    /** Contents of ZIP End of Central Directory (EoCD) of the file. */
    public final ByteBuffer eocd;

    public SignatureInfo(
            ByteBuffer signatureBlock,
            long apkSigningBlockOffset,
            long centralDirOffset,
            long eocdOffset,
            ByteBuffer eocd) {
        this.signatureBlock = signatureBlock;
        this.apkSigningBlockOffset = apkSigningBlockOffset;
        this.centralDirOffset = centralDirOffset;
        this.eocdOffset = eocdOffset;
        this.eocd = eocd;
    }
}
