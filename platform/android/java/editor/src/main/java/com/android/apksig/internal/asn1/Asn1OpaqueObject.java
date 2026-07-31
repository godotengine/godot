/*
 * Copyright (C) 2017 The Android Open Source Project
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

package com.android.apksig.internal.asn1;

import java.nio.ByteBuffer;

/**
 * Opaque holder of encoded ASN.1 stuff.
 */
public class Asn1OpaqueObject {
    private final ByteBuffer mEncoded;

    public Asn1OpaqueObject(ByteBuffer encoded) {
        mEncoded = encoded.slice();
    }

    public Asn1OpaqueObject(byte[] encoded) {
        mEncoded = ByteBuffer.wrap(encoded);
    }

    public ByteBuffer getEncoded() {
        return mEncoded.slice();
    }
}
