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

package com.android.apksig.internal.pkcs7;

import com.android.apksig.internal.asn1.Asn1Class;
import com.android.apksig.internal.asn1.Asn1Field;
import com.android.apksig.internal.asn1.Asn1OpaqueObject;
import com.android.apksig.internal.asn1.Asn1Type;
import com.android.apksig.internal.asn1.Asn1Tagging;
import java.nio.ByteBuffer;
import java.util.List;

/**
 * PKCS #7 {@code SignedData} as specified in RFC 5652.
 */
@Asn1Class(type = Asn1Type.SEQUENCE)
public class SignedData {

    @Asn1Field(index = 0, type = Asn1Type.INTEGER)
    public int version;

    @Asn1Field(index = 1, type = Asn1Type.SET_OF)
    public List<AlgorithmIdentifier> digestAlgorithms;

    @Asn1Field(index = 2, type = Asn1Type.SEQUENCE)
    public EncapsulatedContentInfo encapContentInfo;

    @Asn1Field(
            index = 3,
            type = Asn1Type.SET_OF,
            tagging = Asn1Tagging.IMPLICIT, tagNumber = 0,
            optional = true)
    public List<Asn1OpaqueObject> certificates;

    @Asn1Field(
            index = 4,
            type = Asn1Type.SET_OF,
            tagging = Asn1Tagging.IMPLICIT, tagNumber = 1,
            optional = true)
    public List<ByteBuffer> crls;

    @Asn1Field(index = 5, type = Asn1Type.SET_OF)
    public List<SignerInfo> signerInfos;
}
