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

package com.android.apksig.internal.x509;

import com.android.apksig.internal.asn1.Asn1Class;
import com.android.apksig.internal.asn1.Asn1Field;
import com.android.apksig.internal.asn1.Asn1Type;
import com.android.apksig.internal.asn1.Asn1Tagging;
import com.android.apksig.internal.pkcs7.AlgorithmIdentifier;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.List;

/**
 * To Be Signed Certificate as specified in RFC 5280.
 */
@Asn1Class(type = Asn1Type.SEQUENCE)
public class TBSCertificate {

    @Asn1Field(
            index = 0,
            type = Asn1Type.INTEGER,
            tagging = Asn1Tagging.EXPLICIT, tagNumber = 0)
    public int version;

    @Asn1Field(index = 1, type = Asn1Type.INTEGER)
    public BigInteger serialNumber;

    @Asn1Field(index = 2, type = Asn1Type.SEQUENCE)
    public AlgorithmIdentifier signatureAlgorithm;

    @Asn1Field(index = 3, type = Asn1Type.CHOICE)
    public Name issuer;

    @Asn1Field(index = 4, type = Asn1Type.SEQUENCE)
    public Validity validity;

    @Asn1Field(index = 5, type = Asn1Type.CHOICE)
    public Name subject;

    @Asn1Field(index = 6, type = Asn1Type.SEQUENCE)
    public SubjectPublicKeyInfo subjectPublicKeyInfo;

    @Asn1Field(index = 7,
            type = Asn1Type.BIT_STRING,
            tagging = Asn1Tagging.IMPLICIT,
            optional = true,
            tagNumber = 1)
    public ByteBuffer issuerUniqueID;

    @Asn1Field(index = 8,
            type = Asn1Type.BIT_STRING,
            tagging = Asn1Tagging.IMPLICIT,
            optional = true,
            tagNumber = 2)
    public ByteBuffer subjectUniqueID;

    @Asn1Field(index = 9,
            type = Asn1Type.SEQUENCE_OF,
            tagging = Asn1Tagging.EXPLICIT,
            optional = true,
            tagNumber = 3)
    public List<Extension> extensions;
}
