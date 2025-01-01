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
import com.android.apksig.internal.asn1.Asn1OpaqueObject;
import com.android.apksig.internal.asn1.Asn1Type;
import com.android.apksig.internal.pkcs7.AlgorithmIdentifier;
import com.android.apksig.internal.pkcs7.IssuerAndSerialNumber;
import com.android.apksig.internal.pkcs7.SignerIdentifier;
import com.android.apksig.internal.util.ByteBufferUtils;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import com.android.apksig.internal.util.X509CertificateUtils;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import javax.security.auth.x500.X500Principal;

/**
 * X509 {@code Certificate} as specified in RFC 5280.
 */
@Asn1Class(type = Asn1Type.SEQUENCE)
public class Certificate {
    @Asn1Field(index = 0, type = Asn1Type.SEQUENCE)
    public TBSCertificate certificate;

    @Asn1Field(index = 1, type = Asn1Type.SEQUENCE)
    public AlgorithmIdentifier signatureAlgorithm;

    @Asn1Field(index = 2, type = Asn1Type.BIT_STRING)
    public ByteBuffer signature;

    public static X509Certificate findCertificate(
            Collection<X509Certificate> certs, SignerIdentifier id) {
        for (X509Certificate cert : certs) {
            if (isMatchingCerticicate(cert, id)) {
                return cert;
            }
        }
        return null;
    }

    private static boolean isMatchingCerticicate(X509Certificate cert, SignerIdentifier id) {
        if (id.issuerAndSerialNumber == null) {
            // Android doesn't support any other means of identifying the signing certificate
            return false;
        }
        IssuerAndSerialNumber issuerAndSerialNumber = id.issuerAndSerialNumber;
        byte[] encodedIssuer =
                ByteBufferUtils.toByteArray(issuerAndSerialNumber.issuer.getEncoded());
        X500Principal idIssuer = new X500Principal(encodedIssuer);
        BigInteger idSerialNumber = issuerAndSerialNumber.certificateSerialNumber;
        return idSerialNumber.equals(cert.getSerialNumber())
                && idIssuer.equals(cert.getIssuerX500Principal());
    }

    public static List<X509Certificate> parseCertificates(
            List<Asn1OpaqueObject> encodedCertificates) throws CertificateException {
        if (encodedCertificates.isEmpty()) {
            return Collections.emptyList();
        }

        List<X509Certificate> result = new ArrayList<>(encodedCertificates.size());
        for (int i = 0; i < encodedCertificates.size(); i++) {
            Asn1OpaqueObject encodedCertificate = encodedCertificates.get(i);
            X509Certificate certificate;
            byte[] encodedForm = ByteBufferUtils.toByteArray(encodedCertificate.getEncoded());
            try {
                certificate = X509CertificateUtils.generateCertificate(encodedForm);
            } catch (CertificateException e) {
                throw new CertificateException("Failed to parse certificate #" + (i + 1), e);
            }
            // Wrap the cert so that the result's getEncoded returns exactly the original
            // encoded form. Without this, getEncoded may return a different form from what was
            // stored in the signature. This is because some X509Certificate(Factory)
            // implementations re-encode certificates and/or some implementations of
            // X509Certificate.getEncoded() re-encode certificates.
            certificate = new GuaranteedEncodedFormX509Certificate(certificate, encodedForm);
            result.add(certificate);
        }
        return result;
    }
}
