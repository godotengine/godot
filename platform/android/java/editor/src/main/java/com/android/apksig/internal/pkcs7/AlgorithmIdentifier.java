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

import static com.android.apksig.Constants.OID_RSA_ENCRYPTION;
import static com.android.apksig.internal.asn1.Asn1DerEncoder.ASN1_DER_NULL;
import static com.android.apksig.internal.oid.OidConstants.OID_DIGEST_SHA1;
import static com.android.apksig.internal.oid.OidConstants.OID_DIGEST_SHA256;
import static com.android.apksig.internal.oid.OidConstants.OID_SIG_DSA;
import static com.android.apksig.internal.oid.OidConstants.OID_SIG_EC_PUBLIC_KEY;
import static com.android.apksig.internal.oid.OidConstants.OID_SIG_RSA;
import static com.android.apksig.internal.oid.OidConstants.OID_SIG_SHA256_WITH_DSA;
import static com.android.apksig.internal.oid.OidConstants.OID_TO_JCA_DIGEST_ALG;
import static com.android.apksig.internal.oid.OidConstants.OID_TO_JCA_SIGNATURE_ALG;

import com.android.apksig.internal.apk.v1.DigestAlgorithm;
import com.android.apksig.internal.asn1.Asn1Class;
import com.android.apksig.internal.asn1.Asn1Field;
import com.android.apksig.internal.asn1.Asn1OpaqueObject;
import com.android.apksig.internal.asn1.Asn1Type;
import com.android.apksig.internal.util.Pair;

import java.security.InvalidKeyException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;

/**
 * PKCS #7 {@code AlgorithmIdentifier} as specified in RFC 5652.
 */
@Asn1Class(type = Asn1Type.SEQUENCE)
public class AlgorithmIdentifier {

    @Asn1Field(index = 0, type = Asn1Type.OBJECT_IDENTIFIER)
    public String algorithm;

    @Asn1Field(index = 1, type = Asn1Type.ANY, optional = true)
    public Asn1OpaqueObject parameters;

    public AlgorithmIdentifier() {}

    public AlgorithmIdentifier(String algorithmOid, Asn1OpaqueObject parameters) {
        this.algorithm = algorithmOid;
        this.parameters = parameters;
    }

    /**
     * Returns the PKCS #7 {@code DigestAlgorithm} to use when signing using the specified digest
     * algorithm.
     */
    public static AlgorithmIdentifier getSignerInfoDigestAlgorithmOid(
            DigestAlgorithm digestAlgorithm) {
        switch (digestAlgorithm) {
            case SHA1:
                return new AlgorithmIdentifier(OID_DIGEST_SHA1, ASN1_DER_NULL);
            case SHA256:
                return new AlgorithmIdentifier(OID_DIGEST_SHA256, ASN1_DER_NULL);
        }
        throw new IllegalArgumentException("Unsupported digest algorithm: " + digestAlgorithm);
    }

    /**
     * Returns the JCA {@link Signature} algorithm and PKCS #7 {@code SignatureAlgorithm} to use
     * when signing with the specified key and digest algorithm.
     */
    public static Pair<String, AlgorithmIdentifier> getSignerInfoSignatureAlgorithm(
            PublicKey publicKey, DigestAlgorithm digestAlgorithm, boolean deterministicDsaSigning)
            throws InvalidKeyException {
        String keyAlgorithm = publicKey.getAlgorithm();
        String jcaDigestPrefixForSigAlg;
        switch (digestAlgorithm) {
            case SHA1:
                jcaDigestPrefixForSigAlg = "SHA1";
                break;
            case SHA256:
                jcaDigestPrefixForSigAlg = "SHA256";
                break;
            default:
                throw new IllegalArgumentException(
                        "Unexpected digest algorithm: " + digestAlgorithm);
        }
        if ("RSA".equalsIgnoreCase(keyAlgorithm) || OID_RSA_ENCRYPTION.equals(keyAlgorithm)) {
            return Pair.of(
                    jcaDigestPrefixForSigAlg + "withRSA",
                    new AlgorithmIdentifier(OID_SIG_RSA, ASN1_DER_NULL));
        } else if ("DSA".equalsIgnoreCase(keyAlgorithm)) {
            AlgorithmIdentifier sigAlgId;
            switch (digestAlgorithm) {
                case SHA1:
                    sigAlgId =
                            new AlgorithmIdentifier(OID_SIG_DSA, ASN1_DER_NULL);
                    break;
                case SHA256:
                    // DSA signatures with SHA-256 in SignedData are accepted by Android API Level
                    // 21 and higher. However, there are two ways to specify their SignedData
                    // SignatureAlgorithm: dsaWithSha256 (2.16.840.1.101.3.4.3.2) and
                    // dsa (1.2.840.10040.4.1). The latter works only on API Level 22+. Thus, we use
                    // the former.
                    sigAlgId =
                            new AlgorithmIdentifier(OID_SIG_SHA256_WITH_DSA, ASN1_DER_NULL);
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Unexpected digest algorithm: " + digestAlgorithm);
            }
            String signingAlgorithmName =
                    jcaDigestPrefixForSigAlg + (deterministicDsaSigning ? "withDetDSA" : "withDSA");
            return Pair.of(signingAlgorithmName, sigAlgId);
        } else if ("EC".equalsIgnoreCase(keyAlgorithm)) {
            return Pair.of(
                    jcaDigestPrefixForSigAlg + "withECDSA",
                    new AlgorithmIdentifier(OID_SIG_EC_PUBLIC_KEY, ASN1_DER_NULL));
        } else {
            throw new InvalidKeyException("Unsupported key algorithm: " + keyAlgorithm);
        }
    }

    public static String getJcaSignatureAlgorithm(
            String digestAlgorithmOid,
            String signatureAlgorithmOid) throws SignatureException {
        // First check whether the signature algorithm OID alone is sufficient
        String result = OID_TO_JCA_SIGNATURE_ALG.get(signatureAlgorithmOid);
        if (result != null) {
            return result;
        }

        // Signature algorithm OID alone is insufficient. Need to combine digest algorithm OID
        // with signature algorithm OID.
        String suffix;
        if (OID_SIG_RSA.equals(signatureAlgorithmOid)) {
            suffix = "RSA";
        } else if (OID_SIG_DSA.equals(signatureAlgorithmOid)) {
            suffix = "DSA";
        } else if (OID_SIG_EC_PUBLIC_KEY.equals(signatureAlgorithmOid)) {
            suffix = "ECDSA";
        } else {
            throw new SignatureException(
                    "Unsupported JCA Signature algorithm"
                            + " . Digest algorithm: " + digestAlgorithmOid
                            + ", signature algorithm: " + signatureAlgorithmOid);
        }
        String jcaDigestAlg = getJcaDigestAlgorithm(digestAlgorithmOid);
        // Canonical name for SHA-1 with ... is SHA1with, rather than SHA1. Same for all other
        // SHA algorithms.
        if (jcaDigestAlg.startsWith("SHA-")) {
            jcaDigestAlg = "SHA" + jcaDigestAlg.substring("SHA-".length());
        }
        return jcaDigestAlg + "with" + suffix;
    }

    public static String getJcaDigestAlgorithm(String oid)
            throws SignatureException {
        String result = OID_TO_JCA_DIGEST_ALG.get(oid);
        if (result == null) {
            throw new SignatureException("Unsupported digest algorithm: " + oid);
        }
        return result;
    }
}
