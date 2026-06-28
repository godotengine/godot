/*
 * Copyright (C) 2016 The Android Open Source Project
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

import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.util.Pair;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.PSSParameterSpec;

/**
 * APK Signing Block signature algorithm.
 */
public enum SignatureAlgorithm {
    // TODO reserve the 0x0000 ID to mean null
    /**
     * RSASSA-PSS with SHA2-256 digest, SHA2-256 MGF1, 32 bytes of salt, trailer: 0xbc, content
     * digested using SHA2-256 in 1 MB chunks.
     */
    RSA_PSS_WITH_SHA256(
            0x0101,
            ContentDigestAlgorithm.CHUNKED_SHA256,
            "RSA",
            Pair.of("SHA256withRSA/PSS",
                    new PSSParameterSpec(
                            "SHA-256", "MGF1", MGF1ParameterSpec.SHA256, 256 / 8, 1)),
            AndroidSdkVersion.N,
            AndroidSdkVersion.M),

    /**
     * RSASSA-PSS with SHA2-512 digest, SHA2-512 MGF1, 64 bytes of salt, trailer: 0xbc, content
     * digested using SHA2-512 in 1 MB chunks.
     */
    RSA_PSS_WITH_SHA512(
            0x0102,
            ContentDigestAlgorithm.CHUNKED_SHA512,
            "RSA",
            Pair.of(
                    "SHA512withRSA/PSS",
                    new PSSParameterSpec(
                            "SHA-512", "MGF1", MGF1ParameterSpec.SHA512, 512 / 8, 1)),
            AndroidSdkVersion.N,
            AndroidSdkVersion.M),

    /** RSASSA-PKCS1-v1_5 with SHA2-256 digest, content digested using SHA2-256 in 1 MB chunks. */
    RSA_PKCS1_V1_5_WITH_SHA256(
            0x0103,
            ContentDigestAlgorithm.CHUNKED_SHA256,
            "RSA",
            Pair.of("SHA256withRSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.INITIAL_RELEASE),

    /** RSASSA-PKCS1-v1_5 with SHA2-512 digest, content digested using SHA2-512 in 1 MB chunks. */
    RSA_PKCS1_V1_5_WITH_SHA512(
            0x0104,
            ContentDigestAlgorithm.CHUNKED_SHA512,
            "RSA",
            Pair.of("SHA512withRSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.INITIAL_RELEASE),

    /** ECDSA with SHA2-256 digest, content digested using SHA2-256 in 1 MB chunks. */
    ECDSA_WITH_SHA256(
            0x0201,
            ContentDigestAlgorithm.CHUNKED_SHA256,
            "EC",
            Pair.of("SHA256withECDSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.HONEYCOMB),

    /** ECDSA with SHA2-512 digest, content digested using SHA2-512 in 1 MB chunks. */
    ECDSA_WITH_SHA512(
            0x0202,
            ContentDigestAlgorithm.CHUNKED_SHA512,
            "EC",
            Pair.of("SHA512withECDSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.HONEYCOMB),

    /** DSA with SHA2-256 digest, content digested using SHA2-256 in 1 MB chunks. */
    DSA_WITH_SHA256(
            0x0301,
            ContentDigestAlgorithm.CHUNKED_SHA256,
            "DSA",
            Pair.of("SHA256withDSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.INITIAL_RELEASE),

    /**
     * DSA with SHA2-256 digest, content digested using SHA2-256 in 1 MB chunks. Signing is done
     * deterministically according to RFC 6979.
     */
    DETDSA_WITH_SHA256(
            0x0301,
            ContentDigestAlgorithm.CHUNKED_SHA256,
            "DSA",
            Pair.of("SHA256withDetDSA", null),
            AndroidSdkVersion.N,
            AndroidSdkVersion.INITIAL_RELEASE),

    /**
     * RSASSA-PKCS1-v1_5 with SHA2-256 digest, content digested using SHA2-256 in 4 KB chunks, in
     * the same way fsverity operates. This digest and the content length (before digestion, 8 bytes
     * in little endian) construct the final digest.
     */
    VERITY_RSA_PKCS1_V1_5_WITH_SHA256(
            0x0421,
            ContentDigestAlgorithm.VERITY_CHUNKED_SHA256,
            "RSA",
            Pair.of("SHA256withRSA", null),
            AndroidSdkVersion.P,
            AndroidSdkVersion.INITIAL_RELEASE),

    /**
     * ECDSA with SHA2-256 digest, content digested using SHA2-256 in 4 KB chunks, in the same way
     * fsverity operates. This digest and the content length (before digestion, 8 bytes in little
     * endian) construct the final digest.
     */
    VERITY_ECDSA_WITH_SHA256(
            0x0423,
            ContentDigestAlgorithm.VERITY_CHUNKED_SHA256,
            "EC",
            Pair.of("SHA256withECDSA", null),
            AndroidSdkVersion.P,
            AndroidSdkVersion.HONEYCOMB),

    /**
     * DSA with SHA2-256 digest, content digested using SHA2-256 in 4 KB chunks, in the same way
     * fsverity operates. This digest and the content length (before digestion, 8 bytes in little
     * endian) construct the final digest.
     */
    VERITY_DSA_WITH_SHA256(
            0x0425,
            ContentDigestAlgorithm.VERITY_CHUNKED_SHA256,
            "DSA",
            Pair.of("SHA256withDSA", null),
            AndroidSdkVersion.P,
            AndroidSdkVersion.INITIAL_RELEASE);

    private final int mId;
    private final String mJcaKeyAlgorithm;
    private final ContentDigestAlgorithm mContentDigestAlgorithm;
    private final Pair<String, ? extends AlgorithmParameterSpec> mJcaSignatureAlgAndParams;
    private final int mMinSdkVersion;
    private final int mJcaSigAlgMinSdkVersion;

    SignatureAlgorithm(int id,
            ContentDigestAlgorithm contentDigestAlgorithm,
            String jcaKeyAlgorithm,
            Pair<String, ? extends AlgorithmParameterSpec> jcaSignatureAlgAndParams,
            int minSdkVersion,
            int jcaSigAlgMinSdkVersion) {
        mId = id;
        mContentDigestAlgorithm = contentDigestAlgorithm;
        mJcaKeyAlgorithm = jcaKeyAlgorithm;
        mJcaSignatureAlgAndParams = jcaSignatureAlgAndParams;
        mMinSdkVersion = minSdkVersion;
        mJcaSigAlgMinSdkVersion = jcaSigAlgMinSdkVersion;
    }

    /**
     * Returns the ID of this signature algorithm as used in APK Signature Scheme v2 wire format.
     */
    public int getId() {
        return mId;
    }

    /**
     * Returns the content digest algorithm associated with this signature algorithm.
     */
    public ContentDigestAlgorithm getContentDigestAlgorithm() {
        return mContentDigestAlgorithm;
    }

    /**
     * Returns the JCA {@link java.security.Key} algorithm used by this signature scheme.
     */
    public String getJcaKeyAlgorithm() {
        return mJcaKeyAlgorithm;
    }

    /**
     * Returns the {@link java.security.Signature} algorithm and the {@link AlgorithmParameterSpec}
     * (or null if not needed) to parameterize the {@code Signature}.
     */
    public Pair<String, ? extends AlgorithmParameterSpec> getJcaSignatureAlgorithmAndParams() {
        return mJcaSignatureAlgAndParams;
    }

    public int getMinSdkVersion() {
        return mMinSdkVersion;
    }

    /**
     * Returns the minimum SDK version that supports the JCA signature algorithm.
     */
    public int getJcaSigAlgMinSdkVersion() {
        return mJcaSigAlgMinSdkVersion;
    }

    public static SignatureAlgorithm findById(int id) {
        for (SignatureAlgorithm alg : SignatureAlgorithm.values()) {
            if (alg.getId() == id) {
                return alg;
            }
        }

        return null;
    }
}
