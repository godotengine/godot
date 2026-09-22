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

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_JAR_SIGNATURE_SCHEME;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsLengthPrefixedElement;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsSequenceOfLengthPrefixedElements;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes;

import com.android.apksig.SigningCertificateLineage;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils.SignerConfig;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.util.Pair;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SourceStamp signer.
 *
 * <p>SourceStamp improves traceability of apps with respect to unauthorized distribution.
 *
 * <p>The stamp is part of the APK that is protected by the signing block.
 *
 * <p>The APK contents hash is signed using the stamp key, and is saved as part of the signing
 * block.
 *
 * <p>V2 of the source stamp allows signing the digests of more than one signature schemes.
 */
public class V2SourceStampSigner {
    public static final int V2_SOURCE_STAMP_BLOCK_ID =
            SourceStampConstants.V2_SOURCE_STAMP_BLOCK_ID;

    private final SignerConfig mSourceStampSignerConfig;
    private final Map<Integer, Map<ContentDigestAlgorithm, byte[]>> mSignatureSchemeDigestInfos;
    private final boolean mSourceStampTimestampEnabled;

    /** Hidden constructor to prevent instantiation. */
    private V2SourceStampSigner(Builder builder) {
        mSourceStampSignerConfig = builder.mSourceStampSignerConfig;
        mSignatureSchemeDigestInfos = builder.mSignatureSchemeDigestInfos;
        mSourceStampTimestampEnabled = builder.mSourceStampTimestampEnabled;
    }

    public static Pair<byte[], Integer> generateSourceStampBlock(
            SignerConfig sourceStampSignerConfig,
            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> signatureSchemeDigestInfos)
            throws SignatureException, NoSuchAlgorithmException, InvalidKeyException {
        return new Builder(sourceStampSignerConfig,
                signatureSchemeDigestInfos).build().generateSourceStampBlock();
    }

    public Pair<byte[], Integer> generateSourceStampBlock()
            throws SignatureException, NoSuchAlgorithmException, InvalidKeyException {
        if (mSourceStampSignerConfig.certificates.isEmpty()) {
            throw new SignatureException("No certificates configured for signer");
        }

        // Extract the digests for signature schemes.
        List<Pair<Integer, byte[]>> signatureSchemeDigests = new ArrayList<>();
        getSignedDigestsFor(
                VERSION_APK_SIGNATURE_SCHEME_V3,
                mSignatureSchemeDigestInfos,
                mSourceStampSignerConfig,
                signatureSchemeDigests);
        getSignedDigestsFor(
                VERSION_APK_SIGNATURE_SCHEME_V2,
                mSignatureSchemeDigestInfos,
                mSourceStampSignerConfig,
                signatureSchemeDigests);
        getSignedDigestsFor(
                VERSION_JAR_SIGNATURE_SCHEME,
                mSignatureSchemeDigestInfos,
                mSourceStampSignerConfig,
                signatureSchemeDigests);
        Collections.sort(signatureSchemeDigests, Comparator.comparing(Pair::getFirst));

        SourceStampBlock sourceStampBlock = new SourceStampBlock();

        try {
            sourceStampBlock.stampCertificate =
                    mSourceStampSignerConfig.certificates.get(0).getEncoded();
        } catch (CertificateEncodingException e) {
            throw new SignatureException(
                    "Retrieving the encoded form of the stamp certificate failed", e);
        }

        sourceStampBlock.signedDigests = signatureSchemeDigests;

        sourceStampBlock.stampAttributes = encodeStampAttributes(
                generateStampAttributes(mSourceStampSignerConfig.signingCertificateLineage));
        sourceStampBlock.signedStampAttributes =
                ApkSigningBlockUtils.generateSignaturesOverData(mSourceStampSignerConfig,
                        sourceStampBlock.stampAttributes);

        // FORMAT:
        // * length-prefixed bytes: X.509 certificate (ASN.1 DER encoded)
        // * length-prefixed sequence of length-prefixed signed signature scheme digests:
        //   * uint32: signature scheme id
        //   * length-prefixed bytes: signed digests for the respective signature scheme
        // * length-prefixed bytes: encoded stamp attributes
        // * length-prefixed sequence of length-prefixed signed stamp attributes:
        //   * uint32: signature algorithm id
        //   * length-prefixed bytes: signed stamp attributes for the respective signature algorithm
        byte[] sourceStampSignerBlock =
                encodeAsSequenceOfLengthPrefixedElements(
                        new byte[][]{
                                sourceStampBlock.stampCertificate,
                                encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
                                        sourceStampBlock.signedDigests),
                                sourceStampBlock.stampAttributes,
                                encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
                                        sourceStampBlock.signedStampAttributes),
                        });

        // FORMAT:
        // * length-prefixed stamp block.
        return Pair.of(encodeAsLengthPrefixedElement(sourceStampSignerBlock),
                SourceStampConstants.V2_SOURCE_STAMP_BLOCK_ID);
    }

    private static void getSignedDigestsFor(
            int signatureSchemeVersion,
            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> mSignatureSchemeDigestInfos,
            SignerConfig mSourceStampSignerConfig,
            List<Pair<Integer, byte[]>> signatureSchemeDigests)
            throws NoSuchAlgorithmException, InvalidKeyException, SignatureException {
        if (!mSignatureSchemeDigestInfos.containsKey(signatureSchemeVersion)) {
            return;
        }

        Map<ContentDigestAlgorithm, byte[]> digestInfo =
                mSignatureSchemeDigestInfos.get(signatureSchemeVersion);
        List<Pair<Integer, byte[]>> digests = new ArrayList<>();
        for (Map.Entry<ContentDigestAlgorithm, byte[]> digest : digestInfo.entrySet()) {
            digests.add(Pair.of(digest.getKey().getId(), digest.getValue()));
        }
        Collections.sort(digests, Comparator.comparing(Pair::getFirst));

        // FORMAT:
        // * length-prefixed sequence of length-prefixed digests:
        //   * uint32: digest algorithm id
        //   * length-prefixed bytes: digest of the respective digest algorithm
        byte[] digestBytes =
                encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(digests);

        // FORMAT:
        // * length-prefixed sequence of length-prefixed signed digests:
        //   * uint32: signature algorithm id
        //   * length-prefixed bytes: signed digest for the respective signature algorithm
        List<Pair<Integer, byte[]>> signedDigest =
                ApkSigningBlockUtils.generateSignaturesOverData(
                        mSourceStampSignerConfig, digestBytes);

        // FORMAT:
        // * length-prefixed sequence of length-prefixed signed signature scheme digests:
        //   * uint32: signature scheme id
        //   * length-prefixed bytes: signed digests for the respective signature scheme
        signatureSchemeDigests.add(
                Pair.of(
                        signatureSchemeVersion,
                        encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
                                signedDigest)));
    }

    private static byte[] encodeStampAttributes(Map<Integer, byte[]> stampAttributes) {
        int payloadSize = 0;
        for (byte[] attributeValue : stampAttributes.values()) {
            // Pair size + Attribute ID + Attribute value
            payloadSize += 4 + 4 + attributeValue.length;
        }

        // FORMAT (little endian):
        // * length-prefixed bytes: pair
        //   * uint32: ID
        //   * bytes: value
        ByteBuffer result = ByteBuffer.allocate(4 + payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(payloadSize);
        for (Map.Entry<Integer, byte[]> stampAttribute : stampAttributes.entrySet()) {
            // Pair size
            result.putInt(4 + stampAttribute.getValue().length);
            result.putInt(stampAttribute.getKey());
            result.put(stampAttribute.getValue());
        }
        return result.array();
    }

    private Map<Integer, byte[]> generateStampAttributes(SigningCertificateLineage lineage) {
        HashMap<Integer, byte[]> stampAttributes = new HashMap<>();

        if (mSourceStampTimestampEnabled) {
            // Write the current epoch time as the timestamp for the source stamp.
            long timestamp = Instant.now().getEpochSecond();
            if (timestamp > 0) {
                ByteBuffer attributeBuffer = ByteBuffer.allocate(8);
                attributeBuffer.order(ByteOrder.LITTLE_ENDIAN);
                attributeBuffer.putLong(timestamp);
                stampAttributes.put(SourceStampConstants.STAMP_TIME_ATTR_ID,
                        attributeBuffer.array());
            } else {
                // The epoch time should never be <= 0, and since security decisions can potentially
                // be made based on the value in the timestamp, throw an Exception to ensure the
                // issues with the environment are resolved before allowing the signing.
                throw new IllegalStateException(
                        "Received an invalid value from Instant#getTimestamp: " + timestamp);
            }
        }

        if (lineage != null) {
            stampAttributes.put(SourceStampConstants.PROOF_OF_ROTATION_ATTR_ID,
                    lineage.encodeSigningCertificateLineage());
        }
        return stampAttributes;
    }

    private static final class SourceStampBlock {
        public byte[] stampCertificate;
        public List<Pair<Integer, byte[]>> signedDigests;
        // Optional stamp attributes that are not required for verification.
        public byte[] stampAttributes;
        public List<Pair<Integer, byte[]>> signedStampAttributes;
    }

    /** Builder of {@link V2SourceStampSigner} instances. */
    public static class Builder {
        private final SignerConfig mSourceStampSignerConfig;
        private final Map<Integer, Map<ContentDigestAlgorithm, byte[]>> mSignatureSchemeDigestInfos;
        private boolean mSourceStampTimestampEnabled = true;

        /**
         * Instantiates a new {@code Builder} with the provided {@code sourceStampSignerConfig}
         * and the {@code signatureSchemeDigestInfos}.
         */
        public Builder(SignerConfig sourceStampSignerConfig,
                Map<Integer, Map<ContentDigestAlgorithm, byte[]>> signatureSchemeDigestInfos) {
            mSourceStampSignerConfig = sourceStampSignerConfig;
            mSignatureSchemeDigestInfos = signatureSchemeDigestInfos;
        }

        /**
         * Sets whether the source stamp should contain the timestamp attribute with the time
         * at which the source stamp was signed.
         */
        public Builder setSourceStampTimestampEnabled(boolean value) {
            mSourceStampTimestampEnabled = value;
            return this;
        }

        /**
         * Builds a new V2SourceStampSigner that can be used to generate a new source stamp
         * block signed with the specified signing config.
         */
        public V2SourceStampSigner build() {
            return new V2SourceStampSigner(this);
        }
    }
}
