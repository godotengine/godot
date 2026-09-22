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

import static com.android.apksig.internal.apk.ApkSigningBlockUtilsLite.getLengthPrefixedSlice;
import static com.android.apksig.internal.apk.ApkSigningBlockUtilsLite.getSignaturesToVerify;
import static com.android.apksig.internal.apk.ApkSigningBlockUtilsLite.readLengthPrefixedByteArray;
import static com.android.apksig.internal.apk.ApkSigningBlockUtilsLite.toHex;

import com.android.apksig.ApkVerificationIssue;
import com.android.apksig.Constants;
import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.internal.apk.ApkSignerInfo;
import com.android.apksig.internal.apk.ApkSupportedSignature;
import com.android.apksig.internal.apk.NoApkSupportedSignaturesException;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.util.ByteBufferUtils;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;

import java.io.ByteArrayInputStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Source Stamp verifier.
 *
 * <p>SourceStamp improves traceability of apps with respect to unauthorized distribution.
 *
 * <p>The stamp is part of the APK that is protected by the signing block.
 *
 * <p>The APK contents hash is signed using the stamp key, and is saved as part of the signing
 * block.
 */
class SourceStampVerifier {
    /** Hidden constructor to prevent instantiation. */
    private SourceStampVerifier() {
    }

    /**
     * Parses the SourceStamp block and populates the {@code result}.
     *
     * <p>This verifies signatures over digest provided.
     *
     * <p>This method adds one or more errors to the {@code result} if a verification error is
     * expected to be encountered on an Android platform version in the {@code [minSdkVersion,
     * maxSdkVersion]} range.
     */
    public static void verifyV1SourceStamp(
            ByteBuffer sourceStampBlockData,
            CertificateFactory certFactory,
            ApkSignerInfo result,
            byte[] apkDigest,
            byte[] sourceStampCertificateDigest,
            int minSdkVersion,
            int maxSdkVersion)
            throws ApkFormatException, NoSuchAlgorithmException {
        X509Certificate sourceStampCertificate =
                verifySourceStampCertificate(
                        sourceStampBlockData, certFactory, sourceStampCertificateDigest, result);
        if (result.containsWarnings() || result.containsErrors()) {
            return;
        }

        ByteBuffer apkDigestSignatures = getLengthPrefixedSlice(sourceStampBlockData);
        verifySourceStampSignature(
                apkDigest,
                minSdkVersion,
                maxSdkVersion,
                sourceStampCertificate,
                apkDigestSignatures,
                result);
    }

    /**
     * Parses the SourceStamp block and populates the {@code result}.
     *
     * <p>This verifies signatures over digest of multiple signature schemes provided.
     *
     * <p>This method adds one or more errors to the {@code result} if a verification error is
     * expected to be encountered on an Android platform version in the {@code [minSdkVersion,
     * maxSdkVersion]} range.
     */
    public static void verifyV2SourceStamp(
            ByteBuffer sourceStampBlockData,
            CertificateFactory certFactory,
            ApkSignerInfo result,
            Map<Integer, byte[]> signatureSchemeApkDigests,
            byte[] sourceStampCertificateDigest,
            int minSdkVersion,
            int maxSdkVersion)
            throws ApkFormatException, NoSuchAlgorithmException {
        X509Certificate sourceStampCertificate =
                verifySourceStampCertificate(
                        sourceStampBlockData, certFactory, sourceStampCertificateDigest, result);
        if (result.containsWarnings() || result.containsErrors()) {
            return;
        }

        // Parse signed signature schemes block.
        ByteBuffer signedSignatureSchemes = getLengthPrefixedSlice(sourceStampBlockData);
        Map<Integer, ByteBuffer> signedSignatureSchemeData = new HashMap<>();
        while (signedSignatureSchemes.hasRemaining()) {
            ByteBuffer signedSignatureScheme = getLengthPrefixedSlice(signedSignatureSchemes);
            int signatureSchemeId = signedSignatureScheme.getInt();
            ByteBuffer apkDigestSignatures = getLengthPrefixedSlice(signedSignatureScheme);
            signedSignatureSchemeData.put(signatureSchemeId, apkDigestSignatures);
        }

        for (Map.Entry<Integer, byte[]> signatureSchemeApkDigest :
                signatureSchemeApkDigests.entrySet()) {
            // TODO(b/192301300): Should the new v3.1 be included in the source stamp, or since a
            // v3.0 block must always be present with a v3.1 block is it sufficient to just use the
            // v3.0 block?
            if (signatureSchemeApkDigest.getKey()
                    == Constants.VERSION_APK_SIGNATURE_SCHEME_V31) {
                continue;
            }
            if (!signedSignatureSchemeData.containsKey(signatureSchemeApkDigest.getKey())) {
                result.addWarning(ApkVerificationIssue.SOURCE_STAMP_NO_SIGNATURE);
                return;
            }
            verifySourceStampSignature(
                    signatureSchemeApkDigest.getValue(),
                    minSdkVersion,
                    maxSdkVersion,
                    sourceStampCertificate,
                    signedSignatureSchemeData.get(signatureSchemeApkDigest.getKey()),
                    result);
            if (result.containsWarnings() || result.containsErrors()) {
                return;
            }
        }

        if (sourceStampBlockData.hasRemaining()) {
            // The stamp block contains some additional attributes.
            ByteBuffer stampAttributeData = getLengthPrefixedSlice(sourceStampBlockData);
            ByteBuffer stampAttributeDataSignatures = getLengthPrefixedSlice(sourceStampBlockData);

            byte[] stampAttributeBytes = new byte[stampAttributeData.remaining()];
            stampAttributeData.get(stampAttributeBytes);
            stampAttributeData.flip();

            verifySourceStampSignature(stampAttributeBytes, minSdkVersion, maxSdkVersion,
                    sourceStampCertificate, stampAttributeDataSignatures, result);
            if (result.containsErrors() || result.containsWarnings()) {
                return;
            }
            parseStampAttributes(stampAttributeData, sourceStampCertificate, result);
        }
    }

    private static X509Certificate verifySourceStampCertificate(
            ByteBuffer sourceStampBlockData,
            CertificateFactory certFactory,
            byte[] sourceStampCertificateDigest,
            ApkSignerInfo result)
            throws NoSuchAlgorithmException, ApkFormatException {
        // Parse the SourceStamp certificate.
        byte[] sourceStampEncodedCertificate = readLengthPrefixedByteArray(sourceStampBlockData);
        X509Certificate sourceStampCertificate;
        try {
            sourceStampCertificate = (X509Certificate) certFactory.generateCertificate(
                    new ByteArrayInputStream(sourceStampEncodedCertificate));
        } catch (CertificateException e) {
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_CERTIFICATE, e);
            return null;
        }
        // Wrap the cert so that the result's getEncoded returns exactly the original encoded
        // form. Without this, getEncoded may return a different form from what was stored in
        // the signature. This is because some X509Certificate(Factory) implementations
        // re-encode certificates.
        sourceStampCertificate =
                new GuaranteedEncodedFormX509Certificate(
                        sourceStampCertificate, sourceStampEncodedCertificate);
        result.certs.add(sourceStampCertificate);
        // Verify the SourceStamp certificate found in the signing block is the same as the
        // SourceStamp certificate found in the APK.
        MessageDigest messageDigest = MessageDigest.getInstance("SHA-256");
        messageDigest.update(sourceStampEncodedCertificate);
        byte[] sourceStampBlockCertificateDigest = messageDigest.digest();
        if (!Arrays.equals(sourceStampCertificateDigest, sourceStampBlockCertificateDigest)) {
            result.addWarning(
                    ApkVerificationIssue
                            .SOURCE_STAMP_CERTIFICATE_MISMATCH_BETWEEN_SIGNATURE_BLOCK_AND_APK,
                    toHex(sourceStampBlockCertificateDigest),
                    toHex(sourceStampCertificateDigest));
            return null;
        }
        return sourceStampCertificate;
    }

    private static void verifySourceStampSignature(
            byte[] data,
            int minSdkVersion,
            int maxSdkVersion,
            X509Certificate sourceStampCertificate,
            ByteBuffer signatures,
            ApkSignerInfo result) {
        // Parse the signatures block and identify supported signatures
        int signatureCount = 0;
        List<ApkSupportedSignature> supportedSignatures = new ArrayList<>(1);
        while (signatures.hasRemaining()) {
            signatureCount++;
            try {
                ByteBuffer signature = getLengthPrefixedSlice(signatures);
                int sigAlgorithmId = signature.getInt();
                byte[] sigBytes = readLengthPrefixedByteArray(signature);
                SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.findById(sigAlgorithmId);
                if (signatureAlgorithm == null) {
                    result.addInfoMessage(
                            ApkVerificationIssue.SOURCE_STAMP_UNKNOWN_SIG_ALGORITHM,
                            sigAlgorithmId);
                    continue;
                }
                supportedSignatures.add(
                        new ApkSupportedSignature(signatureAlgorithm, sigBytes));
            } catch (ApkFormatException | BufferUnderflowException e) {
                result.addWarning(
                        ApkVerificationIssue.SOURCE_STAMP_MALFORMED_SIGNATURE, signatureCount);
                return;
            }
        }
        if (supportedSignatures.isEmpty()) {
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_NO_SIGNATURE);
            return;
        }
        // Verify signatures over digests using the SourceStamp's certificate.
        List<ApkSupportedSignature> signaturesToVerify;
        try {
            signaturesToVerify =
                    getSignaturesToVerify(
                            supportedSignatures, minSdkVersion, maxSdkVersion, true);
        } catch (NoApkSupportedSignaturesException e) {
            // To facilitate debugging capture the signature algorithms and resulting exception in
            // the warning.
            StringBuilder signatureAlgorithms = new StringBuilder();
            for (ApkSupportedSignature supportedSignature : supportedSignatures) {
                if (signatureAlgorithms.length() > 0) {
                    signatureAlgorithms.append(", ");
                }
                signatureAlgorithms.append(supportedSignature.algorithm);
            }
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_NO_SUPPORTED_SIGNATURE,
                    signatureAlgorithms.toString(), e);
            return;
        }
        for (ApkSupportedSignature signature : signaturesToVerify) {
            SignatureAlgorithm signatureAlgorithm = signature.algorithm;
            String jcaSignatureAlgorithm =
                    signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getFirst();
            AlgorithmParameterSpec jcaSignatureAlgorithmParams =
                    signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getSecond();
            PublicKey publicKey = sourceStampCertificate.getPublicKey();
            try {
                Signature sig = Signature.getInstance(jcaSignatureAlgorithm);
                sig.initVerify(publicKey);
                if (jcaSignatureAlgorithmParams != null) {
                    sig.setParameter(jcaSignatureAlgorithmParams);
                }
                sig.update(data);
                byte[] sigBytes = signature.signature;
                if (!sig.verify(sigBytes)) {
                    result.addWarning(
                            ApkVerificationIssue.SOURCE_STAMP_DID_NOT_VERIFY, signatureAlgorithm);
                    return;
                }
            } catch (InvalidKeyException
                    | InvalidAlgorithmParameterException
                    | SignatureException
                    | NoSuchAlgorithmException e) {
                result.addWarning(
                        ApkVerificationIssue.SOURCE_STAMP_VERIFY_EXCEPTION, signatureAlgorithm, e);
                return;
            }
        }
    }

    private static void parseStampAttributes(ByteBuffer stampAttributeData,
            X509Certificate sourceStampCertificate, ApkSignerInfo result)
            throws ApkFormatException {
        ByteBuffer stampAttributes = getLengthPrefixedSlice(stampAttributeData);
        int stampAttributeCount = 0;
        while (stampAttributes.hasRemaining()) {
            stampAttributeCount++;
            try {
                ByteBuffer attribute = getLengthPrefixedSlice(stampAttributes);
                int id = attribute.getInt();
                byte[] value = ByteBufferUtils.toByteArray(attribute);
                if (id == SourceStampConstants.PROOF_OF_ROTATION_ATTR_ID) {
                    readStampCertificateLineage(value, sourceStampCertificate, result);
                } else if (id == SourceStampConstants.STAMP_TIME_ATTR_ID) {
                    long timestamp = ByteBuffer.wrap(value).order(
                            ByteOrder.LITTLE_ENDIAN).getLong();
                    if (timestamp > 0) {
                        result.timestamp = timestamp;
                    } else {
                        result.addWarning(ApkVerificationIssue.SOURCE_STAMP_INVALID_TIMESTAMP,
                                timestamp);
                    }
                } else {
                    result.addInfoMessage(ApkVerificationIssue.SOURCE_STAMP_UNKNOWN_ATTRIBUTE, id);
                }
            } catch (ApkFormatException | BufferUnderflowException e) {
                result.addWarning(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_ATTRIBUTE,
                        stampAttributeCount);
                return;
            }
        }
    }

    private static void readStampCertificateLineage(byte[] lineageBytes,
            X509Certificate sourceStampCertificate, ApkSignerInfo result) {
        try {
            // SourceStampCertificateLineage is verified when built
            List<SourceStampCertificateLineage.SigningCertificateNode> nodes =
                    SourceStampCertificateLineage.readSigningCertificateLineage(
                            ByteBuffer.wrap(lineageBytes).order(ByteOrder.LITTLE_ENDIAN));
            for (int i = 0; i < nodes.size(); i++) {
                result.certificateLineage.add(nodes.get(i).signingCert);
            }
            // Make sure that the last cert in the chain matches this signer cert
            if (!sourceStampCertificate.equals(
                    result.certificateLineage.get(result.certificateLineage.size() - 1))) {
                result.addWarning(ApkVerificationIssue.SOURCE_STAMP_POR_CERT_MISMATCH);
            }
        } catch (SecurityException e) {
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_POR_DID_NOT_VERIFY);
        } catch (IllegalArgumentException e) {
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_POR_CERT_MISMATCH);
        } catch (Exception e) {
            result.addWarning(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_LINEAGE);
        }
    }
}
