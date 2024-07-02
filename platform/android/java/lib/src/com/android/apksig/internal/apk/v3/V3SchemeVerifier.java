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

package com.android.apksig.internal.apk.v3;

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.getLengthPrefixedSlice;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.readLengthPrefixedByteArray;

import com.android.apksig.ApkVerificationIssue;
import com.android.apksig.ApkVerifier.Issue;
import com.android.apksig.SigningCertificateLineage;
import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils.SignatureNotFoundException;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.apk.SignatureInfo;
import com.android.apksig.internal.util.ByteBufferUtils;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import com.android.apksig.internal.util.X509CertificateUtils;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.RunnablesExecutor;

import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * APK Signature Scheme v3 verifier.
 *
 * <p>APK Signature Scheme v3, like v2 is a whole-file signature scheme which aims to protect every
 * single bit of the APK, as opposed to the JAR Signature Scheme which protects only the names and
 * uncompressed contents of ZIP entries.
 *
 * @see <a href="https://source.android.com/security/apksigning/v2.html">APK Signature Scheme v2</a>
 */
public class V3SchemeVerifier {
    private final RunnablesExecutor mExecutor;
    private final DataSource mApk;
    private final ApkUtils.ZipSections mZipSections;
    private final ApkSigningBlockUtils.Result mResult;
    private final Set<ContentDigestAlgorithm> mContentDigestsToVerify;
    private final int mMinSdkVersion;
    private final int mMaxSdkVersion;
    private final int mBlockId;
    private final OptionalInt mOptionalRotationMinSdkVersion;
    private final boolean mFullVerification;

    private ByteBuffer mApkSignatureSchemeV3Block;

    private V3SchemeVerifier(
            RunnablesExecutor executor,
            DataSource apk,
            ApkUtils.ZipSections zipSections,
            Set<ContentDigestAlgorithm> contentDigestsToVerify,
            ApkSigningBlockUtils.Result result,
            int minSdkVersion,
            int maxSdkVersion,
            int blockId,
            OptionalInt optionalRotationMinSdkVersion,
            boolean fullVerification) {
        mExecutor = executor;
        mApk = apk;
        mZipSections = zipSections;
        mContentDigestsToVerify = contentDigestsToVerify;
        mResult = result;
        mMinSdkVersion = minSdkVersion;
        mMaxSdkVersion = maxSdkVersion;
        mBlockId = blockId;
        mOptionalRotationMinSdkVersion = optionalRotationMinSdkVersion;
        mFullVerification = fullVerification;
    }

    /**
     * Verifies the provided APK's APK Signature Scheme v3 signatures and returns the result of
     * verification. The APK must be considered verified only if
     * {@link ApkSigningBlockUtils.Result#verified} is
     * {@code true}. If verification fails, the result will contain errors -- see
     * {@link ApkSigningBlockUtils.Result#getErrors()}.
     *
     * <p>Verification succeeds iff the APK's APK Signature Scheme v3 signatures are expected to
     * verify on all Android platform versions in the {@code [minSdkVersion, maxSdkVersion]} range.
     * If the APK's signature is expected to not verify on any of the specified platform versions,
     * this method returns a result with one or more errors and whose
     * {@code Result.verified == false}, or this method throws an exception.
     *
     * <p>This method only verifies the v3.0 signing block without platform targeted rotation from
     * a v3.1 signing block. To verify a v3.1 signing block, or a v3.0 signing block in the presence
     * of a v3.1 block, configure a new {@link V3SchemeVerifier} using the {@code Builder}.
     *
     * @throws NoSuchAlgorithmException if the APK's signatures cannot be verified because a
     *         required cryptographic algorithm implementation is missing
     * @throws SignatureNotFoundException if no APK Signature Scheme v3
     * signatures are found
     * @throws IOException if an I/O error occurs when reading the APK
     */
    public static ApkSigningBlockUtils.Result verify(
            RunnablesExecutor executor,
            DataSource apk,
            ApkUtils.ZipSections zipSections,
            int minSdkVersion,
            int maxSdkVersion)
            throws IOException, NoSuchAlgorithmException, SignatureNotFoundException {
        return new V3SchemeVerifier.Builder(apk, zipSections, minSdkVersion, maxSdkVersion)
                .setRunnablesExecutor(executor)
                .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID)
                .build()
                .verify();
    }

    /**
     * Verifies the provided APK's v3 signatures and outputs the results into the provided
     * {@code result}. APK is considered verified only if there are no errors reported in the
     * {@code result}. See {@link #verify(RunnablesExecutor, DataSource, ApkUtils.ZipSections, int,
     * int)} for more information about the contract of this method.
     *
     * @return {@link ApkSigningBlockUtils.Result} populated with interesting information about the
     *        APK, such as information about signers, and verification errors and warnings
     */
    public ApkSigningBlockUtils.Result verify()
            throws IOException, NoSuchAlgorithmException, SignatureNotFoundException {
        if (mApk == null || mZipSections == null) {
            throw new IllegalStateException(
                    "A non-null apk and zip sections must be specified to verify an APK's v3 "
                            + "signatures");
        }
        SignatureInfo signatureInfo =
                ApkSigningBlockUtils.findSignature(mApk, mZipSections, mBlockId, mResult);
        mApkSignatureSchemeV3Block = signatureInfo.signatureBlock;

        DataSource beforeApkSigningBlock = mApk.slice(0, signatureInfo.apkSigningBlockOffset);
        DataSource centralDir =
                mApk.slice(
                        signatureInfo.centralDirOffset,
                        signatureInfo.eocdOffset - signatureInfo.centralDirOffset);
        ByteBuffer eocd = signatureInfo.eocd;

        parseSigners();

        if (mResult.containsErrors()) {
            return mResult;
        }
        ApkSigningBlockUtils.verifyIntegrity(mExecutor, beforeApkSigningBlock, centralDir, eocd,
                mContentDigestsToVerify, mResult);

        // make sure that the v3 signers cover the entire targeted sdk version ranges and that the
        // longest SigningCertificateHistory, if present, corresponds to the newest platform
        // versions
        SortedMap<Integer, ApkSigningBlockUtils.Result.SignerInfo> sortedSigners = new TreeMap<>();
        for (ApkSigningBlockUtils.Result.SignerInfo signer : mResult.signers) {
            sortedSigners.put(signer.maxSdkVersion, signer);
        }

        // first make sure there is neither overlap nor holes
        int firstMin = 0;
        int lastMax = 0;
        int lastLineageSize = 0;

        // while we're iterating through the signers, build up the list of lineages
        List<SigningCertificateLineage> lineages = new ArrayList<>(mResult.signers.size());

        for (ApkSigningBlockUtils.Result.SignerInfo signer : sortedSigners.values()) {
            int currentMin = signer.minSdkVersion;
            int currentMax = signer.maxSdkVersion;
            if (firstMin == 0) {
                // first round sets up our basis
                firstMin = currentMin;
            } else {
                // A signer's minimum SDK can equal the previous signer's maximum SDK if this signer
                // is targeting a development release.
                if (currentMin != (lastMax + 1)
                        && !(currentMin == lastMax && signerTargetsDevRelease(signer))) {
                    mResult.addError(Issue.V3_INCONSISTENT_SDK_VERSIONS);
                    break;
                }
            }
            lastMax = currentMax;

            // also, while we're here, make sure that the lineage sizes only increase
            if (signer.signingCertificateLineage != null) {
                int currLineageSize = signer.signingCertificateLineage.size();
                if (currLineageSize < lastLineageSize) {
                    mResult.addError(Issue.V3_INCONSISTENT_LINEAGES);
                    break;
                }
                lastLineageSize = currLineageSize;
                lineages.add(signer.signingCertificateLineage);
            }
        }

        // make sure we support our desired sdk ranges; if rotation is present in a v3.1 block
        // then the max level only needs to support up to that sdk version for rotation.
        if (firstMin > mMinSdkVersion
                || lastMax < (mOptionalRotationMinSdkVersion.isPresent()
                    ? mOptionalRotationMinSdkVersion.getAsInt() - 1 : mMaxSdkVersion)) {
            mResult.addError(Issue.V3_MISSING_SDK_VERSIONS, firstMin, lastMax);
        }

        try {
            mResult.signingCertificateLineage =
                    SigningCertificateLineage.consolidateLineages(lineages);
        } catch (IllegalArgumentException e) {
            mResult.addError(Issue.V3_INCONSISTENT_LINEAGES);
        }
        if (!mResult.containsErrors()) {
            mResult.verified = true;
        }
        return mResult;
    }

    /**
     * Parses each signer in the provided APK Signature Scheme v3 block and populates corresponding
     * {@code signerInfos} of the provided {@code result}.
     *
     * <p>This verifies signatures over {@code signed-data} block contained in each signer block.
     * However, this does not verify the integrity of the rest of the APK but rather simply reports
     * the expected digests of the rest of the APK (see {@code contentDigestsToVerify}).
     *
     * <p>This method adds one or more errors to the {@code result} if a verification error is
     * expected to be encountered on an Android platform version in the
     * {@code [minSdkVersion, maxSdkVersion]} range.
     */
    public static void parseSigners(
            ByteBuffer apkSignatureSchemeV3Block,
            Set<ContentDigestAlgorithm> contentDigestsToVerify,
            ApkSigningBlockUtils.Result result) throws NoSuchAlgorithmException {
        try {
            new V3SchemeVerifier.Builder(apkSignatureSchemeV3Block)
                    .setResult(result)
                    .setContentDigestsToVerify(contentDigestsToVerify)
                    .setFullVerification(false)
                    .build()
                    .parseSigners();
        } catch (IOException | SignatureNotFoundException e) {
            // This should never occur since the apkSignatureSchemeV3Block was already provided.
            throw new IllegalStateException("An exception was encountered when attempting to parse"
                    + " the signers from the provided APK Signature Scheme v3 block", e);
        }
    }

    /**
     * Parses each signer in the APK Signature Scheme v3 block and populates corresponding
     * {@link ApkSigningBlockUtils.Result.SignerInfo} instances in the
     * returned {@link ApkSigningBlockUtils.Result}.
     *
     * <p>This verifies signatures over {@code signed-data} block contained in each signer block.
     * However, this does not verify the integrity of the rest of the APK but rather simply reports
     * the expected digests of the rest of the APK (see {@link Builder#setContentDigestsToVerify}).
     *
     * <p>This method adds one or more errors to the returned {@code Result} if a verification error
     * is encountered when parsing the signers.
     */
    public ApkSigningBlockUtils.Result parseSigners()
            throws IOException, NoSuchAlgorithmException, SignatureNotFoundException {
        ByteBuffer signers;
        try {
            if (mApkSignatureSchemeV3Block == null) {
                SignatureInfo signatureInfo =
                        ApkSigningBlockUtils.findSignature(mApk, mZipSections, mBlockId, mResult);
                mApkSignatureSchemeV3Block = signatureInfo.signatureBlock;
            }
            signers = getLengthPrefixedSlice(mApkSignatureSchemeV3Block);
        } catch (ApkFormatException e) {
            mResult.addError(Issue.V3_SIG_MALFORMED_SIGNERS);
            return mResult;
        }
        if (!signers.hasRemaining()) {
            mResult.addError(Issue.V3_SIG_NO_SIGNERS);
            return mResult;
        }

        CertificateFactory certFactory;
        try {
            certFactory = CertificateFactory.getInstance("X.509");
        } catch (CertificateException e) {
            throw new RuntimeException("Failed to obtain X.509 CertificateFactory", e);
        }
        int signerCount = 0;
        while (signers.hasRemaining()) {
            int signerIndex = signerCount;
            signerCount++;
            ApkSigningBlockUtils.Result.SignerInfo signerInfo =
                    new ApkSigningBlockUtils.Result.SignerInfo();
            signerInfo.index = signerIndex;
            mResult.signers.add(signerInfo);
            try {
                ByteBuffer signer = getLengthPrefixedSlice(signers);
                parseSigner(signer, certFactory, signerInfo);
            } catch (ApkFormatException | BufferUnderflowException e) {
                signerInfo.addError(Issue.V3_SIG_MALFORMED_SIGNER);
                return mResult;
            }
        }
        return mResult;
    }

    /**
     * Parses the provided signer block and populates the {@code result}.
     *
     * <p>This verifies signatures over {@code signed-data} contained in this block, as well as
     * the data contained therein, but does not verify the integrity of the rest of the APK. To
     * facilitate APK integrity verification, this method adds the {@code contentDigestsToVerify}.
     * These digests can then be used to verify the integrity of the APK.
     *
     * <p>This method adds one or more errors to the {@code result} if a verification error is
     * expected to be encountered on an Android platform version in the
     * {@code [minSdkVersion, maxSdkVersion]} range.
     */
    private void parseSigner(ByteBuffer signerBlock, CertificateFactory certFactory,
            ApkSigningBlockUtils.Result.SignerInfo result)
            throws ApkFormatException, NoSuchAlgorithmException {
        ByteBuffer signedData = getLengthPrefixedSlice(signerBlock);
        byte[] signedDataBytes = new byte[signedData.remaining()];
        signedData.get(signedDataBytes);
        signedData.flip();
        result.signedData = signedDataBytes;

        int parsedMinSdkVersion = signerBlock.getInt();
        int parsedMaxSdkVersion = signerBlock.getInt();
        result.minSdkVersion = parsedMinSdkVersion;
        result.maxSdkVersion = parsedMaxSdkVersion;
        if (parsedMinSdkVersion < 0 || parsedMinSdkVersion > parsedMaxSdkVersion) {
            result.addError(
                    Issue.V3_SIG_INVALID_SDK_VERSIONS, parsedMinSdkVersion, parsedMaxSdkVersion);
        }
        ByteBuffer signatures = getLengthPrefixedSlice(signerBlock);
        byte[] publicKeyBytes = readLengthPrefixedByteArray(signerBlock);

        // Parse the signatures block and identify supported signatures
        int signatureCount = 0;
        List<ApkSigningBlockUtils.SupportedSignature> supportedSignatures = new ArrayList<>(1);
        while (signatures.hasRemaining()) {
            signatureCount++;
            try {
                ByteBuffer signature = getLengthPrefixedSlice(signatures);
                int sigAlgorithmId = signature.getInt();
                byte[] sigBytes = readLengthPrefixedByteArray(signature);
                result.signatures.add(
                        new ApkSigningBlockUtils.Result.SignerInfo.Signature(
                                sigAlgorithmId, sigBytes));
                SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.findById(sigAlgorithmId);
                if (signatureAlgorithm == null) {
                    result.addWarning(Issue.V3_SIG_UNKNOWN_SIG_ALGORITHM, sigAlgorithmId);
                    continue;
                }
                // TODO consider dropping deprecated signatures for v3 or modifying
                // getSignaturesToVerify (called below)
                supportedSignatures.add(
                        new ApkSigningBlockUtils.SupportedSignature(signatureAlgorithm, sigBytes));
            } catch (ApkFormatException | BufferUnderflowException e) {
                result.addError(Issue.V3_SIG_MALFORMED_SIGNATURE, signatureCount);
                return;
            }
        }
        if (result.signatures.isEmpty()) {
            result.addError(Issue.V3_SIG_NO_SIGNATURES);
            return;
        }

        // Verify signatures over signed-data block using the public key
        List<ApkSigningBlockUtils.SupportedSignature> signaturesToVerify = null;
        try {
            signaturesToVerify =
                    ApkSigningBlockUtils.getSignaturesToVerify(
                            supportedSignatures, result.minSdkVersion, result.maxSdkVersion);
        } catch (ApkSigningBlockUtils.NoSupportedSignaturesException e) {
            result.addError(Issue.V3_SIG_NO_SUPPORTED_SIGNATURES);
            return;
        }
        for (ApkSigningBlockUtils.SupportedSignature signature : signaturesToVerify) {
            SignatureAlgorithm signatureAlgorithm = signature.algorithm;
            String jcaSignatureAlgorithm =
                    signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getFirst();
            AlgorithmParameterSpec jcaSignatureAlgorithmParams =
                    signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getSecond();
            String keyAlgorithm = signatureAlgorithm.getJcaKeyAlgorithm();
            PublicKey publicKey;
            try {
                publicKey =
                        KeyFactory.getInstance(keyAlgorithm).generatePublic(
                                new X509EncodedKeySpec(publicKeyBytes));
            } catch (Exception e) {
                result.addError(Issue.V3_SIG_MALFORMED_PUBLIC_KEY, e);
                return;
            }
            try {
                Signature sig = Signature.getInstance(jcaSignatureAlgorithm);
                sig.initVerify(publicKey);
                if (jcaSignatureAlgorithmParams != null) {
                    sig.setParameter(jcaSignatureAlgorithmParams);
                }
                signedData.position(0);
                sig.update(signedData);
                byte[] sigBytes = signature.signature;
                if (!sig.verify(sigBytes)) {
                    result.addError(Issue.V3_SIG_DID_NOT_VERIFY, signatureAlgorithm);
                    return;
                }
                result.verifiedSignatures.put(signatureAlgorithm, sigBytes);
                mContentDigestsToVerify.add(signatureAlgorithm.getContentDigestAlgorithm());
            } catch (InvalidKeyException | InvalidAlgorithmParameterException
                    | SignatureException e) {
                result.addError(Issue.V3_SIG_VERIFY_EXCEPTION, signatureAlgorithm, e);
                return;
            }
        }

        // At least one signature over signedData has verified. We can now parse signed-data.
        signedData.position(0);
        ByteBuffer digests = getLengthPrefixedSlice(signedData);
        ByteBuffer certificates = getLengthPrefixedSlice(signedData);

        int signedMinSdkVersion = signedData.getInt();
        if (signedMinSdkVersion != parsedMinSdkVersion) {
            result.addError(
                    Issue.V3_MIN_SDK_VERSION_MISMATCH_BETWEEN_SIGNER_AND_SIGNED_DATA_RECORD,
                    parsedMinSdkVersion,
                    signedMinSdkVersion);
        }
        int signedMaxSdkVersion = signedData.getInt();
        if (signedMaxSdkVersion != parsedMaxSdkVersion) {
            result.addError(
                    Issue.V3_MAX_SDK_VERSION_MISMATCH_BETWEEN_SIGNER_AND_SIGNED_DATA_RECORD,
                    parsedMaxSdkVersion,
                    signedMaxSdkVersion);
        }
        ByteBuffer additionalAttributes = getLengthPrefixedSlice(signedData);

        // Parse the certificates block
        int certificateIndex = -1;
        while (certificates.hasRemaining()) {
            certificateIndex++;
            byte[] encodedCert = readLengthPrefixedByteArray(certificates);
            X509Certificate certificate;
            try {
                certificate = X509CertificateUtils.generateCertificate(encodedCert, certFactory);
            } catch (CertificateException e) {
                result.addError(
                        Issue.V3_SIG_MALFORMED_CERTIFICATE,
                        certificateIndex,
                        certificateIndex + 1,
                        e);
                return;
            }
            // Wrap the cert so that the result's getEncoded returns exactly the original encoded
            // form. Without this, getEncoded may return a different form from what was stored in
            // the signature. This is because some X509Certificate(Factory) implementations
            // re-encode certificates.
            certificate = new GuaranteedEncodedFormX509Certificate(certificate, encodedCert);
            result.certs.add(certificate);
        }

        if (result.certs.isEmpty()) {
            result.addError(Issue.V3_SIG_NO_CERTIFICATES);
            return;
        }
        X509Certificate mainCertificate = result.certs.get(0);
        byte[] certificatePublicKeyBytes;
        try {
            certificatePublicKeyBytes = ApkSigningBlockUtils.encodePublicKey(
                    mainCertificate.getPublicKey());
        } catch (InvalidKeyException e) {
            System.out.println("Caught an exception encoding the public key: " + e);
            e.printStackTrace();
            certificatePublicKeyBytes = mainCertificate.getPublicKey().getEncoded();
        }
        if (!Arrays.equals(publicKeyBytes, certificatePublicKeyBytes)) {
            result.addError(
                    Issue.V3_SIG_PUBLIC_KEY_MISMATCH_BETWEEN_CERTIFICATE_AND_SIGNATURES_RECORD,
                    ApkSigningBlockUtils.toHex(certificatePublicKeyBytes),
                    ApkSigningBlockUtils.toHex(publicKeyBytes));
            return;
        }

        // Parse the digests block
        int digestCount = 0;
        while (digests.hasRemaining()) {
            digestCount++;
            try {
                ByteBuffer digest = getLengthPrefixedSlice(digests);
                int sigAlgorithmId = digest.getInt();
                byte[] digestBytes = readLengthPrefixedByteArray(digest);
                result.contentDigests.add(
                        new ApkSigningBlockUtils.Result.SignerInfo.ContentDigest(
                                sigAlgorithmId, digestBytes));
            } catch (ApkFormatException | BufferUnderflowException e) {
                result.addError(Issue.V3_SIG_MALFORMED_DIGEST, digestCount);
                return;
            }
        }

        List<Integer> sigAlgsFromSignaturesRecord = new ArrayList<>(result.signatures.size());
        for (ApkSigningBlockUtils.Result.SignerInfo.Signature signature : result.signatures) {
            sigAlgsFromSignaturesRecord.add(signature.getAlgorithmId());
        }
        List<Integer> sigAlgsFromDigestsRecord = new ArrayList<>(result.contentDigests.size());
        for (ApkSigningBlockUtils.Result.SignerInfo.ContentDigest digest : result.contentDigests) {
            sigAlgsFromDigestsRecord.add(digest.getSignatureAlgorithmId());
        }

        if (!sigAlgsFromSignaturesRecord.equals(sigAlgsFromDigestsRecord)) {
            result.addError(
                    Issue.V3_SIG_SIG_ALG_MISMATCH_BETWEEN_SIGNATURES_AND_DIGESTS_RECORDS,
                    sigAlgsFromSignaturesRecord,
                    sigAlgsFromDigestsRecord);
            return;
        }

        // Parse the additional attributes block.
        int additionalAttributeCount = 0;
        boolean rotationAttrFound = false;
        while (additionalAttributes.hasRemaining()) {
            additionalAttributeCount++;
            try {
                ByteBuffer attribute =
                        getLengthPrefixedSlice(additionalAttributes);
                int id = attribute.getInt();
                byte[] value = ByteBufferUtils.toByteArray(attribute);
                result.additionalAttributes.add(
                        new ApkSigningBlockUtils.Result.SignerInfo.AdditionalAttribute(id, value));
                if (id == V3SchemeConstants.PROOF_OF_ROTATION_ATTR_ID) {
                    try {
                        // SigningCertificateLineage is verified when built
                        result.signingCertificateLineage =
                                SigningCertificateLineage.readFromV3AttributeValue(value);
                        // make sure that the last cert in the chain matches this signer cert
                        SigningCertificateLineage subLineage =
                                result.signingCertificateLineage.getSubLineage(result.certs.get(0));
                        if (result.signingCertificateLineage.size() != subLineage.size()) {
                            result.addError(Issue.V3_SIG_POR_CERT_MISMATCH);
                        }
                    } catch (SecurityException e) {
                        result.addError(Issue.V3_SIG_POR_DID_NOT_VERIFY);
                    } catch (IllegalArgumentException e) {
                        result.addError(Issue.V3_SIG_POR_CERT_MISMATCH);
                    } catch (Exception e) {
                        result.addError(Issue.V3_SIG_MALFORMED_LINEAGE);
                    }
                } else if (id == V3SchemeConstants.ROTATION_MIN_SDK_VERSION_ATTR_ID) {
                    rotationAttrFound = true;
                    // API targeting for rotation was added with V3.1; if the maxSdkVersion
                    // does not support v3.1 then ignore this attribute.
                    if (mMaxSdkVersion >= V3SchemeConstants.MIN_SDK_WITH_V31_SUPPORT
                            && mFullVerification) {
                        int attrRotationMinSdkVersion = ByteBuffer.wrap(value)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        if (mOptionalRotationMinSdkVersion.isPresent()) {
                            int rotationMinSdkVersion = mOptionalRotationMinSdkVersion.getAsInt();
                            if (attrRotationMinSdkVersion != rotationMinSdkVersion) {
                                result.addError(Issue.V31_ROTATION_MIN_SDK_MISMATCH,
                                    attrRotationMinSdkVersion, rotationMinSdkVersion);
                            }
                        } else {
                            result.addError(Issue.V31_BLOCK_MISSING, attrRotationMinSdkVersion);
                        }
                    }
                } else if (id == V3SchemeConstants.ROTATION_ON_DEV_RELEASE_ATTR_ID) {
                    // This attribute should only be used by a v3.1 signer to indicate rotation
                    // is targeting the development release that is using the SDK version of the
                    // previously released platform version.
                    if (mBlockId != V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID) {
                        result.addWarning(Issue.V31_ROTATION_TARGETS_DEV_RELEASE_ATTR_ON_V3_SIGNER);
                    }
                } else {
                    result.addWarning(Issue.V3_SIG_UNKNOWN_ADDITIONAL_ATTRIBUTE, id);
                }
            } catch (ApkFormatException | BufferUnderflowException e) {
                result.addError(
                        Issue.V3_SIG_MALFORMED_ADDITIONAL_ATTRIBUTE, additionalAttributeCount);
                return;
            }
        }
        if (mFullVerification && mOptionalRotationMinSdkVersion.isPresent() && !rotationAttrFound) {
            result.addWarning(Issue.V31_ROTATION_MIN_SDK_ATTR_MISSING,
                    mOptionalRotationMinSdkVersion.getAsInt());
        }
    }

    /**
     * Returns whether the specified {@code signerInfo} is targeting a development release.
     */
    public static boolean signerTargetsDevRelease(
            ApkSigningBlockUtils.Result.SignerInfo signerInfo) {
        boolean result = signerInfo.additionalAttributes.stream()
                .mapToInt(attribute -> attribute.getId())
                .anyMatch(attrId -> attrId == V3SchemeConstants.ROTATION_ON_DEV_RELEASE_ATTR_ID);
        return result;
    }

    /** Builder of {@link V3SchemeVerifier} instances. */
    public static class Builder {
        private RunnablesExecutor mExecutor = RunnablesExecutor.SINGLE_THREADED;
        private DataSource mApk;
        private ApkUtils.ZipSections mZipSections;
        private ByteBuffer mApkSignatureSchemeV3Block;
        private Set<ContentDigestAlgorithm> mContentDigestsToVerify;
        private ApkSigningBlockUtils.Result mResult;
        private int mMinSdkVersion;
        private int mMaxSdkVersion;
        private int mBlockId = V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
        private boolean mFullVerification = true;
        private OptionalInt mOptionalRotationMinSdkVersion = OptionalInt.empty();

        /**
         * Instantiates a new {@code Builder} for a {@code V3SchemeVerifier} that can be used to
         * verify the V3 signing block of the provided {@code apk} with the specified {@code
         * zipSections} over the range from {@code minSdkVersion} to {@code maxSdkVersion}.
         */
        public Builder(DataSource apk, ApkUtils.ZipSections zipSections, int minSdkVersion,
                int maxSdkVersion) {
            mApk = apk;
            mZipSections = zipSections;
            mMinSdkVersion = minSdkVersion;
            mMaxSdkVersion = maxSdkVersion;
        }

        /**
         * Instantiates a new {@code Builder} for a {@code V3SchemeVerifier} that can be used to
         * parse the {@link ApkSigningBlockUtils.Result.SignerInfo} instances from the {@code
         * apkSignatureSchemeV3Block}.
         *
         * <note>Full verification of the v3 signature is not possible when instantiating a new
         * {@code V3SchemeVerifier} with this method.</note>
         */
        public Builder(ByteBuffer apkSignatureSchemeV3Block) {
            mApkSignatureSchemeV3Block = apkSignatureSchemeV3Block;
        }

        /**
         * Sets the {@link RunnablesExecutor} to be used when verifying the APK's content digests.
         */
        public Builder setRunnablesExecutor(RunnablesExecutor executor) {
            mExecutor = executor;
            return this;
        }

        /**
         * Sets the V3 {code blockId} to be verified in the provided APK.
         *
         * <p>This {@code V3SchemeVerifier} currently supports the block IDs for the {@link
         * V3SchemeConstants#APK_SIGNATURE_SCHEME_V3_BLOCK_ID v3.0} and {@link
         * V3SchemeConstants#APK_SIGNATURE_SCHEME_V31_BLOCK_ID v3.1} signature schemes.
         */
        public Builder setBlockId(int blockId) {
            mBlockId = blockId;
            return this;
        }

        /**
         * Sets the {@code rotationMinSdkVersion} to be verified in the v3.0 signer's additional
         * attribute.
         *
         * <p>This value can be obtained from the signers returned when verifying the v3.1 signing
         * block of an APK; in the case of multiple signers targeting different SDK versions in the
         * v3.1 signing block, the minimum SDK version from all the signers should be used.
         */
        public Builder setRotationMinSdkVersion(int rotationMinSdkVersion) {
            mOptionalRotationMinSdkVersion = OptionalInt.of(rotationMinSdkVersion);
            return this;
        }

        /**
         * Sets the {@code result} instance to be used when returning verification results.
         *
         * <p>This method can be used when the caller already has a {@link
         * ApkSigningBlockUtils.Result} and wants to store the verification results in this
         * instance.
         */
        public Builder setResult(ApkSigningBlockUtils.Result result) {
            mResult = result;
            return this;
        }

        /**
         * Sets the instance to be used to store the {@code contentDigestsToVerify}.
         *
         * <p>This method can be used when the caller needs access to the {@code
         * contentDigestsToVerify} computed by this {@code V3SchemeVerifier}.
         */
        public Builder setContentDigestsToVerify(
                Set<ContentDigestAlgorithm> contentDigestsToVerify) {
            mContentDigestsToVerify = contentDigestsToVerify;
            return this;
        }

        /**
         * Sets whether full verification should be performed by the {@code V3SchemeVerifier} built
         * from this instance.
         *
         * <note>{@link #verify()} will always verify the content digests for the APK, but this
         * allows verification of the rotation minimum SDK version stripping attribute to be skipped
         * for scenarios where this value may not have been parsed from a V3.1 signing block (such
         * as when only {@link #parseSigners()} will be invoked.</note>
         */
        public Builder setFullVerification(boolean fullVerification) {
            mFullVerification = fullVerification;
            return this;
        }

        /**
         * Returns a new {@link V3SchemeVerifier} built with the configuration provided to this
         * {@code Builder}.
         */
        public V3SchemeVerifier build() {
            int sigSchemeVersion;
            switch (mBlockId) {
                case V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID:
                    sigSchemeVersion = ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3;
                    mMinSdkVersion = Math.max(mMinSdkVersion,
                            V3SchemeConstants.MIN_SDK_WITH_V3_SUPPORT);
                    break;
                case V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID:
                    sigSchemeVersion = ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V31;
                    // V3.1 supports targeting an SDK version later than that of the initial release
                    // in which it is supported; allow any range for V3.1 as long as V3.0 covers the
                    // rest of the range.
                    mMinSdkVersion = mMaxSdkVersion;
                    break;
                default:
                    throw new IllegalArgumentException(
                            String.format("Unsupported APK Signature Scheme V3 block ID: 0x%08x",
                                    mBlockId));
            }
            if (mResult == null) {
                mResult = new ApkSigningBlockUtils.Result(sigSchemeVersion);
            }
            if (mContentDigestsToVerify == null) {
                mContentDigestsToVerify = new HashSet<>(1);
            }

            V3SchemeVerifier verifier = new V3SchemeVerifier(
                    mExecutor,
                    mApk,
                    mZipSections,
                    mContentDigestsToVerify,
                    mResult,
                    mMinSdkVersion,
                    mMaxSdkVersion,
                    mBlockId,
                    mOptionalRotationMinSdkVersion,
                    mFullVerification);
            if (mApkSignatureSchemeV3Block != null) {
                verifier.mApkSignatureSchemeV3Block = mApkSignatureSchemeV3Block;
            }
            return verifier;
        }
    }
}
