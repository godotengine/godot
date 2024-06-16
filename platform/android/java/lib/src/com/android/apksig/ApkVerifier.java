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

package com.android.apksig;

import static com.android.apksig.apk.ApkUtils.SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME;
import static com.android.apksig.apk.ApkUtils.computeSha256DigestBytes;
import static com.android.apksig.apk.ApkUtils.getTargetSandboxVersionFromBinaryAndroidManifest;
import static com.android.apksig.apk.ApkUtils.getTargetSdkVersionFromBinaryAndroidManifest;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V31;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V31;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V4;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_JAR_SIGNATURE_SCHEME;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_SOURCE_STAMP;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.toHex;
import static com.android.apksig.internal.apk.v1.V1SchemeConstants.MANIFEST_ENTRY_NAME;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.MIN_SDK_WITH_V31_SUPPORT;

import com.android.apksig.ApkVerifier.Result.V2SchemeSignerInfo;
import com.android.apksig.ApkVerifier.Result.V3SchemeSignerInfo;
import com.android.apksig.SigningCertificateLineage.SignerConfig;
import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigResult;
import com.android.apksig.internal.apk.ApkSignerInfo;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils.Result.SignerInfo.ContentDigest;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.apk.SignatureInfo;
import com.android.apksig.internal.apk.SignatureNotFoundException;
import com.android.apksig.internal.apk.stamp.SourceStampConstants;
import com.android.apksig.internal.apk.stamp.V2SourceStampVerifier;
import com.android.apksig.internal.apk.v1.V1SchemeVerifier;
import com.android.apksig.internal.apk.v2.V2SchemeConstants;
import com.android.apksig.internal.apk.v2.V2SchemeVerifier;
import com.android.apksig.internal.apk.v3.V3SchemeConstants;
import com.android.apksig.internal.apk.v3.V3SchemeVerifier;
import com.android.apksig.internal.apk.v4.V4SchemeVerifier;
import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.zip.CentralDirectoryRecord;
import com.android.apksig.internal.zip.LocalFileRecord;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.DataSources;
import com.android.apksig.util.RunnablesExecutor;
import com.android.apksig.zip.ZipFormatException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * APK signature verifier which mimics the behavior of the Android platform.
 *
 * <p>The verifier is designed to closely mimic the behavior of Android platforms. This is to enable
 * the verifier to be used for checking whether an APK's signatures are expected to verify on
 * Android.
 *
 * <p>Use {@link Builder} to obtain instances of this verifier.
 *
 * @see <a href="https://source.android.com/security/apksigning/index.html">Application Signing</a>
 */
public class ApkVerifier {

    private static final Set<Issue> LINEAGE_RELATED_ISSUES = new HashSet<>(Arrays.asList(
        Issue.V3_SIG_MALFORMED_LINEAGE, Issue.V3_INCONSISTENT_LINEAGES,
        Issue.V3_SIG_POR_DID_NOT_VERIFY, Issue.V3_SIG_POR_CERT_MISMATCH));

    private static final Map<Integer, String> SUPPORTED_APK_SIG_SCHEME_NAMES =
            loadSupportedApkSigSchemeNames();

    private static Map<Integer, String> loadSupportedApkSigSchemeNames() {
        Map<Integer, String> supportedMap = new HashMap<>(2);
        supportedMap.put(
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2, "APK Signature Scheme v2");
        supportedMap.put(
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3, "APK Signature Scheme v3");
        return supportedMap;
    }

    private final File mApkFile;
    private final DataSource mApkDataSource;
    private final File mV4SignatureFile;

    private final Integer mMinSdkVersion;
    private final int mMaxSdkVersion;

    private ApkVerifier(
            File apkFile,
            DataSource apkDataSource,
            File v4SignatureFile,
            Integer minSdkVersion,
            int maxSdkVersion) {
        mApkFile = apkFile;
        mApkDataSource = apkDataSource;
        mV4SignatureFile = v4SignatureFile;
        mMinSdkVersion = minSdkVersion;
        mMaxSdkVersion = maxSdkVersion;
    }

    /**
     * Verifies the APK's signatures and returns the result of verification. The APK can be
     * considered verified iff the result's {@link Result#isVerified()} returns {@code true}.
     * The verification result also includes errors, warnings, and information about signers such
     * as their signing certificates.
     *
     * <p>Verification succeeds iff the APK's signature is expected to verify on all Android
     * platform versions specified via the {@link Builder}. If the APK's signature is expected to
     * not verify on any of the specified platform versions, this method returns a result with one
     * or more errors and whose {@link Result#isVerified()} returns {@code false}, or this method
     * throws an exception.
     *
     * @throws IOException              if an I/O error is encountered while reading the APK
     * @throws ApkFormatException       if the APK is malformed
     * @throws NoSuchAlgorithmException if the APK's signatures cannot be verified because a
     *                                  required cryptographic algorithm implementation is missing
     * @throws IllegalStateException    if this verifier's configuration is missing required
     *                                  information.
     */
    public Result verify() throws IOException, ApkFormatException, NoSuchAlgorithmException,
            IllegalStateException {
        Closeable in = null;
        try {
            DataSource apk;
            if (mApkDataSource != null) {
                apk = mApkDataSource;
            } else if (mApkFile != null) {
                RandomAccessFile f = new RandomAccessFile(mApkFile, "r");
                in = f;
                apk = DataSources.asDataSource(f, 0, f.length());
            } else {
                throw new IllegalStateException("APK not provided");
            }
            return verify(apk);
        } finally {
            if (in != null) {
                in.close();
            }
        }
    }

    /**
     * Verifies the APK's signatures and returns the result of verification. The APK can be
     * considered verified iff the result's {@link Result#isVerified()} returns {@code true}.
     * The verification result also includes errors, warnings, and information about signers.
     *
     * @param apk APK file contents
     * @throws IOException              if an I/O error is encountered while reading the APK
     * @throws ApkFormatException       if the APK is malformed
     * @throws NoSuchAlgorithmException if the APK's signatures cannot be verified because a
     *                                  required cryptographic algorithm implementation is missing
     */
    private Result verify(DataSource apk)
            throws IOException, ApkFormatException, NoSuchAlgorithmException {
        int maxSdkVersion = mMaxSdkVersion;

        ApkUtils.ZipSections zipSections;
        try {
            zipSections = ApkUtils.findZipSections(apk);
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Malformed APK: not a ZIP archive", e);
        }

        ByteBuffer androidManifest = null;

        int minSdkVersion = verifyAndGetMinSdkVersion(apk, zipSections);

        Result result = new Result();
        Map<Integer, Map<ContentDigestAlgorithm, byte[]>> signatureSchemeApkContentDigests =
                new HashMap<>();

        // The SUPPORTED_APK_SIG_SCHEME_NAMES contains the mapping from version number to scheme
        // name, but the verifiers use this parameter as the schemes supported by the target SDK
        // range. Since the code below skips signature verification based on max SDK the mapping of
        // supported schemes needs to be modified to ensure the verifiers do not report a stripped
        // signature for an SDK range that does not support that signature version. For instance an
        // APK with V1, V2, and V3 signatures and a max SDK of O would skip the V3 signature
        // verification, but the SUPPORTED_APK_SIG_SCHEME_NAMES contains version 3, so when the V2
        // verification is performed it would see the stripping protection attribute, see that V3
        // is in the list of supported signatures, and report a stripped signature.
        Map<Integer, String> supportedSchemeNames = getSupportedSchemeNames(maxSdkVersion);

        // Android N and newer attempts to verify APKs using the APK Signing Block, which can
        // include v2 and/or v3 signatures.  If none is found, it falls back to JAR signature
        // verification. If the signature is found but does not verify, the APK is rejected.
        Set<Integer> foundApkSigSchemeIds = new HashSet<>(2);
        if (maxSdkVersion >= AndroidSdkVersion.N) {
            RunnablesExecutor executor = RunnablesExecutor.SINGLE_THREADED;
            // Android T and newer attempts to verify APKs using APK Signature Scheme V3.1. v3.0
            // also includes stripping protection for the minimum SDK version on which the rotated
            // signing key should be used.
            int rotationMinSdkVersion = 0;
            if (maxSdkVersion >= MIN_SDK_WITH_V31_SUPPORT) {
                try {
                    ApkSigningBlockUtils.Result v31Result = new V3SchemeVerifier.Builder(apk,
                            zipSections, Math.max(minSdkVersion, MIN_SDK_WITH_V31_SUPPORT),
                            maxSdkVersion)
                            .setRunnablesExecutor(executor)
                            .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID)
                            .build()
                            .verify();
                    foundApkSigSchemeIds.add(VERSION_APK_SIGNATURE_SCHEME_V31);
                    rotationMinSdkVersion = v31Result.signers.stream().mapToInt(
                            signer -> signer.minSdkVersion).min().orElse(0);
                    result.mergeFrom(v31Result);
                    signatureSchemeApkContentDigests.put(
                            VERSION_APK_SIGNATURE_SCHEME_V31,
                            getApkContentDigestsFromSigningSchemeResult(v31Result));
                } catch (ApkSigningBlockUtils.SignatureNotFoundException ignored) {
                    // v3.1 signature not required
                }
                if (result.containsErrors()) {
                    return result;
                }
            }
            // Android P and newer attempts to verify APKs using APK Signature Scheme v3; since a
            // V3.1 block should only be written with a V3.0 block, always perform the V3.0 check
            // if the minSdkVersion supports V3.0.
            if (maxSdkVersion >= AndroidSdkVersion.P) {
                try {
                    V3SchemeVerifier.Builder builder = new V3SchemeVerifier.Builder(apk,
                            zipSections, Math.max(minSdkVersion, AndroidSdkVersion.P),
                            maxSdkVersion)
                            .setRunnablesExecutor(executor)
                            .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID);
                    if (rotationMinSdkVersion > 0) {
                        builder.setRotationMinSdkVersion(rotationMinSdkVersion);
                    }
                    ApkSigningBlockUtils.Result v3Result = builder.build().verify();
                    foundApkSigSchemeIds.add(ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3);
                    result.mergeFrom(v3Result);
                    signatureSchemeApkContentDigests.put(
                            ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3,
                            getApkContentDigestsFromSigningSchemeResult(v3Result));
                } catch (ApkSigningBlockUtils.SignatureNotFoundException ignored) {
                    // v3 signature not required unless a v3.1 signature was found as a v3.1
                    // signature is intended to support key rotation on T+ with the v3 signature
                    // containing the original signing key.
                    if (foundApkSigSchemeIds.contains(
                            VERSION_APK_SIGNATURE_SCHEME_V31)) {
                        result.addError(Issue.V31_BLOCK_FOUND_WITHOUT_V3_BLOCK);
                    }
                }
                if (result.containsErrors()) {
                    return result;
                }
            }

            // Attempt to verify the APK using v2 signing if necessary. Platforms prior to Android P
            // ignore APK Signature Scheme v3 signatures and always attempt to verify either JAR or
            // APK Signature Scheme v2 signatures.  Android P onwards verifies v2 signatures only if
            // no APK Signature Scheme v3 (or newer scheme) signatures were found.
            if (minSdkVersion < AndroidSdkVersion.P || foundApkSigSchemeIds.isEmpty()) {
                try {
                    ApkSigningBlockUtils.Result v2Result =
                            V2SchemeVerifier.verify(
                                    executor,
                                    apk,
                                    zipSections,
                                    supportedSchemeNames,
                                    foundApkSigSchemeIds,
                                    Math.max(minSdkVersion, AndroidSdkVersion.N),
                                    maxSdkVersion);
                    foundApkSigSchemeIds.add(ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2);
                    result.mergeFrom(v2Result);
                    signatureSchemeApkContentDigests.put(
                            ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2,
                            getApkContentDigestsFromSigningSchemeResult(v2Result));
                } catch (ApkSigningBlockUtils.SignatureNotFoundException ignored) {
                    // v2 signature not required
                }
                if (result.containsErrors()) {
                    return result;
                }
            }

            // If v4 file is specified, use additional verification on it
            if (mV4SignatureFile != null) {
                final ApkSigningBlockUtils.Result v4Result =
                        V4SchemeVerifier.verify(apk, mV4SignatureFile);
                foundApkSigSchemeIds.add(
                        ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V4);
                result.mergeFrom(v4Result);
                if (result.containsErrors()) {
                    return result;
                }
            }
        }

        // Android O and newer requires that APKs targeting security sandbox version 2 and higher
        // are signed using APK Signature Scheme v2 or newer.
        if (maxSdkVersion >= AndroidSdkVersion.O) {
            if (androidManifest == null) {
                androidManifest = getAndroidManifestFromApk(apk, zipSections);
            }
            int targetSandboxVersion =
                    getTargetSandboxVersionFromBinaryAndroidManifest(androidManifest.slice());
            if (targetSandboxVersion > 1) {
                if (foundApkSigSchemeIds.isEmpty()) {
                    result.addError(
                            Issue.NO_SIG_FOR_TARGET_SANDBOX_VERSION,
                            targetSandboxVersion);
                }
            }
        }

        List<CentralDirectoryRecord> cdRecords =
                V1SchemeVerifier.parseZipCentralDirectory(apk, zipSections);

        // Attempt to verify the APK using JAR signing if necessary. Platforms prior to Android N
        // ignore APK Signature Scheme v2 signatures and always attempt to verify JAR signatures.
        // Android N onwards verifies JAR signatures only if no APK Signature Scheme v2 (or newer
        // scheme) signatures were found.
        if ((minSdkVersion < AndroidSdkVersion.N) || (foundApkSigSchemeIds.isEmpty())) {
            V1SchemeVerifier.Result v1Result =
                    V1SchemeVerifier.verify(
                            apk,
                            zipSections,
                            supportedSchemeNames,
                            foundApkSigSchemeIds,
                            minSdkVersion,
                            maxSdkVersion);
            result.mergeFrom(v1Result);
            signatureSchemeApkContentDigests.put(
                    ApkSigningBlockUtils.VERSION_JAR_SIGNATURE_SCHEME,
                    getApkContentDigestFromV1SigningScheme(cdRecords, apk, zipSections));
        }
        if (result.containsErrors()) {
            return result;
        }

        // Verify the SourceStamp, if found in the APK.
        try {
            CentralDirectoryRecord sourceStampCdRecord = null;
            for (CentralDirectoryRecord cdRecord : cdRecords) {
                if (SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME.equals(
                        cdRecord.getName())) {
                    sourceStampCdRecord = cdRecord;
                    break;
                }
            }
            // If SourceStamp file is found inside the APK, there must be a SourceStamp
            // block in the APK signing block as well.
            if (sourceStampCdRecord != null) {
                byte[] sourceStampCertificateDigest =
                        LocalFileRecord.getUncompressedData(
                                apk,
                                sourceStampCdRecord,
                                zipSections.getZipCentralDirectoryOffset());
                ApkSigResult sourceStampResult =
                        V2SourceStampVerifier.verify(
                                apk,
                                zipSections,
                                sourceStampCertificateDigest,
                                signatureSchemeApkContentDigests,
                                Math.max(minSdkVersion, AndroidSdkVersion.R),
                                maxSdkVersion);
                result.mergeFrom(sourceStampResult);
            }
        } catch (SignatureNotFoundException ignored) {
            result.addWarning(Issue.SOURCE_STAMP_SIG_MISSING);
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Failed to read APK", e);
        }
        if (result.containsErrors()) {
            return result;
        }

        // Check whether v1 and v2 scheme signer identifies match, provided both v1 and v2
        // signatures verified.
        if ((result.isVerifiedUsingV1Scheme()) && (result.isVerifiedUsingV2Scheme())) {
            ArrayList<Result.V1SchemeSignerInfo> v1Signers =
                    new ArrayList<>(result.getV1SchemeSigners());
            ArrayList<Result.V2SchemeSignerInfo> v2Signers =
                    new ArrayList<>(result.getV2SchemeSigners());
            ArrayList<ByteArray> v1SignerCerts = new ArrayList<>();
            ArrayList<ByteArray> v2SignerCerts = new ArrayList<>();
            for (Result.V1SchemeSignerInfo signer : v1Signers) {
                try {
                    v1SignerCerts.add(new ByteArray(signer.getCertificate().getEncoded()));
                } catch (CertificateEncodingException e) {
                    throw new IllegalStateException(
                            "Failed to encode JAR signer " + signer.getName() + " certs", e);
                }
            }
            for (Result.V2SchemeSignerInfo signer : v2Signers) {
                try {
                    v2SignerCerts.add(new ByteArray(signer.getCertificate().getEncoded()));
                } catch (CertificateEncodingException e) {
                    throw new IllegalStateException(
                            "Failed to encode APK Signature Scheme v2 signer (index: "
                                    + signer.getIndex() + ") certs",
                            e);
                }
            }

            for (int i = 0; i < v1SignerCerts.size(); i++) {
                ByteArray v1Cert = v1SignerCerts.get(i);
                if (!v2SignerCerts.contains(v1Cert)) {
                    Result.V1SchemeSignerInfo v1Signer = v1Signers.get(i);
                    v1Signer.addError(Issue.V2_SIG_MISSING);
                    break;
                }
            }
            for (int i = 0; i < v2SignerCerts.size(); i++) {
                ByteArray v2Cert = v2SignerCerts.get(i);
                if (!v1SignerCerts.contains(v2Cert)) {
                    Result.V2SchemeSignerInfo v2Signer = v2Signers.get(i);
                    v2Signer.addError(Issue.JAR_SIG_MISSING);
                    break;
                }
            }
        }

        // If there is a v3 scheme signer and an earlier scheme signer, make sure that there is a
        // match, or in the event of signing certificate rotation, that the v1/v2 scheme signer
        // matches the oldest signing certificate in the provided SigningCertificateLineage
        if (result.isVerifiedUsingV3Scheme()
                && (result.isVerifiedUsingV1Scheme() || result.isVerifiedUsingV2Scheme())) {
            SigningCertificateLineage lineage = result.getSigningCertificateLineage();
            X509Certificate oldSignerCert;
            if (result.isVerifiedUsingV1Scheme()) {
                List<Result.V1SchemeSignerInfo> v1Signers = result.getV1SchemeSigners();
                if (v1Signers.size() != 1) {
                    // APK Signature Scheme v3 only supports single-signers, error to sign with
                    // multiple and then only one
                    result.addError(Issue.V3_SIG_MULTIPLE_PAST_SIGNERS);
                }
                oldSignerCert = v1Signers.get(0).mCertChain.get(0);
            } else {
                List<Result.V2SchemeSignerInfo> v2Signers = result.getV2SchemeSigners();
                if (v2Signers.size() != 1) {
                    // APK Signature Scheme v3 only supports single-signers, error to sign with
                    // multiple and then only one
                    result.addError(Issue.V3_SIG_MULTIPLE_PAST_SIGNERS);
                }
                oldSignerCert = v2Signers.get(0).mCerts.get(0);
            }
            if (lineage == null) {
                // no signing certificate history with which to contend, just make sure that v3
                // matches previous versions
                List<Result.V3SchemeSignerInfo> v3Signers = result.getV3SchemeSigners();
                if (v3Signers.size() != 1) {
                    // multiple v3 signers should never exist without rotation history, since
                    // multiple signers implies a different signer for different platform versions
                    result.addError(Issue.V3_SIG_MULTIPLE_SIGNERS);
                }
                try {
                    if (!Arrays.equals(oldSignerCert.getEncoded(),
                            v3Signers.get(0).mCerts.get(0).getEncoded())) {
                        result.addError(Issue.V3_SIG_PAST_SIGNERS_MISMATCH);
                    }
                } catch (CertificateEncodingException e) {
                    // we just go the encoding for the v1/v2 certs above, so must be v3
                    throw new RuntimeException(
                            "Failed to encode APK Signature Scheme v3 signer cert", e);
                }
            } else {
                // we have some signing history, make sure that the root of the history is the same
                // as our v1/v2 signer
                try {
                    lineage = lineage.getSubLineage(oldSignerCert);
                    if (lineage.size() != 1) {
                        // the v1/v2 signer was found, but not at the root of the lineage
                        result.addError(Issue.V3_SIG_PAST_SIGNERS_MISMATCH);
                    }
                } catch (IllegalArgumentException e) {
                    // the v1/v2 signer was not found in the lineage
                    result.addError(Issue.V3_SIG_PAST_SIGNERS_MISMATCH);
                }
            }
        }


        // If there is a v4 scheme signer, make sure that their certificates match.
        // The apkDigest field in the v4 signature should match the selected v2/v3.
        if (result.isVerifiedUsingV4Scheme()) {
            List<Result.V4SchemeSignerInfo> v4Signers = result.getV4SchemeSigners();

            List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> digestsFromV4 =
                    v4Signers.get(0).getContentDigests();
            if (digestsFromV4.size() != 1) {
                result.addError(Issue.V4_SIG_UNEXPECTED_DIGESTS, digestsFromV4.size());
                if (digestsFromV4.isEmpty()) {
                    return result;
                }
            }
            final byte[] digestFromV4 = digestsFromV4.get(0).getValue();

            if (result.isVerifiedUsingV3Scheme()) {
                final boolean isV31 = result.isVerifiedUsingV31Scheme();
                final int expectedSize = isV31 ? 2 : 1;
                if (v4Signers.size() != expectedSize) {
                    result.addError(isV31 ? Issue.V41_SIG_NEEDS_TWO_SIGNERS
                            : Issue.V4_SIG_MULTIPLE_SIGNERS);
                    return result;
                }

                checkV4Signer(result.getV3SchemeSigners(), v4Signers.get(0).mCerts, digestFromV4,
                        result);
                if (isV31) {
                    List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> digestsFromV41 =
                            v4Signers.get(1).getContentDigests();
                    if (digestsFromV41.size() != 1) {
                        result.addError(Issue.V4_SIG_UNEXPECTED_DIGESTS, digestsFromV41.size());
                        if (digestsFromV41.isEmpty()) {
                            return result;
                        }
                    }
                    final byte[] digestFromV41 = digestsFromV41.get(0).getValue();
                    checkV4Signer(result.getV31SchemeSigners(), v4Signers.get(1).mCerts,
                            digestFromV41, result);
                }
            } else if (result.isVerifiedUsingV2Scheme()) {
                if (v4Signers.size() != 1) {
                    result.addError(Issue.V4_SIG_MULTIPLE_SIGNERS);
                }

                List<Result.V2SchemeSignerInfo> v2Signers = result.getV2SchemeSigners();
                if (v2Signers.size() != 1) {
                    result.addError(Issue.V4_SIG_MULTIPLE_SIGNERS);
                }

                // Compare certificates.
                checkV4Certificate(v4Signers.get(0).mCerts, v2Signers.get(0).mCerts, result);

                // Compare digests.
                final byte[] digestFromV2 = pickBestDigestForV4(
                        v2Signers.get(0).getContentDigests());
                if (!Arrays.equals(digestFromV4, digestFromV2)) {
                    result.addError(Issue.V4_SIG_V2_V3_DIGESTS_MISMATCH, 2, toHex(digestFromV2),
                            toHex(digestFromV4));
                }
            } else {
                throw new RuntimeException("V4 signature must be also verified with V2/V3");
            }
        }

        // If the targetSdkVersion has a minimum required signature scheme version then verify
        // that the APK was signed with at least that version.
        try {
            if (androidManifest == null) {
                androidManifest = getAndroidManifestFromApk(apk, zipSections);
            }
        } catch (ApkFormatException e) {
            // If the manifest is not available then skip the minimum signature scheme requirement
            // to support bundle verification.
        }
        if (androidManifest != null) {
            int targetSdkVersion = getTargetSdkVersionFromBinaryAndroidManifest(
                    androidManifest.slice());
            int minSchemeVersion = getMinimumSignatureSchemeVersionForTargetSdk(targetSdkVersion);
            // The platform currently only enforces a single minimum signature scheme version, but
            // when later platform versions support another minimum version this will need to be
            // expanded to verify the minimum based on the target and maximum SDK version.
            if (minSchemeVersion > VERSION_JAR_SIGNATURE_SCHEME
                    && maxSdkVersion >= targetSdkVersion) {
                switch (minSchemeVersion) {
                    case VERSION_APK_SIGNATURE_SCHEME_V2:
                        if (result.isVerifiedUsingV2Scheme()) {
                            break;
                        }
                        // Allow this case to fall through to the next as a signature satisfying a
                        // later scheme version will also satisfy this requirement.
                    case VERSION_APK_SIGNATURE_SCHEME_V3:
                        if (result.isVerifiedUsingV3Scheme() || result.isVerifiedUsingV31Scheme()) {
                            break;
                        }
                        result.addError(Issue.MIN_SIG_SCHEME_FOR_TARGET_SDK_NOT_MET,
                                targetSdkVersion,
                                minSchemeVersion);
                }
            }
        }

        if (result.containsErrors()) {
            return result;
        }

        // Verified
        result.setVerified();
        if (result.isVerifiedUsingV31Scheme()) {
            List<Result.V3SchemeSignerInfo> v31Signers = result.getV31SchemeSigners();
            result.addSignerCertificate(v31Signers.get(v31Signers.size() - 1).getCertificate());
        } else if (result.isVerifiedUsingV3Scheme()) {
            List<Result.V3SchemeSignerInfo> v3Signers = result.getV3SchemeSigners();
            result.addSignerCertificate(v3Signers.get(v3Signers.size() - 1).getCertificate());
        } else if (result.isVerifiedUsingV2Scheme()) {
            for (Result.V2SchemeSignerInfo signerInfo : result.getV2SchemeSigners()) {
                result.addSignerCertificate(signerInfo.getCertificate());
            }
        } else if (result.isVerifiedUsingV1Scheme()) {
            for (Result.V1SchemeSignerInfo signerInfo : result.getV1SchemeSigners()) {
                result.addSignerCertificate(signerInfo.getCertificate());
            }
        } else {
            throw new RuntimeException(
                    "APK verified, but has not verified using any of v1, v2 or v3 schemes");
        }

        return result;
    }

    /**
     * Verifies and returns the minimum SDK version, either as provided to the builder or as read
     * from the {@code apk}'s AndroidManifest.xml.
     */
    private int verifyAndGetMinSdkVersion(DataSource apk, ApkUtils.ZipSections zipSections)
            throws ApkFormatException, IOException {
        if (mMinSdkVersion != null) {
            if (mMinSdkVersion < 0) {
                throw new IllegalArgumentException(
                        "minSdkVersion must not be negative: " + mMinSdkVersion);
            }
            if ((mMinSdkVersion != null) && (mMinSdkVersion > mMaxSdkVersion)) {
                throw new IllegalArgumentException(
                        "minSdkVersion (" + mMinSdkVersion + ") > maxSdkVersion (" + mMaxSdkVersion
                                + ")");
            }
            return mMinSdkVersion;
        }

        ByteBuffer androidManifest = null;
        // Need to obtain minSdkVersion from the APK's AndroidManifest.xml
        if (androidManifest == null) {
            androidManifest = getAndroidManifestFromApk(apk, zipSections);
        }
        int minSdkVersion =
                ApkUtils.getMinSdkVersionFromBinaryAndroidManifest(androidManifest.slice());
        if (minSdkVersion > mMaxSdkVersion) {
            throw new IllegalArgumentException(
                    "minSdkVersion from APK (" + minSdkVersion + ") > maxSdkVersion ("
                            + mMaxSdkVersion + ")");
        }
        return minSdkVersion;
    }

    /**
     * Returns the mapping of signature scheme version to signature scheme name for all signature
     * schemes starting from V2 supported by the {@code maxSdkVersion}.
     */
    private static Map<Integer, String> getSupportedSchemeNames(int maxSdkVersion) {
        Map<Integer, String> supportedSchemeNames;
        if (maxSdkVersion >= AndroidSdkVersion.P) {
            supportedSchemeNames = SUPPORTED_APK_SIG_SCHEME_NAMES;
        } else if (maxSdkVersion >= AndroidSdkVersion.N) {
            supportedSchemeNames = new HashMap<>(1);
            supportedSchemeNames.put(ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2,
                    SUPPORTED_APK_SIG_SCHEME_NAMES.get(
                            ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2));
        } else {
            supportedSchemeNames = Collections.emptyMap();
        }
        return supportedSchemeNames;
    }

    /**
     * Verifies the APK's source stamp signature and returns the result of the verification.
     *
     * <p>The APK's source stamp can be considered verified if the result's {@link
     * Result#isVerified} returns {@code true}. The details of the source stamp verification can
     * be obtained from the result's {@link Result#getSourceStampInfo()}} including the success or
     * failure cause from {@link Result.SourceStampInfo#getSourceStampVerificationStatus()}. If the
     * verification fails additional details regarding the failure can be obtained from {@link
     * Result#getAllErrors()}}.
     */
    public Result verifySourceStamp() {
        return verifySourceStamp(null);
    }

    /**
     * Verifies the APK's source stamp signature, including verification that the SHA-256 digest of
     * the stamp signing certificate matches the {@code expectedCertDigest}, and returns the result
     * of the verification.
     *
     * <p>A value of {@code null} for the {@code expectedCertDigest} will verify the source stamp,
     * if present, without verifying the actual source stamp certificate used to sign the source
     * stamp. This can be used to verify an APK contains a properly signed source stamp without
     * verifying a particular signer.
     *
     * @see #verifySourceStamp()
     */
    public Result verifySourceStamp(String expectedCertDigest) {
        Closeable in = null;
        try {
            DataSource apk;
            if (mApkDataSource != null) {
                apk = mApkDataSource;
            } else if (mApkFile != null) {
                RandomAccessFile f = new RandomAccessFile(mApkFile, "r");
                in = f;
                apk = DataSources.asDataSource(f, 0, f.length());
            } else {
                throw new IllegalStateException("APK not provided");
            }
            return verifySourceStamp(apk, expectedCertDigest);
        } catch (IOException e) {
            return createSourceStampResultWithError(
                    Result.SourceStampInfo.SourceStampVerificationStatus.VERIFICATION_ERROR,
                    Issue.UNEXPECTED_EXCEPTION, e);
        } finally {
            if (in != null) {
                try {
                    in.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    /**
     * Compares the digests coming from signature blocks. Returns {@code true} if at least one
     * digest algorithm is present in both digests and actual digests for all common algorithms
     * are the same.
     */
    public static boolean compareDigests(
            Map<ContentDigestAlgorithm, byte[]> firstDigests,
            Map<ContentDigestAlgorithm, byte[]> secondDigests) throws NoSuchAlgorithmException {

        Set<ContentDigestAlgorithm> intersectKeys = new HashSet<>(firstDigests.keySet());
        intersectKeys.retainAll(secondDigests.keySet());
        if (intersectKeys.isEmpty()) {
            return false;
        }

        for (ContentDigestAlgorithm algorithm : intersectKeys) {
            if (!Arrays.equals(firstDigests.get(algorithm),
                    secondDigests.get(algorithm))) {
                return false;
            }
        }
        return true;
    }


    /**
     * Verifies the provided {@code apk}'s source stamp signature, including verification of the
     * SHA-256 digest of the stamp signing certificate matches the {@code expectedCertDigest}, and
     * returns the result of the verification.
     *
     * @see #verifySourceStamp(String)
     */
    private Result verifySourceStamp(DataSource apk, String expectedCertDigest) {
        try {
            ApkUtils.ZipSections zipSections = ApkUtils.findZipSections(apk);
            int minSdkVersion = verifyAndGetMinSdkVersion(apk, zipSections);

            // Attempt to obtain the source stamp's certificate digest from the APK.
            List<CentralDirectoryRecord> cdRecords =
                    V1SchemeVerifier.parseZipCentralDirectory(apk, zipSections);
            CentralDirectoryRecord sourceStampCdRecord = null;
            for (CentralDirectoryRecord cdRecord : cdRecords) {
                if (SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME.equals(cdRecord.getName())) {
                    sourceStampCdRecord = cdRecord;
                    break;
                }
            }

            // If the source stamp's certificate digest is not available within the APK then the
            // source stamp cannot be verified; check if a source stamp signing block is in the
            // APK's signature block to determine the appropriate status to return.
            if (sourceStampCdRecord == null) {
                boolean stampSigningBlockFound;
                try {
                    ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                            VERSION_SOURCE_STAMP);
                    ApkSigningBlockUtils.findSignature(apk, zipSections,
                            SourceStampConstants.V2_SOURCE_STAMP_BLOCK_ID, result);
                    stampSigningBlockFound = true;
                } catch (ApkSigningBlockUtils.SignatureNotFoundException e) {
                    stampSigningBlockFound = false;
                }
                if (stampSigningBlockFound) {
                    return createSourceStampResultWithError(
                            Result.SourceStampInfo.SourceStampVerificationStatus.STAMP_NOT_VERIFIED,
                            Issue.SOURCE_STAMP_SIGNATURE_BLOCK_WITHOUT_CERT_DIGEST);
                } else {
                    return createSourceStampResultWithError(
                            Result.SourceStampInfo.SourceStampVerificationStatus.STAMP_MISSING,
                            Issue.SOURCE_STAMP_CERT_DIGEST_AND_SIG_BLOCK_MISSING);
                }
            }

            // Verify that the contents of the source stamp certificate digest match the expected
            // value, if provided.
            byte[] sourceStampCertificateDigest =
                    LocalFileRecord.getUncompressedData(
                            apk,
                            sourceStampCdRecord,
                            zipSections.getZipCentralDirectoryOffset());
            if (expectedCertDigest != null) {
                String actualCertDigest = ApkSigningBlockUtils.toHex(sourceStampCertificateDigest);
                if (!expectedCertDigest.equalsIgnoreCase(actualCertDigest)) {
                    return createSourceStampResultWithError(
                            Result.SourceStampInfo.SourceStampVerificationStatus
                                    .CERT_DIGEST_MISMATCH,
                            Issue.SOURCE_STAMP_EXPECTED_DIGEST_MISMATCH, actualCertDigest,
                            expectedCertDigest);
                }
            }

            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> signatureSchemeApkContentDigests =
                    new HashMap<>();
            Map<Integer, String> supportedSchemeNames = getSupportedSchemeNames(mMaxSdkVersion);
            Set<Integer> foundApkSigSchemeIds = new HashSet<>(2);

            Result result = new Result();
            ApkSigningBlockUtils.Result v3Result = null;
            if (mMaxSdkVersion >= AndroidSdkVersion.P) {
                v3Result = getApkContentDigests(apk, zipSections, foundApkSigSchemeIds,
                        supportedSchemeNames, signatureSchemeApkContentDigests,
                        VERSION_APK_SIGNATURE_SCHEME_V3,
                        Math.max(minSdkVersion, AndroidSdkVersion.P));
                if (v3Result != null && v3Result.containsErrors()) {
                    result.mergeFrom(v3Result);
                    return mergeSourceStampResult(
                            Result.SourceStampInfo.SourceStampVerificationStatus.VERIFICATION_ERROR,
                            result);
                }
            }

            ApkSigningBlockUtils.Result v2Result = null;
            if (mMaxSdkVersion >= AndroidSdkVersion.N && (minSdkVersion < AndroidSdkVersion.P
                    || foundApkSigSchemeIds.isEmpty())) {
                v2Result = getApkContentDigests(apk, zipSections, foundApkSigSchemeIds,
                        supportedSchemeNames, signatureSchemeApkContentDigests,
                        VERSION_APK_SIGNATURE_SCHEME_V2,
                        Math.max(minSdkVersion, AndroidSdkVersion.N));
                if (v2Result != null && v2Result.containsErrors()) {
                    result.mergeFrom(v2Result);
                    return mergeSourceStampResult(
                            Result.SourceStampInfo.SourceStampVerificationStatus.VERIFICATION_ERROR,
                            result);
                }
            }

            if (minSdkVersion < AndroidSdkVersion.N || foundApkSigSchemeIds.isEmpty()) {
                signatureSchemeApkContentDigests.put(VERSION_JAR_SIGNATURE_SCHEME,
                        getApkContentDigestFromV1SigningScheme(cdRecords, apk, zipSections));
            }

            ApkSigResult sourceStampResult =
                    V2SourceStampVerifier.verify(
                            apk,
                            zipSections,
                            sourceStampCertificateDigest,
                            signatureSchemeApkContentDigests,
                            minSdkVersion,
                            mMaxSdkVersion);
            result.mergeFrom(sourceStampResult);
            // Since the caller is only seeking to verify the source stamp the Result can be marked
            // as verified if the source stamp verification was successful.
            if (sourceStampResult.verified) {
                result.setVerified();
            } else {
                // To prevent APK signature verification with a failed / missing source stamp the
                // source stamp verification will only log warnings; to allow the caller to capture
                // the failure reason treat all warnings as errors.
                result.setWarningsAsErrors(true);
            }
            return result;
        } catch (ApkFormatException | IOException | ZipFormatException e) {
            return createSourceStampResultWithError(
                    Result.SourceStampInfo.SourceStampVerificationStatus.VERIFICATION_ERROR,
                    Issue.MALFORMED_APK, e);
        } catch (NoSuchAlgorithmException e) {
            return createSourceStampResultWithError(
                    Result.SourceStampInfo.SourceStampVerificationStatus.VERIFICATION_ERROR,
                    Issue.UNEXPECTED_EXCEPTION, e);
        } catch (SignatureNotFoundException e) {
            return createSourceStampResultWithError(
                    Result.SourceStampInfo.SourceStampVerificationStatus.STAMP_NOT_VERIFIED,
                    Issue.SOURCE_STAMP_SIG_MISSING);
        }
    }

    /**
     * Creates and returns a {@code Result} that can be returned for source stamp verification
     * with the provided source stamp {@code verificationStatus}, and logs an error for the
     * specified {@code issue} and {@code params}.
     */
    private static Result createSourceStampResultWithError(
            Result.SourceStampInfo.SourceStampVerificationStatus verificationStatus, Issue issue,
            Object... params) {
        Result result = new Result();
        result.addError(issue, params);
        return mergeSourceStampResult(verificationStatus, result);
    }

    /**
     * Creates a new {@link Result.SourceStampInfo} under the provided {@code result} and sets the
     * source stamp status to the provided {@code verificationStatus}.
     */
    private static Result mergeSourceStampResult(
            Result.SourceStampInfo.SourceStampVerificationStatus verificationStatus,
            Result result) {
        result.mSourceStampInfo = new Result.SourceStampInfo(verificationStatus);
        return result;
    }

    /**
     * Gets content digests, signing lineage and certificates from the given {@code schemeId} block
     * alongside encountered errors info and creates a new {@code Result} containing all this
     * information.
     */
    public static Result getSigningBlockResult(
        DataSource apk, ApkUtils.ZipSections zipSections, int sdkVersion, int schemeId)
        throws IOException, NoSuchAlgorithmException{
        Map<Integer, Map<ContentDigestAlgorithm, byte[]>> sigSchemeApkContentDigests =
                new HashMap<>();
        Map<Integer, String> supportedSchemeNames = getSupportedSchemeNames(sdkVersion);
        Set<Integer> foundApkSigSchemeIds = new HashSet<>(2);

        Result result = new Result();
        result.mergeFrom(getApkContentDigests(apk, zipSections,
                foundApkSigSchemeIds, supportedSchemeNames, sigSchemeApkContentDigests,
                schemeId, sdkVersion, sdkVersion));
        return result;
    }

    /**
     * Gets the content digest from the {@code result}'s signers. Ignores {@code ContentDigest}s
     * for which {@code SignatureAlgorithm} is {@code null}.
     */
    public static Map<ContentDigestAlgorithm, byte[]> getContentDigestsFromResult(
        Result result, int schemeId) {
        Map<ContentDigestAlgorithm, byte[]>  apkContentDigests = new HashMap<>();
        if (!(schemeId == VERSION_APK_SIGNATURE_SCHEME_V2
                || schemeId == VERSION_APK_SIGNATURE_SCHEME_V3
                || schemeId == VERSION_APK_SIGNATURE_SCHEME_V31)) {
            return apkContentDigests;
        }
        switch (schemeId) {
            case VERSION_APK_SIGNATURE_SCHEME_V2:
                for (V2SchemeSignerInfo signerInfo : result.getV2SchemeSigners()) {
                    getContentDigests(signerInfo.getContentDigests(), apkContentDigests);
                }
                break;
            case VERSION_APK_SIGNATURE_SCHEME_V3:
                for (Result.V3SchemeSignerInfo signerInfo : result.getV3SchemeSigners()) {
                    getContentDigests(signerInfo.getContentDigests(), apkContentDigests);
                }
                break;
            case  VERSION_APK_SIGNATURE_SCHEME_V31:
                for (Result.V3SchemeSignerInfo signerInfo : result.getV31SchemeSigners()) {
                    getContentDigests(signerInfo.getContentDigests(), apkContentDigests);
                }
                break;
        }
        return apkContentDigests;
    }

    private static void getContentDigests(
            List<ContentDigest> digests, Map<ContentDigestAlgorithm, byte[]> contentDigestsMap) {
        for (ApkSigningBlockUtils.Result.SignerInfo.ContentDigest contentDigest :
            digests) {
            SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.findById(
                    contentDigest.getSignatureAlgorithmId());
            if (signatureAlgorithm == null) {
                continue;
            }
            contentDigestsMap.put(signatureAlgorithm.getContentDigestAlgorithm(),
                    contentDigest.getValue());
        }
    }

    /**
     * Checks whether a given {@code result} contains errors indicating that a signing certificate
     * lineage is incorrect.
     */
    public static boolean containsLineageErrors(
        Result result) {
        if (!result.containsErrors()) {
            return false;
        }

        return (result.getAllErrors().stream().map(i -> i.getIssue())
                .anyMatch(error -> LINEAGE_RELATED_ISSUES.contains(error)));
    }


    /**
     * Gets a lineage from the first signer from a given {@code result}.
     * If the {@code result} contains errors related to the lineage incorrectness or there are no
     * signers or certificates, it returns {@code null}.
     * If the lineage is empty but there is a signer, it returns a 1-element lineage containing
     * the signing key.
     */
    public static SigningCertificateLineage getLineageFromResult(
        Result result, int sdkVersion, int schemeId)
        throws CertificateEncodingException, InvalidKeyException, NoSuchAlgorithmException,
        SignatureException {
        if (!(schemeId == VERSION_APK_SIGNATURE_SCHEME_V3
                        || schemeId == VERSION_APK_SIGNATURE_SCHEME_V31)
                || containsLineageErrors(result)) {
            return null;
        }
        List<V3SchemeSignerInfo> signersInfo =
                schemeId == VERSION_APK_SIGNATURE_SCHEME_V3 ?
                        result.getV3SchemeSigners() : result.getV31SchemeSigners();
        if (signersInfo.isEmpty()) {
            return null;
        }
        V3SchemeSignerInfo firstSignerInfo = signersInfo.get(0);
        SigningCertificateLineage lineage = firstSignerInfo.mSigningCertificateLineage;
        if (lineage == null && firstSignerInfo.getCertificate() != null) {
            try {
                lineage = new SigningCertificateLineage.Builder(
                        new SignerConfig.Builder(
                                /* privateKey= */ null, firstSignerInfo.getCertificate())
                                .build()).build();
            } catch (Exception e) {
                return null;
            }
        }
        return lineage;
    }

    /**
     * Obtains the APK content digest(s) and adds them to the provided {@code
     * sigSchemeApkContentDigests}, returning an {@code ApkSigningBlockUtils.Result} that can be
     * merged with a {@code Result} to notify the client of any errors.
     *
     * <p>Note, this method currently only supports signature scheme V2 and V3; to obtain the
     * content digests for V1 signatures use {@link
     * #getApkContentDigestFromV1SigningScheme(List, DataSource, ApkUtils.ZipSections)}. If a
     * signature scheme version other than V2 or V3 is provided a {@code null} value will be
     * returned.
     */
    private ApkSigningBlockUtils.Result getApkContentDigests(DataSource apk,
            ApkUtils.ZipSections zipSections, Set<Integer> foundApkSigSchemeIds,
            Map<Integer, String> supportedSchemeNames,
            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> sigSchemeApkContentDigests,
            int apkSigSchemeVersion, int minSdkVersion)
            throws IOException, NoSuchAlgorithmException {
        return getApkContentDigests(apk, zipSections, foundApkSigSchemeIds, supportedSchemeNames,
                sigSchemeApkContentDigests, apkSigSchemeVersion, minSdkVersion, mMaxSdkVersion);
    }


    /**
     * Obtains the APK content digest(s) and adds them to the provided {@code
     * sigSchemeApkContentDigests}, returning an {@code ApkSigningBlockUtils.Result} that can be
     * merged with a {@code Result} to notify the client of any errors.
     *
     * <p>Note, this method currently only supports signature scheme V2 and V3; to obtain the
     * content digests for V1 signatures use {@link
     * #getApkContentDigestFromV1SigningScheme(List, DataSource, ApkUtils.ZipSections)}. If a
     * signature scheme version other than V2 or V3 is provided a {@code null} value will be
     * returned.
     */
    private static ApkSigningBlockUtils.Result getApkContentDigests(DataSource apk,
            ApkUtils.ZipSections zipSections, Set<Integer> foundApkSigSchemeIds,
            Map<Integer, String> supportedSchemeNames,
            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> sigSchemeApkContentDigests,
            int apkSigSchemeVersion, int minSdkVersion, int maxSdkVersion)
            throws IOException, NoSuchAlgorithmException {
        if (!(apkSigSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V2
                || apkSigSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V3
                || apkSigSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V31)) {
            return null;
        }
        ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(apkSigSchemeVersion);
        SignatureInfo signatureInfo;
        try {
            int sigSchemeBlockId;
            switch (apkSigSchemeVersion) {
                case VERSION_APK_SIGNATURE_SCHEME_V31:
                    sigSchemeBlockId = V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID;
                    break;
                case VERSION_APK_SIGNATURE_SCHEME_V3:
                    sigSchemeBlockId = V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
                    break;
                default:
                    sigSchemeBlockId =
                        V2SchemeConstants.APK_SIGNATURE_SCHEME_V2_BLOCK_ID;
            }
            signatureInfo = ApkSigningBlockUtils.findSignature(apk, zipSections,
                    sigSchemeBlockId, result);
        } catch (ApkSigningBlockUtils.SignatureNotFoundException e) {
            return null;
        }
        foundApkSigSchemeIds.add(apkSigSchemeVersion);

        Set<ContentDigestAlgorithm> contentDigestsToVerify = new HashSet<>(1);
        if (apkSigSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V2) {
            V2SchemeVerifier.parseSigners(signatureInfo.signatureBlock,
                    contentDigestsToVerify, supportedSchemeNames,
                    foundApkSigSchemeIds, minSdkVersion, maxSdkVersion, result);
        } else {
            V3SchemeVerifier.parseSigners(signatureInfo.signatureBlock,
                    contentDigestsToVerify, result);
        }
        Map<ContentDigestAlgorithm, byte[]> apkContentDigests = new EnumMap<>(
                ContentDigestAlgorithm.class);
        for (ApkSigningBlockUtils.Result.SignerInfo signerInfo : result.signers) {
            for (ApkSigningBlockUtils.Result.SignerInfo.ContentDigest contentDigest :
                    signerInfo.contentDigests) {
                SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.findById(
                        contentDigest.getSignatureAlgorithmId());
                if (signatureAlgorithm == null) {
                    continue;
                }
                apkContentDigests.put(signatureAlgorithm.getContentDigestAlgorithm(),
                        contentDigest.getValue());
            }
        }
        sigSchemeApkContentDigests.put(apkSigSchemeVersion, apkContentDigests);
        return result;
    }

    private static void checkV4Signer(List<Result.V3SchemeSignerInfo> v3Signers,
            List<X509Certificate> v4Certs, byte[] digestFromV4, Result result) {
        if (v3Signers.size() != 1) {
            result.addError(Issue.V4_SIG_MULTIPLE_SIGNERS);
        }

        // Compare certificates.
        checkV4Certificate(v4Certs, v3Signers.get(0).mCerts, result);

        // Compare digests.
        final byte[] digestFromV3 = pickBestDigestForV4(v3Signers.get(0).getContentDigests());
        if (!Arrays.equals(digestFromV4, digestFromV3)) {
            result.addError(Issue.V4_SIG_V2_V3_DIGESTS_MISMATCH, 3, toHex(digestFromV3),
                    toHex(digestFromV4));
        }
    }

    private static void checkV4Certificate(List<X509Certificate> v4Certs,
            List<X509Certificate> v2v3Certs, Result result) {
        try {
            byte[] v4Cert = v4Certs.get(0).getEncoded();
            byte[] cert = v2v3Certs.get(0).getEncoded();
            if (!Arrays.equals(cert, v4Cert)) {
                result.addError(Issue.V4_SIG_V2_V3_SIGNERS_MISMATCH);
            }
        } catch (CertificateEncodingException e) {
            throw new RuntimeException("Failed to encode APK signer cert", e);
        }
    }

    private static byte[] pickBestDigestForV4(
            List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> contentDigests) {
        Map<ContentDigestAlgorithm, byte[]> apkContentDigests = new HashMap<>();
        collectApkContentDigests(contentDigests, apkContentDigests);
        return ApkSigningBlockUtils.pickBestDigestForV4(apkContentDigests);
    }

    private static Map<ContentDigestAlgorithm, byte[]> getApkContentDigestsFromSigningSchemeResult(
            ApkSigningBlockUtils.Result apkSigningSchemeResult) {
        Map<ContentDigestAlgorithm, byte[]> apkContentDigests = new HashMap<>();
        for (ApkSigningBlockUtils.Result.SignerInfo signerInfo : apkSigningSchemeResult.signers) {
            collectApkContentDigests(signerInfo.contentDigests, apkContentDigests);
        }
        return apkContentDigests;
    }

    private static Map<ContentDigestAlgorithm, byte[]> getApkContentDigestFromV1SigningScheme(
            List<CentralDirectoryRecord> cdRecords,
            DataSource apk,
            ApkUtils.ZipSections zipSections)
            throws IOException, ApkFormatException {
        CentralDirectoryRecord manifestCdRecord = null;
        Map<ContentDigestAlgorithm, byte[]> v1ContentDigest = new EnumMap<>(
                ContentDigestAlgorithm.class);
        for (CentralDirectoryRecord cdRecord : cdRecords) {
            if (MANIFEST_ENTRY_NAME.equals(cdRecord.getName())) {
                manifestCdRecord = cdRecord;
                break;
            }
        }
        if (manifestCdRecord == null) {
            // No JAR signing manifest file found. For SourceStamp verification, returning an empty
            // digest is enough since this would affect the final digest signed by the stamp, and
            // thus an empty digest will invalidate that signature.
            return v1ContentDigest;
        }
        try {
            byte[] manifestBytes =
                    LocalFileRecord.getUncompressedData(
                            apk, manifestCdRecord, zipSections.getZipCentralDirectoryOffset());
            v1ContentDigest.put(
                    ContentDigestAlgorithm.SHA256, computeSha256DigestBytes(manifestBytes));
            return v1ContentDigest;
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Failed to read APK", e);
        }
    }

    private static void collectApkContentDigests(
            List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> contentDigests,
            Map<ContentDigestAlgorithm, byte[]> apkContentDigests) {
        for (ApkSigningBlockUtils.Result.SignerInfo.ContentDigest contentDigest : contentDigests) {
            SignatureAlgorithm signatureAlgorithm =
                    SignatureAlgorithm.findById(contentDigest.getSignatureAlgorithmId());
            if (signatureAlgorithm == null) {
                continue;
            }
            ContentDigestAlgorithm contentDigestAlgorithm =
                    signatureAlgorithm.getContentDigestAlgorithm();
            apkContentDigests.put(contentDigestAlgorithm, contentDigest.getValue());
        }

    }

    private static ByteBuffer getAndroidManifestFromApk(
            DataSource apk, ApkUtils.ZipSections zipSections)
            throws IOException, ApkFormatException {
        List<CentralDirectoryRecord> cdRecords =
                V1SchemeVerifier.parseZipCentralDirectory(apk, zipSections);
        try {
            return ApkSigner.getAndroidManifestFromApk(
                    cdRecords,
                    apk.slice(0, zipSections.getZipCentralDirectoryOffset()));
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Failed to read AndroidManifest.xml", e);
        }
    }

    private static int getMinimumSignatureSchemeVersionForTargetSdk(int targetSdkVersion) {
        if (targetSdkVersion >= AndroidSdkVersion.R) {
            return VERSION_APK_SIGNATURE_SCHEME_V2;
        }
        return VERSION_JAR_SIGNATURE_SCHEME;
    }

    /**
     * Result of verifying an APKs signatures. The APK can be considered verified iff
     * {@link #isVerified()} returns {@code true}.
     */
    public static class Result {
        private final List<IssueWithParams> mErrors = new ArrayList<>();
        private final List<IssueWithParams> mWarnings = new ArrayList<>();
        private final List<X509Certificate> mSignerCerts = new ArrayList<>();
        private final List<V1SchemeSignerInfo> mV1SchemeSigners = new ArrayList<>();
        private final List<V1SchemeSignerInfo> mV1SchemeIgnoredSigners = new ArrayList<>();
        private final List<V2SchemeSignerInfo> mV2SchemeSigners = new ArrayList<>();
        private final List<V3SchemeSignerInfo> mV3SchemeSigners = new ArrayList<>();
        private final List<V3SchemeSignerInfo> mV31SchemeSigners = new ArrayList<>();
        private final List<V4SchemeSignerInfo> mV4SchemeSigners = new ArrayList<>();
        private SourceStampInfo mSourceStampInfo;

        private boolean mVerified;
        private boolean mVerifiedUsingV1Scheme;
        private boolean mVerifiedUsingV2Scheme;
        private boolean mVerifiedUsingV3Scheme;
        private boolean mVerifiedUsingV31Scheme;
        private boolean mVerifiedUsingV4Scheme;
        private boolean mSourceStampVerified;
        private boolean mWarningsAsErrors;
        private SigningCertificateLineage mSigningCertificateLineage;

        /**
         * Returns {@code true} if the APK's signatures verified.
         */
        public boolean isVerified() {
            return mVerified;
        }

        private void setVerified() {
            mVerified = true;
        }

        /**
         * Returns {@code true} if the APK's JAR signatures verified.
         */
        public boolean isVerifiedUsingV1Scheme() {
            return mVerifiedUsingV1Scheme;
        }

        /**
         * Returns {@code true} if the APK's APK Signature Scheme v2 signatures verified.
         */
        public boolean isVerifiedUsingV2Scheme() {
            return mVerifiedUsingV2Scheme;
        }

        /**
         * Returns {@code true} if the APK's APK Signature Scheme v3 signature verified.
         */
        public boolean isVerifiedUsingV3Scheme() {
            return mVerifiedUsingV3Scheme;
        }

        /**
         * Returns {@code true} if the APK's APK Signature Scheme v3.1 signature verified.
         */
        public boolean isVerifiedUsingV31Scheme() {
            return mVerifiedUsingV31Scheme;
        }

        /**
         * Returns {@code true} if the APK's APK Signature Scheme v4 signature verified.
         */
        public boolean isVerifiedUsingV4Scheme() {
            return mVerifiedUsingV4Scheme;
        }

        /**
         * Returns {@code true} if the APK's SourceStamp signature verified.
         */
        public boolean isSourceStampVerified() {
            return mSourceStampVerified;
        }

        /**
         * Returns the verified signers' certificates, one per signer.
         */
        public List<X509Certificate> getSignerCertificates() {
            return mSignerCerts;
        }

        private void addSignerCertificate(X509Certificate cert) {
            mSignerCerts.add(cert);
        }

        /**
         * Returns information about JAR signers associated with the APK's signature. These are the
         * signers used by Android.
         *
         * @see #getV1SchemeIgnoredSigners()
         */
        public List<V1SchemeSignerInfo> getV1SchemeSigners() {
            return mV1SchemeSigners;
        }

        /**
         * Returns information about JAR signers ignored by the APK's signature verification
         * process. These signers are ignored by Android. However, each signer's errors or warnings
         * will contain information about why they are ignored.
         *
         * @see #getV1SchemeSigners()
         */
        public List<V1SchemeSignerInfo> getV1SchemeIgnoredSigners() {
            return mV1SchemeIgnoredSigners;
        }

        /**
         * Returns information about APK Signature Scheme v2 signers associated with the APK's
         * signature.
         */
        public List<V2SchemeSignerInfo> getV2SchemeSigners() {
            return mV2SchemeSigners;
        }

        /**
         * Returns information about APK Signature Scheme v3 signers associated with the APK's
         * signature.
         *
         * <note> Multiple signers represent different targeted platform versions, not
         * a signing identity of multiple signers.  APK Signature Scheme v3 only supports single
         * signer identities.</note>
         */
        public List<V3SchemeSignerInfo> getV3SchemeSigners() {
            return mV3SchemeSigners;
        }

        /**
         * Returns information about APK Signature Scheme v3.1 signers associated with the APK's
         * signature.
         *
         * <note> Multiple signers represent different targeted platform versions, not
         * a signing identity of multiple signers.  APK Signature Scheme v3.1 only supports single
         * signer identities.</note>
         */
        public List<V3SchemeSignerInfo> getV31SchemeSigners() {
            return mV31SchemeSigners;
        }

        /**
         * Returns information about APK Signature Scheme v4 signers associated with the APK's
         * signature.
         */
        public List<V4SchemeSignerInfo> getV4SchemeSigners() {
            return mV4SchemeSigners;
        }

        /**
         * Returns information about SourceStamp associated with the APK's signature.
         */
        public SourceStampInfo getSourceStampInfo() {
            return mSourceStampInfo;
        }

        /**
         * Returns the combined SigningCertificateLineage associated with this APK's APK Signature
         * Scheme v3 signing block.
         */
        public SigningCertificateLineage getSigningCertificateLineage() {
            return mSigningCertificateLineage;
        }

        void addError(Issue msg, Object... parameters) {
            mErrors.add(new IssueWithParams(msg, parameters));
        }

        void addWarning(Issue msg, Object... parameters) {
            mWarnings.add(new IssueWithParams(msg, parameters));
        }

        /**
         * Sets whether warnings should be treated as errors.
         */
        void setWarningsAsErrors(boolean value) {
            mWarningsAsErrors = value;
        }

        /**
         * Returns errors encountered while verifying the APK's signatures.
         */
        public List<IssueWithParams> getErrors() {
            if (!mWarningsAsErrors) {
                return mErrors;
            } else {
                List<IssueWithParams> allErrors = new ArrayList<>();
                allErrors.addAll(mErrors);
                allErrors.addAll(mWarnings);
                return allErrors;
            }
        }

        /**
         * Returns warnings encountered while verifying the APK's signatures.
         */
        public List<IssueWithParams> getWarnings() {
            return mWarnings;
        }

        private void mergeFrom(V1SchemeVerifier.Result source) {
            mVerifiedUsingV1Scheme = source.verified;
            mErrors.addAll(source.getErrors());
            mWarnings.addAll(source.getWarnings());
            for (V1SchemeVerifier.Result.SignerInfo signer : source.signers) {
                mV1SchemeSigners.add(new V1SchemeSignerInfo(signer));
            }
            for (V1SchemeVerifier.Result.SignerInfo signer : source.ignoredSigners) {
                mV1SchemeIgnoredSigners.add(new V1SchemeSignerInfo(signer));
            }
        }

        private void mergeFrom(ApkSigResult source) {
            switch (source.signatureSchemeVersion) {
                case VERSION_SOURCE_STAMP:
                    mSourceStampVerified = source.verified;
                    if (!source.mSigners.isEmpty()) {
                        mSourceStampInfo = new SourceStampInfo(source.mSigners.get(0));
                    }
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Unknown ApkSigResult Signing Block Scheme Id "
                                    + source.signatureSchemeVersion);
            }
        }

        private void mergeFrom(ApkSigningBlockUtils.Result source) {
            if (source == null) {
                return;
            }
            if (source.containsErrors()) {
                mErrors.addAll(source.getErrors());
            }
            if (source.containsWarnings()) {
                mWarnings.addAll(source.getWarnings());
            }
            switch (source.signatureSchemeVersion) {
                case VERSION_APK_SIGNATURE_SCHEME_V2:
                    mVerifiedUsingV2Scheme = source.verified;
                    for (ApkSigningBlockUtils.Result.SignerInfo signer : source.signers) {
                        mV2SchemeSigners.add(new V2SchemeSignerInfo(signer));
                    }
                    break;
                case VERSION_APK_SIGNATURE_SCHEME_V3:
                    mVerifiedUsingV3Scheme = source.verified;
                    for (ApkSigningBlockUtils.Result.SignerInfo signer : source.signers) {
                        mV3SchemeSigners.add(new V3SchemeSignerInfo(signer));
                    }
                    // Do not overwrite a previously set lineage from a v3.1 signing block.
                    if (mSigningCertificateLineage == null) {
                        mSigningCertificateLineage = source.signingCertificateLineage;
                    }
                    break;
                case VERSION_APK_SIGNATURE_SCHEME_V31:
                    mVerifiedUsingV31Scheme = source.verified;
                    for (ApkSigningBlockUtils.Result.SignerInfo signer : source.signers) {
                        mV31SchemeSigners.add(new V3SchemeSignerInfo(signer));
                    }
                    mSigningCertificateLineage = source.signingCertificateLineage;
                    break;
                case VERSION_APK_SIGNATURE_SCHEME_V4:
                    mVerifiedUsingV4Scheme = source.verified;
                    for (ApkSigningBlockUtils.Result.SignerInfo signer : source.signers) {
                        mV4SchemeSigners.add(new V4SchemeSignerInfo(signer));
                    }
                    break;
                case VERSION_SOURCE_STAMP:
                    mSourceStampVerified = source.verified;
                    if (!source.signers.isEmpty()) {
                        mSourceStampInfo = new SourceStampInfo(source.signers.get(0));
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Unknown Signing Block Scheme Id");
            }
        }

        /**
         * Returns {@code true} if an error was encountered while verifying the APK. Any error
         * prevents the APK from being considered verified.
         */
        public boolean containsErrors() {
            if (!mErrors.isEmpty()) {
                return true;
            }
            if (mWarningsAsErrors && !mWarnings.isEmpty()) {
                return true;
            }
            if (!mV1SchemeSigners.isEmpty()) {
                for (V1SchemeSignerInfo signer : mV1SchemeSigners) {
                    if (signer.containsErrors()) {
                        return true;
                    }
                    if (mWarningsAsErrors && !signer.getWarnings().isEmpty()) {
                        return true;
                    }
                }
            }
            if (!mV2SchemeSigners.isEmpty()) {
                for (V2SchemeSignerInfo signer : mV2SchemeSigners) {
                    if (signer.containsErrors()) {
                        return true;
                    }
                    if (mWarningsAsErrors && !signer.getWarnings().isEmpty()) {
                        return true;
                    }
                }
            }
            if (!mV3SchemeSigners.isEmpty()) {
                for (V3SchemeSignerInfo signer : mV3SchemeSigners) {
                    if (signer.containsErrors()) {
                        return true;
                    }
                    if (mWarningsAsErrors && !signer.getWarnings().isEmpty()) {
                        return true;
                    }
                }
            }
            if (!mV31SchemeSigners.isEmpty()) {
                for (V3SchemeSignerInfo signer : mV31SchemeSigners) {
                    if (signer.containsErrors()) {
                        return true;
                    }
                    if (mWarningsAsErrors && !signer.getWarnings().isEmpty()) {
                        return true;
                    }
                }
            }
            if (mSourceStampInfo != null) {
                if (mSourceStampInfo.containsErrors()) {
                    return true;
                }
                if (mWarningsAsErrors && !mSourceStampInfo.getWarnings().isEmpty()) {
                    return true;
                }
            }

            return false;
        }

        /**
         * Returns all errors for this result, including any errors from signature scheme signers
         * and the source stamp.
         */
        public List<IssueWithParams> getAllErrors() {
            List<IssueWithParams> errors = new ArrayList<>();
            errors.addAll(mErrors);
            if (mWarningsAsErrors) {
                errors.addAll(mWarnings);
            }
            if (!mV1SchemeSigners.isEmpty()) {
                for (V1SchemeSignerInfo signer : mV1SchemeSigners) {
                    errors.addAll(signer.mErrors);
                    if (mWarningsAsErrors) {
                        errors.addAll(signer.getWarnings());
                    }
                }
            }
            if (!mV2SchemeSigners.isEmpty()) {
                for (V2SchemeSignerInfo signer : mV2SchemeSigners) {
                    errors.addAll(signer.mErrors);
                    if (mWarningsAsErrors) {
                        errors.addAll(signer.getWarnings());
                    }
                }
            }
            if (!mV3SchemeSigners.isEmpty()) {
                for (V3SchemeSignerInfo signer : mV3SchemeSigners) {
                    errors.addAll(signer.mErrors);
                    if (mWarningsAsErrors) {
                        errors.addAll(signer.getWarnings());
                    }
                }
            }
            if (!mV31SchemeSigners.isEmpty()) {
                for (V3SchemeSignerInfo signer : mV31SchemeSigners) {
                    errors.addAll(signer.mErrors);
                    if (mWarningsAsErrors) {
                        errors.addAll(signer.getWarnings());
                    }
                }
            }
            if (mSourceStampInfo != null) {
                errors.addAll(mSourceStampInfo.getErrors());
                if (mWarningsAsErrors) {
                    errors.addAll(mSourceStampInfo.getWarnings());
                }
            }
            return errors;
        }

        /**
         * Information about a JAR signer associated with the APK's signature.
         */
        public static class V1SchemeSignerInfo {
            private final String mName;
            private final List<X509Certificate> mCertChain;
            private final String mSignatureBlockFileName;
            private final String mSignatureFileName;

            private final List<IssueWithParams> mErrors;
            private final List<IssueWithParams> mWarnings;

            private V1SchemeSignerInfo(V1SchemeVerifier.Result.SignerInfo result) {
                mName = result.name;
                mCertChain = result.certChain;
                mSignatureBlockFileName = result.signatureBlockFileName;
                mSignatureFileName = result.signatureFileName;
                mErrors = result.getErrors();
                mWarnings = result.getWarnings();
            }

            /**
             * Returns a user-friendly name of the signer.
             */
            public String getName() {
                return mName;
            }

            /**
             * Returns the name of the JAR entry containing this signer's JAR signature block file.
             */
            public String getSignatureBlockFileName() {
                return mSignatureBlockFileName;
            }

            /**
             * Returns the name of the JAR entry containing this signer's JAR signature file.
             */
            public String getSignatureFileName() {
                return mSignatureFileName;
            }

            /**
             * Returns this signer's signing certificate or {@code null} if not available. The
             * certificate is guaranteed to be available if no errors were encountered during
             * verification (see {@link #containsErrors()}.
             *
             * <p>This certificate contains the signer's public key.
             */
            public X509Certificate getCertificate() {
                return mCertChain.isEmpty() ? null : mCertChain.get(0);
            }

            /**
             * Returns the certificate chain for the signer's public key. The certificate containing
             * the public key is first, followed by the certificate (if any) which issued the
             * signing certificate, and so forth. An empty list may be returned if an error was
             * encountered during verification (see {@link #containsErrors()}).
             */
            public List<X509Certificate> getCertificateChain() {
                return mCertChain;
            }

            /**
             * Returns {@code true} if an error was encountered while verifying this signer's JAR
             * signature. Any error prevents the signer's signature from being considered verified.
             */
            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            /**
             * Returns errors encountered while verifying this signer's JAR signature. Any error
             * prevents the signer's signature from being considered verified.
             */
            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            /**
             * Returns warnings encountered while verifying this signer's JAR signature. Warnings
             * do not prevent the signer's signature from being considered verified.
             */
            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }

            private void addError(Issue msg, Object... parameters) {
                mErrors.add(new IssueWithParams(msg, parameters));
            }
        }

        /**
         * Information about an APK Signature Scheme v2 signer associated with the APK's signature.
         */
        public static class V2SchemeSignerInfo {
            private final int mIndex;
            private final List<X509Certificate> mCerts;

            private final List<IssueWithParams> mErrors;
            private final List<IssueWithParams> mWarnings;
            private final List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest>
                    mContentDigests;

            private V2SchemeSignerInfo(ApkSigningBlockUtils.Result.SignerInfo result) {
                mIndex = result.index;
                mCerts = result.certs;
                mErrors = result.getErrors();
                mWarnings = result.getWarnings();
                mContentDigests = result.contentDigests;
            }

            /**
             * Returns this signer's {@code 0}-based index in the list of signers contained in the
             * APK's APK Signature Scheme v2 signature.
             */
            public int getIndex() {
                return mIndex;
            }

            /**
             * Returns this signer's signing certificate or {@code null} if not available. The
             * certificate is guaranteed to be available if no errors were encountered during
             * verification (see {@link #containsErrors()}.
             *
             * <p>This certificate contains the signer's public key.
             */
            public X509Certificate getCertificate() {
                return mCerts.isEmpty() ? null : mCerts.get(0);
            }

            /**
             * Returns this signer's certificates. The first certificate is for the signer's public
             * key. An empty list may be returned if an error was encountered during verification
             * (see {@link #containsErrors()}).
             */
            public List<X509Certificate> getCertificates() {
                return mCerts;
            }

            private void addError(Issue msg, Object... parameters) {
                mErrors.add(new IssueWithParams(msg, parameters));
            }

            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }

            public List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> getContentDigests() {
                return mContentDigests;
            }
        }

        /**
         * Information about an APK Signature Scheme v3 signer associated with the APK's signature.
         */
        public static class V3SchemeSignerInfo {
            private final int mIndex;
            private final List<X509Certificate> mCerts;

            private final List<IssueWithParams> mErrors;
            private final List<IssueWithParams> mWarnings;
            private final List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest>
                    mContentDigests;
            private final int mMinSdkVersion;
            private final int mMaxSdkVersion;
            private final boolean mRotationTargetsDevRelease;
            private final SigningCertificateLineage mSigningCertificateLineage;

            private V3SchemeSignerInfo(ApkSigningBlockUtils.Result.SignerInfo result) {
                mIndex = result.index;
                mCerts = result.certs;
                mErrors = result.getErrors();
                mWarnings = result.getWarnings();
                mContentDigests = result.contentDigests;
                mMinSdkVersion = result.minSdkVersion;
                mMaxSdkVersion = result.maxSdkVersion;
                mSigningCertificateLineage = result.signingCertificateLineage;
                mRotationTargetsDevRelease = result.additionalAttributes.stream().mapToInt(
                        attribute -> attribute.getId()).anyMatch(
                        attrId -> attrId == V3SchemeConstants.ROTATION_ON_DEV_RELEASE_ATTR_ID);
            }

            /**
             * Returns this signer's {@code 0}-based index in the list of signers contained in the
             * APK's APK Signature Scheme v3 signature.
             */
            public int getIndex() {
                return mIndex;
            }

            /**
             * Returns this signer's signing certificate or {@code null} if not available. The
             * certificate is guaranteed to be available if no errors were encountered during
             * verification (see {@link #containsErrors()}.
             *
             * <p>This certificate contains the signer's public key.
             */
            public X509Certificate getCertificate() {
                return mCerts.isEmpty() ? null : mCerts.get(0);
            }

            /**
             * Returns this signer's certificates. The first certificate is for the signer's public
             * key. An empty list may be returned if an error was encountered during verification
             * (see {@link #containsErrors()}).
             */
            public List<X509Certificate> getCertificates() {
                return mCerts;
            }

            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }

            public List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> getContentDigests() {
                return mContentDigests;
            }

            /**
             * Returns the minimum SDK version on which this signer should be verified.
             */
            public int getMinSdkVersion() {
                return mMinSdkVersion;
            }

            /**
             * Returns the maximum SDK version on which this signer should be verified.
             */
            public int getMaxSdkVersion() {
                return mMaxSdkVersion;
            }

            /**
             * Returns whether rotation is targeting a development release.
             *
             * <p>A development release uses the SDK version of the previously released platform
             * until the SDK of the development release is finalized. To allow rotation to target
             * a development release after T, this attribute must be set to ensure rotation is
             * used on the development release but ignored on the released platform with the same
             * API level.
             */
            public boolean getRotationTargetsDevRelease() {
                return mRotationTargetsDevRelease;
            }

            /**
             * Returns the {@link SigningCertificateLineage} for this signer; when an APK has
             * SDK targeted signing configs, the lineage of each signer could potentially contain
             * a subset of the full signing lineage and / or different capabilities for each signer
             * in the lineage.
             */
            public SigningCertificateLineage getSigningCertificateLineage() {
                return mSigningCertificateLineage;
            }
        }

        /**
         * Information about an APK Signature Scheme V4 signer associated with the APK's
         * signature.
         */
        public static class V4SchemeSignerInfo {
            private final int mIndex;
            private final List<X509Certificate> mCerts;

            private final List<IssueWithParams> mErrors;
            private final List<IssueWithParams> mWarnings;
            private final List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest>
                    mContentDigests;

            private V4SchemeSignerInfo(ApkSigningBlockUtils.Result.SignerInfo result) {
                mIndex = result.index;
                mCerts = result.certs;
                mErrors = result.getErrors();
                mWarnings = result.getWarnings();
                mContentDigests = result.contentDigests;
            }

            /**
             * Returns this signer's {@code 0}-based index in the list of signers contained in the
             * APK's APK Signature Scheme v3 signature.
             */
            public int getIndex() {
                return mIndex;
            }

            /**
             * Returns this signer's signing certificate or {@code null} if not available. The
             * certificate is guaranteed to be available if no errors were encountered during
             * verification (see {@link #containsErrors()}.
             *
             * <p>This certificate contains the signer's public key.
             */
            public X509Certificate getCertificate() {
                return mCerts.isEmpty() ? null : mCerts.get(0);
            }

            /**
             * Returns this signer's certificates. The first certificate is for the signer's public
             * key. An empty list may be returned if an error was encountered during verification
             * (see {@link #containsErrors()}).
             */
            public List<X509Certificate> getCertificates() {
                return mCerts;
            }

            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }

            public List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> getContentDigests() {
                return mContentDigests;
            }
        }

        /**
         * Information about SourceStamp associated with the APK's signature.
         */
        public static class SourceStampInfo {
            public enum SourceStampVerificationStatus {
                /** The stamp is present and was successfully verified. */
                STAMP_VERIFIED,
                /** The stamp is present but failed verification. */
                STAMP_VERIFICATION_FAILED,
                /** The expected cert digest did not match the digest in the APK. */
                CERT_DIGEST_MISMATCH,
                /** The stamp is not present at all. */
                STAMP_MISSING,
                /** The stamp is at least partially present, but was not able to be verified. */
                STAMP_NOT_VERIFIED,
                /** The stamp was not able to be verified due to an unexpected error. */
                VERIFICATION_ERROR
            }

            private final List<X509Certificate> mCertificates;
            private final List<X509Certificate> mCertificateLineage;

            private final List<IssueWithParams> mErrors;
            private final List<IssueWithParams> mWarnings;
            private final List<IssueWithParams> mInfoMessages;

            private final SourceStampVerificationStatus mSourceStampVerificationStatus;

            private final long mTimestamp;

            private SourceStampInfo(ApkSignerInfo result) {
                mCertificates = result.certs;
                mCertificateLineage = result.certificateLineage;
                mErrors = ApkVerificationIssueAdapter.getIssuesFromVerificationIssues(
                        result.getErrors());
                mWarnings = ApkVerificationIssueAdapter.getIssuesFromVerificationIssues(
                        result.getWarnings());
                mInfoMessages = ApkVerificationIssueAdapter.getIssuesFromVerificationIssues(
                        result.getInfoMessages());
                if (mErrors.isEmpty() && mWarnings.isEmpty()) {
                    mSourceStampVerificationStatus = SourceStampVerificationStatus.STAMP_VERIFIED;
                } else {
                    mSourceStampVerificationStatus =
                            SourceStampVerificationStatus.STAMP_VERIFICATION_FAILED;
                }
                mTimestamp = result.timestamp;
            }

            SourceStampInfo(SourceStampVerificationStatus sourceStampVerificationStatus) {
                mCertificates = Collections.emptyList();
                mCertificateLineage = Collections.emptyList();
                mErrors = Collections.emptyList();
                mWarnings = Collections.emptyList();
                mInfoMessages = Collections.emptyList();
                mSourceStampVerificationStatus = sourceStampVerificationStatus;
                mTimestamp = 0;
            }

            /**
             * Returns the SourceStamp's signing certificate or {@code null} if not available. The
             * certificate is guaranteed to be available if no errors were encountered during
             * verification (see {@link #containsErrors()}.
             *
             * <p>This certificate contains the SourceStamp's public key.
             */
            public X509Certificate getCertificate() {
                return mCertificates.isEmpty() ? null : mCertificates.get(0);
            }

            /**
             * Returns a list containing all of the certificates in the stamp certificate lineage.
             */
            public List<X509Certificate> getCertificatesInLineage() {
                return mCertificateLineage;
            }

            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            /**
             * Returns {@code true} if any info messages were encountered during verification of
             * this source stamp.
             */
            public boolean containsInfoMessages() {
                return !mInfoMessages.isEmpty();
            }

            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }

            /**
             * Returns a {@code List} of {@link IssueWithParams} representing info messages
             * that were encountered during verification of the source stamp.
             */
            public List<IssueWithParams> getInfoMessages() {
                return mInfoMessages;
            }

            /**
             * Returns the reason for any source stamp verification failures, or {@code
             * STAMP_VERIFIED} if the source stamp was successfully verified.
             */
            public SourceStampVerificationStatus getSourceStampVerificationStatus() {
                return mSourceStampVerificationStatus;
            }

            /**
             * Returns the epoch timestamp in seconds representing the time this source stamp block
             * was signed, or 0 if the timestamp is not available.
             */
            public long getTimestampEpochSeconds() {
                return mTimestamp;
            }
        }
    }

    /**
     * Error or warning encountered while verifying an APK's signatures.
     */
    public enum Issue {

        /**
         * APK is not JAR-signed.
         */
        JAR_SIG_NO_SIGNATURES("No JAR signatures"),

        /**
         * APK signature scheme v1 has exceeded the maximum number of jar signers.
         * <ul>
         * <li>Parameter 1: maximum allowed signers ({@code Integer})</li>
         * <li>Parameter 2: total number of signers ({@code Integer})</li>
         * </ul>
         */
        JAR_SIG_MAX_SIGNATURES_EXCEEDED(
                "APK Signature Scheme v1 only supports a maximum of %1$d signers, found %2$d"),

        /**
         * APK does not contain any entries covered by JAR signatures.
         */
        JAR_SIG_NO_SIGNED_ZIP_ENTRIES("No JAR entries covered by JAR signatures"),

        /**
         * APK contains multiple entries with the same name.
         *
         * <ul>
         * <li>Parameter 1: name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_DUPLICATE_ZIP_ENTRY("Duplicate entry: %1$s"),

        /**
         * JAR manifest contains a section with a duplicate name.
         *
         * <ul>
         * <li>Parameter 1: section name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_DUPLICATE_MANIFEST_SECTION("Duplicate section in META-INF/MANIFEST.MF: %1$s"),

        /**
         * JAR manifest contains a section without a name.
         *
         * <ul>
         * <li>Parameter 1: section index (1-based) ({@code Integer})</li>
         * </ul>
         */
        JAR_SIG_UNNNAMED_MANIFEST_SECTION(
                "Malformed META-INF/MANIFEST.MF: invidual section #%1$d does not have a name"),

        /**
         * JAR signature file contains a section without a name.
         *
         * <ul>
         * <li>Parameter 1: signature file name ({@code String})</li>
         * <li>Parameter 2: section index (1-based) ({@code Integer})</li>
         * </ul>
         */
        JAR_SIG_UNNNAMED_SIG_FILE_SECTION(
                "Malformed %1$s: invidual section #%2$d does not have a name"),

        /** APK is missing the JAR manifest entry (META-INF/MANIFEST.MF). */
        JAR_SIG_NO_MANIFEST("Missing META-INF/MANIFEST.MF"),

        /**
         * JAR manifest references an entry which is not there in the APK.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_MISSING_ZIP_ENTRY_REFERENCED_IN_MANIFEST(
                "%1$s entry referenced by META-INF/MANIFEST.MF not found in the APK"),

        /**
         * JAR manifest does not list a digest for the specified entry.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_MANIFEST("No digest for %1$s in META-INF/MANIFEST.MF"),

        /**
         * JAR signature does not list a digest for the specified entry.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * <li>Parameter 2: signature file name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_SIG_FILE("No digest for %1$s in %2$s"),

        /**
         * The specified JAR entry is not covered by JAR signature.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_ZIP_ENTRY_NOT_SIGNED("%1$s entry not signed"),

        /**
         * JAR signature uses different set of signers to protect the two specified ZIP entries.
         *
         * <ul>
         * <li>Parameter 1: first entry name ({@code String})</li>
         * <li>Parameter 2: first entry signer names ({@code List<String>})</li>
         * <li>Parameter 3: second entry name ({@code String})</li>
         * <li>Parameter 4: second entry signer names ({@code List<String>})</li>
         * </ul>
         */
        JAR_SIG_ZIP_ENTRY_SIGNERS_MISMATCH(
                "Entries %1$s and %3$s are signed with different sets of signers"
                        + " : <%2$s> vs <%4$s>"),

        /**
         * Digest of the specified ZIP entry's data does not match the digest expected by the JAR
         * signature.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * <li>Parameter 2: digest algorithm (e.g., SHA-256) ({@code String})</li>
         * <li>Parameter 3: name of the entry in which the expected digest is specified
         *     ({@code String})</li>
         * <li>Parameter 4: base64-encoded actual digest ({@code String})</li>
         * <li>Parameter 5: base64-encoded expected digest ({@code String})</li>
         * </ul>
         */
        JAR_SIG_ZIP_ENTRY_DIGEST_DID_NOT_VERIFY(
                "%2$s digest of %1$s does not match the digest specified in %3$s"
                        + ". Expected: <%5$s>, actual: <%4$s>"),

        /**
         * Digest of the JAR manifest main section did not verify.
         *
         * <ul>
         * <li>Parameter 1: digest algorithm (e.g., SHA-256) ({@code String})</li>
         * <li>Parameter 2: name of the entry in which the expected digest is specified
         *     ({@code String})</li>
         * <li>Parameter 3: base64-encoded actual digest ({@code String})</li>
         * <li>Parameter 4: base64-encoded expected digest ({@code String})</li>
         * </ul>
         */
        JAR_SIG_MANIFEST_MAIN_SECTION_DIGEST_DID_NOT_VERIFY(
                "%1$s digest of META-INF/MANIFEST.MF main section does not match the digest"
                        + " specified in %2$s. Expected: <%4$s>, actual: <%3$s>"),

        /**
         * Digest of the specified JAR manifest section does not match the digest expected by the
         * JAR signature.
         *
         * <ul>
         * <li>Parameter 1: section name ({@code String})</li>
         * <li>Parameter 2: digest algorithm (e.g., SHA-256) ({@code String})</li>
         * <li>Parameter 3: name of the signature file in which the expected digest is specified
         *     ({@code String})</li>
         * <li>Parameter 4: base64-encoded actual digest ({@code String})</li>
         * <li>Parameter 5: base64-encoded expected digest ({@code String})</li>
         * </ul>
         */
        JAR_SIG_MANIFEST_SECTION_DIGEST_DID_NOT_VERIFY(
                "%2$s digest of META-INF/MANIFEST.MF section for %1$s does not match the digest"
                        + " specified in %3$s. Expected: <%5$s>, actual: <%4$s>"),

        /**
         * JAR signature file does not contain the whole-file digest of the JAR manifest file. The
         * digest speeds up verification of JAR signature.
         *
         * <ul>
         * <li>Parameter 1: name of the signature file ({@code String})</li>
         * </ul>
         */
        JAR_SIG_NO_MANIFEST_DIGEST_IN_SIG_FILE(
                "%1$s does not specify digest of META-INF/MANIFEST.MF"
                        + ". This slows down verification."),

        /**
         * APK is signed using APK Signature Scheme v2 or newer, but JAR signature file does not
         * contain protections against stripping of these newer scheme signatures.
         *
         * <ul>
         * <li>Parameter 1: name of the signature file ({@code String})</li>
         * </ul>
         */
        JAR_SIG_NO_APK_SIG_STRIP_PROTECTION(
                "APK is signed using APK Signature Scheme v2 but these signatures may be stripped"
                        + " without being detected because %1$s does not contain anti-stripping"
                        + " protections."),

        /**
         * JAR signature of the signer is missing a file/entry.
         *
         * <ul>
         * <li>Parameter 1: name of the encountered file ({@code String})</li>
         * <li>Parameter 2: name of the missing file ({@code String})</li>
         * </ul>
         */
        JAR_SIG_MISSING_FILE("Partial JAR signature. Found: %1$s, missing: %2$s"),

        /**
         * An exception was encountered while verifying JAR signature contained in a signature block
         * against the signature file.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * <li>Parameter 2: name of the signature file ({@code String})</li>
         * <li>Parameter 3: exception ({@code Throwable})</li>
         * </ul>
         */
        JAR_SIG_VERIFY_EXCEPTION("Failed to verify JAR signature %1$s against %2$s: %3$s"),

        /**
         * JAR signature contains unsupported digest algorithm.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * <li>Parameter 2: digest algorithm OID ({@code String})</li>
         * <li>Parameter 3: signature algorithm OID ({@code String})</li>
         * <li>Parameter 4: API Levels on which this combination of algorithms is not supported
         *     ({@code String})</li>
         * <li>Parameter 5: user-friendly variant of digest algorithm ({@code String})</li>
         * <li>Parameter 6: user-friendly variant of signature algorithm ({@code String})</li>
         * </ul>
         */
        JAR_SIG_UNSUPPORTED_SIG_ALG(
                "JAR signature %1$s uses digest algorithm %5$s and signature algorithm %6$s which"
                        + " is not supported on API Level(s) %4$s for which this APK is being"
                        + " verified"),

        /**
         * An exception was encountered while parsing JAR signature contained in a signature block.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * <li>Parameter 2: exception ({@code Throwable})</li>
         * </ul>
         */
        JAR_SIG_PARSE_EXCEPTION("Failed to parse JAR signature %1$s: %2$s"),

        /**
         * An exception was encountered while parsing a certificate contained in the JAR signature
         * block.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * <li>Parameter 2: exception ({@code Throwable})</li>
         * </ul>
         */
        JAR_SIG_MALFORMED_CERTIFICATE("Malformed certificate in JAR signature %1$s: %2$s"),

        /**
         * JAR signature contained in a signature block file did not verify against the signature
         * file.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * <li>Parameter 2: name of the signature file ({@code String})</li>
         * </ul>
         */
        JAR_SIG_DID_NOT_VERIFY("JAR signature %1$s did not verify against %2$s"),

        /**
         * JAR signature contains no verified signers.
         *
         * <ul>
         * <li>Parameter 1: name of the signature block file ({@code String})</li>
         * </ul>
         */
        JAR_SIG_NO_SIGNERS("JAR signature %1$s contains no signers"),

        /**
         * JAR signature file contains a section with a duplicate name.
         *
         * <ul>
         * <li>Parameter 1: signature file name ({@code String})</li>
         * <li>Parameter 1: section name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_DUPLICATE_SIG_FILE_SECTION("Duplicate section in %1$s: %2$s"),

        /**
         * JAR signature file's main section doesn't contain the mandatory Signature-Version
         * attribute.
         *
         * <ul>
         * <li>Parameter 1: signature file name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_MISSING_VERSION_ATTR_IN_SIG_FILE(
                "Malformed %1$s: missing Signature-Version attribute"),

        /**
         * JAR signature file references an unknown APK signature scheme ID.
         *
         * <ul>
         * <li>Parameter 1: name of the signature file ({@code String})</li>
         * <li>Parameter 2: unknown APK signature scheme ID ({@code} Integer)</li>
         * </ul>
         */
        JAR_SIG_UNKNOWN_APK_SIG_SCHEME_ID(
                "JAR signature %1$s references unknown APK signature scheme ID: %2$d"),

        /**
         * JAR signature file indicates that the APK is supposed to be signed with a supported APK
         * signature scheme (in addition to the JAR signature) but no such signature was found in
         * the APK.
         *
         * <ul>
         * <li>Parameter 1: name of the signature file ({@code String})</li>
         * <li>Parameter 2: APK signature scheme ID ({@code} Integer)</li>
         * <li>Parameter 3: APK signature scheme English name ({@code} String)</li>
         * </ul>
         */
        JAR_SIG_MISSING_APK_SIG_REFERENCED(
                "JAR signature %1$s indicates the APK is signed using %3$s but no such signature"
                        + " was found. Signature stripped?"),

        /**
         * JAR entry is not covered by signature and thus unauthorized modifications to its contents
         * will not be detected.
         *
         * <ul>
         * <li>Parameter 1: entry name ({@code String})</li>
         * </ul>
         */
        JAR_SIG_UNPROTECTED_ZIP_ENTRY(
                "%1$s not protected by signature. Unauthorized modifications to this JAR entry"
                        + " will not be detected. Delete or move the entry outside of META-INF/."),

        /**
         * APK which is both JAR-signed and signed using APK Signature Scheme v2 contains an APK
         * Signature Scheme v2 signature from this signer, but does not contain a JAR signature
         * from this signer.
         */
        JAR_SIG_MISSING("No JAR signature from this signer"),

        /**
         * APK is targeting a sandbox version which requires APK Signature Scheme v2 signature but
         * no such signature was found.
         *
         * <ul>
         * <li>Parameter 1: target sandbox version ({@code Integer})</li>
         * </ul>
         */
        NO_SIG_FOR_TARGET_SANDBOX_VERSION(
                "Missing APK Signature Scheme v2 signature required for target sandbox version"
                        + " %1$d"),

        /**
         * APK is targeting an SDK version that requires a minimum signature scheme version, but the
         * APK is not signed with that version or later.
         *
         * <ul>
         *     <li>Parameter 1: target SDK Version (@code Integer})</li>
         *     <li>Parameter 2: minimum signature scheme version ((@code Integer})</li>
         * </ul>
         */
        MIN_SIG_SCHEME_FOR_TARGET_SDK_NOT_MET(
                "Target SDK version %1$d requires a minimum of signature scheme v%2$d; the APK is"
                        + " not signed with this or a later signature scheme"),

        /**
         * APK which is both JAR-signed and signed using APK Signature Scheme v2 contains a JAR
         * signature from this signer, but does not contain an APK Signature Scheme v2 signature
         * from this signer.
         */
        V2_SIG_MISSING("No APK Signature Scheme v2 signature from this signer"),

        /**
         * Failed to parse the list of signers contained in the APK Signature Scheme v2 signature.
         */
        V2_SIG_MALFORMED_SIGNERS("Malformed list of signers"),

        /**
         * Failed to parse this signer's signer block contained in the APK Signature Scheme v2
         * signature.
         */
        V2_SIG_MALFORMED_SIGNER("Malformed signer block"),

        /**
         * Public key embedded in the APK Signature Scheme v2 signature of this signer could not be
         * parsed.
         *
         * <ul>
         * <li>Parameter 1: error details ({@code Throwable})</li>
         * </ul>
         */
        V2_SIG_MALFORMED_PUBLIC_KEY("Malformed public key: %1$s"),

        /**
         * This APK Signature Scheme v2 signer's certificate could not be parsed.
         *
         * <ul>
         * <li>Parameter 1: index ({@code 0}-based) of the certificate in the signer's list of
         *     certificates ({@code Integer})</li>
         * <li>Parameter 2: sequence number ({@code 1}-based) of the certificate in the signer's
         *     list of certificates ({@code Integer})</li>
         * <li>Parameter 3: error details ({@code Throwable})</li>
         * </ul>
         */
        V2_SIG_MALFORMED_CERTIFICATE("Malformed certificate #%2$d: %3$s"),

        /**
         * Failed to parse this signer's signature record contained in the APK Signature Scheme v2
         * signature.
         *
         * <ul>
         * <li>Parameter 1: record number (first record is {@code 1}) ({@code Integer})</li>
         * </ul>
         */
        V2_SIG_MALFORMED_SIGNATURE("Malformed APK Signature Scheme v2 signature record #%1$d"),

        /**
         * Failed to parse this signer's digest record contained in the APK Signature Scheme v2
         * signature.
         *
         * <ul>
         * <li>Parameter 1: record number (first record is {@code 1}) ({@code Integer})</li>
         * </ul>
         */
        V2_SIG_MALFORMED_DIGEST("Malformed APK Signature Scheme v2 digest record #%1$d"),

        /**
         * This APK Signature Scheme v2 signer contains a malformed additional attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute number (first attribute is {@code 1}) {@code Integer})</li>
         * </ul>
         */
        V2_SIG_MALFORMED_ADDITIONAL_ATTRIBUTE("Malformed additional attribute #%1$d"),

        /**
         * APK Signature Scheme v2 signature references an unknown APK signature scheme ID.
         *
         * <ul>
         * <li>Parameter 1: signer index ({@code Integer})</li>
         * <li>Parameter 2: unknown APK signature scheme ID ({@code} Integer)</li>
         * </ul>
         */
        V2_SIG_UNKNOWN_APK_SIG_SCHEME_ID(
                "APK Signature Scheme v2 signer: %1$s references unknown APK signature scheme ID: "
                        + "%2$d"),

        /**
         * APK Signature Scheme v2 signature indicates that the APK is supposed to be signed with a
         * supported APK signature scheme (in addition to the v2 signature) but no such signature
         * was found in the APK.
         *
         * <ul>
         * <li>Parameter 1: signer index ({@code Integer})</li>
         * <li>Parameter 2: APK signature scheme English name ({@code} String)</li>
         * </ul>
         */
        V2_SIG_MISSING_APK_SIG_REFERENCED(
                "APK Signature Scheme v2 signature %1$s indicates the APK is signed using %2$s but "
                        + "no such signature was found. Signature stripped?"),

        /**
         * APK signature scheme v2 has exceeded the maximum number of signers.
         * <ul>
         * <li>Parameter 1: maximum allowed signers ({@code Integer})</li>
         * <li>Parameter 2: total number of signers ({@code Integer})</li>
         * </ul>
         */
        V2_SIG_MAX_SIGNATURES_EXCEEDED(
                "APK Signature Scheme V2 only supports a maximum of %1$d signers, found %2$d"),

        /**
         * APK Signature Scheme v2 signature contains no signers.
         */
        V2_SIG_NO_SIGNERS("No signers in APK Signature Scheme v2 signature"),

        /**
         * This APK Signature Scheme v2 signer contains a signature produced using an unknown
         * algorithm.
         *
         * <ul>
         * <li>Parameter 1: algorithm ID ({@code Integer})</li>
         * </ul>
         */
        V2_SIG_UNKNOWN_SIG_ALGORITHM("Unknown signature algorithm: %1$#x"),

        /**
         * This APK Signature Scheme v2 signer contains an unknown additional attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute ID ({@code Integer})</li>
         * </ul>
         */
        V2_SIG_UNKNOWN_ADDITIONAL_ATTRIBUTE("Unknown additional attribute: ID %1$#x"),

        /**
         * An exception was encountered while verifying APK Signature Scheme v2 signature of this
         * signer.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * <li>Parameter 2: exception ({@code Throwable})</li>
         * </ul>
         */
        V2_SIG_VERIFY_EXCEPTION("Failed to verify %1$s signature: %2$s"),

        /**
         * APK Signature Scheme v2 signature over this signer's signed-data block did not verify.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * </ul>
         */
        V2_SIG_DID_NOT_VERIFY("%1$s signature over signed-data did not verify"),

        /**
         * This APK Signature Scheme v2 signer offers no signatures.
         */
        V2_SIG_NO_SIGNATURES("No signatures"),

        /**
         * This APK Signature Scheme v2 signer offers signatures but none of them are supported.
         */
        V2_SIG_NO_SUPPORTED_SIGNATURES("No supported signatures: %1$s"),

        /**
         * This APK Signature Scheme v2 signer offers no certificates.
         */
        V2_SIG_NO_CERTIFICATES("No certificates"),

        /**
         * This APK Signature Scheme v2 signer's public key listed in the signer's certificate does
         * not match the public key listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: hex-encoded public key from certificate ({@code String})</li>
         * <li>Parameter 2: hex-encoded public key from signatures record ({@code String})</li>
         * </ul>
         */
        V2_SIG_PUBLIC_KEY_MISMATCH_BETWEEN_CERTIFICATE_AND_SIGNATURES_RECORD(
                "Public key mismatch between certificate and signature record: <%1$s> vs <%2$s>"),

        /**
         * This APK Signature Scheme v2 signer's signature algorithms listed in the signatures
         * record do not match the signature algorithms listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: signature algorithms from signatures record ({@code List<Integer>})</li>
         * <li>Parameter 2: signature algorithms from digests record ({@code List<Integer>})</li>
         * </ul>
         */
        V2_SIG_SIG_ALG_MISMATCH_BETWEEN_SIGNATURES_AND_DIGESTS_RECORDS(
                "Signature algorithms mismatch between signatures and digests records"
                        + ": %1$s vs %2$s"),

        /**
         * The APK's digest does not match the digest contained in the APK Signature Scheme v2
         * signature.
         *
         * <ul>
         * <li>Parameter 1: content digest algorithm ({@link ContentDigestAlgorithm})</li>
         * <li>Parameter 2: hex-encoded expected digest of the APK ({@code String})</li>
         * <li>Parameter 3: hex-encoded actual digest of the APK ({@code String})</li>
         * </ul>
         */
        V2_SIG_APK_DIGEST_DID_NOT_VERIFY(
                "APK integrity check failed. %1$s digest mismatch."
                        + " Expected: <%2$s>, actual: <%3$s>"),

        /**
         * Failed to parse the list of signers contained in the APK Signature Scheme v3 signature.
         */
        V3_SIG_MALFORMED_SIGNERS("Malformed list of signers"),

        /**
         * Failed to parse this signer's signer block contained in the APK Signature Scheme v3
         * signature.
         */
        V3_SIG_MALFORMED_SIGNER("Malformed signer block"),

        /**
         * Public key embedded in the APK Signature Scheme v3 signature of this signer could not be
         * parsed.
         *
         * <ul>
         * <li>Parameter 1: error details ({@code Throwable})</li>
         * </ul>
         */
        V3_SIG_MALFORMED_PUBLIC_KEY("Malformed public key: %1$s"),

        /**
         * This APK Signature Scheme v3 signer's certificate could not be parsed.
         *
         * <ul>
         * <li>Parameter 1: index ({@code 0}-based) of the certificate in the signer's list of
         *     certificates ({@code Integer})</li>
         * <li>Parameter 2: sequence number ({@code 1}-based) of the certificate in the signer's
         *     list of certificates ({@code Integer})</li>
         * <li>Parameter 3: error details ({@code Throwable})</li>
         * </ul>
         */
        V3_SIG_MALFORMED_CERTIFICATE("Malformed certificate #%2$d: %3$s"),

        /**
         * Failed to parse this signer's signature record contained in the APK Signature Scheme v3
         * signature.
         *
         * <ul>
         * <li>Parameter 1: record number (first record is {@code 1}) ({@code Integer})</li>
         * </ul>
         */
        V3_SIG_MALFORMED_SIGNATURE("Malformed APK Signature Scheme v3 signature record #%1$d"),

        /**
         * Failed to parse this signer's digest record contained in the APK Signature Scheme v3
         * signature.
         *
         * <ul>
         * <li>Parameter 1: record number (first record is {@code 1}) ({@code Integer})</li>
         * </ul>
         */
        V3_SIG_MALFORMED_DIGEST("Malformed APK Signature Scheme v3 digest record #%1$d"),

        /**
         * This APK Signature Scheme v3 signer contains a malformed additional attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute number (first attribute is {@code 1}) {@code Integer})</li>
         * </ul>
         */
        V3_SIG_MALFORMED_ADDITIONAL_ATTRIBUTE("Malformed additional attribute #%1$d"),

        /**
         * APK Signature Scheme v3 signature contains no signers.
         */
        V3_SIG_NO_SIGNERS("No signers in APK Signature Scheme v3 signature"),

        /**
         * APK Signature Scheme v3 signature contains multiple signers (only one allowed per
         * platform version).
         */
        V3_SIG_MULTIPLE_SIGNERS("Multiple APK Signature Scheme v3 signatures found for a single "
                + " platform version."),

        /**
         * APK Signature Scheme v3 signature found, but multiple v1 and/or multiple v2 signers
         * found, where only one may be used with APK Signature Scheme v3
         */
        V3_SIG_MULTIPLE_PAST_SIGNERS("Multiple signatures found for pre-v3 signing with an APK "
                + " Signature Scheme v3 signer.  Only one allowed."),

        /**
         * APK Signature Scheme v3 signature found, but its signer doesn't match the v1/v2 signers,
         * or have them as the root of its signing certificate history
         */
        V3_SIG_PAST_SIGNERS_MISMATCH(
                "v3 signer differs from v1/v2 signer without proper signing certificate lineage."),

        /**
         * This APK Signature Scheme v3 signer contains a signature produced using an unknown
         * algorithm.
         *
         * <ul>
         * <li>Parameter 1: algorithm ID ({@code Integer})</li>
         * </ul>
         */
        V3_SIG_UNKNOWN_SIG_ALGORITHM("Unknown signature algorithm: %1$#x"),

        /**
         * This APK Signature Scheme v3 signer contains an unknown additional attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute ID ({@code Integer})</li>
         * </ul>
         */
        V3_SIG_UNKNOWN_ADDITIONAL_ATTRIBUTE("Unknown additional attribute: ID %1$#x"),

        /**
         * An exception was encountered while verifying APK Signature Scheme v3 signature of this
         * signer.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * <li>Parameter 2: exception ({@code Throwable})</li>
         * </ul>
         */
        V3_SIG_VERIFY_EXCEPTION("Failed to verify %1$s signature: %2$s"),

        /**
         * The APK Signature Scheme v3 signer contained an invalid value for either min or max SDK
         * versions.
         *
         * <ul>
         * <li>Parameter 1: minSdkVersion ({@code Integer})
         * <li>Parameter 2: maxSdkVersion ({@code Integer})
         * </ul>
         */
        V3_SIG_INVALID_SDK_VERSIONS("Invalid SDK Version parameter(s) encountered in APK Signature "
                + "scheme v3 signature: minSdkVersion %1$s maxSdkVersion: %2$s"),

        /**
         * APK Signature Scheme v3 signature over this signer's signed-data block did not verify.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * </ul>
         */
        V3_SIG_DID_NOT_VERIFY("%1$s signature over signed-data did not verify"),

        /**
         * This APK Signature Scheme v3 signer offers no signatures.
         */
        V3_SIG_NO_SIGNATURES("No signatures"),

        /**
         * This APK Signature Scheme v3 signer offers signatures but none of them are supported.
         */
        V3_SIG_NO_SUPPORTED_SIGNATURES("No supported signatures"),

        /**
         * This APK Signature Scheme v3 signer offers no certificates.
         */
        V3_SIG_NO_CERTIFICATES("No certificates"),

        /**
         * This APK Signature Scheme v3 signer's minSdkVersion listed in the signer's signed data
         * does not match the minSdkVersion listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: minSdkVersion in signature record ({@code Integer}) </li>
         * <li>Parameter 2: minSdkVersion in signed data ({@code Integer}) </li>
         * </ul>
         */
        V3_MIN_SDK_VERSION_MISMATCH_BETWEEN_SIGNER_AND_SIGNED_DATA_RECORD(
                "minSdkVersion mismatch between signed data and signature record:"
                        + " <%1$s> vs <%2$s>"),

        /**
         * This APK Signature Scheme v3 signer's maxSdkVersion listed in the signer's signed data
         * does not match the maxSdkVersion listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: maxSdkVersion in signature record ({@code Integer}) </li>
         * <li>Parameter 2: maxSdkVersion in signed data ({@code Integer}) </li>
         * </ul>
         */
        V3_MAX_SDK_VERSION_MISMATCH_BETWEEN_SIGNER_AND_SIGNED_DATA_RECORD(
                "maxSdkVersion mismatch between signed data and signature record:"
                        + " <%1$s> vs <%2$s>"),

        /**
         * This APK Signature Scheme v3 signer's public key listed in the signer's certificate does
         * not match the public key listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: hex-encoded public key from certificate ({@code String})</li>
         * <li>Parameter 2: hex-encoded public key from signatures record ({@code String})</li>
         * </ul>
         */
        V3_SIG_PUBLIC_KEY_MISMATCH_BETWEEN_CERTIFICATE_AND_SIGNATURES_RECORD(
                "Public key mismatch between certificate and signature record: <%1$s> vs <%2$s>"),

        /**
         * This APK Signature Scheme v3 signer's signature algorithms listed in the signatures
         * record do not match the signature algorithms listed in the signatures record.
         *
         * <ul>
         * <li>Parameter 1: signature algorithms from signatures record ({@code List<Integer>})</li>
         * <li>Parameter 2: signature algorithms from digests record ({@code List<Integer>})</li>
         * </ul>
         */
        V3_SIG_SIG_ALG_MISMATCH_BETWEEN_SIGNATURES_AND_DIGESTS_RECORDS(
                "Signature algorithms mismatch between signatures and digests records"
                        + ": %1$s vs %2$s"),

        /**
         * The APK's digest does not match the digest contained in the APK Signature Scheme v3
         * signature.
         *
         * <ul>
         * <li>Parameter 1: content digest algorithm ({@link ContentDigestAlgorithm})</li>
         * <li>Parameter 2: hex-encoded expected digest of the APK ({@code String})</li>
         * <li>Parameter 3: hex-encoded actual digest of the APK ({@code String})</li>
         * </ul>
         */
        V3_SIG_APK_DIGEST_DID_NOT_VERIFY(
                "APK integrity check failed. %1$s digest mismatch."
                        + " Expected: <%2$s>, actual: <%3$s>"),

        /**
         * The signer's SigningCertificateLineage attribute containd a proof-of-rotation record with
         * signature(s) that did not verify.
         */
        V3_SIG_POR_DID_NOT_VERIFY("SigningCertificateLineage attribute containd a proof-of-rotation"
                + " record with signature(s) that did not verify."),

        /**
         * Failed to parse the SigningCertificateLineage structure in the APK Signature Scheme v3
         * signature's additional attributes section.
         */
        V3_SIG_MALFORMED_LINEAGE("Failed to parse the SigningCertificateLineage structure in the "
                + "APK Signature Scheme v3 signature's additional attributes section."),

        /**
         * The APK's signing certificate does not match the terminal node in the provided
         * proof-of-rotation structure describing the signing certificate history
         */
        V3_SIG_POR_CERT_MISMATCH(
                "APK signing certificate differs from the associated certificate found in the "
                        + "signer's SigningCertificateLineage."),

        /**
         * The APK Signature Scheme v3 signers encountered do not offer a continuous set of
         * supported platform versions.  Either they overlap, resulting in potentially two
         * acceptable signers for a platform version, or there are holes which would create problems
         * in the event of platform version upgrades.
         */
        V3_INCONSISTENT_SDK_VERSIONS("APK Signature Scheme v3 signers supported min/max SDK "
                + "versions are not continuous."),

        /**
         * The APK Signature Scheme v3 signers don't cover all requested SDK versions.
         *
         *  <ul>
         * <li>Parameter 1: minSdkVersion ({@code Integer})
         * <li>Parameter 2: maxSdkVersion ({@code Integer})
         * </ul>
         */
        V3_MISSING_SDK_VERSIONS("APK Signature Scheme v3 signers supported min/max SDK "
                + "versions do not cover the entire desired range.  Found min:  %1$s max %2$s"),

        /**
         * The SigningCertificateLineages for different platform versions using APK Signature Scheme
         * v3 do not go together.  Specifically, each should be a subset of another, with the size
         * of each increasing as the platform level increases.
         */
        V3_INCONSISTENT_LINEAGES("SigningCertificateLineages targeting different platform versions"
                + " using APK Signature Scheme v3 are not all a part of the same overall lineage."),

        /**
         * The v3 stripping protection attribute for rotation is present, but a v3.1 signing block
         * was not found.
         *
         * <ul>
         * <li>Parameter 1: min SDK version supporting rotation from attribute ({@code Integer})
         * </ul>
         */
        V31_BLOCK_MISSING(
                "The v3 signer indicates key rotation should be supported starting from SDK "
                        + "version %1$s, but a v3.1 block was not found"),

        /**
         * The v3 stripping protection attribute for rotation does not match the minimum SDK version
         * targeting rotation in the v3.1 signer block.
         *
         * <ul>
         * <li>Parameter 1: min SDK version supporting rotation from attribute ({@code Integer})
         * <li>Parameter 2: min SDK version supporting rotation from v3.1 block ({@code Integer})
         * </ul>
         */
        V31_ROTATION_MIN_SDK_MISMATCH(
                "The v3 signer indicates key rotation should be supported starting from SDK "
                        + "version %1$s, but the v3.1 block targets %2$s for rotation"),

        /**
         * The APK supports key rotation with SDK version targeting using v3.1, but the rotation min
         * SDK version stripping protection attribute was not written to the v3 signer.
         *
         * <ul>
         * <li>Parameter 1: min SDK version supporting rotation from v3.1 block ({@code Integer})
         * </ul>
         */
        V31_ROTATION_MIN_SDK_ATTR_MISSING(
                "APK supports key rotation starting from SDK version %1$s, but the v3 signer does"
                        + " not contain the attribute to detect if this signature is stripped"),

        /**
         * The APK contains a v3.1 signing block without a v3.0 block. The v3.1 block should only
         * be used for targeting rotation for a later SDK version; if an APK's minSdkVersion is the
         * same as the SDK version for rotation then this should be written to a v3.0 block.
         */
        V31_BLOCK_FOUND_WITHOUT_V3_BLOCK(
                "The APK contains a v3.1 signing block without a v3.0 base block"),

        /**
         * The APK contains a v3.0 signing block with a rotation-targets-dev-release attribute in
         * the signer; this attribute is only intended for v3.1 signers to indicate they should be
         * targeting the next development release that is using the SDK version of the previously
         * released platform SDK version.
         */
        V31_ROTATION_TARGETS_DEV_RELEASE_ATTR_ON_V3_SIGNER(
                "The rotation-targets-dev-release attribute is only supported on v3.1 signers; "
                        + "this attribute will be ignored by the platform in a v3.0 signer"),

        /**
         * APK Signing Block contains an unknown entry.
         *
         * <ul>
         * <li>Parameter 1: entry ID ({@code Integer})</li>
         * </ul>
         */
        APK_SIG_BLOCK_UNKNOWN_ENTRY_ID("APK Signing Block contains unknown entry: ID %1$#x"),

        /**
         * Failed to parse this signer's signature record contained in the APK Signature Scheme
         * V4 signature.
         *
         * <ul>
         * <li>Parameter 1: record number (first record is {@code 1}) ({@code Integer})</li>
         * </ul>
         */
        V4_SIG_MALFORMED_SIGNERS(
                "V4 signature has malformed signer block"),

        /**
         * This APK Signature Scheme V4 signer contains a signature produced using an
         * unknown algorithm.
         *
         * <ul>
         * <li>Parameter 1: algorithm ID ({@code Integer})</li>
         * </ul>
         */
        V4_SIG_UNKNOWN_SIG_ALGORITHM(
                "V4 signature has unknown signing algorithm: %1$#x"),

        /**
         * This APK Signature Scheme V4 signer offers no signatures.
         */
        V4_SIG_NO_SIGNATURES(
                "V4 signature has no signature found"),

        /**
         * This APK Signature Scheme V4 signer offers signatures but none of them are
         * supported.
         */
        V4_SIG_NO_SUPPORTED_SIGNATURES(
                "V4 signature has no supported signature"),

        /**
         * APK Signature Scheme v3 signature over this signer's signed-data block did not verify.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * </ul>
         */
        V4_SIG_DID_NOT_VERIFY("%1$s signature over signed-data did not verify"),

        /**
         * An exception was encountered while verifying APK Signature Scheme v3 signature of this
         * signer.
         *
         * <ul>
         * <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})</li>
         * <li>Parameter 2: exception ({@code Throwable})</li>
         * </ul>
         */
        V4_SIG_VERIFY_EXCEPTION("Failed to verify %1$s signature: %2$s"),

        /**
         * Public key embedded in the APK Signature Scheme v4 signature of this signer could not be
         * parsed.
         *
         * <ul>
         * <li>Parameter 1: error details ({@code Throwable})</li>
         * </ul>
         */
        V4_SIG_MALFORMED_PUBLIC_KEY("Malformed public key: %1$s"),

        /**
         * This APK Signature Scheme V4 signer's certificate could not be parsed.
         *
         * <ul>
         * <li>Parameter 1: index ({@code 0}-based) of the certificate in the signer's list of
         *     certificates ({@code Integer})</li>
         * <li>Parameter 2: sequence number ({@code 1}-based) of the certificate in the signer's
         *     list of certificates ({@code Integer})</li>
         * <li>Parameter 3: error details ({@code Throwable})</li>
         * </ul>
         */
        V4_SIG_MALFORMED_CERTIFICATE(
                "V4 signature has malformed certificate"),

        /**
         * This APK Signature Scheme V4 signer offers no certificate.
         */
        V4_SIG_NO_CERTIFICATE("V4 signature has no certificate"),

        /**
         * This APK Signature Scheme V4 signer's public key listed in the signer's
         * certificate does not match the public key listed in the signature proto.
         *
         * <ul>
         * <li>Parameter 1: hex-encoded public key from certificate ({@code String})</li>
         * <li>Parameter 2: hex-encoded public key from signature proto ({@code String})</li>
         * </ul>
         */
        V4_SIG_PUBLIC_KEY_MISMATCH_BETWEEN_CERTIFICATE_AND_SIGNATURES_RECORD(
                "V4 signature has mismatched certificate and signature: <%1$s> vs <%2$s>"),

        /**
         * The APK's hash root (aka digest) does not match the hash root contained in the Signature
         * Scheme V4 signature.
         *
         * <ul>
         * <li>Parameter 1: content digest algorithm ({@link ContentDigestAlgorithm})</li>
         * <li>Parameter 2: hex-encoded expected digest of the APK ({@code String})</li>
         * <li>Parameter 3: hex-encoded actual digest of the APK ({@code String})</li>
         * </ul>
         */
        V4_SIG_APK_ROOT_DID_NOT_VERIFY(
                "V4 signature's hash tree root (content digest) did not verity"),

        /**
         * The APK's hash tree does not match the hash tree contained in the Signature
         * Scheme V4 signature.
         *
         * <ul>
         * <li>Parameter 1: content digest algorithm ({@link ContentDigestAlgorithm})</li>
         * <li>Parameter 2: hex-encoded expected hash tree of the APK ({@code String})</li>
         * <li>Parameter 3: hex-encoded actual hash tree of the APK ({@code String})</li>
         * </ul>
         */
        V4_SIG_APK_TREE_DID_NOT_VERIFY(
                "V4 signature's hash tree did not verity"),

        /**
         * Using more than one Signer to sign APK Signature Scheme V4 signature.
         */
        V4_SIG_MULTIPLE_SIGNERS(
                "V4 signature only supports one signer"),

        /**
         * V4.1 signature requires two signers to match the v3 and the v3.1.
         */
        V41_SIG_NEEDS_TWO_SIGNERS("V4.1 signature requires two signers"),

        /**
         * The signer used to sign APK Signature Scheme V2/V3 signature does not match the signer
         * used to sign APK Signature Scheme V4 signature.
         */
        V4_SIG_V2_V3_SIGNERS_MISMATCH(
                "V4 signature and V2/V3 signature have mismatched certificates"),

        /**
         * The v4 signature's digest does not match the digest from the corresponding v2 / v3
         * signature.
         *
         * <ul>
         *     <li>Parameter 1: Signature scheme of mismatched digest ({@code int})
         *     <li>Parameter 2: v2/v3 digest ({@code String})
         *     <li>Parameter 3: v4 digest ({@code String})
         * </ul>
         */
        V4_SIG_V2_V3_DIGESTS_MISMATCH(
                "V4 signature and V%1$d signature have mismatched digests, V%1$d digest: %2$s, V4"
                        + " digest: %3$s"),

        /**
         * The v4 signature does not contain the expected number of digests.
         *
         * <ul>
         *     <li>Parameter 1: Number of digests found ({@code int})
         * </ul>
         */
        V4_SIG_UNEXPECTED_DIGESTS(
                "V4 signature does not have the expected number of digests, found %1$d"),

        /**
         * The v4 signature format version isn't the same as the tool's current version, something
         * may go wrong.
         */
        V4_SIG_VERSION_NOT_CURRENT(
                "V4 signature format version %1$d is different from the tool's current "
                        + "version %2$d"),

        /**
         * The APK does not contain the source stamp certificate digest file nor the signature block
         * when verification expected a source stamp to be present.
         */
        SOURCE_STAMP_CERT_DIGEST_AND_SIG_BLOCK_MISSING(
                "Neither the source stamp certificate digest file nor the signature block are "
                        + "present in the APK"),

        /** APK contains SourceStamp file, but does not contain a SourceStamp signature. */
        SOURCE_STAMP_SIG_MISSING("No SourceStamp signature"),

        /**
         * SourceStamp's certificate could not be parsed.
         *
         * <ul>
         *   <li>Parameter 1: error details ({@code Throwable})
         * </ul>
         */
        SOURCE_STAMP_MALFORMED_CERTIFICATE("Malformed certificate: %1$s"),

        /** Failed to parse SourceStamp's signature. */
        SOURCE_STAMP_MALFORMED_SIGNATURE("Malformed SourceStamp signature"),

        /**
         * SourceStamp contains a signature produced using an unknown algorithm.
         *
         * <ul>
         *   <li>Parameter 1: algorithm ID ({@code Integer})
         * </ul>
         */
        SOURCE_STAMP_UNKNOWN_SIG_ALGORITHM("Unknown signature algorithm: %1$#x"),

        /**
         * An exception was encountered while verifying SourceStamp signature.
         *
         * <ul>
         *   <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})
         *   <li>Parameter 2: exception ({@code Throwable})
         * </ul>
         */
        SOURCE_STAMP_VERIFY_EXCEPTION("Failed to verify %1$s signature: %2$s"),

        /**
         * SourceStamp signature block did not verify.
         *
         * <ul>
         *   <li>Parameter 1: signature algorithm ({@link SignatureAlgorithm})
         * </ul>
         */
        SOURCE_STAMP_DID_NOT_VERIFY("%1$s signature over signed-data did not verify"),

        /** SourceStamp offers no signatures. */
        SOURCE_STAMP_NO_SIGNATURE("No signature"),

        /**
         * SourceStamp offers an unsupported signature.
         * <ul>
         *     <li>Parameter 1: list of {@link SignatureAlgorithm}s  in the source stamp
         *     signing block.
         *     <li>Parameter 2: {@code Exception} caught when attempting to obtain the list of
         *     supported signatures.
         * </ul>
         */
        SOURCE_STAMP_NO_SUPPORTED_SIGNATURE("Signature(s) {%1$s} not supported: %2$s"),

        /**
         * SourceStamp's certificate listed in the APK signing block does not match the certificate
         * listed in the SourceStamp file in the APK.
         *
         * <ul>
         *   <li>Parameter 1: SHA-256 hash of certificate from SourceStamp block in APK signing
         *       block ({@code String})
         *   <li>Parameter 2: SHA-256 hash of certificate from SourceStamp file in APK ({@code
         *       String})
         * </ul>
         */
        SOURCE_STAMP_CERTIFICATE_MISMATCH_BETWEEN_SIGNATURE_BLOCK_AND_APK(
                "Certificate mismatch between SourceStamp block in APK signing block and"
                        + " SourceStamp file in APK: <%1$s> vs <%2$s>"),

        /**
         * The APK contains a source stamp signature block without the expected certificate digest
         * in the APK contents.
         */
        SOURCE_STAMP_SIGNATURE_BLOCK_WITHOUT_CERT_DIGEST(
                "A source stamp signature block was found without a corresponding certificate "
                        + "digest in the APK"),

        /**
         * When verifying just the source stamp, the certificate digest in the APK does not match
         * the expected digest.
         * <ul>
         *     <li>Parameter 1: SHA-256 digest of the source stamp certificate in the APK.
         *     <li>Parameter 2: SHA-256 digest of the expected source stamp certificate.
         * </ul>
         */
        SOURCE_STAMP_EXPECTED_DIGEST_MISMATCH(
                "The source stamp certificate digest in the APK, %1$s, does not match the "
                        + "expected digest, %2$s"),

        /**
         * Source stamp block contains a malformed attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute number (first attribute is {@code 1}) {@code Integer})</li>
         * </ul>
         */
        SOURCE_STAMP_MALFORMED_ATTRIBUTE("Malformed stamp attribute #%1$d"),

        /**
         * Source stamp block contains an unknown attribute.
         *
         * <ul>
         * <li>Parameter 1: attribute ID ({@code Integer})</li>
         * </ul>
         */
        SOURCE_STAMP_UNKNOWN_ATTRIBUTE("Unknown stamp attribute: ID %1$#x"),

        /**
         * Failed to parse the SigningCertificateLineage structure in the source stamp
         * attributes section.
         */
        SOURCE_STAMP_MALFORMED_LINEAGE("Failed to parse the SigningCertificateLineage "
                + "structure in the source stamp attributes section."),

        /**
         * The source stamp certificate does not match the terminal node in the provided
         * proof-of-rotation structure describing the stamp certificate history.
         */
        SOURCE_STAMP_POR_CERT_MISMATCH(
                "APK signing certificate differs from the associated certificate found in the "
                        + "signer's SigningCertificateLineage."),

        /**
         * The source stamp SigningCertificateLineage attribute contains a proof-of-rotation record
         * with signature(s) that did not verify.
         */
        SOURCE_STAMP_POR_DID_NOT_VERIFY("Source stamp SigningCertificateLineage attribute "
                + "contains a proof-of-rotation record with signature(s) that did not verify."),

        /**
         * The source stamp timestamp attribute has an invalid value (<= 0).
         * <ul>
         *     <li>Parameter 1: The invalid timestamp value.
         * </ul>
         */
        SOURCE_STAMP_INVALID_TIMESTAMP(
                "The source stamp"
                        + " timestamp attribute has an invalid value: %1$d"),

        /**
         * The APK could not be properly parsed due to a ZIP or APK format exception.
         * <ul>
         *     <li>Parameter 1: The {@code Exception} caught when attempting to parse the APK.
         * </ul>
         */
        MALFORMED_APK(
                "Malformed APK; the following exception was caught when attempting to parse the "
                        + "APK: %1$s"),

        /**
         * An unexpected exception was caught when attempting to verify the signature(s) within the
         * APK.
         * <ul>
         *     <li>Parameter 1: The {@code Exception} caught during verification.
         * </ul>
         */
        UNEXPECTED_EXCEPTION(
                "An unexpected exception was caught when verifying the signature: %1$s");

        private final String mFormat;

        Issue(String format) {
            mFormat = format;
        }

        /**
         * Returns the format string suitable for combining the parameters of this issue into a
         * readable string. See {@link java.util.Formatter} for format.
         */
        private String getFormat() {
            return mFormat;
        }
    }

    /**
     * {@link Issue} with associated parameters. {@link #toString()} produces a readable formatted
     * form.
     */
    public static class IssueWithParams extends ApkVerificationIssue {
        private final Issue mIssue;
        private final Object[] mParams;

        /**
         * Constructs a new {@code IssueWithParams} of the specified type and with provided
         * parameters.
         */
        public IssueWithParams(Issue issue, Object[] params) {
            super(issue.mFormat, params);
            mIssue = issue;
            mParams = params;
        }

        /**
         * Returns the type of this issue.
         */
        public Issue getIssue() {
            return mIssue;
        }

        /**
         * Returns the parameters of this issue.
         */
        public Object[] getParams() {
            return mParams.clone();
        }

        /**
         * Returns a readable form of this issue.
         */
        @Override
        public String toString() {
            return String.format(mIssue.getFormat(), mParams);
        }
    }

    /**
     * Wrapped around {@code byte[]} which ensures that {@code equals} and {@code hashCode} operate
     * on the contents of the arrays rather than on references.
     */
    private static class ByteArray {
        private final byte[] mArray;
        private final int mHashCode;

        private ByteArray(byte[] arr) {
            mArray = arr;
            mHashCode = Arrays.hashCode(mArray);
        }

        @Override
        public int hashCode() {
            return mHashCode;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof ByteArray)) {
                return false;
            }
            ByteArray other = (ByteArray) obj;
            if (hashCode() != other.hashCode()) {
                return false;
            }
            if (!Arrays.equals(mArray, other.mArray)) {
                return false;
            }
            return true;
        }
    }

    /**
     * Builder of {@link ApkVerifier} instances.
     *
     * <p>The resulting verifier by default checks whether the APK will verify on all platform
     * versions supported by the APK, as specified by {@code android:minSdkVersion} attributes in
     * the APK's {@code AndroidManifest.xml}. The range of platform versions can be customized using
     * {@link #setMinCheckedPlatformVersion(int)} and {@link #setMaxCheckedPlatformVersion(int)}.
     */
    public static class Builder {
        private final File mApkFile;
        private final DataSource mApkDataSource;
        private File mV4SignatureFile;

        private Integer mMinSdkVersion;
        private int mMaxSdkVersion = Integer.MAX_VALUE;

        /**
         * Constructs a new {@code Builder} for verifying the provided APK file.
         */
        public Builder(File apk) {
            if (apk == null) {
                throw new NullPointerException("apk == null");
            }
            mApkFile = apk;
            mApkDataSource = null;
        }

        /**
         * Constructs a new {@code Builder} for verifying the provided APK.
         */
        public Builder(DataSource apk) {
            if (apk == null) {
                throw new NullPointerException("apk == null");
            }
            mApkDataSource = apk;
            mApkFile = null;
        }

        /**
         * Sets the oldest Android platform version for which the APK is verified. APK verification
         * will confirm that the APK is expected to install successfully on all known Android
         * platforms starting from the platform version with the provided API Level. The upper end
         * of the platform versions range can be modified via
         * {@link #setMaxCheckedPlatformVersion(int)}.
         *
         * <p>This method is useful for overriding the default behavior which checks that the APK
         * will verify on all platform versions supported by the APK, as specified by
         * {@code android:minSdkVersion} attributes in the APK's {@code AndroidManifest.xml}.
         *
         * @param minSdkVersion API Level of the oldest platform for which to verify the APK
         * @see #setMinCheckedPlatformVersion(int)
         */
        public Builder setMinCheckedPlatformVersion(int minSdkVersion) {
            mMinSdkVersion = minSdkVersion;
            return this;
        }

        /**
         * Sets the newest Android platform version for which the APK is verified. APK verification
         * will confirm that the APK is expected to install successfully on all platform versions
         * supported by the APK up until and including the provided version. The lower end
         * of the platform versions range can be modified via
         * {@link #setMinCheckedPlatformVersion(int)}.
         *
         * @param maxSdkVersion API Level of the newest platform for which to verify the APK
         * @see #setMinCheckedPlatformVersion(int)
         */
        public Builder setMaxCheckedPlatformVersion(int maxSdkVersion) {
            mMaxSdkVersion = maxSdkVersion;
            return this;
        }

        public Builder setV4SignatureFile(File v4SignatureFile) {
            mV4SignatureFile = v4SignatureFile;
            return this;
        }

        /**
         * Returns an {@link ApkVerifier} initialized according to the configuration of this
         * builder.
         */
        public ApkVerifier build() {
            return new ApkVerifier(
                    mApkFile,
                    mApkDataSource,
                    mV4SignatureFile,
                    mMinSdkVersion,
                    mMaxSdkVersion);
        }
    }

    /**
     * Adapter for converting base {@link ApkVerificationIssue} instances to their {@link
     * IssueWithParams} equivalent.
     */
    public static class ApkVerificationIssueAdapter {
        private ApkVerificationIssueAdapter() {
        }

        // This field is visible for testing
        static final Map<Integer, Issue> sVerificationIssueIdToIssue = new HashMap<>();

        static {
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_MALFORMED_SIGNERS,
                    Issue.V2_SIG_MALFORMED_SIGNERS);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_NO_SIGNERS,
                    Issue.V2_SIG_NO_SIGNERS);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_MALFORMED_SIGNER,
                    Issue.V2_SIG_MALFORMED_SIGNER);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_MALFORMED_SIGNATURE,
                    Issue.V2_SIG_MALFORMED_SIGNATURE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_NO_SIGNATURES,
                    Issue.V2_SIG_NO_SIGNATURES);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_MALFORMED_CERTIFICATE,
                    Issue.V2_SIG_MALFORMED_CERTIFICATE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_NO_CERTIFICATES,
                    Issue.V2_SIG_NO_CERTIFICATES);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V2_SIG_MALFORMED_DIGEST,
                    Issue.V2_SIG_MALFORMED_DIGEST);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_MALFORMED_SIGNERS,
                    Issue.V3_SIG_MALFORMED_SIGNERS);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_NO_SIGNERS,
                    Issue.V3_SIG_NO_SIGNERS);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_MALFORMED_SIGNER,
                    Issue.V3_SIG_MALFORMED_SIGNER);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_MALFORMED_SIGNATURE,
                    Issue.V3_SIG_MALFORMED_SIGNATURE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_NO_SIGNATURES,
                    Issue.V3_SIG_NO_SIGNATURES);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_MALFORMED_CERTIFICATE,
                    Issue.V3_SIG_MALFORMED_CERTIFICATE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_NO_CERTIFICATES,
                    Issue.V3_SIG_NO_CERTIFICATES);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.V3_SIG_MALFORMED_DIGEST,
                    Issue.V3_SIG_MALFORMED_DIGEST);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_NO_SIGNATURE,
                    Issue.SOURCE_STAMP_NO_SIGNATURE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_CERTIFICATE,
                    Issue.SOURCE_STAMP_MALFORMED_CERTIFICATE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_UNKNOWN_SIG_ALGORITHM,
                    Issue.SOURCE_STAMP_UNKNOWN_SIG_ALGORITHM);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_SIGNATURE,
                    Issue.SOURCE_STAMP_MALFORMED_SIGNATURE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_DID_NOT_VERIFY,
                    Issue.SOURCE_STAMP_DID_NOT_VERIFY);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_VERIFY_EXCEPTION,
                    Issue.SOURCE_STAMP_VERIFY_EXCEPTION);
            sVerificationIssueIdToIssue.put(
                    ApkVerificationIssue.SOURCE_STAMP_EXPECTED_DIGEST_MISMATCH,
                    Issue.SOURCE_STAMP_EXPECTED_DIGEST_MISMATCH);
            sVerificationIssueIdToIssue.put(
                    ApkVerificationIssue.SOURCE_STAMP_SIGNATURE_BLOCK_WITHOUT_CERT_DIGEST,
                    Issue.SOURCE_STAMP_SIGNATURE_BLOCK_WITHOUT_CERT_DIGEST);
            sVerificationIssueIdToIssue.put(
                    ApkVerificationIssue.SOURCE_STAMP_CERT_DIGEST_AND_SIG_BLOCK_MISSING,
                    Issue.SOURCE_STAMP_CERT_DIGEST_AND_SIG_BLOCK_MISSING);
            sVerificationIssueIdToIssue.put(
                    ApkVerificationIssue.SOURCE_STAMP_NO_SUPPORTED_SIGNATURE,
                    Issue.SOURCE_STAMP_NO_SUPPORTED_SIGNATURE);
            sVerificationIssueIdToIssue.put(
                    ApkVerificationIssue
                            .SOURCE_STAMP_CERTIFICATE_MISMATCH_BETWEEN_SIGNATURE_BLOCK_AND_APK,
                    Issue.SOURCE_STAMP_CERTIFICATE_MISMATCH_BETWEEN_SIGNATURE_BLOCK_AND_APK);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.MALFORMED_APK,
                    Issue.MALFORMED_APK);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.UNEXPECTED_EXCEPTION,
                    Issue.UNEXPECTED_EXCEPTION);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_SIG_MISSING,
                    Issue.SOURCE_STAMP_SIG_MISSING);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_ATTRIBUTE,
                    Issue.SOURCE_STAMP_MALFORMED_ATTRIBUTE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_UNKNOWN_ATTRIBUTE,
                    Issue.SOURCE_STAMP_UNKNOWN_ATTRIBUTE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_MALFORMED_LINEAGE,
                    Issue.SOURCE_STAMP_MALFORMED_LINEAGE);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_POR_CERT_MISMATCH,
                    Issue.SOURCE_STAMP_POR_CERT_MISMATCH);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_POR_DID_NOT_VERIFY,
                    Issue.SOURCE_STAMP_POR_DID_NOT_VERIFY);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.JAR_SIG_NO_SIGNATURES,
                    Issue.JAR_SIG_NO_SIGNATURES);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.JAR_SIG_PARSE_EXCEPTION,
                    Issue.JAR_SIG_PARSE_EXCEPTION);
            sVerificationIssueIdToIssue.put(ApkVerificationIssue.SOURCE_STAMP_INVALID_TIMESTAMP,
                    Issue.SOURCE_STAMP_INVALID_TIMESTAMP);
        }

        /**
         * Converts the provided {@code verificationIssues} to a {@code List} of corresponding
         * {@link IssueWithParams} instances.
         */
        public static List<IssueWithParams> getIssuesFromVerificationIssues(
                List<? extends ApkVerificationIssue> verificationIssues) {
            List<IssueWithParams> result = new ArrayList<>(verificationIssues.size());
            for (ApkVerificationIssue issue : verificationIssues) {
                if (issue instanceof IssueWithParams) {
                    result.add((IssueWithParams) issue);
                } else {
                    result.add(
                            new IssueWithParams(sVerificationIssueIdToIssue.get(issue.getIssueId()),
                                    issue.getParams()));
                }
            }
            return result;
        }
    }
}
