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
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERITY_PADDING_BLOCK_ID;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_JAR_SIGNATURE_SCHEME;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.MIN_SDK_WITH_V31_SUPPORT;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.MIN_SDK_WITH_V3_SUPPORT;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.apk.stamp.V2SourceStampSigner;
import com.android.apksig.internal.apk.v1.DigestAlgorithm;
import com.android.apksig.internal.apk.v1.V1SchemeConstants;
import com.android.apksig.internal.apk.v1.V1SchemeSigner;
import com.android.apksig.internal.apk.v1.V1SchemeVerifier;
import com.android.apksig.internal.apk.v2.V2SchemeSigner;
import com.android.apksig.internal.apk.v3.V3SchemeConstants;
import com.android.apksig.internal.apk.v3.V3SchemeSigner;
import com.android.apksig.internal.apk.v4.V4SchemeSigner;
import com.android.apksig.internal.apk.v4.V4Signature;
import com.android.apksig.internal.jar.ManifestParser;
import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.internal.util.TeeDataSink;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSinks;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.RunnablesExecutor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Default implementation of {@link ApkSignerEngine}.
 *
 * <p>Use {@link Builder} to obtain instances of this engine.
 */
public class DefaultApkSignerEngine implements ApkSignerEngine {

    // IMPLEMENTATION NOTE: This engine generates a signed APK as follows:
    // 1. The engine asks its client to output input JAR entries which are not part of JAR
    //    signature.
    // 2. If JAR signing (v1 signing) is enabled, the engine inspects the output JAR entries to
    //    compute their digests, to be placed into output META-INF/MANIFEST.MF. It also inspects
    //    the contents of input and output META-INF/MANIFEST.MF to borrow the main section of the
    //    file. It does not care about individual (i.e., JAR entry-specific) sections. It then
    //    emits the v1 signature (a set of JAR entries) and asks the client to output them.
    // 3. If APK Signature Scheme v2 (v2 signing) is enabled, the engine emits an APK Signing Block
    //    from outputZipSections() and asks its client to insert this block into the output.
    // 4. If APK Signature Scheme v3 (v3 signing) is enabled, the engine includes it in the APK
    //    Signing BLock output from outputZipSections() and asks its client to insert this block
    //    into the output.  If both v2 and v3 signing is enabled, they are both added to the APK
    //    Signing Block before asking the client to insert it into the output.

    private final boolean mV1SigningEnabled;
    private final boolean mV2SigningEnabled;
    private final boolean mV3SigningEnabled;
    private final boolean mVerityEnabled;
    private final boolean mDebuggableApkPermitted;
    private final boolean mOtherSignersSignaturesPreserved;
    private final String mCreatedBy;
    private final List<SignerConfig> mSignerConfigs;
    private final List<SignerConfig> mTargetedSignerConfigs;
    private final SignerConfig mSourceStampSignerConfig;
    private final SigningCertificateLineage mSourceStampSigningCertificateLineage;
    private final boolean mSourceStampTimestampEnabled;
    private final int mMinSdkVersion;
    private final SigningCertificateLineage mSigningCertificateLineage;

    private List<byte[]> mPreservedV2Signers = Collections.emptyList();
    private List<Pair<byte[], Integer>> mPreservedSignatureBlocks = Collections.emptyList();

    private List<V1SchemeSigner.SignerConfig> mV1SignerConfigs = Collections.emptyList();
    private DigestAlgorithm mV1ContentDigestAlgorithm;

    private boolean mClosed;

    private boolean mV1SignaturePending;

    /** Names of JAR entries which this engine is expected to output as part of v1 signing. */
    private Set<String> mSignatureExpectedOutputJarEntryNames = Collections.emptySet();

    /** Requests for digests of output JAR entries. */
    private final Map<String, GetJarEntryDataDigestRequest> mOutputJarEntryDigestRequests =
            new HashMap<>();

    /** Digests of output JAR entries. */
    private final Map<String, byte[]> mOutputJarEntryDigests = new HashMap<>();

    /** Data of JAR entries emitted by this engine as v1 signature. */
    private final Map<String, byte[]> mEmittedSignatureJarEntryData = new HashMap<>();

    /** Requests for data of output JAR entries which comprise the v1 signature. */
    private final Map<String, GetJarEntryDataRequest> mOutputSignatureJarEntryDataRequests =
            new HashMap<>();
    /**
     * Request to obtain the data of MANIFEST.MF or {@code null} if the request hasn't been issued.
     */
    private GetJarEntryDataRequest mInputJarManifestEntryDataRequest;

    /**
     * Request to obtain the data of AndroidManifest.xml or {@code null} if the request hasn't been
     * issued.
     */
    private GetJarEntryDataRequest mOutputAndroidManifestEntryDataRequest;

    /**
     * Whether the package being signed is marked as {@code android:debuggable} or {@code null} if
     * this is not yet known.
     */
    private Boolean mDebuggable;

    /**
     * Request to output the emitted v1 signature or {@code null} if the request hasn't been issued.
     */
    private OutputJarSignatureRequestImpl mAddV1SignatureRequest;

    private boolean mV2SignaturePending;
    private boolean mV3SignaturePending;

    /**
     * Request to output the emitted v2 and/or v3 signature(s) {@code null} if the request hasn't
     * been issued.
     */
    private OutputApkSigningBlockRequestImpl mAddSigningBlockRequest;

    private RunnablesExecutor mExecutor = RunnablesExecutor.MULTI_THREADED;

    /**
     * A Set of block IDs to be discarded when requesting to preserve the original signatures.
     */
    private static final Set<Integer> DISCARDED_SIGNATURE_BLOCK_IDS;
    static {
        DISCARDED_SIGNATURE_BLOCK_IDS = new HashSet<>(3);
        // The verity padding block is recomputed on an
        // ApkSigningBlockUtils.ANDROID_COMMON_PAGE_ALIGNMENT_BYTES boundary.
        DISCARDED_SIGNATURE_BLOCK_IDS.add(VERITY_PADDING_BLOCK_ID);
        // The source stamp block is not currently preserved; appending a new signature scheme
        // block will invalidate the previous source stamp.
        DISCARDED_SIGNATURE_BLOCK_IDS.add(Constants.V1_SOURCE_STAMP_BLOCK_ID);
        DISCARDED_SIGNATURE_BLOCK_IDS.add(Constants.V2_SOURCE_STAMP_BLOCK_ID);
    }

    private DefaultApkSignerEngine(
            List<SignerConfig> signerConfigs,
            List<SignerConfig> targetedSignerConfigs,
            SignerConfig sourceStampSignerConfig,
            SigningCertificateLineage sourceStampSigningCertificateLineage,
            boolean sourceStampTimestampEnabled,
            int minSdkVersion,
            boolean v1SigningEnabled,
            boolean v2SigningEnabled,
            boolean v3SigningEnabled,
            boolean verityEnabled,
            boolean debuggableApkPermitted,
            boolean otherSignersSignaturesPreserved,
            String createdBy,
            SigningCertificateLineage signingCertificateLineage)
            throws InvalidKeyException {
        if (signerConfigs.isEmpty() && targetedSignerConfigs.isEmpty()) {
            throw new IllegalArgumentException("At least one signer config must be provided");
        }

        mV1SigningEnabled = v1SigningEnabled;
        mV2SigningEnabled = v2SigningEnabled;
        mV3SigningEnabled = v3SigningEnabled;
        mVerityEnabled = verityEnabled;
        mV1SignaturePending = v1SigningEnabled;
        mV2SignaturePending = v2SigningEnabled;
        mV3SignaturePending = v3SigningEnabled;
        mDebuggableApkPermitted = debuggableApkPermitted;
        mOtherSignersSignaturesPreserved = otherSignersSignaturesPreserved;
        mCreatedBy = createdBy;
        mSignerConfigs = signerConfigs;
        mTargetedSignerConfigs = targetedSignerConfigs;
        mSourceStampSignerConfig = sourceStampSignerConfig;
        mSourceStampSigningCertificateLineage = sourceStampSigningCertificateLineage;
        mSourceStampTimestampEnabled = sourceStampTimestampEnabled;
        mMinSdkVersion = minSdkVersion;
        mSigningCertificateLineage = signingCertificateLineage;

        if (v1SigningEnabled) {
            if (v3SigningEnabled) {

                // v3 signing only supports single signers, of which the oldest (first) will be the
                // one to use for v1 and v2 signing
                SignerConfig oldestConfig = !signerConfigs.isEmpty() ? signerConfigs.get(0)
                        : targetedSignerConfigs.get(0);

                // in the event of signing certificate changes, make sure we have the oldest in the
                // signing history to sign with v1
                if (signingCertificateLineage != null) {
                    SigningCertificateLineage subLineage =
                            signingCertificateLineage.getSubLineage(
                                    oldestConfig.mCertificates.get(0));
                    if (subLineage.size() != 1) {
                        throw new IllegalArgumentException(
                                "v1 signing enabled but the oldest signer in the"
                                    + " SigningCertificateLineage is missing.  Please provide the"
                                    + " oldest signer to enable v1 signing");
                    }
                }
                createV1SignerConfigs(Collections.singletonList(oldestConfig), minSdkVersion);
            } else {
                createV1SignerConfigs(signerConfigs, minSdkVersion);
            }
        }
    }

    private void createV1SignerConfigs(List<SignerConfig> signerConfigs, int minSdkVersion)
            throws InvalidKeyException {
        mV1SignerConfigs = new ArrayList<>(signerConfigs.size());
        Map<String, Integer> v1SignerNameToSignerIndex = new HashMap<>(signerConfigs.size());
        DigestAlgorithm v1ContentDigestAlgorithm = null;
        for (int i = 0; i < signerConfigs.size(); i++) {
            SignerConfig signerConfig = signerConfigs.get(i);
            List<X509Certificate> certificates = signerConfig.getCertificates();
            PublicKey publicKey = certificates.get(0).getPublicKey();

            String v1SignerName = V1SchemeSigner.getSafeSignerName(signerConfig.getName());
            // Check whether the signer's name is unique among all v1 signers
            Integer indexOfOtherSignerWithSameName = v1SignerNameToSignerIndex.put(v1SignerName, i);
            if (indexOfOtherSignerWithSameName != null) {
                throw new IllegalArgumentException(
                        "Signers #"
                                + (indexOfOtherSignerWithSameName + 1)
                                + " and #"
                                + (i + 1)
                                + " have the same name: "
                                + v1SignerName
                                + ". v1 signer names must be unique");
            }

            DigestAlgorithm v1SignatureDigestAlgorithm =
                    V1SchemeSigner.getSuggestedSignatureDigestAlgorithm(publicKey, minSdkVersion);
            V1SchemeSigner.SignerConfig v1SignerConfig = new V1SchemeSigner.SignerConfig();
            v1SignerConfig.name = v1SignerName;
            v1SignerConfig.privateKey = signerConfig.getPrivateKey();
            v1SignerConfig.certificates = certificates;
            v1SignerConfig.signatureDigestAlgorithm = v1SignatureDigestAlgorithm;
            v1SignerConfig.deterministicDsaSigning = signerConfig.getDeterministicDsaSigning();
            // For digesting contents of APK entries and of MANIFEST.MF, pick the algorithm
            // of comparable strength to the digest algorithm used for computing the signature.
            // When there are multiple signers, pick the strongest digest algorithm out of their
            // signature digest algorithms. This avoids reducing the digest strength used by any
            // of the signers to protect APK contents.
            if (v1ContentDigestAlgorithm == null) {
                v1ContentDigestAlgorithm = v1SignatureDigestAlgorithm;
            } else {
                if (DigestAlgorithm.BY_STRENGTH_COMPARATOR.compare(
                                v1SignatureDigestAlgorithm, v1ContentDigestAlgorithm)
                        > 0) {
                    v1ContentDigestAlgorithm = v1SignatureDigestAlgorithm;
                }
            }
            mV1SignerConfigs.add(v1SignerConfig);
        }
        mV1ContentDigestAlgorithm = v1ContentDigestAlgorithm;
        mSignatureExpectedOutputJarEntryNames =
                V1SchemeSigner.getOutputEntryNames(mV1SignerConfigs);
    }

    private List<ApkSigningBlockUtils.SignerConfig> createV2SignerConfigs(
            boolean apkSigningBlockPaddingSupported) throws InvalidKeyException {
        if (mV3SigningEnabled) {

            // v3 signing only supports single signers, of which the oldest (first) will be the one
            // to use for v1 and v2 signing
            List<ApkSigningBlockUtils.SignerConfig> signerConfig = new ArrayList<>();

            SignerConfig oldestConfig = !mSignerConfigs.isEmpty() ? mSignerConfigs.get(0)
                    : mTargetedSignerConfigs.get(0);

            // first make sure that if we have signing certificate history that the oldest signer
            // corresponds to the oldest ancestor
            if (mSigningCertificateLineage != null) {
                SigningCertificateLineage subLineage =
                        mSigningCertificateLineage.getSubLineage(oldestConfig.mCertificates.get(0));
                if (subLineage.size() != 1) {
                    throw new IllegalArgumentException(
                            "v2 signing enabled but the oldest signer in"
                                    + " the SigningCertificateLineage is missing.  Please provide"
                                    + " the oldest signer to enable v2 signing.");
                }
            }
            signerConfig.add(
                    createSigningBlockSignerConfig(
                            oldestConfig,
                            apkSigningBlockPaddingSupported,
                            ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2));
            return signerConfig;
        } else {
            return createSigningBlockSignerConfigs(
                    apkSigningBlockPaddingSupported,
                    ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2);
        }
    }

    private List<ApkSigningBlockUtils.SignerConfig> processV3Configs(
            List<ApkSigningBlockUtils.SignerConfig> rawConfigs) throws InvalidKeyException {
        // If the caller only specified targeted signing configs, ensure those configs cover the
        // full range for V3 support (or the APK's minSdkVersion if > P).
        int minRequiredV3SdkVersion = Math.max(AndroidSdkVersion.P, mMinSdkVersion);
        if (mSignerConfigs.isEmpty() &&
                mTargetedSignerConfigs.get(0).getMinSdkVersion() > minRequiredV3SdkVersion) {
            throw new IllegalArgumentException(
                    "The provided targeted signer configs do not cover the SDK range for V3 "
                            + "support; either provide the original signer or ensure a signer "
                            + "targets SDK version " + minRequiredV3SdkVersion);
        }

        List<ApkSigningBlockUtils.SignerConfig> processedConfigs = new ArrayList<>();

        // we have our configs, now touch them up to appropriately cover all SDK levels since APK
        // signature scheme v3 was introduced
        int currentMinSdk = Integer.MAX_VALUE;
        for (int i = rawConfigs.size() - 1; i >= 0; i--) {
            ApkSigningBlockUtils.SignerConfig config = rawConfigs.get(i);
            if (config.signatureAlgorithms == null) {
                // no valid algorithm was found for this signer, and we haven't yet covered all
                // platform versions, something's wrong
                String keyAlgorithm = config.certificates.get(0).getPublicKey().getAlgorithm();
                throw new InvalidKeyException(
                        "Unsupported key algorithm "
                                + keyAlgorithm
                                + " is "
                                + "not supported for APK Signature Scheme v3 signing");
            }
            if (i == rawConfigs.size() - 1) {
                // first go through the loop, config should support all future platform versions.
                // this assumes we don't deprecate support for signers in the future.  If we do,
                // this needs to change
                config.maxSdkVersion = Integer.MAX_VALUE;
            } else {
                // If the previous signer was targeting a development release, then the current
                // signer's maxSdkVersion should overlap with the previous signer's minSdkVersion
                // to ensure the current signer applies to the production release.
                ApkSigningBlockUtils.SignerConfig prevSigner = processedConfigs.get(
                        processedConfigs.size() - 1);
                if (prevSigner.signerTargetsDevRelease) {
                    config.maxSdkVersion = prevSigner.minSdkVersion;
                } else {
                    config.maxSdkVersion = currentMinSdk - 1;
                }
            }
            if (config.minSdkVersion == V3SchemeConstants.DEV_RELEASE) {
                // If the current signer is targeting the current development release, then set
                // the signer's minSdkVersion to the last production release and the flag indicating
                // this signer is targeting a dev release.
                config.minSdkVersion = V3SchemeConstants.PROD_RELEASE;
                config.signerTargetsDevRelease = true;
            } else if (config.minSdkVersion == 0) {
                config.minSdkVersion = getMinSdkFromV3SignatureAlgorithms(
                        config.signatureAlgorithms);
            }
            // Truncate the lineage to the current signer if it is not the latest signer.
            X509Certificate signerCert = config.certificates.get(0);
            if (config.signingCertificateLineage != null
                    && !config.signingCertificateLineage.isCertificateLatestInLineage(signerCert)) {
                config.signingCertificateLineage = config.signingCertificateLineage.getSubLineage(
                        signerCert);
            }
            // we know that this config will be used, so add it to our result, order doesn't matter
            // at this point
            processedConfigs.add(config);
            currentMinSdk = config.minSdkVersion;
            if (config.signerTargetsDevRelease ? currentMinSdk < minRequiredV3SdkVersion
                    : currentMinSdk <= minRequiredV3SdkVersion) {
                // this satisfies all we need, stop here
                break;
            }
        }
        if (currentMinSdk > AndroidSdkVersion.P && currentMinSdk > mMinSdkVersion) {
            // we can't cover all desired SDK versions, abort
            throw new InvalidKeyException(
                    "Provided key algorithms not supported on all desired "
                            + "Android SDK versions");
        }

        return processedConfigs;
    }

    private List<ApkSigningBlockUtils.SignerConfig> createV3SignerConfigs(
            boolean apkSigningBlockPaddingSupported) throws InvalidKeyException {
        return processV3Configs(createSigningBlockSignerConfigs(apkSigningBlockPaddingSupported,
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3));
    }

    private List<ApkSigningBlockUtils.SignerConfig> processV31SignerConfigs(
            List<ApkSigningBlockUtils.SignerConfig> v3SignerConfigs) {
        // The V3.1 signature scheme supports SDK targeted signing config, but this scheme should
        // only be used when a separate signing config exists for the V3.0 block.
        if (v3SignerConfigs.size() == 1) {
            return null;
        }

        // When there are multiple signing configs, the signer with the minimum SDK version should
        // be used for the V3.0 block, and all other signers should be used for the V3.1 block.
        int signerMinSdkVersion = v3SignerConfigs.stream().mapToInt(
                signer -> signer.minSdkVersion).min().orElse(AndroidSdkVersion.P);
        List<ApkSigningBlockUtils.SignerConfig> v31SignerConfigs = new ArrayList<>();
        Iterator<ApkSigningBlockUtils.SignerConfig> v3SignerIterator = v3SignerConfigs.iterator();
        while (v3SignerIterator.hasNext()) {
            ApkSigningBlockUtils.SignerConfig signerConfig = v3SignerIterator.next();
            // If the signer config's minSdkVersion supports V3.1 and is not the min signer in the
            // list, then add it to the V3.1 signer configs and remove it from the V3.0 list. If
            // the signer is targeting the minSdkVersion as a development release, then it should
            // be included in V3.1 to allow the V3.0 block to target the production release of the
            // same SDK version.
            if (signerConfig.minSdkVersion >= MIN_SDK_WITH_V31_SUPPORT
                    && (signerConfig.minSdkVersion > signerMinSdkVersion
                    || (signerConfig.minSdkVersion >= signerMinSdkVersion
                            && signerConfig.signerTargetsDevRelease))) {
                v31SignerConfigs.add(signerConfig);
                v3SignerIterator.remove();
            }
        }
        return v31SignerConfigs;
    }

    private V4SchemeSigner.SignerConfig createV4SignerConfig() throws InvalidKeyException {
        List<ApkSigningBlockUtils.SignerConfig> v4Configs = createSigningBlockSignerConfigs(true,
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V4);
        if (v4Configs.size() != 1) {
            // V4 uses signer config to connect back to v3. Use the same filtering logic.
            v4Configs = processV3Configs(v4Configs);
        }
        List<ApkSigningBlockUtils.SignerConfig> v41configs = processV31SignerConfigs(v4Configs);
        return new V4SchemeSigner.SignerConfig(v4Configs, v41configs);
    }

    private ApkSigningBlockUtils.SignerConfig createSourceStampSignerConfig()
            throws InvalidKeyException {
        ApkSigningBlockUtils.SignerConfig config = createSigningBlockSignerConfig(
                mSourceStampSignerConfig,
                /* apkSigningBlockPaddingSupported= */ false,
                ApkSigningBlockUtils.VERSION_SOURCE_STAMP);
        if (mSourceStampSigningCertificateLineage != null) {
            config.signingCertificateLineage = mSourceStampSigningCertificateLineage.getSubLineage(
                    config.certificates.get(0));
        }
        return config;
    }

    private int getMinSdkFromV3SignatureAlgorithms(List<SignatureAlgorithm> algorithms) {
        int min = Integer.MAX_VALUE;
        for (SignatureAlgorithm algorithm : algorithms) {
            int current = algorithm.getMinSdkVersion();
            if (current < min) {
                if (current <= mMinSdkVersion || current <= AndroidSdkVersion.P) {
                    // this algorithm satisfies all of our needs, no need to keep looking
                    return current;
                } else {
                    min = current;
                }
            }
        }
        return min;
    }

    private List<ApkSigningBlockUtils.SignerConfig> createSigningBlockSignerConfigs(
            boolean apkSigningBlockPaddingSupported, int schemeId) throws InvalidKeyException {
        List<ApkSigningBlockUtils.SignerConfig> signerConfigs =
                new ArrayList<>(mSignerConfigs.size() + mTargetedSignerConfigs.size());
        for (int i = 0; i < mSignerConfigs.size(); i++) {
            SignerConfig signerConfig = mSignerConfigs.get(i);
            signerConfigs.add(
                    createSigningBlockSignerConfig(
                            signerConfig, apkSigningBlockPaddingSupported, schemeId));
        }
        if (schemeId >= VERSION_APK_SIGNATURE_SCHEME_V3) {
            for (int i = 0; i < mTargetedSignerConfigs.size(); i++) {
                SignerConfig signerConfig = mTargetedSignerConfigs.get(i);
                signerConfigs.add(
                        createSigningBlockSignerConfig(
                                signerConfig, apkSigningBlockPaddingSupported, schemeId));
            }
        }
        return signerConfigs;
    }

    private ApkSigningBlockUtils.SignerConfig createSigningBlockSignerConfig(
            SignerConfig signerConfig, boolean apkSigningBlockPaddingSupported, int schemeId)
            throws InvalidKeyException {
        List<X509Certificate> certificates = signerConfig.getCertificates();
        PublicKey publicKey = certificates.get(0).getPublicKey();

        ApkSigningBlockUtils.SignerConfig newSignerConfig = new ApkSigningBlockUtils.SignerConfig();
        newSignerConfig.privateKey = signerConfig.getPrivateKey();
        newSignerConfig.certificates = certificates;
        newSignerConfig.minSdkVersion = signerConfig.getMinSdkVersion();
        newSignerConfig.signerTargetsDevRelease = signerConfig.getSignerTargetsDevRelease();
        newSignerConfig.signingCertificateLineage = signerConfig.getSigningCertificateLineage();

        switch (schemeId) {
            case ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2:
                newSignerConfig.signatureAlgorithms =
                        V2SchemeSigner.getSuggestedSignatureAlgorithms(
                                publicKey,
                                mMinSdkVersion,
                                apkSigningBlockPaddingSupported && mVerityEnabled,
                                signerConfig.getDeterministicDsaSigning());
                break;
            case ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3:
                try {
                    newSignerConfig.signatureAlgorithms =
                            V3SchemeSigner.getSuggestedSignatureAlgorithms(
                                    publicKey,
                                    mMinSdkVersion,
                                    apkSigningBlockPaddingSupported && mVerityEnabled,
                                    signerConfig.getDeterministicDsaSigning());
                } catch (InvalidKeyException e) {

                    // It is possible for a signer used for v1/v2 signing to not be allowed for use
                    // with v3 signing.  This is ok as long as there exists a more recent v3 signer
                    // that covers all supported platform versions.  Populate signatureAlgorithm
                    // with null, it will be cleaned-up in a later step.
                    newSignerConfig.signatureAlgorithms = null;
                }
                break;
            case ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V4:
                try {
                    newSignerConfig.signatureAlgorithms =
                            V4SchemeSigner.getSuggestedSignatureAlgorithms(
                                    publicKey, mMinSdkVersion, apkSigningBlockPaddingSupported,
                                    signerConfig.getDeterministicDsaSigning());
                } catch (InvalidKeyException e) {
                    // V4 is an optional signing schema, ok to proceed without.
                    newSignerConfig.signatureAlgorithms = null;
                }
                break;
            case ApkSigningBlockUtils.VERSION_SOURCE_STAMP:
                newSignerConfig.signatureAlgorithms =
                        Collections.singletonList(
                                SignatureAlgorithm.RSA_PKCS1_V1_5_WITH_SHA256);
                break;
            default:
                throw new IllegalArgumentException("Unknown APK Signature Scheme ID requested");
        }
        return newSignerConfig;
    }

    private boolean isDebuggable(String entryName) {
        return mDebuggableApkPermitted
                || !ApkUtils.ANDROID_MANIFEST_ZIP_ENTRY_NAME.equals(entryName);
    }

    /**
     * Initializes DefaultApkSignerEngine with the existing MANIFEST.MF. This reads existing digests
     * from the MANIFEST.MF file (they are assumed correct) and stores them for the final signature
     * without recalculation. This step has a significant performance benefit in case of incremental
     * build.
     *
     * <p>This method extracts and stored computed digest for every entry that it would compute it
     * for in the {@link #outputJarEntry(String)} method
     *
     * @param manifestBytes raw representation of MANIFEST.MF file
     * @param entryNames a set of expected entries names
     * @return set of entry names which were processed by the engine during the initialization, a
     *     subset of entryNames
     */
    @Override
    @SuppressWarnings("AndroidJdkLibsChecker")
    public Set<String> initWith(byte[] manifestBytes, Set<String> entryNames) {
        V1SchemeVerifier.Result result = new V1SchemeVerifier.Result();
        Pair<ManifestParser.Section, Map<String, ManifestParser.Section>> sections =
                V1SchemeVerifier.parseManifest(manifestBytes, entryNames, result);
        String alg = V1SchemeSigner.getJcaMessageDigestAlgorithm(mV1ContentDigestAlgorithm);
        for (Map.Entry<String, ManifestParser.Section> entry : sections.getSecond().entrySet()) {
            String entryName = entry.getKey();
            if (V1SchemeSigner.isJarEntryDigestNeededInManifest(entry.getKey())
                    && isDebuggable(entryName)) {

                V1SchemeVerifier.NamedDigest extractedDigest = null;
                Collection<V1SchemeVerifier.NamedDigest> digestsToVerify =
                        V1SchemeVerifier.getDigestsToVerify(
                                entry.getValue(), "-Digest", mMinSdkVersion, Integer.MAX_VALUE);
                for (V1SchemeVerifier.NamedDigest digestToVerify : digestsToVerify) {
                    if (digestToVerify.jcaDigestAlgorithm.equals(alg)) {
                        extractedDigest = digestToVerify;
                        break;
                    }
                }
                if (extractedDigest != null) {
                    mOutputJarEntryDigests.put(entryName, extractedDigest.digest);
                }
            }
        }
        return mOutputJarEntryDigests.keySet();
    }

    @Override
    public void setExecutor(RunnablesExecutor executor) {
        mExecutor = executor;
    }

    @Override
    public void inputApkSigningBlock(DataSource apkSigningBlock) {
        checkNotClosed();

        if ((apkSigningBlock == null) || (apkSigningBlock.size() == 0)) {
            return;
        }

        if (mOtherSignersSignaturesPreserved) {
            boolean schemeSignatureBlockPreserved = false;
            mPreservedSignatureBlocks = new ArrayList<>();
            try {
                List<Pair<byte[], Integer>> signatureBlocks =
                        ApkSigningBlockUtils.getApkSignatureBlocks(apkSigningBlock);
                for (Pair<byte[], Integer> signatureBlock : signatureBlocks) {
                    if (signatureBlock.getSecond() == Constants.APK_SIGNATURE_SCHEME_V2_BLOCK_ID) {
                        // If a V2 signature block is found and the engine is configured to use V2
                        // then save any of the previous signers that are not part of the current
                        // signing request.
                        if (mV2SigningEnabled) {
                            List<Pair<List<X509Certificate>, byte[]>> v2Signers =
                                    ApkSigningBlockUtils.getApkSignatureBlockSigners(
                                            signatureBlock.getFirst());
                            mPreservedV2Signers = new ArrayList<>(v2Signers.size());
                            for (Pair<List<X509Certificate>, byte[]> v2Signer : v2Signers) {
                                if (!isConfiguredWithSigner(v2Signer.getFirst())) {
                                    mPreservedV2Signers.add(v2Signer.getSecond());
                                    schemeSignatureBlockPreserved = true;
                                }
                            }
                        } else {
                            // else V2 signing is not enabled; save the entire signature block to be
                            // added to the final APK signing block.
                            mPreservedSignatureBlocks.add(signatureBlock);
                            schemeSignatureBlockPreserved = true;
                        }
                    } else if (signatureBlock.getSecond()
                            == Constants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID) {
                        // Preserving other signers in the presence of a V3 signature block is only
                        // supported if the engine is configured to resign the APK with the V3
                        // signature scheme, and the V3 signer in the signature block is the same
                        // as the engine is configured to use.
                        if (!mV3SigningEnabled) {
                            throw new IllegalStateException(
                                    "Preserving an existing V3 signature is not supported");
                        }
                        List<Pair<List<X509Certificate>, byte[]>> v3Signers =
                                ApkSigningBlockUtils.getApkSignatureBlockSigners(
                                        signatureBlock.getFirst());
                        if (v3Signers.size() > 1) {
                            throw new IllegalArgumentException(
                                    "The provided APK signing block contains " + v3Signers.size()
                                            + " V3 signers; the V3 signature scheme only supports"
                                            + " one signer");
                        }
                        // If there is only a single V3 signer then ensure it is the signer
                        // configured to sign the APK.
                        if (v3Signers.size() == 1
                                && !isConfiguredWithSigner(v3Signers.get(0).getFirst())) {
                            throw new IllegalStateException(
                                    "The V3 signature scheme only supports one signer; a request "
                                            + "was made to preserve the existing V3 signature, "
                                            + "but the engine is configured to sign with a "
                                            + "different signer");
                        }
                    } else if (!DISCARDED_SIGNATURE_BLOCK_IDS.contains(
                            signatureBlock.getSecond())) {
                        mPreservedSignatureBlocks.add(signatureBlock);
                    }
                }
            } catch (ApkFormatException | CertificateException | IOException e) {
                throw new IllegalArgumentException("Unable to parse the provided signing block", e);
            }
            // Signature scheme V3+ only support a single signer; if the engine is configured to
            // sign with V3+ then ensure no scheme signature blocks have been preserved.
            if (mV3SigningEnabled && schemeSignatureBlockPreserved) {
                throw new IllegalStateException(
                        "Signature scheme V3+ only supports a single signer and cannot be "
                                + "appended to the existing signature scheme blocks");
            }
            return;
        }
    }

    /**
     * Returns whether the engine is configured to sign the APK with a signer using the specified
     * {@code signerCerts}.
     */
    private boolean isConfiguredWithSigner(List<X509Certificate> signerCerts) {
        for (SignerConfig signerConfig : mSignerConfigs) {
            if (signerCerts.containsAll(signerConfig.getCertificates())) {
                return true;
            }
        }
        return false;
    }

    @Override
    public InputJarEntryInstructions inputJarEntry(String entryName) {
        checkNotClosed();

        InputJarEntryInstructions.OutputPolicy outputPolicy =
                getInputJarEntryOutputPolicy(entryName);
        switch (outputPolicy) {
            case SKIP:
                return new InputJarEntryInstructions(InputJarEntryInstructions.OutputPolicy.SKIP);
            case OUTPUT:
                return new InputJarEntryInstructions(InputJarEntryInstructions.OutputPolicy.OUTPUT);
            case OUTPUT_BY_ENGINE:
                if (V1SchemeConstants.MANIFEST_ENTRY_NAME.equals(entryName)) {
                    // We copy the main section of the JAR manifest from input to output. Thus, this
                    // invalidates v1 signature and we need to see the entry's data.
                    mInputJarManifestEntryDataRequest = new GetJarEntryDataRequest(entryName);
                    return new InputJarEntryInstructions(
                            InputJarEntryInstructions.OutputPolicy.OUTPUT_BY_ENGINE,
                            mInputJarManifestEntryDataRequest);
                }
                return new InputJarEntryInstructions(
                        InputJarEntryInstructions.OutputPolicy.OUTPUT_BY_ENGINE);
            default:
                throw new RuntimeException("Unsupported output policy: " + outputPolicy);
        }
    }

    @Override
    public InspectJarEntryRequest outputJarEntry(String entryName) {
        checkNotClosed();
        invalidateV2Signature();

        if (!isDebuggable(entryName)) {
            forgetOutputApkDebuggableStatus();
        }

        if (!mV1SigningEnabled) {
            // No need to inspect JAR entries when v1 signing is not enabled.
            if (!isDebuggable(entryName)) {
                // To reject debuggable APKs we need to inspect the APK's AndroidManifest.xml to
                // check whether it declares that the APK is debuggable
                mOutputAndroidManifestEntryDataRequest = new GetJarEntryDataRequest(entryName);
                return mOutputAndroidManifestEntryDataRequest;
            }
            return null;
        }
        // v1 signing is enabled

        if (V1SchemeSigner.isJarEntryDigestNeededInManifest(entryName)) {
            // This entry is covered by v1 signature. We thus need to inspect the entry's data to
            // compute its digest(s) for v1 signature.

            // TODO: Handle the case where other signer's v1 signatures are present and need to be
            // preserved. In that scenario we can't modify MANIFEST.MF and add/remove JAR entries
            // covered by v1 signature.
            invalidateV1Signature();
            GetJarEntryDataDigestRequest dataDigestRequest =
                    new GetJarEntryDataDigestRequest(
                            entryName,
                            V1SchemeSigner.getJcaMessageDigestAlgorithm(mV1ContentDigestAlgorithm));
            mOutputJarEntryDigestRequests.put(entryName, dataDigestRequest);
            mOutputJarEntryDigests.remove(entryName);

            if ((!mDebuggableApkPermitted)
                    && (ApkUtils.ANDROID_MANIFEST_ZIP_ENTRY_NAME.equals(entryName))) {
                // To reject debuggable APKs we need to inspect the APK's AndroidManifest.xml to
                // check whether it declares that the APK is debuggable
                mOutputAndroidManifestEntryDataRequest = new GetJarEntryDataRequest(entryName);
                return new CompoundInspectJarEntryRequest(
                        entryName, mOutputAndroidManifestEntryDataRequest, dataDigestRequest);
            }

            return dataDigestRequest;
        }

        if (mSignatureExpectedOutputJarEntryNames.contains(entryName)) {
            // This entry is part of v1 signature generated by this engine. We need to check whether
            // the entry's data is as output by the engine.
            invalidateV1Signature();
            GetJarEntryDataRequest dataRequest;
            if (V1SchemeConstants.MANIFEST_ENTRY_NAME.equals(entryName)) {
                dataRequest = new GetJarEntryDataRequest(entryName);
                mInputJarManifestEntryDataRequest = dataRequest;
            } else {
                // If this entry is part of v1 signature which has been emitted by this engine,
                // check whether the output entry's data matches what the engine emitted.
                dataRequest =
                        (mEmittedSignatureJarEntryData.containsKey(entryName))
                                ? new GetJarEntryDataRequest(entryName)
                                : null;
            }

            if (dataRequest != null) {
                mOutputSignatureJarEntryDataRequests.put(entryName, dataRequest);
            }
            return dataRequest;
        }

        // This entry is not covered by v1 signature and isn't part of v1 signature.
        return null;
    }

    @Override
    public InputJarEntryInstructions.OutputPolicy inputJarEntryRemoved(String entryName) {
        checkNotClosed();
        return getInputJarEntryOutputPolicy(entryName);
    }

    @Override
    public void outputJarEntryRemoved(String entryName) {
        checkNotClosed();
        invalidateV2Signature();
        if (!mV1SigningEnabled) {
            return;
        }

        if (V1SchemeSigner.isJarEntryDigestNeededInManifest(entryName)) {
            // This entry is covered by v1 signature.
            invalidateV1Signature();
            mOutputJarEntryDigests.remove(entryName);
            mOutputJarEntryDigestRequests.remove(entryName);
            mOutputSignatureJarEntryDataRequests.remove(entryName);
            return;
        }

        if (mSignatureExpectedOutputJarEntryNames.contains(entryName)) {
            // This entry is part of the v1 signature generated by this engine.
            invalidateV1Signature();
            return;
        }
    }

    @Override
    public OutputJarSignatureRequest outputJarEntries()
            throws ApkFormatException, InvalidKeyException, SignatureException,
                    NoSuchAlgorithmException {
        checkNotClosed();

        if (!mV1SignaturePending) {
            return null;
        }

        if ((mInputJarManifestEntryDataRequest != null)
                && (!mInputJarManifestEntryDataRequest.isDone())) {
            throw new IllegalStateException(
                    "Still waiting to inspect input APK's "
                            + mInputJarManifestEntryDataRequest.getEntryName());
        }

        for (GetJarEntryDataDigestRequest digestRequest : mOutputJarEntryDigestRequests.values()) {
            String entryName = digestRequest.getEntryName();
            if (!digestRequest.isDone()) {
                throw new IllegalStateException(
                        "Still waiting to inspect output APK's " + entryName);
            }
            mOutputJarEntryDigests.put(entryName, digestRequest.getDigest());
        }
        if (isEligibleForSourceStamp()) {
            MessageDigest messageDigest =
                    MessageDigest.getInstance(
                            V1SchemeSigner.getJcaMessageDigestAlgorithm(mV1ContentDigestAlgorithm));
            messageDigest.update(generateSourceStampCertificateDigest());
            mOutputJarEntryDigests.put(
                    SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME, messageDigest.digest());
        }
        mOutputJarEntryDigestRequests.clear();

        for (GetJarEntryDataRequest dataRequest : mOutputSignatureJarEntryDataRequests.values()) {
            if (!dataRequest.isDone()) {
                throw new IllegalStateException(
                        "Still waiting to inspect output APK's " + dataRequest.getEntryName());
            }
        }

        List<Integer> apkSigningSchemeIds = new ArrayList<>();
        if (mV2SigningEnabled) {
            apkSigningSchemeIds.add(ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2);
        }
        if (mV3SigningEnabled) {
            apkSigningSchemeIds.add(ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3);
        }
        byte[] inputJarManifest =
                (mInputJarManifestEntryDataRequest != null)
                        ? mInputJarManifestEntryDataRequest.getData()
                        : null;
        if (isEligibleForSourceStamp()) {
            inputJarManifest =
                    V1SchemeSigner.generateManifestFile(
                                    mV1ContentDigestAlgorithm,
                                    mOutputJarEntryDigests,
                                    inputJarManifest)
                            .contents;
        }

        // Check whether the most recently used signature (if present) is still fine.
        checkOutputApkNotDebuggableIfDebuggableMustBeRejected();
        List<Pair<String, byte[]>> signatureZipEntries;
        if ((mAddV1SignatureRequest == null) || (!mAddV1SignatureRequest.isDone())) {
            try {
                signatureZipEntries =
                        V1SchemeSigner.sign(
                                mV1SignerConfigs,
                                mV1ContentDigestAlgorithm,
                                mOutputJarEntryDigests,
                                apkSigningSchemeIds,
                                inputJarManifest,
                                mCreatedBy);
            } catch (CertificateException e) {
                throw new SignatureException("Failed to generate v1 signature", e);
            }
        } else {
            V1SchemeSigner.OutputManifestFile newManifest =
                    V1SchemeSigner.generateManifestFile(
                            mV1ContentDigestAlgorithm, mOutputJarEntryDigests, inputJarManifest);
            byte[] emittedSignatureManifest =
                    mEmittedSignatureJarEntryData.get(V1SchemeConstants.MANIFEST_ENTRY_NAME);
            if (!Arrays.equals(newManifest.contents, emittedSignatureManifest)) {
                // Emitted v1 signature is no longer valid.
                try {
                    signatureZipEntries =
                            V1SchemeSigner.signManifest(
                                    mV1SignerConfigs,
                                    mV1ContentDigestAlgorithm,
                                    apkSigningSchemeIds,
                                    mCreatedBy,
                                    newManifest);
                } catch (CertificateException e) {
                    throw new SignatureException("Failed to generate v1 signature", e);
                }
            } else {
                // Emitted v1 signature is still valid. Check whether the signature is there in the
                // output.
                signatureZipEntries = new ArrayList<>();
                for (Map.Entry<String, byte[]> expectedOutputEntry :
                        mEmittedSignatureJarEntryData.entrySet()) {
                    String entryName = expectedOutputEntry.getKey();
                    byte[] expectedData = expectedOutputEntry.getValue();
                    GetJarEntryDataRequest actualDataRequest =
                            mOutputSignatureJarEntryDataRequests.get(entryName);
                    if (actualDataRequest == null) {
                        // This signature entry hasn't been output.
                        signatureZipEntries.add(Pair.of(entryName, expectedData));
                        continue;
                    }
                    byte[] actualData = actualDataRequest.getData();
                    if (!Arrays.equals(expectedData, actualData)) {
                        signatureZipEntries.add(Pair.of(entryName, expectedData));
                    }
                }
                if (signatureZipEntries.isEmpty()) {
                    // v1 signature in the output is valid
                    return null;
                }
                // v1 signature in the output is not valid.
            }
        }

        if (signatureZipEntries.isEmpty()) {
            // v1 signature in the output is valid
            mV1SignaturePending = false;
            return null;
        }

        List<OutputJarSignatureRequest.JarEntry> sigEntries =
                new ArrayList<>(signatureZipEntries.size());
        for (Pair<String, byte[]> entry : signatureZipEntries) {
            String entryName = entry.getFirst();
            byte[] entryData = entry.getSecond();
            sigEntries.add(new OutputJarSignatureRequest.JarEntry(entryName, entryData));
            mEmittedSignatureJarEntryData.put(entryName, entryData);
        }
        mAddV1SignatureRequest = new OutputJarSignatureRequestImpl(sigEntries);
        return mAddV1SignatureRequest;
    }

    @Deprecated
    @Override
    public OutputApkSigningBlockRequest outputZipSections(
            DataSource zipEntries, DataSource zipCentralDirectory, DataSource zipEocd)
            throws IOException, InvalidKeyException, SignatureException, NoSuchAlgorithmException {
        return outputZipSectionsInternal(zipEntries, zipCentralDirectory, zipEocd, false);
    }

    @Override
    public OutputApkSigningBlockRequest2 outputZipSections2(
            DataSource zipEntries, DataSource zipCentralDirectory, DataSource zipEocd)
            throws IOException, InvalidKeyException, SignatureException, NoSuchAlgorithmException {
        return outputZipSectionsInternal(zipEntries, zipCentralDirectory, zipEocd, true);
    }

    private OutputApkSigningBlockRequestImpl outputZipSectionsInternal(
            DataSource zipEntries,
            DataSource zipCentralDirectory,
            DataSource zipEocd,
            boolean apkSigningBlockPaddingSupported)
            throws IOException, InvalidKeyException, SignatureException, NoSuchAlgorithmException {
        checkNotClosed();
        checkV1SigningDoneIfEnabled();
        if (!mV2SigningEnabled && !mV3SigningEnabled && !isEligibleForSourceStamp()) {
            return null;
        }
        checkOutputApkNotDebuggableIfDebuggableMustBeRejected();

        // adjust to proper padding
        Pair<DataSource, Integer> paddingPair =
                ApkSigningBlockUtils.generateApkSigningBlockPadding(
                        zipEntries, apkSigningBlockPaddingSupported);
        DataSource beforeCentralDir = paddingPair.getFirst();
        int padSizeBeforeApkSigningBlock = paddingPair.getSecond();
        DataSource eocd = ApkSigningBlockUtils.copyWithModifiedCDOffset(beforeCentralDir, zipEocd);

        List<Pair<byte[], Integer>> signingSchemeBlocks = new ArrayList<>();
        ApkSigningBlockUtils.SigningSchemeBlockAndDigests v2SigningSchemeBlockAndDigests = null;
        ApkSigningBlockUtils.SigningSchemeBlockAndDigests v3SigningSchemeBlockAndDigests = null;
        // If the engine is configured to preserve previous signature blocks and any were found in
        // the existing APK signing block then add them to the list to be used to generate the
        // new APK signing block.
        if (mOtherSignersSignaturesPreserved && mPreservedSignatureBlocks != null
                && !mPreservedSignatureBlocks.isEmpty()) {
            signingSchemeBlocks.addAll(mPreservedSignatureBlocks);
        }

        // create APK Signature Scheme V2 Signature if requested
        if (mV2SigningEnabled) {
            invalidateV2Signature();
            List<ApkSigningBlockUtils.SignerConfig> v2SignerConfigs =
                    createV2SignerConfigs(apkSigningBlockPaddingSupported);
            v2SigningSchemeBlockAndDigests =
                    V2SchemeSigner.generateApkSignatureSchemeV2Block(
                            mExecutor,
                            beforeCentralDir,
                            zipCentralDirectory,
                            eocd,
                            v2SignerConfigs,
                            mV3SigningEnabled,
                            mOtherSignersSignaturesPreserved ? mPreservedV2Signers : null);
            signingSchemeBlocks.add(v2SigningSchemeBlockAndDigests.signingSchemeBlock);
        }
        if (mV3SigningEnabled) {
            invalidateV3Signature();
            List<ApkSigningBlockUtils.SignerConfig> v3SignerConfigs =
                    createV3SignerConfigs(apkSigningBlockPaddingSupported);
            List<ApkSigningBlockUtils.SignerConfig> v31SignerConfigs = processV31SignerConfigs(
                    v3SignerConfigs);
            if (v31SignerConfigs != null && v31SignerConfigs.size() > 0) {
                ApkSigningBlockUtils.SigningSchemeBlockAndDigests
                        v31SigningSchemeBlockAndDigests =
                        new V3SchemeSigner.Builder(beforeCentralDir, zipCentralDirectory, eocd,
                                v31SignerConfigs)
                                .setRunnablesExecutor(mExecutor)
                                .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID)
                                .build()
                                .generateApkSignatureSchemeV3BlockAndDigests();
                signingSchemeBlocks.add(v31SigningSchemeBlockAndDigests.signingSchemeBlock);
            }
            V3SchemeSigner.Builder builder = new V3SchemeSigner.Builder(beforeCentralDir,
                zipCentralDirectory, eocd, v3SignerConfigs)
                .setRunnablesExecutor(mExecutor)
                .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID);
            if (v31SignerConfigs != null && !v31SignerConfigs.isEmpty()) {
                // The V3.1 stripping protection writes the minimum SDK version from the targeted
                // signers as an additional attribute in the V3.0 signing block.
                int minSdkVersionForV31 = v31SignerConfigs.stream().mapToInt(
                        signer -> signer.minSdkVersion).min().orElse(MIN_SDK_WITH_V31_SUPPORT);
                builder.setMinSdkVersionForV31(minSdkVersionForV31);
            }
            v3SigningSchemeBlockAndDigests =
                builder.build().generateApkSignatureSchemeV3BlockAndDigests();
            signingSchemeBlocks.add(v3SigningSchemeBlockAndDigests.signingSchemeBlock);
        }
        if (isEligibleForSourceStamp()) {
            ApkSigningBlockUtils.SignerConfig sourceStampSignerConfig =
                    createSourceStampSignerConfig();
            Map<Integer, Map<ContentDigestAlgorithm, byte[]>> signatureSchemeDigestInfos =
                    new HashMap<>();
            if (mV3SigningEnabled) {
                signatureSchemeDigestInfos.put(
                        VERSION_APK_SIGNATURE_SCHEME_V3, v3SigningSchemeBlockAndDigests.digestInfo);
            }
            if (mV2SigningEnabled) {
                signatureSchemeDigestInfos.put(
                        VERSION_APK_SIGNATURE_SCHEME_V2, v2SigningSchemeBlockAndDigests.digestInfo);
            }
            if (mV1SigningEnabled) {
                Map<ContentDigestAlgorithm, byte[]> v1SigningSchemeDigests = new HashMap<>();
                try {
                    // Jar signing related variables must have been already populated at this point
                    // if V1 signing is enabled since it is happening before computations on the APK
                    // signing block (V2/V3/V4/SourceStamp signing).
                    byte[] inputJarManifest =
                            (mInputJarManifestEntryDataRequest != null)
                                    ? mInputJarManifestEntryDataRequest.getData()
                                    : null;
                    byte[] jarManifest =
                            V1SchemeSigner.generateManifestFile(
                                            mV1ContentDigestAlgorithm,
                                            mOutputJarEntryDigests,
                                            inputJarManifest)
                                    .contents;
                    // The digest of the jar manifest does not need to be computed in chunks due to
                    // the small size of the manifest.
                    v1SigningSchemeDigests.put(
                            ContentDigestAlgorithm.SHA256, computeSha256DigestBytes(jarManifest));
                } catch (ApkFormatException e) {
                    throw new RuntimeException("Failed to generate manifest file", e);
                }
                signatureSchemeDigestInfos.put(
                        VERSION_JAR_SIGNATURE_SCHEME, v1SigningSchemeDigests);
            }
            V2SourceStampSigner v2SourceStampSigner =
                    new V2SourceStampSigner.Builder(sourceStampSignerConfig,
                            signatureSchemeDigestInfos)
                            .setSourceStampTimestampEnabled(mSourceStampTimestampEnabled)
                            .build();
            signingSchemeBlocks.add(v2SourceStampSigner.generateSourceStampBlock());
        }

        // create APK Signing Block with v2 and/or v3 and/or SourceStamp blocks
        byte[] apkSigningBlock = ApkSigningBlockUtils.generateApkSigningBlock(signingSchemeBlocks);

        mAddSigningBlockRequest =
                new OutputApkSigningBlockRequestImpl(apkSigningBlock, padSizeBeforeApkSigningBlock);
        return mAddSigningBlockRequest;
    }

    @Override
    public void outputDone() {
        checkNotClosed();
        checkV1SigningDoneIfEnabled();
        checkSigningBlockDoneIfEnabled();
    }

    @Override
    public void signV4(DataSource dataSource, File outputFile, boolean ignoreFailures)
            throws SignatureException {
        if (outputFile == null) {
            if (ignoreFailures) {
                return;
            }
            throw new SignatureException("Missing V4 output file.");
        }
        try {
            V4SchemeSigner.SignerConfig v4SignerConfig = createV4SignerConfig();
            V4SchemeSigner.generateV4Signature(dataSource, v4SignerConfig, outputFile);
        } catch (InvalidKeyException | IOException | NoSuchAlgorithmException e) {
            if (ignoreFailures) {
                return;
            }
            throw new SignatureException("V4 signing failed", e);
        }
    }

    /** For external use only to generate V4 & tree separately. */
    public byte[] produceV4Signature(DataSource dataSource, OutputStream sigOutput)
            throws SignatureException {
        if (sigOutput == null) {
            throw new SignatureException("Missing V4 output streams.");
        }
        try {
            V4SchemeSigner.SignerConfig v4SignerConfig = createV4SignerConfig();
            Pair<V4Signature, byte[]> pair =
                    V4SchemeSigner.generateV4Signature(dataSource, v4SignerConfig);
            pair.getFirst().writeTo(sigOutput);
            return pair.getSecond();
        } catch (InvalidKeyException | IOException | NoSuchAlgorithmException e) {
            throw new SignatureException("V4 signing failed", e);
        }
    }

    @Override
    public boolean isEligibleForSourceStamp() {
        return mSourceStampSignerConfig != null
                && (mV2SigningEnabled || mV3SigningEnabled || mV1SigningEnabled);
    }

    @Override
    public byte[] generateSourceStampCertificateDigest() throws SignatureException {
        if (mSourceStampSignerConfig.getCertificates().isEmpty()) {
            throw new SignatureException("No certificates configured for stamp");
        }
        try {
            return computeSha256DigestBytes(
                    mSourceStampSignerConfig.getCertificates().get(0).getEncoded());
        } catch (CertificateEncodingException e) {
            throw new SignatureException("Failed to encode source stamp certificate", e);
        }
    }

    @Override
    public void close() {
        mClosed = true;

        mAddV1SignatureRequest = null;
        mInputJarManifestEntryDataRequest = null;
        mOutputAndroidManifestEntryDataRequest = null;
        mDebuggable = null;
        mOutputJarEntryDigestRequests.clear();
        mOutputJarEntryDigests.clear();
        mEmittedSignatureJarEntryData.clear();
        mOutputSignatureJarEntryDataRequests.clear();

        mAddSigningBlockRequest = null;
    }

    private void invalidateV1Signature() {
        if (mV1SigningEnabled) {
            mV1SignaturePending = true;
        }
        invalidateV2Signature();
    }

    private void invalidateV2Signature() {
        if (mV2SigningEnabled) {
            mV2SignaturePending = true;
            mAddSigningBlockRequest = null;
        }
    }

    private void invalidateV3Signature() {
        if (mV3SigningEnabled) {
            mV3SignaturePending = true;
            mAddSigningBlockRequest = null;
        }
    }

    private void checkNotClosed() {
        if (mClosed) {
            throw new IllegalStateException("Engine closed");
        }
    }

    private void checkV1SigningDoneIfEnabled() {
        if (!mV1SignaturePending) {
            return;
        }

        if (mAddV1SignatureRequest == null) {
            throw new IllegalStateException(
                    "v1 signature (JAR signature) not yet generated. Skipped outputJarEntries()?");
        }
        if (!mAddV1SignatureRequest.isDone()) {
            throw new IllegalStateException(
                    "v1 signature (JAR signature) addition requested by outputJarEntries() hasn't"
                            + " been fulfilled");
        }
        for (Map.Entry<String, byte[]> expectedOutputEntry :
                mEmittedSignatureJarEntryData.entrySet()) {
            String entryName = expectedOutputEntry.getKey();
            byte[] expectedData = expectedOutputEntry.getValue();
            GetJarEntryDataRequest actualDataRequest =
                    mOutputSignatureJarEntryDataRequests.get(entryName);
            if (actualDataRequest == null) {
                throw new IllegalStateException(
                        "APK entry "
                                + entryName
                                + " not yet output despite this having been"
                                + " requested");
            } else if (!actualDataRequest.isDone()) {
                throw new IllegalStateException(
                        "Still waiting to inspect output APK's " + entryName);
            }
            byte[] actualData = actualDataRequest.getData();
            if (!Arrays.equals(expectedData, actualData)) {
                throw new IllegalStateException(
                        "Output APK entry " + entryName + " data differs from what was requested");
            }
        }
        mV1SignaturePending = false;
    }

    private void checkSigningBlockDoneIfEnabled() {
        if (!mV2SignaturePending && !mV3SignaturePending) {
            return;
        }
        if (mAddSigningBlockRequest == null) {
            throw new IllegalStateException(
                    "Signed APK Signing BLock not yet generated. Skipped outputZipSections()?");
        }
        if (!mAddSigningBlockRequest.isDone()) {
            throw new IllegalStateException(
                    "APK Signing Block addition of signature(s) requested by"
                            + " outputZipSections() hasn't been fulfilled yet");
        }
        mAddSigningBlockRequest = null;
        mV2SignaturePending = false;
        mV3SignaturePending = false;
    }

    private void checkOutputApkNotDebuggableIfDebuggableMustBeRejected() throws SignatureException {
        if (mDebuggableApkPermitted) {
            return;
        }

        try {
            if (isOutputApkDebuggable()) {
                throw new SignatureException(
                        "APK is debuggable (see android:debuggable attribute) and this engine is"
                                + " configured to refuse to sign debuggable APKs");
            }
        } catch (ApkFormatException e) {
            throw new SignatureException("Failed to determine whether the APK is debuggable", e);
        }
    }

    /**
     * Returns whether the output APK is debuggable according to its {@code android:debuggable}
     * declaration.
     */
    private boolean isOutputApkDebuggable() throws ApkFormatException {
        if (mDebuggable != null) {
            return mDebuggable;
        }

        if (mOutputAndroidManifestEntryDataRequest == null) {
            throw new IllegalStateException(
                    "Cannot determine debuggable status of output APK because "
                            + ApkUtils.ANDROID_MANIFEST_ZIP_ENTRY_NAME
                            + " entry contents have not yet been requested");
        }

        if (!mOutputAndroidManifestEntryDataRequest.isDone()) {
            throw new IllegalStateException(
                    "Still waiting to inspect output APK's "
                            + mOutputAndroidManifestEntryDataRequest.getEntryName());
        }
        mDebuggable =
                ApkUtils.getDebuggableFromBinaryAndroidManifest(
                        ByteBuffer.wrap(mOutputAndroidManifestEntryDataRequest.getData()));
        return mDebuggable;
    }

    private void forgetOutputApkDebuggableStatus() {
        mDebuggable = null;
    }

    /** Returns the output policy for the provided input JAR entry. */
    private InputJarEntryInstructions.OutputPolicy getInputJarEntryOutputPolicy(String entryName) {
        if (mSignatureExpectedOutputJarEntryNames.contains(entryName)) {
            return InputJarEntryInstructions.OutputPolicy.OUTPUT_BY_ENGINE;
        }
        if ((mOtherSignersSignaturesPreserved)
                || (V1SchemeSigner.isJarEntryDigestNeededInManifest(entryName))) {
            return InputJarEntryInstructions.OutputPolicy.OUTPUT;
        }
        return InputJarEntryInstructions.OutputPolicy.SKIP;
    }

    private static class OutputJarSignatureRequestImpl implements OutputJarSignatureRequest {
        private final List<JarEntry> mAdditionalJarEntries;
        private volatile boolean mDone;

        private OutputJarSignatureRequestImpl(List<JarEntry> additionalZipEntries) {
            mAdditionalJarEntries =
                    Collections.unmodifiableList(new ArrayList<>(additionalZipEntries));
        }

        @Override
        public List<JarEntry> getAdditionalJarEntries() {
            return mAdditionalJarEntries;
        }

        @Override
        public void done() {
            mDone = true;
        }

        private boolean isDone() {
            return mDone;
        }
    }

    @SuppressWarnings("deprecation")
    private static class OutputApkSigningBlockRequestImpl
            implements OutputApkSigningBlockRequest, OutputApkSigningBlockRequest2 {
        private final byte[] mApkSigningBlock;
        private final int mPaddingBeforeApkSigningBlock;
        private volatile boolean mDone;

        private OutputApkSigningBlockRequestImpl(byte[] apkSigingBlock, int paddingBefore) {
            mApkSigningBlock = apkSigingBlock.clone();
            mPaddingBeforeApkSigningBlock = paddingBefore;
        }

        @Override
        public byte[] getApkSigningBlock() {
            return mApkSigningBlock.clone();
        }

        @Override
        public void done() {
            mDone = true;
        }

        private boolean isDone() {
            return mDone;
        }

        @Override
        public int getPaddingSizeBeforeApkSigningBlock() {
            return mPaddingBeforeApkSigningBlock;
        }
    }

    /** JAR entry inspection request which obtain the entry's uncompressed data. */
    private static class GetJarEntryDataRequest implements InspectJarEntryRequest {
        private final String mEntryName;
        private final Object mLock = new Object();

        private boolean mDone;
        private DataSink mDataSink;
        private ByteArrayOutputStream mDataSinkBuf;

        private GetJarEntryDataRequest(String entryName) {
            mEntryName = entryName;
        }

        @Override
        public String getEntryName() {
            return mEntryName;
        }

        @Override
        public DataSink getDataSink() {
            synchronized (mLock) {
                checkNotDone();
                if (mDataSinkBuf == null) {
                    mDataSinkBuf = new ByteArrayOutputStream();
                }
                if (mDataSink == null) {
                    mDataSink = DataSinks.asDataSink(mDataSinkBuf);
                }
                return mDataSink;
            }
        }

        @Override
        public void done() {
            synchronized (mLock) {
                if (mDone) {
                    return;
                }
                mDone = true;
            }
        }

        private boolean isDone() {
            synchronized (mLock) {
                return mDone;
            }
        }

        private void checkNotDone() throws IllegalStateException {
            synchronized (mLock) {
                if (mDone) {
                    throw new IllegalStateException("Already done");
                }
            }
        }

        private byte[] getData() {
            synchronized (mLock) {
                if (!mDone) {
                    throw new IllegalStateException("Not yet done");
                }
                return (mDataSinkBuf != null) ? mDataSinkBuf.toByteArray() : new byte[0];
            }
        }
    }

    /** JAR entry inspection request which obtains the digest of the entry's uncompressed data. */
    private static class GetJarEntryDataDigestRequest implements InspectJarEntryRequest {
        private final String mEntryName;
        private final String mJcaDigestAlgorithm;
        private final Object mLock = new Object();

        private boolean mDone;
        private DataSink mDataSink;
        private MessageDigest mMessageDigest;
        private byte[] mDigest;

        private GetJarEntryDataDigestRequest(String entryName, String jcaDigestAlgorithm) {
            mEntryName = entryName;
            mJcaDigestAlgorithm = jcaDigestAlgorithm;
        }

        @Override
        public String getEntryName() {
            return mEntryName;
        }

        @Override
        public DataSink getDataSink() {
            synchronized (mLock) {
                checkNotDone();
                if (mDataSink == null) {
                    mDataSink = DataSinks.asDataSink(getMessageDigest());
                }
                return mDataSink;
            }
        }

        private MessageDigest getMessageDigest() {
            synchronized (mLock) {
                if (mMessageDigest == null) {
                    try {
                        mMessageDigest = MessageDigest.getInstance(mJcaDigestAlgorithm);
                    } catch (NoSuchAlgorithmException e) {
                        throw new RuntimeException(
                                mJcaDigestAlgorithm + " MessageDigest not available", e);
                    }
                }
                return mMessageDigest;
            }
        }

        @Override
        public void done() {
            synchronized (mLock) {
                if (mDone) {
                    return;
                }
                mDone = true;
                mDigest = getMessageDigest().digest();
                mMessageDigest = null;
                mDataSink = null;
            }
        }

        private boolean isDone() {
            synchronized (mLock) {
                return mDone;
            }
        }

        private void checkNotDone() throws IllegalStateException {
            synchronized (mLock) {
                if (mDone) {
                    throw new IllegalStateException("Already done");
                }
            }
        }

        private byte[] getDigest() {
            synchronized (mLock) {
                if (!mDone) {
                    throw new IllegalStateException("Not yet done");
                }
                return mDigest.clone();
            }
        }
    }

    /** JAR entry inspection request which transparently satisfies multiple such requests. */
    private static class CompoundInspectJarEntryRequest implements InspectJarEntryRequest {
        private final String mEntryName;
        private final InspectJarEntryRequest[] mRequests;
        private final Object mLock = new Object();

        private DataSink mSink;

        private CompoundInspectJarEntryRequest(
                String entryName, InspectJarEntryRequest... requests) {
            mEntryName = entryName;
            mRequests = requests;
        }

        @Override
        public String getEntryName() {
            return mEntryName;
        }

        @Override
        public DataSink getDataSink() {
            synchronized (mLock) {
                if (mSink == null) {
                    DataSink[] sinks = new DataSink[mRequests.length];
                    for (int i = 0; i < sinks.length; i++) {
                        sinks[i] = mRequests[i].getDataSink();
                    }
                    mSink = new TeeDataSink(sinks);
                }
                return mSink;
            }
        }

        @Override
        public void done() {
            for (InspectJarEntryRequest request : mRequests) {
                request.done();
            }
        }
    }

    /**
     * Configuration of a signer.
     *
     * <p>Use {@link Builder} to obtain configuration instances.
     */
    public static class SignerConfig {
        private final String mName;
        private final PrivateKey mPrivateKey;
        private final List<X509Certificate> mCertificates;
        private final boolean mDeterministicDsaSigning;
        private final int mMinSdkVersion;
        private final boolean mSignerTargetsDevRelease;
        private final SigningCertificateLineage mSigningCertificateLineage;

        private SignerConfig(Builder builder) {
            mName = builder.mName;
            mPrivateKey = builder.mPrivateKey;
            mCertificates = Collections.unmodifiableList(new ArrayList<>(builder.mCertificates));
            mDeterministicDsaSigning = builder.mDeterministicDsaSigning;
            mMinSdkVersion = builder.mMinSdkVersion;
            mSignerTargetsDevRelease = builder.mSignerTargetsDevRelease;
            mSigningCertificateLineage = builder.mSigningCertificateLineage;
        }

        /** Returns the name of this signer. */
        public String getName() {
            return mName;
        }

        /** Returns the signing key of this signer. */
        public PrivateKey getPrivateKey() {
            return mPrivateKey;
        }

        /**
         * Returns the certificate(s) of this signer. The first certificate's public key corresponds
         * to this signer's private key.
         */
        public List<X509Certificate> getCertificates() {
            return mCertificates;
        }

        /**
         * If this signer is a DSA signer, whether or not the signing is done deterministically.
         */
        public boolean getDeterministicDsaSigning() {
            return mDeterministicDsaSigning;
        }

        /** Returns the minimum SDK version for which this signer should be used. */
        public int getMinSdkVersion() {
            return mMinSdkVersion;
        }

        /** Returns whether this signer targets a development release. */
        public boolean getSignerTargetsDevRelease() {
            return mSignerTargetsDevRelease;
        }

        /** Returns the {@link SigningCertificateLineage} for this signer. */
        public SigningCertificateLineage getSigningCertificateLineage() {
            return mSigningCertificateLineage;
        }

        /** Builder of {@link SignerConfig} instances. */
        public static class Builder {
            private final String mName;
            private final PrivateKey mPrivateKey;
            private final List<X509Certificate> mCertificates;
            private final boolean mDeterministicDsaSigning;
            private int mMinSdkVersion;
            private boolean mSignerTargetsDevRelease;
            private SigningCertificateLineage mSigningCertificateLineage;

            /**
             * Constructs a new {@code Builder}.
             *
             * @param name signer's name. The name is reflected in the name of files comprising the
             *     JAR signature of the APK.
             * @param privateKey signing key
             * @param certificates list of one or more X.509 certificates. The subject public key of
             *     the first certificate must correspond to the {@code privateKey}.
             */
            public Builder(String name, PrivateKey privateKey, List<X509Certificate> certificates) {
                this(name, privateKey, certificates, false);
            }

            /**
             * Constructs a new {@code Builder}.
             *
             * @param name signer's name. The name is reflected in the name of files comprising the
             *     JAR signature of the APK.
             * @param privateKey signing key
             * @param certificates list of one or more X.509 certificates. The subject public key of
             *     the first certificate must correspond to the {@code privateKey}.
             * @param deterministicDsaSigning When signing using DSA, whether or not the
             * deterministic signing algorithm variant (RFC6979) should be used.
             */
            public Builder(String name, PrivateKey privateKey, List<X509Certificate> certificates,
                    boolean deterministicDsaSigning) {
                if (name.isEmpty()) {
                    throw new IllegalArgumentException("Empty name");
                }
                mName = name;
                mPrivateKey = privateKey;
                mCertificates = new ArrayList<>(certificates);
                mDeterministicDsaSigning = deterministicDsaSigning;
            }

            /** @see #setLineageForMinSdkVersion(SigningCertificateLineage, int) */
            public Builder setMinSdkVersion(int minSdkVersion) {
                return setLineageForMinSdkVersion(null, minSdkVersion);
            }

            /**
             * Sets the specified {@code minSdkVersion} as the minimum Android platform version
             * (API level) for which the provided {@code lineage} (where applicable) should be used
             * to produce the APK's signature. This method is useful if callers want to specify a
             * particular rotated signer or lineage with restricted capabilities for later
             * platform releases.
             *
             * <p><em>Note:</em>>The V1 and V2 signature schemes do not support key rotation and
             * signing lineages with capabilities; only an app's original signer(s) can be used for
             * the V1 and V2 signature blocks. Because of this, only a value of {@code
             * minSdkVersion} >= 28 (Android P) where support for the V3 signature scheme was
             * introduced can be specified.
             *
             * <p><em>Note:</em>Due to limitations with platform targeting in the V3.0 signature
             * scheme, specifying a {@code minSdkVersion} value <= 32 (Android Sv2) will result in
             * the current {@code SignerConfig} being used in the V3.0 signing block and applied to
             * Android P through at least Sv2 (and later depending on the {@code minSdkVersion} for
             * subsequent {@code SignerConfig} instances). Because of this, only a single {@code
             * SignerConfig} can be instantiated with a minimum SDK version <= 32.
             *
             * @param lineage the {@code SigningCertificateLineage} to target the specified {@code
             *                minSdkVersion}
             * @param minSdkVersion the minimum SDK version for which this {@code SignerConfig}
             *                      should be used
             * @return this {@code Builder} instance
             *
             * @throws IllegalArgumentException if the provided {@code minSdkVersion} < 28 or the
             * certificate provided in the constructor is not in the specified {@code lineage}.
             */
            public Builder setLineageForMinSdkVersion(SigningCertificateLineage lineage,
                    int minSdkVersion) {
                if (minSdkVersion < AndroidSdkVersion.P) {
                    throw new IllegalArgumentException(
                            "SDK targeted signing config is only supported with the V3 signature "
                                    + "scheme on Android P (SDK version "
                                    + AndroidSdkVersion.P + ") and later");
                }
                if (minSdkVersion < MIN_SDK_WITH_V31_SUPPORT) {
                    minSdkVersion = AndroidSdkVersion.P;
                }
                mMinSdkVersion = minSdkVersion;
                // If a lineage is provided, ensure the signing certificate for this signer is in
                // the lineage; in the case of multiple signing certificates, the first is always
                // used in the lineage.
                if (lineage != null && !lineage.isCertificateInLineage(mCertificates.get(0))) {
                    throw new IllegalArgumentException(
                            "The provided lineage does not contain the signing certificate, "
                                    + mCertificates.get(0).getSubjectDN()
                                    + ", for this SignerConfig");
                }
                mSigningCertificateLineage = lineage;
                return this;
            }

            /**
             * Sets whether this signer's min SDK version is intended to target a development
             * release.
             *
             * <p>This is primarily required for a signer testing on a platform's development
             * release; however, it is recommended that signer's use the latest development SDK
             * version instead of explicitly specifying this boolean. This class will properly
             * handle an SDK that is currently targeting a development release and will use the
             * finalized SDK version on release.
             */
            private Builder setSignerTargetsDevRelease(boolean signerTargetsDevRelease) {
                if (signerTargetsDevRelease && mMinSdkVersion < MIN_SDK_WITH_V31_SUPPORT) {
                    throw new IllegalArgumentException(
                            "Rotation can only target a development release for signers targeting "
                                    + MIN_SDK_WITH_V31_SUPPORT + " or later");
                }
                mSignerTargetsDevRelease = signerTargetsDevRelease;
                return this;
            }


            /**
             * Returns a new {@code SignerConfig} instance configured based on the configuration of
             * this builder.
             */
            public SignerConfig build() {
                return new SignerConfig(this);
            }
        }
    }

    /** Builder of {@link DefaultApkSignerEngine} instances. */
    public static class Builder {
        private List<SignerConfig> mSignerConfigs;
        private List<SignerConfig> mTargetedSignerConfigs;
        private SignerConfig mStampSignerConfig;
        private SigningCertificateLineage mSourceStampSigningCertificateLineage;
        private boolean mSourceStampTimestampEnabled = true;
        private final int mMinSdkVersion;

        private boolean mV1SigningEnabled = true;
        private boolean mV2SigningEnabled = true;
        private boolean mV3SigningEnabled = true;
        private int mRotationMinSdkVersion = V3SchemeConstants.DEFAULT_ROTATION_MIN_SDK_VERSION;
        private boolean mRotationTargetsDevRelease = false;
        private boolean mVerityEnabled = false;
        private boolean mDebuggableApkPermitted = true;
        private boolean mOtherSignersSignaturesPreserved;
        private String mCreatedBy = "1.0 (Android)";

        private SigningCertificateLineage mSigningCertificateLineage;

        // APK Signature Scheme v3 only supports a single signing certificate, so to move to v3
        // signing by default, but not require prior clients to update to explicitly disable v3
        // signing for multiple signers, we modify the mV3SigningEnabled depending on the provided
        // inputs (multiple signers and mSigningCertificateLineage in particular).  Maintain two
        // extra variables to record whether or not mV3SigningEnabled has been set directly by a
        // client and so should override the default behavior.
        private boolean mV3SigningExplicitlyDisabled = false;
        private boolean mV3SigningExplicitlyEnabled = false;

        /**
         * Constructs a new {@code Builder}.
         *
         * @param signerConfigs information about signers with which the APK will be signed. At
         *     least one signer configuration must be provided.
         * @param minSdkVersion API Level of the oldest Android platform on which the APK is
         *     supposed to be installed. See {@code minSdkVersion} attribute in the APK's {@code
         *     AndroidManifest.xml}. The higher the version, the stronger signing features will be
         *     enabled.
         */
        public Builder(List<SignerConfig> signerConfigs, int minSdkVersion) {
            if (signerConfigs.isEmpty()) {
                throw new IllegalArgumentException("At least one signer config must be provided");
            }
            if (signerConfigs.size() > 1) {
                // APK Signature Scheme v3 only supports single signer, unless a
                // SigningCertificateLineage is provided, in which case this will be reset to true,
                // since we don't yet have a v4 scheme about which to worry
                mV3SigningEnabled = false;
            }
            mSignerConfigs = new ArrayList<>(signerConfigs);
            mMinSdkVersion = minSdkVersion;
        }

        /**
         * Sets the APK signature schemes that should be enabled based on the options provided by
         * the caller.
         */
        private void setEnabledSignatureSchemes() {
            if (mV3SigningExplicitlyDisabled && mV3SigningExplicitlyEnabled) {
                throw new IllegalStateException(
                        "Builder configured to both enable and disable APK "
                                + "Signature Scheme v3 signing");
            }
            if (mV3SigningExplicitlyDisabled) {
                mV3SigningEnabled = false;
            } else if (mV3SigningExplicitlyEnabled) {
                mV3SigningEnabled = true;
            }
        }

        /**
         * Sets the SDK targeted signer configs based on the signing config and rotation options
         * provided by the caller.
         *
         * @throws InvalidKeyException if a {@link SigningCertificateLineage} cannot be created
         * from the provided options
         */
        private void setTargetedSignerConfigs() throws InvalidKeyException {
            // If the caller specified any SDK targeted signer configs, then the min SDK version
            // should be set for those configs, all others should have a default 0 min SDK version.
            mSignerConfigs.sort(((signerConfig1, signerConfig2) -> signerConfig1.getMinSdkVersion()
                    - signerConfig2.getMinSdkVersion()));
            // With the signer configs sorted, find the first targeted signer config with a min
            // SDK version > 0 to create the separate targeted signer configs.
            mTargetedSignerConfigs = new ArrayList<>();
            for (int i = 0; i < mSignerConfigs.size(); i++) {
                if (mSignerConfigs.get(i).getMinSdkVersion() > 0) {
                    mTargetedSignerConfigs = mSignerConfigs.subList(i, mSignerConfigs.size());
                    mSignerConfigs = mSignerConfigs.subList(0, i);
                    break;
                }
            }

            // A lineage provided outside a targeted signing config is intended for the original
            // rotation; sort the untargeted signing configs based on this lineage and create a new
            // targeted signing config for the initial rotation.
            if (mSigningCertificateLineage != null) {
                if (!mTargetedSignerConfigs.isEmpty()) {
                    // Only the initial rotation can use the rotation-min-sdk-version; all
                    // subsequent targeted rotations must use targeted signing configs.
                    int firstTargetedSdkVersion = mTargetedSignerConfigs.get(0).getMinSdkVersion();
                    if (mRotationMinSdkVersion >= firstTargetedSdkVersion) {
                        throw new IllegalStateException(
                                "The rotation-min-sdk-version, " + mRotationMinSdkVersion
                                        + ", must be less than the first targeted SDK version, "
                                        + firstTargetedSdkVersion);
                    }
                }
                try {
                    mSignerConfigs = mSigningCertificateLineage.sortSignerConfigs(mSignerConfigs);
                } catch (IllegalArgumentException e) {
                    throw new IllegalStateException(
                            "Provided signer configs do not match the "
                                    + "provided SigningCertificateLineage",
                            e);
                }
                // Get the last signer in the lineage, create a new targeted signer from it,
                // and add it as a targeted signer config.
                SignerConfig rotatedSignerConfig = mSignerConfigs.remove(mSignerConfigs.size() - 1);
                SignerConfig.Builder rotatedConfigBuilder = new SignerConfig.Builder(
                        rotatedSignerConfig.getName(), rotatedSignerConfig.getPrivateKey(),
                        rotatedSignerConfig.getCertificates(),
                        rotatedSignerConfig.getDeterministicDsaSigning());
                rotatedConfigBuilder.setLineageForMinSdkVersion(mSigningCertificateLineage,
                        mRotationMinSdkVersion);
                rotatedConfigBuilder.setSignerTargetsDevRelease(mRotationTargetsDevRelease);
                mTargetedSignerConfigs.add(0, rotatedConfigBuilder.build());
            }
            mSigningCertificateLineage = mergeTargetedSigningConfigLineages();
        }

        /**
         * Merges and returns the lineages from any caller provided SDK targeted {@link
         * SignerConfig} instances with an optional {@code lineage} specified as part of the general
         * signing config.
         *
         * <p>If multiple signing configs target the same SDK version, or if any of the lineages
         * cannot be merged, then an {@code IllegalStateException} is thrown.
         */
        private SigningCertificateLineage mergeTargetedSigningConfigLineages()
                throws InvalidKeyException {
            SigningCertificateLineage mergedLineage = null;
            int prevSdkVersion = 0;
            for (SignerConfig signerConfig : mTargetedSignerConfigs) {
                int signerMinSdkVersion = signerConfig.getMinSdkVersion();
                if (signerMinSdkVersion < AndroidSdkVersion.P) {
                    throw new IllegalStateException(
                            "Targeted signing config is not supported prior to SDK version "
                                    + AndroidSdkVersion.P + "; received value "
                                    + signerMinSdkVersion);
                }
                SigningCertificateLineage signerLineage =
                        signerConfig.getSigningCertificateLineage();
                // It is possible for a lineage to be null if the user is using one of the
                // signers from the lineage as the only signer to target an SDK version; create
                // a single element lineage to verify the signer is part of the merged lineage.
                if (signerLineage == null) {
                    try {
                        signerLineage = new SigningCertificateLineage.Builder(
                                new SigningCertificateLineage.SignerConfig.Builder(
                                        signerConfig.mPrivateKey,
                                        signerConfig.mCertificates.get(0))
                                        .build())
                                .build();
                    } catch (CertificateEncodingException | NoSuchAlgorithmException
                            | SignatureException e) {
                        throw new IllegalStateException(
                                "Unable to create a SignerConfig for signer from certificate "
                                        + signerConfig.mCertificates.get(0).getSubjectDN());
                    }
                }
                // The V3.0 signature scheme does not support verified targeted SDK signing
                // configs; if a signer is targeting any SDK version < T, then it will
                // target P with the V3.0 signature scheme.
                if (signerMinSdkVersion < AndroidSdkVersion.T) {
                    signerMinSdkVersion = AndroidSdkVersion.P;
                }
                // Ensure there are no SignerConfigs targeting the same SDK version.
                if (signerMinSdkVersion == prevSdkVersion) {
                    throw new IllegalStateException(
                            "Multiple SignerConfigs were found targeting SDK version "
                                    + signerMinSdkVersion);
                }
                // If multiple lineages have been provided, then verify each subsequent lineage
                // is a valid descendant or ancestor of the previously merged lineages.
                if (mergedLineage == null) {
                    mergedLineage = signerLineage;
                } else {
                    try {
                        mergedLineage = mergedLineage.mergeLineageWith(signerLineage);
                    } catch (IllegalArgumentException e) {
                        throw new IllegalStateException(
                                "The provided lineage targeting SDK " + signerMinSdkVersion
                                        + " is not in the signing history of the other targeted "
                                        + "signing configs", e);
                    }
                }
                prevSdkVersion = signerMinSdkVersion;
            }
            return mergedLineage;
        }

        /**
         * Returns a new {@code DefaultApkSignerEngine} instance configured based on the
         * configuration of this builder.
         */
        public DefaultApkSignerEngine build() throws InvalidKeyException {
            setEnabledSignatureSchemes();
            setTargetedSignerConfigs();

            // make sure our signers are appropriately setup
            if (mSigningCertificateLineage != null) {
                if (!mV3SigningEnabled && mSignerConfigs.size() > 1) {
                    // this is a strange situation: we've provided a valid rotation history, but
                    // are only signing with v1/v2.  blow up, since we don't know for sure with
                    // which signer the user intended to sign
                    throw new IllegalStateException(
                            "Provided multiple signers which are part of the"
                                    + " SigningCertificateLineage, but not signing with APK"
                                    + " Signature Scheme v3");
                }
            } else if (mV3SigningEnabled && mSignerConfigs.size() > 1) {
                throw new IllegalStateException(
                        "Multiple signing certificates provided for use with APK Signature Scheme"
                                + " v3 without an accompanying SigningCertificateLineage");
            }

            return new DefaultApkSignerEngine(
                    mSignerConfigs,
                    mTargetedSignerConfigs,
                    mStampSignerConfig,
                    mSourceStampSigningCertificateLineage,
                    mSourceStampTimestampEnabled,
                    mMinSdkVersion,
                    mV1SigningEnabled,
                    mV2SigningEnabled,
                    mV3SigningEnabled,
                    mVerityEnabled,
                    mDebuggableApkPermitted,
                    mOtherSignersSignaturesPreserved,
                    mCreatedBy,
                    mSigningCertificateLineage);
        }

        /** Sets the signer configuration for the SourceStamp to be embedded in the APK. */
        public Builder setStampSignerConfig(SignerConfig stampSignerConfig) {
            mStampSignerConfig = stampSignerConfig;
            return this;
        }

        /**
         * Sets the source stamp {@link SigningCertificateLineage}. This structure provides proof of
         * signing certificate rotation for certificates previously used to sign source stamps.
         */
        public Builder setSourceStampSigningCertificateLineage(
                SigningCertificateLineage sourceStampSigningCertificateLineage) {
            mSourceStampSigningCertificateLineage = sourceStampSigningCertificateLineage;
            return this;
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
         * Sets whether the APK should be signed using JAR signing (aka v1 signature scheme).
         *
         * <p>By default, the APK will be signed using this scheme.
         */
        public Builder setV1SigningEnabled(boolean enabled) {
            mV1SigningEnabled = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed using APK Signature Scheme v2 (aka v2 signature
         * scheme).
         *
         * <p>By default, the APK will be signed using this scheme.
         */
        public Builder setV2SigningEnabled(boolean enabled) {
            mV2SigningEnabled = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed using APK Signature Scheme v3 (aka v3 signature
         * scheme).
         *
         * <p>By default, the APK will be signed using this scheme.
         */
        public Builder setV3SigningEnabled(boolean enabled) {
            mV3SigningEnabled = enabled;
            if (enabled) {
                mV3SigningExplicitlyEnabled = true;
            } else {
                mV3SigningExplicitlyDisabled = true;
            }
            return this;
        }

        /**
         * Sets whether the APK should be signed using the verity signature algorithm in the v2 and
         * v3 signature blocks.
         *
         * <p>By default, the APK will be signed using the verity signature algorithm for the v2 and
         * v3 signature schemes.
         */
        public Builder setVerityEnabled(boolean enabled) {
            mVerityEnabled = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed even if it is marked as debuggable ({@code
         * android:debuggable="true"} in its {@code AndroidManifest.xml}). For backward
         * compatibility reasons, the default value of this setting is {@code true}.
         *
         * <p>It is dangerous to sign debuggable APKs with production/release keys because Android
         * platform loosens security checks for such APKs. For example, arbitrary unauthorized code
         * may be executed in the context of such an app by anybody with ADB shell access.
         */
        public Builder setDebuggableApkPermitted(boolean permitted) {
            mDebuggableApkPermitted = permitted;
            return this;
        }

        /**
         * Sets whether signatures produced by signers other than the ones configured in this engine
         * should be copied from the input APK to the output APK.
         *
         * <p>By default, signatures of other signers are omitted from the output APK.
         */
        public Builder setOtherSignersSignaturesPreserved(boolean preserved) {
            mOtherSignersSignaturesPreserved = preserved;
            return this;
        }

        /** Sets the value of the {@code Created-By} field in JAR signature files. */
        public Builder setCreatedBy(String createdBy) {
            if (createdBy == null) {
                throw new NullPointerException();
            }
            mCreatedBy = createdBy;
            return this;
        }

        /**
         * Sets the {@link SigningCertificateLineage} to use with the v3 signature scheme. This
         * structure provides proof of signing certificate rotation linking {@link SignerConfig}
         * objects to previous ones.
         */
        public Builder setSigningCertificateLineage(
                SigningCertificateLineage signingCertificateLineage) {
            if (signingCertificateLineage != null) {
                mV3SigningEnabled = true;
                mSigningCertificateLineage = signingCertificateLineage;
            }
            return this;
        }

        /**
         * Sets the minimum Android platform version (API Level) for which an APK's rotated signing
         * key should be used to produce the APK's signature. The original signing key for the APK
         * will be used for all previous platform versions. If a rotated key with signing lineage is
         * not provided then this method is a noop.
         *
         * <p>By default, if a signing lineage is specified with {@link
         * #setSigningCertificateLineage(SigningCertificateLineage)}, then the APK Signature Scheme
         * V3.1 will be used to only apply the rotation on devices running Android T+.
         *
         * <p><em>Note:</em>Specifying a {@code minSdkVersion} value <= 32 (Android Sv2) will result
         * in the original V3 signing block being used without platform targeting.
         */
        public Builder setMinSdkVersionForRotation(int minSdkVersion) {
            // If the provided SDK version does not support v3.1, then use the default SDK version
            // with rotation support.
            if (minSdkVersion < MIN_SDK_WITH_V31_SUPPORT) {
                mRotationMinSdkVersion = MIN_SDK_WITH_V3_SUPPORT;
            } else {
                mRotationMinSdkVersion = minSdkVersion;
            }
            return this;
        }

        /**
         * Sets whether the rotation-min-sdk-version is intended to target a development release;
         * this is primarily required after the T SDK is finalized, and an APK needs to target U
         * during its development cycle for rotation.
         *
         * <p>This is only required after the T SDK is finalized since S and earlier releases do
         * not know about the V3.1 block ID, but once T is released and work begins on U, U will
         * use the SDK version of T during development. Specifying a rotation-min-sdk-version of T's
         * SDK version along with setting {@code enabled} to true will allow an APK to use the
         * rotated key on a device running U while causing this to be bypassed for T.
         *
         * <p><em>Note:</em>If the rotation-min-sdk-version is less than or equal to 32 (Android
         * Sv2), then the rotated signing key will be used in the v3.0 signing block and this call
         * will be a noop.
         */
        public Builder setRotationTargetsDevRelease(boolean enabled) {
            mRotationTargetsDevRelease = enabled;
            return this;
        }
    }
}
