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

package com.android.apksig.internal.apk.v4;

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V31;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeCertificates;
import static com.android.apksig.internal.apk.v2.V2SchemeConstants.APK_SIGNATURE_SCHEME_V2_BLOCK_ID;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;

import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.apk.SignatureInfo;
import com.android.apksig.internal.apk.v2.V2SchemeVerifier;
import com.android.apksig.internal.apk.v3.V3SchemeSigner;
import com.android.apksig.internal.apk.v3.V3SchemeVerifier;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.util.DataSource;
import com.android.apksig.zip.ZipFormatException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * APK Signature Scheme V4 signer. V4 scheme file contains 2 mandatory fields - used during
 * installation. And optional verity tree - has to be present during session commit.
 * <p>
 * The fields:
 * <p>
 * 1. hashingInfo - verity root hash and hashing info,
 * 2. signingInfo - certificate, public key and signature,
 * For more details see V4Signature.
 * </p>
 * (optional) verityTree: integer size prepended bytes of the verity hash tree.
 * <p>
 */
public abstract class V4SchemeSigner {
    /**
     * Hidden constructor to prevent instantiation.
     */
    private V4SchemeSigner() {
    }

    public static class SignerConfig {
        final public ApkSigningBlockUtils.SignerConfig v4Config;
        final public ApkSigningBlockUtils.SignerConfig v41Config;

        public SignerConfig(List<ApkSigningBlockUtils.SignerConfig> v4Configs,
                List<ApkSigningBlockUtils.SignerConfig> v41Configs) throws InvalidKeyException {
            if (v4Configs == null || v4Configs.size() != 1) {
                throw new InvalidKeyException("Only accepting one signer config for V4 Signature.");
            }
            if (v41Configs != null && v41Configs.size() != 1) {
                throw new InvalidKeyException("Only accepting one signer config for V4.1 Signature.");
            }
            this.v4Config = v4Configs.get(0);
            this.v41Config = v41Configs != null ? v41Configs.get(0) : null;
        }
    }

    /**
     * Based on a public key, return a signing algorithm that supports verity.
     */
    public static List<SignatureAlgorithm> getSuggestedSignatureAlgorithms(PublicKey signingKey,
            int minSdkVersion, boolean apkSigningBlockPaddingSupported,
            boolean deterministicDsaSigning)
            throws InvalidKeyException {
        List<SignatureAlgorithm> algorithms = V3SchemeSigner.getSuggestedSignatureAlgorithms(
                signingKey, minSdkVersion,
                apkSigningBlockPaddingSupported, deterministicDsaSigning);
        // Keeping only supported algorithms.
        for (Iterator<SignatureAlgorithm> iter = algorithms.listIterator(); iter.hasNext(); ) {
            final SignatureAlgorithm algorithm = iter.next();
            if (!isSupported(algorithm.getContentDigestAlgorithm(), false)) {
                iter.remove();
            }
        }
        return algorithms;
    }

    /**
     * Compute hash tree and generate v4 signature for a given APK. Write the serialized data to
     * output file.
     */
    public static void generateV4Signature(
        DataSource apkContent, SignerConfig signerConfig, File outputFile)
        throws IOException, InvalidKeyException, NoSuchAlgorithmException {
      Pair<V4Signature, byte[]> pair = generateV4Signature(apkContent, signerConfig);
      try (final OutputStream output = new FileOutputStream(outputFile)) {
        pair.getFirst().writeTo(output);
        V4Signature.writeBytes(output, pair.getSecond());
      } catch (IOException e) {
        outputFile.delete();
        throw e;
      }
    }

    /** Generate v4 signature and hash tree for a given APK. */
    public static Pair<V4Signature, byte[]> generateV4Signature(
            DataSource apkContent,
            SignerConfig signerConfig)
            throws IOException, InvalidKeyException, NoSuchAlgorithmException {
        // Salt has to stay empty for fs-verity compatibility.
        final byte[] salt = null;
        // Not used by apksigner.
        final byte[] additionalData = null;

        final long fileSize = apkContent.size();

        // Obtaining the strongest supported digest for each of the v2/v3/v3.1 blocks
        // (CHUNKED_SHA256 or CHUNKED_SHA512).
        final Map<Integer, byte[]> apkDigests = getApkDigests(apkContent);

        // Obtaining the merkle tree and the root hash in verity format.
        ApkSigningBlockUtils.VerityTreeAndDigest verityContentDigestInfo =
                ApkSigningBlockUtils.computeChunkVerityTreeAndDigest(apkContent);

        final ContentDigestAlgorithm verityContentDigestAlgorithm =
                verityContentDigestInfo.contentDigestAlgorithm;
        final byte[] rootHash = verityContentDigestInfo.rootHash;
        final byte[] tree = verityContentDigestInfo.tree;

        final Pair<Integer, Byte> hashingAlgorithmBlockSizePair = convertToV4HashingInfo(
                verityContentDigestAlgorithm);
        final V4Signature.HashingInfo hashingInfo = new V4Signature.HashingInfo(
                hashingAlgorithmBlockSizePair.getFirst(), hashingAlgorithmBlockSizePair.getSecond(),
                salt, rootHash);

        // Generating SigningInfo and combining everything into V4Signature.
        final V4Signature signature;
        try {
            signature = generateSignature(signerConfig, hashingInfo, apkDigests, additionalData,
                    fileSize);
        } catch (InvalidKeyException | SignatureException | CertificateEncodingException e) {
            throw new InvalidKeyException("Signer failed", e);
        }

        return Pair.of(signature, tree);
    }

    private static V4Signature.SigningInfo generateSigningInfo(
            ApkSigningBlockUtils.SignerConfig signerConfig,
            V4Signature.HashingInfo hashingInfo,
            byte[] apkDigest, byte[] additionalData, long fileSize)
            throws NoSuchAlgorithmException, InvalidKeyException, SignatureException,
            CertificateEncodingException {
        if (signerConfig.certificates.isEmpty()) {
            throw new SignatureException("No certificates configured for signer");
        }
        if (signerConfig.certificates.size() != 1) {
            throw new CertificateEncodingException("Should only have one certificate");
        }

        // Collecting data for signing.
        final PublicKey publicKey = signerConfig.certificates.get(0).getPublicKey();

        final List<byte[]> encodedCertificates = encodeCertificates(signerConfig.certificates);
        final byte[] encodedCertificate = encodedCertificates.get(0);

        final V4Signature.SigningInfo signingInfoNoSignature = new V4Signature.SigningInfo(apkDigest,
                encodedCertificate, additionalData, publicKey.getEncoded(), -1, null);

        final byte[] data = V4Signature.getSignedData(fileSize, hashingInfo,
                signingInfoNoSignature);

        // Signing.
        final List<Pair<Integer, byte[]>> signatures =
                ApkSigningBlockUtils.generateSignaturesOverData(signerConfig, data);
        if (signatures.size() != 1) {
            throw new SignatureException("Should only be one signature generated");
        }

        final int signatureAlgorithmId = signatures.get(0).getFirst();
        final byte[] signature = signatures.get(0).getSecond();

        return new V4Signature.SigningInfo(apkDigest,
                encodedCertificate, additionalData, publicKey.getEncoded(), signatureAlgorithmId,
                signature);
    }

    private static V4Signature generateSignature(
            SignerConfig signerConfig,
            V4Signature.HashingInfo hashingInfo,
            Map<Integer, byte[]> apkDigests, byte[] additionalData, long fileSize)
            throws NoSuchAlgorithmException, InvalidKeyException, SignatureException,
            CertificateEncodingException {
        byte[] apkDigest = apkDigests.containsKey(VERSION_APK_SIGNATURE_SCHEME_V3)
                ? apkDigests.get(VERSION_APK_SIGNATURE_SCHEME_V3)
                    : apkDigests.get(VERSION_APK_SIGNATURE_SCHEME_V2);
        final V4Signature.SigningInfo signingInfo = generateSigningInfo(signerConfig.v4Config,
                hashingInfo, apkDigest, additionalData, fileSize);

        final V4Signature.SigningInfos signingInfos;
        if (signerConfig.v41Config != null) {
            if (!apkDigests.containsKey(VERSION_APK_SIGNATURE_SCHEME_V31)) {
                throw new IllegalStateException(
                        "V4.1 cannot be signed without a V3.1 content digest");
            }
            apkDigest = apkDigests.get(VERSION_APK_SIGNATURE_SCHEME_V31);
            final V4Signature.SigningInfoBlock extSigningBlock = new V4Signature.SigningInfoBlock(
                    APK_SIGNATURE_SCHEME_V31_BLOCK_ID,
                    generateSigningInfo(signerConfig.v41Config, hashingInfo, apkDigest,
                            additionalData, fileSize).toByteArray());
            signingInfos = new V4Signature.SigningInfos(signingInfo, extSigningBlock);
        } else {
            signingInfos = new V4Signature.SigningInfos(signingInfo);
        }

        return new V4Signature(V4Signature.CURRENT_VERSION, hashingInfo.toByteArray(),
                signingInfos.toByteArray());
    }

    /**
     * Returns a {@code Map} from the APK signature scheme version to a {@code byte[]} of the
     * strongest supported content digest found in that version's signature block for the V2,
     * V3, and V3.1 signatures in the provided {@code apk}.
     *
     * <p>If a supported content digest algorithm is not found in any of the signature blocks,
     * or if the APK is not signed by any of these signature schemes, then an {@code IOException}
     * is thrown.
     */
    private static Map<Integer, byte[]> getApkDigests(DataSource apk) throws IOException {
        ApkUtils.ZipSections zipSections;
        try {
            zipSections = ApkUtils.findZipSections(apk);
        } catch (ZipFormatException e) {
            throw new IOException("Malformed APK: not a ZIP archive", e);
        }

        Map<Integer, byte[]> sigSchemeToDigest = new HashMap<>(1);
        try {
            byte[] digest = getBestV3Digest(apk, zipSections, VERSION_APK_SIGNATURE_SCHEME_V31);
            sigSchemeToDigest.put(VERSION_APK_SIGNATURE_SCHEME_V31, digest);
        } catch (SignatureException expected) {
            // It is expected to catch a SignatureException if the APK does not have a v3.1
            // signature.
        }

        SignatureException v3Exception = null;
        try {
            byte[] digest =  getBestV3Digest(apk, zipSections, VERSION_APK_SIGNATURE_SCHEME_V3);
            sigSchemeToDigest.put(VERSION_APK_SIGNATURE_SCHEME_V3, digest);
        } catch (SignatureException e) {
            v3Exception = e;
        }

        SignatureException v2Exception = null;
        try {
            byte[] digest = getBestV2Digest(apk, zipSections);
            sigSchemeToDigest.put(VERSION_APK_SIGNATURE_SCHEME_V2, digest);
        } catch (SignatureException e) {
            v2Exception = e;
        }

        if (sigSchemeToDigest.size() > 0) {
            return sigSchemeToDigest;
        }

        throw new IOException(
                "Failed to obtain v2/v3 digest, v3 exception: " + v3Exception + ", v2 exception: "
                        + v2Exception);
    }

    private static byte[] getBestV3Digest(DataSource apk, ApkUtils.ZipSections zipSections,
            int v3SchemeVersion) throws SignatureException {
        final Set<ContentDigestAlgorithm> contentDigestsToVerify = new HashSet<>(1);
        final ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                v3SchemeVersion);
        final int blockId;
        switch (v3SchemeVersion) {
            case VERSION_APK_SIGNATURE_SCHEME_V31:
                blockId = APK_SIGNATURE_SCHEME_V31_BLOCK_ID;
                break;
            case VERSION_APK_SIGNATURE_SCHEME_V3:
                blockId = APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
                break;
            default:
                throw new IllegalArgumentException(
                        "Invalid V3 scheme provided: " + v3SchemeVersion);
        }
        try {
            final SignatureInfo signatureInfo =
                    ApkSigningBlockUtils.findSignature(apk, zipSections, blockId, result);
            final ByteBuffer apkSignatureSchemeV3Block = signatureInfo.signatureBlock;
            V3SchemeVerifier.parseSigners(apkSignatureSchemeV3Block, contentDigestsToVerify,
                    result);
        } catch (Exception e) {
            throw new SignatureException("Failed to extract and parse v3 block", e);
        }

        if (result.signers.size() != 1) {
            throw new SignatureException("Should only have one signer, errors: " + result.getErrors());
        }

        ApkSigningBlockUtils.Result.SignerInfo signer = result.signers.get(0);
        if (signer.containsErrors()) {
            throw new SignatureException("Parsing failed: " + signer.getErrors());
        }

        final List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> contentDigests =
                result.signers.get(0).contentDigests;
        return pickBestDigest(contentDigests);
    }

    private static byte[] getBestV2Digest(DataSource apk, ApkUtils.ZipSections zipSections)
            throws SignatureException {
        final Set<ContentDigestAlgorithm> contentDigestsToVerify = new HashSet<>(1);
        final Set<Integer> foundApkSigSchemeIds = new HashSet<>(1);
        final ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V2);
        try {
            final SignatureInfo signatureInfo =
                    ApkSigningBlockUtils.findSignature(apk, zipSections,
                            APK_SIGNATURE_SCHEME_V2_BLOCK_ID, result);
            final ByteBuffer apkSignatureSchemeV2Block = signatureInfo.signatureBlock;
            V2SchemeVerifier.parseSigners(
                    apkSignatureSchemeV2Block,
                    contentDigestsToVerify,
                    Collections.emptyMap(),
                    foundApkSigSchemeIds,
                    Integer.MAX_VALUE,
                    Integer.MAX_VALUE,
                    result);
        } catch (Exception e) {
            throw new SignatureException("Failed to extract and parse v2 block", e);
        }

        if (result.signers.size() != 1) {
            throw new SignatureException("Should only have one signer, errors: " + result.getErrors());
        }

        ApkSigningBlockUtils.Result.SignerInfo signer = result.signers.get(0);
        if (signer.containsErrors()) {
            throw new SignatureException("Parsing failed: " + signer.getErrors());
        }

        final List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> contentDigests =
                signer.contentDigests;
        return pickBestDigest(contentDigests);
    }

    private static byte[] pickBestDigest(List<ApkSigningBlockUtils.Result.SignerInfo.ContentDigest> contentDigests) throws SignatureException {
        if (contentDigests == null || contentDigests.isEmpty()) {
            throw new SignatureException("Should have at least one digest");
        }

        int bestAlgorithmOrder = -1;
        byte[] bestDigest = null;
        for (ApkSigningBlockUtils.Result.SignerInfo.ContentDigest contentDigest : contentDigests) {
            final SignatureAlgorithm signatureAlgorithm =
                    SignatureAlgorithm.findById(contentDigest.getSignatureAlgorithmId());
            final ContentDigestAlgorithm contentDigestAlgorithm =
                    signatureAlgorithm.getContentDigestAlgorithm();
            if (!isSupported(contentDigestAlgorithm, true)) {
                continue;
            }
            final int algorithmOrder = digestAlgorithmSortingOrder(contentDigestAlgorithm);
            if (bestAlgorithmOrder < algorithmOrder) {
                bestAlgorithmOrder = algorithmOrder;
                bestDigest = contentDigest.getValue();
            }
        }
        if (bestDigest == null) {
            throw new SignatureException("Failed to find a supported digest in the source APK");
        }
        return bestDigest;
    }

    public static int digestAlgorithmSortingOrder(ContentDigestAlgorithm contentDigestAlgorithm) {
        switch (contentDigestAlgorithm) {
            case CHUNKED_SHA256:
                return 0;
            case VERITY_CHUNKED_SHA256:
                return 1;
            case CHUNKED_SHA512:
                return 2;
            default:
                return -1;
        }
    }

    private static boolean isSupported(final ContentDigestAlgorithm contentDigestAlgorithm,
            boolean forV3Digest) {
        if (contentDigestAlgorithm == null) {
            return false;
        }
        if (contentDigestAlgorithm == ContentDigestAlgorithm.CHUNKED_SHA256
                || contentDigestAlgorithm == ContentDigestAlgorithm.CHUNKED_SHA512
                || (forV3Digest
                     && contentDigestAlgorithm == ContentDigestAlgorithm.VERITY_CHUNKED_SHA256)) {
            return true;
        }
        return false;
    }

    private static Pair<Integer, Byte> convertToV4HashingInfo(ContentDigestAlgorithm algorithm)
            throws NoSuchAlgorithmException {
        switch (algorithm) {
            case VERITY_CHUNKED_SHA256:
                return Pair.of(V4Signature.HASHING_ALGORITHM_SHA256,
                        V4Signature.LOG2_BLOCK_SIZE_4096_BYTES);
            default:
                throw new NoSuchAlgorithmException(
                        "Invalid hash algorithm, only SHA2-256 over 4 KB chunks supported.");
        }
    }
}
