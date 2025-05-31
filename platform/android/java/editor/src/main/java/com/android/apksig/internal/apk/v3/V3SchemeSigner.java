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

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsLengthPrefixedElement;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsSequenceOfLengthPrefixedElements;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodeCertificates;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.encodePublicKey;

import com.android.apksig.SigningCertificateLineage;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils.SigningSchemeBlockAndDigests;
import com.android.apksig.internal.apk.ApkSigningBlockUtils.SignerConfig;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.RunnablesExecutor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.interfaces.ECKey;
import java.security.interfaces.RSAKey;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * APK Signature Scheme v3 signer.
 *
 * <p>APK Signature Scheme v3 builds upon APK Signature Scheme v3, and maintains all of the APK
 * Signature Scheme v2 goals.
 *
 * @see <a href="https://source.android.com/security/apksigning/v2.html">APK Signature Scheme v2</a>
 *     <p>The main contribution of APK Signature Scheme v3 is the introduction of the {@link
 *     SigningCertificateLineage}, which enables an APK to change its signing certificate as long as
 *     it can prove the new siging certificate was signed by the old.
 */
public class V3SchemeSigner {
    public static final int APK_SIGNATURE_SCHEME_V3_BLOCK_ID =
            V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
    public static final int PROOF_OF_ROTATION_ATTR_ID = V3SchemeConstants.PROOF_OF_ROTATION_ATTR_ID;

    private final RunnablesExecutor mExecutor;
    private final DataSource mBeforeCentralDir;
    private final DataSource mCentralDir;
    private final DataSource mEocd;
    private final List<SignerConfig> mSignerConfigs;
    private final int mBlockId;
    private final OptionalInt mOptionalV31MinSdkVersion;
    private final boolean mRotationTargetsDevRelease;

    private V3SchemeSigner(DataSource beforeCentralDir,
            DataSource centralDir,
            DataSource eocd,
            List<SignerConfig> signerConfigs,
            RunnablesExecutor executor,
            int blockId,
            OptionalInt optionalV31MinSdkVersion,
            boolean rotationTargetsDevRelease) {
        mBeforeCentralDir = beforeCentralDir;
        mCentralDir = centralDir;
        mEocd = eocd;
        mSignerConfigs = signerConfigs;
        mExecutor = executor;
        mBlockId = blockId;
        mOptionalV31MinSdkVersion = optionalV31MinSdkVersion;
        mRotationTargetsDevRelease = rotationTargetsDevRelease;
    }

    /**
     * Gets the APK Signature Scheme v3 signature algorithms to be used for signing an APK using the
     * provided key.
     *
     * @param minSdkVersion minimum API Level of the platform on which the APK may be installed (see
     *     AndroidManifest.xml minSdkVersion attribute).
     * @throws InvalidKeyException if the provided key is not suitable for signing APKs using APK
     *     Signature Scheme v3
     */
    public static List<SignatureAlgorithm> getSuggestedSignatureAlgorithms(PublicKey signingKey,
            int minSdkVersion, boolean verityEnabled, boolean deterministicDsaSigning)
            throws InvalidKeyException {
        String keyAlgorithm = signingKey.getAlgorithm();
        if ("RSA".equalsIgnoreCase(keyAlgorithm)) {
            // Use RSASSA-PKCS1-v1_5 signature scheme instead of RSASSA-PSS to guarantee
            // deterministic signatures which make life easier for OTA updates (fewer files
            // changed when deterministic signature schemes are used).

            // Pick a digest which is no weaker than the key.
            int modulusLengthBits = ((RSAKey) signingKey).getModulus().bitLength();
            if (modulusLengthBits <= 3072) {
                // 3072-bit RSA is roughly 128-bit strong, meaning SHA-256 is a good fit.
                List<SignatureAlgorithm> algorithms = new ArrayList<>();
                algorithms.add(SignatureAlgorithm.RSA_PKCS1_V1_5_WITH_SHA256);
                if (verityEnabled) {
                    algorithms.add(SignatureAlgorithm.VERITY_RSA_PKCS1_V1_5_WITH_SHA256);
                }
                return algorithms;
            } else {
                // Keys longer than 3072 bit need to be paired with a stronger digest to avoid the
                // digest being the weak link. SHA-512 is the next strongest supported digest.
                return Collections.singletonList(SignatureAlgorithm.RSA_PKCS1_V1_5_WITH_SHA512);
            }
        } else if ("DSA".equalsIgnoreCase(keyAlgorithm)) {
            // DSA is supported only with SHA-256.
            List<SignatureAlgorithm> algorithms = new ArrayList<>();
            algorithms.add(
                    deterministicDsaSigning ?
                            SignatureAlgorithm.DETDSA_WITH_SHA256 :
                            SignatureAlgorithm.DSA_WITH_SHA256);
            if (verityEnabled) {
                algorithms.add(SignatureAlgorithm.VERITY_DSA_WITH_SHA256);
            }
            return algorithms;
        } else if ("EC".equalsIgnoreCase(keyAlgorithm)) {
            // Pick a digest which is no weaker than the key.
            int keySizeBits = ((ECKey) signingKey).getParams().getOrder().bitLength();
            if (keySizeBits <= 256) {
                // 256-bit Elliptic Curve is roughly 128-bit strong, meaning SHA-256 is a good fit.
                List<SignatureAlgorithm> algorithms = new ArrayList<>();
                algorithms.add(SignatureAlgorithm.ECDSA_WITH_SHA256);
                if (verityEnabled) {
                    algorithms.add(SignatureAlgorithm.VERITY_ECDSA_WITH_SHA256);
                }
                return algorithms;
            } else {
                // Keys longer than 256 bit need to be paired with a stronger digest to avoid the
                // digest being the weak link. SHA-512 is the next strongest supported digest.
                return Collections.singletonList(SignatureAlgorithm.ECDSA_WITH_SHA512);
            }
        } else {
            throw new InvalidKeyException("Unsupported key algorithm: " + keyAlgorithm);
        }
    }

    public static SigningSchemeBlockAndDigests generateApkSignatureSchemeV3Block(
            RunnablesExecutor executor,
            DataSource beforeCentralDir,
            DataSource centralDir,
            DataSource eocd,
            List<SignerConfig> signerConfigs)
            throws IOException, InvalidKeyException, NoSuchAlgorithmException, SignatureException {
        return new V3SchemeSigner.Builder(beforeCentralDir, centralDir, eocd, signerConfigs)
                .setRunnablesExecutor(executor)
                .setBlockId(V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID)
                .build()
                .generateApkSignatureSchemeV3BlockAndDigests();
    }

    public static byte[] generateV3SignerAttribute(
            SigningCertificateLineage signingCertificateLineage) {
        // FORMAT (little endian):
        // * length-prefixed bytes: attribute pair
        //   * uint32: ID
        //   * bytes: value - encoded V3 SigningCertificateLineage
        byte[] encodedLineage = signingCertificateLineage.encodeSigningCertificateLineage();
        int payloadSize = 4 + 4 + encodedLineage.length;
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(4 + encodedLineage.length);
        result.putInt(V3SchemeConstants.PROOF_OF_ROTATION_ATTR_ID);
        result.put(encodedLineage);
        return result.array();
    }

    private static byte[] generateV3RotationMinSdkVersionStrippingProtectionAttribute(
            int rotationMinSdkVersion) {
        // FORMAT (little endian):
        // * length-prefixed bytes: attribute pair
        //   * uint32: ID
        //   * bytes: value - int32 representing minimum SDK version for rotation
        int payloadSize = 4 + 4 + 4;
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(payloadSize - 4);
        result.putInt(V3SchemeConstants.ROTATION_MIN_SDK_VERSION_ATTR_ID);
        result.putInt(rotationMinSdkVersion);
        return result.array();
    }

    private static byte[] generateV31RotationTargetsDevReleaseAttribute() {
        // FORMAT (little endian):
        // * length-prefixed bytes: attribute pair
        //   * uint32: ID
        //   * bytes: value - No value is used for this attribute
        int payloadSize = 4 + 4;
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(payloadSize - 4);
        result.putInt(V3SchemeConstants.ROTATION_ON_DEV_RELEASE_ATTR_ID);
        return result.array();
    }

    /**
     * Generates and returns a new {@link SigningSchemeBlockAndDigests} containing the V3.x
     * signing scheme block and digests based on the parameters provided to the {@link Builder}.
     *
     * @throws IOException if an I/O error occurs
     * @throws NoSuchAlgorithmException if a required cryptographic algorithm implementation is
     *         missing
     * @throws InvalidKeyException if the X.509 encoded form of the public key cannot be obtained
     * @throws SignatureException if an error occurs when computing digests or generating
     *         signatures
     */
    public SigningSchemeBlockAndDigests generateApkSignatureSchemeV3BlockAndDigests()
            throws IOException, InvalidKeyException, NoSuchAlgorithmException, SignatureException {
        Pair<List<SignerConfig>, Map<ContentDigestAlgorithm, byte[]>> digestInfo =
                ApkSigningBlockUtils.computeContentDigests(
                        mExecutor, mBeforeCentralDir, mCentralDir, mEocd, mSignerConfigs);
        return new SigningSchemeBlockAndDigests(
                generateApkSignatureSchemeV3Block(digestInfo.getSecond()), digestInfo.getSecond());
    }

    private Pair<byte[], Integer> generateApkSignatureSchemeV3Block(
            Map<ContentDigestAlgorithm, byte[]> contentDigests)
            throws NoSuchAlgorithmException, InvalidKeyException, SignatureException {
        // FORMAT:
        // * length-prefixed sequence of length-prefixed signer blocks.
        List<byte[]> signerBlocks = new ArrayList<>(mSignerConfigs.size());
        int signerNumber = 0;
        for (SignerConfig signerConfig : mSignerConfigs) {
            signerNumber++;
            byte[] signerBlock;
            try {
                signerBlock = generateSignerBlock(signerConfig, contentDigests);
            } catch (InvalidKeyException e) {
                throw new InvalidKeyException("Signer #" + signerNumber + " failed", e);
            } catch (SignatureException e) {
                throw new SignatureException("Signer #" + signerNumber + " failed", e);
            }
            signerBlocks.add(signerBlock);
        }

        return Pair.of(
                encodeAsSequenceOfLengthPrefixedElements(
                        new byte[][] {
                            encodeAsSequenceOfLengthPrefixedElements(signerBlocks),
                        }),
                mBlockId);
    }

    private byte[] generateSignerBlock(
            SignerConfig signerConfig, Map<ContentDigestAlgorithm, byte[]> contentDigests)
            throws NoSuchAlgorithmException, InvalidKeyException, SignatureException {
        if (signerConfig.certificates.isEmpty()) {
            throw new SignatureException("No certificates configured for signer");
        }
        PublicKey publicKey = signerConfig.certificates.get(0).getPublicKey();

        byte[] encodedPublicKey = encodePublicKey(publicKey);

        V3SignatureSchemeBlock.SignedData signedData = new V3SignatureSchemeBlock.SignedData();
        try {
            signedData.certificates = encodeCertificates(signerConfig.certificates);
        } catch (CertificateEncodingException e) {
            throw new SignatureException("Failed to encode certificates", e);
        }

        List<Pair<Integer, byte[]>> digests =
                new ArrayList<>(signerConfig.signatureAlgorithms.size());
        for (SignatureAlgorithm signatureAlgorithm : signerConfig.signatureAlgorithms) {
            ContentDigestAlgorithm contentDigestAlgorithm =
                    signatureAlgorithm.getContentDigestAlgorithm();
            byte[] contentDigest = contentDigests.get(contentDigestAlgorithm);
            if (contentDigest == null) {
                throw new RuntimeException(
                        contentDigestAlgorithm
                                + " content digest for "
                                + signatureAlgorithm
                                + " not computed");
            }
            digests.add(Pair.of(signatureAlgorithm.getId(), contentDigest));
        }
        signedData.digests = digests;
        signedData.minSdkVersion = signerConfig.minSdkVersion;
        signedData.maxSdkVersion = signerConfig.maxSdkVersion;
        signedData.additionalAttributes = generateAdditionalAttributes(signerConfig);

        V3SignatureSchemeBlock.Signer signer = new V3SignatureSchemeBlock.Signer();

        signer.signedData = encodeSignedData(signedData);

        signer.minSdkVersion = signerConfig.minSdkVersion;
        signer.maxSdkVersion = signerConfig.maxSdkVersion;
        signer.publicKey = encodedPublicKey;
        signer.signatures =
                ApkSigningBlockUtils.generateSignaturesOverData(signerConfig, signer.signedData);

        return encodeSigner(signer);
    }

    private byte[] encodeSigner(V3SignatureSchemeBlock.Signer signer) {
        byte[] signedData = encodeAsLengthPrefixedElement(signer.signedData);
        byte[] signatures =
                encodeAsLengthPrefixedElement(
                        encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
                                signer.signatures));
        byte[] publicKey = encodeAsLengthPrefixedElement(signer.publicKey);

        // FORMAT:
        // * length-prefixed signed data
        // * uint32: minSdkVersion
        // * uint32: maxSdkVersion
        // * length-prefixed sequence of length-prefixed signatures:
        //   * uint32: signature algorithm ID
        //   * length-prefixed bytes: signature of signed data
        // * length-prefixed bytes: public key (X.509 SubjectPublicKeyInfo, ASN.1 DER encoded)
        int payloadSize = signedData.length + 4 + 4 + signatures.length + publicKey.length;

        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.put(signedData);
        result.putInt(signer.minSdkVersion);
        result.putInt(signer.maxSdkVersion);
        result.put(signatures);
        result.put(publicKey);

        return result.array();
    }

    private byte[] encodeSignedData(V3SignatureSchemeBlock.SignedData signedData) {
        byte[] digests =
                encodeAsLengthPrefixedElement(
                        encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
                                signedData.digests));
        byte[] certs =
                encodeAsLengthPrefixedElement(
                        encodeAsSequenceOfLengthPrefixedElements(signedData.certificates));
        byte[] attributes = encodeAsLengthPrefixedElement(signedData.additionalAttributes);

        // FORMAT:
        // * length-prefixed sequence of length-prefixed digests:
        //   * uint32: signature algorithm ID
        //   * length-prefixed bytes: digest of contents
        // * length-prefixed sequence of certificates:
        //   * length-prefixed bytes: X.509 certificate (ASN.1 DER encoded).
        // * uint-32: minSdkVersion
        // * uint-32: maxSdkVersion
        // * length-prefixed sequence of length-prefixed additional attributes:
        //   * uint32: ID
        //   * (length - 4) bytes: value
        //   * uint32: Proof-of-rotation ID: 0x3ba06f8c
        //   * length-prefixed roof-of-rotation structure
        int payloadSize = digests.length + certs.length + 4 + 4 + attributes.length;

        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.put(digests);
        result.put(certs);
        result.putInt(signedData.minSdkVersion);
        result.putInt(signedData.maxSdkVersion);
        result.put(attributes);

        return result.array();
    }

    private byte[] generateAdditionalAttributes(SignerConfig signerConfig) {
        List<byte[]> attributes = new ArrayList<>();
        if (signerConfig.signingCertificateLineage != null) {
            attributes.add(generateV3SignerAttribute(signerConfig.signingCertificateLineage));
        }
        if ((mRotationTargetsDevRelease || signerConfig.signerTargetsDevRelease)
                && mBlockId == V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID) {
            attributes.add(generateV31RotationTargetsDevReleaseAttribute());
        }
        if (mOptionalV31MinSdkVersion.isPresent()
                && mBlockId == V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID) {
            attributes.add(generateV3RotationMinSdkVersionStrippingProtectionAttribute(
                    mOptionalV31MinSdkVersion.getAsInt()));
        }
        int attributesSize = attributes.stream().mapToInt(attribute -> attribute.length).sum();
        byte[] attributesBuffer = new byte[attributesSize];
        if (attributesSize == 0) {
            return new byte[0];
        }
        int index = 0;
        for (byte[] attribute : attributes) {
            System.arraycopy(attribute, 0, attributesBuffer, index, attribute.length);
            index += attribute.length;
        }
        return attributesBuffer;
    }

    private static final class V3SignatureSchemeBlock {
        private static final class Signer {
            public byte[] signedData;
            public int minSdkVersion;
            public int maxSdkVersion;
            public List<Pair<Integer, byte[]>> signatures;
            public byte[] publicKey;
        }

        private static final class SignedData {
            public List<Pair<Integer, byte[]>> digests;
            public List<byte[]> certificates;
            public int minSdkVersion;
            public int maxSdkVersion;
            public byte[] additionalAttributes;
        }
    }

    /** Builder of {@link V3SchemeSigner} instances. */
    public static class Builder {
        private final DataSource mBeforeCentralDir;
        private final DataSource mCentralDir;
        private final DataSource mEocd;
        private final List<SignerConfig> mSignerConfigs;

        private RunnablesExecutor mExecutor = RunnablesExecutor.MULTI_THREADED;
        private int mBlockId = V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID;
        private OptionalInt mOptionalV31MinSdkVersion = OptionalInt.empty();
        private boolean mRotationTargetsDevRelease = false;

        /**
         * Instantiates a new {@code Builder} with an APK's {@code beforeCentralDir}, {@code
         * centralDir}, and {@code eocd}, along with a {@link List} of {@code signerConfigs} to
         * be used to sign the APK.
         */
        public Builder(DataSource beforeCentralDir, DataSource centralDir, DataSource eocd,
                List<SignerConfig> signerConfigs) {
            mBeforeCentralDir = beforeCentralDir;
            mCentralDir = centralDir;
            mEocd = eocd;
            mSignerConfigs = signerConfigs;
        }

        /**
         * Sets the {@link RunnablesExecutor} to be used when computing the APK's content digests.
         */
        public Builder setRunnablesExecutor(RunnablesExecutor executor) {
            mExecutor = executor;
            return this;
        }

        /**
         * Sets the {@code blockId} to be used for the V3 signature block.
         *
         * <p>This {@code V3SchemeSigner} currently supports the block IDs for the {@link
         * V3SchemeConstants#APK_SIGNATURE_SCHEME_V3_BLOCK_ID v3.0} and {@link
         * V3SchemeConstants#APK_SIGNATURE_SCHEME_V31_BLOCK_ID v3.1} signature schemes.
         */
        public Builder setBlockId(int blockId) {
            mBlockId = blockId;
            return this;
        }

        /**
         * Sets the {@code rotationMinSdkVersion} to be written as an additional attribute in each
         * signer's block.
         *
         * <p>This value provides stripping protection to ensure a v3.1 signing block with rotation
         * is not modified or removed from the APK's signature block.
         */
        public Builder setRotationMinSdkVersion(int rotationMinSdkVersion) {
            return setMinSdkVersionForV31(rotationMinSdkVersion);
        }

        /**
         * Sets the {@code minSdkVersion} to be written as an additional attribute in each
         * signer's block.
         *
         * <p>This value provides the stripping protection to ensure a v3.1 signing block is not
         * modified or removed from the APK's signature block.
         */
        public Builder setMinSdkVersionForV31(int minSdkVersion) {
            if (minSdkVersion == V3SchemeConstants.DEV_RELEASE) {
                minSdkVersion = V3SchemeConstants.PROD_RELEASE;
            }
            mOptionalV31MinSdkVersion = OptionalInt.of(minSdkVersion);
            return this;
        }

        /**
         * Sets whether the minimum SDK version of a signer is intended to target a development
         * release; this is primarily required after the T SDK is finalized, and an APK needs to
         * target U during its development cycle for rotation.
         *
         * <p>This is only required after the T SDK is finalized since S and earlier releases do
         * not know about the V3.1 block ID, but once T is released and work begins on U, U will
         * use the SDK version of T during development. A signer with a minimum SDK version of T's
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

        /**
         * Returns a new {@link V3SchemeSigner} built with the configuration provided to this
         * {@code Builder}.
         */
        public V3SchemeSigner build() {
            return new V3SchemeSigner(mBeforeCentralDir,
                    mCentralDir,
                    mEocd,
                    mSignerConfigs,
                    mExecutor,
                    mBlockId,
                    mOptionalV31MinSdkVersion,
                    mRotationTargetsDevRelease);
        }
    }
}
