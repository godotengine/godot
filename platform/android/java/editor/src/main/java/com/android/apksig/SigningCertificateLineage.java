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

package com.android.apksig;

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.getLengthPrefixedSlice;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.apk.SignatureInfo;
import com.android.apksig.internal.apk.v3.V3SchemeConstants;
import com.android.apksig.internal.apk.v3.V3SchemeSigner;
import com.android.apksig.internal.apk.v3.V3SigningCertificateLineage;
import com.android.apksig.internal.apk.v3.V3SigningCertificateLineage.SigningCertificateNode;
import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.util.ByteBufferUtils;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.internal.util.RandomAccessFileDataSink;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.DataSources;
import com.android.apksig.zip.ZipFormatException;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * APK Signer Lineage.
 *
 * <p>The signer lineage contains a history of signing certificates with each ancestor attesting to
 * the validity of its descendant.  Each additional descendant represents a new identity that can be
 * used to sign an APK, and each generation has accompanying attributes which represent how the
 * APK would like to view the older signing certificates, specifically how they should be trusted in
 * certain situations.
 *
 * <p> Its primary use is to enable APK Signing Certificate Rotation.  The Android platform verifies
 * the APK Signer Lineage, and if the current signing certificate for the APK is in the Signer
 * Lineage, and the Lineage contains the certificate the platform associates with the APK, it will
 * allow upgrades to the new certificate.
 *
 * @see <a href="https://source.android.com/security/apksigning/index.html">Application Signing</a>
 */
public class SigningCertificateLineage {

    public final static int MAGIC = 0x3eff39d1;

    private final static int FIRST_VERSION = 1;

    private static final int CURRENT_VERSION = FIRST_VERSION;

    /** accept data from already installed pkg with this cert */
    private static final int PAST_CERT_INSTALLED_DATA = 1;

    /** accept sharedUserId with pkg with this cert */
    private static final int PAST_CERT_SHARED_USER_ID = 2;

    /** grant SIGNATURE permissions to pkgs with this cert */
    private static final int PAST_CERT_PERMISSION = 4;

    /**
     * Enable updates back to this certificate.  WARNING: this effectively removes any benefit of
     * signing certificate changes, since a compromised key could retake control of an app even
     * after change, and should only be used if there is a problem encountered when trying to ditch
     * an older cert.
     */
    private static final int PAST_CERT_ROLLBACK = 8;

    /**
     * Preserve authenticator module-based access in AccountManager gated by signing certificate.
     */
    private static final int PAST_CERT_AUTH = 16;

    private final int mMinSdkVersion;

    /**
     * The signing lineage is just a list of nodes, with the first being the original signing
     * certificate and the most recent being the one with which the APK is to actually be signed.
     */
    private final List<SigningCertificateNode> mSigningLineage;

    private SigningCertificateLineage(int minSdkVersion, List<SigningCertificateNode> list) {
        mMinSdkVersion = minSdkVersion;
        mSigningLineage = list;
    }

    /**
     * Creates a {@code SigningCertificateLineage} with a single signer in the lineage.
     */
    private static SigningCertificateLineage createSigningLineage(int minSdkVersion,
            SignerConfig signer, SignerCapabilities capabilities) {
        SigningCertificateLineage signingCertificateLineage = new SigningCertificateLineage(
                minSdkVersion, new ArrayList<>());
        return signingCertificateLineage.spawnFirstDescendant(signer, capabilities);
    }

    private static SigningCertificateLineage createSigningLineage(
            int minSdkVersion, SignerConfig parent, SignerCapabilities parentCapabilities,
            SignerConfig child, SignerCapabilities childCapabilities)
            throws CertificateEncodingException, InvalidKeyException, NoSuchAlgorithmException,
            SignatureException {
        SigningCertificateLineage signingCertificateLineage =
                new SigningCertificateLineage(minSdkVersion, new ArrayList<>());
        signingCertificateLineage =
                signingCertificateLineage.spawnFirstDescendant(parent, parentCapabilities);
        return signingCertificateLineage.spawnDescendant(parent, child, childCapabilities);
    }

    public static SigningCertificateLineage readFromBytes(byte[] lineageBytes)
            throws IOException {
        return readFromDataSource(DataSources.asDataSource(ByteBuffer.wrap(lineageBytes)));
    }

    public static SigningCertificateLineage readFromFile(File file)
            throws IOException {
        if (file == null) {
            throw new NullPointerException("file == null");
        }
        RandomAccessFile inputFile = new RandomAccessFile(file, "r");
        return readFromDataSource(DataSources.asDataSource(inputFile));
    }

    public static SigningCertificateLineage readFromDataSource(DataSource dataSource)
            throws IOException {
        if (dataSource == null) {
            throw new NullPointerException("dataSource == null");
        }
        ByteBuffer inBuff = dataSource.getByteBuffer(0, (int) dataSource.size());
        inBuff.order(ByteOrder.LITTLE_ENDIAN);
        return read(inBuff);
    }

    /**
     * Extracts a Signing Certificate Lineage from a v3 signer proof-of-rotation attribute.
     *
     * <note>
     *     this may not give a complete representation of an APK's signing certificate history,
     *     since the APK may have multiple signers corresponding to different platform versions.
     *     Use <code> readFromApkFile</code> to handle this case.
     * </note>
     * @param attrValue
     */
    public static SigningCertificateLineage readFromV3AttributeValue(byte[] attrValue)
            throws IOException {
        List<SigningCertificateNode> parsedLineage =
                V3SigningCertificateLineage.readSigningCertificateLineage(ByteBuffer.wrap(
                        attrValue).order(ByteOrder.LITTLE_ENDIAN));
        int minSdkVersion = calculateMinSdkVersion(parsedLineage);
        return  new SigningCertificateLineage(minSdkVersion, parsedLineage);
    }

    /**
     * Extracts a Signing Certificate Lineage from the proof-of-rotation attribute in the V3
     * signature block of the provided APK File.
     *
     * @throws IllegalArgumentException if the provided APK does not contain a V3 signature block,
     * or if the V3 signature block does not contain a valid lineage.
     */
    public static SigningCertificateLineage readFromApkFile(File apkFile)
            throws IOException, ApkFormatException {
        try (RandomAccessFile f = new RandomAccessFile(apkFile, "r")) {
            DataSource apk = DataSources.asDataSource(f, 0, f.length());
            return readFromApkDataSource(apk);
        }
    }

    /**
     * Extracts a Signing Certificate Lineage from the proof-of-rotation attribute in the V3 and
     * V3.1 signature blocks of the provided APK DataSource.
     *
     * @throws IllegalArgumentException if the provided APK does not contain a V3 nor V3.1
     * signature block, or if the V3 and V3.1 signature blocks do not contain a valid lineage.
     */

    public static SigningCertificateLineage readFromApkDataSource(DataSource apk)
            throws IOException, ApkFormatException {
        return readFromApkDataSource(apk, /* readV31Lineage= */ true,  /* readV3Lineage= */true);
    }

    /**
     * Extracts a Signing Certificate Lineage from the proof-of-rotation attribute in the V3.1
     * signature blocks of the provided APK DataSource.
     *
     * @throws IllegalArgumentException if the provided APK does not contain a V3.1 signature block,
     * or if the V3.1 signature block does not contain a valid lineage.
     */

    public static SigningCertificateLineage readV31FromApkDataSource(DataSource apk)
            throws IOException, ApkFormatException {
            return readFromApkDataSource(apk, /* readV31Lineage= */ true,
                        /* readV3Lineage= */ false);
    }

    private static SigningCertificateLineage readFromApkDataSource(
            DataSource apk,
            boolean readV31Lineage,
            boolean readV3Lineage)
            throws IOException, ApkFormatException {
        ApkUtils.ZipSections zipSections;
        try {
            zipSections = ApkUtils.findZipSections(apk);
        } catch (ZipFormatException e) {
            throw new ApkFormatException(e.getMessage());
        }

        List<SignatureInfo> signatureInfoList = new ArrayList<>();
        if (readV31Lineage) {
            try {
                ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                    ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V31);
                signatureInfoList.add(
                    ApkSigningBlockUtils.findSignature(apk, zipSections,
                        V3SchemeConstants.APK_SIGNATURE_SCHEME_V31_BLOCK_ID, result));
            } catch (ApkSigningBlockUtils.SignatureNotFoundException ignored) {
                // This could be expected if there's only a V3 signature block.
            }
        }
        if (readV3Lineage) {
            try {
                ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                    ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V3);
                signatureInfoList.add(
                    ApkSigningBlockUtils.findSignature(apk, zipSections,
                        V3SchemeConstants.APK_SIGNATURE_SCHEME_V3_BLOCK_ID, result));
            } catch (ApkSigningBlockUtils.SignatureNotFoundException ignored) {
                // This could be expected if the provided APK is not signed with the V3 signature
                // scheme
            }
        }
        if (signatureInfoList.isEmpty()) {
            String message;
            if (readV31Lineage && readV3Lineage) {
                message = "The provided APK does not contain a valid V3 nor V3.1 signature block.";
            } else if (readV31Lineage) {
                message = "The provided APK does not contain a valid V3.1 signature block.";
            } else if (readV3Lineage) {
                message = "The provided APK does not contain a valid V3 signature block.";
            } else {
                message = "No signature blocks were requested.";
            }
            throw new IllegalArgumentException(message);
        }

        List<SigningCertificateLineage> lineages = new ArrayList<>(1);
        for (SignatureInfo signatureInfo : signatureInfoList) {
            // FORMAT:
            // * length-prefixed sequence of length-prefixed signers:
            //   * length-prefixed signed data
            //   * minSDK
            //   * maxSDK
            //   * length-prefixed sequence of length-prefixed signatures
            //   * length-prefixed public key
            ByteBuffer signers = getLengthPrefixedSlice(signatureInfo.signatureBlock);
            while (signers.hasRemaining()) {
                ByteBuffer signer = getLengthPrefixedSlice(signers);
                ByteBuffer signedData = getLengthPrefixedSlice(signer);
                try {
                    SigningCertificateLineage lineage = readFromSignedData(signedData);
                    lineages.add(lineage);
                } catch (IllegalArgumentException ignored) {
                    // The current signer block does not contain a valid lineage, but it is possible
                    // another block will.
                }
            }
        }

        SigningCertificateLineage result;
        if (lineages.isEmpty()) {
            throw new IllegalArgumentException(
                    "The provided APK does not contain a valid lineage.");
        } else if (lineages.size() > 1) {
            result = consolidateLineages(lineages);
        } else {
            result = lineages.get(0);
        }
        return result;
    }

    /**
     * Extracts a Signing Certificate Lineage from the proof-of-rotation attribute in the provided
     * signed data portion of a signer in a V3 signature block.
     *
     * @throws IllegalArgumentException if the provided signed data does not contain a valid
     * lineage.
     */
    public static SigningCertificateLineage readFromSignedData(ByteBuffer signedData)
            throws IOException, ApkFormatException {
        // FORMAT:
        //   * length-prefixed sequence of length-prefixed digests:
        //   * length-prefixed sequence of certificates:
        //     * length-prefixed bytes: X.509 certificate (ASN.1 DER encoded).
        //   * uint-32: minSdkVersion
        //   * uint-32: maxSdkVersion
        //   * length-prefixed sequence of length-prefixed additional attributes:
        //     * uint32: ID
        //     * (length - 4) bytes: value
        //     * uint32: Proof-of-rotation ID: 0x3ba06f8c
        //     * length-prefixed proof-of-rotation structure
        // consume the digests through the maxSdkVersion to reach the lineage in the attributes
        getLengthPrefixedSlice(signedData);
        getLengthPrefixedSlice(signedData);
        signedData.getInt();
        signedData.getInt();
        // iterate over the additional attributes adding any lineages to the List
        ByteBuffer additionalAttributes = getLengthPrefixedSlice(signedData);
        List<SigningCertificateLineage> lineages = new ArrayList<>(1);
        while (additionalAttributes.hasRemaining()) {
            ByteBuffer attribute = getLengthPrefixedSlice(additionalAttributes);
            int id = attribute.getInt();
            if (id == V3SchemeConstants.PROOF_OF_ROTATION_ATTR_ID) {
                byte[] value = ByteBufferUtils.toByteArray(attribute);
                SigningCertificateLineage lineage = readFromV3AttributeValue(value);
                lineages.add(lineage);
            }
        }
        SigningCertificateLineage result;
        // There should only be a single attribute with the lineage, but if there are multiple then
        // attempt to consolidate the lineages.
        if (lineages.isEmpty()) {
            throw new IllegalArgumentException("The signed data does not contain a valid lineage.");
        } else if (lineages.size() > 1) {
            result = consolidateLineages(lineages);
        } else {
            result = lineages.get(0);
        }
        return result;
    }

    public byte[] getBytes() {
        return write().array();
    }

    public void writeToFile(File file) throws IOException {
        if (file == null) {
            throw new NullPointerException("file == null");
        }
        RandomAccessFile outputFile = new RandomAccessFile(file, "rw");
        writeToDataSink(new RandomAccessFileDataSink(outputFile));
    }

    public void writeToDataSink(DataSink dataSink) throws IOException {
        if (dataSink == null) {
            throw new NullPointerException("dataSink == null");
        }
        dataSink.consume(write());
    }

    /**
     * Add a new signing certificate to the lineage.  This effectively creates a signing certificate
     * rotation event, forcing APKs which include this lineage to be signed by the new signer. The
     * flags associated with the new signer are set to a default value.
     *
     * @param parent current signing certificate of the containing APK
     * @param child new signing certificate which will sign the APK contents
     */
    public SigningCertificateLineage spawnDescendant(SignerConfig parent, SignerConfig child)
            throws CertificateEncodingException, InvalidKeyException, NoSuchAlgorithmException,
            SignatureException {
        if (parent == null || child == null) {
            throw new NullPointerException("can't add new descendant to lineage with null inputs");
        }
        SignerCapabilities signerCapabilities = new SignerCapabilities.Builder().build();
        return spawnDescendant(parent, child, signerCapabilities);
    }

    /**
     * Add a new signing certificate to the lineage.  This effectively creates a signing certificate
     * rotation event, forcing APKs which include this lineage to be signed by the new signer.
     *
     * @param parent current signing certificate of the containing APK
     * @param child new signing certificate which will sign the APK contents
     * @param childCapabilities flags
     */
    public SigningCertificateLineage spawnDescendant(
            SignerConfig parent, SignerConfig child, SignerCapabilities childCapabilities)
            throws CertificateEncodingException, InvalidKeyException,
            NoSuchAlgorithmException, SignatureException {
        if (parent == null) {
            throw new NullPointerException("parent == null");
        }
        if (child == null) {
            throw new NullPointerException("child == null");
        }
        if (childCapabilities == null) {
            throw new NullPointerException("childCapabilities == null");
        }
        if (mSigningLineage.isEmpty()) {
            throw new IllegalArgumentException("Cannot spawn descendant signing certificate on an"
                    + " empty SigningCertificateLineage: no parent node");
        }

        // make sure that the parent matches our newest generation (leaf node/sink)
        SigningCertificateNode currentGeneration = mSigningLineage.get(mSigningLineage.size() - 1);
        if (!Arrays.equals(currentGeneration.signingCert.getEncoded(),
                parent.getCertificate().getEncoded())) {
            throw new IllegalArgumentException("SignerConfig Certificate containing private key"
                    + " to sign the new SigningCertificateLineage record does not match the"
                    + " existing most recent record");
        }

        // create data to be signed, including the algorithm we're going to use
        SignatureAlgorithm signatureAlgorithm = getSignatureAlgorithm(parent);
        ByteBuffer prefixedSignedData = ByteBuffer.wrap(
                V3SigningCertificateLineage.encodeSignedData(
                        child.getCertificate(), signatureAlgorithm.getId()));
        prefixedSignedData.position(4);
        ByteBuffer signedDataBuffer = ByteBuffer.allocate(prefixedSignedData.remaining());
        signedDataBuffer.put(prefixedSignedData);
        byte[] signedData = signedDataBuffer.array();

        // create SignerConfig to do the signing
        List<X509Certificate> certificates = new ArrayList<>(1);
        certificates.add(parent.getCertificate());
        ApkSigningBlockUtils.SignerConfig newSignerConfig =
                new ApkSigningBlockUtils.SignerConfig();
        newSignerConfig.privateKey = parent.getPrivateKey();
        newSignerConfig.certificates = certificates;
        newSignerConfig.signatureAlgorithms = Collections.singletonList(signatureAlgorithm);

        // sign it
        List<Pair<Integer, byte[]>> signatures =
                ApkSigningBlockUtils.generateSignaturesOverData(newSignerConfig, signedData);

        // finally, add it to our lineage
        SignatureAlgorithm sigAlgorithm = SignatureAlgorithm.findById(signatures.get(0).getFirst());
        byte[] signature = signatures.get(0).getSecond();
        currentGeneration.sigAlgorithm = sigAlgorithm;
        SigningCertificateNode childNode =
                new SigningCertificateNode(
                        child.getCertificate(), sigAlgorithm, null,
                        signature, childCapabilities.getFlags());
        List<SigningCertificateNode> lineageCopy = new ArrayList<>(mSigningLineage);
        lineageCopy.add(childNode);
        return new SigningCertificateLineage(mMinSdkVersion, lineageCopy);
    }

    /**
     * The number of signing certificates in the lineage, including the current signer, which means
     * this value can also be used to V2determine the number of signing certificate rotations by
     * subtracting 1.
     */
    public int size() {
        return mSigningLineage.size();
    }

    private SignatureAlgorithm getSignatureAlgorithm(SignerConfig parent)
            throws InvalidKeyException {
        PublicKey publicKey = parent.getCertificate().getPublicKey();

        // TODO switch to one signature algorithm selection, or add support for multiple algorithms
        List<SignatureAlgorithm> algorithms = V3SchemeSigner.getSuggestedSignatureAlgorithms(
                publicKey, mMinSdkVersion, false /* verityEnabled */,
                false /* deterministicDsaSigning */);
        return algorithms.get(0);
    }

    private SigningCertificateLineage spawnFirstDescendant(
            SignerConfig parent, SignerCapabilities signerCapabilities) {
        if (!mSigningLineage.isEmpty()) {
            throw new IllegalStateException("SigningCertificateLineage already has its first node");
        }

        // check to make sure that the public key for the first node is acceptable for our minSdk
        try {
            getSignatureAlgorithm(parent);
        } catch (InvalidKeyException e) {
            throw new IllegalArgumentException("Algorithm associated with first signing certificate"
                    + " invalid on desired platform versions", e);
        }

        // create "fake" signed data (there will be no signature over it, since there is no parent
        SigningCertificateNode firstNode = new SigningCertificateNode(
                parent.getCertificate(), null, null, new byte[0], signerCapabilities.getFlags());
        return new SigningCertificateLineage(mMinSdkVersion, Collections.singletonList(firstNode));
    }

    private static SigningCertificateLineage read(ByteBuffer inputByteBuffer)
            throws IOException {
        ApkSigningBlockUtils.checkByteOrderLittleEndian(inputByteBuffer);
        if (inputByteBuffer.remaining() < 8) {
            throw new IllegalArgumentException(
                    "Improper SigningCertificateLineage format: insufficient data for header.");
        }

        if (inputByteBuffer.getInt() != MAGIC) {
            throw new IllegalArgumentException(
                    "Improper SigningCertificateLineage format: MAGIC header mismatch.");
        }
        return read(inputByteBuffer, inputByteBuffer.getInt());
    }

    private static SigningCertificateLineage read(ByteBuffer inputByteBuffer, int version)
            throws IOException {
        switch (version) {
            case FIRST_VERSION:
                try {
                    List<SigningCertificateNode> nodes =
                            V3SigningCertificateLineage.readSigningCertificateLineage(
                                    getLengthPrefixedSlice(inputByteBuffer));
                    int minSdkVersion = calculateMinSdkVersion(nodes);
                    return new SigningCertificateLineage(minSdkVersion, nodes);
                } catch (ApkFormatException e) {
                    // unable to get a proper length-prefixed lineage slice
                    throw new IOException("Unable to read list of signing certificate nodes in "
                            + "SigningCertificateLineage", e);
                }
            default:
                throw new IllegalArgumentException(
                        "Improper SigningCertificateLineage format: unrecognized version.");
        }
    }

    private static int calculateMinSdkVersion(List<SigningCertificateNode> nodes) {
        if (nodes == null) {
            throw new IllegalArgumentException("Can't calculate minimum SDK version of null nodes");
        }
        int minSdkVersion = AndroidSdkVersion.P; // lineage introduced in P
        for (SigningCertificateNode node : nodes) {
            if (node.sigAlgorithm != null) {
                int nodeMinSdkVersion = node.sigAlgorithm.getMinSdkVersion();
                if (nodeMinSdkVersion > minSdkVersion) {
                    minSdkVersion = nodeMinSdkVersion;
                }
            }
        }
        return minSdkVersion;
    }

    private ByteBuffer write() {
        byte[] encodedLineage =
                V3SigningCertificateLineage.encodeSigningCertificateLineage(mSigningLineage);
        int payloadSize = 4 + 4 + 4 + encodedLineage.length;
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(MAGIC);
        result.putInt(CURRENT_VERSION);
        result.putInt(encodedLineage.length);
        result.put(encodedLineage);
        result.flip();
        return result;
    }

    public byte[] encodeSigningCertificateLineage() {
        return V3SigningCertificateLineage.encodeSigningCertificateLineage(mSigningLineage);
    }

    public List<DefaultApkSignerEngine.SignerConfig> sortSignerConfigs(
            List<DefaultApkSignerEngine.SignerConfig> signerConfigs) {
        if (signerConfigs == null) {
            throw new NullPointerException("signerConfigs == null");
        }

        // not the most elegant sort, but we expect signerConfigs to be quite small (1 or 2 signers
        // in most cases) and likely already sorted, so not worth the overhead of doing anything
        // fancier
        List<DefaultApkSignerEngine.SignerConfig> sortedSignerConfigs =
                new ArrayList<>(signerConfigs.size());
        for (int i = 0; i < mSigningLineage.size(); i++) {
            for (int j = 0; j < signerConfigs.size(); j++) {
                DefaultApkSignerEngine.SignerConfig config = signerConfigs.get(j);
                if (mSigningLineage.get(i).signingCert.equals(config.getCertificates().get(0))) {
                    sortedSignerConfigs.add(config);
                    break;
                }
            }
        }
        if (sortedSignerConfigs.size() != signerConfigs.size()) {
            throw new IllegalArgumentException("SignerConfigs supplied which are not present in the"
                    + " SigningCertificateLineage");
        }
        return sortedSignerConfigs;
    }

    /**
     * Returns the SignerCapabilities for the signer in the lineage that matches the provided
     * config.
     */
    public SignerCapabilities getSignerCapabilities(SignerConfig config) {
        if (config == null) {
            throw new NullPointerException("config == null");
        }

        X509Certificate cert = config.getCertificate();
        return getSignerCapabilities(cert);
    }

    /**
     * Returns the SignerCapabilities for the signer in the lineage that matches the provided
     * certificate.
     */
    public SignerCapabilities getSignerCapabilities(X509Certificate cert) {
        if (cert == null) {
            throw new NullPointerException("cert == null");
        }

        for (int i = 0; i < mSigningLineage.size(); i++) {
            SigningCertificateNode lineageNode = mSigningLineage.get(i);
            if (lineageNode.signingCert.equals(cert)) {
                int flags = lineageNode.flags;
                return new SignerCapabilities.Builder(flags).build();
            }
        }

        // the provided signer certificate was not found in the lineage
        throw new IllegalArgumentException("Certificate (" + cert.getSubjectDN()
                + ") not found in the SigningCertificateLineage");
    }

    /**
     * Updates the SignerCapabilities for the signer in the lineage that matches the provided
     * config. Only those capabilities that have been modified through the setXX methods will be
     * updated for the signer to prevent unset default values from being applied.
     */
    public void updateSignerCapabilities(SignerConfig config, SignerCapabilities capabilities) {
        if (config == null) {
            throw new NullPointerException("config == null");
        }
        updateSignerCapabilities(config.getCertificate(), capabilities);
    }

    /**
     * Updates the {@code capabilities} for the signer with the provided {@code certificate} in the
     * lineage. Only those capabilities that have been modified through the setXX methods will be
     * updated for the signer to prevent unset default values from being applied.
     */
    public void updateSignerCapabilities(X509Certificate certificate,
            SignerCapabilities capabilities) {
        if (certificate == null) {
            throw new NullPointerException("config == null");
        }

        for (int i = 0; i < mSigningLineage.size(); i++) {
            SigningCertificateNode lineageNode = mSigningLineage.get(i);
            if (lineageNode.signingCert.equals(certificate)) {
                int flags = lineageNode.flags;
                SignerCapabilities newCapabilities = new SignerCapabilities.Builder(
                        flags).setCallerConfiguredCapabilities(capabilities).build();
                lineageNode.flags = newCapabilities.getFlags();
                return;
            }
        }

        // the provided signer config was not found in the lineage
        throw new IllegalArgumentException("Certificate (" + certificate.getSubjectDN()
                + ") not found in the SigningCertificateLineage");
    }

    /**
     * Returns a list containing all of the certificates in the lineage.
     */
    public List<X509Certificate> getCertificatesInLineage() {
        List<X509Certificate> certs = new ArrayList<>();
        for (int i = 0; i < mSigningLineage.size(); i++) {
            X509Certificate cert = mSigningLineage.get(i).signingCert;
            certs.add(cert);
        }
        return certs;
    }

    /**
     * Returns {@code true} if the specified config is in the lineage.
     */
    public boolean isSignerInLineage(SignerConfig config) {
        if (config == null) {
            throw new NullPointerException("config == null");
        }

        X509Certificate cert = config.getCertificate();
        return isCertificateInLineage(cert);
    }

    /**
     * Returns {@code true} if the specified certificate is in the lineage.
     */
    public boolean isCertificateInLineage(X509Certificate cert) {
        if (cert == null) {
            throw new NullPointerException("cert == null");
        }

        for (int i = 0; i < mSigningLineage.size(); i++) {
            if (mSigningLineage.get(i).signingCert.equals(cert)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns whether the provided {@code cert} is the latest signing certificate in the lineage.
     *
     * <p>This method will only compare the provided {@code cert} against the latest signing
     * certificate in the lineage; if a certificate that is not in the lineage is provided, this
     * method will return false.
     */
    public boolean isCertificateLatestInLineage(X509Certificate cert) {
        if (cert == null) {
            throw new NullPointerException("cert == null");
        }

        return mSigningLineage.get(mSigningLineage.size() - 1).signingCert.equals(cert);
    }

    private static int calculateDefaultFlags() {
        return PAST_CERT_INSTALLED_DATA | PAST_CERT_PERMISSION
                | PAST_CERT_SHARED_USER_ID | PAST_CERT_AUTH;
    }

    /**
     * Returns a new SigningCertificateLineage which terminates at the node corresponding to the
     * given certificate.  This is useful in the event of rotating to a new signing algorithm that
     * is only supported on some platform versions.  It enables a v3 signature to be generated using
     * this signing certificate and the shortened proof-of-rotation record from this sub lineage in
     * conjunction with the appropriate SDK version values.
     *
     * @param x509Certificate the signing certificate for which to search
     * @return A new SigningCertificateLineage if the given certificate is present.
     *
     * @throws IllegalArgumentException if the provided certificate is not in the lineage.
     */
    public SigningCertificateLineage getSubLineage(X509Certificate x509Certificate) {
        if (x509Certificate == null) {
            throw new NullPointerException("x509Certificate == null");
        }
        for (int i = 0; i < mSigningLineage.size(); i++) {
            if (mSigningLineage.get(i).signingCert.equals(x509Certificate)) {
                return new SigningCertificateLineage(
                        mMinSdkVersion, new ArrayList<>(mSigningLineage.subList(0, i + 1)));
            }
        }

        // looks like we didn't find the cert,
        throw new IllegalArgumentException("Certificate not found in SigningCertificateLineage");
    }

    /**
     * Consolidates all of the lineages found in an APK into one lineage. In so doing, it also
     * checks that all of the lineages are contained in one common lineage.
     *
     * An APK may contain multiple lineages, one for each signer, which correspond to different
     * supported platform versions.  In this event, the lineage(s) from the earlier platform
     * version(s) should be present in the most recent, either directly or via a sublineage
     * that would allow the earlier lineages to merge with the most recent.
     *
     * <note> This does not verify that the largest lineage corresponds to the most recent supported
     * platform version.  That check is performed during v3 verification. </note>
     */
    public static SigningCertificateLineage consolidateLineages(
            List<SigningCertificateLineage> lineages) {
        if (lineages == null || lineages.isEmpty()) {
            return null;
        }
        SigningCertificateLineage consolidatedLineage = lineages.get(0);
        for (int i = 1; i < lineages.size(); i++) {
            consolidatedLineage = consolidatedLineage.mergeLineageWith(lineages.get(i));
        }
        return consolidatedLineage;
    }

    /**
     * Merges this lineage with the provided {@code otherLineage}.
     *
     * <p>The merged lineage does not currently handle merging capabilities of common signers and
     * should only be used to determine the full signing history of a collection of lineages.
     */
    public SigningCertificateLineage mergeLineageWith(SigningCertificateLineage otherLineage) {
        // Determine the ancestor and descendant lineages; if the original signer is in the other
        // lineage, then it is considered a descendant.
        SigningCertificateLineage ancestorLineage;
        SigningCertificateLineage descendantLineage;
        X509Certificate signerCert = mSigningLineage.get(0).signingCert;
        if (otherLineage.isCertificateInLineage(signerCert)) {
            descendantLineage = this;
            ancestorLineage = otherLineage;
        } else {
            descendantLineage = otherLineage;
            ancestorLineage = this;
        }

        int ancestorIndex = 0;
        int descendantIndex = 0;
        SigningCertificateNode ancestorNode;
        SigningCertificateNode descendantNode = descendantLineage.mSigningLineage.get(
                descendantIndex++);
        List<SigningCertificateNode> mergedLineage = new ArrayList<>();
        // Iterate through the ancestor lineage and add the current node to the resulting lineage
        // until the first node of the descendant is found.
        while (ancestorIndex < ancestorLineage.size()) {
            ancestorNode = ancestorLineage.mSigningLineage.get(ancestorIndex++);
            if (ancestorNode.signingCert.equals(descendantNode.signingCert)) {
                break;
            }
            mergedLineage.add(ancestorNode);
        }
        // If all of the nodes in the ancestor lineage have been added to the merged lineage, then
        // there is no overlap between this and the provided lineage.
        if (ancestorIndex == mergedLineage.size()) {
            throw new IllegalArgumentException(
                    "The provided lineage is not a descendant or an ancestor of this lineage");
        }
        // The descendant lineage's first node was in the ancestor's lineage above; add it to the
        // merged lineage.
        mergedLineage.add(descendantNode);
        while (ancestorIndex < ancestorLineage.size()
                && descendantIndex < descendantLineage.size()) {
            ancestorNode = ancestorLineage.mSigningLineage.get(ancestorIndex++);
            descendantNode = descendantLineage.mSigningLineage.get(descendantIndex++);
            if (!ancestorNode.signingCert.equals(descendantNode.signingCert)) {
                throw new IllegalArgumentException(
                        "The provided lineage diverges from this lineage");
            }
            mergedLineage.add(descendantNode);
        }
        // At this point, one or both of the lineages have been exhausted and all signers to this
        // point were a match between the two lineages; add any remaining elements from either
        // lineage to the merged lineage.
        while (ancestorIndex < ancestorLineage.size()) {
            mergedLineage.add(ancestorLineage.mSigningLineage.get(ancestorIndex++));
        }
        while (descendantIndex < descendantLineage.size()) {
            mergedLineage.add(descendantLineage.mSigningLineage.get(descendantIndex++));
        }
        return new SigningCertificateLineage(Math.min(mMinSdkVersion, otherLineage.mMinSdkVersion),
                mergedLineage);
    }

    /**
     * Checks whether given lineages are compatible. Returns {@code true} if an installed APK with
     * the oldLineage could be updated with an APK with the newLineage.
     */
    public static boolean checkLineagesCompatibility(
        SigningCertificateLineage oldLineage, SigningCertificateLineage newLineage) {

        final ArrayList<X509Certificate> oldCertificates = oldLineage == null ?
                new ArrayList<X509Certificate>()
                : new ArrayList(oldLineage.getCertificatesInLineage());
        final ArrayList<X509Certificate> newCertificates = newLineage == null ?
                new ArrayList<X509Certificate>()
                : new ArrayList(newLineage.getCertificatesInLineage());

        if (oldCertificates.isEmpty()) {
            return true;
        }
        if (newCertificates.isEmpty()) {
            return false;
        }

        // Both lineages contain exactly the same certificates or the new lineage extends
        // the old one. The capabilities of particular certificates may have changed though but it
        // does not matter in terms of current compatibility.
        if (newCertificates.size() >= oldCertificates.size()
                && newCertificates.subList(0, oldCertificates.size()).equals(oldCertificates)) {
            return true;
        }

        ArrayList<X509Certificate> newCertificatesArray = new ArrayList(newCertificates);
        ArrayList<X509Certificate> oldCertificatesArray = new ArrayList(oldCertificates);

        int lastOldCertIndexInNew = newCertificatesArray.lastIndexOf(
                    oldCertificatesArray.get(oldCertificatesArray.size()-1));

        // The new lineage trims some nodes from the beginning of the old lineage and possibly
        // extends it at the end. The new lineage must contain the old signing certificate and
        // the nodes up until the node with signing certificate must be in the same order.
        // Good example 1:
        //    old: A -> B -> C
        //    new: B -> C -> D
        // Good example 2:
        //    old: A -> B -> C
        //    new: C
        // Bad example 1:
        //    old: A -> B -> C
        //    new: A -> C
        // Bad example 1:
        //    old: A -> B
        //    new: C -> B
        if (lastOldCertIndexInNew >= 0) {
            return newCertificatesArray.subList(0, lastOldCertIndexInNew+1).equals(
                    oldCertificatesArray.subList(
                            oldCertificates.size()-1-lastOldCertIndexInNew,
                            oldCertificatesArray.size()));
        }


        // The new lineage can be shorter than the old one only if the last certificate of the new
        // lineage exists in the old lineage and has a rollback capability there.
        // Good example:
        //    old: A -> B_withRollbackCapability -> C
        //    new: A -> B
        // Bad example 1:
        //    old: A -> B -> C
        //    new: A -> B
        // Bad example 2:
        //    old: A -> B_withRollbackCapability -> C
        //    new: A -> B -> D
        return  oldCertificates.subList(0, newCertificates.size()).equals(newCertificates)
                && oldLineage.getSignerCapabilities(
                        oldCertificates.get(newCertificates.size()-1)).hasRollback();
    }

    /**
     * Representation of the capabilities the APK would like to grant to its old signing
     * certificates.  The {@code SigningCertificateLineage} provides two conceptual data structures.
     *   1) proof of rotation - Evidence that other parties can trust an APK's current signing
     *      certificate if they trust an older one in this lineage
     *   2) self-trust - certain capabilities may have been granted by an APK to other parties based
     *      on its own signing certificate.  When it changes its signing certificate it may want to
     *      allow the other parties to retain those capabilities.
     * {@code SignerCapabilties} provides a representation of the second structure.
     *
     * <p>Use {@link Builder} to obtain configuration instances.
     */
    public static class SignerCapabilities {
        private final int mFlags;

        private final int mCallerConfiguredFlags;

        private SignerCapabilities(int flags) {
            this(flags, 0);
        }

        private SignerCapabilities(int flags, int callerConfiguredFlags) {
            mFlags = flags;
            mCallerConfiguredFlags = callerConfiguredFlags;
        }

        private int getFlags() {
            return mFlags;
        }

        /**
         * Returns {@code true} if the capabilities of this object match those of the provided
         * object.
         */
        @Override
        public boolean equals(Object other) {
            if (this == other) return true;
            if (!(other instanceof SignerCapabilities)) return false;

            return this.mFlags == ((SignerCapabilities) other).mFlags;
        }

        @Override
        public int hashCode() {
            return 31 * mFlags;
        }

        /**
         * Returns {@code true} if this object has the installed data capability.
         */
        public boolean hasInstalledData() {
            return (mFlags & PAST_CERT_INSTALLED_DATA) != 0;
        }

        /**
         * Returns {@code true} if this object has the shared UID capability.
         */
        public boolean hasSharedUid() {
            return (mFlags & PAST_CERT_SHARED_USER_ID) != 0;
        }

        /**
         * Returns {@code true} if this object has the permission capability.
         */
        public boolean hasPermission() {
            return (mFlags & PAST_CERT_PERMISSION) != 0;
        }

        /**
         * Returns {@code true} if this object has the rollback capability.
         */
        public boolean hasRollback() {
            return (mFlags & PAST_CERT_ROLLBACK) != 0;
        }

        /**
         * Returns {@code true} if this object has the auth capability.
         */
        public boolean hasAuth() {
            return (mFlags & PAST_CERT_AUTH) != 0;
        }

        /**
         * Builder of {@link SignerCapabilities} instances.
         */
        public static class Builder {
            private int mFlags;

            private int mCallerConfiguredFlags;

            /**
             * Constructs a new {@code Builder}.
             */
            public Builder() {
                mFlags = calculateDefaultFlags();
            }

            /**
             * Constructs a new {@code Builder} with the initial capabilities set to the provided
             * flags.
             */
            public Builder(int flags) {
                mFlags = flags;
            }

            /**
             * Set the {@code PAST_CERT_INSTALLED_DATA} flag in this capabilities object.  This flag
             * is used by the platform to determine if installed data associated with previous
             * signing certificate should be trusted.  In particular, this capability is required to
             * perform signing certificate rotation during an upgrade on-device.  Without it, the
             * platform will not permit the app data from the old signing certificate to
             * propagate to the new version.  Typically, this flag should be set to enable signing
             * certificate rotation, and may be unset later when the app developer is satisfied that
             * their install base is as migrated as it will be.
             */
            public Builder setInstalledData(boolean enabled) {
                mCallerConfiguredFlags |= PAST_CERT_INSTALLED_DATA;
                if (enabled) {
                    mFlags |= PAST_CERT_INSTALLED_DATA;
                } else {
                    mFlags &= ~PAST_CERT_INSTALLED_DATA;
                }
                return this;
            }

            /**
             * Set the {@code PAST_CERT_SHARED_USER_ID} flag in this capabilities object.  This flag
             * is used by the platform to determine if this app is willing to be sharedUid with
             * other apps which are still signed with the associated signing certificate.  This is
             * useful in situations where sharedUserId apps would like to change their signing
             * certificate, but can't guarantee the order of updates to those apps.
             */
            public Builder setSharedUid(boolean enabled) {
                mCallerConfiguredFlags |= PAST_CERT_SHARED_USER_ID;
                if (enabled) {
                    mFlags |= PAST_CERT_SHARED_USER_ID;
                } else {
                    mFlags &= ~PAST_CERT_SHARED_USER_ID;
                }
                return this;
            }

            /**
             * Set the {@code PAST_CERT_PERMISSION} flag in this capabilities object.  This flag
             * is used by the platform to determine if this app is willing to grant SIGNATURE
             * permissions to apps signed with the associated signing certificate.  Without this
             * capability, an application signed with the older certificate will not be granted the
             * SIGNATURE permissions defined by this app.  In addition, if multiple apps define the
             * same SIGNATURE permission, the second one the platform sees will not be installable
             * if this capability is not set and the signing certificates differ.
             */
            public Builder setPermission(boolean enabled) {
                mCallerConfiguredFlags |= PAST_CERT_PERMISSION;
                if (enabled) {
                    mFlags |= PAST_CERT_PERMISSION;
                } else {
                    mFlags &= ~PAST_CERT_PERMISSION;
                }
                return this;
            }

            /**
             * Set the {@code PAST_CERT_ROLLBACK} flag in this capabilities object.  This flag
             * is used by the platform to determine if this app is willing to upgrade to a new
             * version that is signed by one of its past signing certificates.
             *
             * <note> WARNING: this effectively removes any benefit of signing certificate changes,
             * since a compromised key could retake control of an app even after change, and should
             * only be used if there is a problem encountered when trying to ditch an older cert
             * </note>
             */
            public Builder setRollback(boolean enabled) {
                mCallerConfiguredFlags |= PAST_CERT_ROLLBACK;
                if (enabled) {
                    mFlags |= PAST_CERT_ROLLBACK;
                } else {
                    mFlags &= ~PAST_CERT_ROLLBACK;
                }
                return this;
            }

            /**
             * Set the {@code PAST_CERT_AUTH} flag in this capabilities object.  This flag
             * is used by the platform to determine whether or not privileged access based on
             * authenticator module signing certificates should be granted.
             */
            public Builder setAuth(boolean enabled) {
                mCallerConfiguredFlags |= PAST_CERT_AUTH;
                if (enabled) {
                    mFlags |= PAST_CERT_AUTH;
                } else {
                    mFlags &= ~PAST_CERT_AUTH;
                }
                return this;
            }

            /**
             * Applies the capabilities that were explicitly set in the provided capabilities object
             * to this builder. Any values that were not set will not be applied to this builder
             * to prevent unintentinoally setting a capability back to a default value.
             */
            public Builder setCallerConfiguredCapabilities(SignerCapabilities capabilities) {
                // The mCallerConfiguredFlags should have a bit set for each capability that was
                // set by a caller. If a capability was explicitly set then the corresponding bit
                // in mCallerConfiguredFlags should be set. This allows the provided capabilities
                // to take effect for those set by the caller while those that were not set will
                // be cleared by the bitwise and and the initial value for the builder will remain.
                mFlags = (mFlags & ~capabilities.mCallerConfiguredFlags) |
                        (capabilities.mFlags & capabilities.mCallerConfiguredFlags);
                return this;
            }

            /**
             * Returns a new {@code SignerConfig} instance configured based on the configuration of
             * this builder.
             */
            public SignerCapabilities build() {
                return new SignerCapabilities(mFlags, mCallerConfiguredFlags);
            }
        }
    }

    /**
     * Configuration of a signer.  Used to add a new entry to the {@link SigningCertificateLineage}
     *
     * <p>Use {@link Builder} to obtain configuration instances.
     */
    public static class SignerConfig {
        private final PrivateKey mPrivateKey;
        private final X509Certificate mCertificate;

        private SignerConfig(
                PrivateKey privateKey,
                X509Certificate certificate) {
            mPrivateKey = privateKey;
            mCertificate = certificate;
        }

        /**
         * Returns the signing key of this signer.
         */
        public PrivateKey getPrivateKey() {
            return mPrivateKey;
        }

        /**
         * Returns the certificate(s) of this signer. The first certificate's public key corresponds
         * to this signer's private key.
         */
        public X509Certificate getCertificate() {
            return mCertificate;
        }

        /**
         * Builder of {@link SignerConfig} instances.
         */
        public static class Builder {
            private final PrivateKey mPrivateKey;
            private final X509Certificate mCertificate;

            /**
             * Constructs a new {@code Builder}.
             *
             * @param privateKey signing key
             * @param certificate the X.509 certificate with a subject public key of the
             * {@code privateKey}.
             */
            public Builder(
                    PrivateKey privateKey,
                    X509Certificate certificate) {
                mPrivateKey = privateKey;
                mCertificate = certificate;
            }

            /**
             * Returns a new {@code SignerConfig} instance configured based on the configuration of
             * this builder.
             */
            public SignerConfig build() {
                return new SignerConfig(
                        mPrivateKey,
                        mCertificate);
            }
        }
    }

    /**
     * Builder of {@link SigningCertificateLineage} instances.
     */
    public static class Builder {
        private final SignerConfig mOriginalSignerConfig;
        private final SignerConfig mNewSignerConfig;
        private SignerCapabilities mOriginalCapabilities;
        private SignerCapabilities mNewCapabilities;
        private int mMinSdkVersion;
        /**
         * Constructs a new {@code Builder}.
         *
         * @param originalSignerConfig first signer in this lineage, parent of the next
         * @param newSignerConfig new signer in the lineage; the new signing key that the APK will
         *                        use
         */
        public Builder(
                SignerConfig originalSignerConfig,
                SignerConfig newSignerConfig) {
            if (originalSignerConfig == null || newSignerConfig == null) {
                throw new NullPointerException("Can't pass null SignerConfigs when constructing a "
                        + "new SigningCertificateLineage");
            }
            mOriginalSignerConfig = originalSignerConfig;
            mNewSignerConfig = newSignerConfig;
        }

        /**
         * Constructs a new {@code Builder} that is intended to create a {@code
         * SigningCertificateLineage} with a single signer in the signing history.
         *
         * @param originalSignerConfig first signer in this lineage
         */
        public Builder(SignerConfig originalSignerConfig) {
            if (originalSignerConfig == null) {
                throw new NullPointerException("Can't pass null SignerConfigs when constructing a "
                        + "new SigningCertificateLineage");
            }
            mOriginalSignerConfig = originalSignerConfig;
            mNewSignerConfig = null;
        }

        /**
         * Sets the minimum Android platform version (API Level) on which this lineage is expected
         * to validate.  It is possible that newer signers in the lineage may not be recognized on
         * the given platform, but as long as an older signer is, the lineage can still be used to
         * sign an APK for the given platform.
         *
         * <note> By default, this value is set to the value for the
         * P release, since this structure was created for that release, and will also be set to
         * that value if a smaller one is specified. </note>
         */
        public Builder setMinSdkVersion(int minSdkVersion) {
            mMinSdkVersion = minSdkVersion;
            return this;
        }

        /**
         * Sets capabilities to give {@code mOriginalSignerConfig}. These capabilities allow an
         * older signing certificate to still be used in some situations on the platform even though
         * the APK is now being signed by a newer signing certificate.
         */
        public Builder setOriginalCapabilities(SignerCapabilities signerCapabilities) {
            if (signerCapabilities == null) {
                throw new NullPointerException("signerCapabilities == null");
            }
            mOriginalCapabilities = signerCapabilities;
            return this;
        }

        /**
         * Sets capabilities to give {@code mNewSignerConfig}. These capabilities allow an
         * older signing certificate to still be used in some situations on the platform even though
         * the APK is now being signed by a newer signing certificate.  By default, the new signer
         * will have all capabilities, so when first switching to a new signing certificate, these
         * capabilities have no effect, but they will act as the default level of trust when moving
         * to a new signing certificate.
         */
        public Builder setNewCapabilities(SignerCapabilities signerCapabilities) {
            if (signerCapabilities == null) {
                throw new NullPointerException("signerCapabilities == null");
            }
            mNewCapabilities = signerCapabilities;
            return this;
        }

        public SigningCertificateLineage build()
                throws CertificateEncodingException, InvalidKeyException, NoSuchAlgorithmException,
                SignatureException {
            if (mMinSdkVersion < AndroidSdkVersion.P) {
                mMinSdkVersion = AndroidSdkVersion.P;
            }

            if (mOriginalCapabilities == null) {
                mOriginalCapabilities = new SignerCapabilities.Builder().build();
            }

            if (mNewSignerConfig == null) {
                return createSigningLineage(mMinSdkVersion, mOriginalSignerConfig,
                        mOriginalCapabilities);
            }

            if (mNewCapabilities == null) {
                mNewCapabilities = new SignerCapabilities.Builder().build();
            }

            return createSigningLineage(
                    mMinSdkVersion, mOriginalSignerConfig, mOriginalCapabilities,
                    mNewSignerConfig, mNewCapabilities);
        }
    }
}
