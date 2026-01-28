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
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.getLengthPrefixedSlice;
import static com.android.apksig.internal.apk.ApkSigningBlockUtils.readLengthPrefixedByteArray;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import com.android.apksig.internal.util.X509CertificateUtils;

import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

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
public class V3SigningCertificateLineage {

    private final static int FIRST_VERSION = 1;
    private final static int CURRENT_VERSION = FIRST_VERSION;

    /**
     * Deserializes the binary representation of an {@link V3SigningCertificateLineage}. Also
     * verifies that the structure is well-formed, e.g. that the signature for each node is from its
     * parent.
     */
    public static List<SigningCertificateNode> readSigningCertificateLineage(ByteBuffer inputBytes)
            throws IOException {
        List<SigningCertificateNode> result = new ArrayList<>();
        int nodeCount = 0;
        if (inputBytes == null || !inputBytes.hasRemaining()) {
            return null;
        }

        ApkSigningBlockUtils.checkByteOrderLittleEndian(inputBytes);

        // FORMAT (little endian):
        // * uint32: version code
        // * sequence of length-prefixed (uint32): nodes
        //   * length-prefixed bytes: signed data
        //     * length-prefixed bytes: certificate
        //     * uint32: signature algorithm id
        //   * uint32: flags
        //   * uint32: signature algorithm id (used by to sign next cert in lineage)
        //   * length-prefixed bytes: signature over above signed data

        X509Certificate lastCert = null;
        int lastSigAlgorithmId = 0;

        try {
            int version = inputBytes.getInt();
            if (version != CURRENT_VERSION) {
                // we only have one version to worry about right now, so just check it
                throw new IllegalArgumentException("Encoded SigningCertificateLineage has a version"
                        + " different than any of which we are aware");
            }
            HashSet<X509Certificate> certHistorySet = new HashSet<>();
            while (inputBytes.hasRemaining()) {
                nodeCount++;
                ByteBuffer nodeBytes = getLengthPrefixedSlice(inputBytes);
                ByteBuffer signedData = getLengthPrefixedSlice(nodeBytes);
                int flags = nodeBytes.getInt();
                int sigAlgorithmId = nodeBytes.getInt();
                SignatureAlgorithm sigAlgorithm = SignatureAlgorithm.findById(lastSigAlgorithmId);
                byte[] signature = readLengthPrefixedByteArray(nodeBytes);

                if (lastCert != null) {
                    // Use previous level cert to verify current level
                    String jcaSignatureAlgorithm =
                            sigAlgorithm.getJcaSignatureAlgorithmAndParams().getFirst();
                    AlgorithmParameterSpec jcaSignatureAlgorithmParams =
                            sigAlgorithm.getJcaSignatureAlgorithmAndParams().getSecond();
                    PublicKey publicKey = lastCert.getPublicKey();
                    Signature sig = Signature.getInstance(jcaSignatureAlgorithm);
                    sig.initVerify(publicKey);
                    if (jcaSignatureAlgorithmParams != null) {
                        sig.setParameter(jcaSignatureAlgorithmParams);
                    }
                    sig.update(signedData);
                    if (!sig.verify(signature)) {
                        throw new SecurityException("Unable to verify signature of certificate #"
                                + nodeCount + " using " + jcaSignatureAlgorithm + " when verifying"
                                + " V3SigningCertificateLineage object");
                    }
                }

                signedData.rewind();
                byte[] encodedCert = readLengthPrefixedByteArray(signedData);
                int signedSigAlgorithm = signedData.getInt();
                if (lastCert != null && lastSigAlgorithmId != signedSigAlgorithm) {
                    throw new SecurityException("Signing algorithm ID mismatch for certificate #"
                            + nodeBytes + " when verifying V3SigningCertificateLineage object");
                }
                lastCert = X509CertificateUtils.generateCertificate(encodedCert);
                lastCert = new GuaranteedEncodedFormX509Certificate(lastCert, encodedCert);
                if (certHistorySet.contains(lastCert)) {
                    throw new SecurityException("Encountered duplicate entries in "
                            + "SigningCertificateLineage at certificate #" + nodeCount + ".  All "
                            + "signing certificates should be unique");
                }
                certHistorySet.add(lastCert);
                lastSigAlgorithmId = sigAlgorithmId;
                result.add(new SigningCertificateNode(
                        lastCert, SignatureAlgorithm.findById(signedSigAlgorithm),
                        SignatureAlgorithm.findById(sigAlgorithmId), signature, flags));
            }
        } catch(ApkFormatException | BufferUnderflowException e){
            throw new IOException("Failed to parse V3SigningCertificateLineage object", e);
        } catch(NoSuchAlgorithmException | InvalidKeyException
                | InvalidAlgorithmParameterException | SignatureException e){
            throw new SecurityException(
                    "Failed to verify signature over signed data for certificate #" + nodeCount
                            + " when parsing V3SigningCertificateLineage object", e);
        } catch(CertificateException e){
            throw new SecurityException("Failed to decode certificate #" + nodeCount
                    + " when parsing V3SigningCertificateLineage object", e);
        }
        return result;
    }

    /**
     * encode the in-memory representation of this {@code V3SigningCertificateLineage}
     */
    public static byte[] encodeSigningCertificateLineage(
            List<SigningCertificateNode> signingCertificateLineage) {
        // FORMAT (little endian):
        // * version code
        // * sequence of length-prefixed (uint32): nodes
        //   * length-prefixed bytes: signed data
        //     * length-prefixed bytes: certificate
        //     * uint32: signature algorithm id
        //   * uint32: flags
        //   * uint32: signature algorithm id (used by to sign next cert in lineage)

        List<byte[]> nodes = new ArrayList<>();
        for (SigningCertificateNode node : signingCertificateLineage) {
            nodes.add(encodeSigningCertificateNode(node));
        }
        byte [] encodedSigningCertificateLineage = encodeAsSequenceOfLengthPrefixedElements(nodes);

        // add the version code (uint32) on top of the encoded nodes
        int payloadSize = 4 + encodedSigningCertificateLineage.length;
        ByteBuffer encodedWithVersion = ByteBuffer.allocate(payloadSize);
        encodedWithVersion.order(ByteOrder.LITTLE_ENDIAN);
        encodedWithVersion.putInt(CURRENT_VERSION);
        encodedWithVersion.put(encodedSigningCertificateLineage);
        return encodedWithVersion.array();
    }

    public static byte[] encodeSigningCertificateNode(SigningCertificateNode node) {
        // FORMAT (little endian):
        // * length-prefixed bytes: signed data
        //   * length-prefixed bytes: certificate
        //   * uint32: signature algorithm id
        // * uint32: flags
        // * uint32: signature algorithm id (used by to sign next cert in lineage)
        // * length-prefixed bytes: signature over signed data
        int parentSigAlgorithmId = 0;
        if (node.parentSigAlgorithm != null) {
            parentSigAlgorithmId = node.parentSigAlgorithm.getId();
        }
        int sigAlgorithmId = 0;
        if (node.sigAlgorithm != null) {
            sigAlgorithmId = node.sigAlgorithm.getId();
        }
        byte[] prefixedSignedData = encodeSignedData(node.signingCert, parentSigAlgorithmId);
        byte[] prefixedSignature = encodeAsLengthPrefixedElement(node.signature);
        int payloadSize = prefixedSignedData.length + 4 + 4 + prefixedSignature.length;
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.put(prefixedSignedData);
        result.putInt(node.flags);
        result.putInt(sigAlgorithmId);
        result.put(prefixedSignature);
        return result.array();
    }

    public static byte[] encodeSignedData(X509Certificate certificate, int flags) {
        try {
            byte[] prefixedCertificate = encodeAsLengthPrefixedElement(certificate.getEncoded());
            int payloadSize = 4 + prefixedCertificate.length;
            ByteBuffer result = ByteBuffer.allocate(payloadSize);
            result.order(ByteOrder.LITTLE_ENDIAN);
            result.put(prefixedCertificate);
            result.putInt(flags);
            return encodeAsLengthPrefixedElement(result.array());
        } catch (CertificateEncodingException e) {
            throw new RuntimeException(
                    "Failed to encode V3SigningCertificateLineage certificate", e);
        }
    }

    /**
     * Represents one signing certificate in the {@link V3SigningCertificateLineage}, which
     * generally means it is/was used at some point to sign the same APK of the others in the
     * lineage.
     */
    public static class SigningCertificateNode {

        public SigningCertificateNode(
                X509Certificate signingCert,
                SignatureAlgorithm parentSigAlgorithm,
                SignatureAlgorithm sigAlgorithm,
                byte[] signature,
                int flags) {
            this.signingCert = signingCert;
            this.parentSigAlgorithm = parentSigAlgorithm;
            this.sigAlgorithm = sigAlgorithm;
            this.signature = signature;
            this.flags = flags;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SigningCertificateNode)) return false;

            SigningCertificateNode that = (SigningCertificateNode) o;
            if (!signingCert.equals(that.signingCert)) return false;
            if (parentSigAlgorithm != that.parentSigAlgorithm) return false;
            if (sigAlgorithm != that.sigAlgorithm) return false;
            if (!Arrays.equals(signature, that.signature)) return false;
            if (flags != that.flags) return false;

            // we made it
            return true;
        }

        @Override
        public int hashCode() {
            int result = Objects.hash(signingCert, parentSigAlgorithm, sigAlgorithm, flags);
            result = 31 * result + Arrays.hashCode(signature);
            return result;
        }

        /**
         * the signing cert for this node.  This is part of the data signed by the parent node.
         */
        public final X509Certificate signingCert;

        /**
         * the algorithm used by the this node's parent to bless this data.  Its ID value is part of
         * the data signed by the parent node. {@code null} for first node.
         */
        public final SignatureAlgorithm parentSigAlgorithm;

        /**
         * the algorithm used by the this nodeto bless the next node's data.  Its ID value is part
         * of the signed data of the next node. {@code null} for the last node.
         */
        public SignatureAlgorithm sigAlgorithm;

        /**
         * signature over the signed data (above).  The signature is from this node's parent
         * signing certificate, which should correspond to the signing certificate used to sign an
         * APK before rotating to this one, and is formed using {@code signatureAlgorithm}.
         */
        public final byte[] signature;

        /**
         * the flags detailing how the platform should treat this signing cert
         */
        public int flags;
    }
}
