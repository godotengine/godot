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
import static com.android.apksig.internal.apk.ApkSigningBlockUtilsLite.readLengthPrefixedByteArray;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.internal.apk.ApkSigningBlockUtilsLite;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
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
import java.util.HashSet;
import java.util.List;

/** Lightweight version of the V3SigningCertificateLineage to be used for source stamps. */
public class SourceStampCertificateLineage {

    private final static int FIRST_VERSION = 1;
    private final static int CURRENT_VERSION = FIRST_VERSION;

    /**
     * Deserializes the binary representation of a SourceStampCertificateLineage. Also
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

        ApkSigningBlockUtilsLite.checkByteOrderLittleEndian(inputBytes);

        CertificateFactory certFactory;
        try {
            certFactory = CertificateFactory.getInstance("X.509");
        } catch (CertificateException e) {
            throw new IllegalStateException("Failed to obtain X.509 CertificateFactory", e);
        }

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
                                + " SourceStampCertificateLineage object");
                    }
                }

                signedData.rewind();
                byte[] encodedCert = readLengthPrefixedByteArray(signedData);
                int signedSigAlgorithm = signedData.getInt();
                if (lastCert != null && lastSigAlgorithmId != signedSigAlgorithm) {
                    throw new SecurityException("Signing algorithm ID mismatch for certificate #"
                            + nodeBytes + " when verifying SourceStampCertificateLineage object");
                }
                lastCert = (X509Certificate) certFactory.generateCertificate(
                    new ByteArrayInputStream(encodedCert));
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
            throw new IOException("Failed to parse SourceStampCertificateLineage object", e);
        } catch(NoSuchAlgorithmException | InvalidKeyException
                | InvalidAlgorithmParameterException | SignatureException e){
            throw new SecurityException(
                    "Failed to verify signature over signed data for certificate #" + nodeCount
                            + " when parsing SourceStampCertificateLineage object", e);
        } catch(CertificateException e){
            throw new SecurityException("Failed to decode certificate #" + nodeCount
                    + " when parsing SourceStampCertificateLineage object", e);
        }
        return result;
    }

    /**
     * Represents one signing certificate in the SourceStampCertificateLineage, which
     * generally means it is/was used at some point to sign source stamps.
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
            final int prime = 31;
            int result = 1;
            result = prime * result + ((signingCert == null) ? 0 : signingCert.hashCode());
            result = prime * result +
                ((parentSigAlgorithm == null) ? 0 : parentSigAlgorithm.hashCode());
            result = prime * result + ((sigAlgorithm == null) ? 0 : sigAlgorithm.hashCode());
            result = prime * result + Arrays.hashCode(signature);
            result = prime * result + flags;
            return result;
        }

        /**
         * the signing cert for this node.  This is part of the data signed by the parent node.
         */
        public final X509Certificate signingCert;

        /**
         * the algorithm used by this node's parent to bless this data.  Its ID value is part of
         * the data signed by the parent node. {@code null} for first node.
         */
        public final SignatureAlgorithm parentSigAlgorithm;

        /**
         * the algorithm used by this node to bless the next node's data.  Its ID value is part
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
