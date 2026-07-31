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

import static com.android.apksig.internal.apk.ApkSigningBlockUtils.toHex;

import com.android.apksig.ApkVerifier;
import com.android.apksig.ApkVerifier.Issue;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.apk.ContentDigestAlgorithm;
import com.android.apksig.internal.apk.SignatureAlgorithm;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import com.android.apksig.internal.util.X509CertificateUtils;
import com.android.apksig.util.DataSource;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.Arrays;

/**
 * APK Signature Scheme V4 verifier.
 * <p>
 * Verifies the serialized V4Signature file against an APK.
 */
public abstract class V4SchemeVerifier {
    /**
     * Hidden constructor to prevent instantiation.
     */
    private V4SchemeVerifier() {
    }

    /**
     * <p>
     * The main goals of the verifier are: 1) parse V4Signature file fields 2) verifies the PKCS7
     * signature block against the raw root hash bytes in the proto field 3) verifies that the raw
     * root hash matches with the actual hash tree root of the give APK 4) if the file contains a
     * verity tree, verifies that it matches with the actual verity tree computed from the given
     * APK.
     * </p>
     */
    public static ApkSigningBlockUtils.Result verify(DataSource apk, File v4SignatureFile)
            throws IOException, NoSuchAlgorithmException {
        final V4Signature signature;
        final byte[] tree;
        try (InputStream input = new FileInputStream(v4SignatureFile)) {
            signature = V4Signature.readFrom(input);
            tree = V4Signature.readBytes(input);
        }

        final ApkSigningBlockUtils.Result result = new ApkSigningBlockUtils.Result(
                ApkSigningBlockUtils.VERSION_APK_SIGNATURE_SCHEME_V4);

        if (signature == null) {
            result.addError(Issue.V4_SIG_NO_SIGNATURES,
                    "Signature file does not contain a v4 signature.");
            return result;
        }

        if (signature.version != V4Signature.CURRENT_VERSION) {
            result.addWarning(Issue.V4_SIG_VERSION_NOT_CURRENT, signature.version,
                    V4Signature.CURRENT_VERSION);
        }

        V4Signature.HashingInfo hashingInfo = V4Signature.HashingInfo.fromByteArray(
                signature.hashingInfo);

        V4Signature.SigningInfos signingInfos = V4Signature.SigningInfos.fromByteArray(
                signature.signingInfos);

        final ApkSigningBlockUtils.Result.SignerInfo signerInfo;

        // Verify the primary signature over signedData.
        {
            V4Signature.SigningInfo signingInfo = signingInfos.signingInfo;
            final byte[] signedData = V4Signature.getSignedData(apk.size(), hashingInfo,
                    signingInfo);
            signerInfo = parseAndVerifySignatureBlock(signingInfo, signedData);
            result.signers.add(signerInfo);
            if (result.containsErrors()) {
                return result;
            }
        }

        // Verify all subsequent signatures.
        for (V4Signature.SigningInfoBlock signingInfoBlock : signingInfos.signingInfoBlocks) {
            V4Signature.SigningInfo signingInfo = V4Signature.SigningInfo.fromByteArray(
                    signingInfoBlock.signingInfo);
            final byte[] signedData = V4Signature.getSignedData(apk.size(), hashingInfo,
                    signingInfo);
            result.signers.add(parseAndVerifySignatureBlock(signingInfo, signedData));
            if (result.containsErrors()) {
                return result;
            }
        }

        // Check if the root hash and the tree are correct.
        verifyRootHashAndTree(apk, signerInfo, hashingInfo.rawRootHash, tree);
        if (!result.containsErrors()) {
            result.verified = true;
        }

        return result;
    }

    /**
     * Parses the provided signature block and populates the {@code result}.
     * <p>
     * This verifies {@signingInfo} over {@code signedData}, as well as parsing the certificate
     * contained in the signature block. This method adds one or more errors to the {@code result}.
     */
    private static ApkSigningBlockUtils.Result.SignerInfo parseAndVerifySignatureBlock(
            V4Signature.SigningInfo signingInfo,
            final byte[] signedData) throws NoSuchAlgorithmException {
        final ApkSigningBlockUtils.Result.SignerInfo result =
                new ApkSigningBlockUtils.Result.SignerInfo();
        result.index = 0;

        final int sigAlgorithmId = signingInfo.signatureAlgorithmId;
        final byte[] sigBytes = signingInfo.signature;
        result.signatures.add(
                new ApkSigningBlockUtils.Result.SignerInfo.Signature(sigAlgorithmId, sigBytes));

        SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.findById(sigAlgorithmId);
        if (signatureAlgorithm == null) {
            result.addError(Issue.V4_SIG_UNKNOWN_SIG_ALGORITHM, sigAlgorithmId);
            return result;
        }

        String jcaSignatureAlgorithm =
                signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getFirst();
        AlgorithmParameterSpec jcaSignatureAlgorithmParams =
                signatureAlgorithm.getJcaSignatureAlgorithmAndParams().getSecond();

        String keyAlgorithm = signatureAlgorithm.getJcaKeyAlgorithm();

        final byte[] publicKeyBytes = signingInfo.publicKey;
        PublicKey publicKey;
        try {
            publicKey = KeyFactory.getInstance(keyAlgorithm).generatePublic(
                    new X509EncodedKeySpec(publicKeyBytes));
        } catch (Exception e) {
            result.addError(Issue.V4_SIG_MALFORMED_PUBLIC_KEY, e);
            return result;
        }

        try {
            Signature sig = Signature.getInstance(jcaSignatureAlgorithm);
            sig.initVerify(publicKey);
            if (jcaSignatureAlgorithmParams != null) {
                sig.setParameter(jcaSignatureAlgorithmParams);
            }
            sig.update(signedData);
            if (!sig.verify(sigBytes)) {
                result.addError(Issue.V4_SIG_DID_NOT_VERIFY, signatureAlgorithm);
                return result;
            }
            result.verifiedSignatures.put(signatureAlgorithm, sigBytes);
        } catch (InvalidKeyException | InvalidAlgorithmParameterException
                | SignatureException e) {
            result.addError(Issue.V4_SIG_VERIFY_EXCEPTION, signatureAlgorithm, e);
            return result;
        }

        if (signingInfo.certificate == null) {
            result.addError(Issue.V4_SIG_NO_CERTIFICATE);
            return result;
        }

        final X509Certificate certificate;
        try {
            // Wrap the cert so that the result's getEncoded returns exactly the original encoded
            // form. Without this, getEncoded may return a different form from what was stored in
            // the signature. This is because some X509Certificate(Factory) implementations
            // re-encode certificates.
            certificate = new GuaranteedEncodedFormX509Certificate(
                    X509CertificateUtils.generateCertificate(signingInfo.certificate),
                    signingInfo.certificate);
        } catch (CertificateException e) {
            result.addError(Issue.V4_SIG_MALFORMED_CERTIFICATE, e);
            return result;
        }
        result.certs.add(certificate);

        byte[] certificatePublicKeyBytes;
        try {
            certificatePublicKeyBytes = ApkSigningBlockUtils.encodePublicKey(
                    certificate.getPublicKey());
        } catch (InvalidKeyException e) {
            System.out.println("Caught an exception encoding the public key: " + e);
            e.printStackTrace();
            certificatePublicKeyBytes = certificate.getPublicKey().getEncoded();
        }
        if (!Arrays.equals(publicKeyBytes, certificatePublicKeyBytes)) {
            result.addError(
                    Issue.V4_SIG_PUBLIC_KEY_MISMATCH_BETWEEN_CERTIFICATE_AND_SIGNATURES_RECORD,
                    ApkSigningBlockUtils.toHex(certificatePublicKeyBytes),
                    ApkSigningBlockUtils.toHex(publicKeyBytes));
            return result;
        }

        // Add apk digest from the file to the result.
        ApkSigningBlockUtils.Result.SignerInfo.ContentDigest contentDigest =
                new ApkSigningBlockUtils.Result.SignerInfo.ContentDigest(
                        0 /* signature algorithm id doesn't matter here */, signingInfo.apkDigest);
        result.contentDigests.add(contentDigest);

        return result;
    }

    private static void verifyRootHashAndTree(DataSource apkContent,
            ApkSigningBlockUtils.Result.SignerInfo signerInfo, byte[] expectedDigest,
            byte[] expectedTree) throws IOException, NoSuchAlgorithmException {
        ApkSigningBlockUtils.VerityTreeAndDigest actualContentDigestInfo =
                ApkSigningBlockUtils.computeChunkVerityTreeAndDigest(apkContent);

        ContentDigestAlgorithm algorithm = actualContentDigestInfo.contentDigestAlgorithm;
        final byte[] actualDigest = actualContentDigestInfo.rootHash;
        final byte[] actualTree = actualContentDigestInfo.tree;

        if (!Arrays.equals(expectedDigest, actualDigest)) {
            signerInfo.addError(
                    ApkVerifier.Issue.V4_SIG_APK_ROOT_DID_NOT_VERIFY,
                    algorithm,
                    toHex(expectedDigest),
                    toHex(actualDigest));
            return;
        }
        // Only check verity tree if it is not empty
        if (expectedTree != null && !Arrays.equals(expectedTree, actualTree)) {
            signerInfo.addError(
                    ApkVerifier.Issue.V4_SIG_APK_TREE_DID_NOT_VERIFY,
                    algorithm,
                    toHex(expectedDigest),
                    toHex(actualDigest));
            return;
        }

        signerInfo.verifiedContentDigests.put(algorithm, actualDigest);
    }
}
