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

package com.android.apksig.internal.apk.v1;

import static com.android.apksig.Constants.MAX_APK_SIGNERS;
import static com.android.apksig.Constants.OID_RSA_ENCRYPTION;
import static com.android.apksig.internal.pkcs7.AlgorithmIdentifier.getSignerInfoDigestAlgorithmOid;
import static com.android.apksig.internal.pkcs7.AlgorithmIdentifier.getSignerInfoSignatureAlgorithm;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.asn1.Asn1EncodingException;
import com.android.apksig.internal.jar.ManifestWriter;
import com.android.apksig.internal.jar.SignatureFileWriter;
import com.android.apksig.internal.pkcs7.AlgorithmIdentifier;
import com.android.apksig.internal.util.Pair;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.jar.Attributes;
import java.util.jar.Manifest;

/**
 * APK signer which uses JAR signing (aka v1 signing scheme).
 *
 * @see <a href="https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File">Signed JAR File</a>
 */
public abstract class V1SchemeSigner {
    public static final String MANIFEST_ENTRY_NAME = V1SchemeConstants.MANIFEST_ENTRY_NAME;

    private static final Attributes.Name ATTRIBUTE_NAME_CREATED_BY =
            new Attributes.Name("Created-By");
    private static final String ATTRIBUTE_VALUE_MANIFEST_VERSION = "1.0";
    private static final String ATTRIBUTE_VALUE_SIGNATURE_VERSION = "1.0";

    private static final Attributes.Name SF_ATTRIBUTE_NAME_ANDROID_APK_SIGNED_NAME =
            new Attributes.Name(V1SchemeConstants.SF_ATTRIBUTE_NAME_ANDROID_APK_SIGNED_NAME_STR);

    /**
     * Signer configuration.
     */
    public static class SignerConfig {
        /** Name. */
        public String name;

        /** Private key. */
        public PrivateKey privateKey;

        /**
         * Certificates, with the first certificate containing the public key corresponding to
         * {@link #privateKey}.
         */
        public List<X509Certificate> certificates;

        /**
         * Digest algorithm used for the signature.
         */
        public DigestAlgorithm signatureDigestAlgorithm;

        /**
         * If DSA is the signing algorithm, whether or not deterministic DSA signing should be used.
         */
        public boolean deterministicDsaSigning;
    }

    /** Hidden constructor to prevent instantiation. */
    private V1SchemeSigner() {}

    /**
     * Gets the JAR signing digest algorithm to be used for signing an APK using the provided key.
     *
     * @param minSdkVersion minimum API Level of the platform on which the APK may be installed (see
     *        AndroidManifest.xml minSdkVersion attribute)
     *
     * @throws InvalidKeyException if the provided key is not suitable for signing APKs using
     *         JAR signing (aka v1 signature scheme)
     */
    public static DigestAlgorithm getSuggestedSignatureDigestAlgorithm(
            PublicKey signingKey, int minSdkVersion) throws InvalidKeyException {
        String keyAlgorithm = signingKey.getAlgorithm();
        if ("RSA".equalsIgnoreCase(keyAlgorithm) || OID_RSA_ENCRYPTION.equals((keyAlgorithm))) {
            // Prior to API Level 18, only SHA-1 can be used with RSA.
            if (minSdkVersion < 18) {
                return DigestAlgorithm.SHA1;
            }
            return DigestAlgorithm.SHA256;
        } else if ("DSA".equalsIgnoreCase(keyAlgorithm)) {
            // Prior to API Level 21, only SHA-1 can be used with DSA
            if (minSdkVersion < 21) {
                return DigestAlgorithm.SHA1;
            } else {
                return DigestAlgorithm.SHA256;
            }
        } else if ("EC".equalsIgnoreCase(keyAlgorithm)) {
            if (minSdkVersion < 18) {
                throw new InvalidKeyException(
                        "ECDSA signatures only supported for minSdkVersion 18 and higher");
            }
            return DigestAlgorithm.SHA256;
        } else {
            throw new InvalidKeyException("Unsupported key algorithm: " + keyAlgorithm);
        }
    }

    /**
     * Returns a safe version of the provided signer name.
     */
    public static String getSafeSignerName(String name) {
        if (name.isEmpty()) {
            throw new IllegalArgumentException("Empty name");
        }

        // According to https://docs.oracle.com/javase/tutorial/deployment/jar/signing.html, the
        // name must not be longer than 8 characters and may contain only A-Z, 0-9, _, and -.
        StringBuilder result = new StringBuilder();
        char[] nameCharsUpperCase = name.toUpperCase(Locale.US).toCharArray();
        for (int i = 0; i < Math.min(nameCharsUpperCase.length, 8); i++) {
            char c = nameCharsUpperCase[i];
            if (((c >= 'A') && (c <= 'Z'))
                    || ((c >= '0') && (c <= '9'))
                    || (c == '-')
                    || (c == '_')) {
                result.append(c);
            } else {
                result.append('_');
            }
        }
        return result.toString();
    }

    /**
     * Returns a new {@link MessageDigest} instance corresponding to the provided digest algorithm.
     */
    private static MessageDigest getMessageDigestInstance(DigestAlgorithm digestAlgorithm)
            throws NoSuchAlgorithmException {
        String jcaAlgorithm = digestAlgorithm.getJcaMessageDigestAlgorithm();
        return MessageDigest.getInstance(jcaAlgorithm);
    }

    /**
     * Returns the JCA {@link MessageDigest} algorithm corresponding to the provided digest
     * algorithm.
     */
    public static String getJcaMessageDigestAlgorithm(DigestAlgorithm digestAlgorithm) {
        return digestAlgorithm.getJcaMessageDigestAlgorithm();
    }

    /**
     * Returns {@code true} if the provided JAR entry must be mentioned in signed JAR archive's
     * manifest.
     */
    public static boolean isJarEntryDigestNeededInManifest(String entryName) {
        // See https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File

        // Entries which represent directories sould not be listed in the manifest.
        if (entryName.endsWith("/")) {
            return false;
        }

        // Entries outside of META-INF must be listed in the manifest.
        if (!entryName.startsWith("META-INF/")) {
            return true;
        }
        // Entries in subdirectories of META-INF must be listed in the manifest.
        if (entryName.indexOf('/', "META-INF/".length()) != -1) {
            return true;
        }

        // Ignored file names (case-insensitive) in META-INF directory:
        //   MANIFEST.MF
        //   *.SF
        //   *.RSA
        //   *.DSA
        //   *.EC
        //   SIG-*
        String fileNameLowerCase =
                entryName.substring("META-INF/".length()).toLowerCase(Locale.US);
        if (("manifest.mf".equals(fileNameLowerCase))
                || (fileNameLowerCase.endsWith(".sf"))
                || (fileNameLowerCase.endsWith(".rsa"))
                || (fileNameLowerCase.endsWith(".dsa"))
                || (fileNameLowerCase.endsWith(".ec"))
                || (fileNameLowerCase.startsWith("sig-"))) {
            return false;
        }
        return true;
    }

    /**
     * Signs the provided APK using JAR signing (aka v1 signature scheme) and returns the list of
     * JAR entries which need to be added to the APK as part of the signature.
     *
     * @param signerConfigs signer configurations, one for each signer. At least one signer config
     *        must be provided.
     *
     * @throws ApkFormatException if the source manifest is malformed
     * @throws NoSuchAlgorithmException if a required cryptographic algorithm implementation is
     *         missing
     * @throws InvalidKeyException if a signing key is not suitable for this signature scheme or
     *         cannot be used in general
     * @throws SignatureException if an error occurs when computing digests of generating
     *         signatures
     */
    public static List<Pair<String, byte[]>> sign(
            List<SignerConfig> signerConfigs,
            DigestAlgorithm jarEntryDigestAlgorithm,
            Map<String, byte[]> jarEntryDigests,
            List<Integer> apkSigningSchemeIds,
            byte[] sourceManifestBytes,
            String createdBy)
                    throws NoSuchAlgorithmException, ApkFormatException, InvalidKeyException,
                            CertificateException, SignatureException {
        if (signerConfigs.isEmpty()) {
            throw new IllegalArgumentException("At least one signer config must be provided");
        }
        if (signerConfigs.size() > MAX_APK_SIGNERS) {
            throw new IllegalArgumentException(
                    "APK Signature Scheme v1 only supports a maximum of " + MAX_APK_SIGNERS + ", "
                            + signerConfigs.size() + " provided");
        }
        OutputManifestFile manifest =
                generateManifestFile(
                        jarEntryDigestAlgorithm, jarEntryDigests, sourceManifestBytes);

        return signManifest(
                signerConfigs, jarEntryDigestAlgorithm, apkSigningSchemeIds, createdBy, manifest);
    }

    /**
     * Signs the provided APK using JAR signing (aka v1 signature scheme) and returns the list of
     * JAR entries which need to be added to the APK as part of the signature.
     *
     * @param signerConfigs signer configurations, one for each signer. At least one signer config
     *        must be provided.
     *
     * @throws InvalidKeyException if a signing key is not suitable for this signature scheme or
     *         cannot be used in general
     * @throws SignatureException if an error occurs when computing digests of generating
     *         signatures
     */
    public static List<Pair<String, byte[]>> signManifest(
            List<SignerConfig> signerConfigs,
            DigestAlgorithm digestAlgorithm,
            List<Integer> apkSigningSchemeIds,
            String createdBy,
            OutputManifestFile manifest)
                    throws NoSuchAlgorithmException, InvalidKeyException, CertificateException,
                            SignatureException {
        if (signerConfigs.isEmpty()) {
            throw new IllegalArgumentException("At least one signer config must be provided");
        }

        // For each signer output .SF and .(RSA|DSA|EC) file, then output MANIFEST.MF.
        List<Pair<String, byte[]>> signatureJarEntries =
                new ArrayList<>(2 * signerConfigs.size() + 1);
        byte[] sfBytes =
                generateSignatureFile(apkSigningSchemeIds, digestAlgorithm, createdBy, manifest);
        for (SignerConfig signerConfig : signerConfigs) {
            String signerName = signerConfig.name;
            byte[] signatureBlock;
            try {
                signatureBlock = generateSignatureBlock(signerConfig, sfBytes);
            } catch (InvalidKeyException e) {
                throw new InvalidKeyException(
                        "Failed to sign using signer \"" + signerName + "\"", e);
            } catch (CertificateException e) {
                throw new CertificateException(
                        "Failed to sign using signer \"" + signerName + "\"", e);
            } catch (SignatureException e) {
                throw new SignatureException(
                        "Failed to sign using signer \"" + signerName + "\"", e);
            }
            signatureJarEntries.add(Pair.of("META-INF/" + signerName + ".SF", sfBytes));
            PublicKey publicKey = signerConfig.certificates.get(0).getPublicKey();
            String signatureBlockFileName =
                    "META-INF/" + signerName + "."
                            + publicKey.getAlgorithm().toUpperCase(Locale.US);
            signatureJarEntries.add(
                    Pair.of(signatureBlockFileName, signatureBlock));
        }
        signatureJarEntries.add(Pair.of(V1SchemeConstants.MANIFEST_ENTRY_NAME, manifest.contents));
        return signatureJarEntries;
    }

    /**
     * Returns the names of JAR entries which this signer will produce as part of v1 signature.
     */
    public static Set<String> getOutputEntryNames(List<SignerConfig> signerConfigs) {
        Set<String> result = new HashSet<>(2 * signerConfigs.size() + 1);
        for (SignerConfig signerConfig : signerConfigs) {
            String signerName = signerConfig.name;
            result.add("META-INF/" + signerName + ".SF");
            PublicKey publicKey = signerConfig.certificates.get(0).getPublicKey();
            String signatureBlockFileName =
                    "META-INF/" + signerName + "."
                            + publicKey.getAlgorithm().toUpperCase(Locale.US);
            result.add(signatureBlockFileName);
        }
        result.add(V1SchemeConstants.MANIFEST_ENTRY_NAME);
        return result;
    }

    /**
     * Generated and returns the {@code META-INF/MANIFEST.MF} file based on the provided (optional)
     * input {@code MANIFEST.MF} and digests of JAR entries covered by the manifest.
     */
    public static OutputManifestFile generateManifestFile(
            DigestAlgorithm jarEntryDigestAlgorithm,
            Map<String, byte[]> jarEntryDigests,
            byte[] sourceManifestBytes) throws ApkFormatException {
        Manifest sourceManifest = null;
        if (sourceManifestBytes != null) {
            try {
                sourceManifest = new Manifest(new ByteArrayInputStream(sourceManifestBytes));
            } catch (IOException e) {
                throw new ApkFormatException("Malformed source META-INF/MANIFEST.MF", e);
            }
        }
        ByteArrayOutputStream manifestOut = new ByteArrayOutputStream();
        Attributes mainAttrs = new Attributes();
        // Copy the main section from the source manifest (if provided). Otherwise use defaults.
        // NOTE: We don't output our own Created-By header because this signer did not create the
        //       JAR/APK being signed -- the signer only adds signatures to the already existing
        //       JAR/APK.
        if (sourceManifest != null) {
            mainAttrs.putAll(sourceManifest.getMainAttributes());
        } else {
            mainAttrs.put(Attributes.Name.MANIFEST_VERSION, ATTRIBUTE_VALUE_MANIFEST_VERSION);
        }

        try {
            ManifestWriter.writeMainSection(manifestOut, mainAttrs);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write in-memory MANIFEST.MF", e);
        }

        List<String> sortedEntryNames = new ArrayList<>(jarEntryDigests.keySet());
        Collections.sort(sortedEntryNames);
        SortedMap<String, byte[]> invidualSectionsContents = new TreeMap<>();
        String entryDigestAttributeName = getEntryDigestAttributeName(jarEntryDigestAlgorithm);
        for (String entryName : sortedEntryNames) {
            checkEntryNameValid(entryName);
            byte[] entryDigest = jarEntryDigests.get(entryName);
            Attributes entryAttrs = new Attributes();
            entryAttrs.putValue(
                    entryDigestAttributeName,
                    Base64.getEncoder().encodeToString(entryDigest));
            ByteArrayOutputStream sectionOut = new ByteArrayOutputStream();
            byte[] sectionBytes;
            try {
                ManifestWriter.writeIndividualSection(sectionOut, entryName, entryAttrs);
                sectionBytes = sectionOut.toByteArray();
                manifestOut.write(sectionBytes);
            } catch (IOException e) {
                throw new RuntimeException("Failed to write in-memory MANIFEST.MF", e);
            }
            invidualSectionsContents.put(entryName, sectionBytes);
        }

        OutputManifestFile result = new OutputManifestFile();
        result.contents = manifestOut.toByteArray();
        result.mainSectionAttributes = mainAttrs;
        result.individualSectionsContents = invidualSectionsContents;
        return result;
    }

    private static void checkEntryNameValid(String name) throws ApkFormatException {
        // JAR signing spec says CR, LF, and NUL are not permitted in entry names
        // CR or LF in entry names will result in malformed MANIFEST.MF and .SF files because there
        // is no way to escape characters in MANIFEST.MF and .SF files. NUL can, presumably, cause
        // issues when parsing using C and C++ like languages.
        for (char c : name.toCharArray()) {
            if ((c == '\r') || (c == '\n') || (c == 0)) {
                throw new ApkFormatException(
                        String.format(
                                "Unsupported character 0x%1$02x in ZIP entry name \"%2$s\"",
                                (int) c,
                                name));
            }
        }
    }

    public static class OutputManifestFile {
        public byte[] contents;
        public SortedMap<String, byte[]> individualSectionsContents;
        public Attributes mainSectionAttributes;
    }

    private static byte[] generateSignatureFile(
            List<Integer> apkSignatureSchemeIds,
            DigestAlgorithm manifestDigestAlgorithm,
            String createdBy,
            OutputManifestFile manifest) throws NoSuchAlgorithmException {
        Manifest sf = new Manifest();
        Attributes mainAttrs = sf.getMainAttributes();
        mainAttrs.put(Attributes.Name.SIGNATURE_VERSION, ATTRIBUTE_VALUE_SIGNATURE_VERSION);
        mainAttrs.put(ATTRIBUTE_NAME_CREATED_BY, createdBy);
        if (!apkSignatureSchemeIds.isEmpty()) {
            // Add APK Signature Scheme v2 (and newer) signature stripping protection.
            // This attribute indicates that this APK is supposed to have been signed using one or
            // more APK-specific signature schemes in addition to the standard JAR signature scheme
            // used by this code. APK signature verifier should reject the APK if it does not
            // contain a signature for the signature scheme the verifier prefers out of this set.
            StringBuilder attrValue = new StringBuilder();
            for (int id : apkSignatureSchemeIds) {
                if (attrValue.length() > 0) {
                    attrValue.append(", ");
                }
                attrValue.append(String.valueOf(id));
            }
            mainAttrs.put(
                    SF_ATTRIBUTE_NAME_ANDROID_APK_SIGNED_NAME,
                    attrValue.toString());
        }

        // Add main attribute containing the digest of MANIFEST.MF.
        MessageDigest md = getMessageDigestInstance(manifestDigestAlgorithm);
        mainAttrs.putValue(
                getManifestDigestAttributeName(manifestDigestAlgorithm),
                Base64.getEncoder().encodeToString(md.digest(manifest.contents)));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            SignatureFileWriter.writeMainSection(out, mainAttrs);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write in-memory .SF file", e);
        }
        String entryDigestAttributeName = getEntryDigestAttributeName(manifestDigestAlgorithm);
        for (Map.Entry<String, byte[]> manifestSection
                : manifest.individualSectionsContents.entrySet()) {
            String sectionName = manifestSection.getKey();
            byte[] sectionContents = manifestSection.getValue();
            byte[] sectionDigest = md.digest(sectionContents);
            Attributes attrs = new Attributes();
            attrs.putValue(
                    entryDigestAttributeName,
                    Base64.getEncoder().encodeToString(sectionDigest));

            try {
                SignatureFileWriter.writeIndividualSection(out, sectionName, attrs);
            } catch (IOException e) {
                throw new RuntimeException("Failed to write in-memory .SF file", e);
            }
        }

        // A bug in the java.util.jar implementation of Android platforms up to version 1.6 will
        // cause a spurious IOException to be thrown if the length of the signature file is a
        // multiple of 1024 bytes. As a workaround, add an extra CRLF in this case.
        if ((out.size() > 0) && ((out.size() % 1024) == 0)) {
            try {
                SignatureFileWriter.writeSectionDelimiter(out);
            } catch (IOException e) {
                throw new RuntimeException("Failed to write to ByteArrayOutputStream", e);
            }
        }

        return out.toByteArray();
    }



    /**
     * Generates the CMS PKCS #7 signature block corresponding to the provided signature file and
     * signing configuration.
     */
    private static byte[] generateSignatureBlock(
            SignerConfig signerConfig, byte[] signatureFileBytes)
                    throws NoSuchAlgorithmException, InvalidKeyException, CertificateException,
                            SignatureException {
        // Obtain relevant bits of signing configuration
        List<X509Certificate> signerCerts = signerConfig.certificates;
        X509Certificate signingCert = signerCerts.get(0);
        PublicKey publicKey = signingCert.getPublicKey();
        DigestAlgorithm digestAlgorithm = signerConfig.signatureDigestAlgorithm;
        Pair<String, AlgorithmIdentifier> signatureAlgs =
                getSignerInfoSignatureAlgorithm(publicKey, digestAlgorithm,
                        signerConfig.deterministicDsaSigning);
        String jcaSignatureAlgorithm = signatureAlgs.getFirst();

        // Generate the cryptographic signature of the signature file
        byte[] signatureBytes;
        try {
            Signature signature = Signature.getInstance(jcaSignatureAlgorithm);
            signature.initSign(signerConfig.privateKey);
            signature.update(signatureFileBytes);
            signatureBytes = signature.sign();
        } catch (InvalidKeyException e) {
            throw new InvalidKeyException("Failed to sign using " + jcaSignatureAlgorithm, e);
        } catch (SignatureException e) {
            throw new SignatureException("Failed to sign using " + jcaSignatureAlgorithm, e);
        }

        // Verify the signature against the public key in the signing certificate
        try {
            Signature signature = Signature.getInstance(jcaSignatureAlgorithm);
            signature.initVerify(publicKey);
            signature.update(signatureFileBytes);
            if (!signature.verify(signatureBytes)) {
                throw new SignatureException("Signature did not verify");
            }
        } catch (InvalidKeyException e) {
            throw new InvalidKeyException(
                    "Failed to verify generated " + jcaSignatureAlgorithm + " signature using"
                            + " public key from certificate",
                    e);
        } catch (SignatureException e) {
            throw new SignatureException(
                    "Failed to verify generated " + jcaSignatureAlgorithm + " signature using"
                            + " public key from certificate",
                    e);
        }

        AlgorithmIdentifier digestAlgorithmId =
                getSignerInfoDigestAlgorithmOid(digestAlgorithm);
        AlgorithmIdentifier signatureAlgorithmId = signatureAlgs.getSecond();
        try {
            return ApkSigningBlockUtils.generatePkcs7DerEncodedMessage(
                    signatureBytes,
                    null,
                    signerCerts, digestAlgorithmId,
                    signatureAlgorithmId);
        } catch (Asn1EncodingException | CertificateEncodingException ex) {
            throw new SignatureException("Failed to encode signature block");
        }
    }



    private static String getEntryDigestAttributeName(DigestAlgorithm digestAlgorithm) {
        switch (digestAlgorithm) {
            case SHA1:
                return "SHA1-Digest";
            case SHA256:
                return "SHA-256-Digest";
            default:
                throw new IllegalArgumentException(
                        "Unexpected content digest algorithm: " + digestAlgorithm);
        }
    }

    private static String getManifestDigestAttributeName(DigestAlgorithm digestAlgorithm) {
        switch (digestAlgorithm) {
            case SHA1:
                return "SHA1-Digest-Manifest";
            case SHA256:
                return "SHA-256-Digest-Manifest";
            default:
                throw new IllegalArgumentException(
                        "Unexpected content digest algorithm: " + digestAlgorithm);
        }
    }
}
