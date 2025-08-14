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
import static com.android.apksig.internal.oid.OidConstants.getSigAlgSupportedApiLevels;
import static com.android.apksig.internal.pkcs7.AlgorithmIdentifier.getJcaDigestAlgorithm;
import static com.android.apksig.internal.pkcs7.AlgorithmIdentifier.getJcaSignatureAlgorithm;
import static com.android.apksig.internal.x509.Certificate.findCertificate;
import static com.android.apksig.internal.x509.Certificate.parseCertificates;

import com.android.apksig.ApkVerifier.Issue;
import com.android.apksig.ApkVerifier.IssueWithParams;
import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.apk.ApkSigningBlockUtils;
import com.android.apksig.internal.asn1.Asn1BerParser;
import com.android.apksig.internal.asn1.Asn1Class;
import com.android.apksig.internal.asn1.Asn1DecodingException;
import com.android.apksig.internal.asn1.Asn1Field;
import com.android.apksig.internal.asn1.Asn1OpaqueObject;
import com.android.apksig.internal.asn1.Asn1Type;
import com.android.apksig.internal.jar.ManifestParser;
import com.android.apksig.internal.oid.OidConstants;
import com.android.apksig.internal.pkcs7.Attribute;
import com.android.apksig.internal.pkcs7.ContentInfo;
import com.android.apksig.internal.pkcs7.Pkcs7Constants;
import com.android.apksig.internal.pkcs7.Pkcs7DecodingException;
import com.android.apksig.internal.pkcs7.SignedData;
import com.android.apksig.internal.pkcs7.SignerInfo;
import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.util.ByteBufferUtils;
import com.android.apksig.internal.util.InclusiveIntRange;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.internal.zip.CentralDirectoryRecord;
import com.android.apksig.internal.zip.LocalFileRecord;
import com.android.apksig.internal.zip.ZipUtils;
import com.android.apksig.util.DataSinks;
import com.android.apksig.util.DataSource;
import com.android.apksig.zip.ZipFormatException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.Principal;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Base64.Decoder;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.jar.Attributes;

/**
 * APK verifier which uses JAR signing (aka v1 signing scheme).
 *
 * @see <a href="https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File">Signed JAR File</a>
 */
public abstract class V1SchemeVerifier {
    private V1SchemeVerifier() {}

    /**
     * Verifies the provided APK's JAR signatures and returns the result of verification. APK is
     * considered verified only if {@link Result#verified} is {@code true}. If verification fails,
     * the result will contain errors -- see {@link Result#getErrors()}.
     *
     * <p>Verification succeeds iff the APK's JAR signatures are expected to verify on all Android
     * platform versions in the {@code [minSdkVersion, maxSdkVersion]} range. If the APK's signature
     * is expected to not verify on any of the specified platform versions, this method returns a
     * result with one or more errors and whose {@code Result.verified == false}, or this method
     * throws an exception.
     *
     * @throws ApkFormatException if the APK is malformed
     * @throws IOException if an I/O error occurs when reading the APK
     * @throws NoSuchAlgorithmException if the APK's JAR signatures cannot be verified because a
     *         required cryptographic algorithm implementation is missing
     */
    public static Result verify(
            DataSource apk,
            ApkUtils.ZipSections apkSections,
            Map<Integer, String> supportedApkSigSchemeNames,
            Set<Integer> foundApkSigSchemeIds,
            int minSdkVersion,
            int maxSdkVersion) throws IOException, ApkFormatException, NoSuchAlgorithmException {
        if (minSdkVersion > maxSdkVersion) {
            throw new IllegalArgumentException(
                    "minSdkVersion (" + minSdkVersion + ") > maxSdkVersion (" + maxSdkVersion
                            + ")");
        }

        Result result = new Result();

        // Parse the ZIP Central Directory and check that there are no entries with duplicate names.
        List<CentralDirectoryRecord> cdRecords = parseZipCentralDirectory(apk, apkSections);
        Set<String> cdEntryNames = checkForDuplicateEntries(cdRecords, result);
        if (result.containsErrors()) {
            return result;
        }

        // Verify JAR signature(s).
        Signers.verify(
                apk,
                apkSections.getZipCentralDirectoryOffset(),
                cdRecords,
                cdEntryNames,
                supportedApkSigSchemeNames,
                foundApkSigSchemeIds,
                minSdkVersion,
                maxSdkVersion,
                result);

        return result;
    }

    /**
     * Returns the set of entry names and reports any duplicate entry names in the {@code result}
     * as errors.
     */
    private static Set<String> checkForDuplicateEntries(
            List<CentralDirectoryRecord> cdRecords, Result result) {
        Set<String> cdEntryNames = new HashSet<>(cdRecords.size());
        Set<String> duplicateCdEntryNames = null;
        for (CentralDirectoryRecord cdRecord : cdRecords) {
            String entryName = cdRecord.getName();
            if (!cdEntryNames.add(entryName)) {
                // This is an error. Report this once per duplicate name.
                if (duplicateCdEntryNames == null) {
                    duplicateCdEntryNames = new HashSet<>();
                }
                if (duplicateCdEntryNames.add(entryName)) {
                    result.addError(Issue.JAR_SIG_DUPLICATE_ZIP_ENTRY, entryName);
                }
            }
        }
        return cdEntryNames;
    }

    /**
    * Parses raw representation of MANIFEST.MF file into a pair of main entry manifest section
    * representation and a mapping between entry name and its manifest section representation.
    *
    * @param manifestBytes raw representation of Manifest.MF
    * @param cdEntryNames expected set of entry names
    * @param result object to keep track of errors that happened during the parsing
    * @return a pair of main entry manifest section representation and a mapping between entry name
    *     and its manifest section representation
    */
    public static Pair<ManifestParser.Section, Map<String, ManifestParser.Section>> parseManifest(
            byte[] manifestBytes, Set<String> cdEntryNames, Result result) {
        ManifestParser manifest = new ManifestParser(manifestBytes);
        ManifestParser.Section manifestMainSection = manifest.readSection();
        List<ManifestParser.Section> manifestIndividualSections = manifest.readAllSections();
        Map<String, ManifestParser.Section> entryNameToManifestSection =
                new HashMap<>(manifestIndividualSections.size());
        int manifestSectionNumber = 0;
        for (ManifestParser.Section manifestSection : manifestIndividualSections) {
            manifestSectionNumber++;
            String entryName = manifestSection.getName();
            if (entryName == null) {
                result.addError(Issue.JAR_SIG_UNNNAMED_MANIFEST_SECTION, manifestSectionNumber);
                continue;
            }
            if (entryNameToManifestSection.put(entryName, manifestSection) != null) {
                result.addError(Issue.JAR_SIG_DUPLICATE_MANIFEST_SECTION, entryName);
                continue;
            }
            if (!cdEntryNames.contains(entryName)) {
                result.addError(
                        Issue.JAR_SIG_MISSING_ZIP_ENTRY_REFERENCED_IN_MANIFEST, entryName);
                continue;
            }
        }
        return Pair.of(manifestMainSection, entryNameToManifestSection);
    }

    /**
     * All JAR signers of an APK.
     */
    private static class Signers {

        /**
         * Verifies JAR signatures of the provided APK and populates the provided result container
         * with errors, warnings, and information about signers. The APK is considered verified if
         * the {@link Result#verified} is {@code true}.
         */
        private static void verify(
                DataSource apk,
                long cdStartOffset,
                List<CentralDirectoryRecord> cdRecords,
                Set<String> cdEntryNames,
                Map<Integer, String> supportedApkSigSchemeNames,
                Set<Integer> foundApkSigSchemeIds,
                int minSdkVersion,
                int maxSdkVersion,
                Result result) throws ApkFormatException, IOException, NoSuchAlgorithmException {

            // Find JAR manifest and signature block files.
            CentralDirectoryRecord manifestEntry = null;
            Map<String, CentralDirectoryRecord> sigFileEntries = new HashMap<>(1);
            List<CentralDirectoryRecord> sigBlockEntries = new ArrayList<>(1);
            for (CentralDirectoryRecord cdRecord : cdRecords) {
                String entryName = cdRecord.getName();
                if (!entryName.startsWith("META-INF/")) {
                    continue;
                }
                if ((manifestEntry == null) && (V1SchemeConstants.MANIFEST_ENTRY_NAME.equals(
                        entryName))) {
                    manifestEntry = cdRecord;
                    continue;
                }
                if (entryName.endsWith(".SF")) {
                    sigFileEntries.put(entryName, cdRecord);
                    continue;
                }
                if ((entryName.endsWith(".RSA"))
                        || (entryName.endsWith(".DSA"))
                        || (entryName.endsWith(".EC"))) {
                    sigBlockEntries.add(cdRecord);
                    continue;
                }
            }
            if (manifestEntry == null) {
                result.addError(Issue.JAR_SIG_NO_MANIFEST);
                return;
            }

            // Parse the JAR manifest and check that all JAR entries it references exist in the APK.
            byte[] manifestBytes;
            try {
                manifestBytes =
                        LocalFileRecord.getUncompressedData(apk, manifestEntry, cdStartOffset);
            } catch (ZipFormatException e) {
                throw new ApkFormatException("Malformed ZIP entry: " + manifestEntry.getName(), e);
            }

            Pair<ManifestParser.Section, Map<String, ManifestParser.Section>> manifestSections =
                    parseManifest(manifestBytes, cdEntryNames, result);

            if (result.containsErrors()) {
                return;
            }

            ManifestParser.Section manifestMainSection = manifestSections.getFirst();
            Map<String, ManifestParser.Section> entryNameToManifestSection =
                    manifestSections.getSecond();

            // STATE OF AFFAIRS:
            // * All JAR entries listed in JAR manifest are present in the APK.

            // Identify signers
            List<Signer> signers = new ArrayList<>(sigBlockEntries.size());
            for (CentralDirectoryRecord sigBlockEntry : sigBlockEntries) {
                String sigBlockEntryName = sigBlockEntry.getName();
                int extensionDelimiterIndex = sigBlockEntryName.lastIndexOf('.');
                if (extensionDelimiterIndex == -1) {
                    throw new RuntimeException(
                            "Signature block file name does not contain extension: "
                                    + sigBlockEntryName);
                }
                String sigFileEntryName =
                        sigBlockEntryName.substring(0, extensionDelimiterIndex) + ".SF";
                CentralDirectoryRecord sigFileEntry = sigFileEntries.get(sigFileEntryName);
                if (sigFileEntry == null) {
                    result.addWarning(
                            Issue.JAR_SIG_MISSING_FILE, sigBlockEntryName, sigFileEntryName);
                    continue;
                }
                String signerName = sigBlockEntryName.substring("META-INF/".length());
                Result.SignerInfo signerInfo =
                        new Result.SignerInfo(
                                signerName, sigBlockEntryName, sigFileEntry.getName());
                Signer signer = new Signer(signerName, sigBlockEntry, sigFileEntry, signerInfo);
                signers.add(signer);
            }
            if (signers.isEmpty()) {
                result.addError(Issue.JAR_SIG_NO_SIGNATURES);
                return;
            }
            if (signers.size() > MAX_APK_SIGNERS) {
                result.addError(Issue.JAR_SIG_MAX_SIGNATURES_EXCEEDED, MAX_APK_SIGNERS,
                        signers.size());
                return;
            }

            // Verify each signer's signature block file .(RSA|DSA|EC) against the corresponding
            // signature file .SF. Any error encountered for any signer terminates verification, to
            // mimic Android's behavior.
            for (Signer signer : signers) {
                signer.verifySigBlockAgainstSigFile(
                        apk, cdStartOffset, minSdkVersion, maxSdkVersion);
                if (signer.getResult().containsErrors()) {
                    result.signers.add(signer.getResult());
                }
            }
            if (result.containsErrors()) {
                return;
            }
            // STATE OF AFFAIRS:
            // * All JAR entries listed in JAR manifest are present in the APK.
            // * All signature files (.SF) verify against corresponding block files (.RSA|.DSA|.EC).

            // Verify each signer's signature file (.SF) against the JAR manifest.
            List<Signer> remainingSigners = new ArrayList<>(signers.size());
            for (Signer signer : signers) {
                signer.verifySigFileAgainstManifest(
                        manifestBytes,
                        manifestMainSection,
                        entryNameToManifestSection,
                        supportedApkSigSchemeNames,
                        foundApkSigSchemeIds,
                        minSdkVersion,
                        maxSdkVersion);
                if (signer.isIgnored()) {
                    result.ignoredSigners.add(signer.getResult());
                } else {
                    if (signer.getResult().containsErrors()) {
                        result.signers.add(signer.getResult());
                    } else {
                        remainingSigners.add(signer);
                    }
                }
            }
            if (result.containsErrors()) {
                return;
            }
            signers = remainingSigners;
            if (signers.isEmpty()) {
                result.addError(Issue.JAR_SIG_NO_SIGNATURES);
                return;
            }
            // STATE OF AFFAIRS:
            // * All signature files (.SF) verify against corresponding block files (.RSA|.DSA|.EC).
            // * Contents of all JAR manifest sections listed in .SF files verify against .SF files.
            // * All JAR entries listed in JAR manifest are present in the APK.

            // Verify data of JAR entries against JAR manifest and .SF files. On Android, an APK's
            // JAR entry is considered signed by signers associated with an .SF file iff the entry
            // is mentioned in the .SF file and the entry's digest(s) mentioned in the JAR manifest
            // match theentry's uncompressed data. Android requires that all such JAR entries are
            // signed by the same set of signers. This set may be smaller than the set of signers
            // we've identified so far.
            Set<Signer> apkSigners =
                    verifyJarEntriesAgainstManifestAndSigners(
                            apk,
                            cdStartOffset,
                            cdRecords,
                            entryNameToManifestSection,
                            signers,
                            minSdkVersion,
                            maxSdkVersion,
                            result);
            if (result.containsErrors()) {
                return;
            }
            // STATE OF AFFAIRS:
            // * All signature files (.SF) verify against corresponding block files (.RSA|.DSA|.EC).
            // * Contents of all JAR manifest sections listed in .SF files verify against .SF files.
            // * All JAR entries listed in JAR manifest are present in the APK.
            // * All JAR entries present in the APK and supposed to be covered by JAR signature
            //   (i.e., reside outside of META-INF/) are covered by signatures from the same set
            //   of signers.

            // Report any JAR entries which aren't covered by signature.
            Set<String> signatureEntryNames = new HashSet<>(1 + result.signers.size() * 2);
            signatureEntryNames.add(manifestEntry.getName());
            for (Signer signer : apkSigners) {
                signatureEntryNames.add(signer.getSignatureBlockEntryName());
                signatureEntryNames.add(signer.getSignatureFileEntryName());
            }
            for (CentralDirectoryRecord cdRecord : cdRecords) {
                String entryName = cdRecord.getName();
                if ((entryName.startsWith("META-INF/"))
                        && (!entryName.endsWith("/"))
                        && (!signatureEntryNames.contains(entryName))) {
                    result.addWarning(Issue.JAR_SIG_UNPROTECTED_ZIP_ENTRY, entryName);
                }
            }

            // Reflect the sets of used signers and ignored signers in the result.
            for (Signer signer : signers) {
                if (apkSigners.contains(signer)) {
                    result.signers.add(signer.getResult());
                } else {
                    result.ignoredSigners.add(signer.getResult());
                }
            }

            result.verified = true;
        }
    }

    static class Signer {
        private final String mName;
        private final Result.SignerInfo mResult;
        private final CentralDirectoryRecord mSignatureFileEntry;
        private final CentralDirectoryRecord mSignatureBlockEntry;
        private boolean mIgnored;

        private byte[] mSigFileBytes;
        private Set<String> mSigFileEntryNames;

        private Signer(
                String name,
                CentralDirectoryRecord sigBlockEntry,
                CentralDirectoryRecord sigFileEntry,
                Result.SignerInfo result) {
            mName = name;
            mResult = result;
            mSignatureBlockEntry = sigBlockEntry;
            mSignatureFileEntry = sigFileEntry;
        }

        public String getName() {
            return mName;
        }

        public String getSignatureFileEntryName() {
            return mSignatureFileEntry.getName();
        }

        public String getSignatureBlockEntryName() {
            return mSignatureBlockEntry.getName();
        }

        void setIgnored() {
            mIgnored = true;
        }

        public boolean isIgnored() {
            return mIgnored;
        }

        public Set<String> getSigFileEntryNames() {
            return mSigFileEntryNames;
        }

        public Result.SignerInfo getResult() {
            return mResult;
        }

        public void verifySigBlockAgainstSigFile(
                DataSource apk, long cdStartOffset, int minSdkVersion, int maxSdkVersion)
                        throws IOException, ApkFormatException, NoSuchAlgorithmException {
            // Obtain the signature block from the APK
            byte[] sigBlockBytes;
            try {
                sigBlockBytes =
                        LocalFileRecord.getUncompressedData(
                                apk, mSignatureBlockEntry, cdStartOffset);
            } catch (ZipFormatException e) {
                throw new ApkFormatException(
                        "Malformed ZIP entry: " + mSignatureBlockEntry.getName(), e);
            }
            // Obtain the signature file from the APK
            try {
                mSigFileBytes =
                        LocalFileRecord.getUncompressedData(
                                apk, mSignatureFileEntry, cdStartOffset);
            } catch (ZipFormatException e) {
                throw new ApkFormatException(
                        "Malformed ZIP entry: " + mSignatureFileEntry.getName(), e);
            }

            // Extract PKCS #7 SignedData from the signature block
            SignedData signedData;
            try {
                ContentInfo contentInfo =
                        Asn1BerParser.parse(ByteBuffer.wrap(sigBlockBytes), ContentInfo.class);
                if (!Pkcs7Constants.OID_SIGNED_DATA.equals(contentInfo.contentType)) {
                    throw new Asn1DecodingException(
                          "Unsupported ContentInfo.contentType: " + contentInfo.contentType);
                }
                signedData =
                        Asn1BerParser.parse(contentInfo.content.getEncoded(), SignedData.class);
            } catch (Asn1DecodingException e) {
                e.printStackTrace();
                mResult.addError(
                        Issue.JAR_SIG_PARSE_EXCEPTION, mSignatureBlockEntry.getName(), e);
                return;
            }

            if (signedData.signerInfos.isEmpty()) {
                mResult.addError(Issue.JAR_SIG_NO_SIGNERS, mSignatureBlockEntry.getName());
                return;
            }

            // Find the first SignedData.SignerInfos element which verifies against the signature
            // file
            SignerInfo firstVerifiedSignerInfo = null;
            X509Certificate firstVerifiedSignerInfoSigningCertificate = null;
            // Prior to Android N, Android attempts to verify only the first SignerInfo. From N
            // onwards, Android attempts to verify all SignerInfos and then picks the first verified
            // SignerInfo.
            List<SignerInfo> unverifiedSignerInfosToTry;
            if (minSdkVersion < AndroidSdkVersion.N) {
                unverifiedSignerInfosToTry =
                        Collections.singletonList(signedData.signerInfos.get(0));
            } else {
                unverifiedSignerInfosToTry = signedData.signerInfos;
            }
            List<X509Certificate> signedDataCertificates = null;
            for (SignerInfo unverifiedSignerInfo : unverifiedSignerInfosToTry) {
                // Parse SignedData.certificates -- they are needed to verify SignerInfo
                if (signedDataCertificates == null) {
                    try {
                        signedDataCertificates = parseCertificates(signedData.certificates);
                    } catch (CertificateException e) {
                        mResult.addError(
                                Issue.JAR_SIG_PARSE_EXCEPTION, mSignatureBlockEntry.getName(), e);
                        return;
                    }
                }

                // Verify SignerInfo
                X509Certificate signingCertificate;
                try {
                    signingCertificate =
                            verifySignerInfoAgainstSigFile(
                                    signedData,
                                    signedDataCertificates,
                                    unverifiedSignerInfo,
                                    mSigFileBytes,
                                    minSdkVersion,
                                    maxSdkVersion);
                    if (mResult.containsErrors()) {
                        return;
                    }
                    if (signingCertificate != null) {
                        // SignerInfo verified
                        if (firstVerifiedSignerInfo == null) {
                            firstVerifiedSignerInfo = unverifiedSignerInfo;
                            firstVerifiedSignerInfoSigningCertificate = signingCertificate;
                        }
                    }
                } catch (Pkcs7DecodingException e) {
                    mResult.addError(
                            Issue.JAR_SIG_PARSE_EXCEPTION, mSignatureBlockEntry.getName(), e);
                    return;
                } catch (InvalidKeyException | SignatureException e) {
                    mResult.addError(
                            Issue.JAR_SIG_VERIFY_EXCEPTION,
                            mSignatureBlockEntry.getName(),
                            mSignatureFileEntry.getName(),
                            e);
                    return;
                }
            }
            if (firstVerifiedSignerInfo == null) {
                // No SignerInfo verified
                mResult.addError(
                        Issue.JAR_SIG_DID_NOT_VERIFY,
                        mSignatureBlockEntry.getName(),
                        mSignatureFileEntry.getName());
                return;
            }
            // Verified
            List<X509Certificate> signingCertChain =
                    getCertificateChain(
                            signedDataCertificates, firstVerifiedSignerInfoSigningCertificate);
            mResult.certChain.clear();
            mResult.certChain.addAll(signingCertChain);
        }

        /**
         * Returns the signing certificate if the provided {@link SignerInfo} verifies against the
         * contents of the provided signature file, or {@code null} if it does not verify.
         */
        private X509Certificate verifySignerInfoAgainstSigFile(
                SignedData signedData,
                Collection<X509Certificate> signedDataCertificates,
                SignerInfo signerInfo,
                byte[] signatureFile,
                int minSdkVersion,
                int maxSdkVersion)
                        throws Pkcs7DecodingException, NoSuchAlgorithmException,
                                InvalidKeyException, SignatureException {
            String digestAlgorithmOid = signerInfo.digestAlgorithm.algorithm;
            String signatureAlgorithmOid = signerInfo.signatureAlgorithm.algorithm;
            InclusiveIntRange desiredApiLevels =
                    InclusiveIntRange.fromTo(minSdkVersion, maxSdkVersion);
            List<InclusiveIntRange> apiLevelsWhereDigestAndSigAlgorithmSupported =
                    getSigAlgSupportedApiLevels(digestAlgorithmOid, signatureAlgorithmOid);
            List<InclusiveIntRange> apiLevelsWhereDigestAlgorithmNotSupported =
                    desiredApiLevels.getValuesNotIn(apiLevelsWhereDigestAndSigAlgorithmSupported);
            if (!apiLevelsWhereDigestAlgorithmNotSupported.isEmpty()) {
                String digestAlgorithmUserFriendly =
                        OidConstants.OidToUserFriendlyNameMapper.getUserFriendlyNameForOid(
                                digestAlgorithmOid);
                if (digestAlgorithmUserFriendly == null) {
                    digestAlgorithmUserFriendly = digestAlgorithmOid;
                }
                String signatureAlgorithmUserFriendly =
                        OidConstants.OidToUserFriendlyNameMapper.getUserFriendlyNameForOid(
                                signatureAlgorithmOid);
                if (signatureAlgorithmUserFriendly == null) {
                    signatureAlgorithmUserFriendly = signatureAlgorithmOid;
                }
                StringBuilder apiLevelsUserFriendly = new StringBuilder();
                for (InclusiveIntRange range : apiLevelsWhereDigestAlgorithmNotSupported) {
                    if (apiLevelsUserFriendly.length() > 0) {
                        apiLevelsUserFriendly.append(", ");
                    }
                    if (range.getMin() == range.getMax()) {
                        apiLevelsUserFriendly.append(String.valueOf(range.getMin()));
                    } else if (range.getMax() == Integer.MAX_VALUE) {
                        apiLevelsUserFriendly.append(range.getMin() + "+");
                    } else {
                        apiLevelsUserFriendly.append(range.getMin() + "-" + range.getMax());
                    }
                }
                mResult.addError(
                        Issue.JAR_SIG_UNSUPPORTED_SIG_ALG,
                        mSignatureBlockEntry.getName(),
                        digestAlgorithmOid,
                        signatureAlgorithmOid,
                        apiLevelsUserFriendly.toString(),
                        digestAlgorithmUserFriendly,
                        signatureAlgorithmUserFriendly);
                return null;
            }

            // From the bag of certs, obtain the certificate referenced by the SignerInfo,
            // and verify the cryptographic signature in the SignerInfo against the certificate.

            // Locate the signing certificate referenced by the SignerInfo
            X509Certificate signingCertificate =
                    findCertificate(signedDataCertificates, signerInfo.sid);
            if (signingCertificate == null) {
                throw new SignatureException(
                        "Signing certificate referenced in SignerInfo not found in"
                                + " SignedData");
            }

            // Check whether the signing certificate is acceptable. Android performs these
            // checks explicitly, instead of delegating this to
            // Signature.initVerify(Certificate).
            if (signingCertificate.hasUnsupportedCriticalExtension()) {
                throw new SignatureException(
                        "Signing certificate has unsupported critical extensions");
            }
            boolean[] keyUsageExtension = signingCertificate.getKeyUsage();
            if (keyUsageExtension != null) {
                boolean digitalSignature =
                        (keyUsageExtension.length >= 1) && (keyUsageExtension[0]);
                boolean nonRepudiation =
                        (keyUsageExtension.length >= 2) && (keyUsageExtension[1]);
                if ((!digitalSignature) && (!nonRepudiation)) {
                    throw new SignatureException(
                            "Signing certificate not authorized for use in digital signatures"
                                    + ": keyUsage extension missing digitalSignature and"
                                    + " nonRepudiation");
                }
            }

            // Verify the cryptographic signature in SignerInfo against the certificate's
            // public key
            String jcaSignatureAlgorithm =
                    getJcaSignatureAlgorithm(digestAlgorithmOid, signatureAlgorithmOid);
            Signature s = Signature.getInstance(jcaSignatureAlgorithm);
            PublicKey publicKey = signingCertificate.getPublicKey();
            try {
                s.initVerify(publicKey);
            } catch (InvalidKeyException e) {
                // An InvalidKeyException could be caught if the PublicKey in the certificate is not
                // properly encoded; attempt to resolve any encoding errors, generate a new public
                // key, and reattempt the initVerify with the newly encoded key.
                try {
                    byte[] encodedPublicKey = ApkSigningBlockUtils.encodePublicKey(publicKey);
                    publicKey = KeyFactory.getInstance(publicKey.getAlgorithm()).generatePublic(
                            new X509EncodedKeySpec(encodedPublicKey));
                } catch (InvalidKeySpecException ikse) {
                    // If an InvalidKeySpecException is caught then throw the original Exception
                    // since the key couldn't be properly re-encoded, and the original Exception
                    // will have more useful debugging info.
                    throw e;
                }
                s = Signature.getInstance(jcaSignatureAlgorithm);
                s.initVerify(publicKey);
            }

            if (signerInfo.signedAttrs != null) {
                // Signed attributes present -- verify signature against the ASN.1 DER encoded form
                // of signed attributes. This verifies integrity of the signature file because
                // signed attributes must contain the digest of the signature file.
                if (minSdkVersion < AndroidSdkVersion.KITKAT) {
                    // Prior to Android KitKat, APKs with signed attributes are unsafe:
                    // * The APK's contents are not protected by the JAR signature because the
                    //   digest in signed attributes is not verified. This means an attacker can
                    //   arbitrarily modify the APK without invalidating its signature.
                    // * Luckily, the signature over signed attributes was verified incorrectly
                    //   (over the verbatim IMPLICIT [0] form rather than over re-encoded
                    //   UNIVERSAL SET form) which means that JAR signatures which would verify on
                    //   pre-KitKat Android and yet do not protect the APK from modification could
                    //   be generated only by broken tools or on purpose by the entity signing the
                    //   APK.
                    //
                    // We thus reject such unsafe APKs, even if they verify on platforms before
                    // KitKat.
                    throw new SignatureException(
                            "APKs with Signed Attributes broken on platforms with API Level < "
                                    + AndroidSdkVersion.KITKAT);
                }
                try {
                    List<Attribute> signedAttributes =
                            Asn1BerParser.parseImplicitSetOf(
                                    signerInfo.signedAttrs.getEncoded(), Attribute.class);
                    SignedAttributes signedAttrs = new SignedAttributes(signedAttributes);
                    if (maxSdkVersion >= AndroidSdkVersion.N) {
                        // Content Type attribute is checked only on Android N and newer
                        String contentType =
                                signedAttrs.getSingleObjectIdentifierValue(
                                        Pkcs7Constants.OID_CONTENT_TYPE);
                        if (contentType == null) {
                            throw new SignatureException("No Content Type in signed attributes");
                        }
                        if (!contentType.equals(signedData.encapContentInfo.contentType)) {
                            // Did not verify: Content type signed attribute does not match
                            // SignedData.encapContentInfo.eContentType. This fails verification of
                            // this SignerInfo but should not prevent verification of other
                            // SignerInfos. Hence, no exception is thrown.
                            return null;
                        }
                    }
                    byte[] expectedSignatureFileDigest =
                            signedAttrs.getSingleOctetStringValue(
                                    Pkcs7Constants.OID_MESSAGE_DIGEST);
                    if (expectedSignatureFileDigest == null) {
                        throw new SignatureException("No content digest in signed attributes");
                    }
                    byte[] actualSignatureFileDigest =
                            MessageDigest.getInstance(
                                    getJcaDigestAlgorithm(digestAlgorithmOid))
                                    .digest(signatureFile);
                    if (!Arrays.equals(
                            expectedSignatureFileDigest, actualSignatureFileDigest)) {
                        // Skip verification: signature file digest in signed attributes does not
                        // match the signature file. This fails verification of
                        // this SignerInfo but should not prevent verification of other
                        // SignerInfos. Hence, no exception is thrown.
                        return null;
                    }
                } catch (Asn1DecodingException e) {
                    throw new SignatureException("Failed to parse signed attributes", e);
                }
                // PKCS #7 requires that signature is over signed attributes re-encoded as
                // ASN.1 DER. However, Android does not re-encode except for changing the
                // first byte of encoded form from IMPLICIT [0] to UNIVERSAL SET. We do the
                // same for maximum compatibility.
                ByteBuffer signedAttrsOriginalEncoding = signerInfo.signedAttrs.getEncoded();
                s.update((byte) 0x31); // UNIVERSAL SET
                signedAttrsOriginalEncoding.position(1);
                s.update(signedAttrsOriginalEncoding);
            } else {
                // No signed attributes present -- verify signature against the contents of the
                // signature file
                s.update(signatureFile);
            }
            byte[] sigBytes = ByteBufferUtils.toByteArray(signerInfo.signature.slice());
            if (!s.verify(sigBytes)) {
                // Cryptographic signature did not verify. This fails verification of this
                // SignerInfo but should not prevent verification of other SignerInfos. Hence, no
                // exception is thrown.
                return null;
            }
            // Cryptographic signature verified
            return signingCertificate;
        }



        public static List<X509Certificate> getCertificateChain(
                List<X509Certificate> certs, X509Certificate leaf) {
            List<X509Certificate> unusedCerts = new ArrayList<>(certs);
            List<X509Certificate> result = new ArrayList<>(1);
            result.add(leaf);
            unusedCerts.remove(leaf);
            X509Certificate root = leaf;
            while (!root.getSubjectDN().equals(root.getIssuerDN())) {
                Principal targetDn = root.getIssuerDN();
                boolean issuerFound = false;
                for (int i = 0; i < unusedCerts.size(); i++) {
                    X509Certificate unusedCert = unusedCerts.get(i);
                    if (targetDn.equals(unusedCert.getSubjectDN())) {
                        issuerFound = true;
                        unusedCerts.remove(i);
                        result.add(unusedCert);
                        root = unusedCert;
                        break;
                    }
                }
                if (!issuerFound) {
                    break;
                }
            }
            return result;
        }




        public void verifySigFileAgainstManifest(
                byte[] manifestBytes,
                ManifestParser.Section manifestMainSection,
                Map<String, ManifestParser.Section> entryNameToManifestSection,
                Map<Integer, String> supportedApkSigSchemeNames,
                Set<Integer> foundApkSigSchemeIds,
                int minSdkVersion,
                int maxSdkVersion) throws NoSuchAlgorithmException {
            // Inspect the main section of the .SF file.
            ManifestParser sf = new ManifestParser(mSigFileBytes);
            ManifestParser.Section sfMainSection = sf.readSection();
            if (sfMainSection.getAttributeValue(Attributes.Name.SIGNATURE_VERSION) == null) {
                mResult.addError(
                        Issue.JAR_SIG_MISSING_VERSION_ATTR_IN_SIG_FILE,
                        mSignatureFileEntry.getName());
                setIgnored();
                return;
            }

            if (maxSdkVersion >= AndroidSdkVersion.N) {
                // Android N and newer rejects APKs whose .SF file says they were supposed to be
                // signed with APK Signature Scheme v2 (or newer) and yet no such signature was
                // found.
                checkForStrippedApkSignatures(
                        sfMainSection, supportedApkSigSchemeNames, foundApkSigSchemeIds);
                if (mResult.containsErrors()) {
                    return;
                }
            }

            boolean createdBySigntool = false;
            String createdBy = sfMainSection.getAttributeValue("Created-By");
            if (createdBy != null) {
                createdBySigntool = createdBy.indexOf("signtool") != -1;
            }
            boolean manifestDigestVerified =
                    verifyManifestDigest(
                            sfMainSection,
                            createdBySigntool,
                            manifestBytes,
                            minSdkVersion,
                            maxSdkVersion);
            if (!createdBySigntool) {
                verifyManifestMainSectionDigest(
                        sfMainSection,
                        manifestMainSection,
                        manifestBytes,
                        minSdkVersion,
                        maxSdkVersion);
            }
            if (mResult.containsErrors()) {
                return;
            }

            // Inspect per-entry sections of .SF file. Technically, if the digest of JAR manifest
            // verifies, per-entry sections should be ignored. However, most Android platform
            // implementations require that such sections exist.
            List<ManifestParser.Section> sfSections = sf.readAllSections();
            Set<String> sfEntryNames = new HashSet<>(sfSections.size());
            int sfSectionNumber = 0;
            for (ManifestParser.Section sfSection : sfSections) {
                sfSectionNumber++;
                String entryName = sfSection.getName();
                if (entryName == null) {
                    mResult.addError(
                            Issue.JAR_SIG_UNNNAMED_SIG_FILE_SECTION,
                            mSignatureFileEntry.getName(),
                            sfSectionNumber);
                    setIgnored();
                    return;
                }
                if (!sfEntryNames.add(entryName)) {
                    mResult.addError(
                            Issue.JAR_SIG_DUPLICATE_SIG_FILE_SECTION,
                            mSignatureFileEntry.getName(),
                            entryName);
                    setIgnored();
                    return;
                }
                if (manifestDigestVerified) {
                    // No need to verify this entry's corresponding JAR manifest entry because the
                    // JAR manifest verifies in full.
                    continue;
                }
                // Whole-file digest of JAR manifest hasn't been verified. Thus, we need to verify
                // the digest of the JAR manifest section corresponding to this .SF section.
                ManifestParser.Section manifestSection = entryNameToManifestSection.get(entryName);
                if (manifestSection == null) {
                    mResult.addError(
                            Issue.JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_SIG_FILE,
                            entryName,
                            mSignatureFileEntry.getName());
                    setIgnored();
                    continue;
                }
                verifyManifestIndividualSectionDigest(
                        sfSection,
                        createdBySigntool,
                        manifestSection,
                        manifestBytes,
                        minSdkVersion,
                        maxSdkVersion);
            }
            mSigFileEntryNames = sfEntryNames;
        }


        /**
         * Returns {@code true} if the whole-file digest of the manifest against the main section of
         * the .SF file.
         */
        private boolean verifyManifestDigest(
                ManifestParser.Section sfMainSection,
                boolean createdBySigntool,
                byte[] manifestBytes,
                int minSdkVersion,
                int maxSdkVersion) throws NoSuchAlgorithmException {
            Collection<NamedDigest> expectedDigests =
                    getDigestsToVerify(
                            sfMainSection,
                            ((createdBySigntool) ? "-Digest" : "-Digest-Manifest"),
                            minSdkVersion,
                            maxSdkVersion);
            boolean digestFound = !expectedDigests.isEmpty();
            if (!digestFound) {
                mResult.addWarning(
                        Issue.JAR_SIG_NO_MANIFEST_DIGEST_IN_SIG_FILE,
                        mSignatureFileEntry.getName());
                return false;
            }

            boolean verified = true;
            for (NamedDigest expectedDigest : expectedDigests) {
                String jcaDigestAlgorithm = expectedDigest.jcaDigestAlgorithm;
                byte[] actual = digest(jcaDigestAlgorithm, manifestBytes);
                byte[] expected = expectedDigest.digest;
                if (!Arrays.equals(expected, actual)) {
                    mResult.addWarning(
                            Issue.JAR_SIG_ZIP_ENTRY_DIGEST_DID_NOT_VERIFY,
                            V1SchemeConstants.MANIFEST_ENTRY_NAME,
                            jcaDigestAlgorithm,
                            mSignatureFileEntry.getName(),
                            Base64.getEncoder().encodeToString(actual),
                            Base64.getEncoder().encodeToString(expected));
                    verified = false;
                }
            }
            return verified;
        }

        /**
         * Verifies the digest of the manifest's main section against the main section of the .SF
         * file.
         */
        private void verifyManifestMainSectionDigest(
                ManifestParser.Section sfMainSection,
                ManifestParser.Section manifestMainSection,
                byte[] manifestBytes,
                int minSdkVersion,
                int maxSdkVersion) throws NoSuchAlgorithmException {
            Collection<NamedDigest> expectedDigests =
                    getDigestsToVerify(
                            sfMainSection,
                            "-Digest-Manifest-Main-Attributes",
                            minSdkVersion,
                            maxSdkVersion);
            if (expectedDigests.isEmpty()) {
                return;
            }

            for (NamedDigest expectedDigest : expectedDigests) {
                String jcaDigestAlgorithm = expectedDigest.jcaDigestAlgorithm;
                byte[] actual =
                        digest(
                                jcaDigestAlgorithm,
                                manifestBytes,
                                manifestMainSection.getStartOffset(),
                                manifestMainSection.getSizeBytes());
                byte[] expected = expectedDigest.digest;
                if (!Arrays.equals(expected, actual)) {
                    mResult.addError(
                            Issue.JAR_SIG_MANIFEST_MAIN_SECTION_DIGEST_DID_NOT_VERIFY,
                            jcaDigestAlgorithm,
                            mSignatureFileEntry.getName(),
                            Base64.getEncoder().encodeToString(actual),
                            Base64.getEncoder().encodeToString(expected));
                }
            }
        }

        /**
         * Verifies the digest of the manifest's individual section against the corresponding
         * individual section of the .SF file.
         */
        private void verifyManifestIndividualSectionDigest(
                ManifestParser.Section sfIndividualSection,
                boolean createdBySigntool,
                ManifestParser.Section manifestIndividualSection,
                byte[] manifestBytes,
                int minSdkVersion,
                int maxSdkVersion) throws NoSuchAlgorithmException {
            String entryName = sfIndividualSection.getName();
            Collection<NamedDigest> expectedDigests =
                    getDigestsToVerify(
                            sfIndividualSection, "-Digest", minSdkVersion, maxSdkVersion);
            if (expectedDigests.isEmpty()) {
                mResult.addError(
                        Issue.JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_SIG_FILE,
                        entryName,
                        mSignatureFileEntry.getName());
                return;
            }

            int sectionStartIndex = manifestIndividualSection.getStartOffset();
            int sectionSizeBytes = manifestIndividualSection.getSizeBytes();
            if (createdBySigntool) {
                int sectionEndIndex = sectionStartIndex + sectionSizeBytes;
                if ((manifestBytes[sectionEndIndex - 1] == '\n')
                        && (manifestBytes[sectionEndIndex - 2] == '\n')) {
                    sectionSizeBytes--;
                }
            }
            for (NamedDigest expectedDigest : expectedDigests) {
                String jcaDigestAlgorithm = expectedDigest.jcaDigestAlgorithm;
                byte[] actual =
                        digest(
                                jcaDigestAlgorithm,
                                manifestBytes,
                                sectionStartIndex,
                                sectionSizeBytes);
                byte[] expected = expectedDigest.digest;
                if (!Arrays.equals(expected, actual)) {
                    mResult.addError(
                            Issue.JAR_SIG_MANIFEST_SECTION_DIGEST_DID_NOT_VERIFY,
                            entryName,
                            jcaDigestAlgorithm,
                            mSignatureFileEntry.getName(),
                            Base64.getEncoder().encodeToString(actual),
                            Base64.getEncoder().encodeToString(expected));
                }
            }
        }

        private void checkForStrippedApkSignatures(
                ManifestParser.Section sfMainSection,
                Map<Integer, String> supportedApkSigSchemeNames,
                Set<Integer> foundApkSigSchemeIds) {
            String signedWithApkSchemes =
                    sfMainSection.getAttributeValue(
                            V1SchemeConstants.SF_ATTRIBUTE_NAME_ANDROID_APK_SIGNED_NAME_STR);
            // This field contains a comma-separated list of APK signature scheme IDs which were
            // used to sign this APK. Android rejects APKs where an ID is known to the platform but
            // the APK didn't verify using that scheme.

            if (signedWithApkSchemes == null) {
                // APK signature (e.g., v2 scheme) stripping protections not enabled.
                if (!foundApkSigSchemeIds.isEmpty()) {
                    // APK is signed with an APK signature scheme such as v2 scheme.
                    mResult.addWarning(
                            Issue.JAR_SIG_NO_APK_SIG_STRIP_PROTECTION,
                            mSignatureFileEntry.getName());
                }
                return;
            }

            if (supportedApkSigSchemeNames.isEmpty()) {
                return;
            }

            Set<Integer> supportedApkSigSchemeIds = supportedApkSigSchemeNames.keySet();
            Set<Integer> supportedExpectedApkSigSchemeIds = new HashSet<>(1);
            StringTokenizer tokenizer = new StringTokenizer(signedWithApkSchemes, ",");
            while (tokenizer.hasMoreTokens()) {
                String idText = tokenizer.nextToken().trim();
                if (idText.isEmpty()) {
                    continue;
                }
                int id;
                try {
                    id = Integer.parseInt(idText);
                } catch (Exception ignored) {
                    continue;
                }
                // This APK was supposed to be signed with the APK signature scheme having
                // this ID.
                if (supportedApkSigSchemeIds.contains(id)) {
                    supportedExpectedApkSigSchemeIds.add(id);
                } else {
                    mResult.addWarning(
                            Issue.JAR_SIG_UNKNOWN_APK_SIG_SCHEME_ID,
                            mSignatureFileEntry.getName(),
                            id);
                }
            }

            for (int id : supportedExpectedApkSigSchemeIds) {
                if (!foundApkSigSchemeIds.contains(id)) {
                    String apkSigSchemeName = supportedApkSigSchemeNames.get(id);
                    mResult.addError(
                            Issue.JAR_SIG_MISSING_APK_SIG_REFERENCED,
                            mSignatureFileEntry.getName(),
                            id,
                            apkSigSchemeName);
                }
            }
        }
    }

    public static Collection<NamedDigest> getDigestsToVerify(
            ManifestParser.Section section,
            String digestAttrSuffix,
            int minSdkVersion,
            int maxSdkVersion) {
        Decoder base64Decoder = Base64.getDecoder();
        List<NamedDigest> result = new ArrayList<>(1);
        if (minSdkVersion < AndroidSdkVersion.JELLY_BEAN_MR2) {
            // Prior to JB MR2, Android platform's logic for picking a digest algorithm to verify is
            // to rely on the ancient Digest-Algorithms attribute which contains
            // whitespace-separated list of digest algorithms (defaulting to SHA-1) to try. The
            // first digest attribute (with supported digest algorithm) found using the list is
            // used.
            String algs = section.getAttributeValue("Digest-Algorithms");
            if (algs == null) {
                algs = "SHA SHA1";
            }
            StringTokenizer tokens = new StringTokenizer(algs);
            while (tokens.hasMoreTokens()) {
                String alg = tokens.nextToken();
                String attrName = alg + digestAttrSuffix;
                String digestBase64 = section.getAttributeValue(attrName);
                if (digestBase64 == null) {
                    // Attribute not found
                    continue;
                }
                alg = getCanonicalJcaMessageDigestAlgorithm(alg);
                if ((alg == null)
                        || (getMinSdkVersionFromWhichSupportedInManifestOrSignatureFile(alg)
                                > minSdkVersion)) {
                    // Unsupported digest algorithm
                    continue;
                }
                // Supported digest algorithm
                result.add(new NamedDigest(alg, base64Decoder.decode(digestBase64)));
                break;
            }
            // No supported digests found -- this will fail to verify on pre-JB MR2 Androids.
            if (result.isEmpty()) {
                return result;
            }
        }

        if (maxSdkVersion >= AndroidSdkVersion.JELLY_BEAN_MR2) {
            // On JB MR2 and newer, Android platform picks the strongest algorithm out of:
            // SHA-512, SHA-384, SHA-256, SHA-1.
            for (String alg : JB_MR2_AND_NEWER_DIGEST_ALGS) {
                String attrName = getJarDigestAttributeName(alg, digestAttrSuffix);
                String digestBase64 = section.getAttributeValue(attrName);
                if (digestBase64 == null) {
                    // Attribute not found
                    continue;
                }
                byte[] digest = base64Decoder.decode(digestBase64);
                byte[] digestInResult = getDigest(result, alg);
                if ((digestInResult == null) || (!Arrays.equals(digestInResult, digest))) {
                    result.add(new NamedDigest(alg, digest));
                }
                break;
            }
        }

        return result;
    }

    private static final String[] JB_MR2_AND_NEWER_DIGEST_ALGS = {
            "SHA-512",
            "SHA-384",
            "SHA-256",
            "SHA-1",
    };

    private static String getCanonicalJcaMessageDigestAlgorithm(String algorithm) {
        return UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.get(algorithm.toUpperCase(Locale.US));
    }

    public static int getMinSdkVersionFromWhichSupportedInManifestOrSignatureFile(
            String jcaAlgorithmName) {
        Integer result =
                MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.get(
                        jcaAlgorithmName.toUpperCase(Locale.US));
        return (result != null) ? result : Integer.MAX_VALUE;
    }

    private static String getJarDigestAttributeName(
            String jcaDigestAlgorithm, String attrNameSuffix) {
        if ("SHA-1".equalsIgnoreCase(jcaDigestAlgorithm)) {
            return "SHA1" + attrNameSuffix;
        } else {
            return jcaDigestAlgorithm + attrNameSuffix;
        }
    }

    private static final Map<String, String> UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL;
    static {
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL = new HashMap<>(8);
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("MD5", "MD5");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA", "SHA-1");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA1", "SHA-1");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA-1", "SHA-1");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA-256", "SHA-256");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA-384", "SHA-384");
        UPPER_CASE_JCA_DIGEST_ALG_TO_CANONICAL.put("SHA-512", "SHA-512");
    }

    private static final Map<String, Integer>
            MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST;
    static {
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST = new HashMap<>(5);
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.put("MD5", 0);
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.put("SHA-1", 0);
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.put("SHA-256", 0);
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.put(
                "SHA-384", AndroidSdkVersion.GINGERBREAD);
        MIN_SDK_VESION_FROM_WHICH_DIGEST_SUPPORTED_IN_MANIFEST.put(
                "SHA-512", AndroidSdkVersion.GINGERBREAD);
    }

    private static byte[] getDigest(Collection<NamedDigest> digests, String jcaDigestAlgorithm) {
        for (NamedDigest digest : digests) {
            if (digest.jcaDigestAlgorithm.equalsIgnoreCase(jcaDigestAlgorithm)) {
                return digest.digest;
            }
        }
        return null;
    }

    public static List<CentralDirectoryRecord> parseZipCentralDirectory(
            DataSource apk,
            ApkUtils.ZipSections apkSections)
                    throws IOException, ApkFormatException {
        return ZipUtils.parseZipCentralDirectory(apk, apkSections);
    }

    /**
     * Returns {@code true} if the provided JAR entry must be mentioned in signed JAR archive's
     * manifest for the APK to verify on Android.
     */
    private static boolean isJarEntryDigestNeededInManifest(String entryName) {
        // NOTE: This logic is different from what's required by the JAR signing scheme. This is
        // because Android's APK verification logic differs from that spec. In particular, JAR
        // signing spec includes into JAR manifest all files in subdirectories of META-INF and
        // any files inside META-INF not related to signatures.
        if (entryName.startsWith("META-INF/")) {
            return false;
        }
        return !entryName.endsWith("/");
    }

    private static Set<Signer> verifyJarEntriesAgainstManifestAndSigners(
            DataSource apk,
            long cdOffsetInApk,
            Collection<CentralDirectoryRecord> cdRecords,
            Map<String, ManifestParser.Section> entryNameToManifestSection,
            List<Signer> signers,
            int minSdkVersion,
            int maxSdkVersion,
            Result result) throws ApkFormatException, IOException, NoSuchAlgorithmException {
        // Iterate over APK contents as sequentially as possible to improve performance.
        List<CentralDirectoryRecord> cdRecordsSortedByLocalFileHeaderOffset =
                new ArrayList<>(cdRecords);
        Collections.sort(
                cdRecordsSortedByLocalFileHeaderOffset,
                CentralDirectoryRecord.BY_LOCAL_FILE_HEADER_OFFSET_COMPARATOR);
        List<Signer> firstSignedEntrySigners = null;
        String firstSignedEntryName = null;
        for (CentralDirectoryRecord cdRecord : cdRecordsSortedByLocalFileHeaderOffset) {
            String entryName = cdRecord.getName();
            if (!isJarEntryDigestNeededInManifest(entryName)) {
                continue;
            }

            ManifestParser.Section manifestSection = entryNameToManifestSection.get(entryName);
            if (manifestSection == null) {
                result.addError(Issue.JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_MANIFEST, entryName);
                continue;
            }

            List<Signer> entrySigners = new ArrayList<>(signers.size());
            for (Signer signer : signers) {
                if (signer.getSigFileEntryNames().contains(entryName)) {
                    entrySigners.add(signer);
                }
            }
            if (entrySigners.isEmpty()) {
                result.addError(Issue.JAR_SIG_ZIP_ENTRY_NOT_SIGNED, entryName);
                continue;
            }
            if (firstSignedEntrySigners == null) {
                firstSignedEntrySigners = entrySigners;
                firstSignedEntryName = entryName;
            } else if (!entrySigners.equals(firstSignedEntrySigners)) {
                result.addError(
                        Issue.JAR_SIG_ZIP_ENTRY_SIGNERS_MISMATCH,
                        firstSignedEntryName,
                        getSignerNames(firstSignedEntrySigners),
                        entryName,
                        getSignerNames(entrySigners));
                continue;
            }

            List<NamedDigest> expectedDigests =
                    new ArrayList<>(
                            getDigestsToVerify(
                                    manifestSection, "-Digest", minSdkVersion, maxSdkVersion));
            if (expectedDigests.isEmpty()) {
                result.addError(Issue.JAR_SIG_NO_ZIP_ENTRY_DIGEST_IN_MANIFEST, entryName);
                continue;
            }

            MessageDigest[] mds = new MessageDigest[expectedDigests.size()];
            for (int i = 0; i < expectedDigests.size(); i++) {
                mds[i] = getMessageDigest(expectedDigests.get(i).jcaDigestAlgorithm);
            }

            try {
                LocalFileRecord.outputUncompressedData(
                        apk,
                        cdRecord,
                        cdOffsetInApk,
                        DataSinks.asDataSink(mds));
            } catch (ZipFormatException e) {
                throw new ApkFormatException("Malformed ZIP entry: " + entryName, e);
            } catch (IOException e) {
                throw new IOException("Failed to read entry: " + entryName, e);
            }

            for (int i = 0; i < expectedDigests.size(); i++) {
                NamedDigest expectedDigest = expectedDigests.get(i);
                byte[] actualDigest = mds[i].digest();
                if (!Arrays.equals(expectedDigest.digest, actualDigest)) {
                    result.addError(
                            Issue.JAR_SIG_ZIP_ENTRY_DIGEST_DID_NOT_VERIFY,
                            entryName,
                            expectedDigest.jcaDigestAlgorithm,
                            V1SchemeConstants.MANIFEST_ENTRY_NAME,
                            Base64.getEncoder().encodeToString(actualDigest),
                            Base64.getEncoder().encodeToString(expectedDigest.digest));
                }
            }
        }

        if (firstSignedEntrySigners == null) {
            result.addError(Issue.JAR_SIG_NO_SIGNED_ZIP_ENTRIES);
            return Collections.emptySet();
        } else {
            return new HashSet<>(firstSignedEntrySigners);
        }
    }

    private static List<String> getSignerNames(List<Signer> signers) {
        if (signers.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> result = new ArrayList<>(signers.size());
        for (Signer signer : signers) {
            result.add(signer.getName());
        }
        return result;
    }

    private static MessageDigest getMessageDigest(String algorithm)
            throws NoSuchAlgorithmException {
        return MessageDigest.getInstance(algorithm);
    }

    private static byte[] digest(String algorithm, byte[] data, int offset, int length)
            throws NoSuchAlgorithmException {
        MessageDigest md = getMessageDigest(algorithm);
        md.update(data, offset, length);
        return md.digest();
    }

    private static byte[] digest(String algorithm, byte[] data) throws NoSuchAlgorithmException {
        return getMessageDigest(algorithm).digest(data);
    }

    public static class NamedDigest {
        public final String jcaDigestAlgorithm;
        public final byte[] digest;

        private NamedDigest(String jcaDigestAlgorithm, byte[] digest) {
            this.jcaDigestAlgorithm = jcaDigestAlgorithm;
            this.digest = digest;
        }
    }

    public static class Result {

        /** Whether the APK's JAR signature verifies. */
        public boolean verified;

        /** List of APK's signers. These signers are used by Android. */
        public final List<SignerInfo> signers = new ArrayList<>();

        /**
         * Signers encountered in the APK but not included in the set of the APK's signers. These
         * signers are ignored by Android.
         */
        public final List<SignerInfo> ignoredSigners = new ArrayList<>();

        private final List<IssueWithParams> mWarnings = new ArrayList<>();
        private final List<IssueWithParams> mErrors = new ArrayList<>();

        private boolean containsErrors() {
            if (!mErrors.isEmpty()) {
                return true;
            }
            for (SignerInfo signer : signers) {
                if (signer.containsErrors()) {
                    return true;
                }
            }
            return false;
        }

        private void addError(Issue msg, Object... parameters) {
            mErrors.add(new IssueWithParams(msg, parameters));
        }

        private void addWarning(Issue msg, Object... parameters) {
            mWarnings.add(new IssueWithParams(msg, parameters));
        }

        public List<IssueWithParams> getErrors() {
            return mErrors;
        }

        public List<IssueWithParams> getWarnings() {
            return mWarnings;
        }

        public static class SignerInfo {
            public final String name;
            public final String signatureFileName;
            public final String signatureBlockFileName;
            public final List<X509Certificate> certChain = new ArrayList<>();

            private final List<IssueWithParams> mWarnings = new ArrayList<>();
            private final List<IssueWithParams> mErrors = new ArrayList<>();

            private SignerInfo(
                    String name, String signatureBlockFileName, String signatureFileName) {
                this.name = name;
                this.signatureBlockFileName = signatureBlockFileName;
                this.signatureFileName = signatureFileName;
            }

            private boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            private void addError(Issue msg, Object... parameters) {
                mErrors.add(new IssueWithParams(msg, parameters));
            }

            private void addWarning(Issue msg, Object... parameters) {
                mWarnings.add(new IssueWithParams(msg, parameters));
            }

            public List<IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<IssueWithParams> getWarnings() {
                return mWarnings;
            }
        }
    }

    private static class SignedAttributes {
        private Map<String, List<Asn1OpaqueObject>> mAttrs;

        public SignedAttributes(Collection<Attribute> attrs) throws Pkcs7DecodingException {
            Map<String, List<Asn1OpaqueObject>> result = new HashMap<>(attrs.size());
            for (Attribute attr : attrs) {
                if (result.put(attr.attrType, attr.attrValues) != null) {
                    throw new Pkcs7DecodingException("Duplicate signed attribute: " + attr.attrType);
                }
            }
            mAttrs = result;
        }

        private Asn1OpaqueObject getSingleValue(String attrOid) throws Pkcs7DecodingException {
            List<Asn1OpaqueObject> values = mAttrs.get(attrOid);
            if ((values == null) || (values.isEmpty())) {
                return null;
            }
            if (values.size() > 1) {
                throw new Pkcs7DecodingException("Attribute " + attrOid + " has multiple values");
            }
            return values.get(0);
        }

        public String getSingleObjectIdentifierValue(String attrOid) throws Pkcs7DecodingException {
            Asn1OpaqueObject value = getSingleValue(attrOid);
            if (value == null) {
                return null;
            }
            try {
                return Asn1BerParser.parse(value.getEncoded(), ObjectIdentifierChoice.class).value;
            } catch (Asn1DecodingException e) {
                throw new Pkcs7DecodingException("Failed to decode OBJECT IDENTIFIER", e);
            }
        }

        public byte[] getSingleOctetStringValue(String attrOid) throws Pkcs7DecodingException {
            Asn1OpaqueObject value = getSingleValue(attrOid);
            if (value == null) {
                return null;
            }
            try {
                return Asn1BerParser.parse(value.getEncoded(), OctetStringChoice.class).value;
            } catch (Asn1DecodingException e) {
                throw new Pkcs7DecodingException("Failed to decode OBJECT IDENTIFIER", e);
            }
        }
    }

    @Asn1Class(type = Asn1Type.CHOICE)
    public static class OctetStringChoice {
        @Asn1Field(type = Asn1Type.OCTET_STRING)
        public byte[] value;
    }

    @Asn1Class(type = Asn1Type.CHOICE)
    public static class ObjectIdentifierChoice {
        @Asn1Field(type = Asn1Type.OBJECT_IDENTIFIER)
        public String value;
    }
}
