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

import static com.android.apksig.Constants.LIBRARY_PAGE_ALIGNMENT_BYTES;
import static com.android.apksig.apk.ApkUtils.SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.MIN_SDK_WITH_V31_SUPPORT;
import static com.android.apksig.internal.apk.v3.V3SchemeConstants.MIN_SDK_WITH_V3_SUPPORT;

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkSigningBlockNotFoundException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.apk.MinSdkVersionException;
import com.android.apksig.internal.apk.v3.V3SchemeConstants;
import com.android.apksig.internal.util.AndroidSdkVersion;
import com.android.apksig.internal.util.ByteBufferDataSource;
import com.android.apksig.internal.zip.CentralDirectoryRecord;
import com.android.apksig.internal.zip.EocdRecord;
import com.android.apksig.internal.zip.LocalFileRecord;
import com.android.apksig.internal.zip.ZipUtils;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSinks;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.DataSources;
import com.android.apksig.util.ReadableDataSink;
import com.android.apksig.zip.ZipFormatException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.SignatureException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * APK signer.
 *
 * <p>The signer preserves as much of the input APK as possible. For example, it preserves the order
 * of APK entries and preserves their contents, including compressed form and alignment of data.
 *
 * <p>Use {@link Builder} to obtain instances of this signer.
 *
 * @see <a href="https://source.android.com/security/apksigning/index.html">Application Signing</a>
 */
public class ApkSigner {

    /**
     * Extensible data block/field header ID used for storing information about alignment of
     * uncompressed entries as well as for aligning the entries's data. See ZIP appnote.txt section
     * 4.5 Extensible data fields.
     */
    private static final short ALIGNMENT_ZIP_EXTRA_DATA_FIELD_HEADER_ID = (short) 0xd935;

    /**
     * Minimum size (in bytes) of the extensible data block/field used for alignment of uncompressed
     * entries.
     */
    private static final short ALIGNMENT_ZIP_EXTRA_DATA_FIELD_MIN_SIZE_BYTES = 6;

    private static final short ANDROID_FILE_ALIGNMENT_BYTES = 4096;

    /** Name of the Android manifest ZIP entry in APKs. */
    private static final String ANDROID_MANIFEST_ZIP_ENTRY_NAME = "AndroidManifest.xml";

    private final List<SignerConfig> mSignerConfigs;
    private final SignerConfig mSourceStampSignerConfig;
    private final SigningCertificateLineage mSourceStampSigningCertificateLineage;
    private final boolean mForceSourceStampOverwrite;
    private final boolean mSourceStampTimestampEnabled;
    private final Integer mMinSdkVersion;
    private final int mRotationMinSdkVersion;
    private final boolean mRotationTargetsDevRelease;
    private final boolean mV1SigningEnabled;
    private final boolean mV2SigningEnabled;
    private final boolean mV3SigningEnabled;
    private final boolean mV4SigningEnabled;
    private final boolean mAlignFileSize;
    private final boolean mVerityEnabled;
    private final boolean mV4ErrorReportingEnabled;
    private final boolean mDebuggableApkPermitted;
    private final boolean mOtherSignersSignaturesPreserved;
    private final boolean mAlignmentPreserved;
    private final int mLibraryPageAlignmentBytes;
    private final String mCreatedBy;

    private final ApkSignerEngine mSignerEngine;

    private final File mInputApkFile;
    private final DataSource mInputApkDataSource;

    private final File mOutputApkFile;
    private final DataSink mOutputApkDataSink;
    private final DataSource mOutputApkDataSource;

    private final File mOutputV4File;

    private final SigningCertificateLineage mSigningCertificateLineage;

    private ApkSigner(
            List<SignerConfig> signerConfigs,
            SignerConfig sourceStampSignerConfig,
            SigningCertificateLineage sourceStampSigningCertificateLineage,
            boolean forceSourceStampOverwrite,
            boolean sourceStampTimestampEnabled,
            Integer minSdkVersion,
            int rotationMinSdkVersion,
            boolean rotationTargetsDevRelease,
            boolean v1SigningEnabled,
            boolean v2SigningEnabled,
            boolean v3SigningEnabled,
            boolean v4SigningEnabled,
            boolean alignFileSize,
            boolean verityEnabled,
            boolean v4ErrorReportingEnabled,
            boolean debuggableApkPermitted,
            boolean otherSignersSignaturesPreserved,
            boolean alignmentPreserved,
            int libraryPageAlignmentBytes,
            String createdBy,
            ApkSignerEngine signerEngine,
            File inputApkFile,
            DataSource inputApkDataSource,
            File outputApkFile,
            DataSink outputApkDataSink,
            DataSource outputApkDataSource,
            File outputV4File,
            SigningCertificateLineage signingCertificateLineage) {

        mSignerConfigs = signerConfigs;
        mSourceStampSignerConfig = sourceStampSignerConfig;
        mSourceStampSigningCertificateLineage = sourceStampSigningCertificateLineage;
        mForceSourceStampOverwrite = forceSourceStampOverwrite;
        mSourceStampTimestampEnabled = sourceStampTimestampEnabled;
        mMinSdkVersion = minSdkVersion;
        mRotationMinSdkVersion = rotationMinSdkVersion;
        mRotationTargetsDevRelease = rotationTargetsDevRelease;
        mV1SigningEnabled = v1SigningEnabled;
        mV2SigningEnabled = v2SigningEnabled;
        mV3SigningEnabled = v3SigningEnabled;
        mV4SigningEnabled = v4SigningEnabled;
        mAlignFileSize = alignFileSize;
        mVerityEnabled = verityEnabled;
        mV4ErrorReportingEnabled = v4ErrorReportingEnabled;
        mDebuggableApkPermitted = debuggableApkPermitted;
        mOtherSignersSignaturesPreserved = otherSignersSignaturesPreserved;
        mAlignmentPreserved = alignmentPreserved;
        mLibraryPageAlignmentBytes = libraryPageAlignmentBytes;
        mCreatedBy = createdBy;

        mSignerEngine = signerEngine;

        mInputApkFile = inputApkFile;
        mInputApkDataSource = inputApkDataSource;

        mOutputApkFile = outputApkFile;
        mOutputApkDataSink = outputApkDataSink;
        mOutputApkDataSource = outputApkDataSource;

        mOutputV4File = outputV4File;

        mSigningCertificateLineage = signingCertificateLineage;
    }

    /**
     * Signs the input APK and outputs the resulting signed APK. The input APK is not modified.
     *
     * @throws IOException if an I/O error is encountered while reading or writing the APKs
     * @throws ApkFormatException if the input APK is malformed
     * @throws NoSuchAlgorithmException if the APK signatures cannot be produced or verified because
     *     a required cryptographic algorithm implementation is missing
     * @throws InvalidKeyException if a signature could not be generated because a signing key is
     *     not suitable for generating the signature
     * @throws SignatureException if an error occurred while generating or verifying a signature
     * @throws IllegalStateException if this signer's configuration is missing required information
     *     or if the signing engine is in an invalid state.
     */
    public void sign()
            throws IOException, ApkFormatException, NoSuchAlgorithmException, InvalidKeyException,
                    SignatureException, IllegalStateException {
        Closeable in = null;
        DataSource inputApk;
        try {
            if (mInputApkDataSource != null) {
                inputApk = mInputApkDataSource;
            } else if (mInputApkFile != null) {
                RandomAccessFile inputFile = new RandomAccessFile(mInputApkFile, "r");
                in = inputFile;
                inputApk = DataSources.asDataSource(inputFile);
            } else {
                throw new IllegalStateException("Input APK not specified");
            }

            Closeable out = null;
            try {
                DataSink outputApkOut;
                DataSource outputApkIn;
                if (mOutputApkDataSink != null) {
                    outputApkOut = mOutputApkDataSink;
                    outputApkIn = mOutputApkDataSource;
                } else if (mOutputApkFile != null) {
                    RandomAccessFile outputFile = new RandomAccessFile(mOutputApkFile, "rw");
                    out = outputFile;
                    outputFile.setLength(0);
                    outputApkOut = DataSinks.asDataSink(outputFile);
                    outputApkIn = DataSources.asDataSource(outputFile);
                } else {
                    throw new IllegalStateException("Output APK not specified");
                }

                sign(inputApk, outputApkOut, outputApkIn);
            } finally {
                if (out != null) {
                    out.close();
                }
            }
        } finally {
            if (in != null) {
                in.close();
            }
        }
    }

    private void sign(DataSource inputApk, DataSink outputApkOut, DataSource outputApkIn)
            throws IOException, ApkFormatException, NoSuchAlgorithmException, InvalidKeyException,
                    SignatureException {
        // Step 1. Find input APK's main ZIP sections
        ApkUtils.ZipSections inputZipSections;
        try {
            inputZipSections = ApkUtils.findZipSections(inputApk);
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Malformed APK: not a ZIP archive", e);
        }
        long inputApkSigningBlockOffset = -1;
        DataSource inputApkSigningBlock = null;
        try {
            ApkUtils.ApkSigningBlock apkSigningBlockInfo =
                    ApkUtils.findApkSigningBlock(inputApk, inputZipSections);
            inputApkSigningBlockOffset = apkSigningBlockInfo.getStartOffset();
            inputApkSigningBlock = apkSigningBlockInfo.getContents();
        } catch (ApkSigningBlockNotFoundException e) {
            // Input APK does not contain an APK Signing Block. That's OK. APKs are not required to
            // contain this block. It's only needed if the APK is signed using APK Signature Scheme
            // v2 and/or v3.
        }
        DataSource inputApkLfhSection =
                inputApk.slice(
                        0,
                        (inputApkSigningBlockOffset != -1)
                                ? inputApkSigningBlockOffset
                                : inputZipSections.getZipCentralDirectoryOffset());

        // Step 2. Parse the input APK's ZIP Central Directory
        ByteBuffer inputCd = getZipCentralDirectory(inputApk, inputZipSections);
        List<CentralDirectoryRecord> inputCdRecords =
                parseZipCentralDirectory(inputCd, inputZipSections);

        List<Hints.PatternWithRange> pinPatterns =
                extractPinPatterns(inputCdRecords, inputApkLfhSection);
        List<Hints.ByteRange> pinByteRanges = pinPatterns == null ? null : new ArrayList<>();

        // Step 3. Obtain a signer engine instance
        ApkSignerEngine signerEngine;
        if (mSignerEngine != null) {
            // Use the provided signer engine
            signerEngine = mSignerEngine;
        } else {
            // Construct a signer engine from the provided parameters
            int minSdkVersion;
            if (mMinSdkVersion != null) {
                // No need to extract minSdkVersion from the APK's AndroidManifest.xml
                minSdkVersion = mMinSdkVersion;
            } else {
                // Need to extract minSdkVersion from the APK's AndroidManifest.xml
                minSdkVersion = getMinSdkVersionFromApk(inputCdRecords, inputApkLfhSection);
            }
            List<DefaultApkSignerEngine.SignerConfig> engineSignerConfigs =
                    new ArrayList<>(mSignerConfigs.size());
            for (SignerConfig signerConfig : mSignerConfigs) {
                DefaultApkSignerEngine.SignerConfig.Builder signerConfigBuilder =
                        new DefaultApkSignerEngine.SignerConfig.Builder(
                                signerConfig.getName(),
                                signerConfig.getPrivateKey(),
                                signerConfig.getCertificates(),
                                signerConfig.getDeterministicDsaSigning());
                int signerMinSdkVersion = signerConfig.getMinSdkVersion();
                SigningCertificateLineage signerLineage =
                        signerConfig.getSigningCertificateLineage();
                if (signerMinSdkVersion > 0) {
                    signerConfigBuilder.setLineageForMinSdkVersion(signerLineage,
                            signerMinSdkVersion);
                }
                engineSignerConfigs.add(signerConfigBuilder.build());
            }
            DefaultApkSignerEngine.Builder signerEngineBuilder =
                    new DefaultApkSignerEngine.Builder(engineSignerConfigs, minSdkVersion)
                            .setV1SigningEnabled(mV1SigningEnabled)
                            .setV2SigningEnabled(mV2SigningEnabled)
                            .setV3SigningEnabled(mV3SigningEnabled)
                            .setVerityEnabled(mVerityEnabled)
                            .setDebuggableApkPermitted(mDebuggableApkPermitted)
                            .setOtherSignersSignaturesPreserved(mOtherSignersSignaturesPreserved)
                            .setSigningCertificateLineage(mSigningCertificateLineage)
                            .setMinSdkVersionForRotation(mRotationMinSdkVersion)
                            .setRotationTargetsDevRelease(mRotationTargetsDevRelease);
            if (mCreatedBy != null) {
                signerEngineBuilder.setCreatedBy(mCreatedBy);
            }
            if (mSourceStampSignerConfig != null) {
                signerEngineBuilder.setStampSignerConfig(
                        new DefaultApkSignerEngine.SignerConfig.Builder(
                                        mSourceStampSignerConfig.getName(),
                                        mSourceStampSignerConfig.getPrivateKey(),
                                        mSourceStampSignerConfig.getCertificates(),
                                        mSourceStampSignerConfig.getDeterministicDsaSigning())
                                .build());
                signerEngineBuilder.setSourceStampTimestampEnabled(mSourceStampTimestampEnabled);
            }
            if (mSourceStampSigningCertificateLineage != null) {
                signerEngineBuilder.setSourceStampSigningCertificateLineage(
                        mSourceStampSigningCertificateLineage);
            }
            signerEngine = signerEngineBuilder.build();
        }

        // Step 4. Provide the signer engine with the input APK's APK Signing Block (if any)
        if (inputApkSigningBlock != null) {
            signerEngine.inputApkSigningBlock(inputApkSigningBlock);
        }

        // Step 5. Iterate over input APK's entries and output the Local File Header + data of those
        // entries which need to be output. Entries are iterated in the order in which their Local
        // File Header records are stored in the file. This is to achieve better data locality in
        // case Central Directory entries are in the wrong order.
        List<CentralDirectoryRecord> inputCdRecordsSortedByLfhOffset =
                new ArrayList<>(inputCdRecords);
        Collections.sort(
                inputCdRecordsSortedByLfhOffset,
                CentralDirectoryRecord.BY_LOCAL_FILE_HEADER_OFFSET_COMPARATOR);
        int lastModifiedDateForNewEntries = -1;
        int lastModifiedTimeForNewEntries = -1;
        long inputOffset = 0;
        long outputOffset = 0;
        byte[] sourceStampCertificateDigest = null;
        Map<String, CentralDirectoryRecord> outputCdRecordsByName =
                new HashMap<>(inputCdRecords.size());
        for (final CentralDirectoryRecord inputCdRecord : inputCdRecordsSortedByLfhOffset) {
            String entryName = inputCdRecord.getName();
            if (Hints.PIN_BYTE_RANGE_ZIP_ENTRY_NAME.equals(entryName)) {
                continue; // We'll re-add below if needed.
            }
            if (SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME.equals(entryName)) {
                try {
                    sourceStampCertificateDigest =
                            LocalFileRecord.getUncompressedData(
                                    inputApkLfhSection, inputCdRecord, inputApkLfhSection.size());
                } catch (ZipFormatException ex) {
                    throw new ApkFormatException("Bad source stamp entry");
                }
                continue; // Existing source stamp is handled below as needed.
            }
            ApkSignerEngine.InputJarEntryInstructions entryInstructions =
                    signerEngine.inputJarEntry(entryName);
            boolean shouldOutput;
            switch (entryInstructions.getOutputPolicy()) {
                case OUTPUT:
                    shouldOutput = true;
                    break;
                case OUTPUT_BY_ENGINE:
                case SKIP:
                    shouldOutput = false;
                    break;
                default:
                    throw new RuntimeException(
                            "Unknown output policy: " + entryInstructions.getOutputPolicy());
            }

            long inputLocalFileHeaderStartOffset = inputCdRecord.getLocalFileHeaderOffset();
            if (inputLocalFileHeaderStartOffset > inputOffset) {
                // Unprocessed data in input starting at inputOffset and ending and the start of
                // this record's LFH. We output this data verbatim because this signer is supposed
                // to preserve as much of input as possible.
                long chunkSize = inputLocalFileHeaderStartOffset - inputOffset;
                inputApkLfhSection.feed(inputOffset, chunkSize, outputApkOut);
                outputOffset += chunkSize;
                inputOffset = inputLocalFileHeaderStartOffset;
            }
            LocalFileRecord inputLocalFileRecord;
            try {
                inputLocalFileRecord =
                        LocalFileRecord.getRecord(
                                inputApkLfhSection, inputCdRecord, inputApkLfhSection.size());
            } catch (ZipFormatException e) {
                throw new ApkFormatException("Malformed ZIP entry: " + inputCdRecord.getName(), e);
            }
            inputOffset += inputLocalFileRecord.getSize();

            ApkSignerEngine.InspectJarEntryRequest inspectEntryRequest =
                    entryInstructions.getInspectJarEntryRequest();
            if (inspectEntryRequest != null) {
                fulfillInspectInputJarEntryRequest(
                        inputApkLfhSection, inputLocalFileRecord, inspectEntryRequest);
            }

            if (shouldOutput) {
                // Find the max value of last modified, to be used for new entries added by the
                // signer.
                int lastModifiedDate = inputCdRecord.getLastModificationDate();
                int lastModifiedTime = inputCdRecord.getLastModificationTime();
                if ((lastModifiedDateForNewEntries == -1)
                        || (lastModifiedDate > lastModifiedDateForNewEntries)
                        || ((lastModifiedDate == lastModifiedDateForNewEntries)
                                && (lastModifiedTime > lastModifiedTimeForNewEntries))) {
                    lastModifiedDateForNewEntries = lastModifiedDate;
                    lastModifiedTimeForNewEntries = lastModifiedTime;
                }

                inspectEntryRequest = signerEngine.outputJarEntry(entryName);
                if (inspectEntryRequest != null) {
                    fulfillInspectInputJarEntryRequest(
                            inputApkLfhSection, inputLocalFileRecord, inspectEntryRequest);
                }

                // Output entry's Local File Header + data
                long outputLocalFileHeaderOffset = outputOffset;
                OutputSizeAndDataOffset outputLfrResult =
                        outputInputJarEntryLfhRecord(
                                inputApkLfhSection,
                                inputLocalFileRecord,
                                outputApkOut,
                                outputLocalFileHeaderOffset);
                outputOffset += outputLfrResult.outputBytes;
                long outputDataOffset =
                        outputLocalFileHeaderOffset + outputLfrResult.dataOffsetBytes;

                if (pinPatterns != null) {
                    boolean pinFileHeader = false;
                    for (Hints.PatternWithRange pinPattern : pinPatterns) {
                        if (pinPattern.matcher(inputCdRecord.getName()).matches()) {
                            Hints.ByteRange dataRange =
                                    new Hints.ByteRange(outputDataOffset, outputOffset);
                            Hints.ByteRange pinRange =
                                    pinPattern.ClampToAbsoluteByteRange(dataRange);
                            if (pinRange != null) {
                                pinFileHeader = true;
                                pinByteRanges.add(pinRange);
                            }
                        }
                    }
                    if (pinFileHeader) {
                        pinByteRanges.add(
                                new Hints.ByteRange(outputLocalFileHeaderOffset, outputDataOffset));
                    }
                }

                // Enqueue entry's Central Directory record for output
                CentralDirectoryRecord outputCdRecord;
                if (outputLocalFileHeaderOffset == inputLocalFileRecord.getStartOffsetInArchive()) {
                    outputCdRecord = inputCdRecord;
                } else {
                    outputCdRecord =
                            inputCdRecord.createWithModifiedLocalFileHeaderOffset(
                                    outputLocalFileHeaderOffset);
                }
                outputCdRecordsByName.put(entryName, outputCdRecord);
            }
        }
        long inputLfhSectionSize = inputApkLfhSection.size();
        if (inputOffset < inputLfhSectionSize) {
            // Unprocessed data in input starting at inputOffset and ending and the end of the input
            // APK's LFH section. We output this data verbatim because this signer is supposed
            // to preserve as much of input as possible.
            long chunkSize = inputLfhSectionSize - inputOffset;
            inputApkLfhSection.feed(inputOffset, chunkSize, outputApkOut);
            outputOffset += chunkSize;
            inputOffset = inputLfhSectionSize;
        }

        // Step 6. Sort output APK's Central Directory records in the order in which they should
        // appear in the output
        List<CentralDirectoryRecord> outputCdRecords = new ArrayList<>(inputCdRecords.size() + 10);
        for (CentralDirectoryRecord inputCdRecord : inputCdRecords) {
            String entryName = inputCdRecord.getName();
            CentralDirectoryRecord outputCdRecord = outputCdRecordsByName.get(entryName);
            if (outputCdRecord != null) {
                outputCdRecords.add(outputCdRecord);
            }
        }

        if (lastModifiedDateForNewEntries == -1) {
            lastModifiedDateForNewEntries = 0x3a21; // Jan 1 2009 (DOS)
            lastModifiedTimeForNewEntries = 0;
        }

        // Step 7. Generate and output SourceStamp certificate hash, if necessary. This may output
        // more Local File Header + data entries and add to the list of output Central Directory
        // records.
        if (signerEngine.isEligibleForSourceStamp()) {
            byte[] uncompressedData = signerEngine.generateSourceStampCertificateDigest();
            if (mForceSourceStampOverwrite
                    || sourceStampCertificateDigest == null
                    || Arrays.equals(uncompressedData, sourceStampCertificateDigest)) {
                outputOffset +=
                        outputDataToOutputApk(
                                SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME,
                                uncompressedData,
                                outputOffset,
                                outputCdRecords,
                                lastModifiedTimeForNewEntries,
                                lastModifiedDateForNewEntries,
                                outputApkOut);
            } else {
                throw new ApkFormatException(
                        String.format(
                                "Cannot generate SourceStamp. APK contains an existing entry with"
                                    + " the name: %s, and it is different than the provided source"
                                    + " stamp certificate",
                                SOURCE_STAMP_CERTIFICATE_HASH_ZIP_ENTRY_NAME));
            }
        }

        // Step 7.5. Generate pinlist.meta file if necessary.
        // This has to be before the step 8 so that the file is signed.
        if (pinByteRanges != null) {
            // Covers JAR signature and zip central dir entry.
            // The signature files don't have to be pinned, but pinning them isn't that wasteful
            // since the total size is small.
            pinByteRanges.add(new Hints.ByteRange(outputOffset, Long.MAX_VALUE));
            String entryName = Hints.PIN_BYTE_RANGE_ZIP_ENTRY_NAME;
            byte[] uncompressedData = Hints.encodeByteRangeList(pinByteRanges);

            requestOutputEntryInspection(signerEngine, entryName, uncompressedData);
            outputOffset +=
                outputDataToOutputApk(
                    entryName,
                    uncompressedData,
                    outputOffset,
                    outputCdRecords,
                    lastModifiedTimeForNewEntries,
                    lastModifiedDateForNewEntries,
                    outputApkOut);
        }

        // Step 8. Generate and output JAR signatures, if necessary. This may output more Local File
        // Header + data entries and add to the list of output Central Directory records.
        ApkSignerEngine.OutputJarSignatureRequest outputJarSignatureRequest =
                signerEngine.outputJarEntries();
        if (outputJarSignatureRequest != null) {
            for (ApkSignerEngine.OutputJarSignatureRequest.JarEntry entry :
                    outputJarSignatureRequest.getAdditionalJarEntries()) {
                String entryName = entry.getName();
                byte[] uncompressedData = entry.getData();

                requestOutputEntryInspection(signerEngine, entryName, uncompressedData);
                outputOffset +=
                        outputDataToOutputApk(
                                entryName,
                                uncompressedData,
                                outputOffset,
                                outputCdRecords,
                                lastModifiedTimeForNewEntries,
                                lastModifiedDateForNewEntries,
                                outputApkOut);
            }
            outputJarSignatureRequest.done();
        }

        // Step 9. Construct output ZIP Central Directory in an in-memory buffer
        long outputCentralDirSizeBytes = 0;
        for (CentralDirectoryRecord record : outputCdRecords) {
            outputCentralDirSizeBytes += record.getSize();
        }
        if (outputCentralDirSizeBytes > Integer.MAX_VALUE) {
            throw new IOException(
                    "Output ZIP Central Directory too large: "
                            + outputCentralDirSizeBytes
                            + " bytes");
        }
        ByteBuffer outputCentralDir = ByteBuffer.allocate((int) outputCentralDirSizeBytes);
        for (CentralDirectoryRecord record : outputCdRecords) {
            record.copyTo(outputCentralDir);
        }
        outputCentralDir.flip();
        DataSource outputCentralDirDataSource = new ByteBufferDataSource(outputCentralDir);
        long outputCentralDirStartOffset = outputOffset;
        int outputCentralDirRecordCount = outputCdRecords.size();

        // Step 10. Construct output ZIP End of Central Directory record in an in-memory buffer
        // because it can be adjusted in Step 11 due to signing block.
        //   - CD offset (it's shifted by signing block)
        //   - Comments (when the output file needs to be sized 4k-aligned)
        ByteBuffer outputEocd =
                EocdRecord.createWithModifiedCentralDirectoryInfo(
                        inputZipSections.getZipEndOfCentralDirectory(),
                        outputCentralDirRecordCount,
                        outputCentralDirDataSource.size(),
                        outputCentralDirStartOffset);

        // Step 11. Generate and output APK Signature Scheme v2 and/or v3 signatures and/or
        // SourceStamp signatures, if necessary.
        // This may insert an APK Signing Block just before the output's ZIP Central Directory
        ApkSignerEngine.OutputApkSigningBlockRequest2 outputApkSigningBlockRequest =
                signerEngine.outputZipSections2(
                        outputApkIn,
                        outputCentralDirDataSource,
                        DataSources.asDataSource(outputEocd));

        if (outputApkSigningBlockRequest != null) {
            int padding = outputApkSigningBlockRequest.getPaddingSizeBeforeApkSigningBlock();
            byte[] outputApkSigningBlock = outputApkSigningBlockRequest.getApkSigningBlock();
            outputApkSigningBlockRequest.done();

            long fileSize =
                    outputCentralDirStartOffset
                            + outputCentralDirDataSource.size()
                            + padding
                            + outputApkSigningBlock.length
                            + outputEocd.remaining();
            if (mAlignFileSize && (fileSize % ANDROID_FILE_ALIGNMENT_BYTES != 0)) {
                int eocdPadding =
                        (int)
                                (ANDROID_FILE_ALIGNMENT_BYTES
                                        - fileSize % ANDROID_FILE_ALIGNMENT_BYTES);
                // Replace EOCD with padding one so that output file size can be the multiples of
                // alignment.
                outputEocd = EocdRecord.createWithPaddedComment(outputEocd, eocdPadding);

                // Since EoCD has changed, we need to regenerate signing block as well.
                outputApkSigningBlockRequest =
                        signerEngine.outputZipSections2(
                                outputApkIn,
                                new ByteBufferDataSource(outputCentralDir),
                                DataSources.asDataSource(outputEocd));
                outputApkSigningBlock = outputApkSigningBlockRequest.getApkSigningBlock();
                outputApkSigningBlockRequest.done();
            }

            outputApkOut.consume(ByteBuffer.allocate(padding));
            outputApkOut.consume(outputApkSigningBlock, 0, outputApkSigningBlock.length);
            ZipUtils.setZipEocdCentralDirectoryOffset(
                    outputEocd,
                    outputCentralDirStartOffset + padding + outputApkSigningBlock.length);
        }

        // Step 12. Output ZIP Central Directory and ZIP End of Central Directory
        outputCentralDirDataSource.feed(0, outputCentralDirDataSource.size(), outputApkOut);
        outputApkOut.consume(outputEocd);
        signerEngine.outputDone();

        // Step 13. Generate and output APK Signature Scheme v4 signatures, if necessary.
        if (mV4SigningEnabled) {
            signerEngine.signV4(outputApkIn, mOutputV4File, !mV4ErrorReportingEnabled);
        }
    }

    private static void requestOutputEntryInspection(
            ApkSignerEngine signerEngine,
            String entryName,
            byte[] uncompressedData)
            throws IOException {
        ApkSignerEngine.InspectJarEntryRequest inspectEntryRequest =
                signerEngine.outputJarEntry(entryName);
        if (inspectEntryRequest != null) {
            inspectEntryRequest.getDataSink().consume(
                    uncompressedData, 0, uncompressedData.length);
            inspectEntryRequest.done();
        }
    }

    private static long outputDataToOutputApk(
            String entryName,
            byte[] uncompressedData,
            long localFileHeaderOffset,
            List<CentralDirectoryRecord> outputCdRecords,
            int lastModifiedTimeForNewEntries,
            int lastModifiedDateForNewEntries,
            DataSink outputApkOut)
            throws IOException {
        ZipUtils.DeflateResult deflateResult = ZipUtils.deflate(ByteBuffer.wrap(uncompressedData));
        byte[] compressedData = deflateResult.output;
        long uncompressedDataCrc32 = deflateResult.inputCrc32;
        long numOfDataBytes =
                LocalFileRecord.outputRecordWithDeflateCompressedData(
                        entryName,
                        lastModifiedTimeForNewEntries,
                        lastModifiedDateForNewEntries,
                        compressedData,
                        uncompressedDataCrc32,
                        uncompressedData.length,
                        outputApkOut);
        outputCdRecords.add(
                CentralDirectoryRecord.createWithDeflateCompressedData(
                        entryName,
                        lastModifiedTimeForNewEntries,
                        lastModifiedDateForNewEntries,
                        uncompressedDataCrc32,
                        compressedData.length,
                        uncompressedData.length,
                        localFileHeaderOffset));
        return numOfDataBytes;
    }

    private static void fulfillInspectInputJarEntryRequest(
            DataSource lfhSection,
            LocalFileRecord localFileRecord,
            ApkSignerEngine.InspectJarEntryRequest inspectEntryRequest)
            throws IOException, ApkFormatException {
        try {
            localFileRecord.outputUncompressedData(lfhSection, inspectEntryRequest.getDataSink());
        } catch (ZipFormatException e) {
            throw new ApkFormatException("Malformed ZIP entry: " + localFileRecord.getName(), e);
        }
        inspectEntryRequest.done();
    }

    private static class OutputSizeAndDataOffset {
        public long outputBytes;
        public long dataOffsetBytes;

        public OutputSizeAndDataOffset(long outputBytes, long dataOffsetBytes) {
            this.outputBytes = outputBytes;
            this.dataOffsetBytes = dataOffsetBytes;
        }
    }

    private OutputSizeAndDataOffset outputInputJarEntryLfhRecord(
            DataSource inputLfhSection,
            LocalFileRecord inputRecord,
            DataSink outputLfhSection,
            long outputOffset)
            throws IOException {
        long inputOffset = inputRecord.getStartOffsetInArchive();
        if (inputOffset == outputOffset && mAlignmentPreserved) {
            // This record's data will be aligned same as in the input APK.
            return new OutputSizeAndDataOffset(
                    inputRecord.outputRecord(inputLfhSection, outputLfhSection),
                    inputRecord.getDataStartOffsetInRecord());
        }
        int dataAlignmentMultiple = getInputJarEntryDataAlignmentMultiple(inputRecord);
        if ((dataAlignmentMultiple <= 1)
                || ((inputOffset % dataAlignmentMultiple) == (outputOffset % dataAlignmentMultiple)
                        && mAlignmentPreserved)) {
            // This record's data will be aligned same as in the input APK.
            return new OutputSizeAndDataOffset(
                    inputRecord.outputRecord(inputLfhSection, outputLfhSection),
                    inputRecord.getDataStartOffsetInRecord());
        }

        long inputDataStartOffset = inputOffset + inputRecord.getDataStartOffsetInRecord();
        if ((inputDataStartOffset % dataAlignmentMultiple) != 0 && mAlignmentPreserved) {
            // This record's data is not aligned in the input APK. No need to align it in the
            // output.
            return new OutputSizeAndDataOffset(
                    inputRecord.outputRecord(inputLfhSection, outputLfhSection),
                    inputRecord.getDataStartOffsetInRecord());
        }

        // This record's data needs to be re-aligned in the output. This is achieved using the
        // record's extra field.
        ByteBuffer aligningExtra =
                createExtraFieldToAlignData(
                        inputRecord.getExtra(),
                        outputOffset + inputRecord.getExtraFieldStartOffsetInsideRecord(),
                        dataAlignmentMultiple);
        long dataOffset =
                (long) inputRecord.getDataStartOffsetInRecord()
                        + aligningExtra.remaining()
                        - inputRecord.getExtra().remaining();
        return new OutputSizeAndDataOffset(
                inputRecord.outputRecordWithModifiedExtra(
                        inputLfhSection, aligningExtra, outputLfhSection),
                dataOffset);
    }

    private int getInputJarEntryDataAlignmentMultiple(LocalFileRecord entry) {
        if (entry.isDataCompressed()) {
            // Compressed entries don't need to be aligned
            return 1;
        }

        // Attempt to obtain the alignment multiple from the entry's extra field.
        ByteBuffer extra = entry.getExtra();
        if (extra.hasRemaining()) {
            extra.order(ByteOrder.LITTLE_ENDIAN);
            // FORMAT: sequence of fields. Each field consists of:
            //   * uint16 ID
            //   * uint16 size
            //   * 'size' bytes: payload
            while (extra.remaining() >= 4) {
                short headerId = extra.getShort();
                int dataSize = ZipUtils.getUnsignedInt16(extra);
                if (dataSize > extra.remaining()) {
                    // Malformed field -- insufficient input remaining
                    break;
                }
                if (headerId != ALIGNMENT_ZIP_EXTRA_DATA_FIELD_HEADER_ID) {
                    // Skip this field
                    extra.position(extra.position() + dataSize);
                    continue;
                }
                // This is APK alignment field.
                // FORMAT:
                //  * uint16 alignment multiple (in bytes)
                //  * remaining bytes -- padding to achieve alignment of data which starts after
                //    the extra field
                if (dataSize < 2) {
                    // Malformed
                    break;
                }
                return ZipUtils.getUnsignedInt16(extra);
            }
        }

        // Fall back to filename-based defaults
        return (entry.getName().endsWith(".so")) ? mLibraryPageAlignmentBytes : 4;
    }

    private static ByteBuffer createExtraFieldToAlignData(
            ByteBuffer original, long extraStartOffset, int dataAlignmentMultiple) {
        if (dataAlignmentMultiple <= 1) {
            return original;
        }

        // In the worst case scenario, we'll increase the output size by 6 + dataAlignment - 1.
        ByteBuffer result = ByteBuffer.allocate(original.remaining() + 5 + dataAlignmentMultiple);
        result.order(ByteOrder.LITTLE_ENDIAN);

        // Step 1. Output all extra fields other than the one which is to do with alignment
        // FORMAT: sequence of fields. Each field consists of:
        //   * uint16 ID
        //   * uint16 size
        //   * 'size' bytes: payload
        while (original.remaining() >= 4) {
            short headerId = original.getShort();
            int dataSize = ZipUtils.getUnsignedInt16(original);
            if (dataSize > original.remaining()) {
                // Malformed field -- insufficient input remaining
                break;
            }
            if (((headerId == 0) && (dataSize == 0))
                    || (headerId == ALIGNMENT_ZIP_EXTRA_DATA_FIELD_HEADER_ID)) {
                // Ignore the field if it has to do with the old APK data alignment method (filling
                // the extra field with 0x00 bytes) or the new APK data alignment method.
                original.position(original.position() + dataSize);
                continue;
            }
            // Copy this field (including header) to the output
            original.position(original.position() - 4);
            int originalLimit = original.limit();
            original.limit(original.position() + 4 + dataSize);
            result.put(original);
            original.limit(originalLimit);
        }

        // Step 2. Add alignment field
        // FORMAT:
        //  * uint16 extra header ID
        //  * uint16 extra data size
        //        Payload ('data size' bytes)
        //      * uint16 alignment multiple (in bytes)
        //      * remaining bytes -- padding to achieve alignment of data which starts after the
        //        extra field
        long dataMinStartOffset =
                extraStartOffset
                        + result.position()
                        + ALIGNMENT_ZIP_EXTRA_DATA_FIELD_MIN_SIZE_BYTES;
        int paddingSizeBytes =
                (dataAlignmentMultiple - ((int) (dataMinStartOffset % dataAlignmentMultiple)))
                        % dataAlignmentMultiple;
        result.putShort(ALIGNMENT_ZIP_EXTRA_DATA_FIELD_HEADER_ID);
        ZipUtils.putUnsignedInt16(result, 2 + paddingSizeBytes);
        ZipUtils.putUnsignedInt16(result, dataAlignmentMultiple);
        result.position(result.position() + paddingSizeBytes);
        result.flip();

        return result;
    }

    private static ByteBuffer getZipCentralDirectory(
            DataSource apk, ApkUtils.ZipSections apkSections)
            throws IOException, ApkFormatException {
        long cdSizeBytes = apkSections.getZipCentralDirectorySizeBytes();
        if (cdSizeBytes > Integer.MAX_VALUE) {
            throw new ApkFormatException("ZIP Central Directory too large: " + cdSizeBytes);
        }
        long cdOffset = apkSections.getZipCentralDirectoryOffset();
        ByteBuffer cd = apk.getByteBuffer(cdOffset, (int) cdSizeBytes);
        cd.order(ByteOrder.LITTLE_ENDIAN);
        return cd;
    }

    private static List<CentralDirectoryRecord> parseZipCentralDirectory(
            ByteBuffer cd, ApkUtils.ZipSections apkSections) throws ApkFormatException {
        long cdOffset = apkSections.getZipCentralDirectoryOffset();
        int expectedCdRecordCount = apkSections.getZipCentralDirectoryRecordCount();
        List<CentralDirectoryRecord> cdRecords = new ArrayList<>(expectedCdRecordCount);
        Set<String> entryNames = new HashSet<>(expectedCdRecordCount);
        for (int i = 0; i < expectedCdRecordCount; i++) {
            CentralDirectoryRecord cdRecord;
            int offsetInsideCd = cd.position();
            try {
                cdRecord = CentralDirectoryRecord.getRecord(cd);
            } catch (ZipFormatException e) {
                throw new ApkFormatException(
                        "Malformed ZIP Central Directory record #"
                                + (i + 1)
                                + " at file offset "
                                + (cdOffset + offsetInsideCd),
                        e);
            }
            String entryName = cdRecord.getName();
            if (!entryNames.add(entryName)) {
                throw new ApkFormatException(
                        "Multiple ZIP entries with the same name: " + entryName);
            }
            cdRecords.add(cdRecord);
        }
        if (cd.hasRemaining()) {
            throw new ApkFormatException(
                    "Unused space at the end of ZIP Central Directory: "
                            + cd.remaining()
                            + " bytes starting at file offset "
                            + (cdOffset + cd.position()));
        }

        return cdRecords;
    }

    private static CentralDirectoryRecord findCdRecord(
            List<CentralDirectoryRecord> cdRecords, String name) {
        for (CentralDirectoryRecord cdRecord : cdRecords) {
            if (name.equals(cdRecord.getName())) {
                return cdRecord;
            }
        }
        return null;
    }

    /**
     * Returns the contents of the APK's {@code AndroidManifest.xml} or {@code null} if this entry
     * is not present in the APK.
     */
    static ByteBuffer getAndroidManifestFromApk(
            List<CentralDirectoryRecord> cdRecords, DataSource lhfSection)
            throws IOException, ApkFormatException, ZipFormatException {
        CentralDirectoryRecord androidManifestCdRecord =
                findCdRecord(cdRecords, ANDROID_MANIFEST_ZIP_ENTRY_NAME);
        if (androidManifestCdRecord == null) {
            throw new ApkFormatException("Missing " + ANDROID_MANIFEST_ZIP_ENTRY_NAME);
        }

        return ByteBuffer.wrap(
                LocalFileRecord.getUncompressedData(
                        lhfSection, androidManifestCdRecord, lhfSection.size()));
    }

    /**
     * Return list of pin patterns embedded in the pin pattern asset file. If no such file, return
     * {@code null}.
     */
    private static List<Hints.PatternWithRange> extractPinPatterns(
            List<CentralDirectoryRecord> cdRecords, DataSource lhfSection)
            throws IOException, ApkFormatException {
        CentralDirectoryRecord pinListCdRecord =
                findCdRecord(cdRecords, Hints.PIN_HINT_ASSET_ZIP_ENTRY_NAME);
        List<Hints.PatternWithRange> pinPatterns = null;
        if (pinListCdRecord != null) {
            pinPatterns = new ArrayList<>();
            byte[] patternBlob;
            try {
                patternBlob =
                        LocalFileRecord.getUncompressedData(
                                lhfSection, pinListCdRecord, lhfSection.size());
            } catch (ZipFormatException ex) {
                throw new ApkFormatException("Bad " + pinListCdRecord);
            }
            pinPatterns = Hints.parsePinPatterns(patternBlob);
        }
        return pinPatterns;
    }

    /**
     * Returns the minimum Android version (API Level) supported by the provided APK. This is based
     * on the {@code android:minSdkVersion} attributes of the APK's {@code AndroidManifest.xml}.
     */
    private static int getMinSdkVersionFromApk(
            List<CentralDirectoryRecord> cdRecords, DataSource lhfSection)
            throws IOException, MinSdkVersionException {
        ByteBuffer androidManifest;
        try {
            androidManifest = getAndroidManifestFromApk(cdRecords, lhfSection);
        } catch (ZipFormatException | ApkFormatException e) {
            throw new MinSdkVersionException(
                    "Failed to determine APK's minimum supported Android platform version", e);
        }
        return ApkUtils.getMinSdkVersionFromBinaryAndroidManifest(androidManifest);
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
        private final SigningCertificateLineage mSigningCertificateLineage;

        private SignerConfig(Builder builder) {
            mName = builder.mName;
            mPrivateKey = builder.mPrivateKey;
            mCertificates = Collections.unmodifiableList(new ArrayList<>(builder.mCertificates));
            mDeterministicDsaSigning = builder.mDeterministicDsaSigning;
            mMinSdkVersion = builder.mMinSdkVersion;
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
            public Builder(
                    String name,
                    PrivateKey privateKey,
                    List<X509Certificate> certificates) {
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
             *     deterministic variant (RFC6979) should be used.
             */
            public Builder(
                    String name,
                    PrivateKey privateKey,
                    List<X509Certificate> certificates,
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
             * Returns a new {@code SignerConfig} instance configured based on the configuration of
             * this builder.
             */
            public SignerConfig build() {
                return new SignerConfig(this);
            }
        }
    }

    /**
     * Builder of {@link ApkSigner} instances.
     *
     * <p>The builder requires the following information to construct a working {@code ApkSigner}:
     *
     * <ul>
     *   <li>Signer configs or {@link ApkSignerEngine} -- provided in the constructor,
     *   <li>APK to be signed -- see {@link #setInputApk(File) setInputApk} variants,
     *   <li>where to store the output signed APK -- see {@link #setOutputApk(File) setOutputApk}
     *       variants.
     * </ul>
     */
    public static class Builder {
        private final List<SignerConfig> mSignerConfigs;
        private SignerConfig mSourceStampSignerConfig;
        private SigningCertificateLineage mSourceStampSigningCertificateLineage;
        private boolean mForceSourceStampOverwrite = false;
        private boolean mSourceStampTimestampEnabled = true;
        private boolean mV1SigningEnabled = true;
        private boolean mV2SigningEnabled = true;
        private boolean mV3SigningEnabled = true;
        private boolean mV4SigningEnabled = true;
        private boolean mAlignFileSize = false;
        private boolean mVerityEnabled = false;
        private boolean mV4ErrorReportingEnabled = false;
        private boolean mDebuggableApkPermitted = true;
        private boolean mOtherSignersSignaturesPreserved;
        private boolean mAlignmentPreserved = false;
        private int mLibraryPageAlignmentBytes = LIBRARY_PAGE_ALIGNMENT_BYTES;
        private String mCreatedBy;
        private Integer mMinSdkVersion;
        private int mRotationMinSdkVersion = V3SchemeConstants.DEFAULT_ROTATION_MIN_SDK_VERSION;
        private boolean mRotationTargetsDevRelease = false;

        private final ApkSignerEngine mSignerEngine;

        private File mInputApkFile;
        private DataSource mInputApkDataSource;

        private File mOutputApkFile;
        private DataSink mOutputApkDataSink;
        private DataSource mOutputApkDataSource;

        private File mOutputV4File;

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
         * Constructs a new {@code Builder} for an {@code ApkSigner} which signs using the provided
         * signer configurations. The resulting signer may be further customized through this
         * builder's setters, such as {@link #setMinSdkVersion(int)}, {@link
         * #setV1SigningEnabled(boolean)}, {@link #setV2SigningEnabled(boolean)}, {@link
         * #setOtherSignersSignaturesPreserved(boolean)}, {@link #setCreatedBy(String)}.
         *
         * <p>{@link #Builder(ApkSignerEngine)} is an alternative for advanced use cases where more
         * control over low-level details of signing is desired.
         */
        public Builder(List<SignerConfig> signerConfigs) {
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
            mSignerEngine = null;
        }

        /**
         * Constructs a new {@code Builder} for an {@code ApkSigner} which signs using the provided
         * signing engine. This is meant for advanced use cases where more control is needed over
         * the lower-level details of signing. For typical use cases, {@link #Builder(List)} is more
         * appropriate.
         */
        public Builder(ApkSignerEngine signerEngine) {
            if (signerEngine == null) {
                throw new NullPointerException("signerEngine == null");
            }
            mSignerEngine = signerEngine;
            mSignerConfigs = null;
        }

        /** Sets the signing configuration of the source stamp to be embedded in the APK. */
        public Builder setSourceStampSignerConfig(SignerConfig sourceStampSignerConfig) {
            mSourceStampSignerConfig = sourceStampSignerConfig;
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
         * Sets whether the APK should overwrite existing source stamp, if found.
         *
         * @param force {@code true} to require the APK to be overwrite existing source stamp
         */
        public Builder setForceSourceStampOverwrite(boolean force) {
            mForceSourceStampOverwrite = force;
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
         * Sets the APK to be signed.
         *
         * @see #setInputApk(DataSource)
         */
        public Builder setInputApk(File inputApk) {
            if (inputApk == null) {
                throw new NullPointerException("inputApk == null");
            }
            mInputApkFile = inputApk;
            mInputApkDataSource = null;
            return this;
        }

        /**
         * Sets the APK to be signed.
         *
         * @see #setInputApk(File)
         */
        public Builder setInputApk(DataSource inputApk) {
            if (inputApk == null) {
                throw new NullPointerException("inputApk == null");
            }
            mInputApkDataSource = inputApk;
            mInputApkFile = null;
            return this;
        }

        /**
         * Sets the location of the output (signed) APK. {@code ApkSigner} will create this file if
         * it doesn't exist.
         *
         * @see #setOutputApk(ReadableDataSink)
         * @see #setOutputApk(DataSink, DataSource)
         */
        public Builder setOutputApk(File outputApk) {
            if (outputApk == null) {
                throw new NullPointerException("outputApk == null");
            }
            mOutputApkFile = outputApk;
            mOutputApkDataSink = null;
            mOutputApkDataSource = null;
            return this;
        }

        /**
         * Sets the readable data sink which will receive the output (signed) APK. After signing,
         * the contents of the output APK will be available via the {@link DataSource} interface of
         * the sink.
         *
         * <p>This variant of {@code setOutputApk} is useful for avoiding writing the output APK to
         * a file. For example, an in-memory data sink, such as {@link
         * DataSinks#newInMemoryDataSink()}, could be used instead of a file.
         *
         * @see #setOutputApk(File)
         * @see #setOutputApk(DataSink, DataSource)
         */
        public Builder setOutputApk(ReadableDataSink outputApk) {
            if (outputApk == null) {
                throw new NullPointerException("outputApk == null");
            }
            return setOutputApk(outputApk, outputApk);
        }

        /**
         * Sets the sink which will receive the output (signed) APK. Data received by the {@code
         * outputApkOut} sink must be visible through the {@code outputApkIn} data source.
         *
         * <p>This is an advanced variant of {@link #setOutputApk(ReadableDataSink)}, enabling the
         * sink and the source to be different objects.
         *
         * @see #setOutputApk(ReadableDataSink)
         * @see #setOutputApk(File)
         */
        public Builder setOutputApk(DataSink outputApkOut, DataSource outputApkIn) {
            if (outputApkOut == null) {
                throw new NullPointerException("outputApkOut == null");
            }
            if (outputApkIn == null) {
                throw new NullPointerException("outputApkIn == null");
            }
            mOutputApkFile = null;
            mOutputApkDataSink = outputApkOut;
            mOutputApkDataSource = outputApkIn;
            return this;
        }

        /**
         * Sets the location of the V4 output file. {@code ApkSigner} will create this file if it
         * doesn't exist.
         */
        public Builder setV4SignatureOutputFile(File v4SignatureOutputFile) {
            if (v4SignatureOutputFile == null) {
                throw new NullPointerException("v4HashRootOutputFile == null");
            }
            mOutputV4File = v4SignatureOutputFile;
            return this;
        }

        /**
         * Sets the minimum Android platform version (API Level) on which APK signatures produced by
         * the signer being built must verify. This method is useful for overriding the default
         * behavior where the minimum API Level is obtained from the {@code android:minSdkVersion}
         * attribute of the APK's {@code AndroidManifest.xml}.
         *
         * <p><em>Note:</em> This method may result in APK signatures which don't verify on some
         * Android platform versions supported by the APK.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setMinSdkVersion(int minSdkVersion) {
            checkInitializedWithoutEngine();
            mMinSdkVersion = minSdkVersion;
            return this;
        }

        /**
         * Sets the minimum Android platform version (API Level) for which an APK's rotated signing
         * key should be used to produce the APK's signature. The original signing key for the APK
         * will be used for all previous platform versions. If a rotated key with signing lineage is
         * not provided then this method is a noop. This method is useful for overriding the
         * default behavior where Android T is set as the minimum API level for rotation.
         *
         * <p><em>Note:</em>Specifying a {@code minSdkVersion} value <= 32 (Android Sv2) will result
         * in the original V3 signing block being used without platform targeting.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setMinSdkVersionForRotation(int minSdkVersion) {
            checkInitializedWithoutEngine();
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
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         */
        public Builder setRotationTargetsDevRelease(boolean enabled) {
            checkInitializedWithoutEngine();
            mRotationTargetsDevRelease = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed using JAR signing (aka v1 signature scheme).
         *
         * <p>By default, whether APK is signed using JAR signing is determined by {@code
         * ApkSigner}, based on the platform versions supported by the APK or specified using {@link
         * #setMinSdkVersion(int)}. Disabling JAR signing will result in APK signatures which don't
         * verify on Android Marshmallow (Android 6.0, API Level 23) and lower.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @param enabled {@code true} to require the APK to be signed using JAR signing, {@code
         *     false} to require the APK to not be signed using JAR signing.
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         * @see <a
         *     href="https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File">JAR
         *     signing</a>
         */
        public Builder setV1SigningEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
            mV1SigningEnabled = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed using APK Signature Scheme v2 (aka v2 signature
         * scheme).
         *
         * <p>By default, whether APK is signed using APK Signature Scheme v2 is determined by
         * {@code ApkSigner} based on the platform versions supported by the APK or specified using
         * {@link #setMinSdkVersion(int)}.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @param enabled {@code true} to require the APK to be signed using APK Signature Scheme
         *     v2, {@code false} to require the APK to not be signed using APK Signature Scheme v2.
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         * @see <a href="https://source.android.com/security/apksigning/v2.html">APK Signature
         *     Scheme v2</a>
         */
        public Builder setV2SigningEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
            mV2SigningEnabled = enabled;
            return this;
        }

        /**
         * Sets whether the APK should be signed using APK Signature Scheme v3 (aka v3 signature
         * scheme).
         *
         * <p>By default, whether APK is signed using APK Signature Scheme v3 is determined by
         * {@code ApkSigner} based on the platform versions supported by the APK or specified using
         * {@link #setMinSdkVersion(int)}.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * <p><em>Note:</em> APK Signature Scheme v3 only supports a single signing certificate, but
         * may take multiple signers mapping to different targeted platform versions.
         *
         * @param enabled {@code true} to require the APK to be signed using APK Signature Scheme
         *     v3, {@code false} to require the APK to not be signed using APK Signature Scheme v3.
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setV3SigningEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
            mV3SigningEnabled = enabled;
            if (enabled) {
                mV3SigningExplicitlyEnabled = true;
            } else {
                mV3SigningExplicitlyDisabled = true;
            }
            return this;
        }

        /**
         * Sets whether the APK should be signed using APK Signature Scheme v4.
         *
         * <p>V4 signing requires that the APK be v2 or v3 signed.
         *
         * @param enabled {@code true} to require the APK to be signed using APK Signature Scheme v2
         *     or v3 and generate an v4 signature file
         */
        public Builder setV4SigningEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
            mV4SigningEnabled = enabled;
            mV4ErrorReportingEnabled = enabled;
            return this;
        }

        /**
         * Sets whether errors during v4 signing should be reported and halt the signing process.
         *
         * <p>Error reporting for v4 signing is disabled by default, but will be enabled if the
         * caller invokes {@link #setV4SigningEnabled} with a value of true. This method is useful
         * for tools that enable v4 signing by default but don't want to fail the signing process if
         * the user did not explicitly request the v4 signing.
         *
         * @param enabled {@code false} to prevent errors encountered during the V4 signing from
         *     halting the signing process
         */
        public Builder setV4ErrorReportingEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
            mV4ErrorReportingEnabled = enabled;
            return this;
        }

       /**
         * Sets whether the output APK files should be sized as multiples of 4K.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setAlignFileSize(boolean alignFileSize) {
            checkInitializedWithoutEngine();
            mAlignFileSize = alignFileSize;
            return this;
        }

        /**
         * Sets whether to enable the verity signature algorithm for the v2 and v3 signature
         * schemes.
         *
         * @param enabled {@code true} to enable the verity signature algorithm for inclusion in the
         *     v2 and v3 signature blocks.
         */
        public Builder setVerityEnabled(boolean enabled) {
            checkInitializedWithoutEngine();
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
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         */
        public Builder setDebuggableApkPermitted(boolean permitted) {
            checkInitializedWithoutEngine();
            mDebuggableApkPermitted = permitted;
            return this;
        }

        /**
         * Sets whether signatures produced by signers other than the ones configured in this engine
         * should be copied from the input APK to the output APK.
         *
         * <p>By default, signatures of other signers are omitted from the output APK.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setOtherSignersSignaturesPreserved(boolean preserved) {
            checkInitializedWithoutEngine();
            mOtherSignersSignaturesPreserved = preserved;
            return this;
        }

        /**
         * Sets the value of the {@code Created-By} field in JAR signature files.
         *
         * <p><em>Note:</em> This method may only be invoked when this builder is not initialized
         * with an {@link ApkSignerEngine}.
         *
         * @throws IllegalStateException if this builder was initialized with an {@link
         *     ApkSignerEngine}
         */
        public Builder setCreatedBy(String createdBy) {
            checkInitializedWithoutEngine();
            if (createdBy == null) {
                throw new NullPointerException();
            }
            mCreatedBy = createdBy;
            return this;
        }

        private void checkInitializedWithoutEngine() {
            if (mSignerEngine != null) {
                throw new IllegalStateException(
                        "Operation is not available when builder initialized with an engine");
            }
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
         * Sets whether the existing alignment within the APK should be preserved; the
         * default for this setting is false. When this value is false, the value provided to
         * {@link #setLibraryPageAlignmentBytes(int)} will be used to page align native library
         * files and 4 bytes will be used to align all other uncompressed files.
         */
        public Builder setAlignmentPreserved(boolean alignmentPreserved) {
            mAlignmentPreserved = alignmentPreserved;
            return this;
        }

        /**
         * Sets the number of bytes to be used to page align native library files in the APK; the
         * default for this setting is {@link Constants#LIBRARY_PAGE_ALIGNMENT_BYTES}.
         */
        public Builder setLibraryPageAlignmentBytes(int libraryPageAlignmentBytes) {
            mLibraryPageAlignmentBytes = libraryPageAlignmentBytes;
            return this;
        }

        /**
         * Returns a new {@code ApkSigner} instance initialized according to the configuration of
         * this builder.
         */
        public ApkSigner build() {
            if (mV3SigningExplicitlyDisabled && mV3SigningExplicitlyEnabled) {
                throw new IllegalStateException(
                        "Builder configured to both enable and disable APK "
                                + "Signature Scheme v3 signing");
            }

            if (mV3SigningExplicitlyDisabled) {
                mV3SigningEnabled = false;
            }

            if (mV3SigningExplicitlyEnabled) {
                mV3SigningEnabled = true;
            }

            // If V4 signing is not explicitly set, and V2/V3 signing is disabled, then V4 signing
            // must be disabled as well as it is dependent on V2/V3.
            if (mV4SigningEnabled && !mV2SigningEnabled && !mV3SigningEnabled) {
                if (!mV4ErrorReportingEnabled) {
                    mV4SigningEnabled = false;
                } else {
                    throw new IllegalStateException(
                            "APK Signature Scheme v4 signing requires at least "
                                    + "v2 or v3 signing to be enabled");
                }
            }

            // TODO - if v3 signing is enabled, check provided signers and history to see if valid

            return new ApkSigner(
                    mSignerConfigs,
                    mSourceStampSignerConfig,
                    mSourceStampSigningCertificateLineage,
                    mForceSourceStampOverwrite,
                    mSourceStampTimestampEnabled,
                    mMinSdkVersion,
                    mRotationMinSdkVersion,
                    mRotationTargetsDevRelease,
                    mV1SigningEnabled,
                    mV2SigningEnabled,
                    mV3SigningEnabled,
                    mV4SigningEnabled,
                    mAlignFileSize,
                    mVerityEnabled,
                    mV4ErrorReportingEnabled,
                    mDebuggableApkPermitted,
                    mOtherSignersSignaturesPreserved,
                    mAlignmentPreserved,
                    mLibraryPageAlignmentBytes,
                    mCreatedBy,
                    mSignerEngine,
                    mInputApkFile,
                    mInputApkDataSource,
                    mOutputApkFile,
                    mOutputApkDataSink,
                    mOutputApkDataSource,
                    mOutputV4File,
                    mSigningCertificateLineage);
        }
    }
}
