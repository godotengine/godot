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

package com.android.apksig.internal.apk;

import static com.android.apksig.Constants.OID_RSA_ENCRYPTION;
import static com.android.apksig.internal.apk.ContentDigestAlgorithm.CHUNKED_SHA256;
import static com.android.apksig.internal.apk.ContentDigestAlgorithm.CHUNKED_SHA512;
import static com.android.apksig.internal.apk.ContentDigestAlgorithm.VERITY_CHUNKED_SHA256;

import com.android.apksig.ApkVerifier;
import com.android.apksig.SigningCertificateLineage;
import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.apk.ApkUtils;
import com.android.apksig.internal.asn1.Asn1BerParser;
import com.android.apksig.internal.asn1.Asn1DecodingException;
import com.android.apksig.internal.asn1.Asn1DerEncoder;
import com.android.apksig.internal.asn1.Asn1EncodingException;
import com.android.apksig.internal.asn1.Asn1OpaqueObject;
import com.android.apksig.internal.pkcs7.AlgorithmIdentifier;
import com.android.apksig.internal.pkcs7.ContentInfo;
import com.android.apksig.internal.pkcs7.EncapsulatedContentInfo;
import com.android.apksig.internal.pkcs7.IssuerAndSerialNumber;
import com.android.apksig.internal.pkcs7.Pkcs7Constants;
import com.android.apksig.internal.pkcs7.SignedData;
import com.android.apksig.internal.pkcs7.SignerIdentifier;
import com.android.apksig.internal.pkcs7.SignerInfo;
import com.android.apksig.internal.util.ByteBufferDataSource;
import com.android.apksig.internal.util.ChainedDataSource;
import com.android.apksig.internal.util.GuaranteedEncodedFormX509Certificate;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.internal.util.VerityTreeBuilder;
import com.android.apksig.internal.util.X509CertificateUtils;
import com.android.apksig.internal.x509.RSAPublicKey;
import com.android.apksig.internal.x509.SubjectPublicKeyInfo;
import com.android.apksig.internal.zip.ZipUtils;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSinks;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.DataSources;
import com.android.apksig.util.RunnablesExecutor;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.DigestException;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import javax.security.auth.x500.X500Principal;

public class ApkSigningBlockUtils {

    private static final long CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES = 1024 * 1024;
    public static final int ANDROID_COMMON_PAGE_ALIGNMENT_BYTES = 4096;
    private static final byte[] APK_SIGNING_BLOCK_MAGIC =
          new byte[] {
              0x41, 0x50, 0x4b, 0x20, 0x53, 0x69, 0x67, 0x20,
              0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x20, 0x34, 0x32,
          };
    public static final int VERITY_PADDING_BLOCK_ID = 0x42726577;

    private static final ContentDigestAlgorithm[] V4_CONTENT_DIGEST_ALGORITHMS =
            {CHUNKED_SHA512, VERITY_CHUNKED_SHA256, CHUNKED_SHA256};

    public static final int VERSION_SOURCE_STAMP = 0;
    public static final int VERSION_JAR_SIGNATURE_SCHEME = 1;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V2 = 2;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V3 = 3;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V31 = 31;
    public static final int VERSION_APK_SIGNATURE_SCHEME_V4 = 4;

    /**
     * Returns positive number if {@code alg1} is preferred over {@code alg2}, {@code -1} if
     * {@code alg2} is preferred over {@code alg1}, and {@code 0} if there is no preference.
     */
    public static int compareSignatureAlgorithm(SignatureAlgorithm alg1, SignatureAlgorithm alg2) {
        return ApkSigningBlockUtilsLite.compareSignatureAlgorithm(alg1, alg2);
    }

    /**
     * Verifies integrity of the APK outside of the APK Signing Block by computing digests of the
     * APK and comparing them against the digests listed in APK Signing Block. The expected digests
     * are taken from {@code SignerInfos} of the provided {@code result}.
     *
     * <p>This method adds one or more errors to the {@code result} if a verification error is
     * expected to be encountered on Android. No errors are added to the {@code result} if the APK's
     * integrity is expected to verify on Android for each algorithm in
     * {@code contentDigestAlgorithms}.
     *
     * <p>The reason this method is currently not parameterized by a
     * {@code [minSdkVersion, maxSdkVersion]} range is that up until now content digest algorithms
     * exhibit the same behavior on all Android platform versions.
     */
    public static void verifyIntegrity(
            RunnablesExecutor executor,
            DataSource beforeApkSigningBlock,
            DataSource centralDir,
            ByteBuffer eocd,
            Set<ContentDigestAlgorithm> contentDigestAlgorithms,
            Result result) throws IOException, NoSuchAlgorithmException {
        if (contentDigestAlgorithms.isEmpty()) {
            // This should never occur because this method is invoked once at least one signature
            // is verified, meaning at least one content digest is known.
            throw new RuntimeException("No content digests found");
        }

        // For the purposes of verifying integrity, ZIP End of Central Directory (EoCD) must be
        // treated as though its Central Directory offset points to the start of APK Signing Block.
        // We thus modify the EoCD accordingly.
        ByteBuffer modifiedEocd = ByteBuffer.allocate(eocd.remaining());
        int eocdSavedPos = eocd.position();
        modifiedEocd.order(ByteOrder.LITTLE_ENDIAN);
        modifiedEocd.put(eocd);
        modifiedEocd.flip();

        // restore eocd to position prior to modification in case it is to be used elsewhere
        eocd.position(eocdSavedPos);
        ZipUtils.setZipEocdCentralDirectoryOffset(modifiedEocd, beforeApkSigningBlock.size());
        Map<ContentDigestAlgorithm, byte[]> actualContentDigests;
        try {
            actualContentDigests =
                    computeContentDigests(
                            executor,
                            contentDigestAlgorithms,
                            beforeApkSigningBlock,
                            centralDir,
                            new ByteBufferDataSource(modifiedEocd));
            // Special checks for the verity algorithm requirements.
            if (actualContentDigests.containsKey(VERITY_CHUNKED_SHA256)) {
                if ((beforeApkSigningBlock.size() % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES != 0)) {
                    throw new RuntimeException(
                            "APK Signing Block is not aligned on 4k boundary: " +
                            beforeApkSigningBlock.size());
                }

                long centralDirOffset = ZipUtils.getZipEocdCentralDirectoryOffset(eocd);
                long signingBlockSize = centralDirOffset - beforeApkSigningBlock.size();
                if (signingBlockSize % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES != 0) {
                    throw new RuntimeException(
                            "APK Signing Block size is not multiple of page size: " +
                            signingBlockSize);
                }
            }
        } catch (DigestException e) {
            throw new RuntimeException("Failed to compute content digests", e);
        }
        if (!contentDigestAlgorithms.equals(actualContentDigests.keySet())) {
            throw new RuntimeException(
                    "Mismatch between sets of requested and computed content digests"
                            + " . Requested: " + contentDigestAlgorithms
                            + ", computed: " + actualContentDigests.keySet());
        }

        // Compare digests computed over the rest of APK against the corresponding expected digests
        // in signer blocks.
        for (Result.SignerInfo signerInfo : result.signers) {
            for (Result.SignerInfo.ContentDigest expected : signerInfo.contentDigests) {
                SignatureAlgorithm signatureAlgorithm =
                        SignatureAlgorithm.findById(expected.getSignatureAlgorithmId());
                if (signatureAlgorithm == null) {
                    continue;
                }
                ContentDigestAlgorithm contentDigestAlgorithm =
                        signatureAlgorithm.getContentDigestAlgorithm();
                // if the current digest algorithm is not in the list provided by the caller then
                // ignore it; the signer may contain digests not recognized by the specified SDK
                // range.
                if (!contentDigestAlgorithms.contains(contentDigestAlgorithm)) {
                    continue;
                }
                byte[] expectedDigest = expected.getValue();
                byte[] actualDigest = actualContentDigests.get(contentDigestAlgorithm);
                if (!Arrays.equals(expectedDigest, actualDigest)) {
                    if (result.signatureSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V2) {
                        signerInfo.addError(
                                ApkVerifier.Issue.V2_SIG_APK_DIGEST_DID_NOT_VERIFY,
                                contentDigestAlgorithm,
                                toHex(expectedDigest),
                                toHex(actualDigest));
                    } else if (result.signatureSchemeVersion == VERSION_APK_SIGNATURE_SCHEME_V3) {
                        signerInfo.addError(
                                ApkVerifier.Issue.V3_SIG_APK_DIGEST_DID_NOT_VERIFY,
                                contentDigestAlgorithm,
                                toHex(expectedDigest),
                                toHex(actualDigest));
                    }
                    continue;
                }
                signerInfo.verifiedContentDigests.put(contentDigestAlgorithm, actualDigest);
            }
        }
    }

    public static ByteBuffer findApkSignatureSchemeBlock(
            ByteBuffer apkSigningBlock,
            int blockId,
            Result result) throws SignatureNotFoundException {
        try {
            return ApkSigningBlockUtilsLite.findApkSignatureSchemeBlock(apkSigningBlock, blockId);
        } catch (com.android.apksig.internal.apk.SignatureNotFoundException e) {
            throw new SignatureNotFoundException(e.getMessage());
        }
    }

    public static void checkByteOrderLittleEndian(ByteBuffer buffer) {
        ApkSigningBlockUtilsLite.checkByteOrderLittleEndian(buffer);
    }

    public static ByteBuffer getLengthPrefixedSlice(ByteBuffer source) throws ApkFormatException {
        return ApkSigningBlockUtilsLite.getLengthPrefixedSlice(source);
    }

    public static byte[] readLengthPrefixedByteArray(ByteBuffer buf) throws ApkFormatException {
        return ApkSigningBlockUtilsLite.readLengthPrefixedByteArray(buf);
    }

    public static String toHex(byte[] value) {
        return ApkSigningBlockUtilsLite.toHex(value);
    }

    public static Map<ContentDigestAlgorithm, byte[]> computeContentDigests(
            RunnablesExecutor executor,
            Set<ContentDigestAlgorithm> digestAlgorithms,
            DataSource beforeCentralDir,
            DataSource centralDir,
            DataSource eocd) throws IOException, NoSuchAlgorithmException, DigestException {
        Map<ContentDigestAlgorithm, byte[]> contentDigests = new HashMap<>();
        Set<ContentDigestAlgorithm> oneMbChunkBasedAlgorithm = new HashSet<>();
        for (ContentDigestAlgorithm digestAlgorithm : digestAlgorithms) {
            if (digestAlgorithm == ContentDigestAlgorithm.CHUNKED_SHA256
                    || digestAlgorithm == ContentDigestAlgorithm.CHUNKED_SHA512) {
                oneMbChunkBasedAlgorithm.add(digestAlgorithm);
            }
        }
        computeOneMbChunkContentDigests(
                executor,
                oneMbChunkBasedAlgorithm,
                new DataSource[] { beforeCentralDir, centralDir, eocd },
                contentDigests);

        if (digestAlgorithms.contains(VERITY_CHUNKED_SHA256)) {
            computeApkVerityDigest(beforeCentralDir, centralDir, eocd, contentDigests);
        }
        return contentDigests;
    }

    static void computeOneMbChunkContentDigests(
            Set<ContentDigestAlgorithm> digestAlgorithms,
            DataSource[] contents,
            Map<ContentDigestAlgorithm, byte[]> outputContentDigests)
            throws IOException, NoSuchAlgorithmException, DigestException {
        // For each digest algorithm the result is computed as follows:
        // 1. Each segment of contents is split into consecutive chunks of 1 MB in size.
        //    The final chunk will be shorter iff the length of segment is not a multiple of 1 MB.
        //    No chunks are produced for empty (zero length) segments.
        // 2. The digest of each chunk is computed over the concatenation of byte 0xa5, the chunk's
        //    length in bytes (uint32 little-endian) and the chunk's contents.
        // 3. The output digest is computed over the concatenation of the byte 0x5a, the number of
        //    chunks (uint32 little-endian) and the concatenation of digests of chunks of all
        //    segments in-order.

        long chunkCountLong = 0;
        for (DataSource input : contents) {
            chunkCountLong +=
                    getChunkCount(input.size(), CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES);
        }
        if (chunkCountLong > Integer.MAX_VALUE) {
            throw new DigestException("Input too long: " + chunkCountLong + " chunks");
        }
        int chunkCount = (int) chunkCountLong;

        ContentDigestAlgorithm[] digestAlgorithmsArray =
                digestAlgorithms.toArray(new ContentDigestAlgorithm[digestAlgorithms.size()]);
        MessageDigest[] mds = new MessageDigest[digestAlgorithmsArray.length];
        byte[][] digestsOfChunks = new byte[digestAlgorithmsArray.length][];
        int[] digestOutputSizes = new int[digestAlgorithmsArray.length];
        for (int i = 0; i < digestAlgorithmsArray.length; i++) {
            ContentDigestAlgorithm digestAlgorithm = digestAlgorithmsArray[i];
            int digestOutputSizeBytes = digestAlgorithm.getChunkDigestOutputSizeBytes();
            digestOutputSizes[i] = digestOutputSizeBytes;
            byte[] concatenationOfChunkCountAndChunkDigests =
                    new byte[5 + chunkCount * digestOutputSizeBytes];
            concatenationOfChunkCountAndChunkDigests[0] = 0x5a;
            setUnsignedInt32LittleEndian(
                    chunkCount, concatenationOfChunkCountAndChunkDigests, 1);
            digestsOfChunks[i] = concatenationOfChunkCountAndChunkDigests;
            String jcaAlgorithm = digestAlgorithm.getJcaMessageDigestAlgorithm();
            mds[i] = MessageDigest.getInstance(jcaAlgorithm);
        }

        DataSink mdSink = DataSinks.asDataSink(mds);
        byte[] chunkContentPrefix = new byte[5];
        chunkContentPrefix[0] = (byte) 0xa5;
        int chunkIndex = 0;
        // Optimization opportunity: digests of chunks can be computed in parallel. However,
        // determining the number of computations to be performed in parallel is non-trivial. This
        // depends on a wide range of factors, such as data source type (e.g., in-memory or fetched
        // from file), CPU/memory/disk cache bandwidth and latency, interconnect architecture of CPU
        // cores, load on the system from other threads of execution and other processes, size of
        // input.
        // For now, we compute these digests sequentially and thus have the luxury of improving
        // performance by writing the digest of each chunk into a pre-allocated buffer at exactly
        // the right position. This avoids unnecessary allocations, copying, and enables the final
        // digest to be more efficient because it's presented with all of its input in one go.
        for (DataSource input : contents) {
            long inputOffset = 0;
            long inputRemaining = input.size();
            while (inputRemaining > 0) {
                int chunkSize =
                        (int) Math.min(inputRemaining, CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES);
                setUnsignedInt32LittleEndian(chunkSize, chunkContentPrefix, 1);
                for (int i = 0; i < mds.length; i++) {
                    mds[i].update(chunkContentPrefix);
                }
                try {
                    input.feed(inputOffset, chunkSize, mdSink);
                } catch (IOException e) {
                    throw new IOException("Failed to read chunk #" + chunkIndex, e);
                }
                for (int i = 0; i < digestAlgorithmsArray.length; i++) {
                    MessageDigest md = mds[i];
                    byte[] concatenationOfChunkCountAndChunkDigests = digestsOfChunks[i];
                    int expectedDigestSizeBytes = digestOutputSizes[i];
                    int actualDigestSizeBytes =
                            md.digest(
                                    concatenationOfChunkCountAndChunkDigests,
                                    5 + chunkIndex * expectedDigestSizeBytes,
                                    expectedDigestSizeBytes);
                    if (actualDigestSizeBytes != expectedDigestSizeBytes) {
                        throw new RuntimeException(
                                "Unexpected output size of " + md.getAlgorithm()
                                        + " digest: " + actualDigestSizeBytes);
                    }
                }
                inputOffset += chunkSize;
                inputRemaining -= chunkSize;
                chunkIndex++;
            }
        }

        for (int i = 0; i < digestAlgorithmsArray.length; i++) {
            ContentDigestAlgorithm digestAlgorithm = digestAlgorithmsArray[i];
            byte[] concatenationOfChunkCountAndChunkDigests = digestsOfChunks[i];
            MessageDigest md = mds[i];
            byte[] digest = md.digest(concatenationOfChunkCountAndChunkDigests);
            outputContentDigests.put(digestAlgorithm, digest);
        }
    }

    static void computeOneMbChunkContentDigests(
            RunnablesExecutor executor,
            Set<ContentDigestAlgorithm> digestAlgorithms,
            DataSource[] contents,
            Map<ContentDigestAlgorithm, byte[]> outputContentDigests)
            throws NoSuchAlgorithmException, DigestException {
        long chunkCountLong = 0;
        for (DataSource input : contents) {
            chunkCountLong +=
                    getChunkCount(input.size(), CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES);
        }
        if (chunkCountLong > Integer.MAX_VALUE) {
            throw new DigestException("Input too long: " + chunkCountLong + " chunks");
        }
        int chunkCount = (int) chunkCountLong;

        List<ChunkDigests> chunkDigestsList = new ArrayList<>(digestAlgorithms.size());
        for (ContentDigestAlgorithm algorithms : digestAlgorithms) {
            chunkDigestsList.add(new ChunkDigests(algorithms, chunkCount));
        }

        ChunkSupplier chunkSupplier = new ChunkSupplier(contents);
        executor.execute(() -> new ChunkDigester(chunkSupplier, chunkDigestsList));

        // Compute and write out final digest for each algorithm.
        for (ChunkDigests chunkDigests : chunkDigestsList) {
            MessageDigest messageDigest = chunkDigests.createMessageDigest();
            outputContentDigests.put(
                    chunkDigests.algorithm,
                    messageDigest.digest(chunkDigests.concatOfDigestsOfChunks));
        }
    }

    private static class ChunkDigests {
        private final ContentDigestAlgorithm algorithm;
        private final int digestOutputSize;
        private final byte[] concatOfDigestsOfChunks;

        private ChunkDigests(ContentDigestAlgorithm algorithm, int chunkCount) {
            this.algorithm = algorithm;
            digestOutputSize = this.algorithm.getChunkDigestOutputSizeBytes();
            concatOfDigestsOfChunks = new byte[1 + 4 + chunkCount * digestOutputSize];

            // Fill the initial values of the concatenated digests of chunks, which is
            // {0x5a, 4-bytes-of-little-endian-chunk-count, digests*...}.
            concatOfDigestsOfChunks[0] = 0x5a;
            setUnsignedInt32LittleEndian(chunkCount, concatOfDigestsOfChunks, 1);
        }

        private MessageDigest createMessageDigest() throws NoSuchAlgorithmException {
            return MessageDigest.getInstance(algorithm.getJcaMessageDigestAlgorithm());
        }

        private int getOffset(int chunkIndex) {
            return 1 + 4 + chunkIndex * digestOutputSize;
        }
    }

    /**
     * A per-thread digest worker.
     */
    private static class ChunkDigester implements Runnable {
        private final ChunkSupplier dataSupplier;
        private final List<ChunkDigests> chunkDigests;
        private final List<MessageDigest> messageDigests;
        private final DataSink mdSink;

        private ChunkDigester(ChunkSupplier dataSupplier, List<ChunkDigests> chunkDigests) {
            this.dataSupplier = dataSupplier;
            this.chunkDigests = chunkDigests;
            messageDigests = new ArrayList<>(chunkDigests.size());
            for (ChunkDigests chunkDigest : chunkDigests) {
                try {
                    messageDigests.add(chunkDigest.createMessageDigest());
                } catch (NoSuchAlgorithmException ex) {
                    throw new RuntimeException(ex);
                }
            }
            mdSink = DataSinks.asDataSink(messageDigests.toArray(new MessageDigest[0]));
        }

        @Override
        public void run() {
            byte[] chunkContentPrefix = new byte[5];
            chunkContentPrefix[0] = (byte) 0xa5;

            try {
                for (ChunkSupplier.Chunk chunk = dataSupplier.get();
                     chunk != null;
                     chunk = dataSupplier.get()) {
                    int size = chunk.size;
                    if (size > CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES) {
                        throw new RuntimeException("Chunk size greater than expected: " + size);
                    }

                    // First update with the chunk prefix.
                    setUnsignedInt32LittleEndian(size, chunkContentPrefix, 1);
                    mdSink.consume(chunkContentPrefix, 0, chunkContentPrefix.length);

                    // Then update with the chunk data.
                    mdSink.consume(chunk.data);

                    // Now finalize chunk for all algorithms.
                    for (int i = 0; i < chunkDigests.size(); i++) {
                        ChunkDigests chunkDigest = chunkDigests.get(i);
                        int actualDigestSize = messageDigests.get(i).digest(
                                chunkDigest.concatOfDigestsOfChunks,
                                chunkDigest.getOffset(chunk.chunkIndex),
                                chunkDigest.digestOutputSize);
                        if (actualDigestSize != chunkDigest.digestOutputSize) {
                            throw new RuntimeException(
                                    "Unexpected output size of " + chunkDigest.algorithm
                                            + " digest: " + actualDigestSize);
                        }
                    }
                }
            } catch (IOException | DigestException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Thread-safe 1MB DataSource chunk supplier. When bounds are met in a
     * supplied {@link DataSource}, the data from the next {@link DataSource}
     * are NOT concatenated. Only the next call to get() will fetch from the
     * next {@link DataSource} in the input {@link DataSource} array.
     */
    private static class ChunkSupplier implements Supplier<ChunkSupplier.Chunk> {
        private final DataSource[] dataSources;
        private final int[] chunkCounts;
        private final int totalChunkCount;
        private final AtomicInteger nextIndex;

        private ChunkSupplier(DataSource[] dataSources) {
            this.dataSources = dataSources;
            chunkCounts = new int[dataSources.length];
            int totalChunkCount = 0;
            for (int i = 0; i < dataSources.length; i++) {
                long chunkCount = getChunkCount(dataSources[i].size(),
                        CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES);
                if (chunkCount > Integer.MAX_VALUE) {
                    throw new RuntimeException(
                            String.format(
                                    "Number of chunks in dataSource[%d] is greater than max int.",
                                    i));
                }
                chunkCounts[i] = (int)chunkCount;
                totalChunkCount = (int) (totalChunkCount + chunkCount);
            }
            this.totalChunkCount = totalChunkCount;
            nextIndex = new AtomicInteger(0);
        }

        /**
         * We map an integer index to the termination-adjusted dataSources 1MB chunks.
         * Note that {@link Chunk}s could be less than 1MB, namely the last 1MB-aligned
         * blocks in each input {@link DataSource} (unless the DataSource itself is
         * 1MB-aligned).
         */
        @Override
        public ChunkSupplier.Chunk get() {
            int index = nextIndex.getAndIncrement();
            if (index < 0 || index >= totalChunkCount) {
                return null;
            }

            int dataSourceIndex = 0;
            long dataSourceChunkOffset = index;
            for (; dataSourceIndex < dataSources.length; dataSourceIndex++) {
                if (dataSourceChunkOffset < chunkCounts[dataSourceIndex]) {
                    break;
                }
                dataSourceChunkOffset -= chunkCounts[dataSourceIndex];
            }

            long remainingSize = Math.min(
                    dataSources[dataSourceIndex].size() -
                            dataSourceChunkOffset * CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES,
                    CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES);

            final int size = (int)remainingSize;
            final ByteBuffer buffer = ByteBuffer.allocate(size);
            try {
                dataSources[dataSourceIndex].copyTo(
                        dataSourceChunkOffset * CONTENT_DIGESTED_CHUNK_MAX_SIZE_BYTES, size,
                        buffer);
            } catch (IOException e) {
                throw new IllegalStateException("Failed to read chunk", e);
            }
            buffer.rewind();

            return new Chunk(index, buffer, size);
        }

        static class Chunk {
            private final int chunkIndex;
            private final ByteBuffer data;
            private final int size;

            private Chunk(int chunkIndex, ByteBuffer data, int size) {
                this.chunkIndex = chunkIndex;
                this.data = data;
                this.size = size;
            }
        }
    }

    @SuppressWarnings("ByteBufferBackingArray")
    private static void computeApkVerityDigest(DataSource beforeCentralDir, DataSource centralDir,
            DataSource eocd, Map<ContentDigestAlgorithm, byte[]> outputContentDigests)
            throws IOException, NoSuchAlgorithmException {
        ByteBuffer encoded = createVerityDigestBuffer(true);
        // Use 0s as salt for now.  This also needs to be consistent in the fsverify header for
        // kernel to use.
        try (VerityTreeBuilder builder = new VerityTreeBuilder(new byte[8])) {
            byte[] rootHash = builder.generateVerityTreeRootHash(beforeCentralDir, centralDir,
                    eocd);
            encoded.put(rootHash);
            encoded.putLong(beforeCentralDir.size() + centralDir.size() + eocd.size());
            outputContentDigests.put(VERITY_CHUNKED_SHA256, encoded.array());
        }
    }

    private static ByteBuffer createVerityDigestBuffer(boolean includeSourceDataSize) {
        // FORMAT:
        // OFFSET       DATA TYPE  DESCRIPTION
        // * @+0  bytes uint8[32]  Merkle tree root hash of SHA-256
        // * @+32 bytes int64      (optional) Length of source data
        int backBufferSize =
                VERITY_CHUNKED_SHA256.getChunkDigestOutputSizeBytes();
        if (includeSourceDataSize) {
            backBufferSize += Long.SIZE / Byte.SIZE;
        }
        ByteBuffer encoded = ByteBuffer.allocate(backBufferSize);
        encoded.order(ByteOrder.LITTLE_ENDIAN);
        return encoded;
    }

    public static class VerityTreeAndDigest {
        public final ContentDigestAlgorithm contentDigestAlgorithm;
        public final byte[] rootHash;
        public final byte[] tree;

        VerityTreeAndDigest(ContentDigestAlgorithm contentDigestAlgorithm, byte[] rootHash,
                byte[] tree) {
            this.contentDigestAlgorithm = contentDigestAlgorithm;
            this.rootHash = rootHash;
            this.tree = tree;
        }
    }

    @SuppressWarnings("ByteBufferBackingArray")
    public static VerityTreeAndDigest computeChunkVerityTreeAndDigest(DataSource dataSource)
            throws IOException, NoSuchAlgorithmException {
        ByteBuffer encoded = createVerityDigestBuffer(false);
        // Use 0s as salt for now.  This also needs to be consistent in the fsverify header for
        // kernel to use.
        try (VerityTreeBuilder builder = new VerityTreeBuilder(null)) {
            ByteBuffer tree = builder.generateVerityTree(dataSource);
            byte[] rootHash = builder.getRootHashFromTree(tree);
            encoded.put(rootHash);
            return new VerityTreeAndDigest(VERITY_CHUNKED_SHA256, encoded.array(), tree.array());
        }
    }

    private static long getChunkCount(long inputSize, long chunkSize) {
        return (inputSize + chunkSize - 1) / chunkSize;
    }

    private static void setUnsignedInt32LittleEndian(int value, byte[] result, int offset) {
        result[offset] = (byte) (value & 0xff);
        result[offset + 1] = (byte) ((value >> 8) & 0xff);
        result[offset + 2] = (byte) ((value >> 16) & 0xff);
        result[offset + 3] = (byte) ((value >> 24) & 0xff);
    }

    public static byte[] encodePublicKey(PublicKey publicKey)
            throws InvalidKeyException, NoSuchAlgorithmException {
        byte[] encodedPublicKey = null;
        if ("X.509".equals(publicKey.getFormat())) {
            encodedPublicKey = publicKey.getEncoded();
            // if the key is an RSA key check for a negative modulus
            String keyAlgorithm = publicKey.getAlgorithm();
            if ("RSA".equals(keyAlgorithm) || OID_RSA_ENCRYPTION.equals(keyAlgorithm)) {
                try {
                    // Parse the encoded public key into the separate elements of the
                    // SubjectPublicKeyInfo to obtain the SubjectPublicKey.
                    ByteBuffer encodedPublicKeyBuffer = ByteBuffer.wrap(encodedPublicKey);
                    SubjectPublicKeyInfo subjectPublicKeyInfo = Asn1BerParser.parse(
                            encodedPublicKeyBuffer, SubjectPublicKeyInfo.class);
                    // The SubjectPublicKey is encoded as a bit string within the
                    // SubjectPublicKeyInfo. The first byte of the encoding is the number of padding
                    // bits; store this and decode the rest of the bit string into the RSA modulus
                    // and exponent.
                    ByteBuffer subjectPublicKeyBuffer = subjectPublicKeyInfo.subjectPublicKey;
                    byte padding = subjectPublicKeyBuffer.get();
                    RSAPublicKey rsaPublicKey = Asn1BerParser.parse(subjectPublicKeyBuffer,
                            RSAPublicKey.class);
                    // if the modulus is negative then attempt to reencode it with a leading 0 sign
                    // byte.
                    if (rsaPublicKey.modulus.compareTo(BigInteger.ZERO) < 0) {
                        // A negative modulus indicates the leading bit in the integer is 1. Per
                        // ASN.1 encoding rules to encode a positive integer with the leading bit
                        // set to 1 a byte containing all zeros should precede the integer encoding.
                        byte[] encodedModulus = rsaPublicKey.modulus.toByteArray();
                        byte[] reencodedModulus = new byte[encodedModulus.length + 1];
                        reencodedModulus[0] = 0;
                        System.arraycopy(encodedModulus, 0, reencodedModulus, 1,
                                encodedModulus.length);
                        rsaPublicKey.modulus = new BigInteger(reencodedModulus);
                        // Once the modulus has been corrected reencode the RSAPublicKey, then
                        // restore the padding value in the bit string and reencode the entire
                        // SubjectPublicKeyInfo to be returned to the caller.
                        byte[] reencodedRSAPublicKey = Asn1DerEncoder.encode(rsaPublicKey);
                        byte[] reencodedSubjectPublicKey =
                                new byte[reencodedRSAPublicKey.length + 1];
                        reencodedSubjectPublicKey[0] = padding;
                        System.arraycopy(reencodedRSAPublicKey, 0, reencodedSubjectPublicKey, 1,
                                reencodedRSAPublicKey.length);
                        subjectPublicKeyInfo.subjectPublicKey = ByteBuffer.wrap(
                                reencodedSubjectPublicKey);
                        encodedPublicKey = Asn1DerEncoder.encode(subjectPublicKeyInfo);
                    }
                } catch (Asn1DecodingException | Asn1EncodingException e) {
                    System.out.println("Caught a exception encoding the public key: " + e);
                    e.printStackTrace();
                    encodedPublicKey = null;
                }
            }
        }
        if (encodedPublicKey == null) {
            try {
                encodedPublicKey =
                        KeyFactory.getInstance(publicKey.getAlgorithm())
                                .getKeySpec(publicKey, X509EncodedKeySpec.class)
                                .getEncoded();
            } catch (InvalidKeySpecException e) {
                throw new InvalidKeyException(
                        "Failed to obtain X.509 encoded form of public key " + publicKey
                                + " of class " + publicKey.getClass().getName(),
                        e);
            }
        }
        if ((encodedPublicKey == null) || (encodedPublicKey.length == 0)) {
            throw new InvalidKeyException(
                    "Failed to obtain X.509 encoded form of public key " + publicKey
                            + " of class " + publicKey.getClass().getName());
        }
        return encodedPublicKey;
    }

    public static List<byte[]> encodeCertificates(List<X509Certificate> certificates)
            throws CertificateEncodingException {
        List<byte[]> result = new ArrayList<>(certificates.size());
        for (X509Certificate certificate : certificates) {
            result.add(certificate.getEncoded());
        }
        return result;
    }

    public static byte[] encodeAsLengthPrefixedElement(byte[] bytes) {
        byte[][] adapterBytes = new byte[1][];
        adapterBytes[0] = bytes;
        return encodeAsSequenceOfLengthPrefixedElements(adapterBytes);
    }

    public static byte[] encodeAsSequenceOfLengthPrefixedElements(List<byte[]> sequence) {
        return encodeAsSequenceOfLengthPrefixedElements(
                sequence.toArray(new byte[sequence.size()][]));
    }

    public static byte[] encodeAsSequenceOfLengthPrefixedElements(byte[][] sequence) {
        int payloadSize = 0;
        for (byte[] element : sequence) {
            payloadSize += 4 + element.length;
        }
        ByteBuffer result = ByteBuffer.allocate(payloadSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] element : sequence) {
            result.putInt(element.length);
            result.put(element);
        }
        return result.array();
      }

    public static byte[] encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(
            List<Pair<Integer, byte[]>> sequence) {
        return ApkSigningBlockUtilsLite
                .encodeAsSequenceOfLengthPrefixedPairsOfIntAndLengthPrefixedBytes(sequence);
      }

    /**
     * Returns the APK Signature Scheme block contained in the provided APK file for the given ID
     * and the additional information relevant for verifying the block against the file.
     *
     * @param blockId the ID value in the APK Signing Block's sequence of ID-value pairs
     *                identifying the appropriate block to find, e.g. the APK Signature Scheme v2
     *                block ID.
     *
     * @throws SignatureNotFoundException if the APK is not signed using given APK Signature Scheme
     * @throws IOException if an I/O error occurs while reading the APK
     */
    public static SignatureInfo findSignature(
            DataSource apk, ApkUtils.ZipSections zipSections, int blockId, Result result)
                    throws IOException, SignatureNotFoundException {
        try {
            return ApkSigningBlockUtilsLite.findSignature(apk, zipSections, blockId);
        } catch (com.android.apksig.internal.apk.SignatureNotFoundException e) {
            throw new SignatureNotFoundException(e.getMessage());
        }
    }

    /**
     * Generates a new DataSource representing the APK contents before the Central Directory with
     * padding, if padding is requested.  If the existing data entries before the Central Directory
     * are already aligned, or no padding is requested, the original DataSource is used.  This
     * padding is used to allow for verity-based APK verification.
     *
     * @return {@code Pair} containing the potentially new {@code DataSource} and the amount of
     *         padding used.
     */
    public static Pair<DataSource, Integer> generateApkSigningBlockPadding(
            DataSource beforeCentralDir,
            boolean apkSigningBlockPaddingSupported) {

        // Ensure APK Signing Block starts from page boundary.
        int padSizeBeforeSigningBlock = 0;
        if (apkSigningBlockPaddingSupported &&
                (beforeCentralDir.size() % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES != 0)) {
            padSizeBeforeSigningBlock = (int) (
                    ANDROID_COMMON_PAGE_ALIGNMENT_BYTES -
                            beforeCentralDir.size() % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES);
            beforeCentralDir = new ChainedDataSource(
                    beforeCentralDir,
                    DataSources.asDataSource(
                            ByteBuffer.allocate(padSizeBeforeSigningBlock)));
        }
        return Pair.of(beforeCentralDir, padSizeBeforeSigningBlock);
    }

    public static DataSource copyWithModifiedCDOffset(
            DataSource beforeCentralDir, DataSource eocd) throws IOException {

        // Ensure that, when digesting, ZIP End of Central Directory record's Central Directory
        // offset field is treated as pointing to the offset at which the APK Signing Block will
        // start.
        long centralDirOffsetForDigesting = beforeCentralDir.size();
        ByteBuffer eocdBuf = ByteBuffer.allocate((int) eocd.size());
        eocdBuf.order(ByteOrder.LITTLE_ENDIAN);
        eocd.copyTo(0, (int) eocd.size(), eocdBuf);
        eocdBuf.flip();
        ZipUtils.setZipEocdCentralDirectoryOffset(eocdBuf, centralDirOffsetForDigesting);
        return DataSources.asDataSource(eocdBuf);
    }

    public static byte[] generateApkSigningBlock(
            List<Pair<byte[], Integer>> apkSignatureSchemeBlockPairs) {
        // FORMAT:
        // uint64:  size (excluding this field)
        // repeated ID-value pairs:
        //     uint64:           size (excluding this field)
        //     uint32:           ID
        //     (size - 4) bytes: value
        // (extra verity ID-value for padding to make block size a multiple of 4096 bytes)
        // uint64:  size (same as the one above)
        // uint128: magic

        int blocksSize = 0;
        for (Pair<byte[], Integer> schemeBlockPair : apkSignatureSchemeBlockPairs) {
            blocksSize += 8 + 4 + schemeBlockPair.getFirst().length; // size + id + value
        }

        int resultSize =
                8 // size
                + blocksSize
                + 8 // size
                + 16 // magic
                ;
        ByteBuffer paddingPair = null;
        if (resultSize % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES != 0) {
            int padding = ANDROID_COMMON_PAGE_ALIGNMENT_BYTES -
                    (resultSize % ANDROID_COMMON_PAGE_ALIGNMENT_BYTES);
            if (padding < 12) {  // minimum size of an ID-value pair
                padding += ANDROID_COMMON_PAGE_ALIGNMENT_BYTES;
            }
            paddingPair = ByteBuffer.allocate(padding).order(ByteOrder.LITTLE_ENDIAN);
            paddingPair.putLong(padding - 8);
            paddingPair.putInt(VERITY_PADDING_BLOCK_ID);
            paddingPair.rewind();
            resultSize += padding;
        }

        ByteBuffer result = ByteBuffer.allocate(resultSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        long blockSizeFieldValue = resultSize - 8L;
        result.putLong(blockSizeFieldValue);

        for (Pair<byte[], Integer> schemeBlockPair : apkSignatureSchemeBlockPairs) {
            byte[] apkSignatureSchemeBlock = schemeBlockPair.getFirst();
            int apkSignatureSchemeId = schemeBlockPair.getSecond();
            long pairSizeFieldValue = 4L + apkSignatureSchemeBlock.length;
            result.putLong(pairSizeFieldValue);
            result.putInt(apkSignatureSchemeId);
            result.put(apkSignatureSchemeBlock);
        }

        if (paddingPair != null) {
            result.put(paddingPair);
        }

        result.putLong(blockSizeFieldValue);
        result.put(APK_SIGNING_BLOCK_MAGIC);

        return result.array();
    }

    /**
     * Returns the individual APK signature blocks within the provided {@code apkSigningBlock} in a
     * {@code List} of {@code Pair} instances where the first element in the {@code Pair} is the
     * contents / value of the signature block and the second element is the ID of the block.
     *
     * @throws IOException if an error is encountered reading the provided {@code apkSigningBlock}
     */
    public static List<Pair<byte[], Integer>> getApkSignatureBlocks(
            DataSource apkSigningBlock) throws IOException {
        // FORMAT:
        // uint64:  size (excluding this field)
        // repeated ID-value pairs:
        //     uint64:           size (excluding this field)
        //     uint32:           ID
        //     (size - 4) bytes: value
        // (extra verity ID-value for padding to make block size a multiple of 4096 bytes)
        // uint64:  size (same as the one above)
        // uint128: magic
        long apkSigningBlockSize = apkSigningBlock.size();
        if (apkSigningBlock.size() > Integer.MAX_VALUE || apkSigningBlockSize < 32) {
            throw new IllegalArgumentException(
                    "APK signing block size out of range: " + apkSigningBlockSize);
        }
        // Remove the header and footer from the signing block to iterate over only the repeated
        // ID-value pairs.
        ByteBuffer apkSigningBlockBuffer = apkSigningBlock.getByteBuffer(8,
                (int) apkSigningBlock.size() - 32);
        apkSigningBlockBuffer.order(ByteOrder.LITTLE_ENDIAN);
        List<Pair<byte[], Integer>> signatureBlocks = new ArrayList<>();
        while (apkSigningBlockBuffer.hasRemaining()) {
            long blockLength = apkSigningBlockBuffer.getLong();
            if (blockLength > Integer.MAX_VALUE || blockLength < 4) {
                throw new IllegalArgumentException(
                        "Block index " + (signatureBlocks.size() + 1) + " size out of range: "
                                + blockLength);
            }
            int blockId = apkSigningBlockBuffer.getInt();
            // Since the block ID has already been read from the signature block read the next
            // blockLength - 4 bytes as the value.
            byte[] blockValue = new byte[(int) blockLength - 4];
            apkSigningBlockBuffer.get(blockValue);
            signatureBlocks.add(Pair.of(blockValue, blockId));
        }
        return signatureBlocks;
    }

    /**
     * Returns the individual APK signers within the provided {@code signatureBlock} in a {@code
     * List} of {@code Pair} instances where the first element is a {@code List} of {@link
     * X509Certificate}s and the second element is a byte array of the individual signer's block.
     *
     * <p>This method supports any signature block that adheres to the following format up to the
     * signing certificate(s):
     * <pre>
     * * length-prefixed sequence of length-prefixed signers
     *   * length-prefixed signed data
     *     * length-prefixed sequence of length-prefixed digests:
     *       * uint32: signature algorithm ID
     *       * length-prefixed bytes: digest of contents
     *     * length-prefixed sequence of certificates:
     *       * length-prefixed bytes: X.509 certificate (ASN.1 DER encoded).
     * </pre>
     *
     * <p>Note, this is a convenience method to obtain any signers from an existing signature block;
     * the signature of each signer will not be verified.
     *
     * @throws ApkFormatException if an error is encountered while parsing the provided {@code
     * signatureBlock}
     * @throws CertificateException if the signing certificate(s) within an individual signer block
     * cannot be parsed
     */
    public static List<Pair<List<X509Certificate>, byte[]>> getApkSignatureBlockSigners(
            byte[] signatureBlock) throws ApkFormatException, CertificateException {
        ByteBuffer signatureBlockBuffer = ByteBuffer.wrap(signatureBlock);
        signatureBlockBuffer.order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer signersBuffer = getLengthPrefixedSlice(signatureBlockBuffer);
        List<Pair<List<X509Certificate>, byte[]>> signers = new ArrayList<>();
        while (signersBuffer.hasRemaining()) {
            // Parse the next signer block, save all of its bytes for the resulting List, and
            // rewind the buffer to allow the signing certificate(s) to be parsed.
            ByteBuffer signer = getLengthPrefixedSlice(signersBuffer);
            byte[] signerBytes = new byte[signer.remaining()];
            signer.get(signerBytes);
            signer.rewind();

            ByteBuffer signedData = getLengthPrefixedSlice(signer);
            // The first length prefixed slice is the sequence of digests which are not required
            // when obtaining the signing certificate(s).
            getLengthPrefixedSlice(signedData);
            ByteBuffer certificatesBuffer = getLengthPrefixedSlice(signedData);
            List<X509Certificate> certificates = new ArrayList<>();
            while (certificatesBuffer.hasRemaining()) {
                int certLength = certificatesBuffer.getInt();
                byte[] certBytes = new byte[certLength];
                if (certLength > certificatesBuffer.remaining()) {
                    throw new IllegalArgumentException(
                            "Cert index " + (certificates.size() + 1) + " under signer index "
                                    + (signers.size() + 1) + " size out of range: " + certLength);
                }
                certificatesBuffer.get(certBytes);
                GuaranteedEncodedFormX509Certificate signerCert =
                        new GuaranteedEncodedFormX509Certificate(
                                X509CertificateUtils.generateCertificate(certBytes), certBytes);
                certificates.add(signerCert);
            }
            signers.add(Pair.of(certificates, signerBytes));
        }
        return signers;
    }

    /**
     * Computes the digests of the given APK components according to the algorithms specified in the
     * given SignerConfigs.
     *
     * @param signerConfigs signer configurations, one for each signer At least one signer config
     *        must be provided.
     *
     * @throws IOException if an I/O error occurs
     * @throws NoSuchAlgorithmException if a required cryptographic algorithm implementation is
     *         missing
     * @throws SignatureException if an error occurs when computing digests of generating
     *         signatures
     */
    public static Pair<List<SignerConfig>, Map<ContentDigestAlgorithm, byte[]>>
            computeContentDigests(
                    RunnablesExecutor executor,
                    DataSource beforeCentralDir,
                    DataSource centralDir,
                    DataSource eocd,
                    List<SignerConfig> signerConfigs)
                            throws IOException, NoSuchAlgorithmException, SignatureException {
        if (signerConfigs.isEmpty()) {
            throw new IllegalArgumentException(
                    "No signer configs provided. At least one is required");
        }

        // Figure out which digest(s) to use for APK contents.
        Set<ContentDigestAlgorithm> contentDigestAlgorithms = new HashSet<>(1);
        for (SignerConfig signerConfig : signerConfigs) {
            for (SignatureAlgorithm signatureAlgorithm : signerConfig.signatureAlgorithms) {
                contentDigestAlgorithms.add(signatureAlgorithm.getContentDigestAlgorithm());
            }
        }

        // Compute digests of APK contents.
        Map<ContentDigestAlgorithm, byte[]> contentDigests; // digest algorithm ID -> digest
        try {
            contentDigests =
                    computeContentDigests(
                            executor,
                            contentDigestAlgorithms,
                            beforeCentralDir,
                            centralDir,
                            eocd);
        } catch (IOException e) {
            throw new IOException("Failed to read APK being signed", e);
        } catch (DigestException e) {
            throw new SignatureException("Failed to compute digests of APK", e);
        }

        // Sign the digests and wrap the signatures and signer info into an APK Signing Block.
        return Pair.of(signerConfigs, contentDigests);
    }

    /**
     * Returns the subset of signatures which are expected to be verified by at least one Android
     * platform version in the {@code [minSdkVersion, maxSdkVersion]} range. The returned result is
     * guaranteed to contain at least one signature.
     *
     * <p>Each Android platform version typically verifies exactly one signature from the provided
     * {@code signatures} set. This method returns the set of these signatures collected over all
     * requested platform versions. As a result, the result may contain more than one signature.
     *
     * @throws NoSupportedSignaturesException if no supported signatures were
     *         found for an Android platform version in the range.
     */
    public static <T extends ApkSupportedSignature> List<T> getSignaturesToVerify(
            List<T> signatures, int minSdkVersion, int maxSdkVersion)
            throws NoSupportedSignaturesException {
        return getSignaturesToVerify(signatures, minSdkVersion, maxSdkVersion, false);
    }

    /**
     * Returns the subset of signatures which are expected to be verified by at least one Android
     * platform version in the {@code [minSdkVersion, maxSdkVersion]} range. The returned result is
     * guaranteed to contain at least one signature.
     *
     * <p>{@code onlyRequireJcaSupport} can be set to true for cases that only require verifying a
     * signature within the signing block using the standard JCA.
     *
     * <p>Each Android platform version typically verifies exactly one signature from the provided
     * {@code signatures} set. This method returns the set of these signatures collected over all
     * requested platform versions. As a result, the result may contain more than one signature.
     *
     * @throws NoSupportedSignaturesException if no supported signatures were
     *         found for an Android platform version in the range.
     */
    public static <T extends ApkSupportedSignature> List<T> getSignaturesToVerify(
            List<T> signatures, int minSdkVersion, int maxSdkVersion,
            boolean onlyRequireJcaSupport) throws NoSupportedSignaturesException {
        try {
            return ApkSigningBlockUtilsLite.getSignaturesToVerify(signatures, minSdkVersion,
                    maxSdkVersion, onlyRequireJcaSupport);
        } catch (NoApkSupportedSignaturesException e) {
            throw new NoSupportedSignaturesException(e.getMessage());
        }
    }

    public static class NoSupportedSignaturesException extends NoApkSupportedSignaturesException {
        public NoSupportedSignaturesException(String message) {
            super(message);
        }
    }

    public static class SignatureNotFoundException extends Exception {
        private static final long serialVersionUID = 1L;

        public SignatureNotFoundException(String message) {
            super(message);
        }

        public SignatureNotFoundException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * uses the SignatureAlgorithms in the provided signerConfig to sign the provided data
     *
     * @return list of signature algorithm IDs and their corresponding signatures over the data.
     */
    public static List<Pair<Integer, byte[]>> generateSignaturesOverData(
            SignerConfig signerConfig, byte[] data)
                    throws InvalidKeyException, NoSuchAlgorithmException, SignatureException {
        List<Pair<Integer, byte[]>> signatures =
                new ArrayList<>(signerConfig.signatureAlgorithms.size());
        PublicKey publicKey = signerConfig.certificates.get(0).getPublicKey();
        for (SignatureAlgorithm signatureAlgorithm : signerConfig.signatureAlgorithms) {
            Pair<String, ? extends AlgorithmParameterSpec> sigAlgAndParams =
                    signatureAlgorithm.getJcaSignatureAlgorithmAndParams();
            String jcaSignatureAlgorithm = sigAlgAndParams.getFirst();
            AlgorithmParameterSpec jcaSignatureAlgorithmParams = sigAlgAndParams.getSecond();
            byte[] signatureBytes;
            try {
                Signature signature = Signature.getInstance(jcaSignatureAlgorithm);
                signature.initSign(signerConfig.privateKey);
                if (jcaSignatureAlgorithmParams != null) {
                    signature.setParameter(jcaSignatureAlgorithmParams);
                }
                signature.update(data);
                signatureBytes = signature.sign();
            } catch (InvalidKeyException e) {
                throw new InvalidKeyException("Failed to sign using " + jcaSignatureAlgorithm, e);
            } catch (InvalidAlgorithmParameterException | SignatureException e) {
                throw new SignatureException("Failed to sign using " + jcaSignatureAlgorithm, e);
            }

            try {
                Signature signature = Signature.getInstance(jcaSignatureAlgorithm);
                signature.initVerify(publicKey);
                if (jcaSignatureAlgorithmParams != null) {
                    signature.setParameter(jcaSignatureAlgorithmParams);
                }
                signature.update(data);
                if (!signature.verify(signatureBytes)) {
                    throw new SignatureException("Failed to verify generated "
                            + jcaSignatureAlgorithm
                            + " signature using public key from certificate");
                }
            } catch (InvalidKeyException e) {
                throw new InvalidKeyException(
                        "Failed to verify generated " + jcaSignatureAlgorithm + " signature using"
                                + " public key from certificate", e);
            } catch (InvalidAlgorithmParameterException | SignatureException e) {
                throw new SignatureException(
                        "Failed to verify generated " + jcaSignatureAlgorithm + " signature using"
                                + " public key from certificate", e);
            }

            signatures.add(Pair.of(signatureAlgorithm.getId(), signatureBytes));
        }
        return signatures;
    }

    /**
     * Wrap the signature according to CMS PKCS #7 RFC 5652.
     * The high-level simplified structure is as follows:
     * // ContentInfo
     *     //   digestAlgorithm
     *     //   SignedData
     *     //     bag of certificates
     *     //     SignerInfo
     *     //       signing cert issuer and serial number (for locating the cert in the above bag)
     *     //       digestAlgorithm
     *     //       signatureAlgorithm
     *     //       signature
     *
     * @throws Asn1EncodingException if the ASN.1 structure could not be encoded
     */
    public static byte[] generatePkcs7DerEncodedMessage(
            byte[] signatureBytes, ByteBuffer data, List<X509Certificate> signerCerts,
            AlgorithmIdentifier digestAlgorithmId, AlgorithmIdentifier signatureAlgorithmId)
            throws Asn1EncodingException, CertificateEncodingException {
        SignerInfo signerInfo = new SignerInfo();
        signerInfo.version = 1;
        X509Certificate signingCert = signerCerts.get(0);
        X500Principal signerCertIssuer = signingCert.getIssuerX500Principal();
        signerInfo.sid =
                new SignerIdentifier(
                        new IssuerAndSerialNumber(
                                new Asn1OpaqueObject(signerCertIssuer.getEncoded()),
                                signingCert.getSerialNumber()));

        signerInfo.digestAlgorithm = digestAlgorithmId;
        signerInfo.signatureAlgorithm = signatureAlgorithmId;
        signerInfo.signature = ByteBuffer.wrap(signatureBytes);

        SignedData signedData = new SignedData();
        signedData.certificates = new ArrayList<>(signerCerts.size());
        for (X509Certificate cert : signerCerts) {
            signedData.certificates.add(new Asn1OpaqueObject(cert.getEncoded()));
        }
        signedData.version = 1;
        signedData.digestAlgorithms = Collections.singletonList(digestAlgorithmId);
        signedData.encapContentInfo = new EncapsulatedContentInfo(Pkcs7Constants.OID_DATA);
        // If data is not null, data will be embedded as is in the result -- an attached pcsk7
        signedData.encapContentInfo.content = data;
        signedData.signerInfos = Collections.singletonList(signerInfo);
        ContentInfo contentInfo = new ContentInfo();
        contentInfo.contentType = Pkcs7Constants.OID_SIGNED_DATA;
        contentInfo.content = new Asn1OpaqueObject(Asn1DerEncoder.encode(signedData));
        return Asn1DerEncoder.encode(contentInfo);
    }

    /**
     * Picks the correct v2/v3 digest for v4 signature verification.
     *
     * Keep in sync with pickBestDigestForV4 in framework's ApkSigningBlockUtils.
     */
    public static byte[] pickBestDigestForV4(Map<ContentDigestAlgorithm, byte[]> contentDigests) {
        for (ContentDigestAlgorithm algo : V4_CONTENT_DIGEST_ALGORITHMS) {
            if (contentDigests.containsKey(algo)) {
                return contentDigests.get(algo);
            }
        }
        return null;
    }

    /**
     * Signer configuration.
     */
    public static class SignerConfig {
        /** Private key. */
        public PrivateKey privateKey;

        /**
         * Certificates, with the first certificate containing the public key corresponding to
         * {@link #privateKey}.
         */
        public List<X509Certificate> certificates;

        /**
         * List of signature algorithms with which to sign.
         */
        public List<SignatureAlgorithm> signatureAlgorithms;

        public int minSdkVersion;
        public int maxSdkVersion;
        public boolean signerTargetsDevRelease;
        public SigningCertificateLineage signingCertificateLineage;
    }

    public static class Result extends ApkSigResult {
        public SigningCertificateLineage signingCertificateLineage = null;
        public final List<Result.SignerInfo> signers = new ArrayList<>();
        private final List<ApkVerifier.IssueWithParams> mWarnings = new ArrayList<>();
        private final List<ApkVerifier.IssueWithParams> mErrors = new ArrayList<>();

        public Result(int signatureSchemeVersion) {
            super(signatureSchemeVersion);
        }

        public boolean containsErrors() {
            if (!mErrors.isEmpty()) {
                return true;
            }
            if (!signers.isEmpty()) {
                for (Result.SignerInfo signer : signers) {
                    if (signer.containsErrors()) {
                        return true;
                    }
                }
            }
            return false;
        }

        public boolean containsWarnings() {
            if (!mWarnings.isEmpty()) {
                return true;
            }
            if (!signers.isEmpty()) {
                for (Result.SignerInfo signer : signers) {
                    if (signer.containsWarnings()) {
                        return true;
                    }
                }
            }
            return false;
        }

        public void addError(ApkVerifier.Issue msg, Object... parameters) {
            mErrors.add(new ApkVerifier.IssueWithParams(msg, parameters));
        }

        public void addWarning(ApkVerifier.Issue msg, Object... parameters) {
            mWarnings.add(new ApkVerifier.IssueWithParams(msg, parameters));
        }

        @Override
        public List<ApkVerifier.IssueWithParams> getErrors() {
            return mErrors;
        }

        @Override
        public List<ApkVerifier.IssueWithParams> getWarnings() {
            return mWarnings;
        }

        public static class SignerInfo extends ApkSignerInfo {
            public List<ContentDigest> contentDigests = new ArrayList<>();
            public Map<ContentDigestAlgorithm, byte[]> verifiedContentDigests = new HashMap<>();
            public List<Signature> signatures = new ArrayList<>();
            public Map<SignatureAlgorithm, byte[]> verifiedSignatures = new HashMap<>();
            public List<AdditionalAttribute> additionalAttributes = new ArrayList<>();
            public byte[] signedData;
            public int minSdkVersion;
            public int maxSdkVersion;
            public SigningCertificateLineage signingCertificateLineage;

            private final List<ApkVerifier.IssueWithParams> mWarnings = new ArrayList<>();
            private final List<ApkVerifier.IssueWithParams> mErrors = new ArrayList<>();

            public void addError(ApkVerifier.Issue msg, Object... parameters) {
                mErrors.add(new ApkVerifier.IssueWithParams(msg, parameters));
            }

            public void addWarning(ApkVerifier.Issue msg, Object... parameters) {
                mWarnings.add(new ApkVerifier.IssueWithParams(msg, parameters));
            }

            public boolean containsErrors() {
                return !mErrors.isEmpty();
            }

            public boolean containsWarnings() {
                return !mWarnings.isEmpty();
            }

            public List<ApkVerifier.IssueWithParams> getErrors() {
                return mErrors;
            }

            public List<ApkVerifier.IssueWithParams> getWarnings() {
                return mWarnings;
            }

            public static class ContentDigest {
                private final int mSignatureAlgorithmId;
                private final byte[] mValue;

                public ContentDigest(int signatureAlgorithmId, byte[] value) {
                    mSignatureAlgorithmId  = signatureAlgorithmId;
                    mValue = value;
                }

                public int getSignatureAlgorithmId() {
                    return mSignatureAlgorithmId;
                }

                public byte[] getValue() {
                    return mValue;
                }
            }

            public static class Signature {
                private final int mAlgorithmId;
                private final byte[] mValue;

                public Signature(int algorithmId, byte[] value) {
                    mAlgorithmId  = algorithmId;
                    mValue = value;
                }

                public int getAlgorithmId() {
                    return mAlgorithmId;
                }

                public byte[] getValue() {
                    return mValue;
                }
            }

            public static class AdditionalAttribute {
                private final int mId;
                private final byte[] mValue;

                public AdditionalAttribute(int id, byte[] value) {
                    mId  = id;
                    mValue = value.clone();
                }

                public int getId() {
                    return mId;
                }

                public byte[] getValue() {
                    return mValue.clone();
                }
            }
        }
    }

    public static class SupportedSignature extends ApkSupportedSignature {
        public SupportedSignature(SignatureAlgorithm algorithm, byte[] signature) {
            super(algorithm, signature);
        }
    }

    public static class SigningSchemeBlockAndDigests {
        public final Pair<byte[], Integer> signingSchemeBlock;
        public final Map<ContentDigestAlgorithm, byte[]> digestInfo;

        public SigningSchemeBlockAndDigests(
                Pair<byte[], Integer> signingSchemeBlock,
                Map<ContentDigestAlgorithm, byte[]> digestInfo) {
            this.signingSchemeBlock = signingSchemeBlock;
            this.digestInfo = digestInfo;
        }
    }
}
