/*
 * Copyright (C) 2017 The Android Open Source Project
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

package com.android.apksig.internal.util;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.android.apksig.internal.zip.ZipUtils;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import com.android.apksig.util.DataSources;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Phaser;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * VerityTreeBuilder is used to generate the root hash of verity tree built from the input file.
 * The root hash can be used on device for on-access verification.  The tree itself is reproducible
 * on device, and is not shipped with the APK.
 */
public class VerityTreeBuilder implements AutoCloseable {

    /**
     * Maximum size (in bytes) of each node of the tree.
     */
    private final static int CHUNK_SIZE = 4096;
    /**
     * Maximum parallelism while calculating digests.
     */
    private final static int DIGEST_PARALLELISM = Math.min(32,
            Runtime.getRuntime().availableProcessors());
    /**
     * Queue size.
     */
    private final static int MAX_OUTSTANDING_CHUNKS = 4;
    /**
     * Typical prefetch size.
     */
    private final static int MAX_PREFETCH_CHUNKS = 1024;
    /**
     * Minimum chunks to be processed by a single worker task.
     */
    private final static int MIN_CHUNKS_PER_WORKER = 8;

    /**
     * Digest algorithm (JCA Digest algorithm name) used in the tree.
     */
    private final static String JCA_ALGORITHM = "SHA-256";

    /**
     * Optional salt to apply before each digestion.
     */
    private final byte[] mSalt;

    private final MessageDigest mMd;

    private final ExecutorService mExecutor =
            new ThreadPoolExecutor(DIGEST_PARALLELISM, DIGEST_PARALLELISM,
                    0L, MILLISECONDS,
                    new ArrayBlockingQueue<>(MAX_OUTSTANDING_CHUNKS),
                    new ThreadPoolExecutor.CallerRunsPolicy());

    public VerityTreeBuilder(byte[] salt) throws NoSuchAlgorithmException {
        mSalt = salt;
        mMd = getNewMessageDigest();
    }

    @Override
    public void close() {
        mExecutor.shutdownNow();
    }

    /**
     * Returns the root hash of the APK verity tree built from ZIP blocks.
     *
     * Specifically, APK verity tree is built from the APK, but as if the APK Signing Block (which
     * must be page aligned) and the "Central Directory offset" field in End of Central Directory
     * are skipped.
     */
    public byte[] generateVerityTreeRootHash(DataSource beforeApkSigningBlock,
            DataSource centralDir, DataSource eocd) throws IOException {
        if (beforeApkSigningBlock.size() % CHUNK_SIZE != 0) {
            throw new IllegalStateException("APK Signing Block size not a multiple of " + CHUNK_SIZE
                    + ": " + beforeApkSigningBlock.size());
        }

        // Ensure that, when digesting, ZIP End of Central Directory record's Central Directory
        // offset field is treated as pointing to the offset at which the APK Signing Block will
        // start.
        long centralDirOffsetForDigesting = beforeApkSigningBlock.size();
        ByteBuffer eocdBuf = ByteBuffer.allocate((int) eocd.size());
        eocdBuf.order(ByteOrder.LITTLE_ENDIAN);
        eocd.copyTo(0, (int) eocd.size(), eocdBuf);
        eocdBuf.flip();
        ZipUtils.setZipEocdCentralDirectoryOffset(eocdBuf, centralDirOffsetForDigesting);

        return generateVerityTreeRootHash(new ChainedDataSource(beforeApkSigningBlock, centralDir,
                    DataSources.asDataSource(eocdBuf)));
    }

    /**
     * Returns the root hash of the verity tree built from the data source.
     */
    public byte[] generateVerityTreeRootHash(DataSource fileSource) throws IOException {
        ByteBuffer verityBuffer = generateVerityTree(fileSource);
        return getRootHashFromTree(verityBuffer);
    }

    /**
     * Returns the byte buffer that contains the whole verity tree.
     *
     * The tree is built bottom up. The bottom level has 256-bit digest for each 4 KB block in the
     * input file.  If the total size is larger than 4 KB, take this level as input and repeat the
     * same procedure, until the level is within 4 KB.  If salt is given, it will apply to each
     * digestion before the actual data.
     *
     * The returned root hash is calculated from the last level of 4 KB chunk, similarly with salt.
     *
     * The tree is currently stored only in memory and is never written out.  Nevertheless, it is
     * the actual verity tree format on disk, and is supposed to be re-generated on device.
     */
    public ByteBuffer generateVerityTree(DataSource fileSource) throws IOException {
        int digestSize = mMd.getDigestLength();

        // Calculate the summed area table of level size. In other word, this is the offset
        // table of each level, plus the next non-existing level.
        int[] levelOffset = calculateLevelOffset(fileSource.size(), digestSize);

        ByteBuffer verityBuffer = ByteBuffer.allocate(levelOffset[levelOffset.length - 1]);

        // Generate the hash tree bottom-up.
        for (int i = levelOffset.length - 2; i >= 0; i--) {
            DataSink middleBufferSink = new ByteBufferSink(
                    slice(verityBuffer, levelOffset[i], levelOffset[i + 1]));
            DataSource src;
            if (i == levelOffset.length - 2) {
                src = fileSource;
                digestDataByChunks(src, middleBufferSink);
            } else {
                src = DataSources.asDataSource(slice(verityBuffer.asReadOnlyBuffer(),
                        levelOffset[i + 1], levelOffset[i + 2]));
                digestDataByChunks(src, middleBufferSink);
            }

            // If the output is not full chunk, pad with 0s.
            long totalOutput = divideRoundup(src.size(), CHUNK_SIZE) * digestSize;
            int incomplete = (int) (totalOutput % CHUNK_SIZE);
            if (incomplete > 0) {
                byte[] padding = new byte[CHUNK_SIZE - incomplete];
                middleBufferSink.consume(padding, 0, padding.length);
            }
        }
        return verityBuffer;
    }

    /**
     * Returns the digested root hash from the top level (only page) of a verity tree.
     */
    public byte[] getRootHashFromTree(ByteBuffer verityBuffer) throws IOException {
        ByteBuffer firstPage = slice(verityBuffer.asReadOnlyBuffer(), 0, CHUNK_SIZE);
        return saltedDigest(firstPage);
    }

    /**
     * Returns an array of summed area table of level size in the verity tree.  In other words, the
     * returned array is offset of each level in the verity tree file format, plus an additional
     * offset of the next non-existing level (i.e. end of the last level + 1).  Thus the array size
     * is level + 1.
     */
    private static int[] calculateLevelOffset(long dataSize, int digestSize) {
        // Compute total size of each level, bottom to top.
        ArrayList<Long> levelSize = new ArrayList<>();
        while (true) {
            long chunkCount = divideRoundup(dataSize, CHUNK_SIZE);
            long size = CHUNK_SIZE * divideRoundup(chunkCount * digestSize, CHUNK_SIZE);
            levelSize.add(size);
            if (chunkCount * digestSize <= CHUNK_SIZE) {
                break;
            }
            dataSize = chunkCount * digestSize;
        }

        // Reverse and convert to summed area table.
        int[] levelOffset = new int[levelSize.size() + 1];
        levelOffset[0] = 0;
        for (int i = 0; i < levelSize.size(); i++) {
            // We don't support verity tree if it is larger then Integer.MAX_VALUE.
            levelOffset[i + 1] = levelOffset[i] + Math.toIntExact(
                    levelSize.get(levelSize.size() - i - 1));
        }
        return levelOffset;
    }

    /**
     * Digest data source by chunks then feeds them to the sink one by one.  If the last unit is
     * less than the chunk size and padding is desired, feed with extra padding 0 to fill up the
     * chunk before digesting.
     */
    private void digestDataByChunks(DataSource dataSource, DataSink dataSink) throws IOException {
        final long size = dataSource.size();
        final int chunks = (int) divideRoundup(size, CHUNK_SIZE);

        /** Single IO operation size, in chunks. */
        final int ioSizeChunks = MAX_PREFETCH_CHUNKS;

        final byte[][] hashes = new byte[chunks][];

        Phaser tasks = new Phaser(1);

        // Reading the input file as fast as we can.
        final long maxReadSize = ioSizeChunks * CHUNK_SIZE;

        long readOffset = 0;
        int startChunkIndex = 0;
        while (readOffset < size) {
            final long readLimit = Math.min(readOffset + maxReadSize, size);
            final int readSize = (int) (readLimit - readOffset);
            final int bufferSizeChunks = (int) divideRoundup(readSize, CHUNK_SIZE);

            // Overllocating to zero-pad last chunk.
            // With 4MiB block size, 32 threads and 4 queue size we might allocate up to 144MiB.
            final ByteBuffer buffer = ByteBuffer.allocate(bufferSizeChunks * CHUNK_SIZE);
            dataSource.copyTo(readOffset, readSize, buffer);
            buffer.rewind();

            final int readChunkIndex = startChunkIndex;
            Runnable task = () -> {
                final MessageDigest md = cloneMessageDigest();
                for (int offset = 0, finish = buffer.capacity(), chunkIndex = readChunkIndex;
                        offset < finish; offset += CHUNK_SIZE, ++chunkIndex) {
                    ByteBuffer chunk = slice(buffer, offset, offset + CHUNK_SIZE);
                    hashes[chunkIndex] = saltedDigest(md, chunk);
                }
                tasks.arriveAndDeregister();
            };
            tasks.register();
            mExecutor.execute(task);

            startChunkIndex += bufferSizeChunks;
            readOffset += readSize;
        }

        // Waiting for the tasks to complete.
        tasks.arriveAndAwaitAdvance();

        // Streaming hashes back.
        for (byte[] hash : hashes) {
            dataSink.consume(hash, 0, hash.length);
        }
    }

    /** Returns the digest of data with salt prepended. */
    private byte[] saltedDigest(ByteBuffer data) {
        return saltedDigest(mMd, data);
    }

    private byte[] saltedDigest(MessageDigest md, ByteBuffer data) {
        md.reset();
        if (mSalt != null) {
            md.update(mSalt);
        }
        md.update(data);
        return md.digest();
    }

    /** Divides a number and round up to the closest integer. */
    private static long divideRoundup(long dividend, long divisor) {
        return (dividend + divisor - 1) / divisor;
    }

    /** Returns a slice of the buffer with shared the content. */
    private static ByteBuffer slice(ByteBuffer buffer, int begin, int end) {
        ByteBuffer b = buffer.duplicate();
        b.position(0);  // to ensure position <= limit invariant.
        b.limit(end);
        b.position(begin);
        return b.slice();
    }

    /**
     * Obtains a new instance of the message digest algorithm.
     */
    private static MessageDigest getNewMessageDigest() throws NoSuchAlgorithmException {
        return MessageDigest.getInstance(JCA_ALGORITHM);
    }

    /**
     * Clones the existing message digest, or creates a new instance if clone is unavailable.
     */
    private MessageDigest cloneMessageDigest() {
        try {
            return (MessageDigest) mMd.clone();
        } catch (CloneNotSupportedException ignored) {
            try {
                return getNewMessageDigest();
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException(
                        "Failed to obtain an instance of a previously available message digest", e);
            }
        }
    }
}
