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

package com.android.apksig.internal.zip;

import com.android.apksig.internal.util.ByteBufferSink;
import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import com.android.apksig.zip.ZipFormatException;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * ZIP Local File record.
 *
 * <p>The record consists of the Local File Header, file data, and (if present) Data Descriptor.
 */
public class LocalFileRecord {
    private static final int RECORD_SIGNATURE = 0x04034b50;
    private static final int HEADER_SIZE_BYTES = 30;

    private static final int GP_FLAGS_OFFSET = 6;
    private static final int CRC32_OFFSET = 14;
    private static final int COMPRESSED_SIZE_OFFSET = 18;
    private static final int UNCOMPRESSED_SIZE_OFFSET = 22;
    private static final int NAME_LENGTH_OFFSET = 26;
    private static final int EXTRA_LENGTH_OFFSET = 28;
    private static final int NAME_OFFSET = HEADER_SIZE_BYTES;

    private static final int DATA_DESCRIPTOR_SIZE_BYTES_WITHOUT_SIGNATURE = 12;
    private static final int DATA_DESCRIPTOR_SIGNATURE = 0x08074b50;

    private final String mName;
    private final int mNameSizeBytes;
    private final ByteBuffer mExtra;

    private final long mStartOffsetInArchive;
    private final long mSize;

    private final int mDataStartOffset;
    private final long mDataSize;
    private final boolean mDataCompressed;
    private final long mUncompressedDataSize;

    private LocalFileRecord(
            String name,
            int nameSizeBytes,
            ByteBuffer extra,
            long startOffsetInArchive,
            long size,
            int dataStartOffset,
            long dataSize,
            boolean dataCompressed,
            long uncompressedDataSize) {
        mName = name;
        mNameSizeBytes = nameSizeBytes;
        mExtra = extra;
        mStartOffsetInArchive = startOffsetInArchive;
        mSize = size;
        mDataStartOffset = dataStartOffset;
        mDataSize = dataSize;
        mDataCompressed = dataCompressed;
        mUncompressedDataSize = uncompressedDataSize;
    }

    public String getName() {
        return mName;
    }

    public ByteBuffer getExtra() {
        return (mExtra.capacity() > 0) ? mExtra.slice() : mExtra;
    }

    public int getExtraFieldStartOffsetInsideRecord() {
        return HEADER_SIZE_BYTES + mNameSizeBytes;
    }

    public long getStartOffsetInArchive() {
        return mStartOffsetInArchive;
    }

    public int getDataStartOffsetInRecord() {
        return mDataStartOffset;
    }

    /**
     * Returns the size (in bytes) of this record.
     */
    public long getSize() {
        return mSize;
    }

    /**
     * Returns {@code true} if this record's file data is stored in compressed form.
     */
    public boolean isDataCompressed() {
        return mDataCompressed;
    }

    /**
     * Returns the Local File record starting at the current position of the provided buffer
     * and advances the buffer's position immediately past the end of the record. The record
     * consists of the Local File Header, data, and (if present) Data Descriptor.
     */
    public static LocalFileRecord getRecord(
            DataSource apk,
            CentralDirectoryRecord cdRecord,
            long cdStartOffset) throws ZipFormatException, IOException {
        return getRecord(
                apk,
                cdRecord,
                cdStartOffset,
                true, // obtain extra field contents
                true // include Data Descriptor (if present)
                );
    }

    /**
     * Returns the Local File record starting at the current position of the provided buffer
     * and advances the buffer's position immediately past the end of the record. The record
     * consists of the Local File Header, data, and (if present) Data Descriptor.
     */
    private static LocalFileRecord getRecord(
            DataSource apk,
            CentralDirectoryRecord cdRecord,
            long cdStartOffset,
            boolean extraFieldContentsNeeded,
            boolean dataDescriptorIncluded) throws ZipFormatException, IOException {
        // IMPLEMENTATION NOTE: This method attempts to mimic the behavior of Android platform
        // exhibited when reading an APK for the purposes of verifying its signatures.

        String entryName = cdRecord.getName();
        int cdRecordEntryNameSizeBytes = cdRecord.getNameSizeBytes();
        int headerSizeWithName = HEADER_SIZE_BYTES + cdRecordEntryNameSizeBytes;
        long headerStartOffset = cdRecord.getLocalFileHeaderOffset();
        long headerEndOffset = headerStartOffset + headerSizeWithName;
        if (headerEndOffset > cdStartOffset) {
            throw new ZipFormatException(
                    "Local File Header of " + entryName + " extends beyond start of Central"
                            + " Directory. LFH end: " + headerEndOffset
                            + ", CD start: " + cdStartOffset);
        }
        ByteBuffer header;
        try {
            header = apk.getByteBuffer(headerStartOffset, headerSizeWithName);
        } catch (IOException e) {
            throw new IOException("Failed to read Local File Header of " + entryName, e);
        }
        header.order(ByteOrder.LITTLE_ENDIAN);

        int recordSignature = header.getInt();
        if (recordSignature != RECORD_SIGNATURE) {
            throw new ZipFormatException(
                    "Not a Local File Header record for entry " + entryName + ". Signature: 0x"
                            + Long.toHexString(recordSignature & 0xffffffffL));
        }
        short gpFlags = header.getShort(GP_FLAGS_OFFSET);
        boolean dataDescriptorUsed = (gpFlags & ZipUtils.GP_FLAG_DATA_DESCRIPTOR_USED) != 0;
        boolean cdDataDescriptorUsed =
                (cdRecord.getGpFlags() & ZipUtils.GP_FLAG_DATA_DESCRIPTOR_USED) != 0;
        if (dataDescriptorUsed != cdDataDescriptorUsed) {
            throw new ZipFormatException(
                    "Data Descriptor presence mismatch between Local File Header and Central"
                            + " Directory for entry " + entryName
                            + ". LFH: " + dataDescriptorUsed + ", CD: " + cdDataDescriptorUsed);
        }
        long uncompressedDataCrc32FromCdRecord = cdRecord.getCrc32();
        long compressedDataSizeFromCdRecord = cdRecord.getCompressedSize();
        long uncompressedDataSizeFromCdRecord = cdRecord.getUncompressedSize();
        if (!dataDescriptorUsed) {
            long crc32 = ZipUtils.getUnsignedInt32(header, CRC32_OFFSET);
            if (crc32 != uncompressedDataCrc32FromCdRecord) {
                throw new ZipFormatException(
                        "CRC-32 mismatch between Local File Header and Central Directory for entry "
                                + entryName + ". LFH: " + crc32
                                + ", CD: " + uncompressedDataCrc32FromCdRecord);
            }
            long compressedSize = ZipUtils.getUnsignedInt32(header, COMPRESSED_SIZE_OFFSET);
            if (compressedSize != compressedDataSizeFromCdRecord) {
                throw new ZipFormatException(
                        "Compressed size mismatch between Local File Header and Central Directory"
                                + " for entry " + entryName + ". LFH: " + compressedSize
                                + ", CD: " + compressedDataSizeFromCdRecord);
            }
            long uncompressedSize = ZipUtils.getUnsignedInt32(header, UNCOMPRESSED_SIZE_OFFSET);
            if (uncompressedSize != uncompressedDataSizeFromCdRecord) {
                throw new ZipFormatException(
                        "Uncompressed size mismatch between Local File Header and Central Directory"
                                + " for entry " + entryName + ". LFH: " + uncompressedSize
                                + ", CD: " + uncompressedDataSizeFromCdRecord);
            }
        }
        int nameLength = ZipUtils.getUnsignedInt16(header, NAME_LENGTH_OFFSET);
        if (nameLength > cdRecordEntryNameSizeBytes) {
            throw new ZipFormatException(
                    "Name mismatch between Local File Header and Central Directory for entry"
                            + entryName + ". LFH: " + nameLength
                            + " bytes, CD: " + cdRecordEntryNameSizeBytes + " bytes");
        }
        String name = CentralDirectoryRecord.getName(header, NAME_OFFSET, nameLength);
        if (!entryName.equals(name)) {
            throw new ZipFormatException(
                    "Name mismatch between Local File Header and Central Directory. LFH: \""
                            + name + "\", CD: \"" + entryName + "\"");
        }
        int extraLength = ZipUtils.getUnsignedInt16(header, EXTRA_LENGTH_OFFSET);
        long dataStartOffset = headerStartOffset + HEADER_SIZE_BYTES + nameLength + extraLength;
        long dataSize;
        boolean compressed =
                (cdRecord.getCompressionMethod() != ZipUtils.COMPRESSION_METHOD_STORED);
        if (compressed) {
            dataSize = compressedDataSizeFromCdRecord;
        } else {
            dataSize = uncompressedDataSizeFromCdRecord;
        }
        long dataEndOffset = dataStartOffset + dataSize;
        if (dataEndOffset > cdStartOffset) {
            throw new ZipFormatException(
                    "Local File Header data of " + entryName + " overlaps with Central Directory"
                            + ". LFH data start: " + dataStartOffset
                            + ", LFH data end: " + dataEndOffset + ", CD start: " + cdStartOffset);
        }

        ByteBuffer extra = EMPTY_BYTE_BUFFER;
        if ((extraFieldContentsNeeded) && (extraLength > 0)) {
            extra = apk.getByteBuffer(
                    headerStartOffset + HEADER_SIZE_BYTES + nameLength, extraLength);
        }

        long recordEndOffset = dataEndOffset;
        // Include the Data Descriptor (if requested and present) into the record.
        if ((dataDescriptorIncluded) && ((gpFlags & ZipUtils.GP_FLAG_DATA_DESCRIPTOR_USED) != 0)) {
            // The record's data is supposed to be followed by the Data Descriptor. Unfortunately,
            // the descriptor's size is not known in advance because the spec lets the signature
            // field (the first four bytes) be omitted. Thus, there's no 100% reliable way to tell
            // how long the Data Descriptor record is. Most parsers (including Android) check
            // whether the first four bytes look like Data Descriptor record signature and, if so,
            // assume that it is indeed the record's signature. However, this is the wrong
            // conclusion if the record's CRC-32 (next field after the signature) has the same value
            // as the signature. In any case, we're doing what Android is doing.
            long dataDescriptorEndOffset =
                    dataEndOffset + DATA_DESCRIPTOR_SIZE_BYTES_WITHOUT_SIGNATURE;
            if (dataDescriptorEndOffset > cdStartOffset) {
                throw new ZipFormatException(
                        "Data Descriptor of " + entryName + " overlaps with Central Directory"
                                + ". Data Descriptor end: " + dataEndOffset
                                + ", CD start: " + cdStartOffset);
            }
            ByteBuffer dataDescriptorPotentialSig = apk.getByteBuffer(dataEndOffset, 4);
            dataDescriptorPotentialSig.order(ByteOrder.LITTLE_ENDIAN);
            if (dataDescriptorPotentialSig.getInt() == DATA_DESCRIPTOR_SIGNATURE) {
                dataDescriptorEndOffset += 4;
                if (dataDescriptorEndOffset > cdStartOffset) {
                    throw new ZipFormatException(
                            "Data Descriptor of " + entryName + " overlaps with Central Directory"
                                    + ". Data Descriptor end: " + dataEndOffset
                                    + ", CD start: " + cdStartOffset);
                }
            }
            recordEndOffset = dataDescriptorEndOffset;
        }

        long recordSize = recordEndOffset - headerStartOffset;
        int dataStartOffsetInRecord = HEADER_SIZE_BYTES + nameLength + extraLength;

        return new LocalFileRecord(
                entryName,
                cdRecordEntryNameSizeBytes,
                extra,
                headerStartOffset,
                recordSize,
                dataStartOffsetInRecord,
                dataSize,
                compressed,
                uncompressedDataSizeFromCdRecord);
    }

    /**
     * Outputs this record and returns returns the number of bytes output.
     */
    public long outputRecord(DataSource sourceApk, DataSink output) throws IOException {
        long size = getSize();
        sourceApk.feed(getStartOffsetInArchive(), size, output);
        return size;
    }

    /**
     * Outputs this record, replacing its extra field with the provided one, and returns returns the
     * number of bytes output.
     */
    public long outputRecordWithModifiedExtra(
            DataSource sourceApk,
            ByteBuffer extra,
            DataSink output) throws IOException {
        long recordStartOffsetInSource = getStartOffsetInArchive();
        int extraStartOffsetInRecord = getExtraFieldStartOffsetInsideRecord();
        int extraSizeBytes = extra.remaining();
        int headerSize = extraStartOffsetInRecord + extraSizeBytes;
        ByteBuffer header = ByteBuffer.allocate(headerSize);
        header.order(ByteOrder.LITTLE_ENDIAN);
        sourceApk.copyTo(recordStartOffsetInSource, extraStartOffsetInRecord, header);
        header.put(extra.slice());
        header.flip();
        ZipUtils.setUnsignedInt16(header, EXTRA_LENGTH_OFFSET, extraSizeBytes);

        long outputByteCount = header.remaining();
        output.consume(header);
        long remainingRecordSize = getSize() - mDataStartOffset;
        sourceApk.feed(recordStartOffsetInSource + mDataStartOffset, remainingRecordSize, output);
        outputByteCount += remainingRecordSize;
        return outputByteCount;
    }

    /**
     * Outputs the specified Local File Header record with its data and returns the number of bytes
     * output.
     */
    public static long outputRecordWithDeflateCompressedData(
            String name,
            int lastModifiedTime,
            int lastModifiedDate,
            byte[] compressedData,
            long crc32,
            long uncompressedSize,
            DataSink output) throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        int recordSize = HEADER_SIZE_BYTES + nameBytes.length;
        ByteBuffer result = ByteBuffer.allocate(recordSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(RECORD_SIGNATURE);
        ZipUtils.putUnsignedInt16(result,  0x14); // Minimum version needed to extract
        result.putShort(ZipUtils.GP_FLAG_EFS); // General purpose flag: UTF-8 encoded name
        result.putShort(ZipUtils.COMPRESSION_METHOD_DEFLATED);
        ZipUtils.putUnsignedInt16(result, lastModifiedTime);
        ZipUtils.putUnsignedInt16(result, lastModifiedDate);
        ZipUtils.putUnsignedInt32(result, crc32);
        ZipUtils.putUnsignedInt32(result, compressedData.length);
        ZipUtils.putUnsignedInt32(result, uncompressedSize);
        ZipUtils.putUnsignedInt16(result, nameBytes.length);
        ZipUtils.putUnsignedInt16(result, 0); // Extra field length
        result.put(nameBytes);
        if (result.hasRemaining()) {
            throw new RuntimeException("pos: " + result.position() + ", limit: " + result.limit());
        }
        result.flip();

        long outputByteCount = result.remaining();
        output.consume(result);
        outputByteCount += compressedData.length;
        output.consume(compressedData, 0, compressedData.length);
        return outputByteCount;
    }

    private static final ByteBuffer EMPTY_BYTE_BUFFER = ByteBuffer.allocate(0);

    /**
     * Sends uncompressed data of this record into the the provided data sink.
     */
    public void outputUncompressedData(
            DataSource lfhSection,
            DataSink sink) throws IOException, ZipFormatException {
        long dataStartOffsetInArchive = mStartOffsetInArchive + mDataStartOffset;
        try {
            if (mDataCompressed) {
                try (InflateSinkAdapter inflateAdapter = new InflateSinkAdapter(sink)) {
                    lfhSection.feed(dataStartOffsetInArchive, mDataSize, inflateAdapter);
                    long actualUncompressedSize = inflateAdapter.getOutputByteCount();
                    if (actualUncompressedSize != mUncompressedDataSize) {
                        throw new ZipFormatException(
                                "Unexpected size of uncompressed data of " + mName
                                        + ". Expected: " + mUncompressedDataSize + " bytes"
                                        + ", actual: " + actualUncompressedSize + " bytes");
                    }
                } catch (IOException e) {
                    if (e.getCause() instanceof DataFormatException) {
                        throw new ZipFormatException("Data of entry " + mName + " malformed", e);
                    }
                    throw e;
                }
            } else {
                lfhSection.feed(dataStartOffsetInArchive, mDataSize, sink);
                // No need to check whether output size is as expected because DataSource.feed is
                // guaranteed to output exactly the number of bytes requested.
            }
        } catch (IOException e) {
            throw new IOException(
                    "Failed to read data of " + ((mDataCompressed) ? "compressed" : "uncompressed")
                        + " entry " + mName,
                    e);
        }
        // Interestingly, Android doesn't check that uncompressed data's CRC-32 is as expected. We
        // thus don't check either.
    }

    /**
     * Sends uncompressed data pointed to by the provided ZIP Central Directory (CD) record into the
     * provided data sink.
     */
    public static void outputUncompressedData(
            DataSource source,
            CentralDirectoryRecord cdRecord,
            long cdStartOffsetInArchive,
            DataSink sink) throws ZipFormatException, IOException {
        // IMPLEMENTATION NOTE: This method attempts to mimic the behavior of Android platform
        // exhibited when reading an APK for the purposes of verifying its signatures.
        // When verifying an APK, Android doesn't care reading the extra field or the Data
        // Descriptor.
        LocalFileRecord lfhRecord =
                getRecord(
                        source,
                        cdRecord,
                        cdStartOffsetInArchive,
                        false, // don't care about the extra field
                        false // don't read the Data Descriptor
                        );
        lfhRecord.outputUncompressedData(source, sink);
    }

    /**
     * Returns the uncompressed data pointed to by the provided ZIP Central Directory (CD) record.
     */
    public static byte[] getUncompressedData(
            DataSource source,
            CentralDirectoryRecord cdRecord,
            long cdStartOffsetInArchive) throws ZipFormatException, IOException {
        if (cdRecord.getUncompressedSize() > Integer.MAX_VALUE) {
            throw new IOException(
                    cdRecord.getName() + " too large: " + cdRecord.getUncompressedSize());
        }
        byte[] result = null;
        try {
            result = new byte[(int) cdRecord.getUncompressedSize()];
        } catch (OutOfMemoryError e) {
            throw new IOException(
                    cdRecord.getName() + " too large: " + cdRecord.getUncompressedSize(), e);
        }
        ByteBuffer resultBuf = ByteBuffer.wrap(result);
        ByteBufferSink resultSink = new ByteBufferSink(resultBuf);
        outputUncompressedData(
                source,
                cdRecord,
                cdStartOffsetInArchive,
                resultSink);
        return result;
    }

    /**
     * {@link DataSink} which inflates received data and outputs the deflated data into the provided
     * delegate sink.
     */
    private static class InflateSinkAdapter implements DataSink, Closeable {
        private final DataSink mDelegate;

        private Inflater mInflater = new Inflater(true);
        private byte[] mOutputBuffer;
        private byte[] mInputBuffer;
        private long mOutputByteCount;
        private boolean mClosed;

        private InflateSinkAdapter(DataSink delegate) {
            mDelegate = delegate;
        }

        @Override
        public void consume(byte[] buf, int offset, int length) throws IOException {
            checkNotClosed();
            mInflater.setInput(buf, offset, length);
            if (mOutputBuffer == null) {
                mOutputBuffer = new byte[65536];
            }
            while (!mInflater.finished()) {
                int outputChunkSize;
                try {
                    outputChunkSize = mInflater.inflate(mOutputBuffer);
                } catch (DataFormatException e) {
                    throw new IOException("Failed to inflate data", e);
                }
                if (outputChunkSize == 0) {
                    return;
                }
                mDelegate.consume(mOutputBuffer, 0, outputChunkSize);
                mOutputByteCount += outputChunkSize;
            }
        }

        @Override
        public void consume(ByteBuffer buf) throws IOException {
            checkNotClosed();
            if (buf.hasArray()) {
                consume(buf.array(), buf.arrayOffset() + buf.position(), buf.remaining());
                buf.position(buf.limit());
            } else {
                if (mInputBuffer == null) {
                    mInputBuffer = new byte[65536];
                }
                while (buf.hasRemaining()) {
                    int chunkSize = Math.min(buf.remaining(), mInputBuffer.length);
                    buf.get(mInputBuffer, 0, chunkSize);
                    consume(mInputBuffer, 0, chunkSize);
                }
            }
        }

        public long getOutputByteCount() {
            return mOutputByteCount;
        }

        @Override
        public void close() throws IOException {
            mClosed = true;
            mInputBuffer = null;
            mOutputBuffer = null;
            if (mInflater != null) {
                mInflater.end();
                mInflater = null;
            }
        }

        private void checkNotClosed() {
            if (mClosed) {
                throw new IllegalStateException("Closed");
            }
        }
    }
}
