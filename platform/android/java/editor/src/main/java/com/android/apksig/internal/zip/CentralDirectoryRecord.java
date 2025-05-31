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

import com.android.apksig.zip.ZipFormatException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;

/**
 * ZIP Central Directory (CD) Record.
 */
public class CentralDirectoryRecord {

    /**
     * Comparator which compares records by the offset of the corresponding Local File Header in the
     * archive.
     */
    public static final Comparator<CentralDirectoryRecord> BY_LOCAL_FILE_HEADER_OFFSET_COMPARATOR =
            new ByLocalFileHeaderOffsetComparator();

    private static final int RECORD_SIGNATURE = 0x02014b50;
    private static final int HEADER_SIZE_BYTES = 46;

    private static final int GP_FLAGS_OFFSET = 8;
    private static final int LOCAL_FILE_HEADER_OFFSET_OFFSET = 42;
    private static final int NAME_OFFSET = HEADER_SIZE_BYTES;

    private final ByteBuffer mData;
    private final short mGpFlags;
    private final short mCompressionMethod;
    private final int mLastModificationTime;
    private final int mLastModificationDate;
    private final long mCrc32;
    private final long mCompressedSize;
    private final long mUncompressedSize;
    private final long mLocalFileHeaderOffset;
    private final String mName;
    private final int mNameSizeBytes;

    private CentralDirectoryRecord(
            ByteBuffer data,
            short gpFlags,
            short compressionMethod,
            int lastModificationTime,
            int lastModificationDate,
            long crc32,
            long compressedSize,
            long uncompressedSize,
            long localFileHeaderOffset,
            String name,
            int nameSizeBytes) {
        mData = data;
        mGpFlags = gpFlags;
        mCompressionMethod = compressionMethod;
        mLastModificationDate = lastModificationDate;
        mLastModificationTime = lastModificationTime;
        mCrc32 = crc32;
        mCompressedSize = compressedSize;
        mUncompressedSize = uncompressedSize;
        mLocalFileHeaderOffset = localFileHeaderOffset;
        mName = name;
        mNameSizeBytes = nameSizeBytes;
    }

    public int getSize() {
        return mData.remaining();
    }

    public String getName() {
        return mName;
    }

    public int getNameSizeBytes() {
        return mNameSizeBytes;
    }

    public short getGpFlags() {
        return mGpFlags;
    }

    public short getCompressionMethod() {
        return mCompressionMethod;
    }

    public int getLastModificationTime() {
        return mLastModificationTime;
    }

    public int getLastModificationDate() {
        return mLastModificationDate;
    }

    public long getCrc32() {
        return mCrc32;
    }

    public long getCompressedSize() {
        return mCompressedSize;
    }

    public long getUncompressedSize() {
        return mUncompressedSize;
    }

    public long getLocalFileHeaderOffset() {
        return mLocalFileHeaderOffset;
    }

    /**
     * Returns the Central Directory Record starting at the current position of the provided buffer
     * and advances the buffer's position immediately past the end of the record.
     */
    public static CentralDirectoryRecord getRecord(ByteBuffer buf) throws ZipFormatException {
        ZipUtils.assertByteOrderLittleEndian(buf);
        if (buf.remaining() < HEADER_SIZE_BYTES) {
            throw new ZipFormatException(
                    "Input too short. Need at least: " + HEADER_SIZE_BYTES
                            + " bytes, available: " + buf.remaining() + " bytes",
                    new BufferUnderflowException());
        }
        int originalPosition = buf.position();
        int recordSignature = buf.getInt();
        if (recordSignature != RECORD_SIGNATURE) {
            throw new ZipFormatException(
                    "Not a Central Directory record. Signature: 0x"
                            + Long.toHexString(recordSignature & 0xffffffffL));
        }
        buf.position(originalPosition + GP_FLAGS_OFFSET);
        short gpFlags = buf.getShort();
        short compressionMethod = buf.getShort();
        int lastModificationTime = ZipUtils.getUnsignedInt16(buf);
        int lastModificationDate = ZipUtils.getUnsignedInt16(buf);
        long crc32 = ZipUtils.getUnsignedInt32(buf);
        long compressedSize = ZipUtils.getUnsignedInt32(buf);
        long uncompressedSize = ZipUtils.getUnsignedInt32(buf);
        int nameSize = ZipUtils.getUnsignedInt16(buf);
        int extraSize = ZipUtils.getUnsignedInt16(buf);
        int commentSize = ZipUtils.getUnsignedInt16(buf);
        buf.position(originalPosition + LOCAL_FILE_HEADER_OFFSET_OFFSET);
        long localFileHeaderOffset = ZipUtils.getUnsignedInt32(buf);
        buf.position(originalPosition);
        int recordSize = HEADER_SIZE_BYTES + nameSize + extraSize + commentSize;
        if (recordSize > buf.remaining()) {
            throw new ZipFormatException(
                    "Input too short. Need: " + recordSize + " bytes, available: "
                            + buf.remaining() + " bytes",
                    new BufferUnderflowException());
        }
        String name = getName(buf, originalPosition + NAME_OFFSET, nameSize);
        buf.position(originalPosition);
        int originalLimit = buf.limit();
        int recordEndInBuf = originalPosition + recordSize;
        ByteBuffer recordBuf;
        try {
            buf.limit(recordEndInBuf);
            recordBuf = buf.slice();
        } finally {
            buf.limit(originalLimit);
        }
        // Consume this record
        buf.position(recordEndInBuf);
        return new CentralDirectoryRecord(
                recordBuf,
                gpFlags,
                compressionMethod,
                lastModificationTime,
                lastModificationDate,
                crc32,
                compressedSize,
                uncompressedSize,
                localFileHeaderOffset,
                name,
                nameSize);
    }

    public void copyTo(ByteBuffer output) {
        output.put(mData.slice());
    }

    public CentralDirectoryRecord createWithModifiedLocalFileHeaderOffset(
            long localFileHeaderOffset) {
        ByteBuffer result = ByteBuffer.allocate(mData.remaining());
        result.put(mData.slice());
        result.flip();
        result.order(ByteOrder.LITTLE_ENDIAN);
        ZipUtils.setUnsignedInt32(result, LOCAL_FILE_HEADER_OFFSET_OFFSET, localFileHeaderOffset);
        return new CentralDirectoryRecord(
                result,
                mGpFlags,
                mCompressionMethod,
                mLastModificationTime,
                mLastModificationDate,
                mCrc32,
                mCompressedSize,
                mUncompressedSize,
                localFileHeaderOffset,
                mName,
                mNameSizeBytes);
    }

    public static CentralDirectoryRecord createWithDeflateCompressedData(
            String name,
            int lastModifiedTime,
            int lastModifiedDate,
            long crc32,
            long compressedSize,
            long uncompressedSize,
            long localFileHeaderOffset) {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        short gpFlags = ZipUtils.GP_FLAG_EFS; // UTF-8 character encoding used for entry name
        short compressionMethod = ZipUtils.COMPRESSION_METHOD_DEFLATED;
        int recordSize = HEADER_SIZE_BYTES + nameBytes.length;
        ByteBuffer result = ByteBuffer.allocate(recordSize);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.putInt(RECORD_SIGNATURE);
        ZipUtils.putUnsignedInt16(result, 0x14); // Version made by
        ZipUtils.putUnsignedInt16(result, 0x14); // Minimum version needed to extract
        result.putShort(gpFlags);
        result.putShort(compressionMethod);
        ZipUtils.putUnsignedInt16(result, lastModifiedTime);
        ZipUtils.putUnsignedInt16(result, lastModifiedDate);
        ZipUtils.putUnsignedInt32(result, crc32);
        ZipUtils.putUnsignedInt32(result, compressedSize);
        ZipUtils.putUnsignedInt32(result, uncompressedSize);
        ZipUtils.putUnsignedInt16(result, nameBytes.length);
        ZipUtils.putUnsignedInt16(result, 0); // Extra field length
        ZipUtils.putUnsignedInt16(result, 0); // File comment length
        ZipUtils.putUnsignedInt16(result, 0); // Disk number
        ZipUtils.putUnsignedInt16(result, 0); // Internal file attributes
        ZipUtils.putUnsignedInt32(result, 0); // External file attributes
        ZipUtils.putUnsignedInt32(result, localFileHeaderOffset);
        result.put(nameBytes);

        if (result.hasRemaining()) {
            throw new RuntimeException("pos: " + result.position() + ", limit: " + result.limit());
        }
        result.flip();
        return new CentralDirectoryRecord(
                result,
                gpFlags,
                compressionMethod,
                lastModifiedTime,
                lastModifiedDate,
                crc32,
                compressedSize,
                uncompressedSize,
                localFileHeaderOffset,
                name,
                nameBytes.length);
    }

    static String getName(ByteBuffer record, int position, int nameLengthBytes) {
        byte[] nameBytes;
        int nameBytesOffset;
        if (record.hasArray()) {
            nameBytes = record.array();
            nameBytesOffset = record.arrayOffset() + position;
        } else {
            nameBytes = new byte[nameLengthBytes];
            nameBytesOffset = 0;
            int originalPosition = record.position();
            try {
                record.position(position);
                record.get(nameBytes);
            } finally {
                record.position(originalPosition);
            }
        }
        return new String(nameBytes, nameBytesOffset, nameLengthBytes, StandardCharsets.UTF_8);
    }

    private static class ByLocalFileHeaderOffsetComparator
            implements Comparator<CentralDirectoryRecord> {
        @Override
        public int compare(CentralDirectoryRecord r1, CentralDirectoryRecord r2) {
            long offset1 = r1.getLocalFileHeaderOffset();
            long offset2 = r2.getLocalFileHeaderOffset();
            if (offset1 > offset2) {
                return 1;
            } else if (offset1 < offset2) {
                return -1;
            } else {
                return 0;
            }
        }
    }
}
