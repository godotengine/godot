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

import com.android.apksig.apk.ApkFormatException;
import com.android.apksig.internal.util.Pair;
import com.android.apksig.util.DataSource;
import com.android.apksig.zip.ZipFormatException;
import com.android.apksig.zip.ZipSections;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

/**
 * Assorted ZIP format helpers.
 *
 * <p>NOTE: Most helper methods operating on {@code ByteBuffer} instances expect that the byte
 * order of these buffers is little-endian.
 */
public abstract class ZipUtils {
    private ZipUtils() {}

    public static final short COMPRESSION_METHOD_STORED = 0;
    public static final short COMPRESSION_METHOD_DEFLATED = 8;

    public static final short GP_FLAG_DATA_DESCRIPTOR_USED = 0x08;
    public static final short GP_FLAG_EFS = 0x0800;

    private static final int ZIP_EOCD_REC_MIN_SIZE = 22;
    private static final int ZIP_EOCD_REC_SIG = 0x06054b50;
    private static final int ZIP_EOCD_CENTRAL_DIR_TOTAL_RECORD_COUNT_OFFSET = 10;
    private static final int ZIP_EOCD_CENTRAL_DIR_SIZE_FIELD_OFFSET = 12;
    private static final int ZIP_EOCD_CENTRAL_DIR_OFFSET_FIELD_OFFSET = 16;
    private static final int ZIP_EOCD_COMMENT_LENGTH_FIELD_OFFSET = 20;

    private static final int UINT16_MAX_VALUE = 0xffff;

    /**
     * Sets the offset of the start of the ZIP Central Directory in the archive.
     *
     * <p>NOTE: Byte order of {@code zipEndOfCentralDirectory} must be little-endian.
     */
    public static void setZipEocdCentralDirectoryOffset(
            ByteBuffer zipEndOfCentralDirectory, long offset) {
        assertByteOrderLittleEndian(zipEndOfCentralDirectory);
        setUnsignedInt32(
                zipEndOfCentralDirectory,
                zipEndOfCentralDirectory.position() + ZIP_EOCD_CENTRAL_DIR_OFFSET_FIELD_OFFSET,
                offset);
    }

    /**
     * Sets the length of EOCD comment.
     *
     * <p>NOTE: Byte order of {@code zipEndOfCentralDirectory} must be little-endian.
     */
    public static void updateZipEocdCommentLen(ByteBuffer zipEndOfCentralDirectory) {
        assertByteOrderLittleEndian(zipEndOfCentralDirectory);
        int commentLen = zipEndOfCentralDirectory.remaining() - ZIP_EOCD_REC_MIN_SIZE;
        setUnsignedInt16(
                zipEndOfCentralDirectory,
                zipEndOfCentralDirectory.position() + ZIP_EOCD_COMMENT_LENGTH_FIELD_OFFSET,
                commentLen);
    }

    /**
     * Returns the offset of the start of the ZIP Central Directory in the archive.
     *
     * <p>NOTE: Byte order of {@code zipEndOfCentralDirectory} must be little-endian.
     */
    public static long getZipEocdCentralDirectoryOffset(ByteBuffer zipEndOfCentralDirectory) {
        assertByteOrderLittleEndian(zipEndOfCentralDirectory);
        return getUnsignedInt32(
                zipEndOfCentralDirectory,
                zipEndOfCentralDirectory.position() + ZIP_EOCD_CENTRAL_DIR_OFFSET_FIELD_OFFSET);
    }

    /**
     * Returns the size (in bytes) of the ZIP Central Directory.
     *
     * <p>NOTE: Byte order of {@code zipEndOfCentralDirectory} must be little-endian.
     */
    public static long getZipEocdCentralDirectorySizeBytes(ByteBuffer zipEndOfCentralDirectory) {
        assertByteOrderLittleEndian(zipEndOfCentralDirectory);
        return getUnsignedInt32(
                zipEndOfCentralDirectory,
                zipEndOfCentralDirectory.position() + ZIP_EOCD_CENTRAL_DIR_SIZE_FIELD_OFFSET);
    }

    /**
     * Returns the total number of records in ZIP Central Directory.
     *
     * <p>NOTE: Byte order of {@code zipEndOfCentralDirectory} must be little-endian.
     */
    public static int getZipEocdCentralDirectoryTotalRecordCount(
            ByteBuffer zipEndOfCentralDirectory) {
        assertByteOrderLittleEndian(zipEndOfCentralDirectory);
        return getUnsignedInt16(
                zipEndOfCentralDirectory,
                zipEndOfCentralDirectory.position()
                        + ZIP_EOCD_CENTRAL_DIR_TOTAL_RECORD_COUNT_OFFSET);
    }

    /**
     * Returns the ZIP End of Central Directory record of the provided ZIP file.
     *
     * @return contents of the ZIP End of Central Directory record and the record's offset in the
     *         file or {@code null} if the file does not contain the record.
     *
     * @throws IOException if an I/O error occurs while reading the file.
     */
    public static Pair<ByteBuffer, Long> findZipEndOfCentralDirectoryRecord(DataSource zip)
            throws IOException {
        // ZIP End of Central Directory (EOCD) record is located at the very end of the ZIP archive.
        // The record can be identified by its 4-byte signature/magic which is located at the very
        // beginning of the record. A complication is that the record is variable-length because of
        // the comment field.
        // The algorithm for locating the ZIP EOCD record is as follows. We search backwards from
        // end of the buffer for the EOCD record signature. Whenever we find a signature, we check
        // the candidate record's comment length is such that the remainder of the record takes up
        // exactly the remaining bytes in the buffer. The search is bounded because the maximum
        // size of the comment field is 65535 bytes because the field is an unsigned 16-bit number.

        long fileSize = zip.size();
        if (fileSize < ZIP_EOCD_REC_MIN_SIZE) {
            return null;
        }

        // Optimization: 99.99% of APKs have a zero-length comment field in the EoCD record and thus
        // the EoCD record offset is known in advance. Try that offset first to avoid unnecessarily
        // reading more data.
        Pair<ByteBuffer, Long> result = findZipEndOfCentralDirectoryRecord(zip, 0);
        if (result != null) {
            return result;
        }

        // EoCD does not start where we expected it to. Perhaps it contains a non-empty comment
        // field. Expand the search. The maximum size of the comment field in EoCD is 65535 because
        // the comment length field is an unsigned 16-bit number.
        return findZipEndOfCentralDirectoryRecord(zip, UINT16_MAX_VALUE);
    }

    /**
     * Returns the ZIP End of Central Directory record of the provided ZIP file.
     *
     * @param maxCommentSize maximum accepted size (in bytes) of EoCD comment field. The permitted
     *        value is from 0 to 65535 inclusive. The smaller the value, the faster this method
     *        locates the record, provided its comment field is no longer than this value.
     *
     * @return contents of the ZIP End of Central Directory record and the record's offset in the
     *         file or {@code null} if the file does not contain the record.
     *
     * @throws IOException if an I/O error occurs while reading the file.
     */
    private static Pair<ByteBuffer, Long> findZipEndOfCentralDirectoryRecord(
            DataSource zip, int maxCommentSize) throws IOException {
        // ZIP End of Central Directory (EOCD) record is located at the very end of the ZIP archive.
        // The record can be identified by its 4-byte signature/magic which is located at the very
        // beginning of the record. A complication is that the record is variable-length because of
        // the comment field.
        // The algorithm for locating the ZIP EOCD record is as follows. We search backwards from
        // end of the buffer for the EOCD record signature. Whenever we find a signature, we check
        // the candidate record's comment length is such that the remainder of the record takes up
        // exactly the remaining bytes in the buffer. The search is bounded because the maximum
        // size of the comment field is 65535 bytes because the field is an unsigned 16-bit number.

        if ((maxCommentSize < 0) || (maxCommentSize > UINT16_MAX_VALUE)) {
            throw new IllegalArgumentException("maxCommentSize: " + maxCommentSize);
        }

        long fileSize = zip.size();
        if (fileSize < ZIP_EOCD_REC_MIN_SIZE) {
            // No space for EoCD record in the file.
            return null;
        }
        // Lower maxCommentSize if the file is too small.
        maxCommentSize = (int) Math.min(maxCommentSize, fileSize - ZIP_EOCD_REC_MIN_SIZE);

        int maxEocdSize = ZIP_EOCD_REC_MIN_SIZE + maxCommentSize;
        long bufOffsetInFile = fileSize - maxEocdSize;
        ByteBuffer buf = zip.getByteBuffer(bufOffsetInFile, maxEocdSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        int eocdOffsetInBuf = findZipEndOfCentralDirectoryRecord(buf);
        if (eocdOffsetInBuf == -1) {
            // No EoCD record found in the buffer
            return null;
        }
        // EoCD found
        buf.position(eocdOffsetInBuf);
        ByteBuffer eocd = buf.slice();
        eocd.order(ByteOrder.LITTLE_ENDIAN);
        return Pair.of(eocd, bufOffsetInFile + eocdOffsetInBuf);
    }

    /**
     * Returns the position at which ZIP End of Central Directory record starts in the provided
     * buffer or {@code -1} if the record is not present.
     *
     * <p>NOTE: Byte order of {@code zipContents} must be little-endian.
     */
    private static int findZipEndOfCentralDirectoryRecord(ByteBuffer zipContents) {
        assertByteOrderLittleEndian(zipContents);

        // ZIP End of Central Directory (EOCD) record is located at the very end of the ZIP archive.
        // The record can be identified by its 4-byte signature/magic which is located at the very
        // beginning of the record. A complication is that the record is variable-length because of
        // the comment field.
        // The algorithm for locating the ZIP EOCD record is as follows. We search backwards from
        // end of the buffer for the EOCD record signature. Whenever we find a signature, we check
        // the candidate record's comment length is such that the remainder of the record takes up
        // exactly the remaining bytes in the buffer. The search is bounded because the maximum
        // size of the comment field is 65535 bytes because the field is an unsigned 16-bit number.

        int archiveSize = zipContents.capacity();
        if (archiveSize < ZIP_EOCD_REC_MIN_SIZE) {
            return -1;
        }
        int maxCommentLength = Math.min(archiveSize - ZIP_EOCD_REC_MIN_SIZE, UINT16_MAX_VALUE);
        int eocdWithEmptyCommentStartPosition = archiveSize - ZIP_EOCD_REC_MIN_SIZE;
        for (int expectedCommentLength = 0; expectedCommentLength <= maxCommentLength;
                expectedCommentLength++) {
            int eocdStartPos = eocdWithEmptyCommentStartPosition - expectedCommentLength;
            if (zipContents.getInt(eocdStartPos) == ZIP_EOCD_REC_SIG) {
                int actualCommentLength =
                        getUnsignedInt16(
                                zipContents, eocdStartPos + ZIP_EOCD_COMMENT_LENGTH_FIELD_OFFSET);
                if (actualCommentLength == expectedCommentLength) {
                    return eocdStartPos;
                }
            }
        }

        return -1;
    }

    static void assertByteOrderLittleEndian(ByteBuffer buffer) {
        if (buffer.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("ByteBuffer byte order must be little endian");
        }
    }

    public static int getUnsignedInt16(ByteBuffer buffer, int offset) {
        return buffer.getShort(offset) & 0xffff;
    }

    public static int getUnsignedInt16(ByteBuffer buffer) {
        return buffer.getShort() & 0xffff;
    }

    public static List<CentralDirectoryRecord> parseZipCentralDirectory(
            DataSource apk,
            ZipSections apkSections)
            throws IOException, ApkFormatException {
        // Read the ZIP Central Directory
        long cdSizeBytes = apkSections.getZipCentralDirectorySizeBytes();
        if (cdSizeBytes > Integer.MAX_VALUE) {
            throw new ApkFormatException("ZIP Central Directory too large: " + cdSizeBytes);
        }
        long cdOffset = apkSections.getZipCentralDirectoryOffset();
        ByteBuffer cd = apk.getByteBuffer(cdOffset, (int) cdSizeBytes);
        cd.order(ByteOrder.LITTLE_ENDIAN);

        // Parse the ZIP Central Directory
        int expectedCdRecordCount = apkSections.getZipCentralDirectoryRecordCount();
        List<CentralDirectoryRecord> cdRecords = new ArrayList<>(expectedCdRecordCount);
        for (int i = 0; i < expectedCdRecordCount; i++) {
            CentralDirectoryRecord cdRecord;
            int offsetInsideCd = cd.position();
            try {
                cdRecord = CentralDirectoryRecord.getRecord(cd);
            } catch (ZipFormatException e) {
                throw new ApkFormatException(
                        "Malformed ZIP Central Directory record #" + (i + 1)
                                + " at file offset " + (cdOffset + offsetInsideCd),
                        e);
            }
            String entryName = cdRecord.getName();
            if (entryName.endsWith("/")) {
                // Ignore directory entries
                continue;
            }
            cdRecords.add(cdRecord);
        }
        // There may be more data in Central Directory, but we don't warn or throw because Android
        // ignores unused CD data.

        return cdRecords;
    }

    static void setUnsignedInt16(ByteBuffer buffer, int offset, int value) {
        if ((value < 0) || (value > 0xffff)) {
            throw new IllegalArgumentException("uint16 value of out range: " + value);
        }
        buffer.putShort(offset, (short) value);
    }

    static void setUnsignedInt32(ByteBuffer buffer, int offset, long value) {
        if ((value < 0) || (value > 0xffffffffL)) {
            throw new IllegalArgumentException("uint32 value of out range: " + value);
        }
        buffer.putInt(offset, (int) value);
    }

    public static void putUnsignedInt16(ByteBuffer buffer, int value) {
        if ((value < 0) || (value > 0xffff)) {
            throw new IllegalArgumentException("uint16 value of out range: " + value);
        }
        buffer.putShort((short) value);
    }

    static long getUnsignedInt32(ByteBuffer buffer, int offset) {
        return buffer.getInt(offset) & 0xffffffffL;
    }

    static long getUnsignedInt32(ByteBuffer buffer) {
        return buffer.getInt() & 0xffffffffL;
    }

    static void putUnsignedInt32(ByteBuffer buffer, long value) {
        if ((value < 0) || (value > 0xffffffffL)) {
            throw new IllegalArgumentException("uint32 value of out range: " + value);
        }
        buffer.putInt((int) value);
    }

    public static DeflateResult deflate(ByteBuffer input) {
        byte[] inputBuf;
        int inputOffset;
        int inputLength = input.remaining();
        if (input.hasArray()) {
            inputBuf = input.array();
            inputOffset = input.arrayOffset() + input.position();
            input.position(input.limit());
        } else {
            inputBuf = new byte[inputLength];
            inputOffset = 0;
            input.get(inputBuf);
        }
        CRC32 crc32 = new CRC32();
        crc32.update(inputBuf, inputOffset, inputLength);
        long crc32Value = crc32.getValue();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Deflater deflater = new Deflater(9, true);
        deflater.setInput(inputBuf, inputOffset, inputLength);
        deflater.finish();
        byte[] buf = new byte[65536];
        while (!deflater.finished()) {
            int chunkSize = deflater.deflate(buf);
            out.write(buf, 0, chunkSize);
        }
        return new DeflateResult(inputLength, crc32Value, out.toByteArray());
    }

    public static class DeflateResult {
        public final int inputSizeBytes;
        public final long inputCrc32;
        public final byte[] output;

        public DeflateResult(int inputSizeBytes, long inputCrc32, byte[] output) {
            this.inputSizeBytes = inputSizeBytes;
            this.inputCrc32 = inputCrc32;
            this.output = output;
        }
    }
}