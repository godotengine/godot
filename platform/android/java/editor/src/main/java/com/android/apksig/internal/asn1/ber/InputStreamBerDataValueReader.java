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

package com.android.apksig.internal.asn1.ber;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * {@link BerDataValueReader} which reads from an {@link InputStream} returning BER-encoded data
 * values. See {@code X.690} for the encoding.
 */
public class InputStreamBerDataValueReader implements BerDataValueReader {
    private final InputStream mIn;

    public InputStreamBerDataValueReader(InputStream in) {
        if (in == null) {
            throw new NullPointerException("in == null");
        }
        mIn = in;
    }

    @Override
    public BerDataValue readDataValue() throws BerDataValueFormatException {
        return readDataValue(mIn);
    }

    /**
     * Returns the next data value or {@code null} if end of input has been reached.
     *
     * @throws BerDataValueFormatException if the value being read is malformed.
     */
    @SuppressWarnings("resource")
    private static BerDataValue readDataValue(InputStream input)
            throws BerDataValueFormatException {
        RecordingInputStream in = new RecordingInputStream(input);

        try {
            int firstIdentifierByte = in.read();
            if (firstIdentifierByte == -1) {
                // End of input
                return null;
            }
            int tagNumber = readTagNumber(in, firstIdentifierByte);

            int firstLengthByte = in.read();
            if (firstLengthByte == -1) {
                throw new BerDataValueFormatException("Missing length");
            }

            boolean constructed = BerEncoding.isConstructed((byte) firstIdentifierByte);
            int contentsLength;
            int contentsOffsetInDataValue;
            if ((firstLengthByte & 0x80) == 0) {
                // short form length
                contentsLength = readShortFormLength(firstLengthByte);
                contentsOffsetInDataValue = in.getReadByteCount();
                skipDefiniteLengthContents(in, contentsLength);
            } else if ((firstLengthByte & 0xff) != 0x80) {
                // long form length
                contentsLength = readLongFormLength(in, firstLengthByte);
                contentsOffsetInDataValue = in.getReadByteCount();
                skipDefiniteLengthContents(in, contentsLength);
            } else {
                // indefinite length
                contentsOffsetInDataValue = in.getReadByteCount();
                contentsLength =
                        constructed
                                ? skipConstructedIndefiniteLengthContents(in)
                                : skipPrimitiveIndefiniteLengthContents(in);
            }

            byte[] encoded = in.getReadBytes();
            ByteBuffer encodedContents =
                    ByteBuffer.wrap(encoded, contentsOffsetInDataValue, contentsLength);
            return new BerDataValue(
                    ByteBuffer.wrap(encoded),
                    encodedContents,
                    BerEncoding.getTagClass((byte) firstIdentifierByte),
                    constructed,
                    tagNumber);
        } catch (IOException e) {
            throw new BerDataValueFormatException("Failed to read data value", e);
        }
    }

    private static int readTagNumber(InputStream in, int firstIdentifierByte)
            throws IOException, BerDataValueFormatException {
        int tagNumber = BerEncoding.getTagNumber((byte) firstIdentifierByte);
        if (tagNumber == 0x1f) {
            // high-tag-number form
            return readHighTagNumber(in);
        } else {
            // low-tag-number form
            return tagNumber;
        }
    }

    private static int readHighTagNumber(InputStream in)
            throws IOException, BerDataValueFormatException {
        // Base-128 big-endian form, where each byte has the highest bit set, except for the last
        // byte where the highest bit is not set
        int b;
        int result = 0;
        do {
            b = in.read();
            if (b == -1) {
                throw new BerDataValueFormatException("Truncated tag number");
            }
            if (result > Integer.MAX_VALUE >>> 7) {
                throw new BerDataValueFormatException("Tag number too large");
            }
            result <<= 7;
            result |= b & 0x7f;
        } while ((b & 0x80) != 0);
        return result;
    }

    private static int readShortFormLength(int firstLengthByte) {
        return firstLengthByte & 0x7f;
    }

    private static int readLongFormLength(InputStream in, int firstLengthByte)
            throws IOException, BerDataValueFormatException {
        // The low 7 bits of the first byte represent the number of bytes (following the first
        // byte) in which the length is in big-endian base-256 form
        int byteCount = firstLengthByte & 0x7f;
        if (byteCount > 4) {
            throw new BerDataValueFormatException("Length too large: " + byteCount + " bytes");
        }
        int result = 0;
        for (int i = 0; i < byteCount; i++) {
            int b = in.read();
            if (b == -1) {
                throw new BerDataValueFormatException("Truncated length");
            }
            if (result > Integer.MAX_VALUE >>> 8) {
                throw new BerDataValueFormatException("Length too large");
            }
            result <<= 8;
            result |= b & 0xff;
        }
        return result;
    }

    private static void skipDefiniteLengthContents(InputStream in, int len)
            throws IOException, BerDataValueFormatException {
        long bytesRead = 0;
        while (len > 0) {
            int skipped = (int) in.skip(len);
            if (skipped <= 0) {
                throw new BerDataValueFormatException(
                        "Truncated definite-length contents: " + bytesRead + " bytes read"
                                + ", " + len + " missing");
            }
            len -= skipped;
            bytesRead += skipped;
        }
    }

    private static int skipPrimitiveIndefiniteLengthContents(InputStream in)
            throws IOException, BerDataValueFormatException {
        // Contents are terminated by 0x00 0x00
        boolean prevZeroByte = false;
        int bytesRead = 0;
        while (true) {
            int b = in.read();
            if (b == -1) {
                throw new BerDataValueFormatException(
                        "Truncated indefinite-length contents: " + bytesRead + " bytes read");
            }
            bytesRead++;
            if (bytesRead < 0) {
                throw new BerDataValueFormatException("Indefinite-length contents too long");
            }
            if (b == 0) {
                if (prevZeroByte) {
                    // End of contents reached -- we've read the value and its terminator 0x00 0x00
                    return bytesRead - 2;
                }
                prevZeroByte = true;
                continue;
            } else {
                prevZeroByte = false;
            }
        }
    }

    private static int skipConstructedIndefiniteLengthContents(RecordingInputStream in)
            throws BerDataValueFormatException {
        // Contents are terminated by 0x00 0x00. However, this data value is constructed, meaning it
        // can contain data values which are indefinite length encoded as well. As a result, we
        // must parse the direct children of this data value to correctly skip over the contents of
        // this data value.
        int readByteCountBefore = in.getReadByteCount();
        while (true) {
            // We can't easily peek for the 0x00 0x00 terminator using the provided InputStream.
            // Thus, we use the fact that 0x00 0x00 parses as a data value whose encoded form we
            // then check below to see whether it's 0x00 0x00.
            BerDataValue dataValue = readDataValue(in);
            if (dataValue == null) {
                throw new BerDataValueFormatException(
                        "Truncated indefinite-length contents: "
                                + (in.getReadByteCount() - readByteCountBefore) + " bytes read");
            }
            if (in.getReadByteCount() <= 0) {
                throw new BerDataValueFormatException("Indefinite-length contents too long");
            }
            ByteBuffer encoded = dataValue.getEncoded();
            if ((encoded.remaining() == 2) && (encoded.get(0) == 0) && (encoded.get(1) == 0)) {
                // 0x00 0x00 encountered
                return in.getReadByteCount() - readByteCountBefore - 2;
            }
        }
    }

    private static class RecordingInputStream extends InputStream {
        private final InputStream mIn;
        private final ByteArrayOutputStream mBuf;

        private RecordingInputStream(InputStream in) {
            mIn = in;
            mBuf = new ByteArrayOutputStream();
        }

        public byte[] getReadBytes() {
            return mBuf.toByteArray();
        }

        public int getReadByteCount() {
            return mBuf.size();
        }

        @Override
        public int read() throws IOException {
            int b = mIn.read();
            if (b != -1) {
                mBuf.write(b);
            }
            return b;
        }

        @Override
        public int read(byte[] b) throws IOException {
            int len = mIn.read(b);
            if (len > 0) {
                mBuf.write(b, 0, len);
            }
            return len;
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            len = mIn.read(b, off, len);
            if (len > 0) {
                mBuf.write(b, off, len);
            }
            return len;
        }

        @Override
        public long skip(long n) throws IOException {
            if (n <= 0) {
                return mIn.skip(n);
            }

            byte[] buf = new byte[4096];
            int len = mIn.read(buf, 0, (int) Math.min(buf.length, n));
            if (len > 0) {
                mBuf.write(buf, 0, len);
            }
            return (len < 0) ? 0 : len;
        }

        @Override
        public int available() throws IOException {
            return super.available();
        }

        @Override
        public void close() throws IOException {
            super.close();
        }

        @Override
        public synchronized void mark(int readlimit) {}

        @Override
        public synchronized void reset() throws IOException {
            throw new IOException("mark/reset not supported");
        }

        @Override
        public boolean markSupported() {
            return false;
        }
    }
}
