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

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;

public class V4Signature {
    public static final int CURRENT_VERSION = 2;

    public static final int HASHING_ALGORITHM_SHA256 = 1;
    public static final byte LOG2_BLOCK_SIZE_4096_BYTES = 12;

    public static final int MAX_SIGNING_INFOS_SIZE = 7168;

    public static class HashingInfo {
        public final int hashAlgorithm; // only 1 == SHA256 supported
        public final byte log2BlockSize; // only 12 (block size 4096) supported now
        public final byte[] salt; // used exactly as in fs-verity, 32 bytes max
        public final byte[] rawRootHash; // salted digest of the first Merkle tree page

        HashingInfo(int hashAlgorithm, byte log2BlockSize, byte[] salt, byte[] rawRootHash) {
            this.hashAlgorithm = hashAlgorithm;
            this.log2BlockSize = log2BlockSize;
            this.salt = salt;
            this.rawRootHash = rawRootHash;
        }

        static HashingInfo fromByteArray(byte[] bytes) throws IOException {
            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            final int hashAlgorithm = buffer.getInt();
            final byte log2BlockSize = buffer.get();
            byte[] salt = readBytes(buffer);
            byte[] rawRootHash = readBytes(buffer);
            return new HashingInfo(hashAlgorithm, log2BlockSize, salt, rawRootHash);
        }

        byte[] toByteArray() {
            final int size = 4/*hashAlgorithm*/ + 1/*log2BlockSize*/ + bytesSize(this.salt)
                    + bytesSize(this.rawRootHash);
            ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(this.hashAlgorithm);
            buffer.put(this.log2BlockSize);
            writeBytes(buffer, this.salt);
            writeBytes(buffer, this.rawRootHash);
            return buffer.array();
        }
    }

    public static class SigningInfo {
        public final byte[] apkDigest;  // used to match with the corresponding APK
        public final byte[] certificate; // ASN.1 DER form
        public final byte[] additionalData; // a free-form binary data blob
        public final byte[] publicKey; // ASN.1 DER, must match the certificate
        public final int signatureAlgorithmId; // see the APK v2 doc for the list
        public final byte[] signature;

        SigningInfo(byte[] apkDigest, byte[] certificate, byte[] additionalData,
                byte[] publicKey, int signatureAlgorithmId, byte[] signature) {
            this.apkDigest = apkDigest;
            this.certificate = certificate;
            this.additionalData = additionalData;
            this.publicKey = publicKey;
            this.signatureAlgorithmId = signatureAlgorithmId;
            this.signature = signature;
        }

        static SigningInfo fromByteArray(byte[] bytes) throws IOException {
            return fromByteBuffer(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN));
        }

        static SigningInfo fromByteBuffer(ByteBuffer buffer) throws IOException {
            byte[] apkDigest = readBytes(buffer);
            byte[] certificate = readBytes(buffer);
            byte[] additionalData = readBytes(buffer);
            byte[] publicKey = readBytes(buffer);
            int signatureAlgorithmId = buffer.getInt();
            byte[] signature = readBytes(buffer);
            return new SigningInfo(apkDigest, certificate, additionalData, publicKey,
                    signatureAlgorithmId, signature);
        }

        byte[] toByteArray() {
            final int size = bytesSize(this.apkDigest) + bytesSize(this.certificate) + bytesSize(
                    this.additionalData) + bytesSize(this.publicKey) + 4/*signatureAlgorithmId*/
                    + bytesSize(this.signature);
            ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
            writeBytes(buffer, this.apkDigest);
            writeBytes(buffer, this.certificate);
            writeBytes(buffer, this.additionalData);
            writeBytes(buffer, this.publicKey);
            buffer.putInt(this.signatureAlgorithmId);
            writeBytes(buffer, this.signature);
            return buffer.array();
        }
    }

    public static class SigningInfoBlock {
        public final int blockId;
        public final byte[] signingInfo;

        public SigningInfoBlock(int blockId, byte[] signingInfo) {
            this.blockId = blockId;
            this.signingInfo = signingInfo;
        }

        static SigningInfoBlock fromByteBuffer(ByteBuffer buffer) throws IOException {
            int blockId = buffer.getInt();
            byte[] signingInfo = readBytes(buffer);
            return new SigningInfoBlock(blockId, signingInfo);
        }

        byte[] toByteArray() {
            final int size = 4/*blockId*/ + bytesSize(this.signingInfo);
            ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(this.blockId);
            writeBytes(buffer, this.signingInfo);
            return buffer.array();
        }
    }

    public static class SigningInfos {
        public final SigningInfo signingInfo;
        public final SigningInfoBlock[] signingInfoBlocks;

        public SigningInfos(SigningInfo signingInfo) {
            this.signingInfo = signingInfo;
            this.signingInfoBlocks = new SigningInfoBlock[0];
        }

        public SigningInfos(SigningInfo signingInfo, SigningInfoBlock... signingInfoBlocks) {
            this.signingInfo = signingInfo;
            this.signingInfoBlocks = signingInfoBlocks;
        }

        public static SigningInfos fromByteArray(byte[] bytes) throws IOException {
            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            SigningInfo signingInfo = SigningInfo.fromByteBuffer(buffer);
            if (!buffer.hasRemaining()) {
                return new SigningInfos(signingInfo);
            }
            ArrayList<SigningInfoBlock> signingInfoBlocks = new ArrayList<>(1);
            while (buffer.hasRemaining()) {
                signingInfoBlocks.add(SigningInfoBlock.fromByteBuffer(buffer));
            }
            return new SigningInfos(signingInfo,
                    signingInfoBlocks.toArray(new SigningInfoBlock[signingInfoBlocks.size()]));
        }

        byte[] toByteArray() {
            byte[][] arrays = new byte[1 + this.signingInfoBlocks.length][];
            arrays[0] = this.signingInfo.toByteArray();
            int size = arrays[0].length;
            for (int i = 0, isize = this.signingInfoBlocks.length; i < isize; ++i) {
                arrays[i + 1] = this.signingInfoBlocks[i].toByteArray();
                size += arrays[i + 1].length;
            }
            if (size > MAX_SIGNING_INFOS_SIZE) {
                throw new IllegalArgumentException(
                        "Combined SigningInfos length exceeded limit of 7K: " + size);
            }

            // Combine all arrays into one.
            byte[] result = Arrays.copyOf(arrays[0], size);
            int offset = arrays[0].length;
            for (int i = 0, isize = this.signingInfoBlocks.length; i < isize; ++i) {
                System.arraycopy(arrays[i + 1], 0, result, offset, arrays[i + 1].length);
                offset += arrays[i + 1].length;
            }
            return result;
        }
    }

    // Always 2 for now.
    public final int version;
    public final byte[] hashingInfo;
    // Can contain either SigningInfo or SigningInfo + one or multiple SigningInfoBlock.
    // Passed as-is to the kernel. Can be retrieved later.
    public final byte[] signingInfos;

    V4Signature(int version, byte[] hashingInfo, byte[] signingInfos) {
        this.version = version;
        this.hashingInfo = hashingInfo;
        this.signingInfos = signingInfos;
    }

    static V4Signature readFrom(InputStream stream) throws IOException {
        final int version = readIntLE(stream);
        if (version != CURRENT_VERSION) {
            throw new IOException("Invalid signature version.");
        }
        final byte[] hashingInfo = readBytes(stream);
        final byte[] signingInfo = readBytes(stream);
        return new V4Signature(version, hashingInfo, signingInfo);
    }

    public void writeTo(OutputStream stream) throws IOException {
        writeIntLE(stream, this.version);
        writeBytes(stream, this.hashingInfo);
        writeBytes(stream, this.signingInfos);
    }

    static byte[] getSignedData(long fileSize, HashingInfo hashingInfo, SigningInfo signingInfo) {
        final int size =
                4/*size*/ + 8/*fileSize*/ + 4/*hash_algorithm*/ + 1/*log2_blocksize*/ + bytesSize(
                        hashingInfo.salt) + bytesSize(hashingInfo.rawRootHash) + bytesSize(
                        signingInfo.apkDigest) + bytesSize(signingInfo.certificate) + bytesSize(
                        signingInfo.additionalData);
        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(size);
        buffer.putLong(fileSize);
        buffer.putInt(hashingInfo.hashAlgorithm);
        buffer.put(hashingInfo.log2BlockSize);
        writeBytes(buffer, hashingInfo.salt);
        writeBytes(buffer, hashingInfo.rawRootHash);
        writeBytes(buffer, signingInfo.apkDigest);
        writeBytes(buffer, signingInfo.certificate);
        writeBytes(buffer, signingInfo.additionalData);
        return buffer.array();
    }

    // Utility methods.
    static int bytesSize(byte[] bytes) {
        return 4/*length*/ + (bytes == null ? 0 : bytes.length);
    }

    static void readFully(InputStream stream, byte[] buffer) throws IOException {
        int len = buffer.length;
        int n = 0;
        while (n < len) {
            int count = stream.read(buffer, n, len - n);
            if (count < 0) {
                throw new EOFException();
            }
            n += count;
        }
    }

    static int readIntLE(InputStream stream) throws IOException {
        final byte[] buffer = new byte[4];
        readFully(stream, buffer);
        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    static void writeIntLE(OutputStream stream, int v) throws IOException {
        final byte[] buffer = ByteBuffer.wrap(new byte[4]).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
        stream.write(buffer);
    }

    static byte[] readBytes(InputStream stream) throws IOException {
        try {
            final int size = readIntLE(stream);
            final byte[] bytes = new byte[size];
            readFully(stream, bytes);
            return bytes;
        } catch (EOFException ignored) {
            return null;
        }
    }

    static byte[] readBytes(ByteBuffer buffer) throws IOException {
        if (buffer.remaining() < 4) {
            throw new EOFException();
        }
        final int size = buffer.getInt();
        if (buffer.remaining() < size) {
            throw new EOFException();
        }
        final byte[] bytes = new byte[size];
        buffer.get(bytes);
        return bytes;
    }

    static void writeBytes(OutputStream stream, byte[] bytes) throws IOException {
        if (bytes == null) {
            writeIntLE(stream, 0);
            return;
        }
        writeIntLE(stream, bytes.length);
        stream.write(bytes);
    }

    static void writeBytes(ByteBuffer buffer, byte[] bytes) {
        if (bytes == null) {
            buffer.putInt(0);
            return;
        }
        buffer.putInt(bytes.length);
        buffer.put(bytes);
    }
}
