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

package com.android.apksig.internal.util;

import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * {@link DataSource} backed by a {@link FileChannel} for {@link RandomAccessFile} access.
 */
public class FileChannelDataSource implements DataSource {

    private static final int MAX_READ_CHUNK_SIZE = 1024 * 1024;

    private final FileChannel mChannel;
    private final long mOffset;
    private final long mSize;

    /**
     * Constructs a new {@code FileChannelDataSource} based on the data contained in the
     * whole file. Changes to the contents of the file, including the size of the file,
     * will be visible in this data source.
     */
    public FileChannelDataSource(FileChannel channel) {
        mChannel = channel;
        mOffset = 0;
        mSize = -1;
    }

    /**
     * Constructs a new {@code FileChannelDataSource} based on the data contained in the
     * specified region of the provided file. Changes to the contents of the file will be visible in
     * this data source.
     *
     * @throws IndexOutOfBoundsException if {@code offset} or {@code size} is negative.
     */
    public FileChannelDataSource(FileChannel channel, long offset, long size) {
        if (offset < 0) {
            throw new IndexOutOfBoundsException("offset: " + size);
        }
        if (size < 0) {
            throw new IndexOutOfBoundsException("size: " + size);
        }
        mChannel = channel;
        mOffset = offset;
        mSize = size;
    }

    @Override
    public long size() {
        if (mSize == -1) {
            try {
                return mChannel.size();
            } catch (IOException e) {
                return 0;
            }
        } else {
            return mSize;
        }
    }

    @Override
    public FileChannelDataSource slice(long offset, long size) {
        long sourceSize = size();
        checkChunkValid(offset, size, sourceSize);
        if ((offset == 0) && (size == sourceSize)) {
            return this;
        }

        return new FileChannelDataSource(mChannel, mOffset + offset, size);
    }

    @Override
    public void feed(long offset, long size, DataSink sink) throws IOException {
        long sourceSize = size();
        checkChunkValid(offset, size, sourceSize);
        if (size == 0) {
            return;
        }

        long chunkOffsetInFile = mOffset + offset;
        long remaining = size;
        ByteBuffer buf = ByteBuffer.allocateDirect((int) Math.min(remaining, MAX_READ_CHUNK_SIZE));

        while (remaining > 0) {
            int chunkSize = (int) Math.min(remaining, buf.capacity());
            int chunkRemaining = chunkSize;
            buf.limit(chunkSize);
            synchronized (mChannel) {
                mChannel.position(chunkOffsetInFile);
                while (chunkRemaining > 0) {
                    int read = mChannel.read(buf);
                    if (read < 0) {
                        throw new IOException("Unexpected EOF encountered");
                    }
                    chunkRemaining -= read;
                }
            }
            buf.flip();
            sink.consume(buf);
            buf.clear();
            chunkOffsetInFile += chunkSize;
            remaining -= chunkSize;
        }
    }

    @Override
    public void copyTo(long offset, int size, ByteBuffer dest) throws IOException {
        long sourceSize = size();
        checkChunkValid(offset, size, sourceSize);
        if (size == 0) {
            return;
        }
        if (size > dest.remaining()) {
            throw new BufferOverflowException();
        }

        long offsetInFile = mOffset + offset;
        int remaining = size;
        int prevLimit = dest.limit();
        try {
            // FileChannel.read(ByteBuffer) reads up to dest.remaining(). Thus, we need to adjust
            // the buffer's limit to avoid reading more than size bytes.
            dest.limit(dest.position() + size);
            while (remaining > 0) {
                int chunkSize;
                synchronized (mChannel) {
                    mChannel.position(offsetInFile);
                    chunkSize = mChannel.read(dest);
                }
                offsetInFile += chunkSize;
                remaining -= chunkSize;
            }
        } finally {
            dest.limit(prevLimit);
        }
    }

    @Override
    public ByteBuffer getByteBuffer(long offset, int size) throws IOException {
        if (size < 0) {
            throw new IndexOutOfBoundsException("size: " + size);
        }
        ByteBuffer result = ByteBuffer.allocate(size);
        copyTo(offset, size, result);
        result.flip();
        return result;
    }

    private static void checkChunkValid(long offset, long size, long sourceSize) {
        if (offset < 0) {
            throw new IndexOutOfBoundsException("offset: " + offset);
        }
        if (size < 0) {
            throw new IndexOutOfBoundsException("size: " + size);
        }
        if (offset > sourceSize) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") > source size (" + sourceSize + ")");
        }
        long endOffset = offset + size;
        if (endOffset < offset) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") + size (" + size + ") overflow");
        }
        if (endOffset > sourceSize) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") + size (" + size
                            + ") > source size (" + sourceSize  +")");
        }
    }
}
