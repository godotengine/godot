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
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * {@link DataSink} which outputs received data into the associated file, sequentially.
 */
public class RandomAccessFileDataSink implements DataSink {

    private final RandomAccessFile mFile;
    private final FileChannel mFileChannel;
    private long mPosition;

    /**
     * Constructs a new {@code RandomAccessFileDataSink} which stores output starting from the
     * beginning of the provided file.
     */
    public RandomAccessFileDataSink(RandomAccessFile file) {
        this(file, 0);
    }

    /**
     * Constructs a new {@code RandomAccessFileDataSink} which stores output starting from the
     * specified position of the provided file.
     */
    public RandomAccessFileDataSink(RandomAccessFile file, long startPosition) {
        if (file == null) {
            throw new NullPointerException("file == null");
        }
        if (startPosition < 0) {
            throw new IllegalArgumentException("startPosition: " + startPosition);
        }
        mFile = file;
        mFileChannel = file.getChannel();
        mPosition = startPosition;
    }

    /**
     * Returns the underlying {@link RandomAccessFile}.
     */
    public RandomAccessFile getFile() {
        return mFile;
    }

    @Override
    public void consume(byte[] buf, int offset, int length) throws IOException {
        if (offset < 0) {
            // Must perform this check here because RandomAccessFile.write doesn't throw when offset
            // is negative but length is 0
            throw new IndexOutOfBoundsException("offset: " + offset);
        }
        if (offset > buf.length) {
            // Must perform this check here because RandomAccessFile.write doesn't throw when offset
            // is too large but length is 0
            throw new IndexOutOfBoundsException(
                    "offset: " + offset + ", buf.length: " + buf.length);
        }
        if (length == 0) {
            return;
        }

        synchronized (mFile) {
            mFile.seek(mPosition);
            mFile.write(buf, offset, length);
            mPosition += length;
        }
    }

    @Override
    public void consume(ByteBuffer buf) throws IOException {
        int length = buf.remaining();
        if (length == 0) {
            return;
        }

        synchronized (mFile) {
            mFile.seek(mPosition);
            while (buf.hasRemaining()) {
                mFileChannel.write(buf);
            }
            mPosition += length;
        }
    }
}
