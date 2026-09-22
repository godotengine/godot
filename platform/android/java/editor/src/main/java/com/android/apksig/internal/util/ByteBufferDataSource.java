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
import java.nio.ByteBuffer;

/**
 * {@link DataSource} backed by a {@link ByteBuffer}.
 */
public class ByteBufferDataSource implements DataSource {

    private final ByteBuffer mBuffer;
    private final int mSize;

    /**
     * Constructs a new {@code ByteBufferDigestSource} based on the data contained in the provided
     * buffer between the buffer's position and limit.
     */
    public ByteBufferDataSource(ByteBuffer buffer) {
        this(buffer, true);
    }

    /**
     * Constructs a new {@code ByteBufferDigestSource} based on the data contained in the provided
     * buffer between the buffer's position and limit.
     */
    private ByteBufferDataSource(ByteBuffer buffer, boolean sliceRequired) {
        mBuffer = (sliceRequired) ? buffer.slice() : buffer;
        mSize = buffer.remaining();
    }

    @Override
    public long size() {
        return mSize;
    }

    @Override
    public ByteBuffer getByteBuffer(long offset, int size) {
        checkChunkValid(offset, size);

        // checkChunkValid ensures that it's OK to cast offset to int.
        int chunkPosition = (int) offset;
        int chunkLimit = chunkPosition + size;
        // Creating a slice of ByteBuffer modifies the state of the source ByteBuffer (position
        // and limit fields, to be more specific). We thus use synchronization around these
        // state-changing operations to make instances of this class thread-safe.
        synchronized (mBuffer) {
            // ByteBuffer.limit(int) and .position(int) check that that the position >= limit
            // invariant is not broken. Thus, the only way to safely change position and limit
            // without caring about their current values is to first set position to 0 or set the
            // limit to capacity.
            mBuffer.position(0);

            mBuffer.limit(chunkLimit);
            mBuffer.position(chunkPosition);
            return mBuffer.slice();
        }
    }

    @Override
    public void copyTo(long offset, int size, ByteBuffer dest) {
        dest.put(getByteBuffer(offset, size));
    }

    @Override
    public void feed(long offset, long size, DataSink sink) throws IOException {
        if ((size < 0) || (size > mSize)) {
            throw new IndexOutOfBoundsException("size: " + size + ", source size: " + mSize);
        }
        sink.consume(getByteBuffer(offset, (int) size));
    }

    @Override
    public ByteBufferDataSource slice(long offset, long size) {
        if ((offset == 0) && (size == mSize)) {
            return this;
        }
        if ((size < 0) || (size > mSize)) {
            throw new IndexOutOfBoundsException("size: " + size + ", source size: " + mSize);
        }
        return new ByteBufferDataSource(
                getByteBuffer(offset, (int) size),
                false // no need to slice -- it's already a slice
                );
    }

    private void checkChunkValid(long offset, long size) {
        if (offset < 0) {
            throw new IndexOutOfBoundsException("offset: " + offset);
        }
        if (size < 0) {
            throw new IndexOutOfBoundsException("size: " + size);
        }
        if (offset > mSize) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") > source size (" + mSize + ")");
        }
        long endOffset = offset + size;
        if (endOffset < offset) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") + size (" + size + ") overflow");
        }
        if (endOffset > mSize) {
            throw new IndexOutOfBoundsException(
                    "offset (" + offset + ") + size (" + size + ") > source size (" + mSize  +")");
        }
    }
}
