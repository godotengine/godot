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

import com.android.apksig.util.DataSink;
import com.android.apksig.util.DataSource;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

/** Pseudo {@link DataSource} that chains the given {@link DataSource} as a continuous one. */
public class ChainedDataSource implements DataSource {

    private final DataSource[] mSources;
    private final long mTotalSize;

    public ChainedDataSource(DataSource... sources) {
        mSources = sources;
        mTotalSize = Arrays.stream(sources).mapToLong(src -> src.size()).sum();
    }

    @Override
    public long size() {
        return mTotalSize;
    }

    @Override
    public void feed(long offset, long size, DataSink sink) throws IOException {
        if (offset + size > mTotalSize) {
            throw new IndexOutOfBoundsException("Requested more than available");
        }

        for (DataSource src : mSources) {
            // Offset is beyond the current source. Skip.
            if (offset >= src.size()) {
                offset -= src.size();
                continue;
            }

            // If the remaining is enough, finish it.
            long remaining = src.size() - offset;
            if (remaining >= size) {
                src.feed(offset, size, sink);
                break;
            }

            // If the remaining is not enough, consume all.
            src.feed(offset, remaining, sink);
            size -= remaining;
            offset = 0;
        }
    }

    @Override
    public ByteBuffer getByteBuffer(long offset, int size) throws IOException {
        if (offset + size > mTotalSize) {
            throw new IndexOutOfBoundsException("Requested more than available");
        }

        // Skip to the first DataSource we need.
        Pair<Integer, Long> firstSource = locateDataSource(offset);
        int i = firstSource.getFirst();
        offset = firstSource.getSecond();

        // Return the current source's ByteBuffer if it fits.
        if (offset + size <= mSources[i].size()) {
            return mSources[i].getByteBuffer(offset, size);
        }

        // Otherwise, read into a new buffer.
        ByteBuffer buffer = ByteBuffer.allocate(size);
        for (; i < mSources.length && buffer.hasRemaining(); i++) {
            long sizeToCopy = Math.min(mSources[i].size() - offset, buffer.remaining());
            mSources[i].copyTo(offset, Math.toIntExact(sizeToCopy), buffer);
            offset = 0;  // may not be zero for the first source, but reset after that.
        }
        buffer.rewind();
        return buffer;
    }

    @Override
    public void copyTo(long offset, int size, ByteBuffer dest) throws IOException {
        feed(offset, size, new ByteBufferSink(dest));
    }

    @Override
    public DataSource slice(long offset, long size) {
        // Find the first slice.
        Pair<Integer, Long> firstSource = locateDataSource(offset);
        int beginIndex = firstSource.getFirst();
        long beginLocalOffset = firstSource.getSecond();
        DataSource beginSource = mSources[beginIndex];

        if (beginLocalOffset + size <= beginSource.size()) {
            return beginSource.slice(beginLocalOffset, size);
        }

        // Add the first slice to chaining, followed by the middle full slices, then the last.
        ArrayList<DataSource> sources = new ArrayList<>();
        sources.add(beginSource.slice(
                beginLocalOffset, beginSource.size() - beginLocalOffset));

        Pair<Integer, Long> lastSource = locateDataSource(offset + size - 1);
        int endIndex = lastSource.getFirst();
        long endLocalOffset = lastSource.getSecond();

        for (int i = beginIndex + 1; i < endIndex; i++) {
            sources.add(mSources[i]);
        }

        sources.add(mSources[endIndex].slice(0, endLocalOffset + 1));
        return new ChainedDataSource(sources.toArray(new DataSource[0]));
    }

    /**
     * Find the index of DataSource that offset is at.
     * @return Pair of DataSource index and the local offset in the DataSource.
     */
    private Pair<Integer, Long> locateDataSource(long offset) {
        long localOffset = offset;
        for (int i = 0; i < mSources.length; i++) {
            if (localOffset < mSources[i].size()) {
                return Pair.of(i, localOffset);
            }
            localOffset -= mSources[i].size();
        }
        throw new IndexOutOfBoundsException("Access is out of bound, offset: " + offset +
                ", totalSize: " + mTotalSize);
    }
}
