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
import java.io.OutputStream;
import java.nio.ByteBuffer;

/**
 * {@link DataSink} which outputs received data into the associated {@link OutputStream}.
 */
public class OutputStreamDataSink implements DataSink {

    private static final int MAX_READ_CHUNK_SIZE = 65536;

    private final OutputStream mOut;

    /**
     * Constructs a new {@code OutputStreamDataSink} which outputs received data into the provided
     * {@link OutputStream}.
     */
    public OutputStreamDataSink(OutputStream out) {
        if (out == null) {
            throw new NullPointerException("out == null");
        }
        mOut = out;
    }

    /**
     * Returns {@link OutputStream} into which this data sink outputs received data.
     */
    public OutputStream getOutputStream() {
        return mOut;
    }

    @Override
    public void consume(byte[] buf, int offset, int length) throws IOException {
        mOut.write(buf, offset, length);
    }

    @Override
    public void consume(ByteBuffer buf) throws IOException {
        if (!buf.hasRemaining()) {
            return;
        }

        if (buf.hasArray()) {
            mOut.write(
                    buf.array(),
                    buf.arrayOffset() + buf.position(),
                    buf.remaining());
            buf.position(buf.limit());
        } else {
            byte[] tmp = new byte[Math.min(buf.remaining(), MAX_READ_CHUNK_SIZE)];
            while (buf.hasRemaining()) {
                int chunkSize = Math.min(buf.remaining(), tmp.length);
                buf.get(tmp, 0, chunkSize);
                mOut.write(tmp, 0, chunkSize);
            }
        }
    }
}
