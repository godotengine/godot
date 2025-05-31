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
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;

/**
 * Data sink which stores all received data into the associated {@link ByteBuffer}.
 */
public class ByteBufferSink implements DataSink {

    private final ByteBuffer mBuffer;

    public ByteBufferSink(ByteBuffer buffer) {
        mBuffer = buffer;
    }

    public ByteBuffer getBuffer() {
        return mBuffer;
    }

    @Override
    public void consume(byte[] buf, int offset, int length) throws IOException {
        try {
            mBuffer.put(buf, offset, length);
        } catch (BufferOverflowException e) {
            throw new IOException(
                    "Insufficient space in output buffer for " + length + " bytes", e);
        }
    }

    @Override
    public void consume(ByteBuffer buf) throws IOException {
        int length = buf.remaining();
        try {
            mBuffer.put(buf);
        } catch (BufferOverflowException e) {
            throw new IOException(
                    "Insufficient space in output buffer for " + length + " bytes", e);
        }
    }
}
