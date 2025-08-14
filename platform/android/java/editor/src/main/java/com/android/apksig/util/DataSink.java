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

package com.android.apksig.util;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Consumer of input data which may be provided in one go or in chunks.
 */
public interface DataSink {

    /**
     * Consumes the provided chunk of data.
     *
     * <p>This data sink guarantees to not hold references to the provided buffer after this method
     * terminates.
     *
     * @throws IndexOutOfBoundsException if {@code offset} or {@code length} are negative, or if
     *         {@code offset + length} is greater than {@code buf.length}.
     */
    void consume(byte[] buf, int offset, int length) throws IOException;

    /**
     * Consumes all remaining data in the provided buffer and advances the buffer's position
     * to the buffer's limit.
     *
     * <p>This data sink guarantees to not hold references to the provided buffer after this method
     * terminates.
     */
    void consume(ByteBuffer buf) throws IOException;
}
