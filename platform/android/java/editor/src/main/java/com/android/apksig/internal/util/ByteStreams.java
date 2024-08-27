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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Utilities for byte arrays and I/O streams.
 */
public final class ByteStreams {
    private ByteStreams() {}

    /**
     * Returns the data remaining in the provided input stream as a byte array
     */
    public static byte[] toByteArray(InputStream in) throws IOException {
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        byte[] buf = new byte[16384];
        int chunkSize;
        while ((chunkSize = in.read(buf)) != -1) {
            result.write(buf, 0, chunkSize);
        }
        return result.toByteArray();
    }
}
