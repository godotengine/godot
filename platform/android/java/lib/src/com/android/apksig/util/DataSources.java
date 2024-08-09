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

import com.android.apksig.internal.util.ByteBufferDataSource;
import com.android.apksig.internal.util.FileChannelDataSource;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Utility methods for working with {@link DataSource} abstraction.
 */
public abstract class DataSources {
    private DataSources() {}

    /**
     * Returns a {@link DataSource} backed by the provided {@link ByteBuffer}. The data source
     * represents the data contained between the position and limit of the buffer. Changes to the
     * buffer's contents will be visible in the data source.
     */
    public static DataSource asDataSource(ByteBuffer buffer) {
        if (buffer == null) {
            throw new NullPointerException();
        }
        return new ByteBufferDataSource(buffer);
    }

    /**
     * Returns a {@link DataSource} backed by the provided {@link RandomAccessFile}. Changes to the
     * file, including changes to size of file, will be visible in the data source.
     */
    public static DataSource asDataSource(RandomAccessFile file) {
        return asDataSource(file.getChannel());
    }

    /**
     * Returns a {@link DataSource} backed by the provided region of the {@link RandomAccessFile}.
     * Changes to the file will be visible in the data source.
     */
    public static DataSource asDataSource(RandomAccessFile file, long offset, long size) {
        return asDataSource(file.getChannel(), offset, size);
    }

    /**
     * Returns a {@link DataSource} backed by the provided {@link FileChannel}. Changes to the
     * file, including changes to size of file, will be visible in the data source.
     */
    public static DataSource asDataSource(FileChannel channel) {
        if (channel == null) {
            throw new NullPointerException();
        }
        return new FileChannelDataSource(channel);
    }

    /**
     * Returns a {@link DataSource} backed by the provided region of the {@link FileChannel}.
     * Changes to the file will be visible in the data source.
     */
    public static DataSource asDataSource(FileChannel channel, long offset, long size) {
        if (channel == null) {
            throw new NullPointerException();
        }
        return new FileChannelDataSource(channel, offset, size);
    }
}
