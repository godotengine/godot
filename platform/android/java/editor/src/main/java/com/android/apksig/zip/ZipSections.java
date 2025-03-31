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

package com.android.apksig.zip;

import java.nio.ByteBuffer;

/**
 * Base representation of an APK's zip sections containing the central directory's offset, the size
 * of the central directory in bytes, the number of records in the central directory, the offset
 * of the end of central directory, and a ByteBuffer containing the end of central directory
 * contents.
 */
public class ZipSections {
    private final long mCentralDirectoryOffset;
    private final long mCentralDirectorySizeBytes;
    private final int mCentralDirectoryRecordCount;
    private final long mEocdOffset;
    private final ByteBuffer mEocd;

    public ZipSections(
            long centralDirectoryOffset,
            long centralDirectorySizeBytes,
            int centralDirectoryRecordCount,
            long eocdOffset,
            ByteBuffer eocd) {
        mCentralDirectoryOffset = centralDirectoryOffset;
        mCentralDirectorySizeBytes = centralDirectorySizeBytes;
        mCentralDirectoryRecordCount = centralDirectoryRecordCount;
        mEocdOffset = eocdOffset;
        mEocd = eocd;
    }

    /**
     * Returns the start offset of the ZIP Central Directory. This value is taken from the
     * ZIP End of Central Directory record.
     */
    public long getZipCentralDirectoryOffset() {
        return mCentralDirectoryOffset;
    }

    /**
     * Returns the size (in bytes) of the ZIP Central Directory. This value is taken from the
     * ZIP End of Central Directory record.
     */
    public long getZipCentralDirectorySizeBytes() {
        return mCentralDirectorySizeBytes;
    }

    /**
     * Returns the number of records in the ZIP Central Directory. This value is taken from the
     * ZIP End of Central Directory record.
     */
    public int getZipCentralDirectoryRecordCount() {
        return mCentralDirectoryRecordCount;
    }

    /**
     * Returns the start offset of the ZIP End of Central Directory record. The record extends
     * until the very end of the APK.
     */
    public long getZipEndOfCentralDirectoryOffset() {
        return mEocdOffset;
    }

    /**
     * Returns the contents of the ZIP End of Central Directory.
     */
    public ByteBuffer getZipEndOfCentralDirectory() {
        return mEocd;
    }
}