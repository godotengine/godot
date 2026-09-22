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

package com.android.apksig.internal.zip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * ZIP End of Central Directory record.
 */
public class EocdRecord {
    private static final int CD_RECORD_COUNT_ON_DISK_OFFSET = 8;
    private static final int CD_RECORD_COUNT_TOTAL_OFFSET = 10;
    private static final int CD_SIZE_OFFSET = 12;
    private static final int CD_OFFSET_OFFSET = 16;

    public static ByteBuffer createWithModifiedCentralDirectoryInfo(
            ByteBuffer original,
            int centralDirectoryRecordCount,
            long centralDirectorySizeBytes,
            long centralDirectoryOffset) {
        ByteBuffer result = ByteBuffer.allocate(original.remaining());
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.put(original.slice());
        result.flip();
        ZipUtils.setUnsignedInt16(
                result, CD_RECORD_COUNT_ON_DISK_OFFSET, centralDirectoryRecordCount);
        ZipUtils.setUnsignedInt16(
                result, CD_RECORD_COUNT_TOTAL_OFFSET, centralDirectoryRecordCount);
        ZipUtils.setUnsignedInt32(result, CD_SIZE_OFFSET, centralDirectorySizeBytes);
        ZipUtils.setUnsignedInt32(result, CD_OFFSET_OFFSET, centralDirectoryOffset);
        return result;
    }

    public static ByteBuffer createWithPaddedComment(ByteBuffer original, int padding) {
        ByteBuffer result = ByteBuffer.allocate((int) original.remaining() + padding);
        result.order(ByteOrder.LITTLE_ENDIAN);
        result.put(original.slice());
        result.rewind();
        ZipUtils.updateZipEocdCommentLen(result);
        return result;
    }
}
