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

#ifndef AAUDIO_FIXED_BLOCK_ADAPTER_H
#define AAUDIO_FIXED_BLOCK_ADAPTER_H

#include <memory>
#include <stdint.h>
#include <sys/types.h>

/**
 * Interface for a class that needs fixed-size blocks.
 */
class FixedBlockProcessor {
public:
    virtual ~FixedBlockProcessor() = default;
    /**
     *
     * @param buffer Pointer to first byte of data.
     * @param numBytes This will be a fixed size specified in FixedBlockAdapter::open().
     * @return Number of bytes processed or a negative error code.
     */
    virtual int32_t onProcessFixedBlock(uint8_t *buffer, int32_t numBytes) = 0;
};

/**
 * Base class for a variable-to-fixed-size block adapter.
 */
class FixedBlockAdapter
{
public:
    FixedBlockAdapter(FixedBlockProcessor &fixedBlockProcessor)
            : mFixedBlockProcessor(fixedBlockProcessor) {}

    virtual ~FixedBlockAdapter();

    /**
     * Allocate internal resources needed for buffering data.
     */
    virtual int32_t open(int32_t bytesPerFixedBlock);

    /**
     * Free internal resources.
     */
    int32_t close();

protected:
    FixedBlockProcessor  &mFixedBlockProcessor;
    std::unique_ptr<uint8_t[]> mStorage;       // Store data here while assembling buffers.
    int32_t               mSize = 0;           // Size in bytes of the fixed size buffer.
    int32_t               mPosition = 0;       // Offset of the last byte read or written.
};

#endif /* AAUDIO_FIXED_BLOCK_ADAPTER_H */
