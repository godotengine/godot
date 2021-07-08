//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// BlobCache: Stores keyed blobs in memory to support EGL_ANDROID_blob_cache.
// Can be used in conjunction with the platform layer to warm up the cache from
// disk.  MemoryProgramCache uses this to handle caching of compiled programs.

#include "libANGLE/BlobCache.h"
#include "common/utilities.h"
#include "common/version.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/histogram_macros.h"
#include "platform/Platform.h"

namespace egl
{

namespace
{
enum CacheResult
{
    kCacheMiss,
    kCacheHitMemory,
    kCacheHitDisk,
    kCacheResultMax,
};

}  // anonymous namespace

BlobCache::BlobCache(size_t maxCacheSizeBytes)
    : mBlobCache(maxCacheSizeBytes), mSetBlobFunc(nullptr), mGetBlobFunc(nullptr)
{}

BlobCache::~BlobCache() {}

void BlobCache::put(const BlobCache::Key &key, angle::MemoryBuffer &&value)
{
    if (areBlobCacheFuncsSet())
    {
        // Store the result in the application's cache
        mSetBlobFunc(key.data(), key.size(), value.data(), value.size());
    }
    else
    {
        populate(key, std::move(value), CacheSource::Memory);
    }
}

void BlobCache::putApplication(const BlobCache::Key &key, const angle::MemoryBuffer &value)
{
    if (areBlobCacheFuncsSet())
    {
        mSetBlobFunc(key.data(), key.size(), value.data(), value.size());
    }
}

void BlobCache::populate(const BlobCache::Key &key, angle::MemoryBuffer &&value, CacheSource source)
{
    CacheEntry newEntry;
    newEntry.first  = std::move(value);
    newEntry.second = source;

    // Cache it inside blob cache only if caching inside the application is not possible.
    mBlobCache.put(key, std::move(newEntry), newEntry.first.size());
}

bool BlobCache::get(angle::ScratchBuffer *scratchBuffer,
                    const BlobCache::Key &key,
                    BlobCache::Value *valueOut)
{
    // Look into the application's cache, if there is such a cache
    if (areBlobCacheFuncsSet())
    {
        EGLsizeiANDROID valueSize = mGetBlobFunc(key.data(), key.size(), nullptr, 0);
        if (valueSize <= 0)
        {
            return false;
        }

        angle::MemoryBuffer *scratchMemory;
        bool result = scratchBuffer->get(valueSize, &scratchMemory);
        if (!result)
        {
            ERR() << "Failed to allocate memory for binary blob";
            return false;
        }

        EGLsizeiANDROID originalValueSize = valueSize;
        valueSize = mGetBlobFunc(key.data(), key.size(), scratchMemory->data(), valueSize);

        // Make sure the key/value pair still exists/is unchanged after the second call
        // (modifications to the application cache by another thread are a possibility)
        if (valueSize != originalValueSize)
        {
            // This warning serves to find issues with the application cache, none of which are
            // currently known to be thread-safe.  If such a use ever arises, this WARN can be
            // removed.
            WARN() << "Binary blob no longer available in cache (removed by a thread?)";
            return false;
        }

        *valueOut = BlobCache::Value(scratchMemory->data(), scratchMemory->size());
        return true;
    }

    // Otherwise we are doing caching internally, so try to find it there
    const CacheEntry *entry;
    bool result = mBlobCache.get(key, &entry);

    if (result)
    {
        if (entry->second == CacheSource::Memory)
        {
            ANGLE_HISTOGRAM_ENUMERATION("GPU.ANGLE.ProgramCache.CacheResult", kCacheHitMemory,
                                        kCacheResultMax);
        }
        else
        {
            ANGLE_HISTOGRAM_ENUMERATION("GPU.ANGLE.ProgramCache.CacheResult", kCacheHitDisk,
                                        kCacheResultMax);
        }

        *valueOut = BlobCache::Value(entry->first.data(), entry->first.size());
    }
    else
    {
        ANGLE_HISTOGRAM_ENUMERATION("GPU.ANGLE.ProgramCache.CacheResult", kCacheMiss,
                                    kCacheResultMax);
    }

    return result;
}

bool BlobCache::getAt(size_t index, const BlobCache::Key **keyOut, BlobCache::Value *valueOut)
{
    const CacheEntry *valueBuf;
    bool result = mBlobCache.getAt(index, keyOut, &valueBuf);
    if (result)
    {
        *valueOut = BlobCache::Value(valueBuf->first.data(), valueBuf->first.size());
    }
    return result;
}

void BlobCache::remove(const BlobCache::Key &key)
{
    bool result = mBlobCache.eraseByKey(key);
    ASSERT(result);
}

void BlobCache::setBlobCacheFuncs(EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get)
{
    mSetBlobFunc = set;
    mGetBlobFunc = get;
}

bool BlobCache::areBlobCacheFuncsSet() const
{
    // Either none or both of the callbacks should be set.
    ASSERT((mSetBlobFunc != nullptr) == (mGetBlobFunc != nullptr));

    return mSetBlobFunc != nullptr && mGetBlobFunc != nullptr;
}

}  // namespace egl
