//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryProgramCache: Stores compiled and linked programs in memory so they don't
//   always have to be re-compiled. Can be used in conjunction with the platform
//   layer to warm up the cache from disk.

#ifndef LIBANGLE_MEMORY_PROGRAM_CACHE_H_
#define LIBANGLE_MEMORY_PROGRAM_CACHE_H_

#include <array>

#include "common/MemoryBuffer.h"
#include "libANGLE/BlobCache.h"
#include "libANGLE/Error.h"

namespace gl
{
class Context;
class Program;
class ProgramState;

class MemoryProgramCache final : angle::NonCopyable
{
  public:
    explicit MemoryProgramCache(egl::BlobCache &blobCache);
    ~MemoryProgramCache();

    static void ComputeHash(const Context *context,
                            const Program *program,
                            egl::BlobCache::Key *hashOut);

    // Check if the cache contains a binary matching the specified program.
    bool get(const Context *context,
             const egl::BlobCache::Key &programHash,
             egl::BlobCache::Value *programOut);

    // For querying the contents of the cache.
    bool getAt(size_t index,
               const egl::BlobCache::Key **hashOut,
               egl::BlobCache::Value *programOut);

    // Evict a program from the binary cache.
    void remove(const egl::BlobCache::Key &programHash);

    // Helper method that serializes a program.
    void putProgram(const egl::BlobCache::Key &programHash,
                    const Context *context,
                    const Program *program);

    // Same as putProgram but computes the hash.
    void updateProgram(const Context *context, const Program *program);

    // Store a binary directly.  TODO(syoussefi): deprecated.  Will be removed once Chrome supports
    // EGL_ANDROID_blob_cache. http://anglebug.com/2516
    void putBinary(const egl::BlobCache::Key &programHash, const uint8_t *binary, size_t length);

    // Check the cache, and deserialize and load the program if found. Evict existing hash if load
    // fails.
    angle::Result getProgram(const Context *context,
                             Program *program,
                             egl::BlobCache::Key *hashOut);

    // Empty the cache.
    void clear();

    // Resize the cache. Discards current contents.
    void resize(size_t maxCacheSizeBytes);

    // Returns the number of entries in the cache.
    size_t entryCount() const;

    // Reduces the current cache size and returns the number of bytes freed.
    size_t trim(size_t limit);

    // Returns the current cache size in bytes.
    size_t size() const;

    // Returns the maximum cache size in bytes.
    size_t maxSize() const;

  private:
    egl::BlobCache &mBlobCache;
    unsigned int mIssuedWarnings;
};

}  // namespace gl

#endif  // LIBANGLE_MEMORY_PROGRAM_CACHE_H_
