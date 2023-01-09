/*
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DISK_CACHE_H
#define DISK_CACHE_H

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#include <stdio.h>
#include "util/build_id.h"
#endif
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include "util/mesa-sha1.h"
#include "util/detect_os.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Size of cache keys in bytes. */
#define CACHE_KEY_SIZE 20

#define CACHE_DIR_NAME "mesa_shader_cache"
#define CACHE_DIR_NAME_SF "mesa_shader_cache_sf"
#define CACHE_DIR_NAME_DB "mesa_shader_cache_db"

typedef uint8_t cache_key[CACHE_KEY_SIZE];

/* WARNING: 3rd party applications might be reading the cache item metadata.
 * Do not change these values without making the change widely known.
 * Please contact Valve developers and make them aware of this change.
 */
#define CACHE_ITEM_TYPE_UNKNOWN  0x0
#define CACHE_ITEM_TYPE_GLSL     0x1

typedef void
(*disk_cache_put_cb) (const void *key, signed long keySize,
                      const void *value, signed long valueSize);

typedef signed long
(*disk_cache_get_cb) (const void *key, signed long keySize,
                      void *value, signed long valueSize);

struct cache_item_metadata {
   /**
    * The cache item type. This could be used to identify a GLSL cache item,
    * a certain type of IR (tgsi, nir, etc), or signal that it is the final
    * binary form of the shader.
    */
   uint32_t type;

   /** GLSL cache item metadata */
   cache_key *keys;   /* sha1 list of shaders that make up the cache item */
   uint32_t num_keys;
};

struct disk_cache;

static inline char *
disk_cache_format_hex_id(char *buf, const uint8_t *hex_id, unsigned size)
{
   static const char hex_digits[] = "0123456789abcdef";
   unsigned i;

   for (i = 0; i < size; i += 2) {
      buf[i] = hex_digits[hex_id[i >> 1] >> 4];
      buf[i + 1] = hex_digits[hex_id[i >> 1] & 0x0f];
   }
   buf[i] = '\0';

   return buf;
}

#ifdef HAVE_DLADDR
static inline bool
disk_cache_get_function_timestamp(void *ptr, uint32_t* timestamp)
{
   Dl_info info;
   struct stat st;
   if (!dladdr(ptr, &info) || !info.dli_fname) {
      return false;
   }
   if (stat(info.dli_fname, &st)) {
      return false;
   }

   if (!st.st_mtime) {
      fprintf(stderr, "Mesa: The provided filesystem timestamp for the cache "
              "is bogus! Disabling On-disk cache.\n");
      return false;
   }

   *timestamp = st.st_mtime;

   return true;
}

static inline bool
disk_cache_get_function_identifier(void *ptr, struct mesa_sha1 *ctx)
{
   uint32_t timestamp;

#ifdef HAVE_DL_ITERATE_PHDR
   const struct build_id_note *note = NULL;
   if ((note = build_id_find_nhdr_for_addr(ptr))) {
      _mesa_sha1_update(ctx, build_id_data(note), build_id_length(note));
   } else
#endif
   if (disk_cache_get_function_timestamp(ptr, &timestamp)) {
      _mesa_sha1_update(ctx, &timestamp, sizeof(timestamp));
   } else
      return false;
   return true;
}
#elif DETECT_OS_WINDOWS
bool
disk_cache_get_function_identifier(void *ptr, struct mesa_sha1 *ctx);
#else
static inline bool
disk_cache_get_function_identifier(void *ptr, struct mesa_sha1 *ctx)
{
   return false;
}
#endif

/* Provide inlined stub functions if the shader cache is disabled. */

#ifdef ENABLE_SHADER_CACHE

/**
 * Create a new cache object.
 *
 * This function creates the handle necessary for all subsequent cache_*
 * functions.
 *
 * This cache provides two distinct operations:
 *
 *   o Storage and retrieval of arbitrary objects by cryptographic
 *     name (or "key").  This is provided via disk_cache_put() and
 *     disk_cache_get().
 *
 *   o The ability to store a key alone and check later whether the
 *     key was previously stored. This is provided via disk_cache_put_key()
 *     and disk_cache_has_key().
 *
 * The put_key()/has_key() operations are conceptually identical to
 * put()/get() with no data, but are provided separately to allow for
 * a more efficient implementation.
 *
 * In all cases, the keys are sequences of 20 bytes. It is anticipated
 * that callers will compute appropriate SHA-1 signatures for keys,
 * (though nothing in this implementation directly relies on how the
 * names are computed). See mesa-sha1.h and _mesa_sha1_compute for
 * assistance in computing SHA-1 signatures.
 */
struct disk_cache *
disk_cache_create(const char *gpu_name, const char *timestamp,
                  uint64_t driver_flags);

/**
 * Destroy a cache object, (freeing all associated resources).
 */
void
disk_cache_destroy(struct disk_cache *cache);

/* Wait for all previous disk_cache_put() calls to be processed (used for unit
 * testing).
 */
void
disk_cache_wait_for_idle(struct disk_cache *cache);

/**
 * Remove the item in the cache under the name \key.
 */
void
disk_cache_remove(struct disk_cache *cache, const cache_key key);

/**
 * Store an item in the cache under the name \key.
 *
 * The item can be retrieved later with disk_cache_get(), (unless the item has
 * been evicted in the interim).
 *
 * Any call to disk_cache_put() may cause an existing, random item to be
 * evicted from the cache.
 */
void
disk_cache_put(struct disk_cache *cache, const cache_key key,
               const void *data, size_t size,
               struct cache_item_metadata *cache_item_metadata);

/**
 * Store an item in the cache under the name \key without copying the data param.
 *
 * The item can be retrieved later with disk_cache_get(), (unless the item has
 * been evicted in the interim).
 *
 * Any call to disk_cache_put() may cause an existing, random item to be
 * evicted from the cache.
 *
 * @p data will be freed
 */
void
disk_cache_put_nocopy(struct disk_cache *cache, const cache_key key,
                      void *data, size_t size,
                      struct cache_item_metadata *cache_item_metadata);

/**
 * Retrieve an item previously stored in the cache with the name <key>.
 *
 * The item must have been previously stored with a call to disk_cache_put().
 *
 * If \size is non-NULL, then, on successful return, it will be set to the
 * size of the object.
 *
 * \return A pointer to the stored object if found. NULL if the object
 * is not found, or if any error occurs, (memory allocation failure,
 * filesystem error, etc.). The returned data is malloc'ed so the
 * caller should call free() it when finished.
 */
void *
disk_cache_get(struct disk_cache *cache, const cache_key key, size_t *size);

/**
 * Store the name \key within the cache, (without any associated data).
 *
 * Later this key can be checked with disk_cache_has_key(), (unless the key
 * has been evicted in the interim).
 *
 * Any call to disk_cache_put_key() may cause an existing, random key to be
 * evicted from the cache.
 */
void
disk_cache_put_key(struct disk_cache *cache, const cache_key key);

/**
 * Test whether the name \key was previously recorded in the cache.
 *
 * Return value: True if disk_cache_put_key() was previously called with
 * \key, (and the key was not evicted in the interim).
 *
 * Note: disk_cache_has_key() will only return true for keys passed to
 * disk_cache_put_key(). Specifically, a call to disk_cache_put() will not cause
 * disk_cache_has_key() to return true for the same key.
 */
bool
disk_cache_has_key(struct disk_cache *cache, const cache_key key);

/**
 * Compute the name \key from \data of given \size.
 */
void
disk_cache_compute_key(struct disk_cache *cache, const void *data, size_t size,
                       cache_key key);

void
disk_cache_set_callbacks(struct disk_cache *cache, disk_cache_put_cb put,
                         disk_cache_get_cb get);

#else

static inline struct disk_cache *
disk_cache_create(const char *gpu_name, const char *timestamp,
                  uint64_t driver_flags)
{
   return NULL;
}

static inline void
disk_cache_destroy(struct disk_cache *cache)
{
}

static inline void
disk_cache_put(struct disk_cache *cache, const cache_key key,
               const void *data, size_t size,
               struct cache_item_metadata *cache_item_metadata)
{
}

static inline void
disk_cache_put_nocopy(struct disk_cache *cache, const cache_key key,
                      void *data, size_t size,
                      struct cache_item_metadata *cache_item_metadata)
{
}

static inline void
disk_cache_remove(struct disk_cache *cache, const cache_key key)
{
}

static inline uint8_t *
disk_cache_get(struct disk_cache *cache, const cache_key key, size_t *size)
{
   return NULL;
}

static inline void
disk_cache_put_key(struct disk_cache *cache, const cache_key key)
{
}

static inline bool
disk_cache_has_key(struct disk_cache *cache, const cache_key key)
{
   return false;
}

static inline void
disk_cache_compute_key(struct disk_cache *cache, const void *data, size_t size,
                       cache_key key)
{
}

static inline void
disk_cache_set_callbacks(struct disk_cache *cache, disk_cache_put_cb put,
                         disk_cache_get_cb get)
{
}

#endif /* ENABLE_SHADER_CACHE */

#ifdef __cplusplus
}
#endif

#endif /* CACHE_H */
