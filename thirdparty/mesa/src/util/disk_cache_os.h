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

#ifndef DISK_CACHE_OS_H
#define DISK_CACHE_OS_H

#include "util/u_queue.h"

#if DETECT_OS_WINDOWS

/* TODO: implement disk cache support on windows */

#else

#include "util/fossilize_db.h"
#include "util/mesa_cache_db.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of bits to mask off from a cache key to get an index. */
#define CACHE_INDEX_KEY_BITS 16

/* Mask for computing an index from a key. */
#define CACHE_INDEX_KEY_MASK ((1 << CACHE_INDEX_KEY_BITS) - 1)

/* The number of keys that can be stored in the index. */
#define CACHE_INDEX_MAX_KEYS (1 << CACHE_INDEX_KEY_BITS)

enum disk_cache_type {
   DISK_CACHE_MULTI_FILE,
   DISK_CACHE_SINGLE_FILE,
   DISK_CACHE_DATABASE,
};

struct disk_cache {
   /* The path to the cache directory. */
   char *path;
   bool path_init_failed;

   /* Thread queue for compressing and writing cache entries to disk */
   struct util_queue cache_queue;

   struct foz_db foz_db;

   struct mesa_cache_db cache_db;

   enum disk_cache_type type;

   /* Seed for rand, which is used to pick a random directory */
   uint64_t seed_xorshift128plus[2];

   /* A pointer to the mmapped index file within the cache directory. */
   uint8_t *index_mmap;
   size_t index_mmap_size;

   /* Pointer to total size of all objects in cache (within index_mmap) */
   uint64_t *size;

   /* Pointer to stored keys, (within index_mmap). */
   uint8_t *stored_keys;

   /* Maximum size of all cached objects (in bytes). */
   uint64_t max_size;

   /* Driver cache keys. */
   uint8_t *driver_keys_blob;
   size_t driver_keys_blob_size;

   disk_cache_put_cb blob_put_cb;
   disk_cache_get_cb blob_get_cb;

   /* Don't compress cached data. This is for testing purposes only. */
   bool compression_disabled;

   struct {
      bool enabled;
      unsigned hits;
      unsigned misses;
   } stats;

   /* Internal RO FOZ cache for combined use of RO and RW caches. */
   struct disk_cache *foz_ro_cache;
};

struct cache_entry_file_data {
   uint32_t crc32;
   uint32_t uncompressed_size;
};

struct disk_cache_put_job {
   struct util_queue_fence fence;

   struct disk_cache *cache;

   cache_key key;

   /* Copy of cache data to be compressed and written. */
   void *data;

   /* Size of data to be compressed and written. */
   size_t size;

   struct cache_item_metadata cache_item_metadata;
};

char *
disk_cache_generate_cache_dir(void *mem_ctx, const char *gpu_name,
                              const char *driver_id,
                              enum disk_cache_type cache_type);

void
disk_cache_evict_lru_item(struct disk_cache *cache);

void
disk_cache_evict_item(struct disk_cache *cache, char *filename);

void *
disk_cache_load_item_foz(struct disk_cache *cache, const cache_key key,
                         size_t *size);

void *
disk_cache_load_item(struct disk_cache *cache, char *filename, size_t *size);

char *
disk_cache_get_cache_filename(struct disk_cache *cache, const cache_key key);

bool
disk_cache_write_item_to_disk_foz(struct disk_cache_put_job *dc_job);

void
disk_cache_write_item_to_disk(struct disk_cache_put_job *dc_job,
                              char *filename);

bool
disk_cache_enabled(void);

bool
disk_cache_load_cache_index_foz(void *mem_ctx, struct disk_cache *cache);

bool
disk_cache_mmap_cache_index(void *mem_ctx, struct disk_cache *cache,
                            char *path);

void
disk_cache_destroy_mmap(struct disk_cache *cache);

void *
disk_cache_db_load_item(struct disk_cache *cache, const cache_key key,
                        size_t *size);

bool
disk_cache_db_write_item_to_disk(struct disk_cache_put_job *dc_job);

bool
disk_cache_db_load_cache_index(void *mem_ctx, struct disk_cache *cache);

#ifdef __cplusplus
}
#endif

#endif

#endif /* DISK_CACHE_OS_H */
