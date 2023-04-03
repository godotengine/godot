/*
 * Copyright © 2020 Valve Corporation
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

/* This is a basic c implementation of a fossilize db like format intended for
 * use with the Mesa shader cache.
 *
 * The format is compatible enough to allow the fossilize db tools to be used
 * to do things like merge db collections, but unlike fossilize db which uses
 * a zlib implementation for compression of data entries, we use zstd for
 * compression.
 */

#ifndef FOSSILIZE_DB_H
#define FOSSILIZE_DB_H

#ifdef HAVE_FLOCK
#define FOZ_DB_UTIL 1
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "simple_mtx.h"

/* Max number of DBs our implementation can read from at once */
#define FOZ_MAX_DBS 9 /* Default DB + 8 Read only DBs */

#define FOSSILIZE_BLOB_HASH_LENGTH 40

enum {
   FOSSILIZE_COMPRESSION_NONE = 1,
   FOSSILIZE_COMPRESSION_DEFLATE = 2
};

enum {
   FOSSILIZE_FORMAT_VERSION = 6,
   FOSSILIZE_FORMAT_MIN_COMPAT_VERSION = 5
};

struct foz_payload_header {
   uint32_t payload_size;
   uint32_t format;
   uint32_t crc;
   uint32_t uncompressed_size;
};

struct foz_db_entry {
   uint8_t file_idx;
   uint8_t key[20];
   uint64_t offset;
   struct foz_payload_header header;
};

struct foz_dbs_list_updater {
   int inotify_fd;
   int inotify_wd; /* watch descriptor */
   const char *list_filename;
   thrd_t thrd;
};

struct foz_db {
   FILE *file[FOZ_MAX_DBS];          /* An array of all foz dbs */
   FILE *db_idx;                     /* The default writable foz db idx */
   simple_mtx_t mtx;                 /* Mutex for file/hash table read/writes */
   simple_mtx_t flock_mtx;           /* Mutex for flocking the file for writes */
   void *mem_ctx;
   struct hash_table_u64 *index_db;  /* Hash table of all foz db entries */
   bool alive;
   const char *cache_path;
   struct foz_dbs_list_updater updater;
};

bool
foz_prepare(struct foz_db *foz_db, char *cache_path);

void
foz_destroy(struct foz_db *foz_db);

void *
foz_read_entry(struct foz_db *foz_db, const uint8_t *cache_key_160bit,
               size_t *size);

bool
foz_write_entry(struct foz_db *foz_db, const uint8_t *cache_key_160bit,
                const void *blob, size_t size);

#endif /* FOSSILIZE_DB_H */
