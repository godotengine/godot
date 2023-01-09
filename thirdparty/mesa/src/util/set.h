/*
 * Copyright © 2009-2012 Intel Corporation
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
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *
 */

#ifndef _SET_H
#define _SET_H

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct set_entry {
   uint32_t hash;
   const void *key;
};

struct set {
   void *mem_ctx;
   struct set_entry *table;
   uint32_t (*key_hash_function)(const void *key);
   bool (*key_equals_function)(const void *a, const void *b);
   uint32_t size;
   uint32_t rehash;
   uint64_t size_magic;
   uint64_t rehash_magic;
   uint32_t max_entries;
   uint32_t size_index;
   uint32_t entries;
   uint32_t deleted_entries;
};

bool
_mesa_set_init(struct set *ht, void *mem_ctx,
                 uint32_t (*key_hash_function)(const void *key),
                 bool (*key_equals_function)(const void *a,
                                             const void *b));

struct set *
_mesa_set_create(void *mem_ctx,
                 uint32_t (*key_hash_function)(const void *key),
                 bool (*key_equals_function)(const void *a,
                                             const void *b));
struct set *
_mesa_set_create_u32_keys(void *mem_ctx);

struct set *
_mesa_set_clone(struct set *set, void *dst_mem_ctx);

void
_mesa_set_destroy(struct set *set,
                  void (*delete_function)(struct set_entry *entry));
void
_mesa_set_resize(struct set *set, uint32_t entries);
void
_mesa_set_clear(struct set *set,
                void (*delete_function)(struct set_entry *entry));

struct set_entry *
_mesa_set_add(struct set *set, const void *key);
struct set_entry *
_mesa_set_add_pre_hashed(struct set *set, uint32_t hash, const void *key);

struct set_entry *
_mesa_set_search_or_add(struct set *set, const void *key, bool *found);
struct set_entry *
_mesa_set_search_or_add_pre_hashed(struct set *set, uint32_t hash,
                                   const void *key, bool *found);

struct set_entry *
_mesa_set_search(const struct set *set, const void *key);
struct set_entry *
_mesa_set_search_pre_hashed(const struct set *set, uint32_t hash,
                            const void *key);

struct set_entry *
_mesa_set_search_and_add(struct set *set, const void *key, bool *replaced);
struct set_entry *
_mesa_set_search_and_add_pre_hashed(struct set *set, uint32_t hash,
                                    const void *key, bool *replaced);

void
_mesa_set_remove(struct set *set, struct set_entry *entry);
void
_mesa_set_remove_key(struct set *set, const void *key);

struct set_entry *
_mesa_set_next_entry(const struct set *set, struct set_entry *entry);
struct set_entry *
_mesa_set_next_entry_unsafe(const struct set *set, struct set_entry *entry);

struct set *
_mesa_pointer_set_create(void *mem_ctx);

bool
_mesa_set_intersects(struct set *a, struct set *b);

/**
 * This foreach function is safe against deletion, but not against
 * insertion (which may rehash the set, making entry a dangling
 * pointer).
 */
#define set_foreach(set, entry)                                     \
   for (struct set_entry *entry = _mesa_set_next_entry(set, NULL);  \
        entry != NULL;                                              \
        entry = _mesa_set_next_entry(set, entry))

/**
 * This foreach function destroys the table as it iterates.
 * It is not safe to use when inserting or removing entries.
 */
#define set_foreach_remove(set, entry)                              \
   for (struct set_entry *entry = _mesa_set_next_entry_unsafe(set, NULL);  \
        (set)->entries;                                              \
        entry->hash = 0, entry->key = (void*)NULL, (set)->entries--, entry = _mesa_set_next_entry_unsafe(set, entry))

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* _SET_H */
