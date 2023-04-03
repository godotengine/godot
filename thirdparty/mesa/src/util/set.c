/*
 * Copyright © 2009-2012 Intel Corporation
 * Copyright © 1988-2004 Keith Packard and Bart Massey.
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
 * Except as contained in this notice, the names of the authors
 * or their institutions shall not be used in advertising or
 * otherwise to promote the sale, use or other dealings in this
 * Software without prior written authorization from the
 * authors.
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *    Keith Packard <keithp@keithp.com>
 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "hash_table.h"
#include "macros.h"
#include "ralloc.h"
#include "set.h"
#include "fast_urem_by_const.h"

/*
 * From Knuth -- a good choice for hash/rehash values is p, p-2 where
 * p and p-2 are both prime.  These tables are sized to have an extra 10%
 * free to avoid exponential performance degradation as the hash table fills
 */

static const uint32_t deleted_key_value;
static const void *deleted_key = &deleted_key_value;

static const struct {
   uint32_t max_entries, size, rehash;
   uint64_t size_magic, rehash_magic;
} hash_sizes[] = {
#define ENTRY(max_entries, size, rehash) \
   { max_entries, size, rehash, \
      REMAINDER_MAGIC(size), REMAINDER_MAGIC(rehash) }

   ENTRY(2,            5,            3            ),
   ENTRY(4,            7,            5            ),
   ENTRY(8,            13,           11           ),
   ENTRY(16,           19,           17           ),
   ENTRY(32,           43,           41           ),
   ENTRY(64,           73,           71           ),
   ENTRY(128,          151,          149          ),
   ENTRY(256,          283,          281          ),
   ENTRY(512,          571,          569          ),
   ENTRY(1024,         1153,         1151         ),
   ENTRY(2048,         2269,         2267         ),
   ENTRY(4096,         4519,         4517         ),
   ENTRY(8192,         9013,         9011         ),
   ENTRY(16384,        18043,        18041        ),
   ENTRY(32768,        36109,        36107        ),
   ENTRY(65536,        72091,        72089        ),
   ENTRY(131072,       144409,       144407       ),
   ENTRY(262144,       288361,       288359       ),
   ENTRY(524288,       576883,       576881       ),
   ENTRY(1048576,      1153459,      1153457      ),
   ENTRY(2097152,      2307163,      2307161      ),
   ENTRY(4194304,      4613893,      4613891      ),
   ENTRY(8388608,      9227641,      9227639      ),
   ENTRY(16777216,     18455029,     18455027     ),
   ENTRY(33554432,     36911011,     36911009     ),
   ENTRY(67108864,     73819861,     73819859     ),
   ENTRY(134217728,    147639589,    147639587    ),
   ENTRY(268435456,    295279081,    295279079    ),
   ENTRY(536870912,    590559793,    590559791    ),
   ENTRY(1073741824,   1181116273,   1181116271   ),
   ENTRY(2147483648ul, 2362232233ul, 2362232231ul )
};

ASSERTED static inline bool
key_pointer_is_reserved(const void *key)
{
   return key == NULL || key == deleted_key;
}

static int
entry_is_free(struct set_entry *entry)
{
   return entry->key == NULL;
}

static int
entry_is_deleted(struct set_entry *entry)
{
   return entry->key == deleted_key;
}

static int
entry_is_present(struct set_entry *entry)
{
   return entry->key != NULL && entry->key != deleted_key;
}

bool
_mesa_set_init(struct set *ht, void *mem_ctx,
                 uint32_t (*key_hash_function)(const void *key),
                 bool (*key_equals_function)(const void *a,
                                             const void *b))
{
   ht->size_index = 0;
   ht->size = hash_sizes[ht->size_index].size;
   ht->rehash = hash_sizes[ht->size_index].rehash;
   ht->size_magic = hash_sizes[ht->size_index].size_magic;
   ht->rehash_magic = hash_sizes[ht->size_index].rehash_magic;
   ht->max_entries = hash_sizes[ht->size_index].max_entries;
   ht->key_hash_function = key_hash_function;
   ht->key_equals_function = key_equals_function;
   ht->table = rzalloc_array(mem_ctx, struct set_entry, ht->size);
   ht->entries = 0;
   ht->deleted_entries = 0;

   return ht->table != NULL;
}

struct set *
_mesa_set_create(void *mem_ctx,
                 uint32_t (*key_hash_function)(const void *key),
                 bool (*key_equals_function)(const void *a,
                                             const void *b))
{
   struct set *ht;

   ht = ralloc(mem_ctx, struct set);
   if (ht == NULL)
      return NULL;

   if (!_mesa_set_init(ht, ht, key_hash_function, key_equals_function)) {
      ralloc_free(ht);
      return NULL;
   }

   return ht;
}

static uint32_t
key_u32_hash(const void *key)
{
   uint32_t u = (uint32_t)(uintptr_t)key;
   return _mesa_hash_uint(&u);
}

static bool
key_u32_equals(const void *a, const void *b)
{
   return (uint32_t)(uintptr_t)a == (uint32_t)(uintptr_t)b;
}

/* key == 0 and key == deleted_key are not allowed */
struct set *
_mesa_set_create_u32_keys(void *mem_ctx)
{
   return _mesa_set_create(mem_ctx, key_u32_hash, key_u32_equals);
}

struct set *
_mesa_set_clone(struct set *set, void *dst_mem_ctx)
{
   struct set *clone;

   clone = ralloc(dst_mem_ctx, struct set);
   if (clone == NULL)
      return NULL;

   memcpy(clone, set, sizeof(struct set));

   clone->table = ralloc_array(clone, struct set_entry, clone->size);
   if (clone->table == NULL) {
      ralloc_free(clone);
      return NULL;
   }

   memcpy(clone->table, set->table, clone->size * sizeof(struct set_entry));

   return clone;
}

/**
 * Frees the given set.
 *
 * If delete_function is passed, it gets called on each entry present before
 * freeing.
 */
void
_mesa_set_destroy(struct set *ht, void (*delete_function)(struct set_entry *entry))
{
   if (!ht)
      return;

   if (delete_function) {
      set_foreach (ht, entry) {
         delete_function(entry);
      }
   }
   ralloc_free(ht->table);
   ralloc_free(ht);
}


static void
set_clear_fast(struct set *ht)
{
   memset(ht->table, 0, sizeof(struct set_entry) * hash_sizes[ht->size_index].size);
   ht->entries = ht->deleted_entries = 0;
}

/**
 * Clears all values from the given set.
 *
 * If delete_function is passed, it gets called on each entry present before
 * the set is cleared.
 */
void
_mesa_set_clear(struct set *set, void (*delete_function)(struct set_entry *entry))
{
   if (!set)
      return;

   struct set_entry *entry;

   if (delete_function) {
      for (entry = set->table; entry != set->table + set->size; entry++) {
         if (entry_is_present(entry))
            delete_function(entry);

         entry->key = NULL;
      }
      set->entries = 0;
      set->deleted_entries = 0;
   } else
      set_clear_fast(set);
}

/**
 * Finds a set entry with the given key and hash of that key.
 *
 * Returns NULL if no entry is found.
 */
static struct set_entry *
set_search(const struct set *ht, uint32_t hash, const void *key)
{
   assert(!key_pointer_is_reserved(key));

   uint32_t size = ht->size;
   uint32_t start_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = util_fast_urem32(hash, ht->rehash,
                                           ht->rehash_magic) + 1;
   uint32_t hash_address = start_address;
   do {
      struct set_entry *entry = ht->table + hash_address;

      if (entry_is_free(entry)) {
         return NULL;
      } else if (entry_is_present(entry) && entry->hash == hash) {
         if (ht->key_equals_function(key, entry->key)) {
            return entry;
         }
      }

      hash_address += double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (hash_address != start_address);

   return NULL;
}

struct set_entry *
_mesa_set_search(const struct set *set, const void *key)
{
   assert(set->key_hash_function);
   return set_search(set, set->key_hash_function(key), key);
}

struct set_entry *
_mesa_set_search_pre_hashed(const struct set *set, uint32_t hash,
                            const void *key)
{
   assert(set->key_hash_function == NULL ||
          hash == set->key_hash_function(key));
   return set_search(set, hash, key);
}

static void
set_add_rehash(struct set *ht, uint32_t hash, const void *key)
{
   uint32_t size = ht->size;
   uint32_t start_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = util_fast_urem32(hash, ht->rehash,
                                           ht->rehash_magic) + 1;
   uint32_t hash_address = start_address;
   do {
      struct set_entry *entry = ht->table + hash_address;
      if (likely(entry->key == NULL)) {
         entry->hash = hash;
         entry->key = key;
         return;
      }

      hash_address = hash_address + double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (true);
}

static void
set_rehash(struct set *ht, unsigned new_size_index)
{
   struct set old_ht;
   struct set_entry *table;

   if (ht->size_index == new_size_index && ht->deleted_entries == ht->max_entries) {
      set_clear_fast(ht);
      assert(!ht->entries);
      return;
   }

   if (new_size_index >= ARRAY_SIZE(hash_sizes))
      return;

   table = rzalloc_array(ralloc_parent(ht->table), struct set_entry,
                         hash_sizes[new_size_index].size);
   if (table == NULL)
      return;

   old_ht = *ht;

   ht->table = table;
   ht->size_index = new_size_index;
   ht->size = hash_sizes[ht->size_index].size;
   ht->rehash = hash_sizes[ht->size_index].rehash;
   ht->size_magic = hash_sizes[ht->size_index].size_magic;
   ht->rehash_magic = hash_sizes[ht->size_index].rehash_magic;
   ht->max_entries = hash_sizes[ht->size_index].max_entries;
   ht->entries = 0;
   ht->deleted_entries = 0;

   set_foreach(&old_ht, entry) {
      set_add_rehash(ht, entry->hash, entry->key);
   }

   ht->entries = old_ht.entries;

   ralloc_free(old_ht.table);
}

void
_mesa_set_resize(struct set *set, uint32_t entries)
{
   /* You can't shrink a set below its number of entries */
   if (set->entries > entries)
      entries = set->entries;

   unsigned size_index = 0;
   while (hash_sizes[size_index].max_entries < entries)
      size_index++;

   set_rehash(set, size_index);
}

/**
 * Find a matching entry for the given key, or insert it if it doesn't already
 * exist.
 *
 * Note that insertion may rearrange the table on a resize or rehash,
 * so previously found hash_entries are no longer valid after this function.
 */
static struct set_entry *
set_search_or_add(struct set *ht, uint32_t hash, const void *key, bool *found)
{
   struct set_entry *available_entry = NULL;

   assert(!key_pointer_is_reserved(key));

   if (ht->entries >= ht->max_entries) {
      set_rehash(ht, ht->size_index + 1);
   } else if (ht->deleted_entries + ht->entries >= ht->max_entries) {
      set_rehash(ht, ht->size_index);
   }

   uint32_t size = ht->size;
   uint32_t start_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = util_fast_urem32(hash, ht->rehash,
                                           ht->rehash_magic) + 1;
   uint32_t hash_address = start_address;
   do {
      struct set_entry *entry = ht->table + hash_address;

      if (!entry_is_present(entry)) {
         /* Stash the first available entry we find */
         if (available_entry == NULL)
            available_entry = entry;
         if (entry_is_free(entry))
            break;
      }

      if (!entry_is_deleted(entry) &&
          entry->hash == hash &&
          ht->key_equals_function(key, entry->key)) {
         if (found)
            *found = true;
         return entry;
      }

      hash_address = hash_address + double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (hash_address != start_address);

   if (available_entry) {
      /* There is no matching entry, create it. */
      if (entry_is_deleted(available_entry))
         ht->deleted_entries--;
      available_entry->hash = hash;
      available_entry->key = key;
      ht->entries++;
      if (found)
         *found = false;
      return available_entry;
   }

   /* We could hit here if a required resize failed. An unchecked-malloc
    * application could ignore this result.
    */
   return NULL;
}

/**
 * Inserts the key with the given hash into the table.
 *
 * Note that insertion may rearrange the table on a resize or rehash,
 * so previously found hash_entries are no longer valid after this function.
 */
static struct set_entry *
set_add(struct set *ht, uint32_t hash, const void *key)
{
   struct set_entry *entry = set_search_or_add(ht, hash, key, NULL);

   if (unlikely(!entry))
      return NULL;

   /* Note: If a matching entry already exists, this will replace it.  This is
    * a relatively common feature of hash tables, with the alternative
    * generally being "insert the new value as well, and return it first when
    * the key is searched for".
    *
    * Note that the hash table doesn't have a delete callback.  If freeing of
    * old keys is required to avoid memory leaks, use the alternative
    * _mesa_set_search_or_add function and implement the replacement yourself.
    */
   entry->key = key;
   return entry;
}

struct set_entry *
_mesa_set_add(struct set *set, const void *key)
{
   assert(set->key_hash_function);
   return set_add(set, set->key_hash_function(key), key);
}

struct set_entry *
_mesa_set_add_pre_hashed(struct set *set, uint32_t hash, const void *key)
{
   assert(set->key_hash_function == NULL ||
          hash == set->key_hash_function(key));
   return set_add(set, hash, key);
}

struct set_entry *
_mesa_set_search_and_add(struct set *set, const void *key, bool *replaced)
{
   assert(set->key_hash_function);
   return _mesa_set_search_and_add_pre_hashed(set,
                                              set->key_hash_function(key),
                                              key, replaced);
}

struct set_entry *
_mesa_set_search_and_add_pre_hashed(struct set *set, uint32_t hash,
                                    const void *key, bool *replaced)
{
   assert(set->key_hash_function == NULL ||
          hash == set->key_hash_function(key));
   struct set_entry *entry = set_search_or_add(set, hash, key, replaced);

   if (unlikely(!entry))
      return NULL;

   /* This implements the replacement, same as _mesa_set_add(). The user will
    * be notified if we're overwriting a found entry.
    */
   entry->key = key;
   return entry;
}

struct set_entry *
_mesa_set_search_or_add(struct set *set, const void *key, bool *found)
{
   assert(set->key_hash_function);
   return set_search_or_add(set, set->key_hash_function(key), key, found);
}

struct set_entry *
_mesa_set_search_or_add_pre_hashed(struct set *set, uint32_t hash,
                                   const void *key, bool *found)
{
   assert(set->key_hash_function == NULL ||
          hash == set->key_hash_function(key));
   return set_search_or_add(set, hash, key, found);
}

/**
 * This function deletes the given hash table entry.
 *
 * Note that deletion doesn't otherwise modify the table, so an iteration over
 * the table deleting entries is safe.
 */
void
_mesa_set_remove(struct set *ht, struct set_entry *entry)
{
   if (!entry)
      return;

   entry->key = deleted_key;
   ht->entries--;
   ht->deleted_entries++;
}

/**
 * Removes the entry with the corresponding key, if exists.
 */
void
_mesa_set_remove_key(struct set *set, const void *key)
{
   _mesa_set_remove(set, _mesa_set_search(set, key));
}

/**
 * This function is an iterator over the set when no deleted entries are present.
 *
 * Pass in NULL for the first entry, as in the start of a for loop.
 */
struct set_entry *
_mesa_set_next_entry_unsafe(const struct set *ht, struct set_entry *entry)
{
   assert(!ht->deleted_entries);
   if (!ht->entries)
      return NULL;
   if (entry == NULL)
      entry = ht->table;
   else
      entry = entry + 1;
   if (entry != ht->table + ht->size)
      return entry->key ? entry : _mesa_set_next_entry_unsafe(ht, entry);

   return NULL;
}

/**
 * This function is an iterator over the hash table.
 *
 * Pass in NULL for the first entry, as in the start of a for loop.  Note that
 * an iteration over the table is O(table_size) not O(entries).
 */
struct set_entry *
_mesa_set_next_entry(const struct set *ht, struct set_entry *entry)
{
   if (entry == NULL)
      entry = ht->table;
   else
      entry = entry + 1;

   for (; entry != ht->table + ht->size; entry++) {
      if (entry_is_present(entry)) {
         return entry;
      }
   }

   return NULL;
}

/**
 * Helper to create a set with pointer keys.
 */
struct set *
_mesa_pointer_set_create(void *mem_ctx)
{
   return _mesa_set_create(mem_ctx, _mesa_hash_pointer,
                           _mesa_key_pointer_equal);
}

bool
_mesa_set_intersects(struct set *a, struct set *b)
{
   assert(a->key_hash_function == b->key_hash_function);
   assert(a->key_equals_function == b->key_equals_function);

   /* iterate over the set with less entries */
   if (b->entries < a->entries) {
      struct set *tmp = a;
      a = b;
      b = tmp;
   }

   set_foreach(a, entry) {
      if (_mesa_set_search_pre_hashed(b, entry->hash, entry->key))
         return true;
   }
   return false;
}
