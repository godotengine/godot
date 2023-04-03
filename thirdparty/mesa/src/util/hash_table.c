/*
 * Copyright © 2009,2012 Intel Corporation
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

/**
 * Implements an open-addressing, linear-reprobing hash table.
 *
 * For more information, see:
 *
 * http://cgit.freedesktop.org/~anholt/hash_table/tree/README
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hash_table.h"
#include "ralloc.h"
#include "macros.h"
#include "u_memory.h"
#include "fast_urem_by_const.h"
#include "util/u_memory.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

/**
 * Magic number that gets stored outside of the struct hash_table.
 *
 * The hash table needs a particular pointer to be the marker for a key that
 * was deleted from the table, along with NULL for the "never allocated in the
 * table" marker.  Legacy GL allows any GLuint to be used as a GL object name,
 * and we use a 1:1 mapping from GLuints to key pointers, so we need to be
 * able to track a GLuint that happens to match the deleted key outside of
 * struct hash_table.  We tell the hash table to use "1" as the deleted key
 * value, so that we test the deleted-key-in-the-table path as best we can.
 */
#define DELETED_KEY_VALUE 1

static inline void *
uint_key(unsigned id)
{
   return (void *)(uintptr_t) id;
}

static const uint32_t deleted_key_value;

/**
 * From Knuth -- a good choice for hash/rehash values is p, p-2 where
 * p and p-2 are both prime.  These tables are sized to have an extra 10%
 * free to avoid exponential performance degradation as the hash table fills
 */
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
key_pointer_is_reserved(const struct hash_table *ht, const void *key)
{
   return key == NULL || key == ht->deleted_key;
}

static int
entry_is_free(const struct hash_entry *entry)
{
   return entry->key == NULL;
}

static int
entry_is_deleted(const struct hash_table *ht, struct hash_entry *entry)
{
   return entry->key == ht->deleted_key;
}

static int
entry_is_present(const struct hash_table *ht, struct hash_entry *entry)
{
   return entry->key != NULL && entry->key != ht->deleted_key;
}

bool
_mesa_hash_table_init(struct hash_table *ht,
                      void *mem_ctx,
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
   ht->table = rzalloc_array(mem_ctx, struct hash_entry, ht->size);
   ht->entries = 0;
   ht->deleted_entries = 0;
   ht->deleted_key = &deleted_key_value;

   return ht->table != NULL;
}

struct hash_table *
_mesa_hash_table_create(void *mem_ctx,
                        uint32_t (*key_hash_function)(const void *key),
                        bool (*key_equals_function)(const void *a,
                                                    const void *b))
{
   struct hash_table *ht;

   /* mem_ctx is used to allocate the hash table, but the hash table is used
    * to allocate all of the suballocations.
    */
   ht = ralloc(mem_ctx, struct hash_table);
   if (ht == NULL)
      return NULL;

   if (!_mesa_hash_table_init(ht, ht, key_hash_function, key_equals_function)) {
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
struct hash_table *
_mesa_hash_table_create_u32_keys(void *mem_ctx)
{
   return _mesa_hash_table_create(mem_ctx, key_u32_hash, key_u32_equals);
}

struct hash_table *
_mesa_hash_table_clone(struct hash_table *src, void *dst_mem_ctx)
{
   struct hash_table *ht;

   ht = ralloc(dst_mem_ctx, struct hash_table);
   if (ht == NULL)
      return NULL;

   memcpy(ht, src, sizeof(struct hash_table));

   ht->table = ralloc_array(ht, struct hash_entry, ht->size);
   if (ht->table == NULL) {
      ralloc_free(ht);
      return NULL;
   }

   memcpy(ht->table, src->table, ht->size * sizeof(struct hash_entry));

   return ht;
}

/**
 * Frees the given hash table.
 *
 * If delete_function is passed, it gets called on each entry present before
 * freeing.
 */
void
_mesa_hash_table_destroy(struct hash_table *ht,
                         void (*delete_function)(struct hash_entry *entry))
{
   if (!ht)
      return;

   if (delete_function) {
      hash_table_foreach(ht, entry) {
         delete_function(entry);
      }
   }
   ralloc_free(ht);
}

static void
hash_table_clear_fast(struct hash_table *ht)
{
   memset(ht->table, 0, sizeof(struct hash_entry) * hash_sizes[ht->size_index].size);
   ht->entries = ht->deleted_entries = 0;
}

/**
 * Deletes all entries of the given hash table without deleting the table
 * itself or changing its structure.
 *
 * If delete_function is passed, it gets called on each entry present.
 */
void
_mesa_hash_table_clear(struct hash_table *ht,
                       void (*delete_function)(struct hash_entry *entry))
{
   if (!ht)
      return;

   struct hash_entry *entry;

   if (delete_function) {
      for (entry = ht->table; entry != ht->table + ht->size; entry++) {
         if (entry_is_present(ht, entry))
            delete_function(entry);

         entry->key = NULL;
      }
      ht->entries = 0;
      ht->deleted_entries = 0;
   } else
      hash_table_clear_fast(ht);
}

/** Sets the value of the key pointer used for deleted entries in the table.
 *
 * The assumption is that usually keys are actual pointers, so we use a
 * default value of a pointer to an arbitrary piece of storage in the library.
 * But in some cases a consumer wants to store some other sort of value in the
 * table, like a uint32_t, in which case that pointer may conflict with one of
 * their valid keys.  This lets that user select a safe value.
 *
 * This must be called before any keys are actually deleted from the table.
 */
void
_mesa_hash_table_set_deleted_key(struct hash_table *ht, const void *deleted_key)
{
   ht->deleted_key = deleted_key;
}

static struct hash_entry *
hash_table_search(struct hash_table *ht, uint32_t hash, const void *key)
{
   assert(!key_pointer_is_reserved(ht, key));

   uint32_t size = ht->size;
   uint32_t start_hash_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = 1 + util_fast_urem32(hash, ht->rehash,
                                               ht->rehash_magic);
   uint32_t hash_address = start_hash_address;

   do {
      struct hash_entry *entry = ht->table + hash_address;

      if (entry_is_free(entry)) {
         return NULL;
      } else if (entry_is_present(ht, entry) && entry->hash == hash) {
         if (ht->key_equals_function(key, entry->key)) {
            return entry;
         }
      }

      hash_address += double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (hash_address != start_hash_address);

   return NULL;
}

/**
 * Finds a hash table entry with the given key and hash of that key.
 *
 * Returns NULL if no entry is found.  Note that the data pointer may be
 * modified by the user.
 */
struct hash_entry *
_mesa_hash_table_search(struct hash_table *ht, const void *key)
{
   assert(ht->key_hash_function);
   return hash_table_search(ht, ht->key_hash_function(key), key);
}

struct hash_entry *
_mesa_hash_table_search_pre_hashed(struct hash_table *ht, uint32_t hash,
                                  const void *key)
{
   assert(ht->key_hash_function == NULL || hash == ht->key_hash_function(key));
   return hash_table_search(ht, hash, key);
}

static struct hash_entry *
hash_table_insert(struct hash_table *ht, uint32_t hash,
                  const void *key, void *data);

static void
hash_table_insert_rehash(struct hash_table *ht, uint32_t hash,
                         const void *key, void *data)
{
   uint32_t size = ht->size;
   uint32_t start_hash_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = 1 + util_fast_urem32(hash, ht->rehash,
                                               ht->rehash_magic);
   uint32_t hash_address = start_hash_address;
   do {
      struct hash_entry *entry = ht->table + hash_address;

      if (likely(entry->key == NULL)) {
         entry->hash = hash;
         entry->key = key;
         entry->data = data;
         return;
      }

      hash_address += double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (true);
}

static void
_mesa_hash_table_rehash(struct hash_table *ht, unsigned new_size_index)
{
   struct hash_table old_ht;
   struct hash_entry *table;

   if (ht->size_index == new_size_index && ht->deleted_entries == ht->max_entries) {
      hash_table_clear_fast(ht);
      assert(!ht->entries);
      return;
   }

   if (new_size_index >= ARRAY_SIZE(hash_sizes))
      return;

   table = rzalloc_array(ralloc_parent(ht->table), struct hash_entry,
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

   hash_table_foreach(&old_ht, entry) {
      hash_table_insert_rehash(ht, entry->hash, entry->key, entry->data);
   }

   ht->entries = old_ht.entries;

   ralloc_free(old_ht.table);
}

static struct hash_entry *
hash_table_insert(struct hash_table *ht, uint32_t hash,
                  const void *key, void *data)
{
   struct hash_entry *available_entry = NULL;

   assert(!key_pointer_is_reserved(ht, key));

   if (ht->entries >= ht->max_entries) {
      _mesa_hash_table_rehash(ht, ht->size_index + 1);
   } else if (ht->deleted_entries + ht->entries >= ht->max_entries) {
      _mesa_hash_table_rehash(ht, ht->size_index);
   }

   uint32_t size = ht->size;
   uint32_t start_hash_address = util_fast_urem32(hash, size, ht->size_magic);
   uint32_t double_hash = 1 + util_fast_urem32(hash, ht->rehash,
                                               ht->rehash_magic);
   uint32_t hash_address = start_hash_address;
   do {
      struct hash_entry *entry = ht->table + hash_address;

      if (!entry_is_present(ht, entry)) {
         /* Stash the first available entry we find */
         if (available_entry == NULL)
            available_entry = entry;
         if (entry_is_free(entry))
            break;
      }

      /* Implement replacement when another insert happens
       * with a matching key.  This is a relatively common
       * feature of hash tables, with the alternative
       * generally being "insert the new value as well, and
       * return it first when the key is searched for".
       *
       * Note that the hash table doesn't have a delete
       * callback.  If freeing of old data pointers is
       * required to avoid memory leaks, perform a search
       * before inserting.
       */
      if (!entry_is_deleted(ht, entry) &&
          entry->hash == hash &&
          ht->key_equals_function(key, entry->key)) {
         entry->key = key;
         entry->data = data;
         return entry;
      }

      hash_address += double_hash;
      if (hash_address >= size)
         hash_address -= size;
   } while (hash_address != start_hash_address);

   if (available_entry) {
      if (entry_is_deleted(ht, available_entry))
         ht->deleted_entries--;
      available_entry->hash = hash;
      available_entry->key = key;
      available_entry->data = data;
      ht->entries++;
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
struct hash_entry *
_mesa_hash_table_insert(struct hash_table *ht, const void *key, void *data)
{
   assert(ht->key_hash_function);
   return hash_table_insert(ht, ht->key_hash_function(key), key, data);
}

struct hash_entry *
_mesa_hash_table_insert_pre_hashed(struct hash_table *ht, uint32_t hash,
                                   const void *key, void *data)
{
   assert(ht->key_hash_function == NULL || hash == ht->key_hash_function(key));
   return hash_table_insert(ht, hash, key, data);
}

/**
 * This function deletes the given hash table entry.
 *
 * Note that deletion doesn't otherwise modify the table, so an iteration over
 * the table deleting entries is safe.
 */
void
_mesa_hash_table_remove(struct hash_table *ht,
                        struct hash_entry *entry)
{
   if (!entry)
      return;

   entry->key = ht->deleted_key;
   ht->entries--;
   ht->deleted_entries++;
}

/**
 * Removes the entry with the corresponding key, if exists.
 */
void _mesa_hash_table_remove_key(struct hash_table *ht,
                                 const void *key)
{
   _mesa_hash_table_remove(ht, _mesa_hash_table_search(ht, key));
}

/**
 * This function is an iterator over the hash_table when no deleted entries are present.
 *
 * Pass in NULL for the first entry, as in the start of a for loop.
 */
struct hash_entry *
_mesa_hash_table_next_entry_unsafe(const struct hash_table *ht, struct hash_entry *entry)
{
   assert(!ht->deleted_entries);
   if (!ht->entries)
      return NULL;
   if (entry == NULL)
      entry = ht->table;
   else
      entry = entry + 1;
   if (entry != ht->table + ht->size)
      return entry->key ? entry : _mesa_hash_table_next_entry_unsafe(ht, entry);

   return NULL;
}

/**
 * This function is an iterator over the hash table.
 *
 * Pass in NULL for the first entry, as in the start of a for loop.  Note that
 * an iteration over the table is O(table_size) not O(entries).
 */
struct hash_entry *
_mesa_hash_table_next_entry(struct hash_table *ht,
                            struct hash_entry *entry)
{
   if (entry == NULL)
      entry = ht->table;
   else
      entry = entry + 1;

   for (; entry != ht->table + ht->size; entry++) {
      if (entry_is_present(ht, entry)) {
         return entry;
      }
   }

   return NULL;
}

/**
 * Returns a random entry from the hash table.
 *
 * This may be useful in implementing random replacement (as opposed
 * to just removing everything) in caches based on this hash table
 * implementation.  @predicate may be used to filter entries, or may
 * be set to NULL for no filtering.
 */
struct hash_entry *
_mesa_hash_table_random_entry(struct hash_table *ht,
                              bool (*predicate)(struct hash_entry *entry))
{
   struct hash_entry *entry;
   uint32_t i = rand() % ht->size;

   if (ht->entries == 0)
      return NULL;

   for (entry = ht->table + i; entry != ht->table + ht->size; entry++) {
      if (entry_is_present(ht, entry) &&
          (!predicate || predicate(entry))) {
         return entry;
      }
   }

   for (entry = ht->table; entry != ht->table + i; entry++) {
      if (entry_is_present(ht, entry) &&
          (!predicate || predicate(entry))) {
         return entry;
      }
   }

   return NULL;
}


uint32_t
_mesa_hash_data(const void *data, size_t size)
{
   return XXH32(data, size, 0);
}

uint32_t
_mesa_hash_data_with_seed(const void *data, size_t size, uint32_t seed)
{
   return XXH32(data, size, seed);
}

uint32_t
_mesa_hash_int(const void *key)
{
   return XXH32(key, sizeof(int), 0);
}

uint32_t
_mesa_hash_uint(const void *key)
{
   return XXH32(key, sizeof(unsigned), 0);
}

uint32_t
_mesa_hash_u32(const void *key)
{
   return XXH32(key, 4, 0);
}

/** FNV-1a string hash implementation */
uint32_t
_mesa_hash_string(const void *_key)
{
   return _mesa_hash_string_with_length(_key, strlen((const char *)_key));
}

uint32_t
_mesa_hash_string_with_length(const void *_key, unsigned length)
{
   uint32_t hash = 0;
   const char *key = _key;
#if defined(_WIN64) || defined(__x86_64__)
   hash = (uint32_t)XXH64(key, length, hash);
#else
   hash = XXH32(key, length, hash);
#endif
   return hash;
}

uint32_t
_mesa_hash_pointer(const void *pointer)
{
   uintptr_t num = (uintptr_t) pointer;
   return (uint32_t) ((num >> 2) ^ (num >> 6) ^ (num >> 10) ^ (num >> 14));
}

bool
_mesa_key_int_equal(const void *a, const void *b)
{
   return *((const int *)a) == *((const int *)b);
}

bool
_mesa_key_uint_equal(const void *a, const void *b)
{

   return *((const unsigned *)a) == *((const unsigned *)b);
}

bool
_mesa_key_u32_equal(const void *a, const void *b)
{
   return *((const uint32_t *)a) == *((const uint32_t *)b);
}

/**
 * String compare function for use as the comparison callback in
 * _mesa_hash_table_create().
 */
bool
_mesa_key_string_equal(const void *a, const void *b)
{
   return strcmp(a, b) == 0;
}

bool
_mesa_key_pointer_equal(const void *a, const void *b)
{
   return a == b;
}

/**
 * Helper to create a hash table with pointer keys.
 */
struct hash_table *
_mesa_pointer_hash_table_create(void *mem_ctx)
{
   return _mesa_hash_table_create(mem_ctx, _mesa_hash_pointer,
                                  _mesa_key_pointer_equal);
}


bool
_mesa_hash_table_reserve(struct hash_table *ht, unsigned size)
{
   if (size < ht->max_entries)
      return true;
   for (unsigned i = ht->size_index + 1; i < ARRAY_SIZE(hash_sizes); i++) {
      if (hash_sizes[i].max_entries >= size) {
         _mesa_hash_table_rehash(ht, i);
         break;
      }
   }
   return ht->max_entries >= size;
}

/**
 * Hash table wrapper which supports 64-bit keys.
 *
 * TODO: unify all hash table implementations.
 */

struct hash_key_u64 {
   uint64_t value;
};

static uint32_t
key_u64_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct hash_key_u64));
}

static bool
key_u64_equals(const void *a, const void *b)
{
   const struct hash_key_u64 *aa = a;
   const struct hash_key_u64 *bb = b;

   return aa->value == bb->value;
}

#define FREED_KEY_VALUE 0

struct hash_table_u64 *
_mesa_hash_table_u64_create(void *mem_ctx)
{
   STATIC_ASSERT(FREED_KEY_VALUE != DELETED_KEY_VALUE);
   struct hash_table_u64 *ht;

   ht = CALLOC_STRUCT(hash_table_u64);
   if (!ht)
      return NULL;

   if (sizeof(void *) == 8) {
      ht->table = _mesa_hash_table_create(mem_ctx, _mesa_hash_pointer,
                                          _mesa_key_pointer_equal);
   } else {
      ht->table = _mesa_hash_table_create(mem_ctx, key_u64_hash,
                                          key_u64_equals);
   }

   if (ht->table)
      _mesa_hash_table_set_deleted_key(ht->table, uint_key(DELETED_KEY_VALUE));

   return ht;
}

static void
_mesa_hash_table_u64_delete_key(struct hash_entry *entry)
{
   if (sizeof(void *) == 8)
      return;

   struct hash_key_u64 *_key = (struct hash_key_u64 *)entry->key;

   if (_key)
      free(_key);
}

void
_mesa_hash_table_u64_clear(struct hash_table_u64 *ht)
{
   if (!ht)
      return;

   _mesa_hash_table_clear(ht->table, _mesa_hash_table_u64_delete_key);
   ht->freed_key_data = NULL;
   ht->deleted_key_data = NULL;
}

void
_mesa_hash_table_u64_destroy(struct hash_table_u64 *ht)
{
   if (!ht)
      return;

   _mesa_hash_table_u64_clear(ht);
   _mesa_hash_table_destroy(ht->table, NULL);
   free(ht);
}

void
_mesa_hash_table_u64_insert(struct hash_table_u64 *ht, uint64_t key,
                            void *data)
{
   if (key == FREED_KEY_VALUE) {
      ht->freed_key_data = data;
      return;
   }

   if (key == DELETED_KEY_VALUE) {
      ht->deleted_key_data = data;
      return;
   }

   if (sizeof(void *) == 8) {
      _mesa_hash_table_insert(ht->table, (void *)(uintptr_t)key, data);
   } else {
      struct hash_key_u64 *_key = CALLOC_STRUCT(hash_key_u64);

      if (!_key)
         return;
      _key->value = key;

      _mesa_hash_table_insert(ht->table, _key, data);
   }
}

static struct hash_entry *
hash_table_u64_search(struct hash_table_u64 *ht, uint64_t key)
{
   if (sizeof(void *) == 8) {
      return _mesa_hash_table_search(ht->table, (void *)(uintptr_t)key);
   } else {
      struct hash_key_u64 _key = { .value = key };
      return _mesa_hash_table_search(ht->table, &_key);
   }
}

void *
_mesa_hash_table_u64_search(struct hash_table_u64 *ht, uint64_t key)
{
   struct hash_entry *entry;

   if (key == FREED_KEY_VALUE)
      return ht->freed_key_data;

   if (key == DELETED_KEY_VALUE)
      return ht->deleted_key_data;

   entry = hash_table_u64_search(ht, key);
   if (!entry)
      return NULL;

   return entry->data;
}

void
_mesa_hash_table_u64_remove(struct hash_table_u64 *ht, uint64_t key)
{
   struct hash_entry *entry;

   if (key == FREED_KEY_VALUE) {
      ht->freed_key_data = NULL;
      return;
   }

   if (key == DELETED_KEY_VALUE) {
      ht->deleted_key_data = NULL;
      return;
   }

   entry = hash_table_u64_search(ht, key);
   if (!entry)
      return;

   if (sizeof(void *) == 8) {
      _mesa_hash_table_remove(ht->table, entry);
   } else {
      struct hash_key *_key = (struct hash_key *)entry->key;

      _mesa_hash_table_remove(ht->table, entry);
      free(_key);
   }
}
