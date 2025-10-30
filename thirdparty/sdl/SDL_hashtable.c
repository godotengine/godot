/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

typedef struct SDL_HashItem
{
    // TODO: Splitting off values into a separate array might be more cache-friendly
    const void *key;
    const void *value;
    Uint32 hash;
    Uint32 probe_len : 31;
    Uint32 live : 1;
} SDL_HashItem;

// Must be a power of 2 >= sizeof(SDL_HashItem)
#define MAX_HASHITEM_SIZEOF 32u
SDL_COMPILE_TIME_ASSERT(sizeof_SDL_HashItem, sizeof(SDL_HashItem) <= MAX_HASHITEM_SIZEOF);

// Anything larger than this will cause integer overflows
#define MAX_HASHTABLE_SIZE (0x80000000u / (MAX_HASHITEM_SIZEOF))

struct SDL_HashTable
{
    SDL_RWLock *lock;  // NULL if not created threadsafe
    SDL_HashItem *table;
    SDL_HashCallback hash;
    SDL_HashKeyMatchCallback keymatch;
    SDL_HashDestroyCallback destroy;
    void *userdata;
    Uint32 hash_mask;
    Uint32 max_probe_len;
    Uint32 num_occupied_slots;
};


static Uint32 CalculateHashBucketsFromEstimate(int estimated_capacity)
{
    if (estimated_capacity <= 0) {
        return 4;  // start small, grow as necessary.
    }

    const Uint32 estimated32 = (Uint32) estimated_capacity;
    Uint32 buckets = ((Uint32) 1) << SDL_MostSignificantBitIndex32(estimated32);
    if (!SDL_HasExactlyOneBitSet32(estimated32)) {
        buckets <<= 1;  // need next power of two up to fit overflow capacity bits.
    }

    return SDL_min(buckets, MAX_HASHTABLE_SIZE);
}

SDL_HashTable *SDL_CreateHashTable(int estimated_capacity, bool threadsafe, SDL_HashCallback hash,
                                   SDL_HashKeyMatchCallback keymatch,
                                   SDL_HashDestroyCallback destroy, void *userdata)
{
    const Uint32 num_buckets = CalculateHashBucketsFromEstimate(estimated_capacity);
    SDL_HashTable *table = (SDL_HashTable *)SDL_calloc(1, sizeof(SDL_HashTable));
    if (!table) {
        return NULL;
    }

    if (threadsafe) {
        table->lock = SDL_CreateRWLock();
        if (!table->lock) {
            SDL_DestroyHashTable(table);
            return NULL;
        }
    }

    table->table = (SDL_HashItem *)SDL_calloc(num_buckets, sizeof(SDL_HashItem));
    if (!table->table) {
        SDL_DestroyHashTable(table);
        return NULL;
    }

    table->hash_mask = num_buckets - 1;
    table->userdata = userdata;
    table->hash = hash;
    table->keymatch = keymatch;
    table->destroy = destroy;
    return table;
}

static SDL_INLINE Uint32 calc_hash(const SDL_HashTable *table, const void *key)
{
    const Uint32 BitMixer = 0x9E3779B1u;
    return table->hash(table->userdata, key) * BitMixer;
}

static SDL_INLINE Uint32 get_probe_length(Uint32 zero_idx, Uint32 actual_idx, Uint32 num_buckets)
{
    // returns the probe sequence length from zero_idx to actual_idx
    if (actual_idx < zero_idx) {
        return num_buckets - zero_idx + actual_idx;
    }

    return actual_idx - zero_idx;
}

static SDL_HashItem *find_item(const SDL_HashTable *ht, const void *key, Uint32 hash, Uint32 *i, Uint32 *probe_len)
{
    Uint32 hash_mask = ht->hash_mask;
    Uint32 max_probe_len = ht->max_probe_len;

    SDL_HashItem *table = ht->table;

    while (true) {
        SDL_HashItem *item = table + *i;
        Uint32 item_hash = item->hash;

        if (!item->live) {
            return NULL;
        }

        if (item_hash == hash && ht->keymatch(ht->userdata, item->key, key)) {
            return item;
        }

        Uint32 item_probe_len = item->probe_len;
        SDL_assert(item_probe_len == get_probe_length(item_hash & hash_mask, (Uint32)(item - table), hash_mask + 1));

        if (*probe_len > item_probe_len) {
            return NULL;
        }

        if (++*probe_len > max_probe_len) {
            return NULL;
        }

        *i = (*i + 1) & hash_mask;
    }
}

static SDL_HashItem *find_first_item(const SDL_HashTable *ht, const void *key, Uint32 hash)
{
    Uint32 i = hash & ht->hash_mask;
    Uint32 probe_len = 0;
    return find_item(ht, key, hash, &i, &probe_len);
}

static SDL_HashItem *insert_item(SDL_HashItem *item_to_insert, SDL_HashItem *table, Uint32 hash_mask, Uint32 *max_probe_len_ptr)
{
    const Uint32 num_buckets = hash_mask + 1;
    Uint32 idx = item_to_insert->hash & hash_mask;
    SDL_HashItem *target = NULL;
    SDL_HashItem temp_item;

    while (true) {
        SDL_HashItem *candidate = table + idx;

        if (!candidate->live) {
            // Found an empty slot. Put it here and we're done.
            *candidate = *item_to_insert;

            if (target == NULL) {
                target = candidate;
            }

            const Uint32 probe_len = get_probe_length(candidate->hash & hash_mask, idx, num_buckets);
            candidate->probe_len = probe_len;

            if (*max_probe_len_ptr < probe_len) {
                *max_probe_len_ptr = probe_len;
            }

            break;
        }

        const Uint32 candidate_probe_len = candidate->probe_len;
        SDL_assert(candidate_probe_len == get_probe_length(candidate->hash & hash_mask, idx, num_buckets));
        const Uint32 new_probe_len = get_probe_length(item_to_insert->hash & hash_mask, idx, num_buckets);

        if (candidate_probe_len < new_probe_len) {
            // Robin Hood hashing: the item at idx has a better probe length than our item would at this position.
            // Evict it and put our item in its place, then continue looking for a new spot for the displaced item.
            // This algorithm significantly reduces clustering in the table, making lookups take very few probes.

            temp_item = *candidate;
            *candidate = *item_to_insert;

            if (target == NULL) {
                target = candidate;
            }

            *item_to_insert = temp_item;

            SDL_assert(new_probe_len == get_probe_length(candidate->hash & hash_mask, idx, num_buckets));
            candidate->probe_len = new_probe_len;

            if (*max_probe_len_ptr < new_probe_len) {
                *max_probe_len_ptr = new_probe_len;
            }
        }

        idx = (idx + 1) & hash_mask;
    }

    return target;
}

static void delete_item(SDL_HashTable *ht, SDL_HashItem *item)
{
    const Uint32 hash_mask = ht->hash_mask;
    SDL_HashItem *table = ht->table;

    if (ht->destroy) {
        ht->destroy(ht->userdata, item->key, item->value);
    }

    SDL_assert(ht->num_occupied_slots > 0);
    ht->num_occupied_slots--;

    Uint32 idx = (Uint32)(item - ht->table);

    while (true) {
        idx = (idx + 1) & hash_mask;
        SDL_HashItem *next_item = table + idx;

        if (next_item->probe_len < 1) {
            SDL_zerop(item);
            return;
        }

        *item = *next_item;
        item->probe_len -= 1;
        SDL_assert(item->probe_len < ht->max_probe_len);
        item = next_item;
    }
}

static bool resize(SDL_HashTable *ht, Uint32 new_size)
{
    const Uint32 new_hash_mask = new_size - 1;
    SDL_HashItem *new_table = SDL_calloc(new_size, sizeof(*new_table));

    if (!new_table) {
        return false;
    }

    SDL_HashItem *old_table = ht->table;
    const Uint32 old_size = ht->hash_mask + 1;

    ht->max_probe_len = 0;
    ht->hash_mask = new_hash_mask;
    ht->table = new_table;

    for (Uint32 i = 0; i < old_size; ++i) {
        SDL_HashItem *item = old_table + i;
        if (item->live) {
            insert_item(item, new_table, new_hash_mask, &ht->max_probe_len);
        }
    }

    SDL_free(old_table);
    return true;
}

static bool maybe_resize(SDL_HashTable *ht)
{
    const Uint32 capacity = ht->hash_mask + 1;

    if (capacity >= MAX_HASHTABLE_SIZE) {
        return false;
    }

    const Uint32 max_load_factor = 217; // range: 0-255; 217 is roughly 85%
    const Uint32 resize_threshold = (Uint32)((max_load_factor * (Uint64)capacity) >> 8);

    if (ht->num_occupied_slots > resize_threshold) {
        return resize(ht, capacity * 2);
    }

    return true;
}

bool SDL_InsertIntoHashTable(SDL_HashTable *table, const void *key, const void *value, bool replace)
{
    if (!table) {
        return SDL_InvalidParamError("table");
    }

    bool result = false;

    SDL_LockRWLockForWriting(table->lock);

    const Uint32 hash = calc_hash(table, key);
    SDL_HashItem *item = find_first_item(table, key, hash);
    bool do_insert = true;

    if (item) {
        if (replace) {
            delete_item(table, item);
        } else {
            SDL_SetError("key already exists and replace is disabled");
            do_insert = false;
        }
    }

    if (do_insert) {
        SDL_HashItem new_item;
        new_item.key = key;
        new_item.value = value;
        new_item.hash = hash;
        new_item.live = true;
        new_item.probe_len = 0;

        table->num_occupied_slots++;

        if (!maybe_resize(table)) {
            table->num_occupied_slots--;
        } else {
            // This never returns NULL
            insert_item(&new_item, table->table, table->hash_mask, &table->max_probe_len);
            result = true;
        }
    }

    SDL_UnlockRWLock(table->lock);
    return result;
}

bool SDL_FindInHashTable(const SDL_HashTable *table, const void *key, const void **value)
{
    if (!table) {
        if (value) {
            *value = NULL;
        }
        return SDL_InvalidParamError("table");
    }

    SDL_LockRWLockForReading(table->lock);

    bool result = false;
    const Uint32 hash = calc_hash(table, key);
    SDL_HashItem *i = find_first_item(table, key, hash);
    if (i) {
        if (value) {
            *value = i->value;
        }
        result = true;
    }

    SDL_UnlockRWLock(table->lock);

    return result;
}

bool SDL_RemoveFromHashTable(SDL_HashTable *table, const void *key)
{
    if (!table) {
        return SDL_InvalidParamError("table");
    }

    SDL_LockRWLockForWriting(table->lock);

    bool result = false;
    const Uint32 hash = calc_hash(table, key);
    SDL_HashItem *item = find_first_item(table, key, hash);
    if (item) {
        delete_item(table, item);
        result = true;
    }

    SDL_UnlockRWLock(table->lock);
    return result;
}

bool SDL_IterateHashTable(const SDL_HashTable *table, SDL_HashTableIterateCallback callback, void *userdata)
{
    if (!table) {
        return SDL_InvalidParamError("table");
    } else if (!callback) {
        return SDL_InvalidParamError("callback");
    }

    SDL_LockRWLockForReading(table->lock);
    SDL_HashItem *end = table->table + (table->hash_mask + 1);
    Uint32 num_iterated = 0;

    for (SDL_HashItem *item = table->table; item < end; item++) {
        if (item->live) {
            if (!callback(userdata, table, item->key, item->value)) {
                break;  // callback requested iteration stop.
            } else if (++num_iterated >= table->num_occupied_slots) {
                break;  // we can drop out early because we've seen all the live items.
            }
        }
    }

    SDL_UnlockRWLock(table->lock);
    return true;
}

bool SDL_HashTableEmpty(SDL_HashTable *table)
{
    if (!table) {
        return SDL_InvalidParamError("table");
    }

    SDL_LockRWLockForReading(table->lock);
    const bool retval = (table->num_occupied_slots == 0);
    SDL_UnlockRWLock(table->lock);
    return retval;
}


static void destroy_all(SDL_HashTable *table)
{
    SDL_HashDestroyCallback destroy = table->destroy;
    if (destroy) {
        void *userdata = table->userdata;
        SDL_HashItem *end = table->table + (table->hash_mask + 1);
        for (SDL_HashItem *i = table->table; i < end; ++i) {
            if (i->live) {
                i->live = false;
                destroy(userdata, i->key, i->value);
            }
        }
    }
}

void SDL_ClearHashTable(SDL_HashTable *table)
{
    if (table) {
        SDL_LockRWLockForWriting(table->lock);
        {
            destroy_all(table);
            SDL_memset(table->table, 0, sizeof(*table->table) * (table->hash_mask + 1));
            table->num_occupied_slots = 0;
        }
        SDL_UnlockRWLock(table->lock);
    }
}

void SDL_DestroyHashTable(SDL_HashTable *table)
{
    if (table) {
        destroy_all(table);
        if (table->lock) {
            SDL_DestroyRWLock(table->lock);
        }
        SDL_free(table->table);
        SDL_free(table);
    }
}

// this is djb's xor hashing function.
static SDL_INLINE Uint32 hash_string_djbxor(const char *str, size_t len)
{
    Uint32 hash = 5381;
    while (len--) {
        hash = ((hash << 5) + hash) ^ *(str++);
    }
    return hash;
}

Uint32 SDL_HashPointer(void *unused, const void *key)
{
    (void)unused;
    return SDL_murmur3_32(&key, sizeof(key), 0);
}

bool SDL_KeyMatchPointer(void *unused, const void *a, const void *b)
{
    (void)unused;
    return (a == b);
}

Uint32 SDL_HashString(void *unused, const void *key)
{
    (void)unused;
    const char *str = (const char *)key;
    return hash_string_djbxor(str, SDL_strlen(str));
}

bool SDL_KeyMatchString(void *unused, const void *a, const void *b)
{
    const char *a_string = (const char *)a;
    const char *b_string = (const char *)b;

    (void)unused;
    if (a == b) {
        return true; // same pointer, must match.
    } else if (!a || !b) {
        return false; // one pointer is NULL (and first test shows they aren't the same pointer), must not match.
    } else if (a_string[0] != b_string[0]) {
        return false; // we know they don't match
    }
    return (SDL_strcmp(a_string, b_string) == 0); // Check against actual string contents.
}

// We assume we can fit the ID in the key directly
SDL_COMPILE_TIME_ASSERT(SDL_HashID_KeySize, sizeof(Uint32) <= sizeof(const void *));

Uint32 SDL_HashID(void *unused, const void *key)
{
    (void)unused;
    return (Uint32)(uintptr_t)key;
}

bool SDL_KeyMatchID(void *unused, const void *a, const void *b)
{
    (void)unused;
    return (a == b);
}

void SDL_DestroyHashKeyAndValue(void *unused, const void *key, const void *value)
{
    (void)unused;
    SDL_free((void *)key);
    SDL_free((void *)value);
}

void SDL_DestroyHashKey(void *unused, const void *key, const void *value)
{
    (void)value;
    (void)unused;
    SDL_free((void *)key);
}

void SDL_DestroyHashValue(void *unused, const void *key, const void *value)
{
    (void)key;
    (void)unused;
    SDL_free((void *)value);
}
