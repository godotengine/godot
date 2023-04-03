/**
 * \file hash.h
 * Generic hash table.
 */

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef HASH_H
#define HASH_H


#include <stdbool.h>
#include <stdint.h>
#include "util/glheader.h"

#include "c11/threads.h"
#include "util/simple_mtx.h"

struct util_idalloc;

/**
 * Magic GLuint object name that gets stored outside of the struct hash_table.
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

/** @{
 * Mapping from our use of GLuint as both the key and the hash value to the
 * hash_table.h API
 *
 * There exist many integer hash functions, designed to avoid collisions when
 * the integers are spread across key space with some patterns.  In GL, the
 * pattern (in the case of glGen*()ed object IDs) is that the keys are unique
 * contiguous integers starting from 1.  Because of that, we just use the key
 * as the hash value, to minimize the cost of the hash function.  If objects
 * are never deleted, we will never see a collision in the table, because the
 * table resizes itself when it approaches full, and thus key % table_size ==
 * key.
 *
 * The case where we could have collisions for genned objects would be
 * something like: glGenBuffers(&a, 100); glDeleteBuffers(&a + 50, 50);
 * glGenBuffers(&b, 100), because objects 1-50 and 101-200 are allocated at
 * the end of that sequence, instead of 1-150.  So far it doesn't appear to be
 * a problem.
 */
static inline bool
uint_key_compare(const void *a, const void *b)
{
   return a == b;
}

static inline uint32_t
uint_hash(GLuint id)
{
   return id;
}

static inline uint32_t
uint_key_hash(const void *key)
{
   return uint_hash((uintptr_t)key);
}

static inline void *
uint_key(GLuint id)
{
   return (void *)(uintptr_t) id;
}
/** @} */

/**
 * The hash table data structure.
 */
struct _mesa_HashTable {
   struct hash_table *ht;
   GLuint MaxKey;                        /**< highest key inserted so far */
   simple_mtx_t Mutex;                   /**< mutual exclusion lock */
   /* Used when name reuse is enabled */
   struct util_idalloc* id_alloc;

   /** Value that would be in the table for DELETED_KEY_VALUE. */
   void *deleted_key_data;
   #ifndef NDEBUG
   GLboolean InDeleteAll;                /**< Debug check */
   #endif
};

extern struct _mesa_HashTable *_mesa_NewHashTable(void);

extern void _mesa_DeleteHashTable(struct _mesa_HashTable *table);

extern void *_mesa_HashLookup(struct _mesa_HashTable *table, GLuint key);

extern void _mesa_HashInsert(struct _mesa_HashTable *table, GLuint key, void *data,
                             GLboolean isGenName);

extern void _mesa_HashRemove(struct _mesa_HashTable *table, GLuint key);

/**
 * Lock the hash table mutex.
 *
 * This function should be used when multiple objects need
 * to be looked up in the hash table, to avoid having to lock
 * and unlock the mutex each time.
 *
 * \param table the hash table.
 */
static inline void
_mesa_HashLockMutex(struct _mesa_HashTable *table)
{
   assert(table);
   simple_mtx_lock(&table->Mutex);
}


/**
 * Unlock the hash table mutex.
 *
 * \param table the hash table.
 */
static inline void
_mesa_HashUnlockMutex(struct _mesa_HashTable *table)
{
   assert(table);
   simple_mtx_unlock(&table->Mutex);
}

extern void *_mesa_HashLookupLocked(struct _mesa_HashTable *table, GLuint key);

extern void _mesa_HashInsertLocked(struct _mesa_HashTable *table,
                                   GLuint key, void *data, GLboolean isGenName);

extern void _mesa_HashRemoveLocked(struct _mesa_HashTable *table, GLuint key);

extern void
_mesa_HashDeleteAll(struct _mesa_HashTable *table,
                    void (*callback)(void *data, void *userData),
                    void *userData);

extern void
_mesa_HashWalk(const struct _mesa_HashTable *table,
               void (*callback)(void *data, void *userData),
               void *userData);

extern void
_mesa_HashWalkLocked(const struct _mesa_HashTable *table,
                     void (*callback)(void *data, void *userData),
                     void *userData);

extern void _mesa_HashPrint(const struct _mesa_HashTable *table);

extern GLuint _mesa_HashFindFreeKeyBlock(struct _mesa_HashTable *table, GLuint numKeys);

extern bool
_mesa_HashFindFreeKeys(struct _mesa_HashTable *table, GLuint* keys, GLuint numKeys);

extern GLuint
_mesa_HashNumEntries(const struct _mesa_HashTable *table);

extern void _mesa_test_hash_functions(void);

extern void _mesa_HashEnableNameReuse(struct _mesa_HashTable *table);

static inline void
_mesa_HashWalkMaybeLocked(const struct _mesa_HashTable *table,
                            void (*callback)(void *data, void *userData),
                            void *userData, bool locked)
{
   if (locked)
      _mesa_HashWalkLocked(table, callback, userData);
   else
      _mesa_HashWalk(table, callback, userData);
}

static inline struct gl_buffer_object *
_mesa_HashLookupMaybeLocked(struct _mesa_HashTable *table, GLuint key,
                            bool locked)
{
   if (locked)
      return _mesa_HashLookupLocked(table, key);
   else
      return _mesa_HashLookup(table, key);
}

static inline void
_mesa_HashInsertMaybeLocked(struct _mesa_HashTable *table,
                            GLuint key, void *data, GLboolean isGenName,
                            bool locked)
{
   if (locked)
      _mesa_HashInsertLocked(table, key, data, isGenName);
   else
      _mesa_HashInsert(table, key, data, isGenName);
}

static inline void
_mesa_HashLockMaybeLocked(struct _mesa_HashTable *table, bool locked)
{
   if (!locked)
      _mesa_HashLockMutex(table);
}

static inline void
_mesa_HashUnlockMaybeLocked(struct _mesa_HashTable *table, bool locked)
{
   if (!locked)
      _mesa_HashUnlockMutex(table);
}

#endif
