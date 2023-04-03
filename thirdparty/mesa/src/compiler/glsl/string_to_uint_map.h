/*
 * Copyright © 2008 Intel Corporation
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
 */

#ifndef STRING_TO_UINT_MAP_H
#define STRING_TO_UINT_MAP_H

#include <string.h>
#include <limits.h>
#include "util/hash_table.h"

struct string_to_uint_map;

#ifdef __cplusplus
extern "C" {
#endif

struct string_to_uint_map *
string_to_uint_map_ctor();

void
string_to_uint_map_dtor(struct string_to_uint_map *);


#ifdef __cplusplus
}

struct string_map_iterate_wrapper_closure {
   void (*callback)(const char *key, unsigned value, void *closure);
   void *closure;
};

/**
 * Map from a string (name) to an unsigned integer value
 *
 * \note
 * Because of the way this class interacts with the \c hash_table
 * implementation, values of \c UINT_MAX cannot be stored in the map.
 */
struct string_to_uint_map {
public:
   string_to_uint_map()
   {
      this->ht = _mesa_hash_table_create(NULL, _mesa_hash_string,
                                         _mesa_key_string_equal);
   }

   ~string_to_uint_map()
   {
      hash_table_call_foreach(this->ht, delete_key, NULL);
      _mesa_hash_table_destroy(this->ht, NULL);
   }

   /**
    * Remove all mappings from this map.
    */
   void clear()
   {
      hash_table_call_foreach(this->ht, delete_key, NULL);
      _mesa_hash_table_clear(this->ht, NULL);
   }

   /**
    * Runs a passed callback for the hash
    */
   void iterate(void (*func)(const char *, unsigned, void *), void *closure)
   {
      struct string_map_iterate_wrapper_closure *wrapper;

      wrapper = (struct string_map_iterate_wrapper_closure *)
         malloc(sizeof(struct string_map_iterate_wrapper_closure));
      if (wrapper == NULL)
         return;

      wrapper->callback = func;
      wrapper->closure = closure;

      hash_table_call_foreach(this->ht, subtract_one_wrapper, wrapper);
      free(wrapper);
   }

   /**
    * Get the value associated with a particular key
    *
    * \return
    * If \c key is found in the map, \c true is returned.  Otherwise \c false
    * is returned.
    *
    * \note
    * If \c key is not found in the table, \c value is not modified.
    */
   bool get(unsigned &value, const char *key)
   {
      hash_entry *entry = _mesa_hash_table_search(this->ht,
                                                  (const void *) key);

      if (!entry)
         return false;

      const intptr_t v = (intptr_t) entry->data;
      value = (unsigned)(v - 1);
      return true;
   }

   void put(unsigned value, const char *key)
   {
      /* The low-level hash table structure returns NULL if key is not in the
       * hash table.  However, users of this map might want to store zero as a
       * valid value in the table.  Bias the value by +1 so that a
       * user-specified zero is stored as 1.  This enables ::get to tell the
       * difference between a user-specified zero (returned as 1 by
       * _mesa_hash_table_search) and the key not in the table (returned as 0 by
       * _mesa_hash_table_search).
       *
       * The net effect is that we can't store UINT_MAX in the table.  This is
       * because UINT_MAX+1 = 0.
       */
      assert(value != UINT_MAX);
      char *dup_key = strdup(key);

      struct hash_entry *entry = _mesa_hash_table_search(this->ht, dup_key);
      if (entry) {
         entry->data = (void *) (intptr_t) (value + 1);
      } else {
         _mesa_hash_table_insert(this->ht, dup_key,
                                 (void *) (intptr_t) (value + 1));
      }

      if (entry)
         free(dup_key);
   }

private:
   static void delete_key(const void *key, void *data, void *closure)
   {
      (void) data;
      (void) closure;

      free((char *)key);
   }

   static void subtract_one_wrapper(const void *key, void *data, void *closure)
   {
      struct string_map_iterate_wrapper_closure *wrapper =
         (struct string_map_iterate_wrapper_closure *) closure;
      unsigned value = (intptr_t) data;

      value -= 1;

      wrapper->callback((const char *) key, value, wrapper->closure);
   }

   struct hash_table *ht;
};

#endif /* __cplusplus */
#endif /* STRING_TO_UINT_MAP_H */
