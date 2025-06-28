/*
 * Copyright © 2012  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_CACHE_HH
#define HB_CACHE_HH

#include "hb.hh"


/* Implements a lockfree and thread-safe cache for int->int functions,
 * using (optionally) _relaxed_ atomic integer operations.
 *
 * The cache is a fixed-size array of 16-bit or 32-bit integers,
 * typically 256 elements.
 *
 * The key is split into two parts: the cache index (high bits)
 * and the rest (low bits).
 *
 * The cache index is used to index into the array.  The array
 * member is a 16-bit or 32-bit integer that is used *both*
 * to store the low bits of the key, and the value.
 *
 * The value is stored in the least significant bits of the integer.
 * The low bits of the key are stored in the most significant bits
 * of the integer.
 *
 * A cache hit is detected by comparing the low bits of the key
 * with the high bits of the integer at the array position indexed
 * by the high bits of the key. If they match, the value is extracted
 * from the least significant bits of the integer and returned.
 * Otherwise, a cache miss is reported.
 *
 * Cache operations (storage and retrieval) involve just a few
 * arithmetic operations and a single memory access.
 */

template <unsigned int key_bits=16,
	 unsigned int value_bits=8 + 32 - key_bits,
	 unsigned int cache_bits=8,
	 bool thread_safe=true>
struct hb_cache_t
{
  using item_t = typename std::conditional<thread_safe,
					   typename std::conditional<key_bits + value_bits - cache_bits <= 16,
								     hb_atomic_t<unsigned short>,
								     hb_atomic_t<unsigned int>>::type,
					   typename std::conditional<key_bits + value_bits - cache_bits <= 16,
								     unsigned short,
								     unsigned int>::type
					  >::type;

  static_assert ((key_bits >= cache_bits), "");
  static_assert ((key_bits + value_bits <= cache_bits + 8 * sizeof (item_t)), "");

  hb_cache_t () { clear (); }

  void clear ()
  {
    for (auto &v : values)
      v = -1;
  }

  bool get (unsigned int key, unsigned int *value) const
  {
    unsigned int k = key & ((1u<<cache_bits)-1);
    unsigned int v = values[k];
    if ((key_bits + value_bits - cache_bits == 8 * sizeof (item_t) && v == (unsigned int) -1) ||
	(v >> value_bits) != (key >> cache_bits))
      return false;
    *value = v & ((1u<<value_bits)-1);
    return true;
  }

  bool set (unsigned int key, unsigned int value)
  {
    if (unlikely ((key >> key_bits) || (value >> value_bits)))
      return false; /* Overflows */
    unsigned int k = key & ((1u<<cache_bits)-1);
    unsigned int v = ((key>>cache_bits)<<value_bits) | value;
    values[k] = v;
    return true;
  }

  private:
  item_t values[1u<<cache_bits];
};


#endif /* HB_CACHE_HH */
