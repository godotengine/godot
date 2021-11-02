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


/* Implements a lockfree cache for int->int functions. */

template <unsigned int key_bits, unsigned int value_bits, unsigned int cache_bits>
struct hb_cache_t
{
  static_assert ((key_bits >= cache_bits), "");
  static_assert ((key_bits + value_bits - cache_bits <= 8 * sizeof (hb_atomic_int_t)), "");
  static_assert (sizeof (hb_atomic_int_t) == sizeof (unsigned int), "");

  void init () { clear (); }
  void fini () {}

  void clear ()
  {
    for (unsigned i = 0; i < ARRAY_LENGTH (values); i++)
      values[i].set_relaxed (-1);
  }

  bool get (unsigned int key, unsigned int *value) const
  {
    unsigned int k = key & ((1u<<cache_bits)-1);
    unsigned int v = values[k].get_relaxed ();
    if ((key_bits + value_bits - cache_bits == 8 * sizeof (hb_atomic_int_t) && v == (unsigned int) -1) ||
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
    values[k].set_relaxed (v);
    return true;
  }

  private:
  hb_atomic_int_t values[1u<<cache_bits];
};

typedef hb_cache_t<21, 16, 8> hb_cmap_cache_t;
typedef hb_cache_t<16, 24, 8> hb_advance_cache_t;


#endif /* HB_CACHE_HH */
