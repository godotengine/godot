/*
 * Copyright © 2018  Google, Inc.
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

#ifndef HB_MAP_PRIVATE_HH
#define HB_MAP_PRIVATE_HH

#include "hb-private.hh"


template <typename T>
inline uint32_t Hash (const T &v)
{
  /* Knuth's multiplicative method: */
  return (uint32_t) v * 2654435761u;
}


/*
 * hb_map_t
 */

struct hb_map_t
{
  struct item_t
  {
    hb_codepoint_t key;
    hb_codepoint_t value;

    inline bool is_unused (void) const { return key == INVALID; }
    inline bool is_tombstone (void) const { return key != INVALID && value == INVALID; }
  };

  hb_object_header_t header;
  bool successful; /* Allocations successful */
  unsigned int population; /* Not including tombstones. */
  unsigned int occupancy; /* Including tombstones. */
  unsigned int mask;
  unsigned int prime;
  item_t *items;

  inline void init_shallow (void)
  {
    successful = true;
    population = occupancy = 0;
    mask = 0;
    prime = 0;
    items = nullptr;
  }
  inline void init (void)
  {
    hb_object_init (this);
    init_shallow ();
  }
  inline void fini_shallow (void)
  {
    free (items);
  }
  inline void fini (void)
  {
    hb_object_fini (this);
    fini_shallow ();
  }

  inline bool resize (void)
  {
    if (unlikely (!successful)) return false;

    unsigned int power = hb_bit_storage (population * 2 + 8);
    unsigned int new_size = 1u << power;
    item_t *new_items = (item_t *) malloc ((size_t) new_size * sizeof (item_t));
    if (unlikely (!new_items))
    {
      successful = false;
      return false;
    }
    memset (new_items, 0xFF, (size_t) new_size * sizeof (item_t));

    unsigned int old_size = mask + 1;
    item_t *old_items = items;

    /* Switch to new, empty, array. */
    population = occupancy = 0;
    mask = new_size - 1;
    prime = prime_for (power);
    items = new_items;

    /* Insert back old items. */
    if (old_items)
      for (unsigned int i = 0; i < old_size; i++)
	if (old_items[i].key != INVALID && old_items[i].value != INVALID)
	  set (old_items[i].key, old_items[i].value);

    free (old_items);

    return true;
  }

  inline void set (hb_codepoint_t key, hb_codepoint_t value)
  {
    if (unlikely (!successful)) return;
    if (unlikely (key == INVALID)) return;
    if ((occupancy + occupancy / 2) >= mask && !resize ()) return;
    unsigned int i = bucket_for (key);

    if (value == INVALID && items[i].key != key)
      return; /* Trying to delete non-existent key. */

    if (!items[i].is_unused ())
    {
      occupancy--;
      if (items[i].is_tombstone ())
	population--;
    }

    items[i].key = key;
    items[i].value = value;

    occupancy++;
    if (!items[i].is_tombstone ())
      population++;

  }
  inline hb_codepoint_t get (hb_codepoint_t key) const
  {
    if (unlikely (!items)) return INVALID;
    unsigned int i = bucket_for (key);
    return items[i].key == key ? items[i].value : INVALID;
  }

  inline void del (hb_codepoint_t key)
  {
    set (key, INVALID);
  }
  inline bool has (hb_codepoint_t key) const
  {
    return get (key) != INVALID;
  }

  inline hb_codepoint_t operator [] (unsigned int key) const
  { return get (key); }

  static const hb_codepoint_t INVALID = HB_MAP_VALUE_INVALID;

  inline void clear (void)
  {
    memset (items, 0xFF, ((size_t) mask + 1) * sizeof (item_t));
    population = occupancy = 0;
  }

  inline bool is_empty (void) const
  {
    return population != 0;
  }

  inline unsigned int get_population () const
  {
    return population;
  }

  protected:

  inline unsigned int bucket_for (hb_codepoint_t key) const
  {
    unsigned int i = Hash (key) % prime;
    unsigned int step = 0;
    unsigned int tombstone = INVALID;
    while (!items[i].is_unused ())
    {
      if (items[i].key == key)
        return i;
      if (tombstone == INVALID && items[i].is_tombstone ())
        tombstone = i;
      i = (i + ++step) & mask;
    }
    return tombstone == INVALID ? i : tombstone;
  }

  static inline unsigned int prime_for (unsigned int shift)
  {
    /* Following comment and table copied from glib. */
    /* Each table size has an associated prime modulo (the first prime
     * lower than the table size) used to find the initial bucket. Probing
     * then works modulo 2^n. The prime modulo is necessary to get a
     * good distribution with poor hash functions.
     */
    /* Not declaring static to make all kinds of compilers happy... */
    /*static*/ const unsigned int prime_mod [32] =
    {
      1,          /* For 1 << 0 */
      2,
      3,
      7,
      13,
      31,
      61,
      127,
      251,
      509,
      1021,
      2039,
      4093,
      8191,
      16381,
      32749,
      65521,      /* For 1 << 16 */
      131071,
      262139,
      524287,
      1048573,
      2097143,
      4194301,
      8388593,
      16777213,
      33554393,
      67108859,
      134217689,
      268435399,
      536870909,
      1073741789,
      2147483647  /* For 1 << 31 */
    };

    if (unlikely (shift >= ARRAY_LENGTH (prime_mod)))
      return prime_mod[ARRAY_LENGTH (prime_mod) - 1];

    return prime_mod[shift];
  }
};


#endif /* HB_MAP_PRIVATE_HH */
