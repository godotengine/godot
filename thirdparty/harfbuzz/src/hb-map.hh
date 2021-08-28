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

#ifndef HB_MAP_HH
#define HB_MAP_HH

#include "hb.hh"


/*
 * hb_hashmap_t
 */

template <typename K, typename V,
	  K kINVALID = hb_is_pointer (K) ? 0 : hb_is_signed (K) ? hb_int_min (K) : (K) -1,
	  V vINVALID = hb_is_pointer (V) ? 0 : hb_is_signed (V) ? hb_int_min (V) : (V) -1>
struct hb_hashmap_t
{
  HB_DELETE_COPY_ASSIGN (hb_hashmap_t);
  hb_hashmap_t ()  { init (); }
  ~hb_hashmap_t () { fini (); }

  static_assert (hb_is_integral (K) || hb_is_pointer (K), "");
  static_assert (hb_is_integral (V) || hb_is_pointer (V), "");

  struct item_t
  {
    K key;
    V value;
    uint32_t hash;

    void clear () { key = kINVALID; value = vINVALID; hash = 0; }

    bool operator == (const K &o) { return hb_deref (key) == hb_deref (o); }
    bool operator == (const item_t &o) { return *this == o.key; }
    bool is_unused () const    { return key == kINVALID; }
    bool is_tombstone () const { return key != kINVALID && value == vINVALID; }
    bool is_real () const { return key != kINVALID && value != vINVALID; }
    hb_pair_t<K, V> get_pair() const { return hb_pair_t<K, V> (key, value); }
  };

  hb_object_header_t header;
  bool successful; /* Allocations successful */
  unsigned int population; /* Not including tombstones. */
  unsigned int occupancy; /* Including tombstones. */
  unsigned int mask;
  unsigned int prime;
  item_t *items;

  void init_shallow ()
  {
    successful = true;
    population = occupancy = 0;
    mask = 0;
    prime = 0;
    items = nullptr;
  }
  void init ()
  {
    hb_object_init (this);
    init_shallow ();
  }
  void fini_shallow ()
  {
    free (items);
    items = nullptr;
    population = occupancy = 0;
  }
  void fini ()
  {
    hb_object_fini (this);
    fini_shallow ();
  }

  void reset ()
  {
    successful = true;
    clear ();
  }

  bool in_error () const { return !successful; }

  bool resize ()
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
    for (auto &_ : hb_iter (new_items, new_size))
      _.clear ();

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
	if (old_items[i].is_real ())
	  set_with_hash (old_items[i].key,
			 old_items[i].hash,
			 old_items[i].value);

    free (old_items);

    return true;
  }

  void set (K key, V value)
  {
    set_with_hash (key, hb_hash (key), value);
  }

  V get (K key) const
  {
    if (unlikely (!items)) return vINVALID;
    unsigned int i = bucket_for (key);
    return items[i].is_real () && items[i] == key ? items[i].value : vINVALID;
  }

  void del (K key) { set (key, vINVALID); }

  /* Has interface. */
  static constexpr V SENTINEL = vINVALID;
  typedef V value_t;
  value_t operator [] (K k) const { return get (k); }
  bool has (K k, V *vp = nullptr) const
  {
    V v = (*this)[k];
    if (vp) *vp = v;
    return v != SENTINEL;
  }
  /* Projection. */
  V operator () (K k) const { return get (k); }

  void clear ()
  {
    if (items)
      for (auto &_ : hb_iter (items, mask + 1))
	_.clear ();

    population = occupancy = 0;
  }

  bool is_empty () const { return population == 0; }
  explicit operator bool () const { return !is_empty (); }

  unsigned int get_population () const { return population; }

  /*
   * Iterator
   */
  auto iter () const HB_AUTO_RETURN
  (
    + hb_array (items, mask ? mask + 1 : 0)
    | hb_filter (&item_t::is_real)
    | hb_map (&item_t::get_pair)
  )
  auto keys () const HB_AUTO_RETURN
  (
    + hb_array (items, mask ? mask + 1 : 0)
    | hb_filter (&item_t::is_real)
    | hb_map (&item_t::key)
    | hb_map (hb_ridentity)
  )
  auto values () const HB_AUTO_RETURN
  (
    + hb_array (items, mask ? mask + 1 : 0)
    | hb_filter (&item_t::is_real)
    | hb_map (&item_t::value)
    | hb_map (hb_ridentity)
  )

  /* Sink interface. */
  hb_hashmap_t& operator << (const hb_pair_t<K, V>& v)
  { set (v.first, v.second); return *this; }

  protected:

  void set_with_hash (K key, uint32_t hash, V value)
  {
    if (unlikely (!successful)) return;
    if (unlikely (key == kINVALID)) return;
    if ((occupancy + occupancy / 2) >= mask && !resize ()) return;
    unsigned int i = bucket_for_hash (key, hash);

    if (value == vINVALID && items[i].key != key)
      return; /* Trying to delete non-existent key. */

    if (!items[i].is_unused ())
    {
      occupancy--;
      if (items[i].is_tombstone ())
	population--;
    }

    items[i].key = key;
    items[i].value = value;
    items[i].hash = hash;

    occupancy++;
    if (!items[i].is_tombstone ())
      population++;
  }

  unsigned int bucket_for (K key) const
  {
    return bucket_for_hash (key, hb_hash (key));
  }

  unsigned int bucket_for_hash (K key, uint32_t hash) const
  {
    unsigned int i = hash % prime;
    unsigned int step = 0;
    unsigned int tombstone = (unsigned) -1;
    while (!items[i].is_unused ())
    {
      if (items[i].hash == hash && items[i] == key)
	return i;
      if (tombstone == (unsigned) -1 && items[i].is_tombstone ())
	tombstone = i;
      i = (i + ++step) & mask;
    }
    return tombstone == (unsigned) -1 ? i : tombstone;
  }

  static unsigned int prime_for (unsigned int shift)
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

/*
 * hb_map_t
 */

struct hb_map_t : hb_hashmap_t<hb_codepoint_t,
			       hb_codepoint_t,
			       HB_MAP_VALUE_INVALID,
			       HB_MAP_VALUE_INVALID> {};


#endif /* HB_MAP_HH */
