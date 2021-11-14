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

#ifndef HB_SET_DIGEST_HH
#define HB_SET_DIGEST_HH

#include "hb.hh"

/*
 * The set digests here implement various "filters" that support
 * "approximate member query".  Conceptually these are like Bloom
 * Filter and Quotient Filter, however, much smaller, faster, and
 * designed to fit the requirements of our uses for glyph coverage
 * queries.
 *
 * Our filters are highly accurate if the lookup covers fairly local
 * set of glyphs, but fully flooded and ineffective if coverage is
 * all over the place.
 *
 * The frozen-set can be used instead of a digest, to trade more
 * memory for 100% accuracy, but in practice, that doesn't look like
 * an attractive trade-off.
 */

template <typename mask_t, unsigned int shift>
struct hb_set_digest_lowest_bits_t
{
  static constexpr unsigned mask_bytes = sizeof (mask_t);
  static constexpr unsigned mask_bits = sizeof (mask_t) * 8;
  static constexpr unsigned num_bits = 0
				     + (mask_bytes >= 1 ? 3 : 0)
				     + (mask_bytes >= 2 ? 1 : 0)
				     + (mask_bytes >= 4 ? 1 : 0)
				     + (mask_bytes >= 8 ? 1 : 0)
				     + (mask_bytes >= 16? 1 : 0)
				     + 0;

  static_assert ((shift < sizeof (hb_codepoint_t) * 8), "");
  static_assert ((shift + num_bits <= sizeof (hb_codepoint_t) * 8), "");

  void init () { mask = 0; }

  void add (hb_codepoint_t g) { mask |= mask_for (g); }

  bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if ((b >> shift) - (a >> shift) >= mask_bits - 1)
      mask = (mask_t) -1;
    else {
      mask_t ma = mask_for (a);
      mask_t mb = mask_for (b);
      mask |= mb + (mb - ma) - (mb < ma);
    }
    return true;
  }

  template <typename T>
  void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    for (unsigned int i = 0; i < count; i++)
    {
      add (*array);
      array = (const T *) (stride + (const char *) array);
    }
  }
  template <typename T>
  void add_array (const hb_array_t<const T>& arr) { add_array (&arr, arr.len ()); }
  template <typename T>
  bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    for (unsigned int i = 0; i < count; i++)
    {
      add (*array);
      array = (const T *) (stride + (const char *) array);
    }
    return true;
  }
  template <typename T>
  bool add_sorted_array (const hb_sorted_array_t<const T>& arr) { return add_sorted_array (&arr, arr.len ()); }

  bool may_have (hb_codepoint_t g) const
  { return !!(mask & mask_for (g)); }

  private:

  static mask_t mask_for (hb_codepoint_t g)
  { return ((mask_t) 1) << ((g >> shift) & (mask_bits - 1)); }
  mask_t mask;
};

template <typename head_t, typename tail_t>
struct hb_set_digest_combiner_t
{
  void init ()
  {
    head.init ();
    tail.init ();
  }

  void add (hb_codepoint_t g)
  {
    head.add (g);
    tail.add (g);
  }

  bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    head.add_range (a, b);
    tail.add_range (a, b);
    return true;
  }
  template <typename T>
  void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    head.add_array (array, count, stride);
    tail.add_array (array, count, stride);
  }
  template <typename T>
  void add_array (const hb_array_t<const T>& arr) { add_array (&arr, arr.len ()); }
  template <typename T>
  bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    head.add_sorted_array (array, count, stride);
    tail.add_sorted_array (array, count, stride);
    return true;
  }
  template <typename T>
  bool add_sorted_array (const hb_sorted_array_t<const T>& arr) { return add_sorted_array (&arr, arr.len ()); }

  bool may_have (hb_codepoint_t g) const
  {
    return head.may_have (g) && tail.may_have (g);
  }

  private:
  head_t head;
  tail_t tail;
};


/*
 * hb_set_digest_t
 *
 * This is a combination of digests that performs "best".
 * There is not much science to this: it's a result of intuition
 * and testing.
 */
using hb_set_digest_t =
  hb_set_digest_combiner_t
  <
    hb_set_digest_lowest_bits_t<unsigned long, 4>,
    hb_set_digest_combiner_t
    <
      hb_set_digest_lowest_bits_t<unsigned long, 0>,
      hb_set_digest_lowest_bits_t<unsigned long, 9>
    >
  >
;


#endif /* HB_SET_DIGEST_HH */
