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
#include "hb-machinery.hh"

/*
 * The set-digests here implement various "filters" that support
 * "approximate member query".  Conceptually these are like Bloom
 * Filter and Quotient Filter, however, much smaller, faster, and
 * designed to fit the requirements of our uses for glyph coverage
 * queries.
 *
 * Our filters are highly accurate if the lookup covers fairly local
 * set of glyphs, but fully flooded and ineffective if coverage is
 * all over the place.
 *
 * The way these are used is that the filter is first populated by
 * a lookup's or subtable's Coverage table(s), and then when we
 * want to apply the lookup or subtable to a glyph, before trying
 * to apply, we ask the filter if the glyph may be covered. If it's
 * not, we return early.  We can also match a digest against another
 * digest.
 *
 * We use these filters at three levels:
 *   - If the digest for all the glyphs in the buffer as a whole
 *     does not match the digest for the lookup, skip the lookup.
 *   - For each glyph, if it doesn't match the lookup digest,
 *     skip it.
 *   - For each glyph, if it doesn't match the subtable digest,
 *     skip it.
 *
 * The main filter we use is a combination of four bits-pattern
 * filters. A bits-pattern filter checks a number of bits (5 or 6)
 * of the input number (glyph-id in this case) and checks whether
 * its pattern is amongst the patterns of any of the accepted values.
 * The accepted patterns are represented as a "long" integer. The
 * check is done using four bitwise operations only.
 */

static constexpr unsigned hb_set_digest_shifts[] = {4, 0, 6};

struct hb_set_digest_t
{
  // No science in these. Intuition and testing only.
  using mask_t = uint64_t;

  static constexpr unsigned n = ARRAY_LENGTH_CONST (hb_set_digest_shifts);
  static constexpr unsigned mask_bytes = sizeof (mask_t);
  static constexpr unsigned mask_bits = sizeof (mask_t) * 8;
  static constexpr hb_codepoint_t mb1 = mask_bits - 1;
  static constexpr mask_t one = 1;
  static constexpr mask_t all = (mask_t) -1;

  void init ()
  { for (unsigned i = 0; i < n; i++) masks[i] = 0; }

  void clear () { init (); }

  static hb_set_digest_t full ()
  {
    hb_set_digest_t d;
    for (unsigned i = 0; i < n; i++) d.masks[i] = all;
    return d;
  }

  void union_ (const hb_set_digest_t &o)
  { for (unsigned i = 0; i < n; i++) masks[i] |= o.masks[i]; }

  bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    bool ret;

    ret = false;
    for (unsigned i = 0; i < n; i++)
      if (masks[i] != all)
	ret = true;
    if (!ret) return false;

    ret = false;
    for (unsigned i = 0; i < n; i++)
    {
      mask_t shift = hb_set_digest_shifts[i];
      if ((b >> shift) - (a >> shift) >= mb1)
	masks[i] = all;
      else
      {
	mask_t ma = one << ((a >> shift) & mb1);
	mask_t mb = one << ((b >> shift) & mb1);
	masks[i] |= mb + (mb - ma) - (mb < ma);
	ret = true;
      }
    }
    return ret;
  }

  template <typename T>
  void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    for (unsigned int i = 0; i < count; i++)
    {
      add (*array);
      array = &StructAtOffsetUnaligned<T> ((const void *) array, stride);
    }
  }
  template <typename T>
  void add_array (const hb_array_t<const T>& arr) { add_array (&arr, arr.len ()); }
  template <typename T>
  bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  {
    add_array (array, count, stride);
    return true;
  }
  template <typename T>
  bool add_sorted_array (const hb_sorted_array_t<const T>& arr) { return add_sorted_array (&arr, arr.len ()); }

  bool operator [] (hb_codepoint_t g) const
  { return may_have (g); }


  void add (hb_codepoint_t g)
  {
    for (unsigned i = 0; i < n; i++)
      masks[i] |= one << ((g >> hb_set_digest_shifts[i]) & mb1);
  }

  HB_ALWAYS_INLINE
  bool may_have (hb_codepoint_t g) const
  {
    for (unsigned i = 0; i < n; i++)
      if (!(masks[i] & (one << ((g >> hb_set_digest_shifts[i]) & mb1))))
	return false;
    return true;
  }

  bool may_intersect (const hb_set_digest_t &o) const
  {
    for (unsigned i = 0; i < n; i++)
      if (!(masks[i] & o.masks[i]))
	return false;
    return true;
  }

  private:

  mask_t masks[n] = {};
};


#endif /* HB_SET_DIGEST_HH */
