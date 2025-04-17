/*
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
 * Author(s): Behdad Esfahbod
 */

#ifndef HB_BIT_VECTOR_HH
#define HB_BIT_VECTOR_HH

#include "hb.hh"

#include "hb-atomic.hh"

struct hb_min_max_t
{
  void add (hb_codepoint_t v) { min_v = hb_min (min_v, v); max_v = hb_max (max_v, v); }
  void add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    min_v = hb_min (min_v, a);
    max_v = hb_max (max_v, b);
  }

  template <typename set_t>
  void union_ (const set_t &set)
  {
    hb_codepoint_t set_min = set.get_min ();
    if (unlikely (set_min == HB_CODEPOINT_INVALID))
      return;
    hb_codepoint_t set_max = set.get_max ();
    min_v = hb_min (min_v, set_min);
    max_v = hb_max (max_v, set_max);
  }

  hb_codepoint_t get_min () const { return min_v; }
  hb_codepoint_t get_max () const { return max_v; }

  private:
  hb_codepoint_t min_v = HB_CODEPOINT_INVALID;
  hb_codepoint_t max_v = 0;
};

template <bool atomic = false>
struct hb_bit_vector_t
{
  using int_t = uint64_t;
  using elt_t = typename std::conditional<atomic, hb_atomic_t<int_t>, int_t>::type;

  hb_bit_vector_t () = delete;
  hb_bit_vector_t (const hb_bit_vector_t &other) = delete;
  hb_bit_vector_t &operator= (const hb_bit_vector_t &other) = delete;

  // Move
  hb_bit_vector_t (hb_bit_vector_t &&other)
		: min_v (other.min_v), max_v (other.max_v), count (other.count), elts (other.elts)
  {
    other.min_v = other.max_v = other.count = 0;
    other.elts = nullptr;
  }
  hb_bit_vector_t &operator= (hb_bit_vector_t &&other)
  {
    hb_swap (min_v, other.min_v);
    hb_swap (max_v, other.max_v);
    hb_swap (count, other.count);
    hb_swap (elts, other.elts);
    return *this;
  }

  hb_bit_vector_t (unsigned min_v, unsigned max_v)
    : min_v (min_v), max_v (max_v)
  {
    if (unlikely (min_v >= max_v))
    {
      min_v = max_v = count = 0;
      return;
    }

    unsigned num = (max_v - min_v + sizeof (int_t) * 8) / (sizeof (int_t) * 8);
    elts = (elt_t *) hb_calloc (num, sizeof (int_t));
    if (unlikely (!elts))
    {
      min_v = max_v = count = 0;
      return;
    }

    count = max_v - min_v + 1;
  }
  ~hb_bit_vector_t ()
  {
    hb_free (elts);
  }

  void add (hb_codepoint_t g) { elt (g) |= mask (g); }
  void del (hb_codepoint_t g) { elt (g) &= ~mask (g); }
  void set (hb_codepoint_t g, bool value) { if (value) add (g); else del (g); }
  bool get (hb_codepoint_t g) const { return elt (g) & mask (g); }
  bool has (hb_codepoint_t g) const { return get (g); }
  bool may_have (hb_codepoint_t g) const { return get (g); }

  bool operator [] (hb_codepoint_t g) const { return get (g); }
  bool operator () (hb_codepoint_t g) const { return get (g); }

  void add_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if (unlikely (!count || a > b || a < min_v || b > max_v))
      return;

    elt_t *la = &elt (a);
    elt_t *lb = &elt (b);
    if (la == lb)
      *la |= (mask (b) << 1) - mask(a);
    else
    {
      *la |= ~(mask (a) - 1llu);
      la++;

      hb_memset (la, 0xff, (char *) lb - (char *) la);

      *lb |= ((mask (b) << 1) - 1llu);
    }
  }
  void del_range (hb_codepoint_t a, hb_codepoint_t b)
  {
    if (unlikely (!count || a > b || a < min_v || b > max_v))
      return;

    elt_t *la = &elt (a);
    elt_t *lb = &elt (b);
    if (la == lb)
      *la &= ~((mask (b) << 1llu) - mask(a));
    else
    {
      *la &= mask (a) - 1;
      la++;

      hb_memset (la, 0, (char *) lb - (char *) la);

      *lb &= ~((mask (b) << 1) - 1llu);
    }
  }
  void set_range (hb_codepoint_t a, hb_codepoint_t b, bool v)
  { if (v) add_range (a, b); else del_range (a, b); }

  template <typename set_t>
  void union_ (const set_t &set)
  {
    for (hb_codepoint_t g : set)
      add (g);
  }

  static const unsigned int ELT_BITS = sizeof (elt_t) * 8;
  static constexpr unsigned ELT_MASK = ELT_BITS - 1;

  static constexpr elt_t zero = 0;

  elt_t &elt (hb_codepoint_t g)
  {
    g -= min_v;
    if (unlikely (g >= count))
      return Crap(elt_t);
    return elts[g / ELT_BITS];
  }
  const elt_t& elt (hb_codepoint_t g) const
  {
    g -= min_v;
    if (unlikely (g >= count))
      return Null(elt_t);
    return elts[g / ELT_BITS];
  }

  static constexpr int_t mask (hb_codepoint_t g) { return elt_t (1) << (g & ELT_MASK); }

  hb_codepoint_t min_v = 0, max_v = 0, count = 0;
  elt_t *elts = nullptr;
};


#endif /* HB_BIT_VECTOR_HH */
