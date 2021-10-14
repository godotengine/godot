/*
 * Copyright © 2012,2017  Google, Inc.
 * Copyright © 2021 Behdad Esfahbod
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

#ifndef HB_BIT_SET_INVERTIBLE_HH
#define HB_BIT_SET_INVERTIBLE_HH

#include "hb.hh"
#include "hb-bit-set.hh"


struct hb_bit_set_invertible_t
{
  hb_bit_set_t s;
  bool inverted;

  hb_bit_set_invertible_t () { init (); }
  ~hb_bit_set_invertible_t () { fini (); }

  void init () { s.init (); inverted = false; }
  void fini () { s.fini (); }
  void err () { s.err (); }
  bool in_error () const { return s.in_error (); }
  explicit operator bool () const { return !is_empty (); }

  void reset ()
  {
    s.reset ();
    inverted = false;
  }
  void clear ()
  {
    s.clear ();
    if (likely (s.successful))
      inverted = false;
  }
  void invert ()
  {
    if (likely (s.successful))
      inverted = !inverted;
  }

  bool is_empty () const
  {
    hb_codepoint_t v = INVALID;
    next (&v);
    return v == INVALID;
  }
  hb_codepoint_t get_min () const
  {
    hb_codepoint_t v = INVALID;
    next (&v);
    return v;
  }
  hb_codepoint_t get_max () const
  {
    hb_codepoint_t v = INVALID;
    previous (&v);
    return v;
  }
  unsigned int get_population () const
  { return inverted ? INVALID - s.get_population () : s.get_population (); }


  void add (hb_codepoint_t g) { unlikely (inverted) ? s.del (g) : s.add (g); }
  bool add_range (hb_codepoint_t a, hb_codepoint_t b)
  { return unlikely (inverted) ? (s.del_range (a, b), true) : s.add_range (a, b); }

  template <typename T>
  void add_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { inverted ? s.del_array (array, count, stride) : s.add_array (array, count, stride); }
  template <typename T>
  void add_array (const hb_array_t<const T>& arr) { add_array (&arr, arr.len ()); }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename T>
  bool add_sorted_array (const T *array, unsigned int count, unsigned int stride=sizeof(T))
  { return inverted ? s.del_sorted_array (array, count, stride) : s.add_sorted_array (array, count, stride); }
  template <typename T>
  bool add_sorted_array (const hb_sorted_array_t<const T>& arr) { return add_sorted_array (&arr, arr.len ()); }

  void del (hb_codepoint_t g) { unlikely (inverted) ? s.add (g) : s.del (g); }
  void del_range (hb_codepoint_t a, hb_codepoint_t b)
  { unlikely (inverted) ? (void) s.add_range (a, b) : s.del_range (a, b); }

  bool get (hb_codepoint_t g) const { return s.get (g) ^ inverted; }

  /* Has interface. */
  static constexpr bool SENTINEL = false;
  typedef bool value_t;
  value_t operator [] (hb_codepoint_t k) const { return get (k); }
  bool has (hb_codepoint_t k) const { return (*this)[k] != SENTINEL; }
  /* Predicate. */
  bool operator () (hb_codepoint_t k) const { return has (k); }

  /* Sink interface. */
  hb_bit_set_invertible_t& operator << (hb_codepoint_t v)
  { add (v); return *this; }
  hb_bit_set_invertible_t& operator << (const hb_pair_t<hb_codepoint_t, hb_codepoint_t>& range)
  { add_range (range.first, range.second); return *this; }

  bool intersects (hb_codepoint_t first, hb_codepoint_t last) const
  {
    hb_codepoint_t c = first - 1;
    return next (&c) && c <= last;
  }

  void set (const hb_bit_set_invertible_t &other)
  {
    s.set (other.s);
    if (likely (s.successful))
      inverted = other.inverted;
  }

  bool is_equal (const hb_bit_set_invertible_t &other) const
  {
    if (likely (inverted == other.inverted))
      return s.is_equal (other.s);
    else
    {
      /* TODO Add iter_ranges() and use here. */
      auto it1 = iter ();
      auto it2 = other.iter ();
      return hb_all (+ hb_zip (it1, it2)
		     | hb_map ([](hb_pair_t<hb_codepoint_t, hb_codepoint_t> _) { return _.first == _.second; }));
    }
  }

  bool is_subset (const hb_bit_set_invertible_t &larger_set) const
  {
    if (unlikely (inverted != larger_set.inverted))
      return hb_all (hb_iter (s) | hb_map (larger_set.s));
    else
      return unlikely (inverted) ? larger_set.s.is_subset (s) : s.is_subset (larger_set.s);
  }

  protected:
  template <typename Op>
  void process (const Op& op, const hb_bit_set_invertible_t &other)
  { s.process (op, other.s); }
  public:
  void union_ (const hb_bit_set_invertible_t &other)
  {
    if (likely (inverted == other.inverted))
    {
      if (unlikely (inverted))
	process (hb_bitwise_and, other);
      else
	process (hb_bitwise_or, other); /* Main branch. */
    }
    else
    {
      if (unlikely (inverted))
	process (hb_bitwise_gt, other);
      else
	process (hb_bitwise_lt, other);
    }
    if (likely (s.successful))
      inverted = inverted || other.inverted;
  }
  void intersect (const hb_bit_set_invertible_t &other)
  {
    if (likely (inverted == other.inverted))
    {
      if (unlikely (inverted))
	process (hb_bitwise_or, other);
      else
	process (hb_bitwise_and, other); /* Main branch. */
    }
    else
    {
      if (unlikely (inverted))
	process (hb_bitwise_lt, other);
      else
	process (hb_bitwise_gt, other);
    }
    if (likely (s.successful))
      inverted = inverted && other.inverted;
  }
  void subtract (const hb_bit_set_invertible_t &other)
  {
    if (likely (inverted == other.inverted))
    {
      if (unlikely (inverted))
	process (hb_bitwise_lt, other);
      else
	process (hb_bitwise_gt, other); /* Main branch. */
    }
    else
    {
      if (unlikely (inverted))
	process (hb_bitwise_or, other);
      else
	process (hb_bitwise_and, other);
    }
    if (likely (s.successful))
      inverted = inverted && !other.inverted;
  }
  void symmetric_difference (const hb_bit_set_invertible_t &other)
  {
    process (hb_bitwise_xor, other);
    if (likely (s.successful))
      inverted = inverted ^ other.inverted;
  }

  bool next (hb_codepoint_t *codepoint) const
  {
    if (likely (!inverted))
      return s.next (codepoint);

    auto old = *codepoint;
    if (unlikely (old + 1 == INVALID))
    {
      *codepoint = INVALID;
      return false;
    }

    auto v = old;
    s.next (&v);
    if (old + 1 < v)
    {
      *codepoint = old + 1;
      return true;
    }

    v = old;
    s.next_range (&old, &v);

    *codepoint = v + 1;
    return *codepoint != INVALID;
  }
  bool previous (hb_codepoint_t *codepoint) const
  {
    if (likely (!inverted))
      return s.previous (codepoint);

    auto old = *codepoint;
    if (unlikely (old - 1 == INVALID))
    {
      *codepoint = INVALID;
      return false;
    }

    auto v = old;
    s.previous (&v);

    if (old - 1 > v || v == INVALID)
    {
      *codepoint = old - 1;
      return true;
    }

    v = old;
    s.previous_range (&v, &old);

    *codepoint = v - 1;
    return *codepoint != INVALID;
  }
  bool next_range (hb_codepoint_t *first, hb_codepoint_t *last) const
  {
    if (likely (!inverted))
      return s.next_range (first, last);

    if (!next (last))
    {
      *last = *first = INVALID;
      return false;
    }

    *first = *last;
    s.next (last);
    --*last;
    return true;
  }
  bool previous_range (hb_codepoint_t *first, hb_codepoint_t *last) const
  {
    if (likely (!inverted))
      return s.previous_range (first, last);

    if (!previous (first))
    {
      *last = *first = INVALID;
      return false;
    }

    *last = *first;
    s.previous (first);
    ++*first;
    return true;
  }

  static constexpr hb_codepoint_t INVALID = hb_bit_set_t::INVALID;

  /*
   * Iterator implementation.
   */
  struct iter_t : hb_iter_with_fallback_t<iter_t, hb_codepoint_t>
  {
    static constexpr bool is_sorted_iterator = true;
    iter_t (const hb_bit_set_invertible_t &s_ = Null (hb_bit_set_invertible_t),
	    bool init = true) : s (&s_), v (INVALID), l(0)
    {
      if (init)
      {
	l = s->get_population () + 1;
	__next__ ();
      }
    }

    typedef hb_codepoint_t __item_t__;
    hb_codepoint_t __item__ () const { return v; }
    bool __more__ () const { return v != INVALID; }
    void __next__ () { s->next (&v); if (l) l--; }
    void __prev__ () { s->previous (&v); }
    unsigned __len__ () const { return l; }
    iter_t end () const { return iter_t (*s, false); }
    bool operator != (const iter_t& o) const
    { return s != o.s || v != o.v; }

    protected:
    const hb_bit_set_invertible_t *s;
    hb_codepoint_t v;
    unsigned l;
  };
  iter_t iter () const { return iter_t (*this); }
  operator iter_t () const { return iter (); }
};


#endif /* HB_BIT_SET_INVERTIBLE_HH */
