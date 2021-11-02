/*
 * Copyright © 2019 Adobe Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */

#ifndef HB_BIMAP_HH
#define HB_BIMAP_HH

#include "hb.hh"
#include "hb-map.hh"

/* Bi-directional map */
struct hb_bimap_t
{
  hb_bimap_t () { init (); }
  ~hb_bimap_t () { fini (); }

  void init ()
  {
    forw_map.init ();
    back_map.init ();
  }

  void fini ()
  {
    forw_map.fini ();
    back_map.fini ();
  }

  void reset ()
  {
    forw_map.reset ();
    back_map.reset ();
  }

  bool in_error () const { return forw_map.in_error () || back_map.in_error (); }

  void set (hb_codepoint_t lhs, hb_codepoint_t rhs)
  {
    if (in_error ()) return;
    if (unlikely (lhs == HB_MAP_VALUE_INVALID)) return;
    if (unlikely (rhs == HB_MAP_VALUE_INVALID)) { del (lhs); return; }

    forw_map.set (lhs, rhs);
    if (in_error ()) return;

    back_map.set (rhs, lhs);
    if (in_error ()) forw_map.del (lhs);
  }

  hb_codepoint_t get (hb_codepoint_t lhs) const { return forw_map.get (lhs); }
  hb_codepoint_t backward (hb_codepoint_t rhs) const { return back_map.get (rhs); }

  hb_codepoint_t operator [] (hb_codepoint_t lhs) const { return get (lhs); }
  bool has (hb_codepoint_t lhs, hb_codepoint_t *vp = nullptr) const { return forw_map.has (lhs, vp); }

  void del (hb_codepoint_t lhs)
  {
    back_map.del (get (lhs));
    forw_map.del (lhs);
  }

  void clear ()
  {
    forw_map.clear ();
    back_map.clear ();
  }

  bool is_empty () const { return get_population () == 0; }

  unsigned int get_population () const { return forw_map.get_population (); }

  protected:
  hb_map_t  forw_map;
  hb_map_t  back_map;
};

/* Inremental bimap: only lhs is given, rhs is incrementally assigned */
struct hb_inc_bimap_t : hb_bimap_t
{
  hb_inc_bimap_t () { init (); }

  void init ()
  {
    hb_bimap_t::init ();
    next_value = 0;
  }

  /* Add a mapping from lhs to rhs with a unique value if lhs is unknown.
   * Return the rhs value as the result.
   */
  hb_codepoint_t add (hb_codepoint_t lhs)
  {
    hb_codepoint_t  rhs = forw_map[lhs];
    if (rhs == HB_MAP_VALUE_INVALID)
    {
      rhs = next_value++;
      set (lhs, rhs);
    }
    return rhs;
  }

  hb_codepoint_t skip ()
  { return next_value++; }

  hb_codepoint_t get_next_value () const
  { return next_value; }

  void add_set (const hb_set_t *set)
  {
    hb_codepoint_t i = HB_SET_VALUE_INVALID;
    while (hb_set_next (set, &i)) add (i);
  }

  /* Create an identity map. */
  bool identity (unsigned int size)
  {
    clear ();
    for (hb_codepoint_t i = 0; i < size; i++) set (i, i);
    return !in_error ();
  }

  protected:
  static int cmp_id (const void* a, const void* b)
  { return (int)*(const hb_codepoint_t *)a - (int)*(const hb_codepoint_t *)b; }

  public:
  /* Optional: after finished adding all mappings in a random order,
   * reassign rhs to lhs so that they are in the same order. */
  void sort ()
  {
    hb_codepoint_t  count = get_population ();
    hb_vector_t <hb_codepoint_t> work;
    work.resize (count);

    for (hb_codepoint_t rhs = 0; rhs < count; rhs++)
      work[rhs] = back_map[rhs];

    work.qsort (cmp_id);

    clear ();
    for (hb_codepoint_t rhs = 0; rhs < count; rhs++)
      set (work[rhs], rhs);
  }

  protected:
  unsigned int	next_value;
};

#endif /* HB_BIMAP_HH */
