/*
 * Copyright © 2022  Behdad Esfahbod
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
 */

#ifndef HB_MULTIMAP_HH
#define HB_MULTIMAP_HH

#include "hb.hh"
#include "hb-map.hh"
#include "hb-vector.hh"


/*
 * hb_multimap_t
 */

struct hb_multimap_t
{
  void add (hb_codepoint_t k, hb_codepoint_t v)
  {
    hb_vector_t<hb_codepoint_t> *m;
    if (multiples.has (k, &m))
    {
      m->push (v);
      return;
    }

    hb_codepoint_t *old_v;
    if (singulars.has (k, &old_v))
    {
      hb_codepoint_t old = *old_v;
      singulars.del (k);

      multiples.set (k, hb_vector_t<hb_codepoint_t> {old, v});
      return;
    }

    singulars.set (k, v);
  }

  hb_array_t<const hb_codepoint_t> get (hb_codepoint_t k) const
  {
    const hb_codepoint_t *v;
    if (singulars.has (k, &v))
      return hb_array (v, 1);

    hb_vector_t<hb_codepoint_t> *m;
    if (multiples.has (k, &m))
      return m->as_array ();

    return hb_array_t<const hb_codepoint_t> ();
  }

  bool in_error () const
  {
    if (singulars.in_error () || multiples.in_error ())
      return true;
    for (const auto &m : multiples.values_ref ())
      if (m.in_error ())
        return true;
    return false;
  }

  void alloc (unsigned size)
  {
    singulars.alloc (size);
  }

  protected:
  hb_map_t singulars;
  hb_hashmap_t<hb_codepoint_t, hb_vector_t<hb_codepoint_t>> multiples;
};



#endif /* HB_MULTIMAP_HH */
