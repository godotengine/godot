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
 * Google Author(s): Garret Rieger, Roderick Sheeter
 */

#ifndef HB_SUBSET_INPUT_HH
#define HB_SUBSET_INPUT_HH


#include "hb.hh"

#include "hb-subset.h"
#include "hb-map.hh"
#include "hb-set.hh"

#include "hb-font.hh"

HB_MARK_AS_FLAG_T (hb_subset_flags_t);

struct hb_subset_input_t
{
  hb_object_header_t header;

  struct sets_t {
    hb_set_t *glyphs;
    hb_set_t *unicodes;
    hb_set_t *no_subset_tables;
    hb_set_t *drop_tables;
    hb_set_t *name_ids;
    hb_set_t *name_languages;
    hb_set_t *layout_features;
    hb_set_t *layout_scripts;
  };

  union {
    sets_t sets;
    hb_set_t* set_ptrs[sizeof (sets_t) / sizeof (hb_set_t*)];
  };

  unsigned flags;
  hb_hashmap_t<hb_tag_t, float> *axes_location;

  inline unsigned num_sets () const
  {
    return sizeof (set_ptrs) / sizeof (hb_set_t*);
  }

  inline hb_array_t<hb_set_t*> sets_iter ()
  {
    return hb_array_t<hb_set_t*> (set_ptrs, num_sets ());
  }

  bool in_error () const
  {
    for (unsigned i = 0; i < num_sets (); i++)
    {
      if (unlikely (set_ptrs[i]->in_error ()))
        return true;
    }

    return axes_location->in_error ();
  }
};


#endif /* HB_SUBSET_INPUT_HH */
