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
#include "hb-cplusplus.hh"
#include "hb-font.hh"
#include "hb-subset-instancer-solver.hh"

struct hb_ot_name_record_ids_t
{
  hb_ot_name_record_ids_t () = default;
  hb_ot_name_record_ids_t (unsigned platform_id_,
                           unsigned encoding_id_,
                           unsigned language_id_,
                           unsigned name_id_)
      :platform_id (platform_id_),
      encoding_id (encoding_id_),
      language_id (language_id_),
      name_id (name_id_) {}

  bool operator != (const hb_ot_name_record_ids_t o) const
  { return !(*this == o); }

  inline bool operator == (const hb_ot_name_record_ids_t& o) const
  {
    return platform_id == o.platform_id &&
           encoding_id == o.encoding_id &&
           language_id == o.language_id &&
           name_id == o.name_id;
  }

  inline uint32_t hash () const
  {
    uint32_t current = 0;
    current = current * 31 + hb_hash (platform_id);
    current = current * 31 + hb_hash (encoding_id);
    current = current * 31 + hb_hash (language_id);
    current = current * 31 + hb_hash (name_id);
    return current;
  }

  unsigned platform_id;
  unsigned encoding_id;
  unsigned language_id;
  unsigned name_id;
};

typedef struct hb_ot_name_record_ids_t hb_ot_name_record_ids_t;


HB_MARK_AS_FLAG_T (hb_subset_flags_t);

struct hb_subset_input_t
{
  HB_INTERNAL hb_subset_input_t ();

  ~hb_subset_input_t ()
  {
    sets.~sets_t ();

#ifdef HB_EXPERIMENTAL_API
    for (auto _ : name_table_overrides.values ())
      _.fini ();
#endif
  }

  hb_object_header_t header;

  struct sets_t {
    hb::shared_ptr<hb_set_t> glyphs;
    hb::shared_ptr<hb_set_t> unicodes;
    hb::shared_ptr<hb_set_t> no_subset_tables;
    hb::shared_ptr<hb_set_t> drop_tables;
    hb::shared_ptr<hb_set_t> name_ids;
    hb::shared_ptr<hb_set_t> name_languages;
    hb::shared_ptr<hb_set_t> layout_features;
    hb::shared_ptr<hb_set_t> layout_scripts;
  };

  union {
    sets_t sets;
    hb::shared_ptr<hb_set_t> set_ptrs[sizeof (sets_t) / sizeof (hb_set_t*)];
  };

  unsigned flags;
  bool attach_accelerator_data = false;

  // If set loca format will always be the long version.
  bool force_long_loca = false;

  hb_hashmap_t<hb_tag_t, Triple> axes_location;
  hb_map_t glyph_map;
#ifdef HB_EXPERIMENTAL_API
  hb_hashmap_t<hb_ot_name_record_ids_t, hb_bytes_t> name_table_overrides;
#endif

  inline unsigned num_sets () const
  {
    return sizeof (set_ptrs) / sizeof (hb_set_t*);
  }

  inline hb_array_t<hb::shared_ptr<hb_set_t>> sets_iter ()
  {
    return hb_array (set_ptrs);
  }

  bool in_error () const
  {
    for (unsigned i = 0; i < num_sets (); i++)
    {
      if (unlikely (set_ptrs[i]->in_error ()))
        return true;
    }

    return axes_location.in_error ()
#ifdef HB_EXPERIMENTAL_API
	|| name_table_overrides.in_error ()
#endif
	;
  }
};


#endif /* HB_SUBSET_INPUT_HH */
