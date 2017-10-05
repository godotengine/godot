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

#ifndef HB_SUBSET_PLAN_HH
#define HB_SUBSET_PLAN_HH

#include "hb-private.hh"

#include "hb-subset.h"
#include "hb-subset-private.hh"

#include "hb-map-private.hh"

struct hb_subset_plan_t
{
  hb_object_header_t header;
  ASSERT_POD ();

  hb_bool_t drop_hints;
  hb_bool_t drop_ot_layout;

  // For each cp that we'd like to retain maps to the corresponding gid.
  hb_set_t *unicodes;

  // This list contains the complete set of glyphs to retain and may contain
  // more glyphs then the lists above.
  hb_vector_t<hb_codepoint_t> glyphs;

  hb_map_t *codepoint_to_glyph;
  hb_map_t *glyph_map;

  // Plan is only good for a specific source/dest so keep them with it
  hb_face_t *source;
  hb_face_t *dest;

  inline hb_bool_t
  new_gid_for_codepoint (hb_codepoint_t codepoint,
                         hb_codepoint_t *new_gid) const
  {
    hb_codepoint_t old_gid = codepoint_to_glyph->get (codepoint);
    if (old_gid == HB_MAP_VALUE_INVALID)
      return false;

    return new_gid_for_old_gid (old_gid, new_gid);
  }

  inline hb_bool_t
  new_gid_for_old_gid (hb_codepoint_t old_gid,
                      hb_codepoint_t *new_gid) const
  {
    hb_codepoint_t gid = glyph_map->get (old_gid);
    if (gid == HB_MAP_VALUE_INVALID)
      return false;

    *new_gid = gid;
    return true;
  }

  inline hb_bool_t
  add_table (hb_tag_t tag,
             hb_blob_t *contents)
  {
    hb_blob_t *source_blob = source->reference_table (tag);
    DEBUG_MSG(SUBSET, nullptr, "add table %c%c%c%c, dest %d bytes, source %d bytes",
              HB_UNTAG(tag),
              hb_blob_get_length (contents),
              hb_blob_get_length (source_blob));
    hb_blob_destroy (source_blob);
    return hb_subset_face_add_table(dest, tag, contents);
  }
};

typedef struct hb_subset_plan_t hb_subset_plan_t;

HB_INTERNAL hb_subset_plan_t *
hb_subset_plan_create (hb_face_t           *face,
                       hb_subset_profile_t *profile,
                       hb_subset_input_t   *input);

HB_INTERNAL void
hb_subset_plan_destroy (hb_subset_plan_t *plan);

#endif /* HB_SUBSET_PLAN_HH */
