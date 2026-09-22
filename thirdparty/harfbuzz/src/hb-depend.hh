/*
 * Copyright © 2024  Adobe, Inc.
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
 * Adobe Author(s): Skef Iterum
 */

#ifndef HB_DEPEND_HH
#define HB_DEPEND_HH

#include "hb.hh"

#include "hb-subset-depend.h"
#include "hb-depend-data.hh"


/**
 * hb_subset_depend_t:
 *
 * Internal structure implementing the dependency graph API.
 *
 * Initialized via hb_subset_depend_from_face_or_fail() which computes the dependency
 * graph once via hb_depend_data_builder_t::compile(). The graph remains
 * immutable for the lifetime of the object.
 */
struct hb_subset_depend_t
{
  HB_INTERNAL hb_subset_depend_t (hb_face_t *face);

  hb_object_header_t header;

  bool successful;

  bool in_error () const { return !successful; }

  const hb_set_t *get_set_from_index (hb_codepoint_t index)
  {
    return data.get_set_from_index (index);
  }

  hb_depend_data_t data;
};


#endif /* HB_DEPEND_HH */
