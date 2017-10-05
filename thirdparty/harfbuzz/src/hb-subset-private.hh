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

#ifndef HB_SUBSET_PRIVATE_HH
#define HB_SUBSET_PRIVATE_HH


#include "hb-private.hh"

#include "hb-subset.h"

#include "hb-font-private.hh"

typedef struct hb_subset_face_data_t hb_subset_face_data_t;

struct hb_subset_input_t {
  hb_object_header_t header;
  ASSERT_POD ();

  hb_set_t *unicodes;
  hb_set_t *glyphs;

  hb_bool_t drop_hints;
  hb_bool_t drop_ot_layout;
  /* TODO
   *
   * features
   * lookups
   * nameIDs
   * ...
   */
};

HB_INTERNAL hb_face_t *
hb_subset_face_create (void);

HB_INTERNAL hb_bool_t
hb_subset_face_add_table (hb_face_t *face, hb_tag_t tag, hb_blob_t *blob);

#endif /* HB_SUBSET_PRIVATE_HH */
