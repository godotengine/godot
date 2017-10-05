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
 * Google Author(s): Behdad Esfahbod
 */

#include "hb-private.hh"

#include "hb-open-type-private.hh"
#include "hb-ot-layout-common-private.hh"

#include "hb-face-private.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-maxp-table.hh"

#ifndef HB_NO_VISIBILITY

hb_vector_size_impl_t const _hb_NullPool[(HB_NULL_POOL_SIZE + sizeof (hb_vector_size_impl_t) - 1) / sizeof (hb_vector_size_impl_t)] = {};
/*thread_local*/ hb_vector_size_impl_t _hb_CrapPool[(HB_NULL_POOL_SIZE + sizeof (hb_vector_size_impl_t) - 1) / sizeof (hb_vector_size_impl_t)] = {};

DEFINE_NULL_NAMESPACE_BYTES (OT, Index) =  {0xFF,0xFF};
DEFINE_NULL_NAMESPACE_BYTES (OT, LangSys) = {0x00,0x00, 0xFF,0xFF, 0x00,0x00};
DEFINE_NULL_NAMESPACE_BYTES (OT, RangeRecord) = {0x00,0x01, 0x00,0x00, 0x00, 0x00};


void
hb_face_t::load_num_glyphs (void) const
{
  hb_sanitize_context_t c = hb_sanitize_context_t ();
  c.set_num_glyphs (0); /* So we don't recurse ad infinitum. */
  hb_blob_t *maxp_blob = c.reference_table<OT::maxp> (this);
  const OT::maxp *maxp_table = maxp_blob->as<OT::maxp> ();
  num_glyphs = maxp_table->get_num_glyphs ();
  hb_blob_destroy (maxp_blob);
}

void
hb_face_t::load_upem (void) const
{
  hb_blob_t *head_blob = hb_sanitize_context_t ().reference_table<OT::head> (this);
  const OT::head *head_table = head_blob->as<OT::head> ();
  upem = head_table->get_upem ();
  hb_blob_destroy (head_blob);
}

#endif
