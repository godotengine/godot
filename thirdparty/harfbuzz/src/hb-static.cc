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

#include "hb.hh"

#include "hb-open-type.hh"
#include "hb-face.hh"

#include "hb-aat-layout-common.hh"
#include "hb-aat-layout-feat-table.hh"
#include "hb-ot-layout-common.hh"
#include "hb-ot-cmap-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-maxp-table.hh"

#ifndef HB_NO_VISIBILITY
#include "hb-ot-name-language-static.hh"

uint64_t const _hb_NullPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)] = {};
/*thread_local*/ uint64_t _hb_CrapPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)] = {};

DEFINE_NULL_NAMESPACE_BYTES (OT, Index) =  {0xFF,0xFF};
DEFINE_NULL_NAMESPACE_BYTES (OT, LangSys) = {0x00,0x00, 0xFF,0xFF, 0x00,0x00};
DEFINE_NULL_NAMESPACE_BYTES (OT, RangeRecord) = {0x00,0x01, 0x00,0x00, 0x00, 0x00};
DEFINE_NULL_NAMESPACE_BYTES (OT, CmapSubtableLongGroup) = {0x00,0x00,0x00,0x01, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00};
DEFINE_NULL_NAMESPACE_BYTES (AAT, SettingName) = {0xFF,0xFF, 0xFF,0xFF};
/* Hand-coded because Lookup is a template.  Sad. */
const unsigned char _hb_Null_AAT_Lookup[2] = {0xFF, 0xFF};



/* hb_face_t */

unsigned int
hb_face_t::load_num_glyphs () const
{
  hb_sanitize_context_t c = hb_sanitize_context_t ();
  c.set_num_glyphs (0); /* So we don't recurse ad infinitum. */
  hb_blob_t *maxp_blob = c.reference_table<OT::maxp> (this);
  const OT::maxp *maxp_table = maxp_blob->as<OT::maxp> ();

  unsigned int ret = maxp_table->get_num_glyphs ();
  num_glyphs.set_relaxed (ret);
  hb_blob_destroy (maxp_blob);
  return ret;
}

unsigned int
hb_face_t::load_upem () const
{
  unsigned int ret = table.head->get_upem ();
  upem.set_relaxed (ret);
  return ret;
}


/* hb_user_data_array_t */

bool
hb_user_data_array_t::set (hb_user_data_key_t *key,
			   void *              data,
			   hb_destroy_func_t   destroy,
			   hb_bool_t           replace)
{
  if (!key)
    return false;

  if (replace) {
    if (!data && !destroy) {
      items.remove (key, lock);
      return true;
    }
  }
  hb_user_data_item_t item = {key, data, destroy};
  bool ret = !!items.replace_or_insert (item, lock, (bool) replace);

  return ret;
}

void *
hb_user_data_array_t::get (hb_user_data_key_t *key)
{
  hb_user_data_item_t item = {nullptr, nullptr, nullptr};

  return items.find (key, &item, lock) ? item.data : nullptr;
}


#endif
