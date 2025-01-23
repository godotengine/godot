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
#include "hb-cff-interp-common.hh"
#include "hb-ot-layout-common.hh"
#include "hb-ot-cmap-table.hh"
#include "OT/Color/COLR/COLR.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-maxp-table.hh"

#ifndef HB_NO_VISIBILITY
#include "hb-ot-name-language-static.hh"

uint64_t const _hb_NullPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)] = {};
/*thread_local*/ uint64_t _hb_CrapPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)] = {};

DEFINE_NULL_NAMESPACE_BYTES (OT, Index) =  {0xFF,0xFF};
DEFINE_NULL_NAMESPACE_BYTES (OT, VarIdx) =  {0xFF,0xFF,0xFF,0xFF};
DEFINE_NULL_NAMESPACE_BYTES (OT, LangSys) = {0x00,0x00, 0xFF,0xFF, 0x00,0x00};
DEFINE_NULL_NAMESPACE_BYTES (OT, RangeRecord) = {0x01};
DEFINE_NULL_NAMESPACE_BYTES (OT, ClipRecord) = {0x01};
DEFINE_NULL_NAMESPACE_BYTES (OT, CmapSubtableLongGroup) = {0x00,0x00,0x00,0x01, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00};
DEFINE_NULL_NAMESPACE_BYTES (AAT, SettingName) = {0xFF,0xFF, 0xFF,0xFF};
DEFINE_NULL_NAMESPACE_BYTES (AAT, Lookup) = {0xFF,0xFF};


/* hb_map_t */

const hb_codepoint_t minus_1 = -1;
static const unsigned char static_endchar_str[] = {OpCode_endchar};
const unsigned char *endchar_str = static_endchar_str;

/* hb_face_t */

#ifndef HB_NO_BEYOND_64K
static inline unsigned
load_num_glyphs_from_loca (const hb_face_t *face)
{
  unsigned ret = 0;

  unsigned indexToLocFormat = face->table.head->indexToLocFormat;

  if (indexToLocFormat <= 1)
  {
    bool short_offset = 0 == indexToLocFormat;
    hb_blob_t *loca_blob = face->table.loca.get_blob ();
    ret = hb_max (1u, loca_blob->length / (short_offset ? 2 : 4)) - 1;
  }

  return ret;
}
#endif

static inline unsigned
load_num_glyphs_from_maxp (const hb_face_t *face)
{
  return face->table.maxp->get_num_glyphs ();
}

unsigned int
hb_face_t::load_num_glyphs () const
{
  unsigned ret = 0;

#ifndef HB_NO_BEYOND_64K
  ret = hb_max (ret, load_num_glyphs_from_loca (this));
#endif

  ret = hb_max (ret, load_num_glyphs_from_maxp (this));

  num_glyphs = ret;
  return ret;
}

unsigned int
hb_face_t::load_upem () const
{
  unsigned int ret = table.head->get_upem ();
  upem = ret;
  return ret;
}


#ifndef HB_NO_VAR
bool
_glyf_get_leading_bearing_with_var_unscaled (hb_font_t *font, hb_codepoint_t glyph, bool is_vertical,
					     int *lsb)
{
  return font->face->table.glyf->get_leading_bearing_with_var_unscaled (font, glyph, is_vertical, lsb);
}

unsigned
_glyf_get_advance_with_var_unscaled (hb_font_t *font, hb_codepoint_t glyph, bool is_vertical)
{
  return font->face->table.glyf->get_advance_with_var_unscaled (font, glyph, is_vertical);
}
#endif

bool
_glyf_get_leading_bearing_without_var_unscaled (hb_face_t *face, hb_codepoint_t gid, bool is_vertical, int *lsb)
{
  return face->table.glyf->get_leading_bearing_without_var_unscaled (gid, is_vertical, lsb);
}


#endif
