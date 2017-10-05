/*
 * Copyright © 2017  Google, Inc.
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

#include "hb-open-type-private.hh"

#include "hb-ot-layout-private.hh"
#include "hb-ot-layout-gsubgpos-private.hh"

#include "hb-aat-layout-private.hh"
#include "hb-aat-layout-morx-table.hh"

/*
 * mort/morx
 */

static inline const AAT::morx&
_get_morx (hb_face_t *face, hb_blob_t **blob = nullptr)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face)))
  {
    if (blob)
      *blob = hb_blob_get_empty ();
    return OT::Null(AAT::morx);
  }
  hb_ot_layout_t * layout = hb_ot_layout_from_face (face);
  /* XXX this doesn't call set_num_glyphs on sanitizer. */
  const AAT::morx& morx = *(layout->morx.get ());
  if (blob)
    *blob = layout->morx.blob;
  return morx;
}

static inline void
_hb_aat_layout_create (hb_face_t *face)
{
  OT::Sanitizer<AAT::morx> sanitizer;
  sanitizer.set_num_glyphs (face->get_num_glyphs ());
  hb_blob_t *morx_blob = sanitizer.sanitize (face->reference_table (HB_AAT_TAG_MORX));
  OT::Sanitizer<AAT::morx>::lock_instance (morx_blob);

  if (0)
  {
    OT::Sanitizer<AAT::Lookup<OT::GlyphID> >::lock_instance (morx_blob)->get_value (1, face->get_num_glyphs ());
  }
}

void
hb_aat_layout_substitute (hb_font_t *font, hb_buffer_t *buffer)
{
  hb_blob_t *blob;
  const AAT::morx& morx = _get_morx (font->face, &blob);

  AAT::hb_aat_apply_context_t c (font, buffer, blob);
  morx.apply (&c);
}
