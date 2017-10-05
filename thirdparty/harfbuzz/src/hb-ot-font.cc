/*
 * Copyright © 2011,2014  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod, Roozbeh Pournader
 */

#include "hb-private.hh"

#include "hb-ot.h"

#include "hb-font-private.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-cbdt-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-kern-table.hh"
#include "hb-ot-post-table.hh"


struct hb_ot_font_t
{
  OT::cmap::accelerator_t cmap;
  OT::hmtx::accelerator_t h_metrics;
  OT::vmtx::accelerator_t v_metrics;
  OT::hb_lazy_loader_t<OT::glyf::accelerator_t> glyf;
  OT::hb_lazy_loader_t<OT::CBDT::accelerator_t> cbdt;
  OT::hb_lazy_loader_t<OT::post::accelerator_t> post;
  OT::hb_lazy_loader_t<OT::kern::accelerator_t> kern;
};


static hb_ot_font_t *
_hb_ot_font_create (hb_face_t *face)
{
  hb_ot_font_t *ot_font = (hb_ot_font_t *) calloc (1, sizeof (hb_ot_font_t));

  if (unlikely (!ot_font))
    return nullptr;

  ot_font->cmap.init (face);
  ot_font->h_metrics.init (face);
  ot_font->v_metrics.init (face, ot_font->h_metrics.ascender - ot_font->h_metrics.descender); /* TODO Can we do this lazily? */
  ot_font->glyf.init (face);
  ot_font->cbdt.init (face);
  ot_font->post.init (face);
  ot_font->kern.init (face);

  return ot_font;
}

static void
_hb_ot_font_destroy (void *data)
{
  hb_ot_font_t *ot_font = (hb_ot_font_t *) data;

  ot_font->cmap.fini ();
  ot_font->h_metrics.fini ();
  ot_font->v_metrics.fini ();
  ot_font->glyf.fini ();
  ot_font->cbdt.fini ();
  ot_font->post.fini ();
  ot_font->kern.fini ();

  free (ot_font);
}


static hb_bool_t
hb_ot_get_nominal_glyph (hb_font_t *font HB_UNUSED,
			 void *font_data,
			 hb_codepoint_t unicode,
			 hb_codepoint_t *glyph,
			 void *user_data HB_UNUSED)

{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return ot_font->cmap.get_nominal_glyph (unicode, glyph);
}

static hb_bool_t
hb_ot_get_variation_glyph (hb_font_t *font HB_UNUSED,
			   void *font_data,
			   hb_codepoint_t unicode,
			   hb_codepoint_t variation_selector,
			   hb_codepoint_t *glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return ot_font->cmap.get_variation_glyph (unicode, variation_selector, glyph);
}

static hb_position_t
hb_ot_get_glyph_h_advance (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return font->em_scale_x (ot_font->h_metrics.get_advance (glyph, font));
}

static hb_position_t
hb_ot_get_glyph_v_advance (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return font->em_scale_y (-(int) ot_font->v_metrics.get_advance (glyph, font));
}

static hb_position_t
hb_ot_get_glyph_h_kerning (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t left_glyph,
			   hb_codepoint_t right_glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return font->em_scale_x (ot_font->kern->get_h_kerning (left_glyph, right_glyph));
}

static hb_bool_t
hb_ot_get_glyph_extents (hb_font_t *font HB_UNUSED,
			 void *font_data,
			 hb_codepoint_t glyph,
			 hb_glyph_extents_t *extents,
			 void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  bool ret = ot_font->glyf->get_extents (glyph, extents);
  if (!ret)
    ret = ot_font->cbdt->get_extents (glyph, extents);
  // TODO Hook up side-bearings variations.
  extents->x_bearing = font->em_scale_x (extents->x_bearing);
  extents->y_bearing = font->em_scale_y (extents->y_bearing);
  extents->width     = font->em_scale_x (extents->width);
  extents->height    = font->em_scale_y (extents->height);
  return ret;
}

static hb_bool_t
hb_ot_get_glyph_name (hb_font_t *font HB_UNUSED,
                      void *font_data,
                      hb_codepoint_t glyph,
                      char *name, unsigned int size,
                      void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return ot_font->post->get_glyph_name (glyph, name, size);
}

static hb_bool_t
hb_ot_get_glyph_from_name (hb_font_t *font HB_UNUSED,
                           void *font_data,
                           const char *name, int len,
                           hb_codepoint_t *glyph,
                           void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  return ot_font->post->get_glyph_from_name (name, len, glyph);
}

static hb_bool_t
hb_ot_get_font_h_extents (hb_font_t *font HB_UNUSED,
			  void *font_data,
			  hb_font_extents_t *metrics,
			  void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  metrics->ascender = font->em_scale_y (ot_font->h_metrics.ascender);
  metrics->descender = font->em_scale_y (ot_font->h_metrics.descender);
  metrics->line_gap = font->em_scale_y (ot_font->h_metrics.line_gap);
  // TODO Hook up variations.
  return ot_font->h_metrics.has_font_extents;
}

static hb_bool_t
hb_ot_get_font_v_extents (hb_font_t *font HB_UNUSED,
			  void *font_data,
			  hb_font_extents_t *metrics,
			  void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  metrics->ascender = font->em_scale_x (ot_font->v_metrics.ascender);
  metrics->descender = font->em_scale_x (ot_font->v_metrics.descender);
  metrics->line_gap = font->em_scale_x (ot_font->v_metrics.line_gap);
  // TODO Hook up variations.
  return ot_font->v_metrics.has_font_extents;
}

static hb_font_funcs_t *static_ot_funcs = nullptr;

#ifdef HB_USE_ATEXIT
static
void free_static_ot_funcs (void)
{
  hb_font_funcs_destroy (static_ot_funcs);
}
#endif

static hb_font_funcs_t *
_hb_ot_get_font_funcs (void)
{
retry:
  hb_font_funcs_t *funcs = (hb_font_funcs_t *) hb_atomic_ptr_get (&static_ot_funcs);

  if (unlikely (!funcs))
  {
    funcs = hb_font_funcs_create ();

    hb_font_funcs_set_font_h_extents_func (funcs, hb_ot_get_font_h_extents, nullptr, nullptr);
    hb_font_funcs_set_font_v_extents_func (funcs, hb_ot_get_font_v_extents, nullptr, nullptr);
    hb_font_funcs_set_nominal_glyph_func (funcs, hb_ot_get_nominal_glyph, nullptr, nullptr);
    hb_font_funcs_set_variation_glyph_func (funcs, hb_ot_get_variation_glyph, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_advance_func (funcs, hb_ot_get_glyph_h_advance, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_advance_func (funcs, hb_ot_get_glyph_v_advance, nullptr, nullptr);
    //hb_font_funcs_set_glyph_h_origin_func (funcs, hb_ot_get_glyph_h_origin, nullptr, nullptr);
    //hb_font_funcs_set_glyph_v_origin_func (funcs, hb_ot_get_glyph_v_origin, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_kerning_func (funcs, hb_ot_get_glyph_h_kerning, nullptr, nullptr);
    //hb_font_funcs_set_glyph_v_kerning_func (funcs, hb_ot_get_glyph_v_kerning, nullptr, nullptr);
    hb_font_funcs_set_glyph_extents_func (funcs, hb_ot_get_glyph_extents, nullptr, nullptr);
    //hb_font_funcs_set_glyph_contour_point_func (funcs, hb_ot_get_glyph_contour_point, nullptr, nullptr);
    hb_font_funcs_set_glyph_name_func (funcs, hb_ot_get_glyph_name, nullptr, nullptr);
    hb_font_funcs_set_glyph_from_name_func (funcs, hb_ot_get_glyph_from_name, nullptr, nullptr);

    hb_font_funcs_make_immutable (funcs);

    if (!hb_atomic_ptr_cmpexch (&static_ot_funcs, nullptr, funcs)) {
      hb_font_funcs_destroy (funcs);
      goto retry;
    }

#ifdef HB_USE_ATEXIT
    atexit (free_static_ot_funcs); /* First person registers atexit() callback. */
#endif
  };

  return funcs;
}


/**
 * hb_ot_font_set_funcs:
 *
 * Since: 0.9.28
 **/
void
hb_ot_font_set_funcs (hb_font_t *font)
{
  hb_ot_font_t *ot_font = _hb_ot_font_create (font->face);
  if (unlikely (!ot_font))
    return;

  hb_font_set_funcs (font,
		     _hb_ot_get_font_funcs (),
		     ot_font,
		     _hb_ot_font_destroy);
}
