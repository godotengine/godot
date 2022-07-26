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

#include "hb.hh"

#ifndef HB_NO_OT_FONT

#include "hb-ot.h"

#include "hb-cache.hh"
#include "hb-font.hh"
#include "hb-machinery.hh"
#include "hb-ot-face.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-post-table.hh"
#include "hb-ot-stat-table.hh" // Just so we compile it; unused otherwise.
#include "hb-ot-vorg-table.hh"
#include "hb-ot-color-cbdt-table.hh"
#include "hb-ot-color-sbix-table.hh"


/**
 * SECTION:hb-ot-font
 * @title: hb-ot-font
 * @short_description: OpenType font implementation
 * @include: hb-ot.h
 *
 * Functions for using OpenType fonts with hb_shape().  Note that fonts returned
 * by hb_font_create() default to using these functions, so most clients would
 * never need to call these functions directly.
 **/

struct hb_ot_font_t
{
  const hb_ot_face_t *ot_face;

  /* h_advance caching */
  mutable hb_atomic_int_t cached_coords_serial;
  mutable hb_atomic_ptr_t<hb_advance_cache_t> advance_cache;
};

static hb_ot_font_t *
_hb_ot_font_create (hb_font_t *font)
{
  hb_ot_font_t *ot_font = (hb_ot_font_t *) hb_calloc (1, sizeof (hb_ot_font_t));
  if (unlikely (!ot_font))
    return nullptr;

  ot_font->ot_face = &font->face->table;

  return ot_font;
}

static void
_hb_ot_font_destroy (void *font_data)
{
  hb_ot_font_t *ot_font = (hb_ot_font_t *) font_data;

  auto *cache = ot_font->advance_cache.get_relaxed ();
  if (cache)
  {
    cache->fini ();
    hb_free (cache);
  }

  hb_free (ot_font);
}

static hb_bool_t
hb_ot_get_nominal_glyph (hb_font_t *font HB_UNUSED,
			 void *font_data,
			 hb_codepoint_t unicode,
			 hb_codepoint_t *glyph,
			 void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  return ot_face->cmap->get_nominal_glyph (unicode, glyph);
}

static unsigned int
hb_ot_get_nominal_glyphs (hb_font_t *font HB_UNUSED,
			  void *font_data,
			  unsigned int count,
			  const hb_codepoint_t *first_unicode,
			  unsigned int unicode_stride,
			  hb_codepoint_t *first_glyph,
			  unsigned int glyph_stride,
			  void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  return ot_face->cmap->get_nominal_glyphs (count,
					    first_unicode, unicode_stride,
					    first_glyph, glyph_stride);
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
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  return ot_face->cmap->get_variation_glyph (unicode, variation_selector, glyph);
}

static void
hb_ot_get_glyph_h_advances (hb_font_t* font, void* font_data,
			    unsigned count,
			    const hb_codepoint_t *first_glyph,
			    unsigned glyph_stride,
			    hb_position_t *first_advance,
			    unsigned advance_stride,
			    void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  const OT::hmtx_accelerator_t &hmtx = *ot_face->hmtx;

#ifndef HB_NO_VAR
  const OT::HVAR &HVAR = *hmtx.var_table;
  const OT::VariationStore &varStore = &HVAR + HVAR.varStore;
  OT::VariationStore::cache_t *varStore_cache = font->num_coords * count >= 128 ? varStore.create_cache () : nullptr;

  bool use_cache = font->num_coords;
#else
  OT::VariationStore::cache_t *varStore_cache = nullptr;
  bool use_cache = false;
#endif

  hb_advance_cache_t *cache = nullptr;
  if (use_cache)
  {
  retry:
    cache = ot_font->advance_cache.get ();
    if (unlikely (!cache))
    {
      cache = (hb_advance_cache_t *) hb_malloc (sizeof (hb_advance_cache_t));
      if (unlikely (!cache))
      {
	use_cache = false;
	goto out;
      }

      cache->init ();
      if (unlikely (!ot_font->advance_cache.cmpexch (nullptr, cache)))
      {
	hb_free (cache);
	goto retry;
      }
      ot_font->cached_coords_serial.set (font->serial_coords);
    }
  }
  out:

  if (!use_cache)
  {
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->em_scale_x (hmtx.get_advance_with_var_unscaled (*first_glyph, font, varStore_cache));
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
  }
  else
  { /* Use cache. */
    if (ot_font->cached_coords_serial.get () != (int) font->serial_coords)
    {
      ot_font->advance_cache->init ();
      ot_font->cached_coords_serial.set (font->serial_coords);
    }

    for (unsigned int i = 0; i < count; i++)
    {
      hb_position_t v;
      unsigned cv;
      if (ot_font->advance_cache->get (*first_glyph, &cv))
	v = cv;
      else
      {
        v = hmtx.get_advance_with_var_unscaled (*first_glyph, font, varStore_cache);
	ot_font->advance_cache->set (*first_glyph, v);
      }
      *first_advance = font->em_scale_x (v);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
  }

#ifndef HB_NO_VAR
  OT::VariationStore::destroy_cache (varStore_cache);
#endif
}

#ifndef HB_NO_VERTICAL
static void
hb_ot_get_glyph_v_advances (hb_font_t* font, void* font_data,
			    unsigned count,
			    const hb_codepoint_t *first_glyph,
			    unsigned glyph_stride,
			    hb_position_t *first_advance,
			    unsigned advance_stride,
			    void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  const OT::vmtx_accelerator_t &vmtx = *ot_face->vmtx;

  if (vmtx.has_data ())
  {
#ifndef HB_NO_VAR
    const OT::VVAR &VVAR = *vmtx.var_table;
    const OT::VariationStore &varStore = &VVAR + VVAR.varStore;
    OT::VariationStore::cache_t *varStore_cache = font->num_coords ? varStore.create_cache () : nullptr;
#else
    OT::VariationStore::cache_t *varStore_cache = nullptr;
#endif

    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->em_scale_y (-(int) vmtx.get_advance_with_var_unscaled (*first_glyph, font, varStore_cache));
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }

#ifndef HB_NO_VAR
    OT::VariationStore::destroy_cache (varStore_cache);
#endif
  }
  else
  {
    hb_font_extents_t font_extents;
    font->get_h_extents_with_fallback (&font_extents);
    hb_position_t advance = -(font_extents.ascender - font_extents.descender);

    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = advance;
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
  }
}
#endif

#ifndef HB_NO_VERTICAL
static hb_bool_t
hb_ot_get_glyph_v_origin (hb_font_t *font,
			  void *font_data,
			  hb_codepoint_t glyph,
			  hb_position_t *x,
			  hb_position_t *y,
			  void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;

  *x = font->get_glyph_h_advance (glyph) / 2;

  const OT::VORG &VORG = *ot_face->VORG;
  if (VORG.has_data ())
  {
    float delta = 0;

#ifndef HB_NO_VAR
    const OT::vmtx_accelerator_t &vmtx = *ot_face->vmtx;
    const OT::VVAR &VVAR = *vmtx.var_table;
    if (font->num_coords)
      VVAR.get_vorg_delta_unscaled (glyph,
				    font->coords, font->num_coords,
				    &delta);
#endif

    *y = font->em_scalef_y (VORG.get_y_origin (glyph) + delta);
    return true;
  }

  hb_glyph_extents_t extents = {0};
  if (ot_face->glyf->get_extents (font, glyph, &extents))
  {
    const OT::vmtx_accelerator_t &vmtx = *ot_face->vmtx;
    int tsb = 0;
    if (vmtx.get_leading_bearing_with_var_unscaled (font, glyph, &tsb))
    {
      *y = extents.y_bearing + font->em_scale_y (tsb);
      return true;
    }

    hb_font_extents_t font_extents;
    font->get_h_extents_with_fallback (&font_extents);
    hb_position_t advance = font_extents.ascender - font_extents.descender;
    int diff = advance - -extents.height;
    *y = extents.y_bearing + (diff >> 1);
    return true;
  }

  hb_font_extents_t font_extents;
  font->get_h_extents_with_fallback (&font_extents);
  *y = font_extents.ascender;

  return true;
}
#endif

static hb_bool_t
hb_ot_get_glyph_extents (hb_font_t *font,
			 void *font_data,
			 hb_codepoint_t glyph,
			 hb_glyph_extents_t *extents,
			 void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;

#if !defined(HB_NO_OT_FONT_BITMAP) && !defined(HB_NO_COLOR)
  if (ot_face->sbix->get_extents (font, glyph, extents)) return true;
#endif
  if (ot_face->glyf->get_extents (font, glyph, extents)) return true;
#ifndef HB_NO_OT_FONT_CFF
  if (ot_face->cff1->get_extents (font, glyph, extents)) return true;
  if (ot_face->cff2->get_extents (font, glyph, extents)) return true;
#endif
#if !defined(HB_NO_OT_FONT_BITMAP) && !defined(HB_NO_COLOR)
  if (ot_face->CBDT->get_extents (font, glyph, extents)) return true;
#endif

  // TODO Hook up side-bearings variations.
  return false;
}

#ifndef HB_NO_OT_FONT_GLYPH_NAMES
static hb_bool_t
hb_ot_get_glyph_name (hb_font_t *font HB_UNUSED,
		      void *font_data,
		      hb_codepoint_t glyph,
		      char *name, unsigned int size,
		      void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;

  if (ot_face->post->get_glyph_name (glyph, name, size)) return true;
#ifndef HB_NO_OT_FONT_CFF
  if (ot_face->cff1->get_glyph_name (glyph, name, size)) return true;
#endif
  return false;
}
static hb_bool_t
hb_ot_get_glyph_from_name (hb_font_t *font HB_UNUSED,
			   void *font_data,
			   const char *name, int len,
			   hb_codepoint_t *glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;

  if (ot_face->post->get_glyph_from_name (name, len, glyph)) return true;
#ifndef HB_NO_OT_FONT_CFF
    if (ot_face->cff1->get_glyph_from_name (name, len, glyph)) return true;
#endif
  return false;
}
#endif

static hb_bool_t
hb_ot_get_font_h_extents (hb_font_t *font,
			  void *font_data HB_UNUSED,
			  hb_font_extents_t *metrics,
			  void *user_data HB_UNUSED)
{
  return _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER, &metrics->ascender) &&
	 _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER, &metrics->descender) &&
	 _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP, &metrics->line_gap);
}

#ifndef HB_NO_VERTICAL
static hb_bool_t
hb_ot_get_font_v_extents (hb_font_t *font,
			  void *font_data HB_UNUSED,
			  hb_font_extents_t *metrics,
			  void *user_data HB_UNUSED)
{
  return _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_VERTICAL_ASCENDER, &metrics->ascender) &&
	 _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_VERTICAL_DESCENDER, &metrics->descender) &&
	 _hb_ot_metrics_get_position_common (font, HB_OT_METRICS_TAG_VERTICAL_LINE_GAP, &metrics->line_gap);
}
#endif

#ifndef HB_NO_DRAW
static void
hb_ot_get_glyph_shape (hb_font_t *font,
		       void *font_data HB_UNUSED,
		       hb_codepoint_t glyph,
		       hb_draw_funcs_t *draw_funcs, void *draw_data,
		       void *user_data)
{
  hb_draw_session_t draw_session (draw_funcs, draw_data, font->slant_xy);
  if (font->face->table.glyf->get_path (font, glyph, draw_session)) return;
#ifndef HB_NO_CFF
  if (font->face->table.cff1->get_path (font, glyph, draw_session)) return;
  if (font->face->table.cff2->get_path (font, glyph, draw_session)) return;
#endif
}
#endif

static inline void free_static_ot_funcs ();

static struct hb_ot_font_funcs_lazy_loader_t : hb_font_funcs_lazy_loader_t<hb_ot_font_funcs_lazy_loader_t>
{
  static hb_font_funcs_t *create ()
  {
    hb_font_funcs_t *funcs = hb_font_funcs_create ();

    hb_font_funcs_set_nominal_glyph_func (funcs, hb_ot_get_nominal_glyph, nullptr, nullptr);
    hb_font_funcs_set_nominal_glyphs_func (funcs, hb_ot_get_nominal_glyphs, nullptr, nullptr);
    hb_font_funcs_set_variation_glyph_func (funcs, hb_ot_get_variation_glyph, nullptr, nullptr);

    hb_font_funcs_set_font_h_extents_func (funcs, hb_ot_get_font_h_extents, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_advances_func (funcs, hb_ot_get_glyph_h_advances, nullptr, nullptr);
    //hb_font_funcs_set_glyph_h_origin_func (funcs, hb_ot_get_glyph_h_origin, nullptr, nullptr);

#ifndef HB_NO_VERTICAL
    hb_font_funcs_set_font_v_extents_func (funcs, hb_ot_get_font_v_extents, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_advances_func (funcs, hb_ot_get_glyph_v_advances, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_origin_func (funcs, hb_ot_get_glyph_v_origin, nullptr, nullptr);
#endif

#ifndef HB_NO_DRAW
    hb_font_funcs_set_glyph_shape_func (funcs, hb_ot_get_glyph_shape, nullptr, nullptr);
#endif

    hb_font_funcs_set_glyph_extents_func (funcs, hb_ot_get_glyph_extents, nullptr, nullptr);
    //hb_font_funcs_set_glyph_contour_point_func (funcs, hb_ot_get_glyph_contour_point, nullptr, nullptr);

#ifndef HB_NO_OT_FONT_GLYPH_NAMES
    hb_font_funcs_set_glyph_name_func (funcs, hb_ot_get_glyph_name, nullptr, nullptr);
    hb_font_funcs_set_glyph_from_name_func (funcs, hb_ot_get_glyph_from_name, nullptr, nullptr);
#endif

    hb_font_funcs_make_immutable (funcs);

    hb_atexit (free_static_ot_funcs);

    return funcs;
  }
} static_ot_funcs;

static inline
void free_static_ot_funcs ()
{
  static_ot_funcs.free_instance ();
}

static hb_font_funcs_t *
_hb_ot_get_font_funcs ()
{
  return static_ot_funcs.get_unconst ();
}


/**
 * hb_ot_font_set_funcs:
 * @font: #hb_font_t to work upon
 *
 * Sets the font functions to use when working with @font. 
 *
 * Since: 0.9.28
 **/
void
hb_ot_font_set_funcs (hb_font_t *font)
{
  hb_ot_font_t *ot_font = _hb_ot_font_create (font);
  if (unlikely (!ot_font))
    return;

  hb_font_set_funcs (font,
		     _hb_ot_get_font_funcs (),
		     ot_font,
		     _hb_ot_font_destroy);
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


#endif
