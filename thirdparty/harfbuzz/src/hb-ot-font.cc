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
#include "hb-ot-var-gvar-table.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-post-table.hh"
#include "hb-ot-stat-table.hh"
#include "hb-ot-var-varc-table.hh"
#include "hb-ot-vorg-table.hh"
#include "OT/Color/CBDT/CBDT.hh"
#include "OT/Color/COLR/COLR.hh"
#include "OT/Color/sbix/sbix.hh"
#include "OT/Color/svg/svg.hh"


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

using hb_ot_font_advance_cache_t = hb_cache_t<24, 16>;
static_assert (sizeof (hb_ot_font_advance_cache_t) == 1024, "");

using hb_ot_font_origin_cache_t = hb_cache_t<20, 20>;
static_assert (sizeof (hb_ot_font_origin_cache_t) == 1024, "");

struct hb_ot_font_t
{
  const hb_ot_face_t *ot_face;

  mutable hb_atomic_t<int> cached_serial;
  mutable hb_atomic_t<int> cached_coords_serial;

  struct direction_cache_t
  {
    mutable hb_atomic_t<hb_ot_font_advance_cache_t *> advance_cache;
    mutable hb_atomic_t<OT::hb_scalar_cache_t *> varStore_cache;

    ~direction_cache_t ()
    {
      clear ();
    }

    hb_ot_font_advance_cache_t *acquire_advance_cache () const
    {
    retry:
      auto *cache = advance_cache.get_acquire ();
      if (!cache)
      {
        cache = (hb_ot_font_advance_cache_t *) hb_malloc (sizeof (hb_ot_font_advance_cache_t));
	if (!cache)
	  return nullptr;
	new (cache) hb_ot_font_advance_cache_t;
	return cache;
      }
      if (advance_cache.cmpexch (cache, nullptr))
        return cache;
      else
        goto retry;
    }
    void release_advance_cache (hb_ot_font_advance_cache_t *cache) const
    {
      if (!cache)
        return;
      if (!advance_cache.cmpexch (nullptr, cache))
        hb_free (cache);
    }
    void clear_advance_cache () const
    {
    retry:
      auto *cache = advance_cache.get_acquire ();
      if (!cache)
	return;
      if (advance_cache.cmpexch (cache, nullptr))
	hb_free (cache);
      else
        goto retry;
    }

    OT::hb_scalar_cache_t *acquire_varStore_cache (const OT::ItemVariationStore &varStore) const
    {
    retry:
      auto *cache = varStore_cache.get_acquire ();
      if (!cache)
	return varStore.create_cache ();
      if (varStore_cache.cmpexch (cache, nullptr))
	return cache;
      else
	goto retry;
    }
    void release_varStore_cache (OT::hb_scalar_cache_t *cache) const
    {
      if (!cache)
	return;
      if (!varStore_cache.cmpexch (nullptr, cache))
	OT::ItemVariationStore::destroy_cache (cache);
    }
    void clear_varStore_cache () const
    {
    retry:
      auto *cache = varStore_cache.get_acquire ();
      if (!cache)
	return;
      if (varStore_cache.cmpexch (cache, nullptr))
	OT::ItemVariationStore::destroy_cache (cache);
      else
	goto retry;
    }

    void clear () const
    {
      clear_advance_cache ();
      clear_varStore_cache ();
    }

  } h, v;

  struct origin_cache_t
  {
    mutable hb_atomic_t<hb_ot_font_origin_cache_t *> origin_cache;
    mutable hb_atomic_t<OT::hb_scalar_cache_t *> varStore_cache;

    ~origin_cache_t ()
    {
      clear ();
    }

    hb_ot_font_origin_cache_t *acquire_origin_cache () const
    {
    retry:
      auto *cache = origin_cache.get_acquire ();
      if (!cache)
      {
        cache = (hb_ot_font_origin_cache_t *) hb_malloc (sizeof (hb_ot_font_origin_cache_t));
	if (!cache)
	  return nullptr;
	new (cache) hb_ot_font_origin_cache_t;
	return cache;
      }
      if (origin_cache.cmpexch (cache, nullptr))
        return cache;
      else
        goto retry;
    }
    void release_origin_cache (hb_ot_font_origin_cache_t *cache) const
    {
      if (!cache)
        return;
      if (!origin_cache.cmpexch (nullptr, cache))
        hb_free (cache);
    }
    void clear_origin_cache () const
    {
    retry:
      auto *cache = origin_cache.get_acquire ();
      if (!cache)
	return;
      if (origin_cache.cmpexch (cache, nullptr))
	hb_free (cache);
      else
        goto retry;
    }

    OT::hb_scalar_cache_t *acquire_varStore_cache (const OT::ItemVariationStore &varStore) const
    {
    retry:
      auto *cache = varStore_cache.get_acquire ();
      if (!cache)
	return varStore.create_cache ();
      if (varStore_cache.cmpexch (cache, nullptr))
	return cache;
      else
	goto retry;
    }
    void release_varStore_cache (OT::hb_scalar_cache_t *cache) const
    {
      if (!cache)
	return;
      if (!varStore_cache.cmpexch (nullptr, cache))
	OT::ItemVariationStore::destroy_cache (cache);
    }
    void clear_varStore_cache () const
    {
    retry:
      auto *cache = varStore_cache.get_acquire ();
      if (!cache)
	return;
      if (varStore_cache.cmpexch (cache, nullptr))
	OT::ItemVariationStore::destroy_cache (cache);
      else
	goto retry;
    }

    void clear () const
    {
      clear_origin_cache ();
      clear_varStore_cache ();
    }
  } v_origin;

  struct draw_cache_t
  {
    mutable hb_atomic_t<OT::hb_scalar_cache_t *> gvar_cache;

    ~draw_cache_t ()
    {
      clear ();
    }

    OT::hb_scalar_cache_t *acquire_gvar_cache (const OT::gvar_accelerator_t &gvar) const
    {
    retry:
      auto *cache = gvar_cache.get_acquire ();
      if (!cache)
	return gvar.create_cache ();
      if (gvar_cache.cmpexch (cache, nullptr))
	return cache;
      else
	goto retry;
    }
    void release_gvar_cache (OT::hb_scalar_cache_t *cache) const
    {
      if (!cache)
	return;
      if (!gvar_cache.cmpexch (nullptr, cache))
	OT::gvar_accelerator_t::destroy_cache (cache);
    }
    void clear_gvar_cache () const
    {
    retry:
      auto *cache = gvar_cache.get_acquire ();
      if (!cache)
	return;
      if (gvar_cache.cmpexch (cache, nullptr))
	OT::gvar_accelerator_t::destroy_cache (cache);
      else
	goto retry;
    }

    void clear () const
    {
      clear_gvar_cache ();
    }
  } draw;

  void check_serial (hb_font_t *font) const
  {
    int font_serial = font->serial_coords.get_acquire ();
    if (cached_serial.get_acquire () != font_serial)
    {
      /* These caches are dependent on scale and synthetic settings.
       * Any change to the font invalidates them. */
      v_origin.clear ();

      cached_serial.set_release (font_serial);
    }

    int font_serial_coords = font->serial_coords.get_acquire ();
    if (cached_coords_serial.get_acquire () != font_serial_coords)
    {
      /* These caches are independent of scale or synthetic settings.
       * Just variation changes will invalidate them. */
      h.clear ();
      v.clear ();
      draw.clear ();

      cached_coords_serial.set_release (font_serial_coords);
    }
  }
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

  ot_font->~hb_ot_font_t ();

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
  return ot_face->cmap->get_variation_glyph (unicode,
                                             variation_selector, glyph);
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
  // Duplicated in v_advances. Ugly. Keep in sync'ish.

  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  const OT::hmtx_accelerator_t &hmtx = *ot_face->hmtx;

  if (unlikely (!hmtx.has_data ()))
  {
    hb_position_t advance = font->face->get_upem () / 2;
    advance = font->em_scale_x (advance);
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = advance;
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

#ifndef HB_NO_VAR
  if (!font->has_nonzero_coords)
  {
  fallback:
#else
  {
#endif
    // Just plain htmx data. No need to cache.
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->em_scale_x (hmtx.get_advance_without_var_unscaled (*first_glyph));
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

#ifndef HB_NO_VAR
  /* has_nonzero_coords. */

  ot_font->check_serial (font);
  hb_ot_font_advance_cache_t *advance_cache = ot_font->h.acquire_advance_cache ();
  if (!advance_cache)
  {
    // malloc failure. Just use the fallback non-variable path.
    goto fallback;
  }

  /* If HVAR is present, use it.*/
  const OT::HVAR &HVAR = *hmtx.var_table;
  if (HVAR.has_data ())
  {
    const OT::ItemVariationStore &varStore = &HVAR + HVAR.varStore;
    OT::hb_scalar_cache_t *varStore_cache = ot_font->h.acquire_varStore_cache (varStore);

    for (unsigned int i = 0; i < count; i++)
    {
      hb_position_t v;
      unsigned cv;
      if (advance_cache->get (*first_glyph, &cv))
	v = cv;
      else
      {
        v = hmtx.get_advance_with_var_unscaled (*first_glyph, font, varStore_cache);
	advance_cache->set (*first_glyph, v);
      }
      *first_advance = font->em_scale_x (v);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }

    ot_font->h.release_varStore_cache (varStore_cache);
    ot_font->h.release_advance_cache (advance_cache);
    return;
  }

  const auto &gvar = *ot_face->gvar;
  if (gvar.has_data ())
  {
    const auto &glyf = *ot_face->glyf;
    auto *scratch = glyf.acquire_scratch ();
    if (unlikely (!scratch))
    {
      ot_font->h.release_advance_cache (advance_cache);
      goto fallback;
    }
    OT::hb_scalar_cache_t *gvar_cache = ot_font->draw.acquire_gvar_cache (gvar);

    for (unsigned int i = 0; i < count; i++)
    {
      hb_position_t v;
      unsigned cv;
      if (advance_cache->get (*first_glyph, &cv))
	v = cv;
      else
      {
        v = glyf.get_advance_with_var_unscaled (*first_glyph, font, false, *scratch, gvar_cache);
	advance_cache->set (*first_glyph, v);
      }
      *first_advance = font->em_scale_x (v);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }

    ot_font->draw.release_gvar_cache (gvar_cache);
    glyf.release_scratch (scratch);
    ot_font->h.release_advance_cache (advance_cache);
    return;
  }

  ot_font->h.release_advance_cache (advance_cache);
  // No HVAR or GVAR.  Just use the fallback non-variable path.
  goto fallback;
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
  // Duplicated from h_advances. Ugly. Keep in sync'ish.

  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;
  const OT::vmtx_accelerator_t &vmtx = *ot_face->vmtx;

  if (unlikely (!vmtx.has_data ()))
  {
    hb_font_extents_t font_extents;
    font->get_h_extents_with_fallback (&font_extents);
    hb_position_t advance = font_extents.descender - font_extents.ascender;
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = advance;
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

#ifndef HB_NO_VAR
  if (!font->has_nonzero_coords)
  {
  fallback:
#else
  {
#endif
    // Just plain vtmx data. No need to cache.
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->em_scale_y (- (int) vmtx.get_advance_without_var_unscaled (*first_glyph));
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

#ifndef HB_NO_VAR
  /* has_nonzero_coords. */

  ot_font->check_serial (font);
  hb_ot_font_advance_cache_t *advance_cache = ot_font->v.acquire_advance_cache ();
  if (!advance_cache)
  {
    // malloc failure. Just use the fallback non-variable path.
    goto fallback;
  }

  /* If VVAR is present, use it.*/
  const OT::VVAR &VVAR = *vmtx.var_table;
  if (VVAR.has_data ())
  {
    const OT::ItemVariationStore &varStore = &VVAR + VVAR.varStore;
    OT::hb_scalar_cache_t *varStore_cache = ot_font->v.acquire_varStore_cache (varStore);

    for (unsigned int i = 0; i < count; i++)
    {
      hb_position_t v;
      unsigned cv;
      if (advance_cache->get (*first_glyph, &cv))
	v = cv;
      else
      {
        v = vmtx.get_advance_with_var_unscaled (*first_glyph, font, varStore_cache);
	advance_cache->set (*first_glyph, v);
      }
      *first_advance = font->em_scale_y (- (int) v);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }

    ot_font->v.release_varStore_cache (varStore_cache);
    ot_font->v.release_advance_cache (advance_cache);
    return;
  }

  const auto &gvar = *ot_face->gvar;
  if (gvar.has_data ())
  {
    const auto &glyf = *ot_face->glyf;
    auto *scratch = glyf.acquire_scratch ();
    if (unlikely (!scratch))
    {
      ot_font->v.release_advance_cache (advance_cache);
      goto fallback;
    }
    OT::hb_scalar_cache_t *gvar_cache = ot_font->draw.acquire_gvar_cache (gvar);

    for (unsigned int i = 0; i < count; i++)
    {
      hb_position_t v;
      unsigned cv;
      if (advance_cache->get (*first_glyph, &cv))
	v = cv;
      else
      {
        v = glyf.get_advance_with_var_unscaled (*first_glyph, font, true, *scratch, gvar_cache);
	advance_cache->set (*first_glyph, v);
      }
      *first_advance = font->em_scale_y (- (int) v);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }

    ot_font->draw.release_gvar_cache (gvar_cache);
    glyf.release_scratch (scratch);
    ot_font->v.release_advance_cache (advance_cache);
    return;
  }

  ot_font->v.release_advance_cache (advance_cache);
  // No VVAR or GVAR.  Just use the fallback non-variable path.
  goto fallback;
#endif
}
#endif

#ifndef HB_NO_VERTICAL
HB_HOT
static hb_bool_t
hb_ot_get_glyph_v_origins (hb_font_t *font,
			   void *font_data,
			   unsigned int count,
			   const hb_codepoint_t *first_glyph,
			   unsigned glyph_stride,
			   hb_position_t *first_x,
			   unsigned x_stride,
			   hb_position_t *first_y,
			   unsigned y_stride,
			   void *user_data HB_UNUSED)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  const hb_ot_face_t *ot_face = ot_font->ot_face;

  /* First, set all the x values to half the advance width. */
  font->get_glyph_h_advances (count,
			      first_glyph, glyph_stride,
			      first_x, x_stride);
  for (unsigned i = 0; i < count; i++)
  {
    *first_x /= 2;
    first_x = &StructAtOffsetUnaligned<hb_position_t> (first_x, x_stride);
  }

  /* The vertical origin business is messy...
   *
   * We allocate the cache, then have various code paths that use the cache.
   * Each one is responsible to free it before returning.
   */
  hb_ot_font_origin_cache_t *origin_cache = ot_font->v_origin.acquire_origin_cache ();

  /* If there is VORG, always use it. It uses VVAR for variations if necessary. */
  const OT::VORG &VORG = *ot_face->VORG;
  if (origin_cache && VORG.has_data ())
  {
#ifndef HB_NO_VAR
    if (!font->has_nonzero_coords)
#endif
    {
      for (unsigned i = 0; i < count; i++)
      {
	hb_position_t origin;
	unsigned cv;
	if (origin_cache->get (*first_glyph, &cv))
	  origin = font->y_scale < 0 ? -static_cast<hb_position_t>(cv) : static_cast<hb_position_t>(cv);
	else
	{
	  origin = font->em_scalef_y (VORG.get_y_origin (*first_glyph));
	  origin_cache->set (*first_glyph, font->y_scale < 0 ? -origin : origin);
	}

	*first_y = origin;

	first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
	first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
      }
    }
#ifndef HB_NO_VAR
    else
    {
      const OT::VVAR &VVAR = *ot_face->vmtx->var_table;
      const auto &varStore = &VVAR + VVAR.varStore;
      auto *varStore_cache = ot_font->v_origin.acquire_varStore_cache (varStore);
      for (unsigned i = 0; i < count; i++)
      {
	hb_position_t origin;
	unsigned cv;
	if (origin_cache->get (*first_glyph, &cv))
	  origin = font->y_scale < 0 ? -static_cast<hb_position_t>(cv) : static_cast<hb_position_t>(cv);
	else
	{
	  origin = font->em_scalef_y (VORG.get_y_origin (*first_glyph) +
				      VVAR.get_vorg_delta_unscaled (*first_glyph,
								    font->coords, font->num_coords,
								    varStore_cache));
	  origin_cache->set (*first_glyph, font->y_scale < 0 ? -origin : origin);
	}

	*first_y = origin;

	first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
	first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
      }
      ot_font->v_origin.release_varStore_cache (varStore_cache);
    }
#endif
    ot_font->v_origin.release_origin_cache (origin_cache);
    return true;
  }

  /* If and only if `vmtx` is present and it's a `glyf` font,
   * we use the top phantom point, deduced from vmtx,glyf[,gvar]. */
  const auto &vmtx = *ot_face->vmtx;
  const auto &glyf = *ot_face->glyf;
  if (origin_cache && vmtx.has_data() && glyf.has_data ())
  {
    auto *scratch = glyf.acquire_scratch ();
    if (unlikely (!scratch))
    {
      ot_font->v_origin.release_origin_cache (origin_cache);
      return false;
    }
    OT::hb_scalar_cache_t *gvar_cache = font->has_nonzero_coords ?
					ot_font->draw.acquire_gvar_cache (*ot_face->gvar) :
					nullptr;

    for (unsigned i = 0; i < count; i++)
    {
      hb_position_t origin;
      unsigned cv;
      if (origin_cache->get (*first_glyph, &cv))
	origin = font->y_scale < 0 ? -static_cast<hb_position_t>(cv) : static_cast<hb_position_t>(cv);
      else
      {
	origin = font->em_scalef_y (glyf.get_v_origin_with_var_unscaled (*first_glyph, font, *scratch, gvar_cache));
	origin_cache->set (*first_glyph, font->y_scale < 0 ? -origin : origin);
      }

      *first_y = origin;

      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
    }

    if (gvar_cache)
      ot_font->draw.release_gvar_cache (gvar_cache);
    glyf.release_scratch (scratch);
    ot_font->v_origin.release_origin_cache (origin_cache);
    return true;
  }

  /* Otherwise, use glyph extents to center the glyph vertically.
   * If getting glyph extents failed, just use the font ascender. */
  if (origin_cache && font->has_glyph_extents_func ())
  {
    hb_font_extents_t font_extents;
    font->get_h_extents_with_fallback (&font_extents);
    hb_position_t font_advance = font_extents.ascender - font_extents.descender;

    for (unsigned i = 0; i < count; i++)
    {
      hb_position_t origin;
      unsigned cv;

      if (origin_cache->get (*first_glyph, &cv))
	origin = font->y_scale < 0 ? -static_cast<hb_position_t>(cv) : static_cast<hb_position_t>(cv);
      else
      {
	hb_glyph_extents_t extents = {0};
	if (likely (font->get_glyph_extents (*first_glyph, &extents)))
	  origin = extents.y_bearing + ((font_advance - -extents.height) >> 1);
	else
	  origin = font_extents.ascender;

	origin_cache->set (*first_glyph, font->y_scale < 0 ? -origin : origin);
      }

      *first_y = origin;

      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
    }
  }

  ot_font->v_origin.release_origin_cache (origin_cache);
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
  if (ot_face->CBDT->get_extents (font, glyph, extents)) return true;
#endif
#if !defined(HB_NO_COLOR) && !defined(HB_NO_PAINT)
  if (ot_face->COLR->get_extents (font, glyph, extents)) return true;
#endif
#ifndef HB_NO_VAR_COMPOSITES
  if (ot_face->VARC->get_extents (font, glyph, extents)) return true;
#endif
  if (ot_face->glyf->get_extents (font, glyph, extents)) return true;
#ifndef HB_NO_OT_FONT_CFF
  if (ot_face->cff2->get_extents (font, glyph, extents)) return true;
  if (ot_face->cff1->get_extents (font, glyph, extents)) return true;
#endif

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
static hb_bool_t
hb_ot_draw_glyph_or_fail (hb_font_t *font,
			  void *font_data HB_UNUSED,
			  hb_codepoint_t glyph,
			  hb_draw_funcs_t *draw_funcs, void *draw_data,
			  void *user_data)
{
  const hb_ot_font_t *ot_font = (const hb_ot_font_t *) font_data;
  hb_draw_session_t draw_session {draw_funcs, draw_data};
  bool ret = false;

  OT::hb_scalar_cache_t *gvar_cache = nullptr;
  if (font->num_coords)
  {
    ot_font->check_serial (font);
    gvar_cache = ot_font->draw.acquire_gvar_cache (*ot_font->ot_face->gvar);
  }

#ifndef HB_NO_VAR_COMPOSITES
  if (font->face->table.VARC->get_path (font, glyph, draw_session)) { ret = true; goto done; }
#endif
  // Keep the following in synch with VARC::get_path_at()
  if (font->face->table.glyf->get_path (font, glyph, draw_session, gvar_cache)) { ret = true; goto done; }

#ifndef HB_NO_CFF
  if (font->face->table.cff2->get_path (font, glyph, draw_session)) { ret = true; goto done; }
  if (font->face->table.cff1->get_path (font, glyph, draw_session)) { ret = true; goto done; }
#endif

done:

  ot_font->draw.release_gvar_cache (gvar_cache);

  return ret;
}
#endif

#ifndef HB_NO_PAINT
static hb_bool_t
hb_ot_paint_glyph_or_fail (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t glyph,
			   hb_paint_funcs_t *paint_funcs, void *paint_data,
			   unsigned int palette,
			   hb_color_t foreground,
			   void *user_data)
{
#ifndef HB_NO_COLOR
  if (font->face->table.COLR->paint_glyph (font, glyph, paint_funcs, paint_data, palette, foreground)) return true;
  if (font->face->table.SVG->paint_glyph (font, glyph, paint_funcs, paint_data)) return true;
#ifndef HB_NO_OT_FONT_BITMAP
  if (font->face->table.CBDT->paint_glyph (font, glyph, paint_funcs, paint_data)) return true;
  if (font->face->table.sbix->paint_glyph (font, glyph, paint_funcs, paint_data)) return true;
#endif
#endif
  return false;
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

#ifndef HB_NO_VERTICAL
    hb_font_funcs_set_font_v_extents_func (funcs, hb_ot_get_font_v_extents, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_advances_func (funcs, hb_ot_get_glyph_v_advances, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_origins_func (funcs, hb_ot_get_glyph_v_origins, nullptr, nullptr);
#endif

#ifndef HB_NO_DRAW
    hb_font_funcs_set_draw_glyph_or_fail_func (funcs, hb_ot_draw_glyph_or_fail, nullptr, nullptr);
#endif

#ifndef HB_NO_PAINT
    hb_font_funcs_set_paint_glyph_or_fail_func (funcs, hb_ot_paint_glyph_or_fail, nullptr, nullptr);
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
 * Sets the font functions to use when working with @font to
 * the HarfBuzz's native implementation. This is the default
 * for fonts newly created.
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

#endif
