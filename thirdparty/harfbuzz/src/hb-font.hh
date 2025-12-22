/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2011  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_FONT_HH
#define HB_FONT_HH

#include "hb.hh"

#include "hb-face.hh"
#include "hb-atomic.hh"
#include "hb-draw.hh"
#include "hb-paint-extents.hh"
#include "hb-shaper.hh"
#include "hb-outline.hh"


/*
 * hb_font_funcs_t
 */

#define HB_FONT_FUNCS_IMPLEMENT_CALLBACKS \
  HB_FONT_FUNC_IMPLEMENT (get_,font_h_extents) \
  HB_FONT_FUNC_IMPLEMENT (get_,font_v_extents) \
  HB_FONT_FUNC_IMPLEMENT (get_,nominal_glyph) \
  HB_FONT_FUNC_IMPLEMENT (get_,nominal_glyphs) \
  HB_FONT_FUNC_IMPLEMENT (get_,variation_glyph) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_h_advance) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_v_advance) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_h_advances) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_v_advances) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_h_origin) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_v_origin) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_h_origins) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_v_origins) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_h_kerning) \
  HB_IF_NOT_DEPRECATED (HB_FONT_FUNC_IMPLEMENT (get_,glyph_v_kerning)) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_extents) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_contour_point) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_name) \
  HB_FONT_FUNC_IMPLEMENT (get_,glyph_from_name) \
  HB_FONT_FUNC_IMPLEMENT (,draw_glyph_or_fail) \
  HB_FONT_FUNC_IMPLEMENT (,paint_glyph_or_fail) \
  /* ^--- Add new callbacks here */

struct hb_font_funcs_t
{
  hb_object_header_t header;

  struct {
#define HB_FONT_FUNC_IMPLEMENT(get_,name) void *name;
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  } *user_data;

  struct {
#define HB_FONT_FUNC_IMPLEMENT(get_,name) hb_destroy_func_t name;
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  } *destroy;

  /* Don't access these directly.  Call font->get_*() instead. */
  union get_t {
    struct get_funcs_t {
#define HB_FONT_FUNC_IMPLEMENT(get_,name) hb_font_##get_##name##_func_t name;
      HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
    } f;
    void (*array[0
#define HB_FONT_FUNC_IMPLEMENT(get_,name) +1
      HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
		]) ();
  } get;
};
DECLARE_NULL_INSTANCE (hb_font_funcs_t);


/*
 * hb_font_t
 */

#define HB_SHAPER_IMPLEMENT(shaper) HB_SHAPER_DATA_INSTANTIATE_SHAPERS(shaper, font);
#include "hb-shaper-list.hh"
#undef HB_SHAPER_IMPLEMENT

struct hb_font_t
{
  hb_object_header_t header;
  hb_atomic_t<unsigned> serial;
  hb_atomic_t<unsigned> serial_coords;

  hb_font_t *parent;
  hb_face_t *face;

  int32_t x_scale;
  int32_t y_scale;

  bool is_synthetic;

  float x_embolden;
  float y_embolden;
  bool embolden_in_place;
  int32_t x_strength; /* x_embolden, in scaled units. */
  int32_t y_strength; /* y_embolden, in scaled units. */

  float slant;
  float slant_xy;

  float x_multf;
  float y_multf;
  int64_t x_mult;
  int64_t y_mult;

  unsigned int x_ppem;
  unsigned int y_ppem;

  float ptem;

  /* Font variation coordinates. */
  unsigned int instance_index;
  bool has_nonzero_coords;
  unsigned int num_coords;
  int *coords;
  float *design_coords;

  hb_font_funcs_t   *klass;
  void              *user_data;
  hb_destroy_func_t  destroy;

  hb_shaper_object_dataset_t<hb_font_t> data; /* Various shaper data. */


  /* Convert from font-space to user-space */
  int64_t dir_mult (hb_direction_t direction)
  { return HB_DIRECTION_IS_VERTICAL(direction) ? y_mult : x_mult; }
  hb_position_t em_scale_x (int16_t v) { return em_mult (v, x_mult); }
  hb_position_t em_scale_y (int16_t v) { return em_mult (v, y_mult); }
  hb_position_t em_scalef_x (float v) { return em_multf (v, x_multf); }
  hb_position_t em_scalef_y (float v) { return em_multf (v, y_multf); }
  float em_fscale_x (int16_t v) { return em_fmult (v, x_multf); }
  float em_fscale_y (int16_t v) { return em_fmult (v, y_multf); }
  float em_fscalef_x (float v) { return em_fmultf (v, x_multf); }
  float em_fscalef_y (float v) { return em_fmultf (v, y_multf); }
  hb_position_t em_scale_dir (int16_t v, hb_direction_t direction)
  { return em_mult (v, dir_mult (direction)); }

  /* Convert from parent-font user-space to our user-space */
  hb_position_t parent_scale_x_distance (hb_position_t v)
  {
    if (unlikely (parent && parent->x_scale != x_scale))
      return (hb_position_t) (v * (int64_t) this->x_scale / this->parent->x_scale);
    return v;
  }
  hb_position_t parent_scale_y_distance (hb_position_t v)
  {
    if (unlikely (parent && parent->y_scale != y_scale))
      return (hb_position_t) (v * (int64_t) this->y_scale / this->parent->y_scale);
    return v;
  }
  hb_position_t parent_scale_x_position (hb_position_t v)
  { return parent_scale_x_distance (v); }
  hb_position_t parent_scale_y_position (hb_position_t v)
  { return parent_scale_y_distance (v); }

  void parent_scale_distance (hb_position_t *x, hb_position_t *y)
  {
    *x = parent_scale_x_distance (*x);
    *y = parent_scale_y_distance (*y);
  }
  void parent_scale_position (hb_position_t *x, hb_position_t *y)
  {
    *x = parent_scale_x_position (*x);
    *y = parent_scale_y_position (*y);
  }

  void scale_glyph_extents (hb_glyph_extents_t *extents)
  {
    float x1 = em_scale_x (extents->x_bearing);
    float y1 = em_scale_y (extents->y_bearing);
    float x2 = em_scale_x (extents->x_bearing + extents->width);
    float y2 = em_scale_y (extents->y_bearing + extents->height);

    extents->x_bearing = roundf (x1);
    extents->y_bearing = roundf (y1);
    extents->width = roundf (x2) - extents->x_bearing;
    extents->height = roundf (y2) - extents->y_bearing;
  }

  void synthetic_glyph_extents (hb_glyph_extents_t *extents)
  {
    /* Slant. */
    if (slant_xy)
    {
      hb_position_t x1 = extents->x_bearing;
      hb_position_t y1 = extents->y_bearing;
      hb_position_t x2 = extents->x_bearing + extents->width;
      hb_position_t y2 = extents->y_bearing + extents->height;

      x1 += floorf (hb_min (y1 * slant_xy, y2 * slant_xy));
      x2 += ceilf (hb_max (y1 * slant_xy, y2 * slant_xy));

      extents->x_bearing = x1;
      extents->width = x2 - extents->x_bearing;
    }

    /* Embolden. */
    if (x_strength || y_strength)
    {
      /* Y */
      int y_shift = y_strength;
      if (y_scale < 0) y_shift = -y_shift;
      extents->y_bearing += y_shift;
      extents->height -= y_shift;

      /* X */
      int x_shift = x_strength;
      if (x_scale < 0) x_shift = -x_shift;
      if (embolden_in_place)
	extents->x_bearing -= x_shift / 2;
      extents->width += x_shift;
    }
  }


  /* Public getters */

  HB_INTERNAL bool has_func (unsigned int i);
  HB_INTERNAL bool has_func_set (unsigned int i);

  /* has_* ... */
#define HB_FONT_FUNC_IMPLEMENT(get_,name) \
  bool \
  has_##name##_func () \
  { \
    hb_font_funcs_t *funcs = this->klass; \
    unsigned int i = offsetof (hb_font_funcs_t::get_t::get_funcs_t, name) / sizeof (funcs->get.array[0]); \
    return has_func (i); \
  } \
  bool \
  has_##name##_func_set () \
  { \
    hb_font_funcs_t *funcs = this->klass; \
    unsigned int i = offsetof (hb_font_funcs_t::get_t::get_funcs_t, name) / sizeof (funcs->get.array[0]); \
    return has_func_set (i); \
  }
  HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT

  hb_bool_t get_font_h_extents (hb_font_extents_t *extents,
				bool synthetic = true)
  {
    hb_memset (extents, 0, sizeof (*extents));
    bool ret = klass->get.f.font_h_extents (this, user_data,
					    extents,
					    !klass->user_data ? nullptr : klass->user_data->font_h_extents);

    if (synthetic && ret)
    {
      /* Embolden */
      int y_shift = y_scale < 0 ? -y_strength : y_strength;
      extents->ascender += y_shift;
    }

    return ret;
  }
  hb_bool_t get_font_v_extents (hb_font_extents_t *extents,
				bool synthetic = true)
  {
    hb_memset (extents, 0, sizeof (*extents));
    bool ret = klass->get.f.font_v_extents (this, user_data,
					    extents,
					    !klass->user_data ? nullptr : klass->user_data->font_v_extents);

    if (synthetic && ret)
    {
      /* Embolden */
      int x_shift = x_scale < 0 ? -x_strength : x_strength;
      if (embolden_in_place)
      {
	extents->ascender += x_shift / 2;
	extents->descender -= x_shift - x_shift / 2;
      }
      else
	extents->ascender += x_shift;
    }

    return ret;
  }

  bool has_glyph (hb_codepoint_t unicode)
  {
    hb_codepoint_t glyph;
    return get_nominal_glyph (unicode, &glyph);
  }

  hb_bool_t get_nominal_glyph (hb_codepoint_t unicode,
			       hb_codepoint_t *glyph,
			       hb_codepoint_t not_found = 0)
  {
    *glyph = not_found;
    return klass->get.f.nominal_glyph (this, user_data,
				       unicode, glyph,
				       !klass->user_data ? nullptr : klass->user_data->nominal_glyph);
  }
  unsigned int get_nominal_glyphs (unsigned int count,
				   const hb_codepoint_t *first_unicode,
				   unsigned int unicode_stride,
				   hb_codepoint_t *first_glyph,
				   unsigned int glyph_stride)
  {
    return klass->get.f.nominal_glyphs (this, user_data,
					count,
					first_unicode, unicode_stride,
					first_glyph, glyph_stride,
					!klass->user_data ? nullptr : klass->user_data->nominal_glyphs);
  }

  hb_bool_t get_variation_glyph (hb_codepoint_t unicode, hb_codepoint_t variation_selector,
				 hb_codepoint_t *glyph,
				 hb_codepoint_t not_found = 0)
  {
    *glyph = not_found;
    return klass->get.f.variation_glyph (this, user_data,
					 unicode, variation_selector, glyph,
					 !klass->user_data ? nullptr : klass->user_data->variation_glyph);
  }

  hb_position_t get_glyph_h_advance (hb_codepoint_t glyph,
				     bool synthetic = true)
  {
    hb_position_t advance = klass->get.f.glyph_h_advance (this, user_data,
							  glyph,
							  !klass->user_data ? nullptr : klass->user_data->glyph_h_advance);

    if (synthetic && x_strength && !embolden_in_place)
    {
      /* Embolden */
      hb_position_t strength = x_scale >= 0 ? x_strength : -x_strength;
      advance += advance ? strength : 0;
    }

    return advance;
  }

  hb_position_t get_glyph_v_advance (hb_codepoint_t glyph,
				     bool synthetic = true)
  {
    hb_position_t advance = klass->get.f.glyph_v_advance (this, user_data,
							  glyph,
							  !klass->user_data ? nullptr : klass->user_data->glyph_v_advance);

    if (synthetic && y_strength && !embolden_in_place)
    {
      /* Embolden */
      hb_position_t strength = y_scale >= 0 ? y_strength : -y_strength;
      advance += advance ? strength : 0;
    }

    return advance;
  }

  void get_glyph_h_advances (unsigned int count,
			     const hb_codepoint_t *first_glyph,
			     unsigned int glyph_stride,
			     hb_position_t *first_advance,
			     unsigned int advance_stride,
			     bool synthetic = true)
  {
    klass->get.f.glyph_h_advances (this, user_data,
				   count,
				   first_glyph, glyph_stride,
				   first_advance, advance_stride,
				   !klass->user_data ? nullptr : klass->user_data->glyph_h_advances);

    if (synthetic && x_strength && !embolden_in_place)
    {
      /* Embolden */
      hb_position_t strength = x_scale >= 0 ? x_strength : -x_strength;
      for (unsigned int i = 0; i < count; i++)
      {
	*first_advance += *first_advance ? strength : 0;
	first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
      }
    }
  }

  void get_glyph_v_advances (unsigned int count,
			     const hb_codepoint_t *first_glyph,
			     unsigned int glyph_stride,
			     hb_position_t *first_advance,
			     unsigned int advance_stride,
			     bool synthetic = true)
  {
    klass->get.f.glyph_v_advances (this, user_data,
				   count,
				   first_glyph, glyph_stride,
				   first_advance, advance_stride,
				   !klass->user_data ? nullptr : klass->user_data->glyph_v_advances);

    if (synthetic && y_strength && !embolden_in_place)
    {
      /* Embolden */
      hb_position_t strength = y_scale >= 0 ? y_strength : -y_strength;
      for (unsigned int i = 0; i < count; i++)
      {
	*first_advance += *first_advance ? strength : 0;
	first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
      }
    }
  }

  hb_bool_t get_glyph_h_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y,
				bool synthetic = true)
  {
    *x = *y = 0;
    bool ret = klass->get.f.glyph_h_origin (this, user_data,
					    glyph, x, y,
					    !klass->user_data ? nullptr : klass->user_data->glyph_h_origin);

    if (synthetic && ret)
    {
      /* Slant is ignored as it does not affect glyph origin */

      /* Embolden */
      if (!embolden_in_place)
      {
        *x += x_scale < 0 ? -x_strength : x_strength;
	*y += y_scale < 0 ? -y_strength : y_strength;
      }
    }

    return ret;
  }

  hb_bool_t get_glyph_v_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y,
				bool synthetic = true)
  {
    *x = *y = 0;
    bool ret = klass->get.f.glyph_v_origin (this, user_data,
					    glyph, x, y,
					    !klass->user_data ? nullptr : klass->user_data->glyph_v_origin);

    if (synthetic && ret)
    {
      /* Slant is ignored as it does not affect glyph origin */

      /* Embolden */
      if (!embolden_in_place)
      {
        *x += x_scale < 0 ? -x_strength : x_strength;
	*y += y_scale < 0 ? -y_strength : y_strength;
      }
    }

    return ret;
  }

  hb_bool_t get_glyph_h_origins (unsigned int count,
				 const hb_codepoint_t *first_glyph,
				 unsigned int glyph_stride,
				 hb_position_t *first_x,
				 unsigned int x_stride,
				 hb_position_t *first_y,
				 unsigned int y_stride,
				 bool synthetic = true)

  {
    bool ret = klass->get.f.glyph_h_origins (this, user_data,
					     count,
					     first_glyph, glyph_stride,
					     first_x, x_stride, first_y, y_stride,
					     !klass->user_data ? nullptr : klass->user_data->glyph_h_origins);

    if (synthetic && ret)
    {
      hb_position_t x_shift = x_scale < 0 ? -x_strength : x_strength;
      hb_position_t y_shift = y_scale < 0 ? -y_strength : y_strength;
      for (unsigned i = 0; i < count; i++)
      {
	/* Slant is ignored as it does not affect glyph origin */

	/* Embolden */
	if (!embolden_in_place)
	{
	  *first_x += x_shift;
	  *first_y += y_shift;
	}
	first_x = &StructAtOffsetUnaligned<hb_position_t> (first_x, x_stride);
	first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
      }
    }

    return ret;
  }

  hb_bool_t get_glyph_v_origins (unsigned int count,
				 const hb_codepoint_t *first_glyph,
				 unsigned int glyph_stride,
				 hb_position_t *first_x,
				 unsigned int x_stride,
				 hb_position_t *first_y,
				 unsigned int y_stride,
				 bool synthetic = true)

  {
    bool ret = klass->get.f.glyph_v_origins (this, user_data,
					     count,
					     first_glyph, glyph_stride,
					     first_x, x_stride, first_y, y_stride,
					     !klass->user_data ? nullptr : klass->user_data->glyph_v_origins);

    if (synthetic && is_synthetic && ret)
    {
      hb_position_t x_shift = x_scale < 0 ? -x_strength : x_strength;
      hb_position_t y_shift = y_scale < 0 ? -y_strength : y_strength;
      for (unsigned i = 0; i < count; i++)
      {
	/* Slant is ignored as it does not affect glyph origin */

	/* Embolden */
	if (!embolden_in_place)
	{
	  *first_x += x_shift;
	  *first_y += y_shift;
	}
	first_x = &StructAtOffsetUnaligned<hb_position_t> (first_x, x_stride);
	first_y = &StructAtOffsetUnaligned<hb_position_t> (first_y, y_stride);
      }
    }

    return ret;
  }

  hb_position_t get_glyph_h_kerning (hb_codepoint_t left_glyph,
				     hb_codepoint_t right_glyph)
  {
#ifdef HB_DISABLE_DEPRECATED
    return 0;
#else
    return klass->get.f.glyph_h_kerning (this, user_data,
					 left_glyph, right_glyph,
					 !klass->user_data ? nullptr : klass->user_data->glyph_h_kerning);
#endif
  }

  hb_position_t get_glyph_v_kerning (hb_codepoint_t top_glyph,
				     hb_codepoint_t bottom_glyph)
  {
#ifdef HB_DISABLE_DEPRECATED
    return 0;
#else
    return klass->get.f.glyph_v_kerning (this, user_data,
					 top_glyph, bottom_glyph,
					 !klass->user_data ? nullptr : klass->user_data->glyph_v_kerning);
#endif
  }

  hb_bool_t get_glyph_extents (hb_codepoint_t glyph,
			       hb_glyph_extents_t *extents,
			       bool synthetic = true)
  {
    hb_memset (extents, 0, sizeof (*extents));

    /* This is rather messy, but necessary. */

    if (!synthetic)
    {
      return klass->get.f.glyph_extents (this, user_data,
					 glyph,
					 extents,
					 !klass->user_data ? nullptr : klass->user_data->glyph_extents);
    }
    if (!is_synthetic &&
	klass->get.f.glyph_extents (this, user_data,
				    glyph,
				    extents,
				    !klass->user_data ? nullptr : klass->user_data->glyph_extents))
      return true;

    /* Try getting extents from paint(), then draw(), *then* get_extents()
     * and apply synthetic settings in the last case. */

#ifndef HB_NO_PAINT
    hb_paint_extents_context_t paint_extents;
    if (paint_glyph_or_fail (glyph,
			     hb_paint_extents_get_funcs (), &paint_extents,
			     0, 0))
    {
      *extents = paint_extents.get_extents ().to_glyph_extents ();
      return true;
    }
#endif

#ifndef HB_NO_DRAW
    hb_extents_t<> draw_extents;
    if (draw_glyph_or_fail (glyph,
			    hb_draw_extents_get_funcs (), &draw_extents))
    {
      *extents = draw_extents.to_glyph_extents ();
      return true;
    }
#endif

    bool ret = klass->get.f.glyph_extents (this, user_data,
					   glyph,
					   extents,
					   !klass->user_data ? nullptr : klass->user_data->glyph_extents);
    if (ret)
      synthetic_glyph_extents (extents);

    return ret;
  }

  hb_bool_t get_glyph_contour_point (hb_codepoint_t glyph, unsigned int point_index,
				     hb_position_t *x, hb_position_t *y,
				     bool synthetic = true)
  {
    *x = *y = 0;
    bool ret = klass->get.f.glyph_contour_point (this, user_data,
						 glyph, point_index,
						 x, y,
						 !klass->user_data ? nullptr : klass->user_data->glyph_contour_point);

    if (synthetic && ret)
    {
      /* Slant */
      if (slant_xy)
        *x += roundf (*y * slant_xy);

      /* Embolden */
      if (!embolden_in_place)
      {
	int x_shift = x_scale < 0 ? -x_strength : x_strength;
	*x += x_shift;
      }
    }

    return ret;
  }

  hb_bool_t get_glyph_name (hb_codepoint_t glyph,
			    char *name, unsigned int size)
  {
    if (size) *name = '\0';
    return klass->get.f.glyph_name (this, user_data,
				    glyph,
				    name, size,
				    !klass->user_data ? nullptr : klass->user_data->glyph_name);
  }

  hb_bool_t get_glyph_from_name (const char *name, int len, /* -1 means nul-terminated */
				 hb_codepoint_t *glyph)
  {
    *glyph = 0;
    if (len == -1) len = strlen (name);
    return klass->get.f.glyph_from_name (this, user_data,
					 name, len,
					 glyph,
					 !klass->user_data ? nullptr : klass->user_data->glyph_from_name);
  }

  bool draw_glyph_or_fail (hb_codepoint_t glyph,
			   hb_draw_funcs_t *draw_funcs, void *draw_data,
			   bool synthetic = true)
  {
#ifndef HB_NO_DRAW
#ifndef HB_NO_OUTLINE
    bool embolden = x_strength || y_strength;
    bool slanted = slant_xy;
    synthetic = synthetic && (embolden || slanted);
#else
    synthetic = false;
#endif

    if (!synthetic)
    {
      return klass->get.f.draw_glyph_or_fail (this, user_data,
					      glyph,
					      draw_funcs, draw_data,
					      !klass->user_data ? nullptr : klass->user_data->draw_glyph_or_fail);
    }

#ifndef HB_NO_OUTLINE

    hb_outline_t outline;
    if (!klass->get.f.draw_glyph_or_fail (this, user_data,
					  glyph,
					  hb_outline_recording_pen_get_funcs (), &outline,
					  !klass->user_data ? nullptr : klass->user_data->draw_glyph_or_fail))
      return false;

    // Slant before embolden; produces nicer results.

    if (slanted)
      outline.slant (slant_xy);

    if (embolden)
    {
      float x_shift = embolden_in_place ? 0 : (float) x_strength / 2;
      float y_shift = (float) y_strength / 2;
      if (x_scale < 0) x_shift = -x_shift;
      if (y_scale < 0) y_shift = -y_shift;
      outline.embolden (x_strength, y_strength, x_shift, y_shift);
    }

    outline.replay (draw_funcs, draw_data);

    return true;
#endif
#endif
    return false;
  }

  bool paint_glyph_or_fail (hb_codepoint_t glyph,
			    hb_paint_funcs_t *paint_funcs, void *paint_data,
			    unsigned int palette,
			    hb_color_t foreground,
			    bool synthetic = true)
  {
#ifndef HB_NO_PAINT
    /* Slant */
    if (synthetic && slant_xy)
      hb_paint_push_transform (paint_funcs, paint_data,
			       1.f, 0.f,
			       slant_xy, 1.f,
			       0.f, 0.f);

    bool ret = klass->get.f.paint_glyph_or_fail (this, user_data,
						 glyph,
						 paint_funcs, paint_data,
						 palette, foreground,
						 !klass->user_data ? nullptr : klass->user_data->paint_glyph_or_fail);

    if (synthetic && slant_xy)
      hb_paint_pop_transform (paint_funcs, paint_data);

    return ret;
#endif
    return false;
  }

  /* A bit higher-level, and with fallback */

  HB_INTERNAL
  void paint_glyph (hb_codepoint_t glyph,
		    hb_paint_funcs_t *paint_funcs, void *paint_data,
		    unsigned int palette,
		    hb_color_t foreground);

  void get_h_extents_with_fallback (hb_font_extents_t *extents)
  {
    if (!get_font_h_extents (extents))
    {
      extents->ascender = y_scale * .8;
      extents->descender = extents->ascender - y_scale;
      extents->line_gap = 0;
    }
  }
  void get_v_extents_with_fallback (hb_font_extents_t *extents)
  {
    if (!get_font_v_extents (extents))
    {
      extents->ascender = x_scale / 2;
      extents->descender = extents->ascender - x_scale;
      extents->line_gap = 0;
    }
  }

  void get_extents_for_direction (hb_direction_t direction,
				  hb_font_extents_t *extents)
  {
    if (likely (HB_DIRECTION_IS_HORIZONTAL (direction)))
      get_h_extents_with_fallback (extents);
    else
      get_v_extents_with_fallback (extents);
  }

  void get_glyph_advance_for_direction (hb_codepoint_t glyph,
					hb_direction_t direction,
					hb_position_t *x, hb_position_t *y)
  {
    *x = *y = 0;
    if (likely (HB_DIRECTION_IS_HORIZONTAL (direction)))
      *x = get_glyph_h_advance (glyph);
    else
      *y = get_glyph_v_advance (glyph);
  }
  void get_glyph_advances_for_direction (hb_direction_t direction,
					 unsigned int count,
					 const hb_codepoint_t *first_glyph,
					 unsigned glyph_stride,
					 hb_position_t *first_advance,
					 unsigned advance_stride)
  {
    if (likely (HB_DIRECTION_IS_HORIZONTAL (direction)))
      get_glyph_h_advances (count, first_glyph, glyph_stride, first_advance, advance_stride);
    else
      get_glyph_v_advances (count, first_glyph, glyph_stride, first_advance, advance_stride);
  }

  void apply_offset (hb_position_t *x, hb_position_t *y,
		     hb_position_t dx, hb_position_t dy,
		     signed mult)
  {
    assert (mult == -1 || mult == +1);

    *x += dx * mult;
    *y += dy * mult;
  }
  void add_offset (hb_position_t *x, hb_position_t *y,
		   hb_position_t dx, hb_position_t dy)
  {
    *x += dx;
    *y += dy;
  }
  void subtract_offset (hb_position_t *x, hb_position_t *y,
			hb_position_t dx, hb_position_t dy)
  {
    *x -= dx;
    *y -= dy;
  }

  void guess_v_origin_minus_h_origin (hb_codepoint_t glyph,
				      hb_position_t *x, hb_position_t *y)
  {
    *x = get_glyph_h_advance (glyph) / 2;

    hb_font_extents_t extents;
    get_h_extents_with_fallback (&extents);
    *y = extents.ascender;
  }

  void apply_glyph_h_origins_with_fallback (hb_buffer_t *buf, int mult)
  {
    bool has_ascender = false;
    hb_position_t ascender = 0;

    struct { hb_position_t x, y; } origins[32];

    unsigned int offset = 0;
    unsigned int count = buf->len;
    while (offset < count)
    {
      unsigned n = hb_min (count - offset, ARRAY_LENGTH (origins));
      if (!get_glyph_h_origins (n,
				&buf->info[offset].codepoint, sizeof (hb_glyph_info_t),
				&origins[0].x, sizeof (origins[0]),
				&origins[0].y, sizeof (origins[0])))
      {
        if (get_glyph_v_origins (n,
				  &buf->info[offset].codepoint, sizeof (hb_glyph_info_t),
				  &origins[0].x, sizeof (origins[0]),
				  &origins[0].y, sizeof (origins[0])))
	{
	  if (!has_ascender)
	  {
	    hb_font_extents_t extents;
	    get_h_extents_with_fallback (&extents);
	    ascender = extents.ascender;
	    has_ascender = true;
	  }

	  /* We got the v_origins, adjust them to h_origins. */
	  for (unsigned j = 0; j < n; j++)
	  {
	    hb_codepoint_t glyph = buf->info[offset + j].codepoint;
	    origins[j].x -= get_glyph_h_advance (glyph) / 2;
	    origins[j].y -= ascender;
	  }
	}
	else
	{
	  for (unsigned j = 0; j < n; j++)
	  {
	    origins[j].x = 0;
	    origins[j].y = 0;
	  }
	}
      }

      assert (mult == -1 || mult == +1);
      if (mult == +1)
        for (unsigned j = 0; j < n; j++)
	{
	  hb_glyph_position_t *pos = &buf->pos[offset + j];
	  add_offset (&pos->x_offset, &pos->y_offset,
		      origins[j].x, origins[j].y);
	}
      else /* mult == -1 */
	for (unsigned j = 0; j < n; j++)
	{
	  hb_glyph_position_t *pos = &buf->pos[offset + j];
	  subtract_offset (&pos->x_offset, &pos->y_offset,
			   origins[j].x, origins[j].y);
	}

      offset += n;
    }
  }
  void apply_glyph_v_origins_with_fallback (hb_buffer_t *buf, int mult)
  {
    bool has_ascender = false;
    hb_position_t ascender = 0;

    struct { hb_position_t x, y; } origins[32];

    unsigned int offset = 0;
    unsigned int count = buf->len;
    while (offset < count)
    {
      unsigned n = hb_min (count - offset, ARRAY_LENGTH (origins));
      if (!get_glyph_v_origins (n,
				&buf->info[offset].codepoint, sizeof (hb_glyph_info_t),
				&origins[0].x, sizeof (origins[0]),
				&origins[0].y, sizeof (origins[0])))
      {
	if (get_glyph_h_origins (n,
				 &buf->info[offset].codepoint, sizeof (hb_glyph_info_t),
				 &origins[0].x, sizeof (origins[0]),
				 &origins[0].y, sizeof (origins[0])))
	{
	  if (!has_ascender)
	  {
	    hb_font_extents_t extents;
	    get_h_extents_with_fallback (&extents);
	    ascender = extents.ascender;
	    has_ascender = true;
	  }

	  /* We got the h_origins, adjust them to v_origins. */
	  for (unsigned j = 0; j < n; j++)
	  {
	    hb_codepoint_t glyph = buf->info[offset + j].codepoint;
	    origins[j].x += get_glyph_h_advance (glyph) / 2;
	    origins[j].y += ascender;
	  }
	}
	else
	{
	  for (unsigned j = 0; j < n; j++)
	  {
	    origins[j].x = 0;
	    origins[j].y = 0;
	  }
	}
      }

      assert (mult == -1 || mult == +1);
      if (mult == +1)
        for (unsigned j = 0; j < n; j++)
	{
	  hb_glyph_position_t *pos = &buf->pos[offset + j];
	  add_offset (&pos->x_offset, &pos->y_offset,
		      origins[j].x, origins[j].y);
	}
      else /* mult == -1 */
	for (unsigned j = 0; j < n; j++)
	{
	  hb_glyph_position_t *pos = &buf->pos[offset + j];
	  subtract_offset (&pos->x_offset, &pos->y_offset,
			   origins[j].x, origins[j].y);
	}

      offset += n;
    }
  }

  void get_glyph_h_origin_with_fallback (hb_codepoint_t glyph,
					 hb_position_t *x, hb_position_t *y)
  {
    if (!get_glyph_h_origin (glyph, x, y) &&
	 get_glyph_v_origin (glyph, x, y))
    {
      hb_position_t dx, dy;
      guess_v_origin_minus_h_origin (glyph, &dx, &dy);
      subtract_offset (x, y, dx, dy);
    }
  }
  void get_glyph_v_origin_with_fallback (hb_codepoint_t glyph,
					 hb_position_t *x, hb_position_t *y)
  {
    if (!get_glyph_v_origin (glyph, x, y) &&
	 get_glyph_h_origin (glyph, x, y))
    {
      hb_position_t dx, dy;
      guess_v_origin_minus_h_origin (glyph, &dx, &dy);
      add_offset (x, y, dx, dy);
    }
  }

  void get_glyph_origin_for_direction (hb_codepoint_t glyph,
				       hb_direction_t direction,
				       hb_position_t *x, hb_position_t *y)
  {
    if (likely (HB_DIRECTION_IS_HORIZONTAL (direction)))
      get_glyph_h_origin_with_fallback (glyph, x, y);
    else
      get_glyph_v_origin_with_fallback (glyph, x, y);
  }

  void add_glyph_h_origins (hb_buffer_t *buf)
  {
    apply_glyph_h_origins_with_fallback (buf, +1);
  }
  void add_glyph_v_origins (hb_buffer_t *buf)
  {
    apply_glyph_v_origins_with_fallback (buf, +1);
  }
  void add_glyph_origin_for_direction (hb_codepoint_t glyph,
				       hb_direction_t direction,
				       hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;
    get_glyph_origin_for_direction (glyph, direction, &origin_x, &origin_y);
    add_offset (x, y, origin_x, origin_y);
  }

  void subtract_glyph_h_origins (hb_buffer_t *buf)
  {
    apply_glyph_h_origins_with_fallback (buf, -1);
  }
  void subtract_glyph_v_origins (hb_buffer_t *buf)
  {
    apply_glyph_v_origins_with_fallback (buf, -1);
  }
  void subtract_glyph_origin_for_direction (hb_codepoint_t glyph,
					    hb_direction_t direction,
					    hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;
    get_glyph_origin_for_direction (glyph, direction, &origin_x, &origin_y);
    subtract_offset (x, y, origin_x, origin_y);
  }

  void get_glyph_kerning_for_direction (hb_codepoint_t first_glyph, hb_codepoint_t second_glyph,
					hb_direction_t direction,
					hb_position_t *x, hb_position_t *y)
  {
    if (likely (HB_DIRECTION_IS_HORIZONTAL (direction))) {
      *y = 0;
      *x = get_glyph_h_kerning (first_glyph, second_glyph);
    } else {
      *x = 0;
      *y = get_glyph_v_kerning (first_glyph, second_glyph);
    }
  }

  hb_bool_t get_glyph_extents_for_origin (hb_codepoint_t glyph,
					  hb_direction_t direction,
					  hb_glyph_extents_t *extents)
  {
    hb_bool_t ret = get_glyph_extents (glyph, extents);

    if (ret)
      subtract_glyph_origin_for_direction (glyph, direction, &extents->x_bearing, &extents->y_bearing);

    return ret;
  }

  hb_bool_t get_glyph_contour_point_for_origin (hb_codepoint_t glyph, unsigned int point_index,
						hb_direction_t direction,
						hb_position_t *x, hb_position_t *y)
  {
    hb_bool_t ret = get_glyph_contour_point (glyph, point_index, x, y);

    if (ret)
      subtract_glyph_origin_for_direction (glyph, direction, x, y);

    return ret;
  }

  /* Generates gidDDD if glyph has no name. */
  void
  glyph_to_string (hb_codepoint_t glyph,
		   char *s, unsigned int size)
  {
    if (get_glyph_name (glyph, s, size)) return;

    if (size && snprintf (s, size, "gid%" PRIu32, glyph) < 0)
      *s = '\0';
  }

  /* Parses gidDDD and uniUUUU strings automatically. */
  hb_bool_t
  glyph_from_string (const char *s, int len, /* -1 means nul-terminated */
		     hb_codepoint_t *glyph)
  {
    if (get_glyph_from_name (s, len, glyph)) return true;

    if (len == -1) len = strlen (s);

    /* Straight glyph index. */
    if (hb_codepoint_parse (s, len, 10, glyph))
      return true;

    if (len > 3)
    {
      /* gidDDD syntax for glyph indices. */
      if (0 == strncmp (s, "gid", 3) &&
	  hb_codepoint_parse (s + 3, len - 3, 10, glyph))
	return true;

      /* uniUUUU and other Unicode character indices. */
      hb_codepoint_t unichar;
      if (0 == strncmp (s, "uni", 3) &&
	  hb_codepoint_parse (s + 3, len - 3, 16, &unichar) &&
	  get_nominal_glyph (unichar, glyph))
	return true;
    }

    return false;
  }

  void changed ()
  {
    float upem = face->get_upem ();

    x_multf = x_scale / upem;
    y_multf = y_scale / upem;
    bool x_neg = x_scale < 0;
    x_mult = (x_neg ? -((int64_t) -x_scale << 16) : ((int64_t) x_scale << 16)) / upem;
    bool y_neg = y_scale < 0;
    y_mult = (y_neg ? -((int64_t) -y_scale << 16) : ((int64_t) y_scale << 16)) / upem;

    is_synthetic =  x_embolden || y_embolden || slant;

    x_strength = roundf (abs (x_scale) * x_embolden);
    y_strength = roundf (abs (y_scale) * y_embolden);

    slant_xy = y_scale ? slant * x_scale / y_scale : 0.f;

    data.fini ();

    serial++;
  }

  hb_position_t em_mult (int16_t v, int64_t mult)
  { return (hb_position_t) ((v * mult + 32768) >> 16); }
  hb_position_t em_multf (float v, float mult)
  { return (hb_position_t) roundf (em_fmultf (v, mult)); }
  float em_fmultf (float v, float mult)
  { return v * mult; }
  float em_fmult (int16_t v, float mult)
  { return (float) v * mult; }
};
DECLARE_NULL_INSTANCE (hb_font_t);


#endif /* HB_FONT_HH */
