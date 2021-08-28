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
#include "hb-shaper.hh"


/*
 * hb_font_funcs_t
 */

#define HB_FONT_FUNCS_IMPLEMENT_CALLBACKS \
  HB_FONT_FUNC_IMPLEMENT (font_h_extents) \
  HB_FONT_FUNC_IMPLEMENT (font_v_extents) \
  HB_FONT_FUNC_IMPLEMENT (nominal_glyph) \
  HB_FONT_FUNC_IMPLEMENT (nominal_glyphs) \
  HB_FONT_FUNC_IMPLEMENT (variation_glyph) \
  HB_FONT_FUNC_IMPLEMENT (glyph_h_advance) \
  HB_FONT_FUNC_IMPLEMENT (glyph_v_advance) \
  HB_FONT_FUNC_IMPLEMENT (glyph_h_advances) \
  HB_FONT_FUNC_IMPLEMENT (glyph_v_advances) \
  HB_FONT_FUNC_IMPLEMENT (glyph_h_origin) \
  HB_FONT_FUNC_IMPLEMENT (glyph_v_origin) \
  HB_FONT_FUNC_IMPLEMENT (glyph_h_kerning) \
  HB_IF_NOT_DEPRECATED (HB_FONT_FUNC_IMPLEMENT (glyph_v_kerning)) \
  HB_FONT_FUNC_IMPLEMENT (glyph_extents) \
  HB_FONT_FUNC_IMPLEMENT (glyph_contour_point) \
  HB_FONT_FUNC_IMPLEMENT (glyph_name) \
  HB_FONT_FUNC_IMPLEMENT (glyph_from_name) \
  /* ^--- Add new callbacks here */

struct hb_font_funcs_t
{
  hb_object_header_t header;

  struct {
#define HB_FONT_FUNC_IMPLEMENT(name) void *name;
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  } user_data;

  struct {
#define HB_FONT_FUNC_IMPLEMENT(name) hb_destroy_func_t name;
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  } destroy;

  /* Don't access these directly.  Call font->get_*() instead. */
  union get_t {
    struct get_funcs_t {
#define HB_FONT_FUNC_IMPLEMENT(name) hb_font_get_##name##_func_t name;
      HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
    } f;
    void (*array[0
#define HB_FONT_FUNC_IMPLEMENT(name) +1
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

  hb_font_t *parent;
  hb_face_t *face;

  int32_t x_scale;
  int32_t y_scale;
  int64_t x_mult;
  int64_t y_mult;

  unsigned int x_ppem;
  unsigned int y_ppem;

  float ptem;

  /* Font variation coordinates. */
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
  hb_position_t em_scalef_x (float v) { return em_scalef (v, x_scale); }
  hb_position_t em_scalef_y (float v) { return em_scalef (v, y_scale); }
  float em_fscale_x (int16_t v) { return em_fscale (v, x_scale); }
  float em_fscale_y (int16_t v) { return em_fscale (v, y_scale); }
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


  /* Public getters */

  HB_INTERNAL bool has_func (unsigned int i);
  HB_INTERNAL bool has_func_set (unsigned int i);

  /* has_* ... */
#define HB_FONT_FUNC_IMPLEMENT(name) \
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

  hb_bool_t get_font_h_extents (hb_font_extents_t *extents)
  {
    memset (extents, 0, sizeof (*extents));
    return klass->get.f.font_h_extents (this, user_data,
					extents,
					klass->user_data.font_h_extents);
  }
  hb_bool_t get_font_v_extents (hb_font_extents_t *extents)
  {
    memset (extents, 0, sizeof (*extents));
    return klass->get.f.font_v_extents (this, user_data,
					extents,
					klass->user_data.font_v_extents);
  }

  bool has_glyph (hb_codepoint_t unicode)
  {
    hb_codepoint_t glyph;
    return get_nominal_glyph (unicode, &glyph);
  }

  hb_bool_t get_nominal_glyph (hb_codepoint_t unicode,
			       hb_codepoint_t *glyph)
  {
    *glyph = 0;
    return klass->get.f.nominal_glyph (this, user_data,
				       unicode, glyph,
				       klass->user_data.nominal_glyph);
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
					klass->user_data.nominal_glyphs);
  }

  hb_bool_t get_variation_glyph (hb_codepoint_t unicode, hb_codepoint_t variation_selector,
				 hb_codepoint_t *glyph)
  {
    *glyph = 0;
    return klass->get.f.variation_glyph (this, user_data,
					 unicode, variation_selector, glyph,
					 klass->user_data.variation_glyph);
  }

  hb_position_t get_glyph_h_advance (hb_codepoint_t glyph)
  {
    return klass->get.f.glyph_h_advance (this, user_data,
					 glyph,
					 klass->user_data.glyph_h_advance);
  }

  hb_position_t get_glyph_v_advance (hb_codepoint_t glyph)
  {
    return klass->get.f.glyph_v_advance (this, user_data,
					 glyph,
					 klass->user_data.glyph_v_advance);
  }

  void get_glyph_h_advances (unsigned int count,
			     const hb_codepoint_t *first_glyph,
			     unsigned int glyph_stride,
			     hb_position_t *first_advance,
			     unsigned int advance_stride)
  {
    return klass->get.f.glyph_h_advances (this, user_data,
					  count,
					  first_glyph, glyph_stride,
					  first_advance, advance_stride,
					  klass->user_data.glyph_h_advances);
  }

  void get_glyph_v_advances (unsigned int count,
			     const hb_codepoint_t *first_glyph,
			     unsigned int glyph_stride,
			     hb_position_t *first_advance,
			     unsigned int advance_stride)
  {
    return klass->get.f.glyph_v_advances (this, user_data,
					  count,
					  first_glyph, glyph_stride,
					  first_advance, advance_stride,
					  klass->user_data.glyph_v_advances);
  }

  hb_bool_t get_glyph_h_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y)
  {
    *x = *y = 0;
    return klass->get.f.glyph_h_origin (this, user_data,
					glyph, x, y,
					klass->user_data.glyph_h_origin);
  }

  hb_bool_t get_glyph_v_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y)
  {
    *x = *y = 0;
    return klass->get.f.glyph_v_origin (this, user_data,
					glyph, x, y,
					klass->user_data.glyph_v_origin);
  }

  hb_position_t get_glyph_h_kerning (hb_codepoint_t left_glyph,
				     hb_codepoint_t right_glyph)
  {
#ifdef HB_DISABLE_DEPRECATED
    return 0;
#else
    return klass->get.f.glyph_h_kerning (this, user_data,
					 left_glyph, right_glyph,
					 klass->user_data.glyph_h_kerning);
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
					 klass->user_data.glyph_v_kerning);
#endif
  }

  hb_bool_t get_glyph_extents (hb_codepoint_t glyph,
			       hb_glyph_extents_t *extents)
  {
    memset (extents, 0, sizeof (*extents));
    return klass->get.f.glyph_extents (this, user_data,
				       glyph,
				       extents,
				       klass->user_data.glyph_extents);
  }

  hb_bool_t get_glyph_contour_point (hb_codepoint_t glyph, unsigned int point_index,
				     hb_position_t *x, hb_position_t *y)
  {
    *x = *y = 0;
    return klass->get.f.glyph_contour_point (this, user_data,
					     glyph, point_index,
					     x, y,
					     klass->user_data.glyph_contour_point);
  }

  hb_bool_t get_glyph_name (hb_codepoint_t glyph,
			    char *name, unsigned int size)
  {
    if (size) *name = '\0';
    return klass->get.f.glyph_name (this, user_data,
				    glyph,
				    name, size,
				    klass->user_data.glyph_name);
  }

  hb_bool_t get_glyph_from_name (const char *name, int len, /* -1 means nul-terminated */
				 hb_codepoint_t *glyph)
  {
    *glyph = 0;
    if (len == -1) len = strlen (name);
    return klass->get.f.glyph_from_name (this, user_data,
					 name, len,
					 glyph,
					 klass->user_data.glyph_from_name);
  }


  /* A bit higher-level, and with fallback */

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

  void guess_v_origin_minus_h_origin (hb_codepoint_t glyph,
				      hb_position_t *x, hb_position_t *y)
  {
    *x = get_glyph_h_advance (glyph) / 2;

    /* TODO cache this somehow?! */
    hb_font_extents_t extents;
    get_h_extents_with_fallback (&extents);
    *y = extents.ascender;
  }

  void get_glyph_h_origin_with_fallback (hb_codepoint_t glyph,
					 hb_position_t *x, hb_position_t *y)
  {
    if (!get_glyph_h_origin (glyph, x, y) &&
	 get_glyph_v_origin (glyph, x, y))
    {
      hb_position_t dx, dy;
      guess_v_origin_minus_h_origin (glyph, &dx, &dy);
      *x -= dx; *y -= dy;
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
      *x += dx; *y += dy;
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

  void add_glyph_h_origin (hb_codepoint_t glyph,
			   hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_h_origin_with_fallback (glyph, &origin_x, &origin_y);

    *x += origin_x;
    *y += origin_y;
  }
  void add_glyph_v_origin (hb_codepoint_t glyph,
			   hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_v_origin_with_fallback (glyph, &origin_x, &origin_y);

    *x += origin_x;
    *y += origin_y;
  }
  void add_glyph_origin_for_direction (hb_codepoint_t glyph,
				       hb_direction_t direction,
				       hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_origin_for_direction (glyph, direction, &origin_x, &origin_y);

    *x += origin_x;
    *y += origin_y;
  }

  void subtract_glyph_h_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_h_origin_with_fallback (glyph, &origin_x, &origin_y);

    *x -= origin_x;
    *y -= origin_y;
  }
  void subtract_glyph_v_origin (hb_codepoint_t glyph,
				hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_v_origin_with_fallback (glyph, &origin_x, &origin_y);

    *x -= origin_x;
    *y -= origin_y;
  }
  void subtract_glyph_origin_for_direction (hb_codepoint_t glyph,
					    hb_direction_t direction,
					    hb_position_t *x, hb_position_t *y)
  {
    hb_position_t origin_x, origin_y;

    get_glyph_origin_for_direction (glyph, direction, &origin_x, &origin_y);

    *x -= origin_x;
    *y -= origin_y;
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

    if (size && snprintf (s, size, "gid%u", glyph) < 0)
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

  void mults_changed ()
  {
    signed upem = face->get_upem ();
    x_mult = ((int64_t) x_scale << 16) / upem;
    y_mult = ((int64_t) y_scale << 16) / upem;
  }

  hb_position_t em_mult (int16_t v, int64_t mult)
  {
    return (hb_position_t) ((v * mult) >> 16);
  }
  hb_position_t em_scalef (float v, int scale)
  { return (hb_position_t) roundf (v * scale / face->get_upem ()); }
  float em_fscale (int16_t v, int scale)
  { return (float) v * scale / face->get_upem (); }
};
DECLARE_NULL_INSTANCE (hb_font_t);


#endif /* HB_FONT_HH */
