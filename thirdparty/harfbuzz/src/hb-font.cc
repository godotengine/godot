/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2012  Google, Inc.
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

#include "hb.hh"

#include "hb-font.hh"
#include "hb-machinery.hh"

#include "hb-ot.h"

#include "hb-ot-var-avar-table.hh"
#include "hb-ot-var-fvar-table.hh"


/**
 * SECTION:hb-font
 * @title: hb-font
 * @short_description: Font objects
 * @include: hb.h
 *
 * Font objects represent a font face at a certain size and other
 * parameters (pixels per EM, points per EM, variation settings.)
 * Fonts are created from font faces, and are used as input to
 * hb_shape() among other things.
 **/


/*
 * hb_font_funcs_t
 */

static hb_bool_t
hb_font_get_font_h_extents_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_font_extents_t *extents,
				void *user_data HB_UNUSED)
{
  memset (extents, 0, sizeof (*extents));
  return false;
}
static hb_bool_t
hb_font_get_font_h_extents_default (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_font_extents_t *extents,
				    void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_font_h_extents (extents);
  if (ret) {
    extents->ascender = font->parent_scale_y_distance (extents->ascender);
    extents->descender = font->parent_scale_y_distance (extents->descender);
    extents->line_gap = font->parent_scale_y_distance (extents->line_gap);
  }
  return ret;
}

static hb_bool_t
hb_font_get_font_v_extents_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_font_extents_t *extents,
				void *user_data HB_UNUSED)
{
  memset (extents, 0, sizeof (*extents));
  return false;
}
static hb_bool_t
hb_font_get_font_v_extents_default (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_font_extents_t *extents,
				    void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_font_v_extents (extents);
  if (ret) {
    extents->ascender = font->parent_scale_x_distance (extents->ascender);
    extents->descender = font->parent_scale_x_distance (extents->descender);
    extents->line_gap = font->parent_scale_x_distance (extents->line_gap);
  }
  return ret;
}

static hb_bool_t
hb_font_get_nominal_glyph_nil (hb_font_t *font HB_UNUSED,
			       void *font_data HB_UNUSED,
			       hb_codepoint_t unicode HB_UNUSED,
			       hb_codepoint_t *glyph,
			       void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_nominal_glyph_default (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_codepoint_t unicode,
				   hb_codepoint_t *glyph,
				   void *user_data HB_UNUSED)
{
  if (font->has_nominal_glyphs_func_set ())
  {
    return font->get_nominal_glyphs (1, &unicode, 0, glyph, 0);
  }
  return font->parent->get_nominal_glyph (unicode, glyph);
}

#define hb_font_get_nominal_glyphs_nil hb_font_get_nominal_glyphs_default
static unsigned int
hb_font_get_nominal_glyphs_default (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    unsigned int count,
				    const hb_codepoint_t *first_unicode,
				    unsigned int unicode_stride,
				    hb_codepoint_t *first_glyph,
				    unsigned int glyph_stride,
				    void *user_data HB_UNUSED)
{
  if (font->has_nominal_glyph_func_set ())
  {
    for (unsigned int i = 0; i < count; i++)
    {
      if (!font->get_nominal_glyph (*first_unicode, first_glyph))
	return i;

      first_unicode = &StructAtOffsetUnaligned<hb_codepoint_t> (first_unicode, unicode_stride);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
    }
    return count;
  }

  return font->parent->get_nominal_glyphs (count,
					   first_unicode, unicode_stride,
					   first_glyph, glyph_stride);
}

static hb_bool_t
hb_font_get_variation_glyph_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t unicode HB_UNUSED,
				 hb_codepoint_t variation_selector HB_UNUSED,
				 hb_codepoint_t *glyph,
				 void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_variation_glyph_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t unicode,
				     hb_codepoint_t variation_selector,
				     hb_codepoint_t *glyph,
				     void *user_data HB_UNUSED)
{
  return font->parent->get_variation_glyph (unicode, variation_selector, glyph);
}


static hb_position_t
hb_font_get_glyph_h_advance_nil (hb_font_t *font,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t glyph HB_UNUSED,
				 void *user_data HB_UNUSED)
{
  return font->x_scale;
}
static hb_position_t
hb_font_get_glyph_h_advance_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t glyph,
				     void *user_data HB_UNUSED)
{
  if (font->has_glyph_h_advances_func_set ())
  {
    hb_position_t ret;
    font->get_glyph_h_advances (1, &glyph, 0, &ret, 0);
    return ret;
  }
  return font->parent_scale_x_distance (font->parent->get_glyph_h_advance (glyph));
}

static hb_position_t
hb_font_get_glyph_v_advance_nil (hb_font_t *font,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t glyph HB_UNUSED,
				 void *user_data HB_UNUSED)
{
  /* TODO use font_extents.ascender+descender */
  return font->y_scale;
}
static hb_position_t
hb_font_get_glyph_v_advance_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t glyph,
				     void *user_data HB_UNUSED)
{
  if (font->has_glyph_v_advances_func_set ())
  {
    hb_position_t ret;
    font->get_glyph_v_advances (1, &glyph, 0, &ret, 0);
    return ret;
  }
  return font->parent_scale_y_distance (font->parent->get_glyph_v_advance (glyph));
}

#define hb_font_get_glyph_h_advances_nil hb_font_get_glyph_h_advances_default
static void
hb_font_get_glyph_h_advances_default (hb_font_t* font,
				      void* font_data HB_UNUSED,
				      unsigned int count,
				      const hb_codepoint_t *first_glyph,
				      unsigned int glyph_stride,
				      hb_position_t *first_advance,
				      unsigned int advance_stride,
				      void *user_data HB_UNUSED)
{
  if (font->has_glyph_h_advance_func_set ())
  {
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->get_glyph_h_advance (*first_glyph);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

  font->parent->get_glyph_h_advances (count,
				      first_glyph, glyph_stride,
				      first_advance, advance_stride);
  for (unsigned int i = 0; i < count; i++)
  {
    *first_advance = font->parent_scale_x_distance (*first_advance);
    first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
  }
}

#define hb_font_get_glyph_v_advances_nil hb_font_get_glyph_v_advances_default
static void
hb_font_get_glyph_v_advances_default (hb_font_t* font,
				      void* font_data HB_UNUSED,
				      unsigned int count,
				      const hb_codepoint_t *first_glyph,
				      unsigned int glyph_stride,
				      hb_position_t *first_advance,
				      unsigned int advance_stride,
				      void *user_data HB_UNUSED)
{
  if (font->has_glyph_v_advance_func_set ())
  {
    for (unsigned int i = 0; i < count; i++)
    {
      *first_advance = font->get_glyph_v_advance (*first_glyph);
      first_glyph = &StructAtOffsetUnaligned<hb_codepoint_t> (first_glyph, glyph_stride);
      first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
    }
    return;
  }

  font->parent->get_glyph_v_advances (count,
				      first_glyph, glyph_stride,
				      first_advance, advance_stride);
  for (unsigned int i = 0; i < count; i++)
  {
    *first_advance = font->parent_scale_y_distance (*first_advance);
    first_advance = &StructAtOffsetUnaligned<hb_position_t> (first_advance, advance_stride);
  }
}

static hb_bool_t
hb_font_get_glyph_h_origin_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_codepoint_t glyph HB_UNUSED,
				hb_position_t *x,
				hb_position_t *y,
				void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return true;
}
static hb_bool_t
hb_font_get_glyph_h_origin_default (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t glyph,
				    hb_position_t *x,
				    hb_position_t *y,
				    void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_glyph_h_origin (glyph, x, y);
  if (ret)
    font->parent_scale_position (x, y);
  return ret;
}

static hb_bool_t
hb_font_get_glyph_v_origin_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_codepoint_t glyph HB_UNUSED,
				hb_position_t *x,
				hb_position_t *y,
				void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_v_origin_default (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t glyph,
				    hb_position_t *x,
				    hb_position_t *y,
				    void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_glyph_v_origin (glyph, x, y);
  if (ret)
    font->parent_scale_position (x, y);
  return ret;
}

static hb_position_t
hb_font_get_glyph_h_kerning_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t left_glyph HB_UNUSED,
				 hb_codepoint_t right_glyph HB_UNUSED,
				 void *user_data HB_UNUSED)
{
  return 0;
}
static hb_position_t
hb_font_get_glyph_h_kerning_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t left_glyph,
				     hb_codepoint_t right_glyph,
				     void *user_data HB_UNUSED)
{
  return font->parent_scale_x_distance (font->parent->get_glyph_h_kerning (left_glyph, right_glyph));
}

#ifndef HB_DISABLE_DEPRECATED
static hb_position_t
hb_font_get_glyph_v_kerning_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t top_glyph HB_UNUSED,
				 hb_codepoint_t bottom_glyph HB_UNUSED,
				 void *user_data HB_UNUSED)
{
  return 0;
}
static hb_position_t
hb_font_get_glyph_v_kerning_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t top_glyph,
				     hb_codepoint_t bottom_glyph,
				     void *user_data HB_UNUSED)
{
  return font->parent_scale_y_distance (font->parent->get_glyph_v_kerning (top_glyph, bottom_glyph));
}
#endif

static hb_bool_t
hb_font_get_glyph_extents_nil (hb_font_t *font HB_UNUSED,
			       void *font_data HB_UNUSED,
			       hb_codepoint_t glyph HB_UNUSED,
			       hb_glyph_extents_t *extents,
			       void *user_data HB_UNUSED)
{
  memset (extents, 0, sizeof (*extents));
  return false;
}
static hb_bool_t
hb_font_get_glyph_extents_default (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_codepoint_t glyph,
				   hb_glyph_extents_t *extents,
				   void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_glyph_extents (glyph, extents);
  if (ret) {
    font->parent_scale_position (&extents->x_bearing, &extents->y_bearing);
    font->parent_scale_distance (&extents->width, &extents->height);
  }
  return ret;
}

static hb_bool_t
hb_font_get_glyph_contour_point_nil (hb_font_t *font HB_UNUSED,
				     void *font_data HB_UNUSED,
				     hb_codepoint_t glyph HB_UNUSED,
				     unsigned int point_index HB_UNUSED,
				     hb_position_t *x,
				     hb_position_t *y,
				     void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_contour_point_default (hb_font_t *font,
					 void *font_data HB_UNUSED,
					 hb_codepoint_t glyph,
					 unsigned int point_index,
					 hb_position_t *x,
					 hb_position_t *y,
					 void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_glyph_contour_point (glyph, point_index, x, y);
  if (ret)
    font->parent_scale_position (x, y);
  return ret;
}

static hb_bool_t
hb_font_get_glyph_name_nil (hb_font_t *font HB_UNUSED,
			    void *font_data HB_UNUSED,
			    hb_codepoint_t glyph HB_UNUSED,
			    char *name, unsigned int size,
			    void *user_data HB_UNUSED)
{
  if (size) *name = '\0';
  return false;
}
static hb_bool_t
hb_font_get_glyph_name_default (hb_font_t *font,
				void *font_data HB_UNUSED,
				hb_codepoint_t glyph,
				char *name, unsigned int size,
				void *user_data HB_UNUSED)
{
  return font->parent->get_glyph_name (glyph, name, size);
}

static hb_bool_t
hb_font_get_glyph_from_name_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 const char *name HB_UNUSED,
				 int len HB_UNUSED, /* -1 means nul-terminated */
				 hb_codepoint_t *glyph,
				 void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_from_name_default (hb_font_t *font,
				     void *font_data HB_UNUSED,
				     const char *name, int len, /* -1 means nul-terminated */
				     hb_codepoint_t *glyph,
				     void *user_data HB_UNUSED)
{
  return font->parent->get_glyph_from_name (name, len, glyph);
}

DEFINE_NULL_INSTANCE (hb_font_funcs_t) =
{
  HB_OBJECT_HEADER_STATIC,

  {
#define HB_FONT_FUNC_IMPLEMENT(name) nullptr,
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  },
  {
#define HB_FONT_FUNC_IMPLEMENT(name) nullptr,
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  },
  {
    {
#define HB_FONT_FUNC_IMPLEMENT(name) hb_font_get_##name##_nil,
      HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
    }
  }
};

static const hb_font_funcs_t _hb_font_funcs_default = {
  HB_OBJECT_HEADER_STATIC,

  {
#define HB_FONT_FUNC_IMPLEMENT(name) nullptr,
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  },
  {
#define HB_FONT_FUNC_IMPLEMENT(name) nullptr,
    HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
  },
  {
    {
#define HB_FONT_FUNC_IMPLEMENT(name) hb_font_get_##name##_default,
      HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT
    }
  }
};


/**
 * hb_font_funcs_create: (Xconstructor)
 *
 *
 *
 * Return value: (transfer full):
 *
 * Since: 0.9.2
 **/
hb_font_funcs_t *
hb_font_funcs_create ()
{
  hb_font_funcs_t *ffuncs;

  if (!(ffuncs = hb_object_create<hb_font_funcs_t> ()))
    return hb_font_funcs_get_empty ();

  ffuncs->get = _hb_font_funcs_default.get;

  return ffuncs;
}

/**
 * hb_font_funcs_get_empty:
 *
 *
 *
 * Return value: (transfer full):
 *
 * Since: 0.9.2
 **/
hb_font_funcs_t *
hb_font_funcs_get_empty ()
{
  return const_cast<hb_font_funcs_t *> (&_hb_font_funcs_default);
}

/**
 * hb_font_funcs_reference: (skip)
 * @ffuncs: font functions.
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_font_funcs_t *
hb_font_funcs_reference (hb_font_funcs_t *ffuncs)
{
  return hb_object_reference (ffuncs);
}

/**
 * hb_font_funcs_destroy: (skip)
 * @ffuncs: font functions.
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_funcs_destroy (hb_font_funcs_t *ffuncs)
{
  if (!hb_object_destroy (ffuncs)) return;

#define HB_FONT_FUNC_IMPLEMENT(name) if (ffuncs->destroy.name) \
  ffuncs->destroy.name (ffuncs->user_data.name);
  HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT

  free (ffuncs);
}

/**
 * hb_font_funcs_set_user_data: (skip)
 * @ffuncs: font functions.
 * @key:
 * @data:
 * @destroy:
 * @replace:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_funcs_set_user_data (hb_font_funcs_t    *ffuncs,
			     hb_user_data_key_t *key,
			     void *              data,
			     hb_destroy_func_t   destroy,
			     hb_bool_t           replace)
{
  return hb_object_set_user_data (ffuncs, key, data, destroy, replace);
}

/**
 * hb_font_funcs_get_user_data: (skip)
 * @ffuncs: font functions.
 * @key:
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
void *
hb_font_funcs_get_user_data (hb_font_funcs_t    *ffuncs,
			     hb_user_data_key_t *key)
{
  return hb_object_get_user_data (ffuncs, key);
}


/**
 * hb_font_funcs_make_immutable:
 * @ffuncs: font functions.
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_funcs_make_immutable (hb_font_funcs_t *ffuncs)
{
  if (hb_object_is_immutable (ffuncs))
    return;

  hb_object_make_immutable (ffuncs);
}

/**
 * hb_font_funcs_is_immutable:
 * @ffuncs: font functions.
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_funcs_is_immutable (hb_font_funcs_t *ffuncs)
{
  return hb_object_is_immutable (ffuncs);
}


#define HB_FONT_FUNC_IMPLEMENT(name) \
									 \
void                                                                     \
hb_font_funcs_set_##name##_func (hb_font_funcs_t             *ffuncs,    \
				 hb_font_get_##name##_func_t  func,      \
				 void                        *user_data, \
				 hb_destroy_func_t            destroy)   \
{                                                                        \
  if (hb_object_is_immutable (ffuncs))                                   \
  {                                                                      \
    if (destroy)                                                         \
      destroy (user_data);                                               \
    return;                                                              \
  }                                                                      \
									 \
  if (ffuncs->destroy.name)                                              \
    ffuncs->destroy.name (ffuncs->user_data.name);                       \
									 \
  if (func) {                                                            \
    ffuncs->get.f.name = func;                                           \
    ffuncs->user_data.name = user_data;                                  \
    ffuncs->destroy.name = destroy;                                      \
  } else {                                                               \
    ffuncs->get.f.name = hb_font_get_##name##_default;                   \
    ffuncs->user_data.name = nullptr;                                    \
    ffuncs->destroy.name = nullptr;                                      \
  }                                                                      \
}

HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT

bool
hb_font_t::has_func_set (unsigned int i)
{
  return this->klass->get.array[i] != _hb_font_funcs_default.get.array[i];
}

bool
hb_font_t::has_func (unsigned int i)
{
  return has_func_set (i) ||
	 (parent && parent != &_hb_Null_hb_font_t && parent->has_func (i));
}

/* Public getters */

/**
 * hb_font_get_h_extents:
 * @font: a font.
 * @extents: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 1.1.3
 **/
hb_bool_t
hb_font_get_h_extents (hb_font_t *font,
		       hb_font_extents_t *extents)
{
  return font->get_font_h_extents (extents);
}

/**
 * hb_font_get_v_extents:
 * @font: a font.
 * @extents: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 1.1.3
 **/
hb_bool_t
hb_font_get_v_extents (hb_font_t *font,
		       hb_font_extents_t *extents)
{
  return font->get_font_v_extents (extents);
}

/**
 * hb_font_get_glyph:
 * @font: a font.
 * @unicode:
 * @variation_selector:
 * @glyph: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph (hb_font_t *font,
		   hb_codepoint_t unicode, hb_codepoint_t variation_selector,
		   hb_codepoint_t *glyph)
{
  if (unlikely (variation_selector))
    return font->get_variation_glyph (unicode, variation_selector, glyph);
  return font->get_nominal_glyph (unicode, glyph);
}

/**
 * hb_font_get_nominal_glyph:
 * @font: a font.
 * @unicode:
 * @glyph: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 1.2.3
 **/
hb_bool_t
hb_font_get_nominal_glyph (hb_font_t *font,
			   hb_codepoint_t unicode,
			   hb_codepoint_t *glyph)
{
  return font->get_nominal_glyph (unicode, glyph);
}

/**
 * hb_font_get_nominal_glyphs:
 * @font: a font.
 *
 *
 *
 * Return value:
 *
 * Since: 2.6.3
 **/
unsigned int
hb_font_get_nominal_glyphs (hb_font_t *font,
			    unsigned int count,
			    const hb_codepoint_t *first_unicode,
			    unsigned int unicode_stride,
			    hb_codepoint_t *first_glyph,
			    unsigned int glyph_stride)
{
  return font->get_nominal_glyphs (count,
				   first_unicode, unicode_stride,
				   first_glyph, glyph_stride);
}

/**
 * hb_font_get_variation_glyph:
 * @font: a font.
 * @unicode:
 * @variation_selector:
 * @glyph: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 1.2.3
 **/
hb_bool_t
hb_font_get_variation_glyph (hb_font_t *font,
			     hb_codepoint_t unicode, hb_codepoint_t variation_selector,
			     hb_codepoint_t *glyph)
{
  return font->get_variation_glyph (unicode, variation_selector, glyph);
}

/**
 * hb_font_get_glyph_h_advance:
 * @font: a font.
 * @glyph:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_position_t
hb_font_get_glyph_h_advance (hb_font_t *font,
			     hb_codepoint_t glyph)
{
  return font->get_glyph_h_advance (glyph);
}

/**
 * hb_font_get_glyph_v_advance:
 * @font: a font.
 * @glyph:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_position_t
hb_font_get_glyph_v_advance (hb_font_t *font,
			     hb_codepoint_t glyph)
{
  return font->get_glyph_v_advance (glyph);
}

/**
 * hb_font_get_glyph_h_advances:
 * @font: a font.
 *
 *
 *
 * Since: 1.8.6
 **/
void
hb_font_get_glyph_h_advances (hb_font_t* font,
			      unsigned int count,
			      const hb_codepoint_t *first_glyph,
			      unsigned glyph_stride,
			      hb_position_t *first_advance,
			      unsigned advance_stride)
{
  font->get_glyph_h_advances (count, first_glyph, glyph_stride, first_advance, advance_stride);
}
/**
 * hb_font_get_glyph_v_advances:
 * @font: a font.
 *
 *
 *
 * Since: 1.8.6
 **/
void
hb_font_get_glyph_v_advances (hb_font_t* font,
			      unsigned int count,
			      const hb_codepoint_t *first_glyph,
			      unsigned glyph_stride,
			      hb_position_t *first_advance,
			      unsigned advance_stride)
{
  font->get_glyph_v_advances (count, first_glyph, glyph_stride, first_advance, advance_stride);
}

/**
 * hb_font_get_glyph_h_origin:
 * @font: a font.
 * @glyph:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_h_origin (hb_font_t *font,
			    hb_codepoint_t glyph,
			    hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_h_origin (glyph, x, y);
}

/**
 * hb_font_get_glyph_v_origin:
 * @font: a font.
 * @glyph:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_v_origin (hb_font_t *font,
			    hb_codepoint_t glyph,
			    hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_v_origin (glyph, x, y);
}

/**
 * hb_font_get_glyph_h_kerning:
 * @font: a font.
 * @left_glyph:
 * @right_glyph:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_position_t
hb_font_get_glyph_h_kerning (hb_font_t *font,
			     hb_codepoint_t left_glyph, hb_codepoint_t right_glyph)
{
  return font->get_glyph_h_kerning (left_glyph, right_glyph);
}

#ifndef HB_DISABLE_DEPRECATED
/**
 * hb_font_get_glyph_v_kerning:
 * @font: a font.
 * @top_glyph:
 * @bottom_glyph:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 * Deprecated: 2.0.0
 **/
hb_position_t
hb_font_get_glyph_v_kerning (hb_font_t *font,
			     hb_codepoint_t top_glyph, hb_codepoint_t bottom_glyph)
{
  return font->get_glyph_v_kerning (top_glyph, bottom_glyph);
}
#endif

/**
 * hb_font_get_glyph_extents:
 * @font: a font.
 * @glyph:
 * @extents: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_extents (hb_font_t *font,
			   hb_codepoint_t glyph,
			   hb_glyph_extents_t *extents)
{
  return font->get_glyph_extents (glyph, extents);
}

/**
 * hb_font_get_glyph_contour_point:
 * @font: a font.
 * @glyph:
 * @point_index:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_contour_point (hb_font_t *font,
				 hb_codepoint_t glyph, unsigned int point_index,
				 hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_contour_point (glyph, point_index, x, y);
}

/**
 * hb_font_get_glyph_name:
 * @font: a font.
 * @glyph:
 * @name: (array length=size):
 * @size:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_name (hb_font_t *font,
			hb_codepoint_t glyph,
			char *name, unsigned int size)
{
  return font->get_glyph_name (glyph, name, size);
}

/**
 * hb_font_get_glyph_from_name:
 * @font: a font.
 * @name: (array length=len):
 * @len:
 * @glyph: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_from_name (hb_font_t *font,
			     const char *name, int len, /* -1 means nul-terminated */
			     hb_codepoint_t *glyph)
{
  return font->get_glyph_from_name (name, len, glyph);
}


/* A bit higher-level, and with fallback */

/**
 * hb_font_get_extents_for_direction:
 * @font: a font.
 * @direction:
 * @extents: (out):
 *
 *
 *
 * Since: 1.1.3
 **/
void
hb_font_get_extents_for_direction (hb_font_t *font,
				   hb_direction_t direction,
				   hb_font_extents_t *extents)
{
  return font->get_extents_for_direction (direction, extents);
}
/**
 * hb_font_get_glyph_advance_for_direction:
 * @font: a font.
 * @glyph:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_get_glyph_advance_for_direction (hb_font_t *font,
					 hb_codepoint_t glyph,
					 hb_direction_t direction,
					 hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_advance_for_direction (glyph, direction, x, y);
}
/**
 * hb_font_get_glyph_advances_for_direction:
 * @font: a font.
 * @direction:
 *
 *
 *
 * Since: 1.8.6
 **/
HB_EXTERN void
hb_font_get_glyph_advances_for_direction (hb_font_t* font,
					  hb_direction_t direction,
					  unsigned int count,
					  const hb_codepoint_t *first_glyph,
					  unsigned glyph_stride,
					  hb_position_t *first_advance,
					  unsigned advance_stride)
{
  font->get_glyph_advances_for_direction (direction, count, first_glyph, glyph_stride, first_advance, advance_stride);
}

/**
 * hb_font_get_glyph_origin_for_direction:
 * @font: a font.
 * @glyph:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_get_glyph_origin_for_direction (hb_font_t *font,
					hb_codepoint_t glyph,
					hb_direction_t direction,
					hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_origin_for_direction (glyph, direction, x, y);
}

/**
 * hb_font_add_glyph_origin_for_direction:
 * @font: a font.
 * @glyph:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_add_glyph_origin_for_direction (hb_font_t *font,
					hb_codepoint_t glyph,
					hb_direction_t direction,
					hb_position_t *x, hb_position_t *y)
{
  return font->add_glyph_origin_for_direction (glyph, direction, x, y);
}

/**
 * hb_font_subtract_glyph_origin_for_direction:
 * @font: a font.
 * @glyph:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_subtract_glyph_origin_for_direction (hb_font_t *font,
					     hb_codepoint_t glyph,
					     hb_direction_t direction,
					     hb_position_t *x, hb_position_t *y)
{
  return font->subtract_glyph_origin_for_direction (glyph, direction, x, y);
}

/**
 * hb_font_get_glyph_kerning_for_direction:
 * @font: a font.
 * @first_glyph:
 * @second_glyph:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_get_glyph_kerning_for_direction (hb_font_t *font,
					 hb_codepoint_t first_glyph, hb_codepoint_t second_glyph,
					 hb_direction_t direction,
					 hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_kerning_for_direction (first_glyph, second_glyph, direction, x, y);
}

/**
 * hb_font_get_glyph_extents_for_origin:
 * @font: a font.
 * @glyph:
 * @direction:
 * @extents: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_extents_for_origin (hb_font_t *font,
				      hb_codepoint_t glyph,
				      hb_direction_t direction,
				      hb_glyph_extents_t *extents)
{
  return font->get_glyph_extents_for_origin (glyph, direction, extents);
}

/**
 * hb_font_get_glyph_contour_point_for_origin:
 * @font: a font.
 * @glyph:
 * @point_index:
 * @direction:
 * @x: (out):
 * @y: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_get_glyph_contour_point_for_origin (hb_font_t *font,
					    hb_codepoint_t glyph, unsigned int point_index,
					    hb_direction_t direction,
					    hb_position_t *x, hb_position_t *y)
{
  return font->get_glyph_contour_point_for_origin (glyph, point_index, direction, x, y);
}

/* Generates gidDDD if glyph has no name. */
/**
 * hb_font_glyph_to_string:
 * @font: a font.
 * @glyph:
 * @s: (array length=size):
 * @size:
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_glyph_to_string (hb_font_t *font,
			 hb_codepoint_t glyph,
			 char *s, unsigned int size)
{
  font->glyph_to_string (glyph, s, size);
}

/* Parses gidDDD and uniUUUU strings automatically. */
/**
 * hb_font_glyph_from_string:
 * @font: a font.
 * @s: (array length=len) (element-type uint8_t):
 * @len:
 * @glyph: (out):
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_glyph_from_string (hb_font_t *font,
			   const char *s, int len, /* -1 means nul-terminated */
			   hb_codepoint_t *glyph)
{
  return font->glyph_from_string (s, len, glyph);
}


/*
 * hb_font_t
 */

DEFINE_NULL_INSTANCE (hb_font_t) =
{
  HB_OBJECT_HEADER_STATIC,

  nullptr, /* parent */
  const_cast<hb_face_t *> (&_hb_Null_hb_face_t),

  1000, /* x_scale */
  1000, /* y_scale */
  1<<16, /* x_mult */
  1<<16, /* y_mult */

  0, /* x_ppem */
  0, /* y_ppem */
  0, /* ptem */

  0, /* num_coords */
  nullptr, /* coords */
  nullptr, /* design_coords */

  const_cast<hb_font_funcs_t *> (&_hb_Null_hb_font_funcs_t),

  /* Zero for the rest is fine. */
};


static hb_font_t *
_hb_font_create (hb_face_t *face)
{
  hb_font_t *font;

  if (unlikely (!face))
    face = hb_face_get_empty ();
  if (!(font = hb_object_create<hb_font_t> ()))
    return hb_font_get_empty ();

  hb_face_make_immutable (face);
  font->parent = hb_font_get_empty ();
  font->face = hb_face_reference (face);
  font->klass = hb_font_funcs_get_empty ();
  font->data.init0 (font);
  font->x_scale = font->y_scale = hb_face_get_upem (face);
  font->x_mult = font->y_mult = 1 << 16;

  return font;
}

/**
 * hb_font_create: (Xconstructor)
 * @face: a face.
 *
 *
 *
 * Return value: (transfer full):
 *
 * Since: 0.9.2
 **/
hb_font_t *
hb_font_create (hb_face_t *face)
{
  hb_font_t *font = _hb_font_create (face);

#ifndef HB_NO_OT_FONT
  /* Install our in-house, very lightweight, funcs. */
  hb_ot_font_set_funcs (font);
#endif

  return font;
}

static void
_hb_font_adopt_var_coords (hb_font_t *font,
			   int *coords, /* 2.14 normalized */
			   float *design_coords,
			   unsigned int coords_length)
{
  free (font->coords);
  free (font->design_coords);

  font->coords = coords;
  font->design_coords = design_coords;
  font->num_coords = coords_length;
}

/**
 * hb_font_create_sub_font:
 * @parent: parent font.
 *
 *
 *
 * Return value: (transfer full):
 *
 * Since: 0.9.2
 **/
hb_font_t *
hb_font_create_sub_font (hb_font_t *parent)
{
  if (unlikely (!parent))
    parent = hb_font_get_empty ();

  hb_font_t *font = _hb_font_create (parent->face);

  if (unlikely (hb_object_is_immutable (font)))
    return font;

  font->parent = hb_font_reference (parent);

  font->x_scale = parent->x_scale;
  font->y_scale = parent->y_scale;
  font->mults_changed ();
  font->x_ppem = parent->x_ppem;
  font->y_ppem = parent->y_ppem;
  font->ptem = parent->ptem;

  unsigned int num_coords = parent->num_coords;
  if (num_coords)
  {
    int *coords = (int *) calloc (num_coords, sizeof (parent->coords[0]));
    float *design_coords = (float *) calloc (num_coords, sizeof (parent->design_coords[0]));
    if (likely (coords && design_coords))
    {
      memcpy (coords, parent->coords, num_coords * sizeof (parent->coords[0]));
      memcpy (design_coords, parent->design_coords, num_coords * sizeof (parent->design_coords[0]));
      _hb_font_adopt_var_coords (font, coords, design_coords, num_coords);
    }
    else
    {
      free (coords);
      free (design_coords);
    }
  }

  return font;
}

/**
 * hb_font_get_empty:
 *
 *
 *
 * Return value: (transfer full)
 *
 * Since: 0.9.2
 **/
hb_font_t *
hb_font_get_empty ()
{
  return const_cast<hb_font_t *> (&Null (hb_font_t));
}

/**
 * hb_font_reference: (skip)
 * @font: a font.
 *
 *
 *
 * Return value: (transfer full):
 *
 * Since: 0.9.2
 **/
hb_font_t *
hb_font_reference (hb_font_t *font)
{
  return hb_object_reference (font);
}

/**
 * hb_font_destroy: (skip)
 * @font: a font.
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_destroy (hb_font_t *font)
{
  if (!hb_object_destroy (font)) return;

  font->data.fini ();

  if (font->destroy)
    font->destroy (font->user_data);

  hb_font_destroy (font->parent);
  hb_face_destroy (font->face);
  hb_font_funcs_destroy (font->klass);

  free (font->coords);
  free (font->design_coords);

  free (font);
}

/**
 * hb_font_set_user_data: (skip)
 * @font: a font.
 * @key:
 * @data:
 * @destroy:
 * @replace:
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_set_user_data (hb_font_t          *font,
		       hb_user_data_key_t *key,
		       void *              data,
		       hb_destroy_func_t   destroy,
		       hb_bool_t           replace)
{
  return hb_object_set_user_data (font, key, data, destroy, replace);
}

/**
 * hb_font_get_user_data: (skip)
 * @font: a font.
 * @key:
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
void *
hb_font_get_user_data (hb_font_t          *font,
		       hb_user_data_key_t *key)
{
  return hb_object_get_user_data (font, key);
}

/**
 * hb_font_make_immutable:
 * @font: a font.
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_make_immutable (hb_font_t *font)
{
  if (hb_object_is_immutable (font))
    return;

  if (font->parent)
    hb_font_make_immutable (font->parent);

  hb_object_make_immutable (font);
}

/**
 * hb_font_is_immutable:
 * @font: a font.
 *
 *
 *
 * Return value:
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_font_is_immutable (hb_font_t *font)
{
  return hb_object_is_immutable (font);
}

/**
 * hb_font_set_parent:
 * @font: a font.
 * @parent: new parent.
 *
 * Sets parent font of @font.
 *
 * Since: 1.0.5
 **/
void
hb_font_set_parent (hb_font_t *font,
		    hb_font_t *parent)
{
  if (hb_object_is_immutable (font))
    return;

  if (!parent)
    parent = hb_font_get_empty ();

  hb_font_t *old = font->parent;

  font->parent = hb_font_reference (parent);

  hb_font_destroy (old);
}

/**
 * hb_font_get_parent:
 * @font: a font.
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
hb_font_t *
hb_font_get_parent (hb_font_t *font)
{
  return font->parent;
}

/**
 * hb_font_set_face:
 * @font: a font.
 * @face: new face.
 *
 * Sets font-face of @font.
 *
 * Since: 1.4.3
 **/
void
hb_font_set_face (hb_font_t *font,
		  hb_face_t *face)
{
  if (hb_object_is_immutable (font))
    return;

  if (unlikely (!face))
    face = hb_face_get_empty ();

  hb_face_t *old = font->face;

  hb_face_make_immutable (face);
  font->face = hb_face_reference (face);
  font->mults_changed ();

  hb_face_destroy (old);
}

/**
 * hb_font_get_face:
 * @font: a font.
 *
 *
 *
 * Return value: (transfer none):
 *
 * Since: 0.9.2
 **/
hb_face_t *
hb_font_get_face (hb_font_t *font)
{
  return font->face;
}


/**
 * hb_font_set_funcs:
 * @font: a font.
 * @klass: (closure font_data) (destroy destroy) (scope notified):
 * @font_data:
 * @destroy:
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_set_funcs (hb_font_t         *font,
		   hb_font_funcs_t   *klass,
		   void              *font_data,
		   hb_destroy_func_t  destroy)
{
  if (hb_object_is_immutable (font))
  {
    if (destroy)
      destroy (font_data);
    return;
  }

  if (font->destroy)
    font->destroy (font->user_data);

  if (!klass)
    klass = hb_font_funcs_get_empty ();

  hb_font_funcs_reference (klass);
  hb_font_funcs_destroy (font->klass);
  font->klass = klass;
  font->user_data = font_data;
  font->destroy = destroy;
}

/**
 * hb_font_set_funcs_data:
 * @font: a font.
 * @font_data: (destroy destroy) (scope notified):
 * @destroy:
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_set_funcs_data (hb_font_t         *font,
			void              *font_data,
			hb_destroy_func_t  destroy)
{
  /* Destroy user_data? */
  if (hb_object_is_immutable (font))
  {
    if (destroy)
      destroy (font_data);
    return;
  }

  if (font->destroy)
    font->destroy (font->user_data);

  font->user_data = font_data;
  font->destroy = destroy;
}


/**
 * hb_font_set_scale:
 * @font: a font.
 * @x_scale:
 * @y_scale:
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_set_scale (hb_font_t *font,
		   int x_scale,
		   int y_scale)
{
  if (hb_object_is_immutable (font))
    return;

  font->x_scale = x_scale;
  font->y_scale = y_scale;
  font->mults_changed ();
}

/**
 * hb_font_get_scale:
 * @font: a font.
 * @x_scale: (out):
 * @y_scale: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_get_scale (hb_font_t *font,
		   int *x_scale,
		   int *y_scale)
{
  if (x_scale) *x_scale = font->x_scale;
  if (y_scale) *y_scale = font->y_scale;
}

/**
 * hb_font_set_ppem:
 * @font: a font.
 * @x_ppem:
 * @y_ppem:
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_set_ppem (hb_font_t *font,
		  unsigned int x_ppem,
		  unsigned int y_ppem)
{
  if (hb_object_is_immutable (font))
    return;

  font->x_ppem = x_ppem;
  font->y_ppem = y_ppem;
}

/**
 * hb_font_get_ppem:
 * @font: a font.
 * @x_ppem: (out):
 * @y_ppem: (out):
 *
 *
 *
 * Since: 0.9.2
 **/
void
hb_font_get_ppem (hb_font_t *font,
		  unsigned int *x_ppem,
		  unsigned int *y_ppem)
{
  if (x_ppem) *x_ppem = font->x_ppem;
  if (y_ppem) *y_ppem = font->y_ppem;
}

/**
 * hb_font_set_ptem:
 * @font: a font.
 * @ptem: font size in points.
 *
 * Sets "point size" of the font.  Set to 0 to unset.
 *
 * There are 72 points in an inch.
 *
 * Since: 1.6.0
 **/
void
hb_font_set_ptem (hb_font_t *font, float ptem)
{
  if (hb_object_is_immutable (font))
    return;

  font->ptem = ptem;
}

/**
 * hb_font_get_ptem:
 * @font: a font.
 *
 * Gets the "point size" of the font.  A value of 0 means unset.
 *
 * Return value: Point size.
 *
 * Since: 0.9.2
 **/
float
hb_font_get_ptem (hb_font_t *font)
{
  return font->ptem;
}

#ifndef HB_NO_VAR
/*
 * Variations
 */

/**
 * hb_font_set_variations:
 *
 * Since: 1.4.2
 */
void
hb_font_set_variations (hb_font_t *font,
			const hb_variation_t *variations,
			unsigned int variations_length)
{
  if (hb_object_is_immutable (font))
    return;

  if (!variations_length)
  {
    hb_font_set_var_coords_normalized (font, nullptr, 0);
    return;
  }

  unsigned int coords_length = hb_ot_var_get_axis_count (font->face);

  int *normalized = coords_length ? (int *) calloc (coords_length, sizeof (int)) : nullptr;
  float *design_coords = coords_length ? (float *) calloc (coords_length, sizeof (float)) : nullptr;

  if (unlikely (coords_length && !(normalized && design_coords)))
  {
    free (normalized);
    free (design_coords);
    return;
  }

  const OT::fvar &fvar = *font->face->table.fvar;
  for (unsigned int i = 0; i < variations_length; i++)
  {
    hb_ot_var_axis_info_t info;
    if (hb_ot_var_find_axis_info (font->face, variations[i].tag, &info) &&
	info.axis_index < coords_length)
    {
      float v = variations[i].value;
      design_coords[info.axis_index] = v;
      normalized[info.axis_index] = fvar.normalize_axis_value (info.axis_index, v);
    }
  }
  font->face->table.avar->map_coords (normalized, coords_length);

  _hb_font_adopt_var_coords (font, normalized, design_coords, coords_length);
}

/**
 * hb_font_set_var_coords_design:
 *
 * Since: 1.4.2
 */
void
hb_font_set_var_coords_design (hb_font_t *font,
			       const float *coords,
			       unsigned int coords_length)
{
  if (hb_object_is_immutable (font))
    return;

  int *normalized = coords_length ? (int *) calloc (coords_length, sizeof (int)) : nullptr;
  float *design_coords = coords_length ? (float *) calloc (coords_length, sizeof (float)) : nullptr;

  if (unlikely (coords_length && !(normalized && design_coords)))
  {
    free (normalized);
    free (design_coords);
    return;
  }

  if (coords_length)
    memcpy (design_coords, coords, coords_length * sizeof (font->design_coords[0]));

  hb_ot_var_normalize_coords (font->face, coords_length, coords, normalized);
  _hb_font_adopt_var_coords (font, normalized, design_coords, coords_length);
}

/**
 * hb_font_set_var_named_instance:
 * @font: a font.
 * @instance_index: named instance index.
 *
 * Sets design coords of a font from a named instance index.
 *
 * Since: 2.6.0
 */
void
hb_font_set_var_named_instance (hb_font_t *font,
				unsigned instance_index)
{
  if (hb_object_is_immutable (font))
    return;

  unsigned int coords_length = hb_ot_var_named_instance_get_design_coords (font->face, instance_index, nullptr, nullptr);

  float *coords = coords_length ? (float *) calloc (coords_length, sizeof (float)) : nullptr;
  if (unlikely (coords_length && !coords))
    return;

  hb_ot_var_named_instance_get_design_coords (font->face, instance_index, &coords_length, coords);
  hb_font_set_var_coords_design (font, coords, coords_length);
  free (coords);
}

/**
 * hb_font_set_var_coords_normalized:
 *
 * Since: 1.4.2
 */
void
hb_font_set_var_coords_normalized (hb_font_t *font,
				   const int *coords, /* 2.14 normalized */
				   unsigned int coords_length)
{
  if (hb_object_is_immutable (font))
    return;

  int *copy = coords_length ? (int *) calloc (coords_length, sizeof (coords[0])) : nullptr;
  int *unmapped = coords_length ? (int *) calloc (coords_length, sizeof (coords[0])) : nullptr;
  float *design_coords = coords_length ? (float *) calloc (coords_length, sizeof (design_coords[0])) : nullptr;

  if (unlikely (coords_length && !(copy && unmapped && design_coords)))
  {
    free (copy);
    free (unmapped);
    free (design_coords);
    return;
  }

  if (coords_length)
  {
    memcpy (copy, coords, coords_length * sizeof (coords[0]));
    memcpy (unmapped, coords, coords_length * sizeof (coords[0]));
  }

  /* Best effort design coords simulation */
  font->face->table.avar->unmap_coords (unmapped, coords_length);
  for (unsigned int i = 0; i < coords_length; ++i)
    design_coords[i] = font->face->table.fvar->unnormalize_axis_value (i, unmapped[i]);
  free (unmapped);

  _hb_font_adopt_var_coords (font, copy, design_coords, coords_length);
}

/**
 * hb_font_get_var_coords_normalized:
 *
 * Return value is valid as long as variation coordinates of the font
 * are not modified.
 *
 * Since: 1.4.2
 */
const int *
hb_font_get_var_coords_normalized (hb_font_t *font,
				   unsigned int *length)
{
  if (length)
    *length = font->num_coords;

  return font->coords;
}

#ifdef HB_EXPERIMENTAL_API
/**
 * hb_font_get_var_coords_design:
 *
 * Return value is valid as long as variation coordinates of the font
 * are not modified.
 *
 * Since: EXPERIMENTAL
 */
const float *
hb_font_get_var_coords_design (hb_font_t *font,
			       unsigned int *length)
{
  if (length)
    *length = font->num_coords;

  return font->design_coords;
}
#endif
#endif

#ifndef HB_DISABLE_DEPRECATED
/*
 * Deprecated get_glyph_func():
 */

struct hb_trampoline_closure_t
{
  void *user_data;
  hb_destroy_func_t destroy;
  unsigned int ref_count;
};

template <typename FuncType>
struct hb_trampoline_t
{
  hb_trampoline_closure_t closure; /* Must be first. */
  FuncType func;
};

template <typename FuncType>
static hb_trampoline_t<FuncType> *
trampoline_create (FuncType           func,
		   void              *user_data,
		   hb_destroy_func_t  destroy)
{
  typedef hb_trampoline_t<FuncType> trampoline_t;

  trampoline_t *trampoline = (trampoline_t *) calloc (1, sizeof (trampoline_t));

  if (unlikely (!trampoline))
    return nullptr;

  trampoline->closure.user_data = user_data;
  trampoline->closure.destroy = destroy;
  trampoline->closure.ref_count = 1;
  trampoline->func = func;

  return trampoline;
}

static void
trampoline_reference (hb_trampoline_closure_t *closure)
{
  closure->ref_count++;
}

static void
trampoline_destroy (void *user_data)
{
  hb_trampoline_closure_t *closure = (hb_trampoline_closure_t *) user_data;

  if (--closure->ref_count)
    return;

  if (closure->destroy)
    closure->destroy (closure->user_data);
  free (closure);
}

typedef hb_trampoline_t<hb_font_get_glyph_func_t> hb_font_get_glyph_trampoline_t;

static hb_bool_t
hb_font_get_nominal_glyph_trampoline (hb_font_t *font,
				      void *font_data,
				      hb_codepoint_t unicode,
				      hb_codepoint_t *glyph,
				      void *user_data)
{
  hb_font_get_glyph_trampoline_t *trampoline = (hb_font_get_glyph_trampoline_t *) user_data;
  return trampoline->func (font, font_data, unicode, 0, glyph, trampoline->closure.user_data);
}

static hb_bool_t
hb_font_get_variation_glyph_trampoline (hb_font_t *font,
					void *font_data,
					hb_codepoint_t unicode,
					hb_codepoint_t variation_selector,
					hb_codepoint_t *glyph,
					void *user_data)
{
  hb_font_get_glyph_trampoline_t *trampoline = (hb_font_get_glyph_trampoline_t *) user_data;
  return trampoline->func (font, font_data, unicode, variation_selector, glyph, trampoline->closure.user_data);
}

/**
 * hb_font_funcs_set_glyph_func:
 * @ffuncs: font functions.
 * @func: (closure user_data) (destroy destroy) (scope notified): callback function.
 * @user_data: data to pass to @func.
 * @destroy: function to call when @user_data is not needed anymore.
 *
 * Deprecated.  Use hb_font_funcs_set_nominal_glyph_func() and
 * hb_font_funcs_set_variation_glyph_func() instead.
 *
 * Since: 0.9.2
 * Deprecated: 1.2.3
 **/
void
hb_font_funcs_set_glyph_func (hb_font_funcs_t *ffuncs,
			      hb_font_get_glyph_func_t func,
			      void *user_data, hb_destroy_func_t destroy)
{
  if (hb_object_is_immutable (ffuncs))
  {
    if (destroy)
      destroy (user_data);
    return;
  }

  hb_font_get_glyph_trampoline_t *trampoline;

  trampoline = trampoline_create (func, user_data, destroy);
  if (unlikely (!trampoline))
  {
    if (destroy)
      destroy (user_data);
    return;
  }

  hb_font_funcs_set_nominal_glyph_func (ffuncs,
					hb_font_get_nominal_glyph_trampoline,
					trampoline,
					trampoline_destroy);

  trampoline_reference (&trampoline->closure);
  hb_font_funcs_set_variation_glyph_func (ffuncs,
					  hb_font_get_variation_glyph_trampoline,
					  trampoline,
					  trampoline_destroy);
}
#endif
