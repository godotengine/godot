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

#include "hb-private.hh"

#include "hb-font-private.hh"


/*
 * hb_font_funcs_t
 */

static hb_bool_t
hb_font_get_font_h_extents_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_font_extents_t *metrics,
				void *user_data HB_UNUSED)
{
  memset (metrics, 0, sizeof (*metrics));
  return false;
}
static hb_bool_t
hb_font_get_font_h_extents_parent (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_font_extents_t *metrics,
				   void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_font_h_extents (metrics);
  if (ret) {
    metrics->ascender = font->parent_scale_y_distance (metrics->ascender);
    metrics->descender = font->parent_scale_y_distance (metrics->descender);
    metrics->line_gap = font->parent_scale_y_distance (metrics->line_gap);
  }
  return ret;
}

static hb_bool_t
hb_font_get_font_v_extents_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_font_extents_t *metrics,
				void *user_data HB_UNUSED)
{
  memset (metrics, 0, sizeof (*metrics));
  return false;
}
static hb_bool_t
hb_font_get_font_v_extents_parent (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_font_extents_t *metrics,
				   void *user_data HB_UNUSED)
{
  hb_bool_t ret = font->parent->get_font_v_extents (metrics);
  if (ret) {
    metrics->ascender = font->parent_scale_x_distance (metrics->ascender);
    metrics->descender = font->parent_scale_x_distance (metrics->descender);
    metrics->line_gap = font->parent_scale_x_distance (metrics->line_gap);
  }
  return ret;
}

static hb_bool_t
hb_font_get_nominal_glyph_nil (hb_font_t *font HB_UNUSED,
			       void *font_data HB_UNUSED,
			       hb_codepoint_t unicode,
			       hb_codepoint_t *glyph,
			       void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_nominal_glyph_parent (hb_font_t *font,
				  void *font_data HB_UNUSED,
				  hb_codepoint_t unicode,
				  hb_codepoint_t *glyph,
				  void *user_data HB_UNUSED)
{
  return font->parent->get_nominal_glyph (unicode, glyph);
}

static hb_bool_t
hb_font_get_variation_glyph_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t unicode,
				 hb_codepoint_t variation_selector,
				 hb_codepoint_t *glyph,
				 void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_variation_glyph_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t unicode,
				    hb_codepoint_t variation_selector,
				    hb_codepoint_t *glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent->get_variation_glyph (unicode, variation_selector, glyph);
}


static hb_position_t
hb_font_get_glyph_h_advance_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t glyph,
				 void *user_data HB_UNUSED)
{
  return font->x_scale;
}
static hb_position_t
hb_font_get_glyph_h_advance_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent_scale_x_distance (font->parent->get_glyph_h_advance (glyph));
}

static hb_position_t
hb_font_get_glyph_v_advance_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t glyph,
				 void *user_data HB_UNUSED)
{
  /* TODO use font_extents.ascender+descender */
  return font->y_scale;
}
static hb_position_t
hb_font_get_glyph_v_advance_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent_scale_y_distance (font->parent->get_glyph_v_advance (glyph));
}

static hb_bool_t
hb_font_get_glyph_h_origin_nil (hb_font_t *font HB_UNUSED,
				void *font_data HB_UNUSED,
				hb_codepoint_t glyph,
				hb_position_t *x,
				hb_position_t *y,
				void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return true;
}
static hb_bool_t
hb_font_get_glyph_h_origin_parent (hb_font_t *font,
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
				hb_codepoint_t glyph,
				hb_position_t *x,
				hb_position_t *y,
				void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_v_origin_parent (hb_font_t *font,
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
				 hb_codepoint_t left_glyph,
				 hb_codepoint_t right_glyph,
				 void *user_data HB_UNUSED)
{
  return 0;
}
static hb_position_t
hb_font_get_glyph_h_kerning_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t left_glyph,
				    hb_codepoint_t right_glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent_scale_x_distance (font->parent->get_glyph_h_kerning (left_glyph, right_glyph));
}

static hb_position_t
hb_font_get_glyph_v_kerning_nil (hb_font_t *font HB_UNUSED,
				 void *font_data HB_UNUSED,
				 hb_codepoint_t top_glyph,
				 hb_codepoint_t bottom_glyph,
				 void *user_data HB_UNUSED)
{
  return 0;
}
static hb_position_t
hb_font_get_glyph_v_kerning_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    hb_codepoint_t top_glyph,
				    hb_codepoint_t bottom_glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent_scale_y_distance (font->parent->get_glyph_v_kerning (top_glyph, bottom_glyph));
}

static hb_bool_t
hb_font_get_glyph_extents_nil (hb_font_t *font HB_UNUSED,
			       void *font_data HB_UNUSED,
			       hb_codepoint_t glyph,
			       hb_glyph_extents_t *extents,
			       void *user_data HB_UNUSED)
{
  memset (extents, 0, sizeof (*extents));
  return false;
}
static hb_bool_t
hb_font_get_glyph_extents_parent (hb_font_t *font,
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
				     hb_codepoint_t glyph,
				     unsigned int point_index,
				     hb_position_t *x,
				     hb_position_t *y,
				     void *user_data HB_UNUSED)
{
  *x = *y = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_contour_point_parent (hb_font_t *font,
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
			    hb_codepoint_t glyph,
			    char *name, unsigned int size,
			    void *user_data HB_UNUSED)
{
  if (size) *name = '\0';
  return false;
}
static hb_bool_t
hb_font_get_glyph_name_parent (hb_font_t *font,
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
				 const char *name, int len, /* -1 means nul-terminated */
				 hb_codepoint_t *glyph,
				 void *user_data HB_UNUSED)
{
  *glyph = 0;
  return false;
}
static hb_bool_t
hb_font_get_glyph_from_name_parent (hb_font_t *font,
				    void *font_data HB_UNUSED,
				    const char *name, int len, /* -1 means nul-terminated */
				    hb_codepoint_t *glyph,
				    void *user_data HB_UNUSED)
{
  return font->parent->get_glyph_from_name (name, len, glyph);
}

static const hb_font_funcs_t _hb_font_funcs_nil = {
  HB_OBJECT_HEADER_STATIC,

  true, /* immutable */

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
static const hb_font_funcs_t _hb_font_funcs_parent = {
  HB_OBJECT_HEADER_STATIC,

  true, /* immutable */

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
#define HB_FONT_FUNC_IMPLEMENT(name) hb_font_get_##name##_parent,
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
hb_font_funcs_create (void)
{
  hb_font_funcs_t *ffuncs;

  if (!(ffuncs = hb_object_create<hb_font_funcs_t> ()))
    return hb_font_funcs_get_empty ();

  ffuncs->get = _hb_font_funcs_parent.get;

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
hb_font_funcs_get_empty (void)
{
  return const_cast<hb_font_funcs_t *> (&_hb_font_funcs_parent);
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
  if (unlikely (hb_object_is_inert (ffuncs)))
    return;

  ffuncs->immutable = true;
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
  return ffuncs->immutable;
}


#define HB_FONT_FUNC_IMPLEMENT(name) \
                                                                         \
void                                                                     \
hb_font_funcs_set_##name##_func (hb_font_funcs_t             *ffuncs,    \
                                 hb_font_get_##name##_func_t  func,      \
                                 void                        *user_data, \
                                 hb_destroy_func_t            destroy)   \
{                                                                        \
  if (ffuncs->immutable) {                                               \
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
    ffuncs->get.f.name = hb_font_get_##name##_parent;                    \
    ffuncs->user_data.name = nullptr;                                       \
    ffuncs->destroy.name = nullptr;                                         \
  }                                                                      \
}

HB_FONT_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_FONT_FUNC_IMPLEMENT

bool
hb_font_t::has_func (unsigned int i)
{
  if (parent && parent != hb_font_get_empty () && parent->has_func (i))
    return true;
  return this->klass->get.array[i] != _hb_font_funcs_parent.get.array[i];
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
 **/
hb_position_t
hb_font_get_glyph_v_kerning (hb_font_t *font,
			     hb_codepoint_t top_glyph, hb_codepoint_t bottom_glyph)
{
  return font->get_glyph_v_kerning (top_glyph, bottom_glyph);
}

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
 * @extents:
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
  hb_font_t *font;

  if (unlikely (!face))
    face = hb_face_get_empty ();
  if (!(font = hb_object_create<hb_font_t> ()))
    return hb_font_get_empty ();

  hb_face_make_immutable (face);
  font->parent = hb_font_get_empty ();
  font->face = hb_face_reference (face);
  font->klass = hb_font_funcs_get_empty ();

  font->x_scale = font->y_scale = hb_face_get_upem (face);

  return font;
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

  hb_font_t *font = hb_font_create (parent->face);

  if (unlikely (hb_object_is_inert (font)))
    return font;

  font->parent = hb_font_reference (parent);

  font->x_scale = parent->x_scale;
  font->y_scale = parent->y_scale;
  font->x_ppem = parent->x_ppem;
  font->y_ppem = parent->y_ppem;
  font->ptem = parent->ptem;

  font->num_coords = parent->num_coords;
  if (!font->num_coords)
    font->coords = nullptr;
  else
  {
    unsigned int size = parent->num_coords * sizeof (parent->coords[0]);
    font->coords = (int *) malloc (size);
    if (unlikely (!font->coords))
      font->num_coords = 0;
    else
      memcpy (font->coords, parent->coords, size);
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
hb_font_get_empty (void)
{
  static const hb_font_t _hb_font_nil = {
    HB_OBJECT_HEADER_STATIC,

    true, /* immutable */

    nullptr, /* parent */
    const_cast<hb_face_t *> (&_hb_face_nil),

    1000, /* x_scale */
    1000, /* y_scale */

    0, /* x_ppem */
    0, /* y_ppem */
    0, /* ptem */

    0, /* num_coords */
    nullptr, /* coords */

    const_cast<hb_font_funcs_t *> (&_hb_font_funcs_nil), /* klass */
    nullptr, /* user_data */
    nullptr, /* destroy */

    {
#define HB_SHAPER_IMPLEMENT(shaper) HB_SHAPER_DATA_INVALID,
#include "hb-shaper-list.hh"
#undef HB_SHAPER_IMPLEMENT
    }
  };

  return const_cast<hb_font_t *> (&_hb_font_nil);
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

#define HB_SHAPER_IMPLEMENT(shaper) HB_SHAPER_DATA_DESTROY(shaper, font);
#include "hb-shaper-list.hh"
#undef HB_SHAPER_IMPLEMENT

  if (font->destroy)
    font->destroy (font->user_data);

  hb_font_destroy (font->parent);
  hb_face_destroy (font->face);
  hb_font_funcs_destroy (font->klass);

  free (font->coords);

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
  if (unlikely (hb_object_is_inert (font)))
    return;

  if (font->parent)
    hb_font_make_immutable (font->parent);

  font->immutable = true;
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
  return font->immutable;
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
  if (font->immutable)
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
  if (font->immutable)
    return;

  if (unlikely (!face))
    face = hb_face_get_empty ();

  hb_face_t *old = font->face;

  font->face = hb_face_reference (face);

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
  if (font->immutable) {
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
  if (font->immutable) {
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
  if (font->immutable)
    return;

  font->x_scale = x_scale;
  font->y_scale = y_scale;
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
  if (font->immutable)
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
 * @ptem: 
 *
 * Sets "point size" of the font.
 *
 * Since: 1.6.0
 **/
void
hb_font_set_ptem (hb_font_t *font, float ptem)
{
  if (font->immutable)
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

/*
 * Variations
 */

static void
_hb_font_adopt_var_coords_normalized (hb_font_t *font,
				      int *coords, /* 2.14 normalized */
				      unsigned int coords_length)
{
  free (font->coords);

  font->coords = coords;
  font->num_coords = coords_length;
}

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
  if (font->immutable)
    return;

  if (!variations_length)
  {
    hb_font_set_var_coords_normalized (font, nullptr, 0);
    return;
  }

  unsigned int coords_length = hb_ot_var_get_axis_count (font->face);

  int *normalized = coords_length ? (int *) calloc (coords_length, sizeof (int)) : nullptr;
  if (unlikely (coords_length && !normalized))
    return;

  hb_ot_var_normalize_variations (font->face,
				  variations, variations_length,
				  normalized, coords_length);
  _hb_font_adopt_var_coords_normalized (font, normalized, coords_length);
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
  if (font->immutable)
    return;

  int *normalized = coords_length ? (int *) calloc (coords_length, sizeof (int)) : nullptr;
  if (unlikely (coords_length && !normalized))
    return;

  hb_ot_var_normalize_coords (font->face, coords_length, coords, normalized);
  _hb_font_adopt_var_coords_normalized (font, normalized, coords_length);
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
  if (font->immutable)
    return;

  int *copy = coords_length ? (int *) calloc (coords_length, sizeof (coords[0])) : nullptr;
  if (unlikely (coords_length && !copy))
    return;

  if (coords_length)
    memcpy (copy, coords, coords_length * sizeof (coords[0]));

  _hb_font_adopt_var_coords_normalized (font, copy, coords_length);
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
 * @func: (closure user_data) (destroy destroy) (scope notified):
 * @user_data:
 * @destroy:
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

#endif /* HB_DISABLE_DEPRECATED */
