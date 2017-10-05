/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2009  Keith Stribley
 * Copyright © 2015  Google, Inc.
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

#include "hb-ft.h"

#include "hb-font-private.hh"
#include "hb-machinery-private.hh"

#include FT_ADVANCES_H
#include FT_MULTIPLE_MASTERS_H
#include FT_TRUETYPE_TABLES_H


/* TODO:
 *
 * In general, this file does a fine job of what it's supposed to do.
 * There are, however, things that need more work:
 *
 *   - I remember seeing FT_Get_Advance() without the NO_HINTING flag to be buggy.
 *     Have not investigated.
 *
 *   - FreeType works in 26.6 mode.  Clients can decide to use that mode, and everything
 *     would work fine.  However, we also abuse this API for performing in font-space,
 *     but don't pass the correct flags to FreeType.  We just abuse the no-hinting mode
 *     for that, such that no rounding etc happens.  As such, we don't set ppem, and
 *     pass NO_HINTING as load_flags.  Would be much better to use NO_SCALE, and scale
 *     ourselves, like we do in uniscribe, etc.
 *
 *   - We don't handle / allow for emboldening / obliqueing.
 *
 *   - In the future, we should add constructors to create fonts in font space?
 *
 *   - FT_Load_Glyph() is extremely costly.  Do something about it?
 */


struct hb_ft_font_t
{
  FT_Face ft_face;
  int load_flags;
  bool symbol; /* Whether selected cmap is symbol cmap. */
  bool unref; /* Whether to destroy ft_face when done. */
};

static hb_ft_font_t *
_hb_ft_font_create (FT_Face ft_face, bool symbol, bool unref)
{
  hb_ft_font_t *ft_font = (hb_ft_font_t *) calloc (1, sizeof (hb_ft_font_t));

  if (unlikely (!ft_font))
    return nullptr;

  ft_font->ft_face = ft_face;
  ft_font->symbol = symbol;
  ft_font->unref = unref;

  ft_font->load_flags = FT_LOAD_DEFAULT | FT_LOAD_NO_HINTING;

  return ft_font;
}

static void
_hb_ft_face_destroy (void *data)
{
  FT_Done_Face ((FT_Face) data);
}

static void
_hb_ft_font_destroy (void *data)
{
  hb_ft_font_t *ft_font = (hb_ft_font_t *) data;

  if (ft_font->unref)
    _hb_ft_face_destroy (ft_font->ft_face);

  free (ft_font);
}

/**
 * hb_ft_font_set_load_flags:
 * @font:
 * @load_flags:
 *
 *
 *
 * Since: 1.0.5
 **/
void
hb_ft_font_set_load_flags (hb_font_t *font, int load_flags)
{
  if (font->immutable)
    return;

  if (font->destroy != (hb_destroy_func_t) _hb_ft_font_destroy)
    return;

  hb_ft_font_t *ft_font = (hb_ft_font_t *) font->user_data;

  ft_font->load_flags = load_flags;
}

/**
 * hb_ft_font_get_load_flags:
 * @font:
 *
 *
 *
 * Return value:
 * Since: 1.0.5
 **/
int
hb_ft_font_get_load_flags (hb_font_t *font)
{
  if (font->destroy != (hb_destroy_func_t) _hb_ft_font_destroy)
    return 0;

  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font->user_data;

  return ft_font->load_flags;
}

FT_Face
hb_ft_font_get_face (hb_font_t *font)
{
  if (font->destroy != (hb_destroy_func_t) _hb_ft_font_destroy)
    return nullptr;

  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font->user_data;

  return ft_font->ft_face;
}



static hb_bool_t
hb_ft_get_nominal_glyph (hb_font_t *font HB_UNUSED,
			 void *font_data,
			 hb_codepoint_t unicode,
			 hb_codepoint_t *glyph,
			 void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  unsigned int g = FT_Get_Char_Index (ft_font->ft_face, unicode);

  if (unlikely (!g))
  {
    if (unlikely (ft_font->symbol) && unicode <= 0x00FFu)
    {
      /* For symbol-encoded OpenType fonts, we duplicate the
       * U+F000..F0FF range at U+0000..U+00FF.  That's what
       * Windows seems to do, and that's hinted about at:
       * https://docs.microsoft.com/en-us/typography/opentype/spec/recom
       * under "Non-Standard (Symbol) Fonts". */
      g = FT_Get_Char_Index (ft_font->ft_face, 0xF000u + unicode);
      if (!g)
	return false;
    }
    else
      return false;
  }

  *glyph = g;
  return true;
}

static hb_bool_t
hb_ft_get_variation_glyph (hb_font_t *font HB_UNUSED,
			   void *font_data,
			   hb_codepoint_t unicode,
			   hb_codepoint_t variation_selector,
			   hb_codepoint_t *glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  unsigned int g = FT_Face_GetCharVariantIndex (ft_font->ft_face, unicode, variation_selector);

  if (unlikely (!g))
    return false;

  *glyph = g;
  return true;
}

static hb_position_t
hb_ft_get_glyph_h_advance (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Fixed v;

  if (unlikely (FT_Get_Advance (ft_font->ft_face, glyph, ft_font->load_flags, &v)))
    return 0;

  if (font->x_scale < 0)
    v = -v;

  return (v + (1<<9)) >> 10;
}

static hb_position_t
hb_ft_get_glyph_v_advance (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Fixed v;

  if (unlikely (FT_Get_Advance (ft_font->ft_face, glyph, ft_font->load_flags | FT_LOAD_VERTICAL_LAYOUT, &v)))
    return 0;

  if (font->y_scale < 0)
    v = -v;

  /* Note: FreeType's vertical metrics grows downward while other FreeType coordinates
   * have a Y growing upward.  Hence the extra negation. */
  return (-v + (1<<9)) >> 10;
}

static hb_bool_t
hb_ft_get_glyph_v_origin (hb_font_t *font,
			  void *font_data,
			  hb_codepoint_t glyph,
			  hb_position_t *x,
			  hb_position_t *y,
			  void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Face ft_face = ft_font->ft_face;

  if (unlikely (FT_Load_Glyph (ft_face, glyph, ft_font->load_flags)))
    return false;

  /* Note: FreeType's vertical metrics grows downward while other FreeType coordinates
   * have a Y growing upward.  Hence the extra negation. */
  *x = ft_face->glyph->metrics.horiBearingX -   ft_face->glyph->metrics.vertBearingX;
  *y = ft_face->glyph->metrics.horiBearingY - (-ft_face->glyph->metrics.vertBearingY);

  if (font->x_scale < 0)
    *x = -*x;
  if (font->y_scale < 0)
    *y = -*y;

  return true;
}

static hb_position_t
hb_ft_get_glyph_h_kerning (hb_font_t *font,
			   void *font_data,
			   hb_codepoint_t left_glyph,
			   hb_codepoint_t right_glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Vector kerningv;

  FT_Kerning_Mode mode = font->x_ppem ? FT_KERNING_DEFAULT : FT_KERNING_UNFITTED;
  if (FT_Get_Kerning (ft_font->ft_face, left_glyph, right_glyph, mode, &kerningv))
    return 0;

  return kerningv.x;
}

static hb_bool_t
hb_ft_get_glyph_extents (hb_font_t *font,
			 void *font_data,
			 hb_codepoint_t glyph,
			 hb_glyph_extents_t *extents,
			 void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Face ft_face = ft_font->ft_face;

  if (unlikely (FT_Load_Glyph (ft_face, glyph, ft_font->load_flags)))
    return false;

  extents->x_bearing = ft_face->glyph->metrics.horiBearingX;
  extents->y_bearing = ft_face->glyph->metrics.horiBearingY;
  extents->width = ft_face->glyph->metrics.width;
  extents->height = -ft_face->glyph->metrics.height;
  if (font->x_scale < 0)
  {
    extents->x_bearing = -extents->x_bearing;
    extents->width = -extents->width;
  }
  if (font->y_scale < 0)
  {
    extents->y_bearing = -extents->y_bearing;
    extents->height = -extents->height;
  }
  return true;
}

static hb_bool_t
hb_ft_get_glyph_contour_point (hb_font_t *font HB_UNUSED,
			       void *font_data,
			       hb_codepoint_t glyph,
			       unsigned int point_index,
			       hb_position_t *x,
			       hb_position_t *y,
			       void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Face ft_face = ft_font->ft_face;

  if (unlikely (FT_Load_Glyph (ft_face, glyph, ft_font->load_flags)))
      return false;

  if (unlikely (ft_face->glyph->format != FT_GLYPH_FORMAT_OUTLINE))
      return false;

  if (unlikely (point_index >= (unsigned int) ft_face->glyph->outline.n_points))
      return false;

  *x = ft_face->glyph->outline.points[point_index].x;
  *y = ft_face->glyph->outline.points[point_index].y;

  return true;
}

static hb_bool_t
hb_ft_get_glyph_name (hb_font_t *font HB_UNUSED,
		      void *font_data,
		      hb_codepoint_t glyph,
		      char *name, unsigned int size,
		      void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;

  hb_bool_t ret = !FT_Get_Glyph_Name (ft_font->ft_face, glyph, name, size);
  if (ret && (size && !*name))
    ret = false;

  return ret;
}

static hb_bool_t
hb_ft_get_glyph_from_name (hb_font_t *font HB_UNUSED,
			   void *font_data,
			   const char *name, int len, /* -1 means nul-terminated */
			   hb_codepoint_t *glyph,
			   void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Face ft_face = ft_font->ft_face;

  if (len < 0)
    *glyph = FT_Get_Name_Index (ft_face, (FT_String *) name);
  else {
    /* Make a nul-terminated version. */
    char buf[128];
    len = MIN (len, (int) sizeof (buf) - 1);
    strncpy (buf, name, len);
    buf[len] = '\0';
    *glyph = FT_Get_Name_Index (ft_face, buf);
  }

  if (*glyph == 0)
  {
    /* Check whether the given name was actually the name of glyph 0. */
    char buf[128];
    if (!FT_Get_Glyph_Name(ft_face, 0, buf, sizeof (buf)) &&
        len < 0 ? !strcmp (buf, name) : !strncmp (buf, name, len))
      return true;
  }

  return *glyph != 0;
}

static hb_bool_t
hb_ft_get_font_h_extents (hb_font_t *font HB_UNUSED,
			  void *font_data,
			  hb_font_extents_t *metrics,
			  void *user_data HB_UNUSED)
{
  const hb_ft_font_t *ft_font = (const hb_ft_font_t *) font_data;
  FT_Face ft_face = ft_font->ft_face;
  metrics->ascender = ft_face->size->metrics.ascender;
  metrics->descender = ft_face->size->metrics.descender;
  metrics->line_gap = ft_face->size->metrics.height - (ft_face->size->metrics.ascender - ft_face->size->metrics.descender);
  if (font->y_scale < 0)
  {
    metrics->ascender = -metrics->ascender;
    metrics->descender = -metrics->descender;
    metrics->line_gap = -metrics->line_gap;
  }
  return true;
}

static void free_static_ft_funcs (void);

static struct hb_ft_font_funcs_lazy_loader_t : hb_font_funcs_lazy_loader_t<hb_ft_font_funcs_lazy_loader_t>
{
  static inline hb_font_funcs_t *create (void)
  {
    hb_font_funcs_t *funcs = hb_font_funcs_create ();

    hb_font_funcs_set_font_h_extents_func (funcs, hb_ft_get_font_h_extents, nullptr, nullptr);
    //hb_font_funcs_set_font_v_extents_func (funcs, hb_ft_get_font_v_extents, nullptr, nullptr);
    hb_font_funcs_set_nominal_glyph_func (funcs, hb_ft_get_nominal_glyph, nullptr, nullptr);
    hb_font_funcs_set_variation_glyph_func (funcs, hb_ft_get_variation_glyph, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_advance_func (funcs, hb_ft_get_glyph_h_advance, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_advance_func (funcs, hb_ft_get_glyph_v_advance, nullptr, nullptr);
    //hb_font_funcs_set_glyph_h_origin_func (funcs, hb_ft_get_glyph_h_origin, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_origin_func (funcs, hb_ft_get_glyph_v_origin, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_kerning_func (funcs, hb_ft_get_glyph_h_kerning, nullptr, nullptr);
    //hb_font_funcs_set_glyph_v_kerning_func (funcs, hb_ft_get_glyph_v_kerning, nullptr, nullptr);
    hb_font_funcs_set_glyph_extents_func (funcs, hb_ft_get_glyph_extents, nullptr, nullptr);
    hb_font_funcs_set_glyph_contour_point_func (funcs, hb_ft_get_glyph_contour_point, nullptr, nullptr);
    hb_font_funcs_set_glyph_name_func (funcs, hb_ft_get_glyph_name, nullptr, nullptr);
    hb_font_funcs_set_glyph_from_name_func (funcs, hb_ft_get_glyph_from_name, nullptr, nullptr);

    hb_font_funcs_make_immutable (funcs);

#ifdef HB_USE_ATEXIT
    atexit (free_static_ft_funcs);
#endif

    return funcs;
  }
} static_ft_funcs;

#ifdef HB_USE_ATEXIT
static
void free_static_ft_funcs (void)
{
  static_ft_funcs.free_instance ();
}
#endif

static hb_font_funcs_t *
_hb_ft_get_font_funcs (void)
{
  return static_ft_funcs.get_unconst ();
}

static void
_hb_ft_font_set_funcs (hb_font_t *font, FT_Face ft_face, bool unref)
{
  bool symbol = ft_face->charmap && ft_face->charmap->encoding == FT_ENCODING_MS_SYMBOL;

  hb_font_set_funcs (font,
		     _hb_ft_get_font_funcs (),
		     _hb_ft_font_create (ft_face, symbol, unref),
		     _hb_ft_font_destroy);
}


static hb_blob_t *
reference_table  (hb_face_t *face HB_UNUSED, hb_tag_t tag, void *user_data)
{
  FT_Face ft_face = (FT_Face) user_data;
  FT_Byte *buffer;
  FT_ULong  length = 0;
  FT_Error error;

  /* Note: FreeType like HarfBuzz uses the NONE tag for fetching the entire blob */

  error = FT_Load_Sfnt_Table (ft_face, tag, 0, nullptr, &length);
  if (error)
    return nullptr;

  buffer = (FT_Byte *) malloc (length);
  if (!buffer)
    return nullptr;

  error = FT_Load_Sfnt_Table (ft_face, tag, 0, buffer, &length);
  if (error)
  {
    free (buffer);
    return nullptr;
  }

  return hb_blob_create ((const char *) buffer, length,
			 HB_MEMORY_MODE_WRITABLE,
			 buffer, free);
}

/**
 * hb_ft_face_create:
 * @ft_face: (destroy destroy) (scope notified):
 * @destroy:
 *
 *
 *
 * Return value: (transfer full):
 * Since: 0.9.2
 **/
hb_face_t *
hb_ft_face_create (FT_Face           ft_face,
		   hb_destroy_func_t destroy)
{
  hb_face_t *face;

  if (!ft_face->stream->read) {
    hb_blob_t *blob;

    blob = hb_blob_create ((const char *) ft_face->stream->base,
			   (unsigned int) ft_face->stream->size,
			   HB_MEMORY_MODE_READONLY,
			   ft_face, destroy);
    face = hb_face_create (blob, ft_face->face_index);
    hb_blob_destroy (blob);
  } else {
    face = hb_face_create_for_tables (reference_table, ft_face, destroy);
  }

  hb_face_set_index (face, ft_face->face_index);
  hb_face_set_upem (face, ft_face->units_per_EM);

  return face;
}

/**
 * hb_ft_face_create_referenced:
 * @ft_face:
 *
 *
 *
 * Return value: (transfer full):
 * Since: 0.9.38
 **/
hb_face_t *
hb_ft_face_create_referenced (FT_Face ft_face)
{
  FT_Reference_Face (ft_face);
  return hb_ft_face_create (ft_face, _hb_ft_face_destroy);
}

static void
hb_ft_face_finalize (FT_Face ft_face)
{
  hb_face_destroy ((hb_face_t *) ft_face->generic.data);
}

/**
 * hb_ft_face_create_cached:
 * @ft_face:
 *
 *
 *
 * Return value: (transfer full):
 * Since: 0.9.2
 **/
hb_face_t *
hb_ft_face_create_cached (FT_Face ft_face)
{
  if (unlikely (!ft_face->generic.data || ft_face->generic.finalizer != (FT_Generic_Finalizer) hb_ft_face_finalize))
  {
    if (ft_face->generic.finalizer)
      ft_face->generic.finalizer (ft_face);

    ft_face->generic.data = hb_ft_face_create (ft_face, nullptr);
    ft_face->generic.finalizer = (FT_Generic_Finalizer) hb_ft_face_finalize;
  }

  return hb_face_reference ((hb_face_t *) ft_face->generic.data);
}


/**
 * hb_ft_font_create:
 * @ft_face: (destroy destroy) (scope notified):
 * @destroy:
 *
 *
 *
 * Return value: (transfer full):
 * Since: 0.9.2
 **/
hb_font_t *
hb_ft_font_create (FT_Face           ft_face,
		   hb_destroy_func_t destroy)
{
  hb_font_t *font;
  hb_face_t *face;

  face = hb_ft_face_create (ft_face, destroy);
  font = hb_font_create (face);
  hb_face_destroy (face);
  _hb_ft_font_set_funcs (font, ft_face, false);
  hb_ft_font_changed (font);
  return font;
}

void
hb_ft_font_changed (hb_font_t *font)
{
  if (font->destroy != (hb_destroy_func_t) _hb_ft_font_destroy)
    return;

  hb_ft_font_t *ft_font = (hb_ft_font_t *) font->user_data;
  FT_Face ft_face = ft_font->ft_face;

  hb_font_set_scale (font,
		     (int) (((uint64_t) ft_face->size->metrics.x_scale * (uint64_t) ft_face->units_per_EM + (1u<<15)) >> 16),
		     (int) (((uint64_t) ft_face->size->metrics.y_scale * (uint64_t) ft_face->units_per_EM + (1u<<15)) >> 16));
#if 0 /* hb-ft works in no-hinting model */
  hb_font_set_ppem (font,
		    ft_face->size->metrics.x_ppem,
		    ft_face->size->metrics.y_ppem);
#endif

#ifdef HAVE_FT_GET_VAR_BLEND_COORDINATES
  FT_MM_Var *mm_var = nullptr;
  if (!FT_Get_MM_Var (ft_face, &mm_var))
  {
    FT_Fixed *ft_coords = (FT_Fixed *) calloc (mm_var->num_axis, sizeof (FT_Fixed));
    int *coords = (int *) calloc (mm_var->num_axis, sizeof (int));
    if (coords && ft_coords)
    {
      if (!FT_Get_Var_Blend_Coordinates (ft_face, mm_var->num_axis, ft_coords))
      {
	bool nonzero = false;

	for (unsigned int i = 0; i < mm_var->num_axis; ++i)
	 {
	  coords[i] = ft_coords[i] >>= 2;
	  nonzero = nonzero || coords[i];
	 }

	if (nonzero)
	  hb_font_set_var_coords_normalized (font, coords, mm_var->num_axis);
	else
	  hb_font_set_var_coords_normalized (font, nullptr, 0);
      }
    }
    free (coords);
    free (ft_coords);
#ifdef HAVE_FT_DONE_MM_VAR
    FT_Done_MM_Var (ft_face->glyph->library, mm_var);
#else
    free (mm_var);
#endif
  }
#endif
}

/**
 * hb_ft_font_create_referenced:
 * @ft_face:
 *
 *
 *
 * Return value: (transfer full):
 * Since: 0.9.38
 **/
hb_font_t *
hb_ft_font_create_referenced (FT_Face ft_face)
{
  FT_Reference_Face (ft_face);
  return hb_ft_font_create (ft_face, _hb_ft_face_destroy);
}


static void free_static_ft_library (void);

static struct hb_ft_library_lazy_loader_t : hb_lazy_loader_t<hb_remove_ptr_t<FT_Library>::value,
							     hb_ft_library_lazy_loader_t>
{
  static inline FT_Library create (void)
  {
    FT_Library l;
    if (FT_Init_FreeType (&l))
      return nullptr;

#ifdef HB_USE_ATEXIT
    atexit (free_static_ft_library);
#endif

    return l;
  }
  static inline void destroy (FT_Library l)
  {
    FT_Done_FreeType (l);
  }
  static inline FT_Library get_null (void)
  {
    return nullptr;
  }
} static_ft_library;

#ifdef HB_USE_ATEXIT
static
void free_static_ft_library (void)
{
  static_ft_library.free_instance ();
}
#endif

static FT_Library
get_ft_library (void)
{
  return static_ft_library.get_unconst ();
}

static void
_release_blob (FT_Face ft_face)
{
  hb_blob_destroy ((hb_blob_t *) ft_face->generic.data);
}

void
hb_ft_font_set_funcs (hb_font_t *font)
{
  hb_blob_t *blob = hb_face_reference_blob (font->face);
  unsigned int blob_length;
  const char *blob_data = hb_blob_get_data (blob, &blob_length);
  if (unlikely (!blob_length))
    DEBUG_MSG (FT, font, "Font face has empty blob");

  FT_Face ft_face = nullptr;
  FT_Error err = FT_New_Memory_Face (get_ft_library (),
				     (const FT_Byte *) blob_data,
				     blob_length,
				     hb_face_get_index (font->face),
				     &ft_face);

  if (unlikely (err)) {
    hb_blob_destroy (blob);
    DEBUG_MSG (FT, font, "Font face FT_New_Memory_Face() failed");
    return;
  }

  if (FT_Select_Charmap (ft_face, FT_ENCODING_UNICODE))
    FT_Select_Charmap (ft_face, FT_ENCODING_MS_SYMBOL);

  FT_Set_Char_Size (ft_face,
		    abs (font->x_scale), abs (font->y_scale),
		    0, 0);
#if 0
		    font->x_ppem * 72 * 64 / font->x_scale,
		    font->y_ppem * 72 * 64 / font->y_scale);
#endif
  if (font->x_scale < 0 || font->y_scale < 0)
  {
    FT_Matrix matrix = { font->x_scale < 0 ? -1 : +1, 0,
			  0, font->y_scale < 0 ? -1 : +1};
    FT_Set_Transform (ft_face, &matrix, nullptr);
  }

#ifdef HAVE_FT_SET_VAR_BLEND_COORDINATES
  unsigned int num_coords;
  const int *coords = hb_font_get_var_coords_normalized (font, &num_coords);
  if (num_coords)
  {
    FT_Fixed *ft_coords = (FT_Fixed *) calloc (num_coords, sizeof (FT_Fixed));
    if (ft_coords)
    {
      for (unsigned int i = 0; i < num_coords; i++)
	ft_coords[i] = coords[i] << 2;
      FT_Set_Var_Blend_Coordinates (ft_face, num_coords, ft_coords);
      free (ft_coords);
    }
  }
#endif

  ft_face->generic.data = blob;
  ft_face->generic.finalizer = (FT_Generic_Finalizer) _release_blob;

  _hb_ft_font_set_funcs (font, ft_face, true);
  hb_ft_font_set_load_flags (font, FT_LOAD_DEFAULT | FT_LOAD_NO_HINTING);
}
