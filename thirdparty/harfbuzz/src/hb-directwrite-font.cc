/*
 * Copyright © 2025  Google, Inc.
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
 * Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifdef HAVE_DIRECTWRITE

#include "hb-directwrite.h"

#include <d2d1.h>

#include "hb-draw.hh"
#include "hb-font.hh"
#include "hb-machinery.hh"

#define MAX_GLYPHS 256u

static unsigned int
hb_directwrite_get_nominal_glyphs (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   unsigned int count,
				   const hb_codepoint_t *first_unicode,
				   unsigned int unicode_stride,
				   hb_codepoint_t *first_glyph,
				   unsigned int glyph_stride,
				   void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  for (unsigned i = 0; i < count;)
  {
    UINT32 unicodes[MAX_GLYPHS];
    UINT16 gids[MAX_GLYPHS];

    unsigned n = hb_min (MAX_GLYPHS, count - i);

    for (unsigned j = 0; j < n; j++)
    {
      unicodes[j] = *first_unicode;
      first_unicode = &StructAtOffset<const hb_codepoint_t> (first_unicode, unicode_stride);
    }

    if (!SUCCEEDED (dw_face->GetGlyphIndices (unicodes, n, gids)))
      return i;

    for (unsigned j = 0; j < n; j++)
    {
      if (!gids[j])
        return i + j;
      *first_glyph = gids[j];
      first_glyph = &StructAtOffset<hb_codepoint_t> (first_glyph, glyph_stride);
    }

    i += n;
  }

  return count;
}

static hb_bool_t
hb_directwrite_get_font_h_extents (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_font_extents_t *metrics,
				   void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  DWRITE_FONT_METRICS dw_metrics;
  dw_face->GetMetrics (&dw_metrics);

  metrics->ascender = font->em_scale_y (dw_metrics.ascent);
  metrics->descender = -font->em_scale_y (dw_metrics.descent);
  metrics->line_gap = font->em_scale_y (dw_metrics.lineGap);

  return true;
}

static void
hb_directwrite_get_glyph_h_advances (hb_font_t* font,
				     void* font_data HB_UNUSED,
				     unsigned count,
				     const hb_codepoint_t *first_glyph,
				     unsigned glyph_stride,
				     hb_position_t *first_advance,
				     unsigned advance_stride,
				     void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  IDWriteFontFace1 *dw_face1 = nullptr;
  dw_face->QueryInterface (__uuidof(IDWriteFontFace1), (void**)&dw_face1);
  assert (dw_face1);

  unsigned int num_glyphs = font->face->get_num_glyphs ();

  for (unsigned i = 0; i < count;)
  {
    UINT16 gids[MAX_GLYPHS];
    INT32 advances[MAX_GLYPHS];

    unsigned n = hb_min (MAX_GLYPHS, count - i);

    for (unsigned j = 0; j < n; j++)
    {
      gids[j] = *first_glyph;
      advances[j] = 0;
      first_glyph = &StructAtOffset<const hb_codepoint_t> (first_glyph, glyph_stride);
    }
    dw_face1->GetDesignGlyphAdvances (n, gids, advances, false);
    for (unsigned j = 0; j < n; j++)
    {
      // https://github.com/harfbuzz/harfbuzz/issues/5319
      auto advance = gids[j] < num_glyphs ? advances[j] : 0;
      *first_advance = font->em_scale_x (advance);
      first_advance = &StructAtOffset<hb_position_t> (first_advance, advance_stride);
    }

    i += n;
  }
}

#ifndef HB_NO_VERTICAL

static void
hb_directwrite_get_glyph_v_advances (hb_font_t* font,
				     void* font_data HB_UNUSED,
				     unsigned count,
				     const hb_codepoint_t *first_glyph,
				     unsigned glyph_stride,
				     hb_position_t *first_advance,
				     unsigned advance_stride,
				     void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  IDWriteFontFace1 *dw_face1 = nullptr;
  dw_face->QueryInterface (__uuidof(IDWriteFontFace1), (void**)&dw_face1);
  assert (dw_face1);

  for (unsigned i = 0; i < count;)
  {
    UINT16 gids[MAX_GLYPHS];
    INT32 advances[MAX_GLYPHS];

    unsigned n = hb_min (MAX_GLYPHS, count - i);

    for (unsigned j = 0; j < n; j++)
    {
      gids[j] = *first_glyph;
      advances[j] = 0;
      first_glyph = &StructAtOffset<const hb_codepoint_t> (first_glyph, glyph_stride);
    }
    dw_face1->GetDesignGlyphAdvances (n, gids, advances, true);
    for (unsigned j = 0; j < n; j++)
    {
      *first_advance = -font->em_scale_y (advances[j]);
      first_advance = &StructAtOffset<hb_position_t> (first_advance, advance_stride);
    }

    i += n;
  }
}

static hb_bool_t
hb_directwrite_get_glyph_v_origin (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_codepoint_t glyph,
				   hb_position_t *x,
				   hb_position_t *y,
				   void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  UINT16 gid = glyph;
  DWRITE_GLYPH_METRICS metrics;

  if (FAILED (dw_face->GetDesignGlyphMetrics (&gid, 1, &metrics)))
    return false;

  *x = font->em_scale_x (metrics.advanceWidth / 2);
  *y = font->em_scale_y (metrics.verticalOriginY); // Untested

  return true;
}
#endif

static hb_bool_t
hb_directwrite_get_glyph_extents (hb_font_t *font,
				  void *font_data HB_UNUSED,
				  hb_codepoint_t glyph,
				  hb_glyph_extents_t *extents,
				  void *user_data HB_UNUSED)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  UINT16 gid = glyph;
  DWRITE_GLYPH_METRICS metrics;

  if (FAILED (dw_face->GetDesignGlyphMetrics (&gid, 1, &metrics)))
    return false;

  extents->x_bearing = font->em_scale_x (metrics.leftSideBearing);
  extents->y_bearing = font->em_scale_y (metrics.verticalOriginY - metrics.topSideBearing);
  extents->width = font->em_scale_x (metrics.advanceWidth - metrics.rightSideBearing) - extents->x_bearing;
  extents->height = font->em_scale_y (metrics.verticalOriginY - metrics.advanceHeight + metrics.bottomSideBearing) - extents->y_bearing; // Magic

  return true;
}


#ifndef HB_NO_DRAW

class GeometrySink : public IDWriteGeometrySink
{
  hb_font_t *font;
  hb_draw_session_t drawing;

public:
  GeometrySink(hb_font_t *font,
	       hb_draw_funcs_t *draw_funcs,
	       void *draw_data)
    : font (font), drawing ({draw_funcs, draw_data}) {}

  virtual ~GeometrySink() {}

  HRESULT STDMETHODCALLTYPE Close() override { return S_OK; }
  void STDMETHODCALLTYPE SetFillMode(D2D1_FILL_MODE) override {}
  void STDMETHODCALLTYPE SetSegmentFlags(D2D1_PATH_SEGMENT) override {}

  IFACEMETHOD(QueryInterface)(REFIID, void **) override { return E_NOINTERFACE; }
  IFACEMETHOD_(ULONG, AddRef)() override { return 1; }
  IFACEMETHOD_(ULONG, Release)() override { return 1; }

  void STDMETHODCALLTYPE BeginFigure(D2D1_POINT_2F startPoint, D2D1_FIGURE_BEGIN) override
  {
    drawing.move_to (font->em_scalef_x (startPoint.x), -font->em_scalef_y (startPoint.y));
  }

  void STDMETHODCALLTYPE AddBeziers(const D2D1_BEZIER_SEGMENT *beziers, UINT beziersCount) override
  {
    for (unsigned i = 0; i < beziersCount; ++i)
      drawing.cubic_to (font->em_scalef_x (beziers[i].point1.x), -font->em_scalef_y (beziers[i].point1.y),
			font->em_scalef_x (beziers[i].point2.x), -font->em_scalef_y (beziers[i].point2.y),
			font->em_scalef_x (beziers[i].point3.x), -font->em_scalef_y (beziers[i].point3.y));
  }

  void STDMETHODCALLTYPE AddLines(const D2D1_POINT_2F *points, UINT pointsCount) override
  {
    for (unsigned i = 0; i < pointsCount; ++i)
      drawing.line_to (font->em_scalef_x (points[i].x), -font->em_scalef_y (points[i].y));
  }

  void STDMETHODCALLTYPE EndFigure(D2D1_FIGURE_END) override
  {
    drawing.close_path ();
  }
};

static hb_bool_t
hb_directwrite_draw_glyph_or_fail (hb_font_t *font,
				   void *font_data HB_UNUSED,
				   hb_codepoint_t glyph,
				   hb_draw_funcs_t *draw_funcs, void *draw_data,
				   void *user_data)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;

  GeometrySink sink (font, draw_funcs, draw_data);
  UINT16 gid = static_cast<UINT16>(glyph);
  unsigned upem = font->face->get_upem();

  return S_OK == dw_face->GetGlyphRunOutline (upem,
					      &gid, nullptr, nullptr,
					      1,
					      false, false,
					      &sink);
}

#endif

static inline void free_static_directwrite_funcs ();

static struct hb_directwrite_font_funcs_lazy_loader_t : hb_font_funcs_lazy_loader_t<hb_directwrite_font_funcs_lazy_loader_t>
{
  static hb_font_funcs_t *create ()
  {
    hb_font_funcs_t *funcs = hb_font_funcs_create ();

    hb_font_funcs_set_nominal_glyphs_func (funcs, hb_directwrite_get_nominal_glyphs, nullptr, nullptr);
    //hb_font_funcs_set_variation_glyph_func (funcs, hb_directwrite_get_variation_glyph, nullptr, nullptr);

    hb_font_funcs_set_font_h_extents_func (funcs, hb_directwrite_get_font_h_extents, nullptr, nullptr);
    hb_font_funcs_set_glyph_h_advances_func (funcs, hb_directwrite_get_glyph_h_advances, nullptr, nullptr);

#ifndef HB_NO_VERTICAL
    hb_font_funcs_set_glyph_v_advances_func (funcs, hb_directwrite_get_glyph_v_advances, nullptr, nullptr);
    hb_font_funcs_set_glyph_v_origin_func (funcs, hb_directwrite_get_glyph_v_origin, nullptr, nullptr);
#endif

#ifndef HB_NO_DRAW
    hb_font_funcs_set_draw_glyph_or_fail_func (funcs, hb_directwrite_draw_glyph_or_fail, nullptr, nullptr);
#endif

    hb_font_funcs_set_glyph_extents_func (funcs, hb_directwrite_get_glyph_extents, nullptr, nullptr);

#ifndef HB_NO_OT_FONT_GLYPH_NAMES
    //hb_font_funcs_set_glyph_name_func (funcs, hb_directwrite_get_glyph_name, nullptr, nullptr);
    //hb_font_funcs_set_glyph_from_name_func (funcs, hb_directwrite_get_glyph_from_name, nullptr, nullptr);
#endif

    hb_font_funcs_make_immutable (funcs);

    hb_atexit (free_static_directwrite_funcs);

    return funcs;
  }
} static_directwrite_funcs;

static inline
void free_static_directwrite_funcs ()
{
  static_directwrite_funcs.free_instance ();
}

static hb_font_funcs_t *
_hb_directwrite_get_font_funcs ()
{
  return static_directwrite_funcs.get_unconst ();
}

/**
 * hb_directwrite_font_set_funcs:
 * @font: #hb_font_t to work upon
 *
 * Configures the font-functions structure of the specified
 * #hb_font_t font object to use DirectWrite font functions.
 *
 * In particular, you can use this function to configure an
 * existing #hb_face_t face object for use with DirectWrite font
 * functions even if that #hb_face_t face object was initially
 * created with hb_face_create(), and therefore was not
 * initially configured to use DirectWrite font functions.
 *
 * <note>Note: Internally, this function creates a DirectWrite font.
 * </note>
 *
 * Since: 11.0.0
 **/
void
hb_directwrite_font_set_funcs (hb_font_t *font)
{
  IDWriteFontFace *dw_face = (IDWriteFontFace *) (const void *) font->data.directwrite;
  if (unlikely (!dw_face))
  {
    hb_font_set_funcs (font,
		       hb_font_funcs_get_empty (),
		       nullptr, nullptr);
    return;
  }

  dw_face->AddRef ();
  hb_font_set_funcs (font,
		     _hb_directwrite_get_font_funcs (),
		     nullptr, nullptr);
}

#undef MAX_GLYPHS

#endif
