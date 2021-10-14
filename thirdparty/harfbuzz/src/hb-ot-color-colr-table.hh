/*
 * Copyright © 2018  Ebrahim Byagowi
 * Copyright © 2020  Google, Inc.
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
 * Google Author(s): Calder Kitagawa
 */

#ifndef HB_OT_COLOR_COLR_TABLE_HH
#define HB_OT_COLOR_COLR_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-common.hh"

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
#define HB_OT_TAG_COLR HB_TAG('C','O','L','R')

#ifndef COLRV1_MAX_NESTING_LEVEL
#define COLRV1_MAX_NESTING_LEVEL	100
#endif

#ifndef COLRV1_ENABLE_SUBSETTING
#define COLRV1_ENABLE_SUBSETTING 0
#endif

namespace OT {

struct COLR;
struct hb_colrv1_closure_context_t :
       hb_dispatch_context_t<hb_colrv1_closure_context_t>
{
  template <typename T>
  return_t dispatch (const T &obj)
  {
    if (unlikely (nesting_level_left == 0))
      return hb_empty_t ();

    if (paint_visited (&obj))
      return hb_empty_t ();

    nesting_level_left--;
    obj.closurev1 (this);
    nesting_level_left++;
    return hb_empty_t ();
  }
  static return_t default_return_value () { return hb_empty_t (); }

  bool paint_visited (const void *paint)
  {
    hb_codepoint_t delta = (hb_codepoint_t) ((uintptr_t) paint - (uintptr_t) base);
     if (visited_paint.has (delta))
      return true;

    visited_paint.add (delta);
    return false;
  }

  const COLR* get_colr_table () const
  { return reinterpret_cast<const COLR *> (base); }

  void add_glyph (unsigned glyph_id)
  { glyphs->add (glyph_id); }

  void add_layer_indices (unsigned first_layer_index, unsigned num_of_layers)
  { layer_indices->add_range (first_layer_index, first_layer_index + num_of_layers - 1); }

  void add_palette_index (unsigned palette_index)
  { palette_indices->add (palette_index); }

  public:
  const void *base;
  hb_set_t visited_paint;
  hb_set_t *glyphs;
  hb_set_t *layer_indices;
  hb_set_t *palette_indices;
  unsigned nesting_level_left;

  hb_colrv1_closure_context_t (const void *base_,
                               hb_set_t *glyphs_,
                               hb_set_t *layer_indices_,
                               hb_set_t *palette_indices_,
                               unsigned nesting_level_left_ = COLRV1_MAX_NESTING_LEVEL) :
                          base (base_),
                          glyphs (glyphs_),
                          layer_indices (layer_indices_),
                          palette_indices (palette_indices_),
                          nesting_level_left (nesting_level_left_)
  {}
};

struct LayerRecord
{
  operator hb_ot_color_layer_t () const { return {glyphId, colorIdx}; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBGlyphID	glyphId;	/* Glyph ID of layer glyph */
  Index		colorIdx;	/* Index value to use with a
				 * selected color palette.
				 * An index value of 0xFFFF
				 * is a special case indicating
				 * that the text foreground
				 * color (defined by a
				 * higher-level client) should
				 * be used and shall not be
				 * treated as actual index
				 * into CPAL ColorRecord array. */
  public:
  DEFINE_SIZE_STATIC (4);
};

struct BaseGlyphRecord
{
  int cmp (hb_codepoint_t g) const
  { return g < glyphId ? -1 : g > glyphId ? 1 : 0; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  public:
  HBGlyphID	glyphId;	/* Glyph ID of reference glyph */
  HBUINT16	firstLayerIdx;	/* Index (from beginning of
				 * the Layer Records) to the
				 * layer record. There will be
				 * numLayers consecutive entries
				 * for this base glyph. */
  HBUINT16	numLayers;	/* Number of color layers
				 * associated with this glyph */
  public:
  DEFINE_SIZE_STATIC (6);
};

template <typename T>
struct Variable
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  T      value;
  VarIdx varIdx;
  public:
  DEFINE_SIZE_STATIC (4 + T::static_size);
};

template <typename T>
struct NoVariable
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  T      value;
  public:
  DEFINE_SIZE_STATIC (T::static_size);
};

// Color structures

template <template<typename> class Var>
struct ColorIndex
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (*this);
    if (unlikely (!out)) return_trace (false);
    return_trace (c->serializer->check_assign (out->paletteIndex, c->plan->colr_palettes->get (paletteIndex),
                                               HB_SERIALIZE_ERROR_INT_OVERFLOW));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16	paletteIndex;
  Var<F2DOT14>	alpha;
  public:
  DEFINE_SIZE_STATIC (2 + Var<F2DOT14>::static_size);
};

template <template<typename> class Var>
struct ColorStop
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    if (unlikely (!c->serializer->embed (stopOffset))) return_trace (false);
    return_trace (color.subset (c));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  Var<F2DOT14>		stopOffset;
  ColorIndex<Var>		color;
  public:
  DEFINE_SIZE_STATIC (Var<F2DOT14>::static_size + ColorIndex<Var>::static_size);
};

struct Extend : HBUINT8
{
  enum {
    EXTEND_PAD     = 0,
    EXTEND_REPEAT  = 1,
    EXTEND_REFLECT = 2,
  };
  public:
  DEFINE_SIZE_STATIC (1);
};

template <template<typename> class Var>
struct ColorLine
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (this);
    if (unlikely (!out)) return_trace (false);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    if (!c->serializer->check_assign (out->extend, extend, HB_SERIALIZE_ERROR_INT_OVERFLOW)) return_trace (false);
    if (!c->serializer->check_assign (out->stops.len, stops.len, HB_SERIALIZE_ERROR_ARRAY_OVERFLOW)) return_trace (false);

    for (const auto& stop : stops.iter ())
    {
      if (!stop.subset (c)) return_trace (false);
    }
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  stops.sanitize (c));
  }

  Extend			extend;
  Array16Of<ColorStop<Var>>	stops;
  public:
  DEFINE_SIZE_ARRAY_SIZED (3, stops);
};

// Composition modes

// Compositing modes are taken from https://www.w3.org/TR/compositing-1/
// NOTE: a brief audit of major implementations suggests most support most
// or all of the specified modes.
struct CompositeMode : HBUINT8
{
  enum {
    // Porter-Duff modes
    // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators
    COMPOSITE_CLEAR          =  0,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_clear
    COMPOSITE_SRC            =  1,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_src
    COMPOSITE_DEST           =  2,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dst
    COMPOSITE_SRC_OVER       =  3,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcover
    COMPOSITE_DEST_OVER      =  4,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstover
    COMPOSITE_SRC_IN         =  5,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcin
    COMPOSITE_DEST_IN        =  6,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstin
    COMPOSITE_SRC_OUT        =  7,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcout
    COMPOSITE_DEST_OUT       =  8,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstout
    COMPOSITE_SRC_ATOP       =  9,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcatop
    COMPOSITE_DEST_ATOP      = 10,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstatop
    COMPOSITE_XOR            = 11,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_xor
    COMPOSITE_PLUS           = 12,  // https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_plus

    // Blend modes
    // https://www.w3.org/TR/compositing-1/#blending
    COMPOSITE_SCREEN         = 13,  // https://www.w3.org/TR/compositing-1/#blendingscreen
    COMPOSITE_OVERLAY        = 14,  // https://www.w3.org/TR/compositing-1/#blendingoverlay
    COMPOSITE_DARKEN         = 15,  // https://www.w3.org/TR/compositing-1/#blendingdarken
    COMPOSITE_LIGHTEN        = 16,  // https://www.w3.org/TR/compositing-1/#blendinglighten
    COMPOSITE_COLOR_DODGE    = 17,  // https://www.w3.org/TR/compositing-1/#blendingcolordodge
    COMPOSITE_COLOR_BURN     = 18,  // https://www.w3.org/TR/compositing-1/#blendingcolorburn
    COMPOSITE_HARD_LIGHT     = 19,  // https://www.w3.org/TR/compositing-1/#blendinghardlight
    COMPOSITE_SOFT_LIGHT     = 20,  // https://www.w3.org/TR/compositing-1/#blendingsoftlight
    COMPOSITE_DIFFERENCE     = 21,  // https://www.w3.org/TR/compositing-1/#blendingdifference
    COMPOSITE_EXCLUSION      = 22,  // https://www.w3.org/TR/compositing-1/#blendingexclusion
    COMPOSITE_MULTIPLY       = 23,  // https://www.w3.org/TR/compositing-1/#blendingmultiply

    // Modes that, uniquely, do not operate on components
    // https://www.w3.org/TR/compositing-1/#blendingnonseparable
    COMPOSITE_HSL_HUE        = 24,  // https://www.w3.org/TR/compositing-1/#blendinghue
    COMPOSITE_HSL_SATURATION = 25,  // https://www.w3.org/TR/compositing-1/#blendingsaturation
    COMPOSITE_HSL_COLOR      = 26,  // https://www.w3.org/TR/compositing-1/#blendingcolor
    COMPOSITE_HSL_LUMINOSITY = 27,  // https://www.w3.org/TR/compositing-1/#blendingluminosity
  };
  public:
  DEFINE_SIZE_STATIC (1);
};

template <template<typename> class Var>
struct Affine2x3
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  Var<HBFixed> xx;
  Var<HBFixed> yx;
  Var<HBFixed> xy;
  Var<HBFixed> yy;
  Var<HBFixed> dx;
  Var<HBFixed> dy;
  public:
  DEFINE_SIZE_STATIC (6 * Var<HBFixed>::static_size);
};

struct PaintColrLayers
{
  void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);
    return_trace (c->serializer->check_assign (out->firstLayerIndex, c->plan->colrv1_layers->get (firstLayerIndex),
                                               HB_SERIALIZE_ERROR_INT_OVERFLOW));

    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT8	format; /* format = 1 */
  HBUINT8	numLayers;
  HBUINT32	firstLayerIndex;  /* index into COLRv1::layersV1 */
  public:
  DEFINE_SIZE_STATIC (6);
};

template <template<typename> class Var>
struct PaintSolid
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { c->add_palette_index (color.paletteIndex); }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    if (unlikely (!c->serializer->embed (format))) return_trace (false);
    return_trace (color.subset (c));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT8		format; /* format = 2(noVar) or 3(Var)*/
  ColorIndex<Var>	color;
  public:
  DEFINE_SIZE_STATIC (1 + ColorIndex<Var>::static_size);
};

template <template<typename> class Var>
struct PaintLinearGradient
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  {
    for (const auto &stop : (this+colorLine).stops.iter ())
      c->add_palette_index (stop.color.paletteIndex);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->colorLine.serialize_subset (c, colorLine, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && colorLine.sanitize (c, this));
  }

  HBUINT8			format; /* format = 4(noVar) or 5 (Var) */
  Offset24To<ColorLine<Var>>	colorLine; /* Offset (from beginning of PaintLinearGradient
                                            * table) to ColorLine subtable. */
  Var<FWORD>			x0;
  Var<FWORD>			y0;
  Var<FWORD>			x1;
  Var<FWORD>			y1;
  Var<FWORD>			x2;
  Var<FWORD>			y2;
  public:
  DEFINE_SIZE_STATIC (4 + 6 * Var<FWORD>::static_size);
};

template <template<typename> class Var>
struct PaintRadialGradient
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->colorLine.serialize_subset (c, colorLine, this));
  }

  void closurev1 (hb_colrv1_closure_context_t* c) const
  {
    for (const auto &stop : (this+colorLine).stops.iter ())
      c->add_palette_index (stop.color.paletteIndex);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && colorLine.sanitize (c, this));
  }

  HBUINT8			format; /* format = 6(noVar) or 7 (Var) */
  Offset24To<ColorLine<Var>>	colorLine; /* Offset (from beginning of PaintRadialGradient
                                            * table) to ColorLine subtable. */
  Var<FWORD>			x0;
  Var<FWORD>			y0;
  Var<UFWORD>			radius0;
  Var<FWORD>			x1;
  Var<FWORD>			y1;
  Var<UFWORD>			radius1;
  public:
  DEFINE_SIZE_STATIC (4 + 6 * Var<FWORD>::static_size);
};

template <template<typename> class Var>
struct PaintSweepGradient
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->colorLine.serialize_subset (c, colorLine, this));
  }

  void closurev1 (hb_colrv1_closure_context_t* c) const
  {
    for (const auto &stop : (this+colorLine).stops.iter ())
      c->add_palette_index (stop.color.paletteIndex);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && colorLine.sanitize (c, this));
  }

  HBUINT8			format; /* format = 8(noVar) or 9 (Var) */
  Offset24To<ColorLine<Var>>	colorLine; /* Offset (from beginning of PaintSweepGradient
                                            * table) to ColorLine subtable. */
  Var<FWORD>			centerX;
  Var<FWORD>			centerY;
  Var<HBFixed>			startAngle;
  Var<HBFixed>			endAngle;
  public:
  DEFINE_SIZE_STATIC (2 * Var<FWORD>::static_size + 2 * Var<HBFixed>::static_size);
};

struct Paint;
// Paint a non-COLR glyph, filled as indicated by paint.
struct PaintGlyph
{
  void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    if (! c->serializer->check_assign (out->gid, c->plan->glyph_map->get (gid),
                                       HB_SERIALIZE_ERROR_INT_OVERFLOW))
      return_trace (false);

    return_trace (out->paint.serialize_subset (c, paint, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && paint.sanitize (c, this));
  }

  HBUINT8		format; /* format = 10 */
  Offset24To<Paint>	paint;  /* Offset (from beginning of PaintGlyph table) to Paint subtable. */
  HBUINT16		gid;
  public:
  DEFINE_SIZE_STATIC (6);
};

struct PaintColrGlyph
{
  void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (c->serializer->check_assign (out->gid, c->plan->glyph_map->get (gid),
                                               HB_SERIALIZE_ERROR_INT_OVERFLOW));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT8	format; /* format = 11 */
  HBUINT16	gid;
  public:
  DEFINE_SIZE_STATIC (3);
};

template <template<typename> class Var>
struct PaintTransform
{
  HB_INTERNAL void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->src.serialize_subset (c, src, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && src.sanitize (c, this));
  }

  HBUINT8		format; /* format = 12(noVar) or 13 (Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintTransform table) to Paint subtable. */
  Affine2x3<Var>	transform;
  public:
  DEFINE_SIZE_STATIC (4 + Affine2x3<Var>::static_size);
};

template <template<typename> class Var>
struct PaintTranslate
{
  HB_INTERNAL void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->src.serialize_subset (c, src, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && src.sanitize (c, this));
  }

  HBUINT8		format; /* format = 14(noVar) or 15 (Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintTranslate table) to Paint subtable. */
  Var<HBFixed>		dx;
  Var<HBFixed>		dy;
  public:
  DEFINE_SIZE_STATIC (4 + Var<HBFixed>::static_size);
};

template <template<typename> class Var>
struct PaintRotate
{
  HB_INTERNAL void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->src.serialize_subset (c, src, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && src.sanitize (c, this));
  }

  HBUINT8		format; /* format = 16 (noVar) or 17(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintRotate table) to Paint subtable. */
  Var<HBFixed>		angle;
  Var<HBFixed>		centerX;
  Var<HBFixed>		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + 3 * Var<HBFixed>::static_size);
};

template <template<typename> class Var>
struct PaintSkew
{
  HB_INTERNAL void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    return_trace (out->src.serialize_subset (c, src, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && src.sanitize (c, this));
  }

  HBUINT8		format; /* format = 18(noVar) or 19 (Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintSkew table) to Paint subtable. */
  Var<HBFixed>		xSkewAngle;
  Var<HBFixed>		ySkewAngle;
  Var<HBFixed>		centerX;
  Var<HBFixed>		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + 4 * Var<HBFixed>::static_size);
};

struct PaintComposite
{
  void closurev1 (hb_colrv1_closure_context_t* c) const;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    if (!out->src.serialize_subset (c, src, this)) return_trace (false);
    return_trace (out->backdrop.serialize_subset (c, backdrop, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  src.sanitize (c, this) &&
                  backdrop.sanitize (c, this));
  }

  HBUINT8		format; /* format = 20 */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintComposite table) to source Paint subtable. */
  CompositeMode		mode;   /* If mode is unrecognized use COMPOSITE_CLEAR */
  Offset24To<Paint>	backdrop; /* Offset (from beginning of PaintComposite table) to backdrop Paint subtable. */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct Paint
{
  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.paintformat1, hb_forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.paintformat2, hb_forward<Ts> (ds)...));
    case 3: return_trace (c->dispatch (u.paintformat3, hb_forward<Ts> (ds)...));
    case 4: return_trace (c->dispatch (u.paintformat4, hb_forward<Ts> (ds)...));
    case 5: return_trace (c->dispatch (u.paintformat5, hb_forward<Ts> (ds)...));
    case 6: return_trace (c->dispatch (u.paintformat6, hb_forward<Ts> (ds)...));
    case 7: return_trace (c->dispatch (u.paintformat7, hb_forward<Ts> (ds)...));
    case 8: return_trace (c->dispatch (u.paintformat8, hb_forward<Ts> (ds)...));
    case 9: return_trace (c->dispatch (u.paintformat9, hb_forward<Ts> (ds)...));
    case 10: return_trace (c->dispatch (u.paintformat10, hb_forward<Ts> (ds)...));
    case 11: return_trace (c->dispatch (u.paintformat11, hb_forward<Ts> (ds)...));
    case 12: return_trace (c->dispatch (u.paintformat12, hb_forward<Ts> (ds)...));
    case 13: return_trace (c->dispatch (u.paintformat13, hb_forward<Ts> (ds)...));
    case 14: return_trace (c->dispatch (u.paintformat14, hb_forward<Ts> (ds)...));
    case 15: return_trace (c->dispatch (u.paintformat15, hb_forward<Ts> (ds)...));
    case 16: return_trace (c->dispatch (u.paintformat16, hb_forward<Ts> (ds)...));
    case 17: return_trace (c->dispatch (u.paintformat17, hb_forward<Ts> (ds)...));
    case 18: return_trace (c->dispatch (u.paintformat18, hb_forward<Ts> (ds)...));
    case 19: return_trace (c->dispatch (u.paintformat19, hb_forward<Ts> (ds)...));
    case 20: return_trace (c->dispatch (u.paintformat20, hb_forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT8				format;
  PaintColrLayers			paintformat1;
  PaintSolid<NoVariable>		paintformat2;
  PaintSolid<Variable>			paintformat3;
  PaintLinearGradient<NoVariable>	paintformat4;
  PaintLinearGradient<Variable>		paintformat5;
  PaintRadialGradient<NoVariable>	paintformat6;
  PaintRadialGradient<Variable>		paintformat7;
  PaintSweepGradient<NoVariable>	paintformat8;
  PaintSweepGradient<Variable>		paintformat9;
  PaintGlyph				paintformat10;
  PaintColrGlyph			paintformat11;
  PaintTransform<NoVariable>		paintformat12;
  PaintTransform<Variable>		paintformat13;
  PaintTranslate<NoVariable>		paintformat14;
  PaintTranslate<Variable>		paintformat15;
  PaintRotate<NoVariable>		paintformat16;
  PaintRotate<Variable>			paintformat17;
  PaintSkew<NoVariable>			paintformat18;
  PaintSkew<Variable>			paintformat19;
  PaintComposite			paintformat20;
  } u;
};

struct BaseGlyphV1Record
{
  int cmp (hb_codepoint_t g) const
  { return g < glyphId ? -1 : g > glyphId ? 1 : 0; }

  bool serialize (hb_serialize_context_t *s, const hb_map_t* glyph_map,
                  const void* src_base, hb_subset_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    auto *out = s->embed (this);
    if (unlikely (!out)) return_trace (false);
    if (!s->check_assign (out->glyphId, glyph_map->get (glyphId),
                          HB_SERIALIZE_ERROR_INT_OVERFLOW))
      return_trace (false);

    return_trace (out->paint.serialize_subset (c, paint, src_base));
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) && paint.sanitize (c, base)));
  }

  public:
  HBGlyphID		glyphId;    /* Glyph ID of reference glyph */
  Offset32To<Paint>	paint;      /* Offset (from beginning of BaseGlyphV1Record array) to Paint,
                                     * Typically PaintColrLayers */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseGlyphV1List : SortedArray32Of<BaseGlyphV1Record>
{
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out)))  return_trace (false);
    const hb_set_t* glyphset = c->plan->_glyphset;

    for (const auto& _ : as_array ())
    {
      unsigned gid = _.glyphId;
      if (!glyphset->has (gid)) continue;

      if (_.serialize (c->serializer, c->plan->glyph_map, this, c)) out->len++;
      else return_trace (false);
    }

    return_trace (out->len != 0);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (SortedArray32Of<BaseGlyphV1Record>::sanitize (c, this));
  }
};

struct LayerV1List : Array32OfOffset32To<Paint>
{
  const Paint& get_paint (unsigned i) const
  { return this+(*this)[i]; }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out)))  return_trace (false);

    for (const auto& _ : + hb_enumerate (*this)
                         | hb_filter (c->plan->colrv1_layers, hb_first))

    {
      auto *o = out->serialize_append (c->serializer);
      if (unlikely (!o) || !o->serialize_subset (c, _.second, this))
        return_trace (false);
    }
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (Array32OfOffset32To<Paint>::sanitize (c, this));
  }
};

struct COLR
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_COLR;

  bool has_data () const { return numBaseGlyphs; }

  unsigned int get_glyph_layers (hb_codepoint_t       glyph,
				 unsigned int         start_offset,
				 unsigned int        *count, /* IN/OUT.  May be NULL. */
				 hb_ot_color_layer_t *layers /* OUT.     May be NULL. */) const
  {
    const BaseGlyphRecord &record = (this+baseGlyphsZ).bsearch (numBaseGlyphs, glyph);

    hb_array_t<const LayerRecord> all_layers = (this+layersZ).as_array (numLayers);
    hb_array_t<const LayerRecord> glyph_layers = all_layers.sub_array (record.firstLayerIdx,
								       record.numLayers);
    if (count)
    {
      + glyph_layers.sub_array (start_offset, count)
      | hb_sink (hb_array (layers, *count))
      ;
    }
    return glyph_layers.length;
  }

  struct accelerator_t
  {
    accelerator_t () {}
    ~accelerator_t () { fini (); }

    void init (hb_face_t *face)
    { colr = hb_sanitize_context_t ().reference_table<COLR> (face); }

    void fini () { this->colr.destroy (); }

    bool is_valid () { return colr.get_blob ()->length; }

    void closure_glyphs (hb_codepoint_t glyph,
			 hb_set_t *related_ids /* OUT */) const
    { colr->closure_glyphs (glyph, related_ids); }

    void closure_V0palette_indices (const hb_set_t *glyphs,
				    hb_set_t *palettes /* OUT */) const
    { colr->closure_V0palette_indices (glyphs, palettes); }

    void closure_forV1 (hb_set_t *glyphset,
                        hb_set_t *layer_indices,
                        hb_set_t *palette_indices) const
    { colr->closure_forV1 (glyphset, layer_indices, palette_indices); }

    private:
    hb_blob_ptr_t<COLR> colr;
  };

  void closure_glyphs (hb_codepoint_t glyph,
		       hb_set_t *related_ids /* OUT */) const
  {
    const BaseGlyphRecord *record = get_base_glyph_record (glyph);
    if (!record) return;

    auto glyph_layers = (this+layersZ).as_array (numLayers).sub_array (record->firstLayerIdx,
								       record->numLayers);
    if (!glyph_layers.length) return;
    related_ids->add_array (&glyph_layers[0].glyphId, glyph_layers.length, LayerRecord::min_size);
  }

  void closure_V0palette_indices (const hb_set_t *glyphs,
				  hb_set_t *palettes /* OUT */) const
  {
    if (!numBaseGlyphs || !numLayers) return;
    hb_array_t<const BaseGlyphRecord> baseGlyphs = (this+baseGlyphsZ).as_array (numBaseGlyphs);
    hb_array_t<const LayerRecord> all_layers = (this+layersZ).as_array (numLayers);

    for (const BaseGlyphRecord record : baseGlyphs)
    {
      if (!glyphs->has (record.glyphId)) continue;
      hb_array_t<const LayerRecord> glyph_layers = all_layers.sub_array (record.firstLayerIdx,
                                                                   record.numLayers);
      for (const LayerRecord layer : glyph_layers)
        palettes->add (layer.colorIdx);
    }
  }

  void closure_forV1 (hb_set_t *glyphset,
                      hb_set_t *layer_indices,
                      hb_set_t *palette_indices) const
  {
    if (version != 1) return;
    hb_set_t visited_glyphs;

    hb_colrv1_closure_context_t c (this, &visited_glyphs, layer_indices, palette_indices);
    const BaseGlyphV1List &baseglyphV1_records = this+baseGlyphsV1List;

    for (const BaseGlyphV1Record &baseglyphV1record: baseglyphV1_records.iter ())
    {
      unsigned gid = baseglyphV1record.glyphId;
      if (!glyphset->has (gid)) continue;

      const Paint &paint = &baseglyphV1_records+baseglyphV1record.paint;
      paint.dispatch (&c);
    }
    hb_set_union (glyphset, &visited_glyphs);
  }

  const LayerV1List& get_layerV1List () const
  { return (this+layersV1); }

  const BaseGlyphV1List& get_baseglyphV1List () const
  { return (this+baseGlyphsV1List); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  (this+baseGlyphsZ).sanitize (c, numBaseGlyphs) &&
                  (this+layersZ).sanitize (c, numLayers) &&
                  (version == 0 ||
		   (COLRV1_ENABLE_SUBSETTING && version == 1 &&
		    baseGlyphsV1List.sanitize (c, this) &&
		    layersV1.sanitize (c, this) &&
		    varStore.sanitize (c, this))));
  }

  template<typename BaseIterator, typename LayerIterator,
	   hb_requires (hb_is_iterator (BaseIterator)),
	   hb_requires (hb_is_iterator (LayerIterator))>
  bool serialize_V0 (hb_serialize_context_t *c,
		     unsigned version,
		     BaseIterator base_it,
		     LayerIterator layer_it)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (base_it.len () != layer_it.len ()))
      return_trace (false);

    if (unlikely (!c->extend_min (this))) return_trace (false);
    this->version = version;
    numLayers = 0;
    numBaseGlyphs = base_it.len ();
    if (base_it.len () == 0)
    {
      baseGlyphsZ = 0;
      layersZ = 0;
      return_trace (true);
    }
    baseGlyphsZ = COLR::min_size;
    layersZ = COLR::min_size + numBaseGlyphs * BaseGlyphRecord::min_size;

    for (const hb_item_type<BaseIterator> _ : + base_it.iter ())
    {
      auto* record = c->embed (_);
      if (unlikely (!record)) return_trace (false);
      record->firstLayerIdx = numLayers;
      numLayers += record->numLayers;
    }

    for (const hb_item_type<LayerIterator>& _ : + layer_it.iter ())
      _.as_array ().copy (c);

    return_trace (true);
  }

  const BaseGlyphRecord* get_base_glyph_record (hb_codepoint_t gid) const
  {
    if ((unsigned int) gid == 0) // Ignore notdef.
      return nullptr;
    const BaseGlyphRecord* record = &(this+baseGlyphsZ).bsearch (numBaseGlyphs, (unsigned int) gid);
    if ((record && (hb_codepoint_t) record->glyphId != gid))
      record = nullptr;
    return record;
  }

  const BaseGlyphV1Record* get_base_glyphV1_record (hb_codepoint_t gid) const
  {
    const BaseGlyphV1Record* record = &(this+baseGlyphsV1List).bsearch ((unsigned) gid);
    if ((record && (hb_codepoint_t) record->glyphId != gid))
      record = nullptr;
    return record;
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    const hb_map_t &reverse_glyph_map = *c->plan->reverse_glyph_map;

    auto base_it =
    + hb_range (c->plan->num_output_glyphs ())
    | hb_map_retains_sorting ([&](hb_codepoint_t new_gid)
			      {
				hb_codepoint_t old_gid = reverse_glyph_map.get (new_gid);

				const BaseGlyphRecord* old_record = get_base_glyph_record (old_gid);
				if (unlikely (!old_record))
				  return hb_pair_t<bool, BaseGlyphRecord> (false, Null (BaseGlyphRecord));

				BaseGlyphRecord new_record = {};
				new_record.glyphId = new_gid;
				new_record.numLayers = old_record->numLayers;
				return hb_pair_t<bool, BaseGlyphRecord> (true, new_record);
			      })
    | hb_filter (hb_first)
    | hb_map_retains_sorting (hb_second)
    ;

    auto layer_it =
    + hb_range (c->plan->num_output_glyphs ())
    | hb_map (reverse_glyph_map)
    | hb_map_retains_sorting ([&](hb_codepoint_t old_gid)
			      {
				const BaseGlyphRecord* old_record = get_base_glyph_record (old_gid);
				hb_vector_t<LayerRecord> out_layers;

				if (unlikely (!old_record ||
					      old_record->firstLayerIdx >= numLayers ||
					      old_record->firstLayerIdx + old_record->numLayers > numLayers))
				  return hb_pair_t<bool, hb_vector_t<LayerRecord>> (false, out_layers);

				auto layers = (this+layersZ).as_array (numLayers).sub_array (old_record->firstLayerIdx,
											     old_record->numLayers);
				out_layers.resize (layers.length);
				for (unsigned int i = 0; i < layers.length; i++) {
				  out_layers[i] = layers[i];
				  hb_codepoint_t new_gid = 0;
				  if (unlikely (!c->plan->new_gid_for_old_gid (out_layers[i].glyphId, &new_gid)))
				    return hb_pair_t<bool, hb_vector_t<LayerRecord>> (false, out_layers);
				  out_layers[i].glyphId = new_gid;
				  out_layers[i].colorIdx = c->plan->colr_palettes->get (layers[i].colorIdx);
				}

				return hb_pair_t<bool, hb_vector_t<LayerRecord>> (true, out_layers);
			      })
    | hb_filter (hb_first)
    | hb_map_retains_sorting (hb_second)
    ;

    if (version == 0 && (!base_it || !layer_it))
      return_trace (false);

    COLR *colr_prime = c->serializer->start_embed<COLR> ();
    bool ret = colr_prime->serialize_V0 (c->serializer, version, base_it, layer_it);

    if (version == 0) return_trace (ret);
    auto snap = c->serializer->snapshot ();
    if (!c->serializer->allocate_size<void> (3 * HBUINT32::static_size)) return_trace (false);
    if (!colr_prime->baseGlyphsV1List.serialize_subset (c, baseGlyphsV1List, this))
    {
      if (c->serializer->in_error ()) return_trace (false);
      //no more COLRv1 glyphs: downgrade to version 0
      c->serializer->revert (snap);
      colr_prime->version = 0;
      return_trace (true);
    }

    if (!colr_prime->layersV1.serialize_subset (c, layersV1, this)) return_trace (false);

    colr_prime->varStore = 0;
    //TODO: subset varStore once it's implemented in fonttools
    return_trace (true);
  }

  protected:
  HBUINT16	version;	/* Table version number (starts at 0). */
  HBUINT16	numBaseGlyphs;	/* Number of Base Glyph Records. */
  NNOffset32To<SortedUnsizedArrayOf<BaseGlyphRecord>>
		baseGlyphsZ;	/* Offset to Base Glyph records. */
  NNOffset32To<UnsizedArrayOf<LayerRecord>>
		layersZ;	/* Offset to Layer Records. */
  HBUINT16	numLayers;	/* Number of Layer Records. */
  // Version-1 additions
  Offset32To<BaseGlyphV1List>		baseGlyphsV1List;
  Offset32To<LayerV1List>		layersV1;
  Offset32To<VariationStore>		varStore;
  public:
  DEFINE_SIZE_MIN (14);
};

} /* namespace OT */


#endif /* HB_OT_COLOR_COLR_TABLE_HH */
