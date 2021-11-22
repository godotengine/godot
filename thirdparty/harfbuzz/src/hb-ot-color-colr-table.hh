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
#include "hb-ot-var-common.hh"

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
#define HB_OT_TAG_COLR HB_TAG('C','O','L','R')

#ifndef COLRV1_MAX_NESTING_LEVEL
#define COLRV1_MAX_NESTING_LEVEL	100
#endif

#ifndef COLRV1_ENABLE_SUBSETTING
#define COLRV1_ENABLE_SUBSETTING 1
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
  HBGlyphID16	glyphId;	/* Glyph ID of layer glyph */
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
  HBGlyphID16	glyphId;	/* Glyph ID of reference glyph */
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
  Variable<T>* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed (this));
  }

  void closurev1 (hb_colrv1_closure_context_t* c) const
  { value.closurev1 (c); }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    if (!value.subset (c)) return_trace (false);
    return_trace (c->serializer->embed (varIdxBase));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && value.sanitize (c));
  }

  protected:
  T      value;
  VarIdx varIdxBase;
  public:
  DEFINE_SIZE_STATIC (4 + T::static_size);
};

template <typename T>
struct NoVariable
{
  NoVariable<T>* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed (this));
  }

  void closurev1 (hb_colrv1_closure_context_t* c) const
  { value.closurev1 (c); }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    return_trace (value.subset (c));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && value.sanitize (c));
  }

  T      value;
  public:
  DEFINE_SIZE_STATIC (T::static_size);
};

// Color structures

struct ColorStop
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { c->add_palette_index (paletteIndex); }

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

  F2DOT14	stopOffset;
  HBUINT16	paletteIndex;
  F2DOT14	alpha;
  public:
  DEFINE_SIZE_STATIC (2 + 2 * F2DOT14::static_size);
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
  void closurev1 (hb_colrv1_closure_context_t* c) const
  {
    for (const auto &stop : stops.iter ())
      stop.closurev1 (c);
  }

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

  Extend	extend;
  Array16Of<Var<ColorStop>>	stops;
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

struct Affine2x3
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBFixed xx;
  HBFixed yx;
  HBFixed xy;
  HBFixed yy;
  HBFixed dx;
  HBFixed dy;
  public:
  DEFINE_SIZE_STATIC (6 * HBFixed::static_size);
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
  HBUINT32	firstLayerIndex;  /* index into COLRv1::layerList */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct PaintSolid
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { c->add_palette_index (paletteIndex); }

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

  HBUINT8	format; /* format = 2(noVar) or 3(Var)*/
  HBUINT16	paletteIndex;
  F2DOT14	alpha;
  public:
  DEFINE_SIZE_STATIC (3 + F2DOT14::static_size);
};

template <template<typename> class Var>
struct PaintLinearGradient
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { (this+colorLine).closurev1 (c); }

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
  FWORD			x0;
  FWORD			y0;
  FWORD			x1;
  FWORD			y1;
  FWORD			x2;
  FWORD			y2;
  public:
  DEFINE_SIZE_STATIC (4 + 6 * FWORD::static_size);
};

template <template<typename> class Var>
struct PaintRadialGradient
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { (this+colorLine).closurev1 (c); }

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

  HBUINT8			format; /* format = 6(noVar) or 7 (Var) */
  Offset24To<ColorLine<Var>>	colorLine; /* Offset (from beginning of PaintRadialGradient
                                            * table) to ColorLine subtable. */
  FWORD			x0;
  FWORD			y0;
  UFWORD		radius0;
  FWORD			x1;
  FWORD			y1;
  UFWORD		radius1;
  public:
  DEFINE_SIZE_STATIC (4 + 6 * FWORD::static_size);
};

template <template<typename> class Var>
struct PaintSweepGradient
{
  void closurev1 (hb_colrv1_closure_context_t* c) const
  { (this+colorLine).closurev1 (c); }

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

  HBUINT8			format; /* format = 8(noVar) or 9 (Var) */
  Offset24To<ColorLine<Var>>	colorLine; /* Offset (from beginning of PaintSweepGradient
                                            * table) to ColorLine subtable. */
  FWORD			centerX;
  FWORD			centerY;
  F2DOT14		startAngle;
  F2DOT14		endAngle;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * FWORD::static_size + 2 * F2DOT14::static_size);
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
    if (!out->transform.serialize_copy (c->serializer, transform, this)) return_trace (false);
    return_trace (out->src.serialize_subset (c, src, this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  src.sanitize (c, this) &&
                  transform.sanitize (c, this));
  }

  HBUINT8			format; /* format = 12(noVar) or 13 (Var) */
  Offset24To<Paint>		src; /* Offset (from beginning of PaintTransform table) to Paint subtable. */
  Offset24To<Var<Affine2x3>>	transform;
  public:
  DEFINE_SIZE_STATIC (7);
};

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
  FWORD		dx;
  FWORD		dy;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * FWORD::static_size);
};

struct PaintScale
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
  Offset24To<Paint>	src; /* Offset (from beginning of PaintScale table) to Paint subtable. */
  F2DOT14		scaleX;
  F2DOT14		scaleY;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * F2DOT14::static_size);
};

struct PaintScaleAroundCenter
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

  HBUINT8		format; /* format = 18 (noVar) or 19(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintScaleAroundCenter table) to Paint subtable. */
  F2DOT14	scaleX;
  F2DOT14	scaleY;
  FWORD		centerX;
  FWORD		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * F2DOT14::static_size + 2 * FWORD::static_size);
};

struct PaintScaleUniform
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

  HBUINT8		format; /* format = 20 (noVar) or 21(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintScaleUniform table) to Paint subtable. */
  F2DOT14		scale;
  public:
  DEFINE_SIZE_STATIC (4 + F2DOT14::static_size);
};

struct PaintScaleUniformAroundCenter
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

  HBUINT8		format; /* format = 22 (noVar) or 23(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintScaleUniformAroundCenter table) to Paint subtable. */
  F2DOT14	scale;
  FWORD		centerX;
  FWORD		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + F2DOT14::static_size + 2 * FWORD::static_size);
};

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

  HBUINT8		format; /* format = 24 (noVar) or 25(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintRotate table) to Paint subtable. */
  F2DOT14		angle;
  public:
  DEFINE_SIZE_STATIC (4 + F2DOT14::static_size);
};

struct PaintRotateAroundCenter
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

  HBUINT8		format; /* format = 26 (noVar) or 27(Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintRotateAroundCenter table) to Paint subtable. */
  F2DOT14	angle;
  FWORD		centerX;
  FWORD		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + F2DOT14::static_size + 2 * FWORD::static_size);
};

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

  HBUINT8		format; /* format = 28(noVar) or 29 (Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintSkew table) to Paint subtable. */
  F2DOT14		xSkewAngle;
  F2DOT14		ySkewAngle;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * F2DOT14::static_size);
};

struct PaintSkewAroundCenter
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

  HBUINT8		format; /* format = 30(noVar) or 31 (Var) */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintSkewAroundCenter table) to Paint subtable. */
  F2DOT14	xSkewAngle;
  F2DOT14	ySkewAngle;
  FWORD		centerX;
  FWORD		centerY;
  public:
  DEFINE_SIZE_STATIC (4 + 2 * F2DOT14::static_size + 2 * FWORD::static_size);
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

  HBUINT8		format; /* format = 32 */
  Offset24To<Paint>	src; /* Offset (from beginning of PaintComposite table) to source Paint subtable. */
  CompositeMode		mode;   /* If mode is unrecognized use COMPOSITE_CLEAR */
  Offset24To<Paint>	backdrop; /* Offset (from beginning of PaintComposite table) to backdrop Paint subtable. */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct ClipBoxFormat1
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBUINT8	format; /* format = 1(noVar) or 2(Var)*/
  FWORD		xMin;
  FWORD		yMin;
  FWORD		xMax;
  FWORD		yMax;
  public:
  DEFINE_SIZE_STATIC (1 + 4 * FWORD::static_size);
};

struct ClipBoxFormat2 : Variable<ClipBoxFormat1> {};

struct ClipBox
{
  ClipBox* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    switch (u.format) {
    case 1: return_trace (reinterpret_cast<ClipBox *> (c->embed (u.format1)));
    case 2: return_trace (reinterpret_cast<ClipBox *> (c->embed (u.format2)));
    default:return_trace (nullptr);
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT8		format;         /* Format identifier */
  ClipBoxFormat1	format1;
  ClipBoxFormat2	format2;
  } u;
};

struct ClipRecord
{
  ClipRecord* copy (hb_serialize_context_t *c, const void *base) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->embed (this);
    if (unlikely (!out)) return_trace (nullptr);
    if (!out->clipBox.serialize_copy (c, clipBox, base)) return_trace (nullptr);
    return_trace (out);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && clipBox.sanitize (c, base));
  }

  public:
  HBUINT16		startGlyphID;  // first gid clip applies to
  HBUINT16		endGlyphID;    // last gid clip applies to, inclusive
  Offset24To<ClipBox>	clipBox;   // Box or VarBox
  public:
  DEFINE_SIZE_STATIC (7);
};

struct ClipList
{
  unsigned serialize_clip_records (hb_serialize_context_t *c,
                                   const hb_set_t& gids,
                                   const hb_map_t& gid_offset_map) const
  {
    TRACE_SERIALIZE (this);
    if (gids.is_empty () ||
        gid_offset_map.get_population () != gids.get_population ())
      return_trace (0);

    unsigned count  = 0;

    hb_codepoint_t start_gid= gids.get_min ();
    hb_codepoint_t prev_gid = start_gid;

    unsigned offset = gid_offset_map.get (start_gid);
    unsigned prev_offset = offset;
    for (const hb_codepoint_t _ : gids.iter ())
    {
      if (_ == start_gid) continue;
      
      offset = gid_offset_map.get (_);
      if (_ == prev_gid + 1 &&  offset == prev_offset)
      {
        prev_gid = _;
        continue;
      }

      ClipRecord record;
      record.startGlyphID = start_gid;
      record.endGlyphID = prev_gid;
      record.clipBox = prev_offset;

      if (!c->copy (record, this)) return_trace (0);
      count++;

      start_gid = _;
      prev_gid = _;
      prev_offset = offset;
    }

    //last one
    {
      ClipRecord record;
      record.startGlyphID = start_gid;
      record.endGlyphID = prev_gid;
      record.clipBox = prev_offset;
      if (!c->copy (record, this)) return_trace (0);
      count++;
    }
    return_trace (count);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    if (!c->serializer->check_assign (out->format, format, HB_SERIALIZE_ERROR_INT_OVERFLOW)) return_trace (false);

    const hb_set_t& glyphset = *c->plan->_glyphset;
    const hb_map_t &glyph_map = *c->plan->glyph_map;
    
    hb_map_t new_gid_offset_map;
    hb_set_t new_gids;
    for (const ClipRecord& record : clips.iter ())
    {
      unsigned start_gid = record.startGlyphID;
      unsigned end_gid = record.endGlyphID;
      for (unsigned gid = start_gid; gid <= end_gid; gid++)
      {
        if (!glyphset.has (gid) || !glyph_map.has (gid)) continue;
        unsigned new_gid = glyph_map.get (gid);
        new_gid_offset_map.set (new_gid, record.clipBox);
        new_gids.add (new_gid);
      }
    }

    unsigned count = serialize_clip_records (c->serializer, new_gids, new_gid_offset_map);
    if (!count) return_trace (false);
    return_trace (c->serializer->check_assign (out->clips.len, count, HB_SERIALIZE_ERROR_INT_OVERFLOW));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && clips.sanitize (c, this));
  }

  HBUINT8			format;  // Set to 1.
  Array32Of<ClipRecord>		clips;  // Clip records, sorted by startGlyphID
  public:
  DEFINE_SIZE_ARRAY_SIZED (5, clips);
};

struct Paint
{
  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.paintformat1, std::forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.paintformat2, std::forward<Ts> (ds)...));
    case 3: return_trace (c->dispatch (u.paintformat3, std::forward<Ts> (ds)...));
    case 4: return_trace (c->dispatch (u.paintformat4, std::forward<Ts> (ds)...));
    case 5: return_trace (c->dispatch (u.paintformat5, std::forward<Ts> (ds)...));
    case 6: return_trace (c->dispatch (u.paintformat6, std::forward<Ts> (ds)...));
    case 7: return_trace (c->dispatch (u.paintformat7, std::forward<Ts> (ds)...));
    case 8: return_trace (c->dispatch (u.paintformat8, std::forward<Ts> (ds)...));
    case 9: return_trace (c->dispatch (u.paintformat9, std::forward<Ts> (ds)...));
    case 10: return_trace (c->dispatch (u.paintformat10, std::forward<Ts> (ds)...));
    case 11: return_trace (c->dispatch (u.paintformat11, std::forward<Ts> (ds)...));
    case 12: return_trace (c->dispatch (u.paintformat12, std::forward<Ts> (ds)...));
    case 13: return_trace (c->dispatch (u.paintformat13, std::forward<Ts> (ds)...));
    case 14: return_trace (c->dispatch (u.paintformat14, std::forward<Ts> (ds)...));
    case 15: return_trace (c->dispatch (u.paintformat15, std::forward<Ts> (ds)...));
    case 16: return_trace (c->dispatch (u.paintformat16, std::forward<Ts> (ds)...));
    case 17: return_trace (c->dispatch (u.paintformat17, std::forward<Ts> (ds)...));
    case 18: return_trace (c->dispatch (u.paintformat18, std::forward<Ts> (ds)...));
    case 19: return_trace (c->dispatch (u.paintformat19, std::forward<Ts> (ds)...));
    case 20: return_trace (c->dispatch (u.paintformat20, std::forward<Ts> (ds)...));
    case 21: return_trace (c->dispatch (u.paintformat21, std::forward<Ts> (ds)...));
    case 22: return_trace (c->dispatch (u.paintformat22, std::forward<Ts> (ds)...));
    case 23: return_trace (c->dispatch (u.paintformat23, std::forward<Ts> (ds)...));
    case 24: return_trace (c->dispatch (u.paintformat24, std::forward<Ts> (ds)...));
    case 25: return_trace (c->dispatch (u.paintformat25, std::forward<Ts> (ds)...));
    case 26: return_trace (c->dispatch (u.paintformat26, std::forward<Ts> (ds)...));
    case 27: return_trace (c->dispatch (u.paintformat27, std::forward<Ts> (ds)...));
    case 28: return_trace (c->dispatch (u.paintformat28, std::forward<Ts> (ds)...));
    case 29: return_trace (c->dispatch (u.paintformat29, std::forward<Ts> (ds)...));
    case 30: return_trace (c->dispatch (u.paintformat30, std::forward<Ts> (ds)...));
    case 31: return_trace (c->dispatch (u.paintformat31, std::forward<Ts> (ds)...));
    case 32: return_trace (c->dispatch (u.paintformat32, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT8					format;
  PaintColrLayers				paintformat1;
  PaintSolid					paintformat2;
  Variable<PaintSolid>				paintformat3;
  PaintLinearGradient<NoVariable>		paintformat4;
  Variable<PaintLinearGradient<Variable>>	paintformat5;
  PaintRadialGradient<NoVariable>		paintformat6;
  Variable<PaintRadialGradient<Variable>>	paintformat7;
  PaintSweepGradient<NoVariable>		paintformat8;
  Variable<PaintSweepGradient<Variable>>	paintformat9;
  PaintGlyph					paintformat10;
  PaintColrGlyph				paintformat11;
  PaintTransform<NoVariable>			paintformat12;
  PaintTransform<Variable>			paintformat13;
  PaintTranslate				paintformat14;
  Variable<PaintTranslate>			paintformat15;
  PaintScale					paintformat16;
  Variable<PaintScale>				paintformat17;
  PaintScaleAroundCenter			paintformat18;
  Variable<PaintScaleAroundCenter>		paintformat19;
  PaintScaleUniform				paintformat20;
  Variable<PaintScaleUniform>			paintformat21;
  PaintScaleUniformAroundCenter			paintformat22;
  Variable<PaintScaleUniformAroundCenter>	paintformat23;
  PaintRotate					paintformat24;
  Variable<PaintRotate>				paintformat25;
  PaintRotateAroundCenter			paintformat26;
  Variable<PaintRotateAroundCenter>		paintformat27;
  PaintSkew					paintformat28;
  Variable<PaintSkew>				paintformat29;
  PaintSkewAroundCenter				paintformat30;
  Variable<PaintSkewAroundCenter>		paintformat31;
  PaintComposite				paintformat32;
  } u;
};

struct BaseGlyphPaintRecord
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
  HBGlyphID16		glyphId;    /* Glyph ID of reference glyph */
  Offset32To<Paint>	paint;      /* Offset (from beginning of BaseGlyphPaintRecord array) to Paint,
                                     * Typically PaintColrLayers */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseGlyphList : SortedArray32Of<BaseGlyphPaintRecord>
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
    return_trace (SortedArray32Of<BaseGlyphPaintRecord>::sanitize (c, this));
  }
};

struct LayerList : Array32OfOffset32To<Paint>
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
    const BaseGlyphList &baseglyph_paintrecords = this+baseGlyphList;

    for (const BaseGlyphPaintRecord &baseglyph_paintrecord: baseglyph_paintrecords.iter ())
    {
      unsigned gid = baseglyph_paintrecord.glyphId;
      if (!glyphset->has (gid)) continue;

      const Paint &paint = &baseglyph_paintrecords+baseglyph_paintrecord.paint;
      paint.dispatch (&c);
    }
    hb_set_union (glyphset, &visited_glyphs);
  }

  const LayerList& get_layerList () const
  { return (this+layerList); }

  const BaseGlyphList& get_baseglyphList () const
  { return (this+baseGlyphList); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  (this+baseGlyphsZ).sanitize (c, numBaseGlyphs) &&
                  (this+layersZ).sanitize (c, numLayers) &&
                  (version == 0 ||
		   (COLRV1_ENABLE_SUBSETTING && version == 1 &&
		    baseGlyphList.sanitize (c, this) &&
		    layerList.sanitize (c, this) &&
		    clipList.sanitize (c, this) &&
		    varIdxMap.sanitize (c, this) &&
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

    this->version = version;
    numLayers = 0;
    numBaseGlyphs = base_it.len ();
    if (numBaseGlyphs == 0)
    {
      baseGlyphsZ = 0;
      layersZ = 0;
      return_trace (true);
    }

    c->push ();
    for (const hb_item_type<BaseIterator> _ : + base_it.iter ())
    {
      auto* record = c->embed (_);
      if (unlikely (!record)) return_trace (false);
      record->firstLayerIdx = numLayers;
      numLayers += record->numLayers;
    }
    c->add_link (baseGlyphsZ, c->pop_pack ());

    c->push ();
    for (const hb_item_type<LayerIterator>& _ : + layer_it.iter ())
      _.as_array ().copy (c);

    c->add_link (layersZ, c->pop_pack ());

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

  const BaseGlyphPaintRecord* get_base_glyph_paintrecord (hb_codepoint_t gid) const
  {
    const BaseGlyphPaintRecord* record = &(this+baseGlyphList).bsearch ((unsigned) gid);
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
    if (unlikely (!c->serializer->extend_min (colr_prime)))  return_trace (false);

    if (version == 0)
    return_trace (colr_prime->serialize_V0 (c->serializer, version, base_it, layer_it));

    auto snap = c->serializer->snapshot ();
    if (!c->serializer->allocate_size<void> (5 * HBUINT32::static_size)) return_trace (false);
    if (!colr_prime->baseGlyphList.serialize_subset (c, baseGlyphList, this))
    {
      if (c->serializer->in_error ()) return_trace (false);
      //no more COLRv1 glyphs: downgrade to version 0
      c->serializer->revert (snap);
      return_trace (colr_prime->serialize_V0 (c->serializer, 0, base_it, layer_it));
    }

    if (!colr_prime->serialize_V0 (c->serializer, version, base_it, layer_it)) return_trace (false);

    colr_prime->layerList.serialize_subset (c, layerList, this);
    colr_prime->clipList.serialize_subset (c, clipList, this);
    colr_prime->varIdxMap.serialize_copy (c->serializer, varIdxMap, this);
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
  Offset32To<BaseGlyphList>		baseGlyphList;
  Offset32To<LayerList>			layerList;
  Offset32To<ClipList>			clipList;   // Offset to ClipList table (may be NULL)
  Offset32To<DeltaSetIndexMap>		varIdxMap;  // Offset to DeltaSetIndexMap table (may be NULL)
  Offset32To<VariationStore>		varStore;
  public:
  DEFINE_SIZE_MIN (14);
};

} /* namespace OT */


#endif /* HB_OT_COLOR_COLR_TABLE_HH */
