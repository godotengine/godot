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
 */

#ifndef OT_COLOR_COLR_COLRV1_CLOSURE_HH
#define OT_COLOR_COLR_COLRV1_CLOSURE_HH

#include "../../../hb-open-type.hh"
#include "COLR.hh"

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
namespace OT {

HB_INTERNAL void PaintColrLayers::closurev1 (hb_colrv1_closure_context_t* c) const
{
  c->add_layer_indices (firstLayerIndex, numLayers);
  const LayerList &paint_offset_lists = c->get_colr_table ()->get_layerList ();
  for (unsigned i = firstLayerIndex; i < firstLayerIndex + numLayers; i++)
  {
    const Paint &paint = std::addressof (paint_offset_lists) + paint_offset_lists[i];
    paint.dispatch (c);
  }
}

HB_INTERNAL void PaintGlyph::closurev1 (hb_colrv1_closure_context_t* c) const
{
  c->add_glyph (gid);
  (this+paint).dispatch (c);
}

HB_INTERNAL void PaintColrGlyph::closurev1 (hb_colrv1_closure_context_t* c) const
{
  const COLR *colr_table = c->get_colr_table ();
  const BaseGlyphPaintRecord* baseglyph_paintrecord = colr_table->get_base_glyph_paintrecord (gid);
  if (!baseglyph_paintrecord) return;
  c->add_glyph (gid);

  const BaseGlyphList &baseglyph_list = colr_table->get_baseglyphList ();
  (&baseglyph_list+baseglyph_paintrecord->paint).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintTransform<Var>::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintTranslate::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintScale::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintScaleAroundCenter::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintScaleUniform::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintScaleUniformAroundCenter::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintRotate::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintRotateAroundCenter::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintSkew::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintSkewAroundCenter::closurev1 (hb_colrv1_closure_context_t* c) const
{ (this+src).dispatch (c); }

HB_INTERNAL void PaintComposite::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
  (this+backdrop).dispatch (c);
}

} /* namespace OT */


#endif /* OT_COLOR_COLR_COLRV1_CLOSURE_HH */
