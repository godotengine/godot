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

#ifndef HB_OT_COLR_COLRV1_CLOSURE_HH
#define HB_OT_COLR_COLRV1_CLOSURE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-common.hh"
#include "hb-ot-color-colr-table.hh"

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
namespace OT {

HB_INTERNAL void PaintColrLayers::closurev1 (hb_colrv1_closure_context_t* c) const
{
  c->add_layer_indices (firstLayerIndex, numLayers);
  const LayerV1List &paint_offset_lists = c->get_colr_table ()->get_layerV1List ();
  for (unsigned i = firstLayerIndex; i < firstLayerIndex + numLayers; i++)
  {
    const Paint &paint = hb_addressof (paint_offset_lists) + paint_offset_lists[i];
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
  const BaseGlyphV1Record* baseglyphV1_record = colr_table->get_base_glyphV1_record (gid);
  if (!baseglyphV1_record) return;
  c->add_glyph (gid);

  const BaseGlyphV1List &baseglyphV1_list = colr_table->get_baseglyphV1List ();
  (&baseglyphV1_list+baseglyphV1_record->paint).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintTransform<Var>::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintTranslate<Var>::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintRotate<Var>::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintSkew<Var>::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintComposite::closurev1 (hb_colrv1_closure_context_t* c) const
{
  (this+src).dispatch (c);
  (this+backdrop).dispatch (c);
}

} /* namespace OT */


#endif /* HB_OT_COLR_COLRV1_CLOSURE_HH */
