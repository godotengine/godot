/*
 * Copyright © 2025  Skef Iterum
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

#ifndef OT_COLOR_COLR_COLRV1_DEPEND_HH
#define OT_COLOR_COLR_COLRV1_DEPEND_HH

#include "../../../hb-open-type.hh"
#include "COLR.hh"

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
namespace OT {


HB_INTERNAL void PaintColrLayers::dependv1 (hb_colrv1_depend_context_t* c) const
{
  const LayerList &paint_offset_lists = c->get_colr_table ()->get_layerList ();
  for (unsigned i = firstLayerIndex; i < firstLayerIndex + numLayers; i++)
  {
    const Paint &paint = paint_offset_lists.get_paint (i);
    paint.dispatch (c);
  }
}

HB_INTERNAL void PaintGlyph::dependv1 (hb_colrv1_depend_context_t* c) const
{
  // PaintGlyph (Format 10) references glyph outlines (from glyf/CFF), not paint graphs.
  // Self-reference is valid per OpenType spec: a base glyph can reference its own
  // outline as geometry to fill. This is not a dependency in the traditional sense
  // since the outline is part of the same glyph data, so we skip it.
  if (gid != c->source_gid)
    c->depend_data->add_depend(c->source_gid, HB_OT_TAG_COLR, gid);
  (this+paint).dispatch (c);
}

HB_INTERNAL void PaintColrGlyph::dependv1 (hb_colrv1_depend_context_t* c) const
{
  const COLR *colr_table = c->get_colr_table ();
  const BaseGlyphPaintRecord* baseglyph_paintrecord = colr_table->get_base_glyph_paintrecord (gid);
  if (!baseglyph_paintrecord) return;
  c->depend_data->add_depend(c->source_gid, HB_OT_TAG_COLR, gid);

  const BaseGlyphList &baseglyph_list = colr_table->get_baseglyphList ();
  (&baseglyph_list+baseglyph_paintrecord->paint).dispatch (c);
}

template <template<typename> class Var>
HB_INTERNAL void PaintTransform<Var>::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintTranslate::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintScale::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintScaleAroundCenter::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintScaleUniform::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintScaleUniformAroundCenter::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintRotate::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintRotateAroundCenter::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintSkew::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintSkewAroundCenter::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
}

HB_INTERNAL void PaintComposite::dependv1 (hb_colrv1_depend_context_t* c) const
{
  (this+src).dispatch (c);
  (this+backdrop).dispatch (c);
}


} /* namespace OT */


#endif /* OT_COLOR_COLR_COLRV1_DEPEND_HH */
