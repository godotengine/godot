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

/*
 * COLR -- Color
 * https://docs.microsoft.com/en-us/typography/opentype/spec/colr
 */
#define HB_OT_TAG_COLR HB_TAG('C','O','L','R')


namespace OT {


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

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (this+baseGlyphsZ).sanitize (c, numBaseGlyphs) &&
			  (this+layersZ).sanitize (c, numLayers)));
  }

  template<typename BaseIterator, typename LayerIterator,
	   hb_requires (hb_is_iterator (BaseIterator)),
	   hb_requires (hb_is_iterator (LayerIterator))>
  bool serialize (hb_serialize_context_t *c,
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
				}

				return hb_pair_t<bool, hb_vector_t<LayerRecord>> (true, out_layers);
			      })
    | hb_filter (hb_first)
    | hb_map_retains_sorting (hb_second)
    ;

    if (unlikely (!base_it || !layer_it || base_it.len () != layer_it.len ()))
      return_trace (false);

    COLR *colr_prime = c->serializer->start_embed<COLR> ();
    return_trace (colr_prime->serialize (c->serializer, version, base_it, layer_it));
  }

  protected:
  HBUINT16	version;	/* Table version number (starts at 0). */
  HBUINT16	numBaseGlyphs;	/* Number of Base Glyph Records. */
  LNNOffsetTo<SortedUnsizedArrayOf<BaseGlyphRecord>>
		baseGlyphsZ;	/* Offset to Base Glyph records. */
  LNNOffsetTo<UnsizedArrayOf<LayerRecord>>
		layersZ;	/* Offset to Layer Records. */
  HBUINT16	numLayers;	/* Number of Layer Records. */
  public:
  DEFINE_SIZE_STATIC (14);
};

} /* namespace OT */


#endif /* HB_OT_COLOR_COLR_TABLE_HH */
