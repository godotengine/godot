/*
 * Copyright © 2016  Google, Inc.
 * Copyright © 2018  Ebrahim Byagowi
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
 * Google Author(s): Sascha Brawer
 */

#ifndef HB_OT_COLOR_CPAL_TABLE_HH
#define HB_OT_COLOR_CPAL_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-color.h"
#include "hb-ot-name.h"


/*
 * CPAL -- Color Palette
 * https://docs.microsoft.com/en-us/typography/opentype/spec/cpal
 */
#define HB_OT_TAG_CPAL HB_TAG('C','P','A','L')

namespace OT {


struct CPALV1Tail
{
  friend struct CPAL;

  private:
  hb_ot_color_palette_flags_t get_palette_flags (const void *base,
						 unsigned int palette_index,
						 unsigned int palette_count) const
  {
    if (!paletteFlagsZ) return HB_OT_COLOR_PALETTE_FLAG_DEFAULT;
    return (hb_ot_color_palette_flags_t) (uint32_t)
	   (base+paletteFlagsZ).as_array (palette_count)[palette_index];
  }

  hb_ot_name_id_t get_palette_name_id (const void *base,
				       unsigned int palette_index,
				       unsigned int palette_count) const
  {
    if (!paletteLabelsZ) return HB_OT_NAME_ID_INVALID;
    return (base+paletteLabelsZ).as_array (palette_count)[palette_index];
  }

  hb_ot_name_id_t get_color_name_id (const void *base,
				     unsigned int color_index,
				     unsigned int color_count) const
  {
    if (!colorLabelsZ) return HB_OT_NAME_ID_INVALID;
    return (base+colorLabelsZ).as_array (color_count)[color_index];
  }

  public:
  bool serialize (hb_serialize_context_t *c,
                  unsigned palette_count,
                  unsigned color_count,
                  const void *base,
                  const hb_map_t *color_index_map) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->allocate_size<CPALV1Tail> (static_size);
    if (unlikely (!out)) return_trace (false);

    out->paletteFlagsZ = 0;
    if (paletteFlagsZ)
      out->paletteFlagsZ.serialize_copy (c, paletteFlagsZ, base, 0, hb_serialize_context_t::Head, palette_count);

    out->paletteLabelsZ = 0;
    if (paletteLabelsZ)
      out->paletteLabelsZ.serialize_copy (c, paletteLabelsZ, base, 0, hb_serialize_context_t::Head, palette_count);

    const hb_array_t<const NameID> colorLabels = (base+colorLabelsZ).as_array (color_count);
    if (colorLabelsZ)
    {
      c->push ();
      for (const auto _ : colorLabels)
      {
        if (!color_index_map->has (_)) continue;
        NameID new_color_idx;
        new_color_idx = color_index_map->get (_);
        if (!c->copy<NameID> (new_color_idx))
        {
          c->pop_discard ();
          return_trace (false);
        }
      }
      c->add_link (out->colorLabelsZ, c->pop_pack ());
    }
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c,
		 const void *base,
		 unsigned int palette_count,
		 unsigned int color_count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  (!paletteFlagsZ  || (base+paletteFlagsZ).sanitize (c, palette_count)) &&
		  (!paletteLabelsZ || (base+paletteLabelsZ).sanitize (c, palette_count)) &&
		  (!colorLabelsZ   || (base+colorLabelsZ).sanitize (c, color_count)));
  }

  protected:
  // TODO(garretrieger): these offsets can hold nulls so we should not be using non-null offsets
  //                     here. Currently they are needed since UnsizedArrayOf doesn't define null_size
  NNOffset32To<UnsizedArrayOf<HBUINT32>>
		paletteFlagsZ;		/* Offset from the beginning of CPAL table to
					 * the Palette Type Array. Set to 0 if no array
					 * is provided. */
  NNOffset32To<UnsizedArrayOf<NameID>>
		paletteLabelsZ;		/* Offset from the beginning of CPAL table to
					 * the palette labels array. Set to 0 if no
					 * array is provided. */
  NNOffset32To<UnsizedArrayOf<NameID>>
		colorLabelsZ;		/* Offset from the beginning of CPAL table to
					 * the color labels array. Set to 0
					 * if no array is provided. */
  public:
  DEFINE_SIZE_STATIC (12);
};

typedef HBUINT32 BGRAColor;

struct CPAL
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_CPAL;

  bool has_data () const { return numPalettes; }

  unsigned int get_size () const
  { return min_size + numPalettes * sizeof (colorRecordIndicesZ[0]); }

  unsigned int get_palette_count () const { return numPalettes; }
  unsigned int   get_color_count () const { return numColors; }

  hb_ot_color_palette_flags_t get_palette_flags (unsigned int palette_index) const
  { return v1 ().get_palette_flags (this, palette_index, numPalettes); }

  hb_ot_name_id_t get_palette_name_id (unsigned int palette_index) const
  { return v1 ().get_palette_name_id (this, palette_index, numPalettes); }

  hb_ot_name_id_t get_color_name_id (unsigned int color_index) const
  { return v1 ().get_color_name_id (this, color_index, numColors); }

  unsigned int get_palette_colors (unsigned int  palette_index,
				   unsigned int  start_offset,
				   unsigned int *color_count, /* IN/OUT.  May be NULL. */
				   hb_color_t   *colors       /* OUT.     May be NULL. */) const
  {
    if (unlikely (palette_index >= numPalettes))
    {
      if (color_count) *color_count = 0;
      return 0;
    }
    unsigned int start_index = colorRecordIndicesZ[palette_index];
    hb_array_t<const BGRAColor> all_colors ((this+colorRecordsZ).arrayZ, numColorRecords);
    hb_array_t<const BGRAColor> palette_colors = all_colors.sub_array (start_index,
								       numColors);
    if (color_count)
    {
      + palette_colors.sub_array (start_offset, color_count)
      | hb_sink (hb_array (colors, *color_count))
      ;
    }
    return numColors;
  }

  private:
  const CPALV1Tail& v1 () const
  {
    if (version == 0) return Null (CPALV1Tail);
    return StructAfter<CPALV1Tail> (*this);
  }

  public:
  bool serialize (hb_serialize_context_t *c,
                  const hb_array_t<const BGRAColor> &color_records,
                  const hb_array_t<const HBUINT16> &color_record_indices,
                  const hb_map_t &color_record_index_map,
                  const hb_set_t &retained_color_record_indices) const
  {
    TRACE_SERIALIZE (this);

    for (const auto idx : color_record_indices)
    {
      HBUINT16 new_idx;
      if (idx == 0) new_idx = 0;
      else new_idx = color_record_index_map.get (idx);
      if (!c->copy<HBUINT16> (new_idx)) return_trace (false);
    }

    c->push ();
    for (const auto _ : retained_color_record_indices.iter ())
    {
      if (!c->copy<BGRAColor> (color_records[_]))
      {
        c->pop_discard ();
        return_trace (false);
      }
    }
    c->add_link (colorRecordsZ, c->pop_pack ());
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_map_t *color_index_map = c->plan->colr_palettes;
    if (color_index_map->is_empty ()) return_trace (false);

    hb_set_t retained_color_indices;
    for (const auto _ : color_index_map->keys ())
    {
      if (_ == 0xFFFF) continue;
      retained_color_indices.add (_);
    }
    if (retained_color_indices.is_empty ()) return_trace (false);

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    out->version = version;
    out->numColors = retained_color_indices.get_population ();
    out->numPalettes = numPalettes;

    const hb_array_t<const HBUINT16> colorRecordIndices = colorRecordIndicesZ.as_array (numPalettes);
    hb_map_t color_record_index_map;
    hb_set_t retained_color_record_indices;

    unsigned record_count = 0;
    for (const auto first_color_record_idx : colorRecordIndices)
    {
      for (unsigned retained_color_idx : retained_color_indices.iter ())
      {
        unsigned color_record_idx = first_color_record_idx + retained_color_idx;
        if (color_record_index_map.has (color_record_idx)) continue;
        color_record_index_map.set (color_record_idx, record_count);
        retained_color_record_indices.add (color_record_idx);
        record_count++;
      }
    }

    out->numColorRecords = record_count;
    const hb_array_t<const BGRAColor> color_records = (this+colorRecordsZ).as_array (numColorRecords);
    if (!out->serialize (c->serializer, color_records, colorRecordIndices, color_record_index_map, retained_color_record_indices))
      return_trace (false);

    if (version == 1)
      return_trace (v1 ().serialize (c->serializer, numPalettes, numColors, this, color_index_map));

    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  (this+colorRecordsZ).sanitize (c, numColorRecords) &&
		  colorRecordIndicesZ.sanitize (c, numPalettes) &&
		  (version == 0 || v1 ().sanitize (c, this, numPalettes, numColors)));
  }

  protected:
  HBUINT16	version;		/* Table version number */
  /* Version 0 */
  HBUINT16	numColors;		/* Number of colors in each palette. */
  HBUINT16	numPalettes;		/* Number of palettes in the table. */
  HBUINT16	numColorRecords;	/* Total number of color records, combined for
					 * all palettes. */
  NNOffset32To<UnsizedArrayOf<BGRAColor>>
		colorRecordsZ;		/* Offset from the beginning of CPAL table to
					 * the first ColorRecord. */
  UnsizedArrayOf<HBUINT16>
		colorRecordIndicesZ;	/* Index of each palette’s first color record in
					 * the combined color record array. */
/*CPALV1Tail	v1;*/
  public:
  DEFINE_SIZE_ARRAY (12, colorRecordIndicesZ);
};

} /* namespace OT */


#endif /* HB_OT_COLOR_CPAL_TABLE_HH */
