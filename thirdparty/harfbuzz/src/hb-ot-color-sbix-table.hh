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

#ifndef HB_OT_COLOR_SBIX_TABLE_HH
#define HB_OT_COLOR_SBIX_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-common.hh"

/*
 * sbix -- Standard Bitmap Graphics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/sbix
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6sbix.html
 */
#define HB_OT_TAG_sbix HB_TAG('s','b','i','x')


namespace OT {


struct SBIXGlyph
{
  SBIXGlyph* copy (hb_serialize_context_t *c, unsigned int data_length) const
  {
    TRACE_SERIALIZE (this);
    SBIXGlyph* new_glyph = c->start_embed<SBIXGlyph> ();
    if (unlikely (!new_glyph)) return_trace (nullptr);
    if (unlikely (!c->extend_min (new_glyph))) return_trace (nullptr);

    new_glyph->xOffset = xOffset;
    new_glyph->yOffset = yOffset;
    new_glyph->graphicType = graphicType;
    data.copy (c, data_length);
    return_trace (new_glyph);
  }

  HBINT16	xOffset;	/* The horizontal (x-axis) offset from the left
				 * edge of the graphic to the glyph’s origin.
				 * That is, the x-coordinate of the point on the
				 * baseline at the left edge of the glyph. */
  HBINT16	yOffset;	/* The vertical (y-axis) offset from the bottom
				 * edge of the graphic to the glyph’s origin.
				 * That is, the y-coordinate of the point on the
				 * baseline at the left edge of the glyph. */
  Tag		graphicType;	/* Indicates the format of the embedded graphic
				 * data: one of 'jpg ', 'png ' or 'tiff', or the
				 * special format 'dupe'. */
  UnsizedArrayOf<HBUINT8>
		data;		/* The actual embedded graphic data. The total
				 * length is inferred from sequential entries in
				 * the glyphDataOffsets array and the fixed size
				 * (8 bytes) of the preceding fields. */
  public:
  DEFINE_SIZE_ARRAY (8, data);
};

struct SBIXStrike
{
  static unsigned int get_size (unsigned num_glyphs)
  { return min_size + num_glyphs * HBUINT32::static_size; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  imageOffsetsZ.sanitize_shallow (c, c->get_num_glyphs () + 1));
  }

  hb_blob_t *get_glyph_blob (unsigned int  glyph_id,
			     hb_blob_t    *sbix_blob,
			     hb_tag_t      file_type,
			     int          *x_offset,
			     int          *y_offset,
			     unsigned int  num_glyphs,
			     unsigned int *strike_ppem) const
  {
    if (unlikely (!ppem)) return hb_blob_get_empty (); /* To get Null() object out of the way. */

    unsigned int retry_count = 8;
    unsigned int sbix_len = sbix_blob->length;
    unsigned int strike_offset = (const char *) this - (const char *) sbix_blob->data;
    assert (strike_offset < sbix_len);

  retry:
    if (unlikely (glyph_id >= num_glyphs ||
		  imageOffsetsZ[glyph_id + 1] <= imageOffsetsZ[glyph_id] ||
		  imageOffsetsZ[glyph_id + 1] - imageOffsetsZ[glyph_id] <= SBIXGlyph::min_size ||
		  (unsigned int) imageOffsetsZ[glyph_id + 1] > sbix_len - strike_offset))
      return hb_blob_get_empty ();

    unsigned int glyph_offset = strike_offset + (unsigned int) imageOffsetsZ[glyph_id] + SBIXGlyph::min_size;
    unsigned int glyph_length = imageOffsetsZ[glyph_id + 1] - imageOffsetsZ[glyph_id] - SBIXGlyph::min_size;

    const SBIXGlyph *glyph = &(this+imageOffsetsZ[glyph_id]);

    if (glyph->graphicType == HB_TAG ('d','u','p','e'))
    {
      if (glyph_length >= 2)
      {
	glyph_id = *((HBUINT16 *) &glyph->data);
	if (retry_count--)
	  goto retry;
      }
      return hb_blob_get_empty ();
    }

    if (unlikely (file_type != glyph->graphicType))
      return hb_blob_get_empty ();

    if (strike_ppem) *strike_ppem = ppem;
    if (x_offset) *x_offset = glyph->xOffset;
    if (y_offset) *y_offset = glyph->yOffset;
    return hb_blob_create_sub_blob (sbix_blob, glyph_offset, glyph_length);
  }

  bool subset (hb_subset_context_t *c, unsigned int available_len) const
  {
    TRACE_SUBSET (this);
    unsigned int num_output_glyphs = c->plan->num_output_glyphs ();

    auto* out = c->serializer->start_embed<SBIXStrike> ();
    if (unlikely (!out)) return_trace (false);
    auto snap = c->serializer->snapshot ();
    if (unlikely (!c->serializer->extend (*out, num_output_glyphs + 1))) return_trace (false);
    out->ppem = ppem;
    out->resolution = resolution;
    HBUINT32 head;
    head = get_size (num_output_glyphs + 1);

    bool has_glyphs = false;
    for (unsigned new_gid = 0; new_gid < num_output_glyphs; new_gid++)
    {
      hb_codepoint_t old_gid;
      if (!c->plan->old_gid_for_new_gid (new_gid, &old_gid) ||
	  unlikely (imageOffsetsZ[old_gid].is_null () ||
		    imageOffsetsZ[old_gid + 1].is_null () ||
		    imageOffsetsZ[old_gid + 1] <= imageOffsetsZ[old_gid] ||
		    imageOffsetsZ[old_gid + 1] - imageOffsetsZ[old_gid] <= SBIXGlyph::min_size) ||
		    (unsigned int) imageOffsetsZ[old_gid + 1] > available_len)
      {
	out->imageOffsetsZ[new_gid] = head;
	continue;
      }
      has_glyphs = true;
      unsigned int delta = imageOffsetsZ[old_gid + 1] - imageOffsetsZ[old_gid];
      unsigned int glyph_data_length = delta - SBIXGlyph::min_size;
      if (!(this+imageOffsetsZ[old_gid]).copy (c->serializer, glyph_data_length))
	return_trace (false);
      out->imageOffsetsZ[new_gid] = head;
      head += delta;
    }
    if (has_glyphs)
      out->imageOffsetsZ[num_output_glyphs] = head;
    else
      c->serializer->revert (snap);
    return_trace (has_glyphs);
  }

  public:
  HBUINT16	ppem;		/* The PPEM size for which this strike was designed. */
  HBUINT16	resolution;	/* The device pixel density (in PPI) for which this
				 * strike was designed. (E.g., 96 PPI, 192 PPI.) */
  protected:
  UnsizedArrayOf<LOffsetTo<SBIXGlyph>>
		imageOffsetsZ;	/* Offset from the beginning of the strike data header
				 * to bitmap data for an individual glyph ID. */
  public:
  DEFINE_SIZE_ARRAY (4, imageOffsetsZ);
};

struct sbix
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_sbix;

  bool has_data () const { return version; }

  const SBIXStrike &get_strike (unsigned int i) const { return this+strikes[i]; }

  struct accelerator_t
  {
    void init (hb_face_t *face)
    {
      table = hb_sanitize_context_t ().reference_table<sbix> (face);
      num_glyphs = face->get_num_glyphs ();
    }
    void fini () { table.destroy (); }

    bool has_data () const { return table->has_data (); }

    bool get_extents (hb_font_t          *font,
		      hb_codepoint_t      glyph,
		      hb_glyph_extents_t *extents) const
    {
      /* We only support PNG right now, and following function checks type. */
      return get_png_extents (font, glyph, extents);
    }

    hb_blob_t *reference_png (hb_font_t      *font,
			      hb_codepoint_t  glyph_id,
			      int            *x_offset,
			      int            *y_offset,
			      unsigned int   *available_ppem) const
    {
      return choose_strike (font).get_glyph_blob (glyph_id, table.get_blob (),
						  HB_TAG ('p','n','g',' '),
						  x_offset, y_offset,
						  num_glyphs, available_ppem);
    }

    private:

    const SBIXStrike &choose_strike (hb_font_t *font) const
    {
      unsigned count = table->strikes.len;
      if (unlikely (!count))
	return Null (SBIXStrike);

      unsigned int requested_ppem = hb_max (font->x_ppem, font->y_ppem);
      if (!requested_ppem)
	requested_ppem = 1<<30; /* Choose largest strike. */
      /* TODO Add DPI sensitivity as well? */
      unsigned int best_i = 0;
      unsigned int best_ppem = table->get_strike (0).ppem;

      for (unsigned int i = 1; i < count; i++)
      {
	unsigned int ppem = (table->get_strike (i)).ppem;
	if ((requested_ppem <= ppem && ppem < best_ppem) ||
	    (requested_ppem > best_ppem && ppem > best_ppem))
	{
	  best_i = i;
	  best_ppem = ppem;
	}
      }

      return table->get_strike (best_i);
    }

    struct PNGHeader
    {
      HBUINT8	signature[8];
      struct
      {
	struct
	{
	  HBUINT32	length;
	  Tag		type;
	}		header;
	HBUINT32	width;
	HBUINT32	height;
	HBUINT8		bitDepth;
	HBUINT8		colorType;
	HBUINT8		compressionMethod;
	HBUINT8		filterMethod;
	HBUINT8		interlaceMethod;
      } IHDR;

      public:
      DEFINE_SIZE_STATIC (29);
    };

    bool get_png_extents (hb_font_t          *font,
			  hb_codepoint_t      glyph,
			  hb_glyph_extents_t *extents) const
    {
      /* Following code is safe to call even without data.
       * But faster to short-circuit. */
      if (!has_data ())
	return false;

      int x_offset = 0, y_offset = 0;
      unsigned int strike_ppem = 0;
      hb_blob_t *blob = reference_png (font, glyph, &x_offset, &y_offset, &strike_ppem);

      const PNGHeader &png = *blob->as<PNGHeader>();

      extents->x_bearing = x_offset;
      extents->y_bearing = png.IHDR.height + y_offset;
      extents->width     = png.IHDR.width;
      extents->height    = -1 * png.IHDR.height;

      /* Convert to font units. */
      if (strike_ppem)
      {
	float scale = font->face->get_upem () / (float) strike_ppem;
	extents->x_bearing = font->em_scalef_x (extents->x_bearing * scale);
	extents->y_bearing = font->em_scalef_y (extents->y_bearing * scale);
	extents->width = font->em_scalef_x (extents->width * scale);
	extents->height = font->em_scalef_y (extents->height * scale);
      }
      else
      {
	extents->x_bearing = font->em_scale_x (extents->x_bearing);
	extents->y_bearing = font->em_scale_y (extents->y_bearing);
	extents->width = font->em_scale_x (extents->width);
	extents->height = font->em_scale_y (extents->height);
      }

      hb_blob_destroy (blob);

      return strike_ppem;
    }

    private:
    hb_blob_ptr_t<sbix> table;

    unsigned int num_glyphs;
  };

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  version >= 1 &&
			  strikes.sanitize (c, this)));
  }

  bool
  add_strike (hb_subset_context_t *c, unsigned i) const
  {
    if (strikes[i].is_null () || c->source_blob->length < (unsigned) strikes[i])
      return false;

    return (this+strikes[i]).subset (c, c->source_blob->length - (unsigned) strikes[i]);
  }

  bool serialize_strike_offsets (hb_subset_context_t *c) const
  {
    TRACE_SERIALIZE (this);

    auto *out = c->serializer->start_embed<LOffsetLArrayOf<SBIXStrike>> ();
    if (unlikely (!out)) return_trace (false);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    hb_vector_t<LOffsetTo<SBIXStrike>*> new_strikes;
    hb_vector_t<hb_serialize_context_t::objidx_t> objidxs;
    for (int i = strikes.len - 1; i >= 0; --i)
    {
      auto* o = out->serialize_append (c->serializer);
      if (unlikely (!o)) return_trace (false);
      *o = 0;
      auto snap = c->serializer->snapshot ();
      c->serializer->push ();
      bool ret = add_strike (c, i);
      if (!ret)
      {
	c->serializer->pop_discard ();
	out->pop ();
	c->serializer->revert (snap);
      }
      else
      {
	objidxs.push (c->serializer->pop_pack ());
	new_strikes.push (o);
      }
    }
    for (unsigned int i = 0; i < new_strikes.length; ++i)
      c->serializer->add_link (*new_strikes[i], objidxs[new_strikes.length - 1 - i]);

    return_trace (true);
  }

  bool subset (hb_subset_context_t* c) const
  {
    TRACE_SUBSET (this);

    sbix *sbix_prime = c->serializer->start_embed<sbix> ();
    if (unlikely (!sbix_prime)) return_trace (false);
    if (unlikely (!c->serializer->embed (this->version))) return_trace (false);
    if (unlikely (!c->serializer->embed (this->flags))) return_trace (false);

    return_trace (serialize_strike_offsets (c));
  }

  protected:
  HBUINT16	version;	/* Table version number — set to 1 */
  HBUINT16	flags;		/* Bit 0: Set to 1. Bit 1: Draw outlines.
				 * Bits 2 to 15: reserved (set to 0). */
  LOffsetLArrayOf<SBIXStrike>
		strikes;	/* Offsets from the beginning of the 'sbix'
				 * table to data for each individual bitmap strike. */
  public:
  DEFINE_SIZE_ARRAY (8, strikes);
};

struct sbix_accelerator_t : sbix::accelerator_t {};

} /* namespace OT */

#endif /* HB_OT_COLOR_SBIX_TABLE_HH */
