/*
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_GLYF_TABLE_HH
#define HB_OT_GLYF_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-ot-head-table.hh"


namespace OT {


/*
 * loca -- Index to Location
 */

#define HB_OT_TAG_loca HB_TAG('l','o','c','a')


struct loca
{
  friend struct glyf;

  static const hb_tag_t tableTag = HB_OT_TAG_loca;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (true);
  }

  protected:
  HBUINT8		dataX[VAR];		/* Location data. */
  DEFINE_SIZE_ARRAY (0, dataX);
};


/*
 * glyf -- TrueType Glyph Data
 */

#define HB_OT_TAG_glyf HB_TAG('g','l','y','f')


struct glyf
{
  static const hb_tag_t tableTag = HB_OT_TAG_glyf;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    /* We don't check for anything specific here.  The users of the
     * struct do all the hard work... */
    return_trace (true);
  }

  struct GlyphHeader
  {
    HBINT16		numberOfContours;	/* If the number of contours is
					   * greater than or equal to zero,
					   * this is a simple glyph; if negative,
					   * this is a composite glyph. */
    FWORD		xMin;			/* Minimum x for coordinate data. */
    FWORD		yMin;			/* Minimum y for coordinate data. */
    FWORD		xMax;			/* Maximum x for coordinate data. */
    FWORD		yMax;			/* Maximum y for coordinate data. */

    DEFINE_SIZE_STATIC (10);
  };

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      hb_blob_t *head_blob = Sanitizer<head>().sanitize (face->reference_table (HB_OT_TAG_head));
      const head *head_table = Sanitizer<head>::lock_instance (head_blob);
      if ((unsigned int) head_table->indexToLocFormat > 1 || head_table->glyphDataFormat != 0)
      {
	/* Unknown format.  Leave num_glyphs=0, that takes care of disabling us. */
	hb_blob_destroy (head_blob);
	return;
      }
      short_offset = 0 == head_table->indexToLocFormat;
      hb_blob_destroy (head_blob);

      loca_blob = Sanitizer<loca>().sanitize (face->reference_table (HB_OT_TAG_loca));
      loca_table = Sanitizer<loca>::lock_instance (loca_blob);
      glyf_blob = Sanitizer<glyf>().sanitize (face->reference_table (HB_OT_TAG_glyf));
      glyf_table = Sanitizer<glyf>::lock_instance (glyf_blob);

      num_glyphs = MAX (1u, hb_blob_get_length (loca_blob) / (short_offset ? 2 : 4)) - 1;
      glyf_len = hb_blob_get_length (glyf_blob);
    }

    inline void fini (void)
    {
      hb_blob_destroy (loca_blob);
      hb_blob_destroy (glyf_blob);
    }

    inline bool get_extents (hb_codepoint_t glyph,
			     hb_glyph_extents_t *extents) const
    {
      if (unlikely (glyph >= num_glyphs))
	return false;

      unsigned int start_offset, end_offset;
      if (short_offset)
      {
        const HBUINT16 *offsets = (const HBUINT16 *) loca_table->dataX;
	start_offset = 2 * offsets[glyph];
	end_offset   = 2 * offsets[glyph + 1];
      }
      else
      {
        const HBUINT32 *offsets = (const HBUINT32 *) loca_table->dataX;
	start_offset = offsets[glyph];
	end_offset   = offsets[glyph + 1];
      }

      if (start_offset > end_offset || end_offset > glyf_len)
	return false;

      if (end_offset - start_offset < GlyphHeader::static_size)
	return true; /* Empty glyph; zero extents. */

      const GlyphHeader &glyph_header = StructAtOffset<GlyphHeader> (glyf_table, start_offset);

      extents->x_bearing = MIN (glyph_header.xMin, glyph_header.xMax);
      extents->y_bearing = MAX (glyph_header.yMin, glyph_header.yMax);
      extents->width     = MAX (glyph_header.xMin, glyph_header.xMax) - extents->x_bearing;
      extents->height    = MIN (glyph_header.yMin, glyph_header.yMax) - extents->y_bearing;

      return true;
    }

    private:
    bool short_offset;
    unsigned int num_glyphs;
    const loca *loca_table;
    const glyf *glyf_table;
    hb_blob_t *loca_blob;
    hb_blob_t *glyf_blob;
    unsigned int glyf_len;
  };

  protected:
  HBUINT8		dataX[VAR];		/* Glyphs data. */

  DEFINE_SIZE_ARRAY (0, dataX);
};

} /* namespace OT */


#endif /* HB_OT_GLYF_TABLE_HH */
