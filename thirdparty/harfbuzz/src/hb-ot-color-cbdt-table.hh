/*
 * Copyright © 2016  Google, Inc.
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
 * Google Author(s): Seigo Nonaka
 */

#ifndef HB_OT_COLOR_CBDT_TABLE_HH
#define HB_OT_COLOR_CBDT_TABLE_HH

#include "hb-open-type-private.hh"

/*
 * CBLC -- Color Bitmap Location
 * https://docs.microsoft.com/en-us/typography/opentype/spec/cblc
 * https://docs.microsoft.com/en-us/typography/opentype/spec/eblc
 * CBDT -- Color Bitmap Data
 * https://docs.microsoft.com/en-us/typography/opentype/spec/cbdt
 * https://docs.microsoft.com/en-us/typography/opentype/spec/ebdt
 */
#define HB_OT_TAG_CBLC HB_TAG('C','B','L','C')
#define HB_OT_TAG_CBDT HB_TAG('C','B','D','T')


namespace OT {

struct SmallGlyphMetrics
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  inline void get_extents (hb_glyph_extents_t *extents) const
  {
    extents->x_bearing = bearingX;
    extents->y_bearing = bearingY;
    extents->width = width;
    extents->height = -height;
  }

  HBUINT8	height;
  HBUINT8	width;
  HBINT8	bearingX;
  HBINT8	bearingY;
  HBUINT8	advance;
  public:
  DEFINE_SIZE_STATIC(5);
};

struct BigGlyphMetrics : SmallGlyphMetrics
{
  HBINT8	vertBearingX;
  HBINT8	vertBearingY;
  HBUINT8	vertAdvance;
  public:
  DEFINE_SIZE_STATIC(8);
};

struct SBitLineMetrics
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBINT8	ascender;
  HBINT8	decender;
  HBUINT8	widthMax;
  HBINT8	caretSlopeNumerator;
  HBINT8	caretSlopeDenominator;
  HBINT8	caretOffset;
  HBINT8	minOriginSB;
  HBINT8	minAdvanceSB;
  HBINT8	maxBeforeBL;
  HBINT8	minAfterBL;
  HBINT8	padding1;
  HBINT8	padding2;
  public:
  DEFINE_SIZE_STATIC(12);
};


/*
 * Index Subtables.
 */

struct IndexSubtableHeader
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16	indexFormat;
  HBUINT16	imageFormat;
  HBUINT32	imageDataOffset;
  public:
  DEFINE_SIZE_STATIC(8);
};

template <typename OffsetType>
struct IndexSubtableFormat1Or3
{
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int glyph_count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  c->check_array (offsetArrayZ, offsetArrayZ[0].static_size, glyph_count + 1));
  }

  bool get_image_data (unsigned int idx,
		       unsigned int *offset,
		       unsigned int *length) const
  {
    if (unlikely (offsetArrayZ[idx + 1] <= offsetArrayZ[idx]))
      return false;

    *offset = header.imageDataOffset + offsetArrayZ[idx];
    *length = offsetArrayZ[idx + 1] - offsetArrayZ[idx];
    return true;
  }

  IndexSubtableHeader	header;
  Offset<OffsetType>	offsetArrayZ[VAR];
  public:
  DEFINE_SIZE_ARRAY(8, offsetArrayZ);
};

struct IndexSubtableFormat1 : IndexSubtableFormat1Or3<HBUINT32> {};
struct IndexSubtableFormat3 : IndexSubtableFormat1Or3<HBUINT16> {};

struct IndexSubtable
{
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int glyph_count) const
  {
    TRACE_SANITIZE (this);
    if (!u.header.sanitize (c)) return_trace (false);
    switch (u.header.indexFormat) {
    case 1: return_trace (u.format1.sanitize (c, glyph_count));
    case 3: return_trace (u.format3.sanitize (c, glyph_count));
    default:return_trace (true);
    }
  }

  inline bool get_extents (hb_glyph_extents_t *extents) const
  {
    switch (u.header.indexFormat) {
    case 2: case 5: /* TODO */
    case 1: case 3: case 4: /* Variable-metrics formats do not have metrics here. */
    default:return (false);
    }
  }

  bool get_image_data (unsigned int idx,
		       unsigned int *offset,
		       unsigned int *length,
		       unsigned int *format) const
  {
    *format = u.header.imageFormat;
    switch (u.header.indexFormat) {
    case 1: return u.format1.get_image_data (idx, offset, length);
    case 3: return u.format3.get_image_data (idx, offset, length);
    default: return false;
    }
  }

  protected:
  union {
  IndexSubtableHeader	header;
  IndexSubtableFormat1	format1;
  IndexSubtableFormat3	format3;
  /* TODO: Format 2, 4, 5. */
  } u;
  public:
  DEFINE_SIZE_UNION (8, header);
};

struct IndexSubtableRecord
{
  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  firstGlyphIndex <= lastGlyphIndex &&
		  offsetToSubtable.sanitize (c, this, lastGlyphIndex - firstGlyphIndex + 1));
  }

  inline bool get_extents (hb_glyph_extents_t *extents) const
  {
    return (this+offsetToSubtable).get_extents (extents);
  }

  bool get_image_data (unsigned int gid,
		       unsigned int *offset,
		       unsigned int *length,
		       unsigned int *format) const
  {
    if (gid < firstGlyphIndex || gid > lastGlyphIndex)
    {
      return false;
    }
    return (this+offsetToSubtable).get_image_data (gid - firstGlyphIndex,
						   offset, length, format);
  }

  GlyphID			firstGlyphIndex;
  GlyphID			lastGlyphIndex;
  LOffsetTo<IndexSubtable>	offsetToSubtable;
  public:
  DEFINE_SIZE_STATIC(8);
};

struct IndexSubtableArray
{
  friend struct CBDT;

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_array (&indexSubtablesZ, indexSubtablesZ[0].static_size, count)))
      return_trace (false);
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!indexSubtablesZ[i].sanitize (c, this)))
	return_trace (false);
    return_trace (true);
  }

  public:
  const IndexSubtableRecord* find_table (hb_codepoint_t glyph, unsigned int numTables) const
  {
    for (unsigned int i = 0; i < numTables; ++i)
    {
      unsigned int firstGlyphIndex = indexSubtablesZ[i].firstGlyphIndex;
      unsigned int lastGlyphIndex = indexSubtablesZ[i].lastGlyphIndex;
      if (firstGlyphIndex <= glyph && glyph <= lastGlyphIndex) {
        return &indexSubtablesZ[i];
      }
    }
    return nullptr;
  }

  protected:
  IndexSubtableRecord	indexSubtablesZ[VAR];
  public:
  DEFINE_SIZE_ARRAY(0, indexSubtablesZ);
};

struct BitmapSizeTable
{
  friend struct CBLC;
  friend struct CBDT;

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  indexSubtableArrayOffset.sanitize (c, base, numberOfIndexSubtables) &&
		  c->check_range (&(base+indexSubtableArrayOffset), indexTablesSize) &&
		  horizontal.sanitize (c) &&
		  vertical.sanitize (c));
  }

  const IndexSubtableRecord *find_table (hb_codepoint_t glyph, const void *base) const
  {
    return (base+indexSubtableArrayOffset).find_table (glyph, numberOfIndexSubtables);
  }

  protected:
  LOffsetTo<IndexSubtableArray>
			indexSubtableArrayOffset;
  HBUINT32		indexTablesSize;
  HBUINT32		numberOfIndexSubtables;
  HBUINT32		colorRef;
  SBitLineMetrics	horizontal;
  SBitLineMetrics	vertical;
  GlyphID		startGlyphIndex;
  GlyphID		endGlyphIndex;
  HBUINT8		ppemX;
  HBUINT8		ppemY;
  HBUINT8		bitDepth;
  HBINT8		flags;
  public:
  DEFINE_SIZE_STATIC(48);
};


/*
 * Glyph Bitmap Data Formats.
 */

struct GlyphBitmapDataFormat17
{
  SmallGlyphMetrics	glyphMetrics;
  LArrayOf<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY(9, data);
};

struct GlyphBitmapDataFormat18
{
  BigGlyphMetrics	glyphMetrics;
  LArrayOf<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY(12, data);
};

struct GlyphBitmapDataFormat19
{
  LArrayOf<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY(4, data);
};

struct CBLC
{
  friend struct CBDT;

  static const hb_tag_t tableTag = HB_OT_TAG_CBLC;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 2 || version.major == 3) &&
		  sizeTables.sanitize (c, this));
  }

  protected:
  const IndexSubtableRecord *find_table (hb_codepoint_t glyph,
					 unsigned int *x_ppem, unsigned int *y_ppem) const
  {
    /* TODO: Make it possible to select strike. */

    unsigned int count = sizeTables.len;
    for (uint32_t i = 0; i < count; ++i)
    {
      unsigned int startGlyphIndex = sizeTables.arrayZ[i].startGlyphIndex;
      unsigned int endGlyphIndex = sizeTables.arrayZ[i].endGlyphIndex;
      if (startGlyphIndex <= glyph && glyph <= endGlyphIndex)
      {
	*x_ppem = sizeTables[i].ppemX;
	*y_ppem = sizeTables[i].ppemY;
	return sizeTables[i].find_table (glyph, this);
      }
    }

    return nullptr;
  }

  protected:
  FixedVersion<>		version;
  LArrayOf<BitmapSizeTable>	sizeTables;
  public:
  DEFINE_SIZE_ARRAY(8, sizeTables);
};

struct CBDT
{
  static const hb_tag_t tableTag = HB_OT_TAG_CBDT;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 2 || version.major == 3));
  }

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      upem = hb_face_get_upem (face);

      cblc_blob = hb_sanitize_context_t().reference_table<CBLC> (face);
      cbdt_blob = hb_sanitize_context_t().reference_table<CBDT> (face);
      cbdt_len = hb_blob_get_length (cbdt_blob);

      if (hb_blob_get_length (cblc_blob) == 0) {
	cblc = nullptr;
	cbdt = nullptr;
	return;  /* Not a bitmap font. */
      }
      cblc = cblc_blob->as<CBLC> ();
      cbdt = cbdt_blob->as<CBDT> ();

    }

    inline void fini (void)
    {
      hb_blob_destroy (this->cblc_blob);
      hb_blob_destroy (this->cbdt_blob);
    }

    inline bool get_extents (hb_codepoint_t glyph, hb_glyph_extents_t *extents) const
    {
      unsigned int x_ppem = upem, y_ppem = upem; /* TODO Use font ppem if available. */

      if (!cblc)
	return false;  // Not a color bitmap font.

      const IndexSubtableRecord *subtable_record = this->cblc->find_table(glyph, &x_ppem, &y_ppem);
      if (!subtable_record || !x_ppem || !y_ppem)
	return false;

      if (subtable_record->get_extents (extents))
	return true;

      unsigned int image_offset = 0, image_length = 0, image_format = 0;
      if (!subtable_record->get_image_data (glyph, &image_offset, &image_length, &image_format))
	return false;

      {
	if (unlikely (image_offset > cbdt_len || cbdt_len - image_offset < image_length))
	  return false;

	switch (image_format)
	{
	  case 17: {
	    if (unlikely (image_length < GlyphBitmapDataFormat17::min_size))
	      return false;

	    const GlyphBitmapDataFormat17& glyphFormat17 =
		StructAtOffset<GlyphBitmapDataFormat17> (this->cbdt, image_offset);
	    glyphFormat17.glyphMetrics.get_extents (extents);
	  }
	  break;
	  default:
	    // TODO: Support other image formats.
	    return false;
	}
      }

      /* Convert to the font units. */
      extents->x_bearing *= upem / (float) x_ppem;
      extents->y_bearing *= upem / (float) y_ppem;
      extents->width *= upem / (float) x_ppem;
      extents->height *= upem / (float) y_ppem;

      return true;
    }

    inline void dump (void (*callback) (const uint8_t* data, unsigned int length,
        unsigned int group, unsigned int gid)) const
    {
      if (!cblc)
	return;  // Not a color bitmap font.

      for (unsigned int i = 0; i < cblc->sizeTables.len; ++i)
      {
        const BitmapSizeTable &sizeTable = cblc->sizeTables[i];
        const IndexSubtableArray &subtable_array = cblc+sizeTable.indexSubtableArrayOffset;
        for (unsigned int j = 0; j < sizeTable.numberOfIndexSubtables; ++j)
        {
          const IndexSubtableRecord &subtable_record = subtable_array.indexSubtablesZ[j];
          for (unsigned int gid = subtable_record.firstGlyphIndex;
                gid <= subtable_record.lastGlyphIndex; ++gid)
          {
            unsigned int image_offset = 0, image_length = 0, image_format = 0;

            if (!subtable_record.get_image_data (gid,
                  &image_offset, &image_length, &image_format))
              continue;

            switch (image_format)
            {
            case 17: {
              const GlyphBitmapDataFormat17& glyphFormat17 =
                StructAtOffset<GlyphBitmapDataFormat17> (this->cbdt, image_offset);
              callback ((const uint8_t *) &glyphFormat17.data.arrayZ,
                glyphFormat17.data.len, i, gid);
            }
            break;
            case 18: {
              const GlyphBitmapDataFormat18& glyphFormat18 =
                StructAtOffset<GlyphBitmapDataFormat18> (this->cbdt, image_offset);
              callback ((const uint8_t *) &glyphFormat18.data.arrayZ,
                glyphFormat18.data.len, i, gid);
            }
            break;
            case 19: {
              const GlyphBitmapDataFormat19& glyphFormat19 =
                StructAtOffset<GlyphBitmapDataFormat19> (this->cbdt, image_offset);
              callback ((const uint8_t *) &glyphFormat19.data.arrayZ,
                glyphFormat19.data.len, i, gid);
            }
            break;
            default:
              continue;
            }
          }
        }
      }
    }

    private:
    hb_blob_t *cblc_blob;
    hb_blob_t *cbdt_blob;
    const CBLC *cblc;
    const CBDT *cbdt;

    unsigned int cbdt_len;
    unsigned int upem;
  };


  protected:
  FixedVersion<>	version;
  HBUINT8		dataZ[VAR];
  public:
  DEFINE_SIZE_ARRAY(4, dataZ);
};

} /* namespace OT */

#endif /* HB_OT_COLOR_CBDT_TABLE_HH */
