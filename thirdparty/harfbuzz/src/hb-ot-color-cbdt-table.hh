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
 * Google Author(s): Seigo Nonaka, Calder Kitagawa
 */

#ifndef HB_OT_COLOR_CBDT_TABLE_HH
#define HB_OT_COLOR_CBDT_TABLE_HH

#include "hb-open-type.hh"

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

struct cblc_bitmap_size_subset_context_t
{
  const char *cbdt;
  unsigned int cbdt_length;
  hb_vector_t<char> *cbdt_prime;
  unsigned int size;		/* INOUT
				 *  Input: old size of IndexSubtable
				 *  Output: new size of IndexSubtable
				 */
  unsigned int num_tables;	/* INOUT
				 *  Input: old number of subtables.
				 *  Output: new number of subtables.
				 */
  hb_codepoint_t start_glyph;	/* OUT */
  hb_codepoint_t end_glyph;	/* OUT */
};

static inline bool
_copy_data_to_cbdt (hb_vector_t<char> *cbdt_prime,
		    const void        *data,
		    unsigned           length)
{
  unsigned int new_len = cbdt_prime->length + length;
  if (unlikely (!cbdt_prime->alloc (new_len))) return false;
  memcpy (cbdt_prime->arrayZ + cbdt_prime->length, data, length);
  cbdt_prime->length = new_len;
  return true;
}

struct SmallGlyphMetrics
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  void get_extents (hb_font_t *font, hb_glyph_extents_t *extents) const
  {
    extents->x_bearing = font->em_scale_x (bearingX);
    extents->y_bearing = font->em_scale_y (bearingY);
    extents->width = font->em_scale_x (width);
    extents->height = font->em_scale_y (-static_cast<int>(height));
  }

  HBUINT8	height;
  HBUINT8	width;
  HBINT8	bearingX;
  HBINT8	bearingY;
  HBUINT8	advance;
  public:
  DEFINE_SIZE_STATIC (5);
};

struct BigGlyphMetrics : SmallGlyphMetrics
{
  HBINT8	vertBearingX;
  HBINT8	vertBearingY;
  HBUINT8	vertAdvance;
  public:
  DEFINE_SIZE_STATIC (8);
};

struct SBitLineMetrics
{
  bool sanitize (hb_sanitize_context_t *c) const
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
  DEFINE_SIZE_STATIC (12);
};


/*
 * Index Subtables.
 */

struct IndexSubtableHeader
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16	indexFormat;
  HBUINT16	imageFormat;
  HBUINT32	imageDataOffset;
  public:
  DEFINE_SIZE_STATIC (8);
};

template <typename OffsetType>
struct IndexSubtableFormat1Or3
{
  bool sanitize (hb_sanitize_context_t *c, unsigned int glyph_count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  offsetArrayZ.sanitize (c, glyph_count + 1));
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

  bool add_offset (hb_serialize_context_t *c,
		   unsigned int offset,
		   unsigned int *size /* OUT (accumulated) */)
  {
    TRACE_SERIALIZE (this);
    Offset<OffsetType> embedded_offset;
    embedded_offset = offset;
    *size += sizeof (OffsetType);
    auto *o = c->embed (embedded_offset);
    return_trace ((bool) o);
  }

  IndexSubtableHeader	header;
  UnsizedArrayOf<Offset<OffsetType>>
			offsetArrayZ;
  public:
  DEFINE_SIZE_ARRAY (8, offsetArrayZ);
};

struct IndexSubtableFormat1 : IndexSubtableFormat1Or3<HBUINT32> {};
struct IndexSubtableFormat3 : IndexSubtableFormat1Or3<HBUINT16> {};

struct IndexSubtable
{
  bool sanitize (hb_sanitize_context_t *c, unsigned int glyph_count) const
  {
    TRACE_SANITIZE (this);
    if (!u.header.sanitize (c)) return_trace (false);
    switch (u.header.indexFormat)
    {
    case 1: return_trace (u.format1.sanitize (c, glyph_count));
    case 3: return_trace (u.format3.sanitize (c, glyph_count));
    default:return_trace (true);
    }
  }

  bool
  finish_subtable (hb_serialize_context_t *c,
		   unsigned int cbdt_prime_len,
		   unsigned int num_glyphs,
		   unsigned int *size /* OUT (accumulated) */)
  {
    TRACE_SERIALIZE (this);

    unsigned int local_offset = cbdt_prime_len - u.header.imageDataOffset;
    switch (u.header.indexFormat)
    {
    case 1: return_trace (u.format1.add_offset (c, local_offset, size));
    case 3: {
      if (!u.format3.add_offset (c, local_offset, size))
	return_trace (false);
      if (!(num_glyphs & 0x01))  // Pad to 32-bit alignment if needed.
	return_trace (u.format3.add_offset (c, 0, size));
      return_trace (true);
    }
    // TODO: implement 2, 4, 5.
    case 2: case 4:  // No-op.
    case 5:  // Pad to 32-bit aligned.
    default: return_trace (false);
    }
  }

  bool
  fill_missing_glyphs (hb_serialize_context_t *c,
		       unsigned int cbdt_prime_len,
		       unsigned int num_missing,
		       unsigned int *size /* OUT (accumulated) */,
		       unsigned int *num_glyphs /* OUT (accumulated) */)
  {
    TRACE_SERIALIZE (this);

    unsigned int local_offset = cbdt_prime_len - u.header.imageDataOffset;
    switch (u.header.indexFormat)
    {
    case 1: {
      for (unsigned int i = 0; i < num_missing; i++)
      {
	if (unlikely (!u.format1.add_offset (c, local_offset, size)))
	  return_trace (false);
	*num_glyphs += 1;
      }
      return_trace (true);
    }
    case 3: {
      for (unsigned int i = 0; i < num_missing; i++)
      {
	if (unlikely (!u.format3.add_offset (c, local_offset, size)))
	  return_trace (false);
	*num_glyphs += 1;
      }
      return_trace (true);
    }
    // TODO: implement 2, 4, 5.
    case 2:  // Add empty space in cbdt_prime?.
    case 4: case 5:  // No-op as sparse is supported.
    default: return_trace (false);
    }
  }

  bool
  copy_glyph_at_idx (hb_serialize_context_t *c, unsigned int idx,
		     const char *cbdt, unsigned int cbdt_length,
		     hb_vector_t<char> *cbdt_prime /* INOUT */,
		     IndexSubtable *subtable_prime /* INOUT */,
		     unsigned int *size /* OUT (accumulated) */) const
  {
    TRACE_SERIALIZE (this);

    unsigned int offset, length, format;
    if (unlikely (!get_image_data (idx, &offset, &length, &format))) return_trace (false);
    if (unlikely (offset > cbdt_length || cbdt_length - offset < length)) return_trace (false);

    auto *header_prime = subtable_prime->get_header ();
    unsigned int new_local_offset = cbdt_prime->length - (unsigned int) header_prime->imageDataOffset;
    if (unlikely (!_copy_data_to_cbdt (cbdt_prime, cbdt + offset, length))) return_trace (false);

    return_trace (subtable_prime->add_offset (c, new_local_offset, size));
  }

  bool
  add_offset (hb_serialize_context_t *c, unsigned int local_offset,
	      unsigned int *size /* OUT (accumulated) */)
  {
    TRACE_SERIALIZE (this);
    switch (u.header.indexFormat)
    {
    case 1: return_trace (u.format1.add_offset (c, local_offset, size));
    case 3: return_trace (u.format3.add_offset (c, local_offset, size));
    // TODO: Implement tables 2, 4, 5
    case 2:  // Should be a no-op.
    case 4: case 5:  // Handle sparse cases.
    default: return_trace (false);
    }
  }

  bool get_extents (hb_glyph_extents_t *extents HB_UNUSED) const
  {
    switch (u.header.indexFormat)
    {
    case 2: case 5: /* TODO */
    case 1: case 3: case 4: /* Variable-metrics formats do not have metrics here. */
    default:return (false);
    }
  }

  bool
  get_image_data (unsigned int idx, unsigned int *offset,
		  unsigned int *length, unsigned int *format) const
  {
    *format = u.header.imageFormat;
    switch (u.header.indexFormat)
    {
    case 1: return u.format1.get_image_data (idx, offset, length);
    case 3: return u.format3.get_image_data (idx, offset, length);
    default: return false;
    }
  }

  const IndexSubtableHeader* get_header () const { return &u.header; }

  void populate_header (unsigned index_format,
			unsigned image_format,
			unsigned int image_data_offset,
			unsigned int *size)
  {
    u.header.indexFormat = index_format;
    u.header.imageFormat = image_format;
    u.header.imageDataOffset = image_data_offset;
    switch (u.header.indexFormat)
    {
    case 1: *size += IndexSubtableFormat1::min_size; break;
    case 3: *size += IndexSubtableFormat3::min_size; break;
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
  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  firstGlyphIndex <= lastGlyphIndex &&
		  offsetToSubtable.sanitize (c, base, lastGlyphIndex - firstGlyphIndex + 1));
  }

  const IndexSubtable* get_subtable (const void *base) const
  {
    return &(base+offsetToSubtable);
  }

  bool add_new_subtable (hb_subset_context_t* c,
			 cblc_bitmap_size_subset_context_t *bitmap_size_context,
			 IndexSubtableRecord *record,
			 const hb_vector_t<hb_pair_t<hb_codepoint_t, const IndexSubtableRecord*>> *lookup, /* IN */
			 const void *base,
			 unsigned int *start /* INOUT */) const
  {
    TRACE_SERIALIZE (this);

    auto *subtable = c->serializer->start_embed<IndexSubtable> ();
    if (unlikely (!subtable)) return_trace (false);
    if (unlikely (!c->serializer->extend_min (subtable))) return_trace (false);

    auto *old_subtable = get_subtable (base);
    auto *old_header = old_subtable->get_header ();

    subtable->populate_header (old_header->indexFormat,
			       old_header->imageFormat,
			       bitmap_size_context->cbdt_prime->length,
			       &bitmap_size_context->size);

    unsigned int num_glyphs = 0;
    bool early_exit = false;
    for (unsigned int i = *start; i < lookup->length; i++)
    {
      hb_codepoint_t new_gid = (*lookup)[i].first;
      const IndexSubtableRecord *next_record = (*lookup)[i].second;
      const IndexSubtable *next_subtable = next_record->get_subtable (base);
      auto *next_header = next_subtable->get_header ();
      if (next_header != old_header)
      {
	*start = i;
	early_exit = true;
	break;
      }
      unsigned int num_missing = record->add_glyph_for_subset (new_gid);
      if (unlikely (!subtable->fill_missing_glyphs (c->serializer,
						    bitmap_size_context->cbdt_prime->length,
						    num_missing,
						    &bitmap_size_context->size,
						    &num_glyphs)))
	return_trace (false);

      hb_codepoint_t old_gid = 0;
      c->plan->old_gid_for_new_gid (new_gid, &old_gid);
      if (old_gid < next_record->firstGlyphIndex)
	return_trace (false);

      unsigned int old_idx = (unsigned int) old_gid - next_record->firstGlyphIndex;
      if (unlikely (!next_subtable->copy_glyph_at_idx (c->serializer,
						       old_idx,
						       bitmap_size_context->cbdt,
						       bitmap_size_context->cbdt_length,
						       bitmap_size_context->cbdt_prime,
						       subtable,
						       &bitmap_size_context->size)))
	return_trace (false);
      num_glyphs += 1;
    }
    if (!early_exit)
      *start = lookup->length;
    if (unlikely (!subtable->finish_subtable (c->serializer,
					      bitmap_size_context->cbdt_prime->length,
					      num_glyphs,
					      &bitmap_size_context->size)))
      return_trace (false);
    return_trace (true);
  }

  bool add_new_record (hb_subset_context_t *c,
		       cblc_bitmap_size_subset_context_t *bitmap_size_context,
		       const hb_vector_t<hb_pair_t<hb_codepoint_t, const IndexSubtableRecord*>> *lookup, /* IN */
		       const void *base,
		       unsigned int *start, /* INOUT */
		       hb_vector_t<IndexSubtableRecord>* records /* INOUT */) const
  {
    TRACE_SERIALIZE (this);
    auto snap = c->serializer->snapshot ();
    unsigned int old_size = bitmap_size_context->size;
    unsigned int old_cbdt_prime_length = bitmap_size_context->cbdt_prime->length;

    // Set to invalid state to indicate filling glyphs is not yet started.
    if (unlikely (!c->serializer->check_success (records->resize (records->length + 1))))
      return_trace (false);

    (*records)[records->length - 1].firstGlyphIndex = 1;
    (*records)[records->length - 1].lastGlyphIndex = 0;
    bitmap_size_context->size += IndexSubtableRecord::min_size;

    c->serializer->push ();

    if (unlikely (!add_new_subtable (c, bitmap_size_context, &((*records)[records->length - 1]), lookup, base, start)))
    {
      c->serializer->pop_discard ();
      c->serializer->revert (snap);
      bitmap_size_context->cbdt_prime->shrink (old_cbdt_prime_length);
      bitmap_size_context->size = old_size;
      records->resize (records->length - 1);
      return_trace (false);
    }

    bitmap_size_context->num_tables += 1;
    return_trace (true);
  }

  unsigned int add_glyph_for_subset (hb_codepoint_t gid)
  {
    if (firstGlyphIndex > lastGlyphIndex)
    {
      firstGlyphIndex = gid;
      lastGlyphIndex = gid;
      return 0;
    }
    // TODO maybe assert? this shouldn't occur.
    if (lastGlyphIndex > gid)
      return 0;
    unsigned int num_missing = (unsigned int) (gid - lastGlyphIndex - 1);
    lastGlyphIndex = gid;
    return num_missing;
  }

  bool get_extents (hb_glyph_extents_t *extents, const void *base) const
  { return (base+offsetToSubtable).get_extents (extents); }

  bool get_image_data (unsigned int  gid,
		       const void   *base,
		       unsigned int *offset,
		       unsigned int *length,
		       unsigned int *format) const
  {
    if (gid < firstGlyphIndex || gid > lastGlyphIndex) return false;
    return (base+offsetToSubtable).get_image_data (gid - firstGlyphIndex,
						   offset, length, format);
  }

  HBGlyphID			firstGlyphIndex;
  HBGlyphID			lastGlyphIndex;
  Offset32To<IndexSubtable>	offsetToSubtable;
  public:
  DEFINE_SIZE_STATIC (8);
};

struct IndexSubtableArray
{
  friend struct CBDT;

  bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    return_trace (indexSubtablesZ.sanitize (c, count, this));
  }

  void
  build_lookup (hb_subset_context_t *c, cblc_bitmap_size_subset_context_t *bitmap_size_context,
		hb_vector_t<hb_pair_t<hb_codepoint_t,
		const IndexSubtableRecord*>> *lookup /* OUT */) const
  {
    bool start_glyph_is_set = false;
    for (hb_codepoint_t new_gid = 0; new_gid < c->plan->num_output_glyphs (); new_gid++)
    {
      hb_codepoint_t old_gid;
      if (unlikely (!c->plan->old_gid_for_new_gid (new_gid, &old_gid))) continue;

      const IndexSubtableRecord* record = find_table (old_gid, bitmap_size_context->num_tables);
      if (unlikely (!record)) continue;

      // Don't add gaps to the lookup. The best way to determine if a glyph is a
      // gap is that it has no image data.
      unsigned int offset, length, format;
      if (unlikely (!record->get_image_data (old_gid, this, &offset, &length, &format))) continue;

      lookup->push (hb_pair_t<hb_codepoint_t, const IndexSubtableRecord*> (new_gid, record));

      if (!start_glyph_is_set)
      {
	bitmap_size_context->start_glyph = new_gid;
	start_glyph_is_set = true;
      }

      bitmap_size_context->end_glyph = new_gid;
    }
  }

  bool
  subset (hb_subset_context_t *c,
	  cblc_bitmap_size_subset_context_t *bitmap_size_context) const
  {
    TRACE_SUBSET (this);

    auto *dst = c->serializer->start_embed<IndexSubtableArray> ();
    if (unlikely (!dst)) return_trace (false);

    hb_vector_t<hb_pair_t<hb_codepoint_t, const IndexSubtableRecord*>> lookup;
    build_lookup (c, bitmap_size_context, &lookup);
    if (unlikely (!c->serializer->propagate_error (lookup)))
      return false;

    bitmap_size_context->size = 0;
    bitmap_size_context->num_tables = 0;
    hb_vector_t<IndexSubtableRecord> records;
    for (unsigned int start = 0; start < lookup.length;)
    {
      if (unlikely (!lookup[start].second->add_new_record (c, bitmap_size_context, &lookup, this, &start, &records)))
      {
	// Discard any leftover pushes to the serializer from successful records.
	for (unsigned int i = 0; i < records.length; i++)
	  c->serializer->pop_discard ();
	return_trace (false);
      }
    }

    /* Workaround to ensure offset ordering is from least to greatest when
     * resolving links. */
    hb_vector_t<hb_serialize_context_t::objidx_t> objidxs;
    for (unsigned int i = 0; i < records.length; i++)
      objidxs.push (c->serializer->pop_pack ());
    for (unsigned int i = 0; i < records.length; i++)
    {
      IndexSubtableRecord* record = c->serializer->embed (records[i]);
      if (unlikely (!record)) return_trace (false);
      c->serializer->add_link (record->offsetToSubtable, objidxs[records.length - 1 - i]);
    }
    return_trace (true);
  }

  public:
  const IndexSubtableRecord* find_table (hb_codepoint_t glyph, unsigned int numTables) const
  {
    for (unsigned int i = 0; i < numTables; ++i)
    {
      unsigned int firstGlyphIndex = indexSubtablesZ[i].firstGlyphIndex;
      unsigned int lastGlyphIndex = indexSubtablesZ[i].lastGlyphIndex;
      if (firstGlyphIndex <= glyph && glyph <= lastGlyphIndex)
	return &indexSubtablesZ[i];
    }
    return nullptr;
  }

  protected:
  UnsizedArrayOf<IndexSubtableRecord>	indexSubtablesZ;
};

struct BitmapSizeTable
{
  friend struct CBLC;
  friend struct CBDT;

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  indexSubtableArrayOffset.sanitize (c, base, numberOfIndexSubtables) &&
		  horizontal.sanitize (c) &&
		  vertical.sanitize (c));
  }

  const IndexSubtableRecord *
  find_table (hb_codepoint_t glyph, const void *base, const void **out_base) const
  {
    *out_base = &(base+indexSubtableArrayOffset);
    return (base+indexSubtableArrayOffset).find_table (glyph, numberOfIndexSubtables);
  }

  bool
  subset (hb_subset_context_t *c, const void *base,
	  const char *cbdt, unsigned int cbdt_length,
	  hb_vector_t<char> *cbdt_prime /* INOUT */) const
  {
    TRACE_SUBSET (this);
    auto *out_table = c->serializer->embed (this);
    if (unlikely (!out_table)) return_trace (false);

    cblc_bitmap_size_subset_context_t bitmap_size_context;
    bitmap_size_context.cbdt = cbdt;
    bitmap_size_context.cbdt_length = cbdt_length;
    bitmap_size_context.cbdt_prime = cbdt_prime;
    bitmap_size_context.size = indexTablesSize;
    bitmap_size_context.num_tables = numberOfIndexSubtables;
    bitmap_size_context.start_glyph = 1;
    bitmap_size_context.end_glyph = 0;

    if (!out_table->indexSubtableArrayOffset.serialize_subset (c,
							       indexSubtableArrayOffset,
							       base,
							       &bitmap_size_context))
      return_trace (false);
    if (!bitmap_size_context.size ||
	!bitmap_size_context.num_tables ||
	bitmap_size_context.start_glyph > bitmap_size_context.end_glyph)
      return_trace (false);

    out_table->indexTablesSize = bitmap_size_context.size;
    out_table->numberOfIndexSubtables = bitmap_size_context.num_tables;
    out_table->startGlyphIndex = bitmap_size_context.start_glyph;
    out_table->endGlyphIndex = bitmap_size_context.end_glyph;
    return_trace (true);
  }

  protected:
  NNOffset32To<IndexSubtableArray>
			indexSubtableArrayOffset;
  HBUINT32		indexTablesSize;
  HBUINT32		numberOfIndexSubtables;
  HBUINT32		colorRef;
  SBitLineMetrics	horizontal;
  SBitLineMetrics	vertical;
  HBGlyphID		startGlyphIndex;
  HBGlyphID		endGlyphIndex;
  HBUINT8		ppemX;
  HBUINT8		ppemY;
  HBUINT8		bitDepth;
  HBINT8		flags;
  public:
  DEFINE_SIZE_STATIC (48);
};


/*
 * Glyph Bitmap Data Formats.
 */

struct GlyphBitmapDataFormat17
{
  SmallGlyphMetrics	glyphMetrics;
  Array32Of<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY (9, data);
};

struct GlyphBitmapDataFormat18
{
  BigGlyphMetrics	glyphMetrics;
  Array32Of<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY (12, data);
};

struct GlyphBitmapDataFormat19
{
  Array32Of<HBUINT8>	data;
  public:
  DEFINE_SIZE_ARRAY (4, data);
};

struct CBLC
{
  friend struct CBDT;

  static constexpr hb_tag_t tableTag = HB_OT_TAG_CBLC;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 2 || version.major == 3) &&
		  sizeTables.sanitize (c, this));
  }

  static bool
  sink_cbdt (hb_subset_context_t *c, hb_vector_t<char>* cbdt_prime)
  {
    hb_blob_t *cbdt_prime_blob = hb_blob_create (cbdt_prime->arrayZ,
						 cbdt_prime->length,
						 HB_MEMORY_MODE_WRITABLE,
						 cbdt_prime->arrayZ,
						 hb_free);
    cbdt_prime->init ();  // Leak arrayZ to the blob.
    bool ret = c->plan->add_table (HB_OT_TAG_CBDT, cbdt_prime_blob);
    hb_blob_destroy (cbdt_prime_blob);
    return ret;
  }

  bool
  subset_size_table (hb_subset_context_t *c, const BitmapSizeTable& table,
		     const char *cbdt /* IN */, unsigned int cbdt_length,
		     CBLC *cblc_prime /* INOUT */, hb_vector_t<char> *cbdt_prime /* INOUT */) const
  {
    TRACE_SUBSET (this);
    cblc_prime->sizeTables.len++;

    auto snap = c->serializer->snapshot ();
    auto cbdt_prime_len = cbdt_prime->length;

    if (!table.subset (c, this, cbdt, cbdt_length, cbdt_prime))
    {
      cblc_prime->sizeTables.len--;
      c->serializer->revert (snap);
      cbdt_prime->shrink (cbdt_prime_len);
      return_trace (false);
    }
    return_trace (true);
  }

  // Implemented in cc file as it depends on definition of CBDT.
  HB_INTERNAL bool subset (hb_subset_context_t *c) const;

  protected:
  const BitmapSizeTable &choose_strike (hb_font_t *font) const
  {
    unsigned count = sizeTables.len;
    if (unlikely (!count))
      return Null (BitmapSizeTable);

    unsigned int requested_ppem = hb_max (font->x_ppem, font->y_ppem);
    if (!requested_ppem)
      requested_ppem = 1<<30; /* Choose largest strike. */
    unsigned int best_i = 0;
    unsigned int best_ppem = hb_max (sizeTables[0].ppemX, sizeTables[0].ppemY);

    for (unsigned int i = 1; i < count; i++)
    {
      unsigned int ppem = hb_max (sizeTables[i].ppemX, sizeTables[i].ppemY);
      if ((requested_ppem <= ppem && ppem < best_ppem) ||
	  (requested_ppem > best_ppem && ppem > best_ppem))
      {
	best_i = i;
	best_ppem = ppem;
      }
    }

    return sizeTables[best_i];
  }

  protected:
  FixedVersion<>		version;
  Array32Of<BitmapSizeTable>	sizeTables;
  public:
  DEFINE_SIZE_ARRAY (8, sizeTables);
};

struct CBDT
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_CBDT;

  struct accelerator_t
  {
    void init (hb_face_t *face)
    {
      cblc = hb_sanitize_context_t ().reference_table<CBLC> (face);
      cbdt = hb_sanitize_context_t ().reference_table<CBDT> (face);

      upem = hb_face_get_upem (face);
    }

    void fini ()
    {
      this->cblc.destroy ();
      this->cbdt.destroy ();
    }

    bool
    get_extents (hb_font_t *font, hb_codepoint_t glyph, hb_glyph_extents_t *extents) const
    {
      const void *base;
      const BitmapSizeTable &strike = this->cblc->choose_strike (font);
      const IndexSubtableRecord *subtable_record = strike.find_table (glyph, cblc, &base);
      if (!subtable_record || !strike.ppemX || !strike.ppemY)
	return false;

      if (subtable_record->get_extents (extents, base))
	return true;

      unsigned int image_offset = 0, image_length = 0, image_format = 0;
      if (!subtable_record->get_image_data (glyph, base, &image_offset, &image_length, &image_format))
	return false;

      unsigned int cbdt_len = cbdt.get_length ();
      if (unlikely (image_offset > cbdt_len || cbdt_len - image_offset < image_length))
	return false;

      switch (image_format)
      {
      case 17: {
	if (unlikely (image_length < GlyphBitmapDataFormat17::min_size))
	  return false;
	auto &glyphFormat17 = StructAtOffset<GlyphBitmapDataFormat17> (this->cbdt, image_offset);
	glyphFormat17.glyphMetrics.get_extents (font, extents);
	break;
      }
      case 18: {
	if (unlikely (image_length < GlyphBitmapDataFormat18::min_size))
	  return false;
	auto &glyphFormat18 = StructAtOffset<GlyphBitmapDataFormat18> (this->cbdt, image_offset);
	glyphFormat18.glyphMetrics.get_extents (font, extents);
	break;
      }
      default: return false; /* TODO: Support other image formats. */
      }

      /* Convert to font units. */
      float x_scale = upem / (float) strike.ppemX;
      float y_scale = upem / (float) strike.ppemY;
      extents->x_bearing = roundf (extents->x_bearing * x_scale);
      extents->y_bearing = roundf (extents->y_bearing * y_scale);
      extents->width = roundf (extents->width * x_scale);
      extents->height = roundf (extents->height * y_scale);

      return true;
    }

    hb_blob_t*
    reference_png (hb_font_t *font, hb_codepoint_t glyph) const
    {
      const void *base;
      const BitmapSizeTable &strike = this->cblc->choose_strike (font);
      const IndexSubtableRecord *subtable_record = strike.find_table (glyph, cblc, &base);
      if (!subtable_record || !strike.ppemX || !strike.ppemY)
	return hb_blob_get_empty ();

      unsigned int image_offset = 0, image_length = 0, image_format = 0;
      if (!subtable_record->get_image_data (glyph, base, &image_offset, &image_length, &image_format))
	return hb_blob_get_empty ();

      unsigned int cbdt_len = cbdt.get_length ();
      if (unlikely (image_offset > cbdt_len || cbdt_len - image_offset < image_length))
	return hb_blob_get_empty ();

      switch (image_format)
      {
      case 17:
      {
	if (unlikely (image_length < GlyphBitmapDataFormat17::min_size))
	  return hb_blob_get_empty ();
	auto &glyphFormat17 = StructAtOffset<GlyphBitmapDataFormat17> (this->cbdt, image_offset);
	return hb_blob_create_sub_blob (cbdt.get_blob (),
					image_offset + GlyphBitmapDataFormat17::min_size,
					glyphFormat17.data.len);
      }
      case 18:
      {
	if (unlikely (image_length < GlyphBitmapDataFormat18::min_size))
	  return hb_blob_get_empty ();
	auto &glyphFormat18 = StructAtOffset<GlyphBitmapDataFormat18> (this->cbdt, image_offset);
	return hb_blob_create_sub_blob (cbdt.get_blob (),
					image_offset + GlyphBitmapDataFormat18::min_size,
					glyphFormat18.data.len);
      }
      case 19:
      {
	if (unlikely (image_length < GlyphBitmapDataFormat19::min_size))
	  return hb_blob_get_empty ();
	auto &glyphFormat19 = StructAtOffset<GlyphBitmapDataFormat19> (this->cbdt, image_offset);
	return hb_blob_create_sub_blob (cbdt.get_blob (),
					image_offset + GlyphBitmapDataFormat19::min_size,
					glyphFormat19.data.len);
      }
      default: return hb_blob_get_empty (); /* TODO: Support other image formats. */
      }
    }

    bool has_data () const { return cbdt.get_length (); }

    private:
    hb_blob_ptr_t<CBLC> cblc;
    hb_blob_ptr_t<CBDT> cbdt;

    unsigned int upem;
  };

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 2 || version.major == 3));
  }

  protected:
  FixedVersion<>		version;
  UnsizedArrayOf<HBUINT8>	dataZ;
  public:
  DEFINE_SIZE_ARRAY (4, dataZ);
};

inline bool
CBLC::subset (hb_subset_context_t *c) const
{
  TRACE_SUBSET (this);

  auto *cblc_prime = c->serializer->start_embed<CBLC> ();

  // Use a vector as a secondary buffer as the tables need to be built in parallel.
  hb_vector_t<char> cbdt_prime;

  if (unlikely (!cblc_prime)) return_trace (false);
  if (unlikely (!c->serializer->extend_min (cblc_prime))) return_trace (false);
  cblc_prime->version = version;

  hb_blob_t* cbdt_blob = hb_sanitize_context_t ().reference_table<CBDT> (c->plan->source);
  unsigned int cbdt_length;
  CBDT* cbdt = (CBDT *) hb_blob_get_data (cbdt_blob, &cbdt_length);
  if (unlikely (cbdt_length < CBDT::min_size))
  {
    hb_blob_destroy (cbdt_blob);
    return_trace (false);
  }
  _copy_data_to_cbdt (&cbdt_prime, cbdt, CBDT::min_size);

  for (const BitmapSizeTable& table : + sizeTables.iter ())
    subset_size_table (c, table, (const char *) cbdt, cbdt_length, cblc_prime, &cbdt_prime);

  hb_blob_destroy (cbdt_blob);

  return_trace (CBLC::sink_cbdt (c, &cbdt_prime));
}

struct CBDT_accelerator_t : CBDT::accelerator_t {};

} /* namespace OT */

#endif /* HB_OT_COLOR_CBDT_TABLE_HH */
