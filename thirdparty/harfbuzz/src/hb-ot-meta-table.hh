/*
 * Copyright © 2019  Ebrahim Byagowi
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
 */

#ifndef HB_OT_META_TABLE_HH
#define HB_OT_META_TABLE_HH

#include "hb-open-type.hh"

/*
 * meta -- Metadata Table
 * https://docs.microsoft.com/en-us/typography/opentype/spec/meta
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6meta.html
 */
#define HB_OT_TAG_meta HB_TAG ('m','e','t','a')


namespace OT {


struct DataMap
{
  int cmp (hb_tag_t a) const { return tag.cmp (a); }

  hb_tag_t get_tag () const { return tag; }

  hb_blob_t *reference_entry (hb_blob_t *meta_blob) const
  { return hb_blob_create_sub_blob (meta_blob, dataZ, dataLength); }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  dataZ.sanitize (c, base, dataLength)));
  }

  protected:
  Tag		tag;		/* A tag indicating the type of metadata. */
  LNNOffsetTo<UnsizedArrayOf<HBUINT8>>
		dataZ;		/* Offset in bytes from the beginning of the
				 * metadata table to the data for this tag. */
  HBUINT32	dataLength;	/* Length of the data. The data is not required to
				 * be padded to any byte boundary. */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct meta
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_meta;

  struct accelerator_t
  {
    void init (hb_face_t *face)
    { table = hb_sanitize_context_t ().reference_table<meta> (face); }
    void fini () { table.destroy (); }

    hb_blob_t *reference_entry (hb_tag_t tag) const
    { return table->dataMaps.lsearch (tag).reference_entry (table.get_blob ()); }

    unsigned int get_entries (unsigned int      start_offset,
			      unsigned int     *count,
			      hb_ot_meta_tag_t *entries) const
    {
      if (count)
      {
	+ table->dataMaps.sub_array (start_offset, count)
	| hb_map (&DataMap::get_tag)
	| hb_map ([](hb_tag_t tag) { return (hb_ot_meta_tag_t) tag; })
	| hb_sink (hb_array (entries, *count))
	;
      }
      return table->dataMaps.len;
    }

    private:
    hb_blob_ptr_t<meta> table;
  };

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  version == 1 &&
			  dataMaps.sanitize (c, this)));
  }

  protected:
  HBUINT32	version;	/* Version number of the metadata table — set to 1. */
  HBUINT32	flags;		/* Flags — currently unused; set to 0. */
  HBUINT32	dataOffset;
				/* Per Apple specification:
				 * Offset from the beginning of the table to the data.
				 * Per OT specification:
				 * Reserved. Not used; should be set to 0. */
  LArrayOf<DataMap>
		dataMaps;/* Array of data map records. */
  public:
  DEFINE_SIZE_ARRAY (16, dataMaps);
};

struct meta_accelerator_t : meta::accelerator_t {};

} /* namespace OT */


#endif /* HB_OT_META_TABLE_HH */
