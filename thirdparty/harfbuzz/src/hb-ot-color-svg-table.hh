/*
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
 */

#ifndef HB_OT_COLOR_SVG_TABLE_HH
#define HB_OT_COLOR_SVG_TABLE_HH

#include "hb-open-type-private.hh"

/*
 * SVG -- SVG (Scalable Vector Graphics)
 * https://docs.microsoft.com/en-us/typography/opentype/spec/svg
 */

#define HB_OT_TAG_SVG HB_TAG('S','V','G',' ')


namespace OT {


struct SVGDocumentIndexEntry
{
  friend struct SVG;

  inline bool sanitize (hb_sanitize_context_t *c, const void* base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  (base+svgDoc).sanitize (c, svgDocLength));
  }

  protected:
  HBUINT16	startGlyphID;	/* The first glyph ID in the range described by
				 * this index entry. */
  HBUINT16	endGlyphID;	/* The last glyph ID in the range described by
				 * this index entry. Must be >= startGlyphID. */
  LOffsetTo<UnsizedArrayOf<HBUINT8> >
		svgDoc;		/* Offset from the beginning of the SVG Document Index
				 * to an SVG document. Must be non-zero. */
  HBUINT32 svgDocLength;	/* Length of the SVG document.
				 * Must be non-zero. */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct SVGDocumentIndex
{
  friend struct SVG;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  entries.sanitize (c, this));
  }

  protected:
  ArrayOf<SVGDocumentIndexEntry>
		entries;	/* Array of SVG Document Index Entries. */
  public:
  DEFINE_SIZE_ARRAY (2, entries);
};

struct SVG
{
  static const hb_tag_t tableTag = HB_OT_TAG_SVG;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (this+svgDocIndex).sanitize (c)));
  }

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      svg_blob = hb_sanitize_context_t().reference_table<SVG> (face);
      svg_len = hb_blob_get_length (svg_blob);
      svg = svg_blob->as<SVG> ();
    }

    inline void fini (void)
    {
      hb_blob_destroy (svg_blob);
    }

    inline void
    dump (void (*callback) (const uint8_t* data, unsigned int length,
			    unsigned int start_glyph, unsigned int end_glyph)) const
    {
      const SVGDocumentIndex &index = svg+svg->svgDocIndex;
      const ArrayOf<SVGDocumentIndexEntry> &entries = index.entries;
      for (unsigned int i = 0; i < entries.len; ++i)
      {
	const SVGDocumentIndexEntry &entry = entries[i];
	callback ((const uint8_t*) &entry.svgDoc (&index), entry.svgDocLength,
						  entry.startGlyphID, entry.endGlyphID);
      }
    }

    private:
    hb_blob_t *svg_blob;
    const SVG *svg;

    unsigned int svg_len;
  };

  protected:
  HBUINT16	version;	/* Table version (starting at 0). */
  LOffsetTo<SVGDocumentIndex>
		svgDocIndex;	/* Offset (relative to the start of the SVG table) to the
				 * SVG Documents Index. Must be non-zero. */
  HBUINT32	reserved;	/* Set to 0. */
  public:
  DEFINE_SIZE_STATIC (10);
};

} /* namespace OT */


#endif /* HB_OT_COLOR_SVG_TABLE_HH */
