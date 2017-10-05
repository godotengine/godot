/*
 * Copyright © 2018  Ebrahim Byagowi
 * Copyright © 2018  Google, Inc.
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

#ifndef HB_AAT_LAYOUT_KERX_TABLE_HH
#define HB_AAT_LAYOUT_KERX_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-aat-layout-common-private.hh"
#include "hb-aat-layout-ankr-table.hh"

/*
 * kerx -- Extended Kerning
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6kerx.html
 */
#define HB_AAT_TAG_kerx HB_TAG('k','e','r','x')


namespace AAT {

using namespace OT;


struct KerxFormat0Records
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  GlyphID	left;
  GlyphID	right;
  FWORD		value;
  public:
  DEFINE_SIZE_STATIC (6);
};

struct KerxSubTableFormat0
{
  // TODO(ebraminio) Enable when we got suitable BinSearchArrayOf
  // inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right) const
  // {
  //   hb_glyph_pair_t pair = {left, right};
  //   int i = pairs.bsearch (pair);
  //   if (i == -1)
  //     return 0;
  //   return pairs[i].get_kerning ();
  // }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  recordsZ.sanitize (c, nPairs)));
  }

  protected:
  // TODO(ebraminio): A custom version of "BinSearchArrayOf<KerxPair> pairs;" is
  // needed here to use HBUINT32 instead
  HBUINT32	nPairs;		/* The number of kerning pairs in this subtable */
  HBUINT32	searchRange;	/* The largest power of two less than or equal to the value of nPairs,
				 * multiplied by the size in bytes of an entry in the subtable. */
  HBUINT32	entrySelector;	/* This is calculated as log2 of the largest power of two less
				 * than or equal to the value of nPairs. */
  HBUINT32	rangeShift;	/* The value of nPairs minus the largest power of two less than or equal to nPairs. */
  UnsizedArrayOf<KerxFormat0Records>
		recordsZ;	/* VAR=nPairs */
  public:
  DEFINE_SIZE_ARRAY (16, recordsZ);
};

struct KerxSubTableFormat1
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  stateHeader.sanitize (c)));
  }

  protected:
  StateTable<HBUINT16>		stateHeader;
  LOffsetTo<ArrayOf<HBUINT16> >	valueTable;
  public:
  DEFINE_SIZE_STATIC (20);
};

// TODO(ebraminio): Maybe this can be replaced with Lookup<HBUINT16>?
struct KerxClassTable
{
  inline unsigned int get_class (hb_codepoint_t g) const { return classes[g - firstGlyph]; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (firstGlyph.sanitize (c) &&
			  classes.sanitize (c)));
  }

  protected:
  HBUINT16		firstGlyph;	/* First glyph in class range. */
  ArrayOf<HBUINT16>	classes;	/* Glyph classes. */
  public:
  DEFINE_SIZE_ARRAY (4, classes);
};

struct KerxSubTableFormat2
{
  inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right, const char *end) const
  {
    unsigned int l = (this+leftClassTable).get_class (left);
    unsigned int r = (this+leftClassTable).get_class (left);
    unsigned int offset = l * rowWidth + r * sizeof (FWORD);
    const FWORD *arr = &(this+array);
    if (unlikely ((const void *) arr < (const void *) this || (const void *) arr >= (const void *) end))
      return 0;
    const FWORD *v = &StructAtOffset<FWORD> (arr, offset);
    if (unlikely ((const void *) v < (const void *) arr || (const void *) (v + 1) > (const void *) end))
      return 0;
    return *v;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  rowWidth.sanitize (c) &&
			  leftClassTable.sanitize (c, this) &&
			  rightClassTable.sanitize (c, this) &&
			  array.sanitize (c, this)));
  }

  protected:
  HBUINT32	rowWidth;	/* The width, in bytes, of a row in the table. */
  LOffsetTo<KerxClassTable>
		leftClassTable;	/* Offset from beginning of this subtable to
				 * left-hand class table. */
  LOffsetTo<KerxClassTable>
		rightClassTable;/* Offset from beginning of this subtable to
				 * right-hand class table. */
  LOffsetTo<FWORD>
		array;		/* Offset from beginning of this subtable to
				 * the start of the kerning array. */
  public:
  DEFINE_SIZE_STATIC (16);
};

struct KerxSubTableFormat4
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  rowWidth.sanitize (c) &&
			  leftClassTable.sanitize (c, this) &&
			  rightClassTable.sanitize (c, this) &&
			  array.sanitize (c, this)));
  }

  protected:
  HBUINT32	rowWidth;	/* The width, in bytes, of a row in the table. */
  LOffsetTo<KerxClassTable>
		leftClassTable;	/* Offset from beginning of this subtable to
				 * left-hand class table. */
  LOffsetTo<KerxClassTable>
		rightClassTable;/* Offset from beginning of this subtable to
				 * right-hand class table. */
  LOffsetTo<FWORD>
		array;		/* Offset from beginning of this subtable to
				 * the start of the kerning array. */
  public:
  DEFINE_SIZE_STATIC (16);
};

struct KerxSubTableFormat6
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  rowIndexTable.sanitize (c, this) &&
			  columnIndexTable.sanitize (c, this) &&
			  kerningArray.sanitize (c, this) &&
			  kerningVector.sanitize (c, this)));
  }

  protected:
  HBUINT32	flags;
  HBUINT16	rowCount;
  HBUINT16	columnCount;
  LOffsetTo<Lookup<HBUINT16> >	rowIndexTable;
  LOffsetTo<Lookup<HBUINT16> >	columnIndexTable;
  LOffsetTo<Lookup<HBUINT16> >	kerningArray;
  LOffsetTo<Lookup<HBUINT16> >	kerningVector;
  public:
  DEFINE_SIZE_STATIC (24);
};

enum coverage_flags_t
{
  COVERAGE_VERTICAL_FLAG	= 0x80u,
  COVERAGE_CROSSSTREAM_FLAG	= 0x40u,
  COVERAGE_VARIATION_FLAG	= 0x20u,
  COVERAGE_PROCESS_DIRECTION	= 0x10u,
};

struct KerxTable
{
  inline bool apply (hb_aat_apply_context_t *c, const AAT::ankr *ankr) const
  {
    TRACE_APPLY (this);
    /* TODO */
    return_trace (false);
  }

  inline unsigned int get_size (void) const { return length; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    switch (format) {
    case 0: return u.format0.sanitize (c);
    case 1: return u.format1.sanitize (c);
    case 2: return u.format2.sanitize (c);
    case 4: return u.format4.sanitize (c);
    case 6: return u.format6.sanitize (c);
    default:return_trace (false);
    }
  }

protected:
  HBUINT32	length;
  HBUINT8	coverage;
  HBUINT16	unused;
  HBUINT8	format;
  HBUINT32	tupleIndex;
  union {
  KerxSubTableFormat0	format0;
  KerxSubTableFormat1	format1;
  KerxSubTableFormat2	format2;
  KerxSubTableFormat4	format4;
  KerxSubTableFormat6	format6;
  } u;
public:
  DEFINE_SIZE_MIN (12);
};

struct SubtableGlyphCoverageArray
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT32	length;
  HBUINT32	coverage;
  HBUINT32	tupleCount;
  public:
  DEFINE_SIZE_STATIC (12);
};

struct kerx
{
  static const hb_tag_t tableTag = HB_AAT_TAG_kerx;

  inline bool apply (hb_aat_apply_context_t *c, const AAT::ankr *ankr) const
  {
    TRACE_APPLY (this);
    const KerxTable &table = StructAfter<KerxTable> (*this);
    return_trace (table.apply (c, ankr));
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(c->check_struct (this))))
     return_trace (false);

    /* TODO: Something like `morx`s ChainSubtable should be done here instead */
    const KerxTable *table = &StructAfter<KerxTable> (*this);
    if (unlikely (!(table->sanitize (c))))
      return_trace (false);

    for (unsigned int i = 0; i < nTables - 1; ++i)
    {
      table = &StructAfter<KerxTable> (*table);
      if (unlikely (!(table->sanitize (c))))
        return_trace (false);
    }

    // If version is less than 3, we are done here; otherwise better to check footer also
    if (version < 3)
      return_trace (true);

    // TODO: Investigate why this just work on some fonts no matter of version
    // const SubtableGlyphCoverageArray &footer =
    //   StructAfter<SubtableGlyphCoverageArray> (*table);
    // return_trace (footer.sanitize (c));

    return_trace (true);
  }

  protected:
  HBUINT16		version;
  HBUINT16		padding;
  HBUINT32		nTables;
/*KerxTable tablesZ[VAR]; XXX ArrayOf??? */
/*SubtableGlyphCoverageArray coverage_array;*/
  public:
  DEFINE_SIZE_STATIC (8);
};

} /* namespace AAT */


#endif /* HB_AAT_LAYOUT_KERX_TABLE_HH */
