/*
 * Copyright © 2017  Google, Inc.
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

#ifndef HB_OT_KERN_TABLE_HH
#define HB_OT_KERN_TABLE_HH

#include "hb-open-type-private.hh"

namespace OT {


/*
 * kern -- Kerning
 */

#define HB_OT_TAG_kern HB_TAG('k','e','r','n')

struct hb_glyph_pair_t
{
  hb_codepoint_t left;
  hb_codepoint_t right;
};

struct KernPair
{
  inline int get_kerning (void) const
  { return value; }

  inline int cmp (const hb_glyph_pair_t &o) const
  {
    int ret = left.cmp (o.left);
    if (ret) return ret;
    return right.cmp (o.right);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  GlyphID	left;
  GlyphID	right;
  FWORD		value;
  public:
  DEFINE_SIZE_STATIC (6);
};

struct KernSubTableFormat0
{
  inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right) const
  {
    hb_glyph_pair_t pair = {left, right};
    int i = pairs.bsearch (pair);
    if (i == -1)
      return 0;
    return pairs[i].get_kerning ();
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (pairs.sanitize (c));
  }

  protected:
  BinSearchArrayOf<KernPair> pairs;	/* Array of kerning pairs. */
  public:
  DEFINE_SIZE_ARRAY (8, pairs);
};

struct KernClassTable
{
  inline unsigned int get_class (hb_codepoint_t g) const { return classes[g - firstGlyph]; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (firstGlyph.sanitize (c) && classes.sanitize (c));
  }

  protected:
  HBUINT16		firstGlyph;	/* First glyph in class range. */
  ArrayOf<HBUINT16>	classes;	/* Glyph classes. */
  public:
  DEFINE_SIZE_ARRAY (4, classes);
};

struct KernSubTableFormat2
{
  inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right, const char *end) const
  {
    unsigned int l = (this+leftClassTable).get_class (left);
    unsigned int r = (this+rightClassTable).get_class (right);
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
    return_trace (rowWidth.sanitize (c) &&
		  leftClassTable.sanitize (c, this) &&
		  rightClassTable.sanitize (c, this) &&
		  array.sanitize (c, this));
  }

  protected:
  HBUINT16	rowWidth;	/* The width, in bytes, of a row in the table. */
  OffsetTo<KernClassTable>
		leftClassTable;	/* Offset from beginning of this subtable to
				 * left-hand class table. */
  OffsetTo<KernClassTable>
		rightClassTable;/* Offset from beginning of this subtable to
				 * right-hand class table. */
  OffsetTo<FWORD>
		array;		/* Offset from beginning of this subtable to
				 * the start of the kerning array. */
  public:
  DEFINE_SIZE_MIN (8);
};

struct KernSubTable
{
  inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right, const char *end, unsigned int format) const
  {
    switch (format) {
    case 0: return u.format0.get_kerning (left, right);
    case 2: return u.format2.get_kerning (left, right, end);
    default:return 0;
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int format) const
  {
    TRACE_SANITIZE (this);
    switch (format) {
    case 0: return_trace (u.format0.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  union {
  KernSubTableFormat0	format0;
  KernSubTableFormat2	format2;
  } u;
  public:
  DEFINE_SIZE_MIN (0);
};


template <typename T>
struct KernSubTableWrapper
{
  /* https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern */
  inline const T* thiz (void) const { return static_cast<const T *> (this); }

  inline bool is_horizontal (void) const
  { return (thiz()->coverage & T::COVERAGE_CHECK_FLAGS) == T::COVERAGE_CHECK_HORIZONTAL; }

  inline bool is_override (void) const
  { return bool (thiz()->coverage & T::COVERAGE_OVERRIDE_FLAG); }

  inline int get_kerning (hb_codepoint_t left, hb_codepoint_t right, const char *end) const
  { return thiz()->subtable.get_kerning (left, right, end, thiz()->format); }

  inline int get_h_kerning (hb_codepoint_t left, hb_codepoint_t right, const char *end) const
  { return is_horizontal () ? get_kerning (left, right, end) : 0; }

  inline unsigned int get_size (void) const { return thiz()->length; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (thiz()) &&
		  thiz()->length >= thiz()->min_size &&
		  c->check_array (thiz(), 1, thiz()->length) &&
		  thiz()->subtable.sanitize (c, thiz()->format));
  }
};

template <typename T>
struct KernTable
{
  /* https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern */
  inline const T* thiz (void) const { return static_cast<const T *> (this); }

  inline int get_h_kerning (hb_codepoint_t left, hb_codepoint_t right, unsigned int table_length) const
  {
    int v = 0;
    const typename T::SubTableWrapper *st = CastP<typename T::SubTableWrapper> (thiz()->data);
    unsigned int count = thiz()->nTables;
    for (unsigned int i = 0; i < count; i++)
    {
      if (st->is_override ())
        v = 0;
      v += st->get_h_kerning (left, right, table_length + (const char *) this);
      st = &StructAfter<typename T::SubTableWrapper> (*st);
    }
    return v;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (thiz()) ||
		  thiz()->version != T::VERSION))
      return_trace (false);

    const typename T::SubTableWrapper *st = CastP<typename T::SubTableWrapper> (thiz()->data);
    unsigned int count = thiz()->nTables;
    for (unsigned int i = 0; i < count; i++)
    {
      if (unlikely (!st->sanitize (c)))
	return_trace (false);
      st = &StructAfter<typename T::SubTableWrapper> (*st);
    }

    return_trace (true);
  }
};

struct KernOT : KernTable<KernOT>
{
  friend struct KernTable<KernOT>;

  static const uint16_t VERSION = 0x0000u;

  struct SubTableWrapper : KernSubTableWrapper<SubTableWrapper>
  {
    friend struct KernSubTableWrapper<SubTableWrapper>;

    enum coverage_flags_t {
      COVERAGE_DIRECTION_FLAG	= 0x01u,
      COVERAGE_MINIMUM_FLAG	= 0x02u,
      COVERAGE_CROSSSTREAM_FLAG	= 0x04u,
      COVERAGE_OVERRIDE_FLAG	= 0x08u,

      COVERAGE_VARIATION_FLAG	= 0x00u, /* Not supported. */

      COVERAGE_CHECK_FLAGS	= 0x07u,
      COVERAGE_CHECK_HORIZONTAL	= 0x01u
    };

    protected:
    HBUINT16	versionZ;	/* Unused. */
    HBUINT16	length;		/* Length of the subtable (including this header). */
    HBUINT8	format;		/* Subtable format. */
    HBUINT8	coverage;	/* Coverage bits. */
    KernSubTable subtable;	/* Subtable data. */
    public:
    DEFINE_SIZE_MIN (6);
  };

  protected:
  HBUINT16	version;	/* Version--0x0000u */
  HBUINT16	nTables;	/* Number of subtables in the kerning table. */
  HBUINT8		data[VAR];
  public:
  DEFINE_SIZE_ARRAY (4, data);
};

struct KernAAT : KernTable<KernAAT>
{
  friend struct KernTable<KernAAT>;

  static const uint32_t VERSION = 0x00010000u;

  struct SubTableWrapper : KernSubTableWrapper<SubTableWrapper>
  {
    friend struct KernSubTableWrapper<SubTableWrapper>;

    enum coverage_flags_t {
      COVERAGE_DIRECTION_FLAG	= 0x80u,
      COVERAGE_CROSSSTREAM_FLAG	= 0x40u,
      COVERAGE_VARIATION_FLAG	= 0x20u,

      COVERAGE_OVERRIDE_FLAG	= 0x00u, /* Not supported. */

      COVERAGE_CHECK_FLAGS	= 0xE0u,
      COVERAGE_CHECK_HORIZONTAL	= 0x00u
    };

    protected:
    HBUINT32	length;		/* Length of the subtable (including this header). */
    HBUINT8	coverage;	/* Coverage bits. */
    HBUINT8	format;		/* Subtable format. */
    HBUINT16	tupleIndex;	/* The tuple index (used for variations fonts).
				 * This value specifies which tuple this subtable covers. */
    KernSubTable subtable;	/* Subtable data. */
    public:
    DEFINE_SIZE_MIN (8);
  };

  protected:
  HBUINT32		version;	/* Version--0x00010000u */
  HBUINT32		nTables;	/* Number of subtables in the kerning table. */
  HBUINT8		data[VAR];
  public:
  DEFINE_SIZE_ARRAY (8, data);
};

struct kern
{
  static const hb_tag_t tableTag = HB_OT_TAG_kern;

  inline int get_h_kerning (hb_codepoint_t left, hb_codepoint_t right, unsigned int table_length) const
  {
    switch (u.major) {
    case 0: return u.ot.get_h_kerning (left, right, table_length);
    case 1: return u.aat.get_h_kerning (left, right, table_length);
    default:return 0;
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.major.sanitize (c)) return_trace (false);
    switch (u.major) {
    case 0: return_trace (u.ot.sanitize (c));
    case 1: return_trace (u.aat.sanitize (c));
    default:return_trace (true);
    }
  }

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      blob = Sanitizer<kern>().sanitize (face->reference_table (HB_OT_TAG_kern));
      table = Sanitizer<kern>::lock_instance (blob);
      table_length = hb_blob_get_length (blob);
    }
    inline void fini (void)
    {
      hb_blob_destroy (blob);
    }

    inline int get_h_kerning (hb_codepoint_t left, hb_codepoint_t right) const
    { return table->get_h_kerning (left, right, table_length); }

    private:
    hb_blob_t *blob;
    const kern *table;
    unsigned int table_length;
  };

  protected:
  union {
  HBUINT16		major;
  KernOT		ot;
  KernAAT		aat;
  } u;
  public:
  DEFINE_SIZE_UNION (2, major);
};

} /* namespace OT */


#endif /* HB_OT_KERN_TABLE_HH */
