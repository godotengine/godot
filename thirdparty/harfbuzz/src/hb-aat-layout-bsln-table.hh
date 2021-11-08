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

#ifndef HB_AAT_LAYOUT_BSLN_TABLE_HH
#define HB_AAT_LAYOUT_BSLN_TABLE_HH

#include "hb-aat-layout-common.hh"

/*
 * bsln -- Baseline
 * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6bsln.html
 */
#define HB_AAT_TAG_bsln HB_TAG('b','s','l','n')


namespace AAT {


struct BaselineTableFormat0Part
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  // Roman, Ideographic centered, Ideographic low, Hanging and Math
  // are the default defined ones, but any other maybe accessed also.
  HBINT16	deltas[32];	/* These are the FUnit distance deltas from
				 * the font's natural baseline to the other
				 * baselines used in the font. */
  public:
  DEFINE_SIZE_STATIC (64);
};

struct BaselineTableFormat1Part
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  lookupTable.sanitize (c)));
  }

  protected:
  HBINT16	deltas[32];	/* ditto */
  Lookup<HBUINT16>
		lookupTable;	/* Lookup table that maps glyphs to their
				 * baseline values. */
  public:
  DEFINE_SIZE_MIN (66);
};

struct BaselineTableFormat2Part
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBGlyphID16	stdGlyph;	/* The specific glyph index number in this
				 * font that is used to set the baseline values.
				 * This is the standard glyph.
				 * This glyph must contain a set of control points
				 * (whose numbers are contained in the ctlPoints field)
				 * that are used to determine baseline distances. */
  HBUINT16	ctlPoints[32];	/* Set of control point numbers,
				 * associated with the standard glyph.
				 * A value of 0xFFFF means there is no corresponding
				 * control point in the standard glyph. */
  public:
  DEFINE_SIZE_STATIC (66);
};

struct BaselineTableFormat3Part
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) && lookupTable.sanitize (c)));
  }

  protected:
  HBGlyphID16	stdGlyph;	/* ditto */
  HBUINT16	ctlPoints[32];	/* ditto */
  Lookup<HBUINT16>
		lookupTable;	/* Lookup table that maps glyphs to their
				 * baseline values. */
  public:
  DEFINE_SIZE_MIN (68);
};

struct bsln
{
  static constexpr hb_tag_t tableTag = HB_AAT_TAG_bsln;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(c->check_struct (this) && defaultBaseline < 32)))
      return_trace (false);

    switch (format)
    {
    case 0: return_trace (parts.format0.sanitize (c));
    case 1: return_trace (parts.format1.sanitize (c));
    case 2: return_trace (parts.format2.sanitize (c));
    case 3: return_trace (parts.format3.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  FixedVersion<>version;	/* Version number of the Baseline table. */
  HBUINT16	format;		/* Format of the baseline table. Only one baseline
				 * format may be selected for the font. */
  HBUINT16	defaultBaseline;/* Default baseline value for all glyphs.
				 * This value can be from 0 through 31. */
  union {
  // Distance-Based Formats
  BaselineTableFormat0Part	format0;
  BaselineTableFormat1Part	format1;
  // Control Point-based Formats
  BaselineTableFormat2Part	format2;
  BaselineTableFormat3Part	format3;
  } parts;
  public:
  DEFINE_SIZE_MIN (8);
};

} /* namespace AAT */


#endif /* HB_AAT_LAYOUT_BSLN_TABLE_HH */
