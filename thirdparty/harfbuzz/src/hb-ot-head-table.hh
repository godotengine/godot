/*
 * Copyright © 2010  Red Hat, Inc.
 * Copyright © 2012  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_HEAD_TABLE_HH
#define HB_OT_HEAD_TABLE_HH

#include "hb-open-type.hh"

/*
 * head -- Font Header
 * https://docs.microsoft.com/en-us/typography/opentype/spec/head
 */
#define HB_OT_TAG_head HB_TAG('h','e','a','d')


namespace OT {


struct head
{
  friend struct OpenTypeOffsetTable;

  static constexpr hb_tag_t tableTag = HB_OT_TAG_head;

  unsigned int get_upem () const
  {
    unsigned int upem = unitsPerEm;
    /* If no valid head table found, assume 1000, which matches typical Type1 usage. */
    return 16 <= upem && upem <= 16384 ? upem : 1000;
  }

  bool serialize (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace ((bool) c->embed (this));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    head *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    if (c->plan->normalized_coords)
    {
      if (unlikely (!c->serializer->check_assign (out->xMin, c->plan->head_maxp_info.xMin,
                                                  HB_SERIALIZE_ERROR_INT_OVERFLOW)))
        return_trace (false);
      if (unlikely (!c->serializer->check_assign (out->xMax, c->plan->head_maxp_info.xMax,
                                                  HB_SERIALIZE_ERROR_INT_OVERFLOW)))
        return_trace (false);
      if (unlikely (!c->serializer->check_assign (out->yMin, c->plan->head_maxp_info.yMin,
                                                  HB_SERIALIZE_ERROR_INT_OVERFLOW)))
        return_trace (false);
      if (unlikely (!c->serializer->check_assign (out->yMax, c->plan->head_maxp_info.yMax,
                                                  HB_SERIALIZE_ERROR_INT_OVERFLOW)))
        return_trace (false);
    }
    return_trace (true);
  }

  enum mac_style_flag_t {
    BOLD	= 1u<<0,
    ITALIC	= 1u<<1,
    UNDERLINE	= 1u<<2,
    OUTLINE	= 1u<<3,
    SHADOW	= 1u<<4,
    CONDENSED	= 1u<<5,
    EXPANDED	= 1u<<6,
  };

  bool is_bold () const      { return macStyle & BOLD; }
  bool is_italic () const    { return macStyle & ITALIC; }
  bool is_condensed () const { return macStyle & CONDENSED; }
  bool is_expanded () const  { return macStyle & EXPANDED; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  hb_barrier () &&
		  version.major == 1 &&
		  magicNumber == 0x5F0F3CF5u);
  }

  protected:
  FixedVersion<>version;		/* Version of the head table--currently
					 * 0x00010000u for version 1.0. */
  FixedVersion<>fontRevision;		/* Set by font manufacturer. */
  HBUINT32	checkSumAdjustment;	/* To compute: set it to 0, sum the
					 * entire font as HBUINT32, then store
					 * 0xB1B0AFBAu - sum. */
  HBUINT32	magicNumber;		/* Set to 0x5F0F3CF5u. */
  public:
  HBUINT16	flags;			/* Bit 0: Baseline for font at y=0;
					 * Bit 1: Left sidebearing point at x=0;
					 * Bit 2: Instructions may depend on point size;
					 * Bit 3: Force ppem to integer values for all
					 *   internal scaler math; may use fractional
					 *   ppem sizes if this bit is clear;
					 * Bit 4: Instructions may alter advance width
					 *   (the advance widths might not scale linearly);
					 * Bits 5-10: These should be set according to
					 *   Apple's specification. However, they are not
					 *   implemented in OpenType.
					 * Bit 5: This bit should be set in fonts that are
					 *   intended to e laid out vertically, and in
					 *   which the glyphs have been drawn such that an
					 *   x-coordinate of 0 corresponds to the desired
					 *   vertical baseline.
					 * Bit 6: This bit must be set to zero.
					 * Bit 7: This bit should be set if the font
					 *   requires layout for correct linguistic
					 *   rendering (e.g. Arabic fonts).
					 * Bit 8: This bit should be set for a GX font
					 *   which has one or more metamorphosis effects
					 *   designated as happening by default.
					 * Bit 9: This bit should be set if the font
					 *   contains any strong right-to-left glyphs.
					 * Bit 10: This bit should be set if the font
					 *   contains Indic-style rearrangement effects.
					 * Bit 11: Font data is 'lossless,' as a result
					 *   of having been compressed and decompressed
					 *   with the Agfa MicroType Express engine.
					 * Bit 12: Font converted (produce compatible metrics)
					 * Bit 13: Font optimized for ClearType™.
					 *   Note, fonts that rely on embedded bitmaps (EBDT)
					 *   for rendering should not be considered optimized
					 *   for ClearType, and therefore should keep this bit
					 *   cleared.
					 * Bit 14: Last Resort font. If set, indicates that
					 * the glyphs encoded in the cmap subtables are simply
					 * generic symbolic representations of code point
					 * ranges and don’t truly represent support for those
					 * code points. If unset, indicates that the glyphs
					 * encoded in the cmap subtables represent proper
					 * support for those code points.
					 * Bit 15: Reserved, set to 0. */
  protected:
  HBUINT16	unitsPerEm;		/* Valid range is from 16 to 16384. This value
					 * should be a power of 2 for fonts that have
					 * TrueType outlines. */
  LONGDATETIME	created;		/* Number of seconds since 12:00 midnight,
					   January 1, 1904. 64-bit integer */
  LONGDATETIME	modified;		/* Number of seconds since 12:00 midnight,
					   January 1, 1904. 64-bit integer */
  public:
  HBINT16	xMin;			/* For all glyph bounding boxes. */
  HBINT16	yMin;			/* For all glyph bounding boxes. */
  HBINT16	xMax;			/* For all glyph bounding boxes. */
  HBINT16	yMax;			/* For all glyph bounding boxes. */
  protected:
  HBUINT16	macStyle;		/* Bit 0: Bold (if set to 1);
					 * Bit 1: Italic (if set to 1)
					 * Bit 2: Underline (if set to 1)
					 * Bit 3: Outline (if set to 1)
					 * Bit 4: Shadow (if set to 1)
					 * Bit 5: Condensed (if set to 1)
					 * Bit 6: Extended (if set to 1)
					 * Bits 7-15: Reserved (set to 0). */
  HBUINT16	lowestRecPPEM;		/* Smallest readable size in pixels. */
  HBINT16	fontDirectionHint;	/* Deprecated (Set to 2).
					 * 0: Fully mixed directional glyphs;
					 * 1: Only strongly left to right;
					 * 2: Like 1 but also contains neutrals;
					 * -1: Only strongly right to left;
					 * -2: Like -1 but also contains neutrals. */
  public:
  HBUINT16	indexToLocFormat;	/* 0 for short offsets, 1 for long. */
  HBUINT16	glyphDataFormat;	/* 0 for current format. */

  DEFINE_SIZE_STATIC (54);
};


} /* namespace OT */


#endif /* HB_OT_HEAD_TABLE_HH */
