/*
 * Copyright © 2016  Igalia S.L.
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
 * Igalia Author(s): Frédéric Wang
 */

#ifndef HB_OT_MATH_TABLE_HH
#define HB_OT_MATH_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-ot-layout-common-private.hh"
#include "hb-ot-math.h"

namespace OT {


struct MathValueRecord
{
  inline hb_position_t get_x_value (hb_font_t *font, const void *base) const
  { return font->em_scale_x (value) + (base+deviceTable).get_x_delta (font); }
  inline hb_position_t get_y_value (hb_font_t *font, const void *base) const
  { return font->em_scale_y (value) + (base+deviceTable).get_y_delta (font); }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && deviceTable.sanitize (c, base));
  }

  protected:
  HBINT16			value;		/* The X or Y value in design units */
  OffsetTo<Device>	deviceTable;	/* Offset to the device table - from the
					 * beginning of parent table. May be nullptr.
					 * Suggested format for device table is 1. */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct MathConstants
{
  inline bool sanitize_math_value_records (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    unsigned int count = ARRAY_LENGTH (mathValueRecords);
    for (unsigned int i = 0; i < count; i++)
      if (!mathValueRecords[i].sanitize (c, this))
	return_trace (false);

    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && sanitize_math_value_records(c));
  }

  inline hb_position_t get_value (hb_ot_math_constant_t constant,
				  hb_font_t *font) const
  {
    switch (constant) {

    case HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN:
    case HB_OT_MATH_CONSTANT_SCRIPT_SCRIPT_PERCENT_SCALE_DOWN:
      return percentScaleDown[constant - HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN];

    case HB_OT_MATH_CONSTANT_DELIMITED_SUB_FORMULA_MIN_HEIGHT:
    case HB_OT_MATH_CONSTANT_DISPLAY_OPERATOR_MIN_HEIGHT:
      return font->em_scale_y (minHeight[constant - HB_OT_MATH_CONSTANT_DELIMITED_SUB_FORMULA_MIN_HEIGHT]);

    case HB_OT_MATH_CONSTANT_RADICAL_KERN_AFTER_DEGREE:
    case HB_OT_MATH_CONSTANT_RADICAL_KERN_BEFORE_DEGREE:
    case HB_OT_MATH_CONSTANT_SKEWED_FRACTION_HORIZONTAL_GAP:
    case HB_OT_MATH_CONSTANT_SPACE_AFTER_SCRIPT:
      return mathValueRecords[constant - HB_OT_MATH_CONSTANT_MATH_LEADING].get_x_value(font, this);

    case HB_OT_MATH_CONSTANT_ACCENT_BASE_HEIGHT:
    case HB_OT_MATH_CONSTANT_AXIS_HEIGHT:
    case HB_OT_MATH_CONSTANT_FLATTENED_ACCENT_BASE_HEIGHT:
    case HB_OT_MATH_CONSTANT_FRACTION_DENOMINATOR_DISPLAY_STYLE_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_FRACTION_DENOMINATOR_GAP_MIN:
    case HB_OT_MATH_CONSTANT_FRACTION_DENOMINATOR_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_FRACTION_DENOM_DISPLAY_STYLE_GAP_MIN:
    case HB_OT_MATH_CONSTANT_FRACTION_NUMERATOR_DISPLAY_STYLE_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_FRACTION_NUMERATOR_GAP_MIN:
    case HB_OT_MATH_CONSTANT_FRACTION_NUMERATOR_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_FRACTION_NUM_DISPLAY_STYLE_GAP_MIN:
    case HB_OT_MATH_CONSTANT_FRACTION_RULE_THICKNESS:
    case HB_OT_MATH_CONSTANT_LOWER_LIMIT_BASELINE_DROP_MIN:
    case HB_OT_MATH_CONSTANT_LOWER_LIMIT_GAP_MIN:
    case HB_OT_MATH_CONSTANT_MATH_LEADING:
    case HB_OT_MATH_CONSTANT_OVERBAR_EXTRA_ASCENDER:
    case HB_OT_MATH_CONSTANT_OVERBAR_RULE_THICKNESS:
    case HB_OT_MATH_CONSTANT_OVERBAR_VERTICAL_GAP:
    case HB_OT_MATH_CONSTANT_RADICAL_DISPLAY_STYLE_VERTICAL_GAP:
    case HB_OT_MATH_CONSTANT_RADICAL_EXTRA_ASCENDER:
    case HB_OT_MATH_CONSTANT_RADICAL_RULE_THICKNESS:
    case HB_OT_MATH_CONSTANT_RADICAL_VERTICAL_GAP:
    case HB_OT_MATH_CONSTANT_SKEWED_FRACTION_VERTICAL_GAP:
    case HB_OT_MATH_CONSTANT_STACK_BOTTOM_DISPLAY_STYLE_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_STACK_BOTTOM_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_STACK_DISPLAY_STYLE_GAP_MIN:
    case HB_OT_MATH_CONSTANT_STACK_GAP_MIN:
    case HB_OT_MATH_CONSTANT_STACK_TOP_DISPLAY_STYLE_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_STACK_TOP_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_STRETCH_STACK_BOTTOM_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_STRETCH_STACK_GAP_ABOVE_MIN:
    case HB_OT_MATH_CONSTANT_STRETCH_STACK_GAP_BELOW_MIN:
    case HB_OT_MATH_CONSTANT_STRETCH_STACK_TOP_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_SUBSCRIPT_BASELINE_DROP_MIN:
    case HB_OT_MATH_CONSTANT_SUBSCRIPT_SHIFT_DOWN:
    case HB_OT_MATH_CONSTANT_SUBSCRIPT_TOP_MAX:
    case HB_OT_MATH_CONSTANT_SUB_SUPERSCRIPT_GAP_MIN:
    case HB_OT_MATH_CONSTANT_SUPERSCRIPT_BASELINE_DROP_MAX:
    case HB_OT_MATH_CONSTANT_SUPERSCRIPT_BOTTOM_MAX_WITH_SUBSCRIPT:
    case HB_OT_MATH_CONSTANT_SUPERSCRIPT_BOTTOM_MIN:
    case HB_OT_MATH_CONSTANT_SUPERSCRIPT_SHIFT_UP:
    case HB_OT_MATH_CONSTANT_SUPERSCRIPT_SHIFT_UP_CRAMPED:
    case HB_OT_MATH_CONSTANT_UNDERBAR_EXTRA_DESCENDER:
    case HB_OT_MATH_CONSTANT_UNDERBAR_RULE_THICKNESS:
    case HB_OT_MATH_CONSTANT_UNDERBAR_VERTICAL_GAP:
    case HB_OT_MATH_CONSTANT_UPPER_LIMIT_BASELINE_RISE_MIN:
    case HB_OT_MATH_CONSTANT_UPPER_LIMIT_GAP_MIN:
      return mathValueRecords[constant - HB_OT_MATH_CONSTANT_MATH_LEADING].get_y_value(font, this);

    case HB_OT_MATH_CONSTANT_RADICAL_DEGREE_BOTTOM_RAISE_PERCENT:
      return radicalDegreeBottomRaisePercent;

    default:
      return 0;
    }
  }

  protected:
  HBINT16 percentScaleDown[2];
  HBUINT16 minHeight[2];
  MathValueRecord mathValueRecords[51];
  HBINT16 radicalDegreeBottomRaisePercent;

  public:
  DEFINE_SIZE_STATIC (214);
};

struct MathItalicsCorrectionInfo
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  coverage.sanitize (c, this) &&
		  italicsCorrection.sanitize (c, this));
  }

  inline hb_position_t get_value (hb_codepoint_t glyph,
				  hb_font_t *font) const
  {
    unsigned int index = (this+coverage).get_coverage (glyph);
    return italicsCorrection[index].get_x_value (font, this);
  }

  protected:
  OffsetTo<Coverage>       coverage;		/* Offset to Coverage table -
						 * from the beginning of
						 * MathItalicsCorrectionInfo
						 * table. */
  ArrayOf<MathValueRecord> italicsCorrection;	/* Array of MathValueRecords
						 * defining italics correction
						 * values for each
						 * covered glyph. */

  public:
  DEFINE_SIZE_ARRAY (4, italicsCorrection);
};

struct MathTopAccentAttachment
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  topAccentCoverage.sanitize (c, this) &&
		  topAccentAttachment.sanitize (c, this));
  }

  inline hb_position_t get_value (hb_codepoint_t glyph,
				  hb_font_t *font) const
  {
    unsigned int index = (this+topAccentCoverage).get_coverage (glyph);
    if (index == NOT_COVERED)
      return font->get_glyph_h_advance (glyph) / 2;
    return topAccentAttachment[index].get_x_value(font, this);
  }

  protected:
  OffsetTo<Coverage>       topAccentCoverage;   /* Offset to Coverage table -
						 * from the beginning of
						 * MathTopAccentAttachment
						 * table. */
  ArrayOf<MathValueRecord> topAccentAttachment; /* Array of MathValueRecords
						 * defining top accent
						 * attachment points for each
						 * covered glyph. */

  public:
  DEFINE_SIZE_ARRAY (2 + 2, topAccentAttachment);
};

struct MathKern
{
  inline bool sanitize_math_value_records (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    unsigned int count = 2 * heightCount + 1;
    for (unsigned int i = 0; i < count; i++)
      if (!mathValueRecords[i].sanitize (c, this)) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  c->check_array (mathValueRecords,
				  mathValueRecords[0].static_size,
				  2 * heightCount + 1) &&
		  sanitize_math_value_records (c));
  }

  inline hb_position_t get_value (hb_position_t correction_height, hb_font_t *font) const
  {
    const MathValueRecord* correctionHeight = mathValueRecords;
    const MathValueRecord* kernValue = mathValueRecords + heightCount;
    int sign = font->y_scale < 0 ? -1 : +1;

    /* The description of the MathKern table is a ambiguous, but interpreting
     * "between the two heights found at those indexes" for 0 < i < len as
     *
     *   correctionHeight[i-1] < correction_height <= correctionHeight[i]
     *
     * makes the result consistent with the limit cases and we can just use the
     * binary search algorithm of std::upper_bound:
     */
    unsigned int i = 0;
    unsigned int count = heightCount;
    while (count > 0)
    {
      unsigned int half = count / 2;
      hb_position_t height = correctionHeight[i + half].get_y_value(font, this);
      if (sign * height < sign * correction_height)
      {
	i += half + 1;
	count -= half + 1;
      } else
	count = half;
    }
    return kernValue[i].get_x_value(font, this);
  }

  protected:
  HBUINT16	  heightCount;
  MathValueRecord mathValueRecords[VAR]; /* Array of correction heights at
					  * which the kern value changes.
					  * Sorted by the height value in
					  * design units (heightCount entries),
					  * Followed by:
					  * Array of kern values corresponding
					  * to heights. (heightCount+1 entries).
					  */

  public:
  DEFINE_SIZE_ARRAY (2, mathValueRecords);
};

struct MathKernInfoRecord
{
  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);

    unsigned int count = ARRAY_LENGTH (mathKern);
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!mathKern[i].sanitize (c, base)))
	return_trace (false);

    return_trace (true);
  }

  inline hb_position_t get_kerning (hb_ot_math_kern_t kern,
				    hb_position_t correction_height,
				    hb_font_t *font,
				    const void *base) const
  {
    unsigned int idx = kern;
    if (unlikely (idx >= ARRAY_LENGTH (mathKern))) return 0;
    return (base+mathKern[idx]).get_value (correction_height, font);
  }

  protected:
  /* Offset to MathKern table for each corner -
   * from the beginning of MathKernInfo table. May be nullptr. */
  OffsetTo<MathKern> mathKern[4];

  public:
  DEFINE_SIZE_STATIC (8);
};

struct MathKernInfo
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  mathKernCoverage.sanitize (c, this) &&
		  mathKernInfoRecords.sanitize (c, this));
  }

  inline hb_position_t get_kerning (hb_codepoint_t glyph,
				    hb_ot_math_kern_t kern,
				    hb_position_t correction_height,
				    hb_font_t *font) const
  {
    unsigned int index = (this+mathKernCoverage).get_coverage (glyph);
    return mathKernInfoRecords[index].get_kerning (kern, correction_height, font, this);
  }

  protected:
  OffsetTo<Coverage>		mathKernCoverage;    /* Offset to Coverage table -
						      * from the beginning of the
						      * MathKernInfo table. */
  ArrayOf<MathKernInfoRecord>	mathKernInfoRecords; /* Array of
						      * MathKernInfoRecords,
						      * per-glyph information for
						      * mathematical positioning
						      * of subscripts and
						      * superscripts. */

  public:
  DEFINE_SIZE_ARRAY (4, mathKernInfoRecords);
};

struct MathGlyphInfo
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  mathItalicsCorrectionInfo.sanitize (c, this) &&
		  mathTopAccentAttachment.sanitize (c, this) &&
		  extendedShapeCoverage.sanitize (c, this) &&
		  mathKernInfo.sanitize(c, this));
  }

  inline hb_position_t
  get_italics_correction (hb_codepoint_t  glyph, hb_font_t *font) const
  { return (this+mathItalicsCorrectionInfo).get_value (glyph, font); }

  inline hb_position_t
  get_top_accent_attachment (hb_codepoint_t  glyph, hb_font_t *font) const
  { return (this+mathTopAccentAttachment).get_value (glyph, font); }

  inline bool is_extended_shape (hb_codepoint_t glyph) const
  { return (this+extendedShapeCoverage).get_coverage (glyph) != NOT_COVERED; }

  inline hb_position_t get_kerning (hb_codepoint_t glyph,
				    hb_ot_math_kern_t kern,
				    hb_position_t correction_height,
				    hb_font_t *font) const
  { return (this+mathKernInfo).get_kerning (glyph, kern, correction_height, font); }

  protected:
  /* Offset to MathItalicsCorrectionInfo table -
   * from the beginning of MathGlyphInfo table. */
  OffsetTo<MathItalicsCorrectionInfo> mathItalicsCorrectionInfo;

  /* Offset to MathTopAccentAttachment table -
   * from the beginning of MathGlyphInfo table. */
  OffsetTo<MathTopAccentAttachment> mathTopAccentAttachment;

  /* Offset to coverage table for Extended Shape glyphs -
   * from the beginning of MathGlyphInfo table. When the left or right glyph of
   * a box is an extended shape variant, the (ink) box (and not the default
   * position defined by values in MathConstants table) should be used for
   * vertical positioning purposes. May be nullptr.. */
  OffsetTo<Coverage> extendedShapeCoverage;

   /* Offset to MathKernInfo table -
    * from the beginning of MathGlyphInfo table. */
  OffsetTo<MathKernInfo> mathKernInfo;

  public:
  DEFINE_SIZE_STATIC (8);
};

struct MathGlyphVariantRecord
{
  friend struct MathGlyphConstruction;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  GlyphID variantGlyph;       /* Glyph ID for the variant. */
  HBUINT16  advanceMeasurement; /* Advance width/height, in design units, of the
			       * variant, in the direction of requested
			       * glyph extension. */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct PartFlags : HBUINT16
{
  enum Flags {
    Extender	= 0x0001u, /* If set, the part can be skipped or repeated. */

    Defined	= 0x0001u, /* All defined flags. */
  };

  public:
  DEFINE_SIZE_STATIC (2);
};

struct MathGlyphPartRecord
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  inline void extract (hb_ot_math_glyph_part_t &out,
		       int scale,
		       hb_font_t *font) const
  {
    out.glyph			= glyph;

    out.start_connector_length	= font->em_scale (startConnectorLength, scale);
    out.end_connector_length	= font->em_scale (endConnectorLength, scale);
    out.full_advance		= font->em_scale (fullAdvance, scale);

    static_assert ((unsigned int) HB_MATH_GLYPH_PART_FLAG_EXTENDER ==
		   (unsigned int) PartFlags::Extender, "");

    out.flags = (hb_ot_math_glyph_part_flags_t)
		(unsigned int)
		(partFlags & PartFlags::Defined);
  }

  protected:
  GlyphID   glyph;		  /* Glyph ID for the part. */
  HBUINT16    startConnectorLength; /* Advance width/ height of the straight bar
				   * connector material, in design units, is at
				   * the beginning of the glyph, in the
				   * direction of the extension. */
  HBUINT16    endConnectorLength;   /* Advance width/ height of the straight bar
				   * connector material, in design units, is at
				   * the end of the glyph, in the direction of
				   * the extension. */
  HBUINT16    fullAdvance;	  /* Full advance width/height for this part,
				   * in the direction of the extension.
				   * In design units. */
  PartFlags partFlags;		  /* Part qualifiers. */

  public:
  DEFINE_SIZE_STATIC (10);
};

struct MathGlyphAssembly
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  italicsCorrection.sanitize(c, this) &&
		  partRecords.sanitize(c));
  }

  inline unsigned int get_parts (hb_direction_t direction,
				 hb_font_t *font,
				 unsigned int start_offset,
				 unsigned int *parts_count, /* IN/OUT */
				 hb_ot_math_glyph_part_t *parts /* OUT */,
				 hb_position_t *italics_correction /* OUT */) const
  {
    if (parts_count)
    {
      int scale = font->dir_scale (direction);
      const MathGlyphPartRecord *arr =
	    partRecords.sub_array (start_offset, parts_count);
      unsigned int count = *parts_count;
      for (unsigned int i = 0; i < count; i++)
	arr[i].extract (parts[i], scale, font);
    }

    if (italics_correction)
      *italics_correction = italicsCorrection.get_x_value (font, this);

    return partRecords.len;
  }

  protected:
  MathValueRecord	   italicsCorrection; /* Italics correction of this
					       * MathGlyphAssembly. Should not
					       * depend on the assembly size. */
  ArrayOf<MathGlyphPartRecord> partRecords;   /* Array of part records, from
					       * left to right and bottom to
					       * top. */

  public:
  DEFINE_SIZE_ARRAY (6, partRecords);
};

struct MathGlyphConstruction
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  glyphAssembly.sanitize(c, this) &&
		  mathGlyphVariantRecord.sanitize(c));
  }

  inline const MathGlyphAssembly &get_assembly (void) const
  { return this+glyphAssembly; }

  inline unsigned int get_variants (hb_direction_t direction,
				    hb_font_t *font,
				    unsigned int start_offset,
				    unsigned int *variants_count, /* IN/OUT */
				    hb_ot_math_glyph_variant_t *variants /* OUT */) const
  {
    if (variants_count)
    {
      int scale = font->dir_scale (direction);
      const MathGlyphVariantRecord *arr =
	    mathGlyphVariantRecord.sub_array (start_offset, variants_count);
      unsigned int count = *variants_count;
      for (unsigned int i = 0; i < count; i++)
      {
	variants[i].glyph = arr[i].variantGlyph;
	variants[i].advance = font->em_scale (arr[i].advanceMeasurement, scale);
      }
    }
    return mathGlyphVariantRecord.len;
  }

  protected:
  /* Offset to MathGlyphAssembly table for this shape - from the beginning of
     MathGlyphConstruction table. May be nullptr. */
  OffsetTo<MathGlyphAssembly>	  glyphAssembly;

  /* MathGlyphVariantRecords for alternative variants of the glyphs. */
  ArrayOf<MathGlyphVariantRecord> mathGlyphVariantRecord;

  public:
  DEFINE_SIZE_ARRAY (4, mathGlyphVariantRecord);
};

struct MathVariants
{
  inline bool sanitize_offsets (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    unsigned int count = vertGlyphCount + horizGlyphCount;
    for (unsigned int i = 0; i < count; i++)
      if (!glyphConstruction[i].sanitize (c, this)) return_trace (false);
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  vertGlyphCoverage.sanitize (c, this) &&
		  horizGlyphCoverage.sanitize (c, this) &&
		  c->check_array (glyphConstruction,
				  glyphConstruction[0].static_size,
				  vertGlyphCount + horizGlyphCount) &&
		  sanitize_offsets (c));
  }

  inline hb_position_t get_min_connector_overlap (hb_direction_t direction,
						  hb_font_t *font) const
  { return font->em_scale_dir (minConnectorOverlap, direction); }

  inline unsigned int get_glyph_variants (hb_codepoint_t glyph,
					  hb_direction_t direction,
					  hb_font_t *font,
					  unsigned int start_offset,
					  unsigned int *variants_count, /* IN/OUT */
					  hb_ot_math_glyph_variant_t *variants /* OUT */) const
  { return get_glyph_construction (glyph, direction, font)
	   .get_variants (direction, font, start_offset, variants_count, variants); }

  inline unsigned int get_glyph_parts (hb_codepoint_t glyph,
				       hb_direction_t direction,
				       hb_font_t *font,
				       unsigned int start_offset,
				       unsigned int *parts_count, /* IN/OUT */
				       hb_ot_math_glyph_part_t *parts /* OUT */,
				       hb_position_t *italics_correction /* OUT */) const
  { return get_glyph_construction (glyph, direction, font)
	   .get_assembly ()
	   .get_parts (direction, font,
		       start_offset, parts_count, parts,
		       italics_correction); }

  private:
  inline const MathGlyphConstruction &
		get_glyph_construction (hb_codepoint_t glyph,
					hb_direction_t direction,
					hb_font_t *font) const
  {
    bool vertical = HB_DIRECTION_IS_VERTICAL (direction);
    unsigned int count = vertical ? vertGlyphCount : horizGlyphCount;
    const OffsetTo<Coverage> &coverage = vertical ? vertGlyphCoverage
						  : horizGlyphCoverage;

    unsigned int index = (this+coverage).get_coverage (glyph);
    if (unlikely (index >= count)) return Null(MathGlyphConstruction);

    if (!vertical)
      index += vertGlyphCount;

    return this+glyphConstruction[index];
  }

  protected:
  HBUINT16	     minConnectorOverlap; /* Minimum overlap of connecting
					   * glyphs during glyph construction,
					   * in design units. */
  OffsetTo<Coverage> vertGlyphCoverage;   /* Offset to Coverage table -
					   * from the beginning of MathVariants
					   * table. */
  OffsetTo<Coverage> horizGlyphCoverage;  /* Offset to Coverage table -
					   * from the beginning of MathVariants
					   * table. */
  HBUINT16	     vertGlyphCount;      /* Number of glyphs for which
					   * information is provided for
					   * vertically growing variants. */
  HBUINT16	     horizGlyphCount;     /* Number of glyphs for which
					   * information is provided for
					   * horizontally growing variants. */

  /* Array of offsets to MathGlyphConstruction tables - from the beginning of
     the MathVariants table, for shapes growing in vertical/horizontal
     direction. */
  OffsetTo<MathGlyphConstruction> glyphConstruction[VAR];

  public:
  DEFINE_SIZE_ARRAY (10, glyphConstruction);
};


/*
 * MATH -- Mathematical typesetting
 * https://docs.microsoft.com/en-us/typography/opentype/spec/math
 */

struct MATH
{
  static const hb_tag_t tableTag	= HB_OT_TAG_MATH;

  inline bool has_data (void) const { return version.to_int () != 0; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  mathConstants.sanitize (c, this) &&
		  mathGlyphInfo.sanitize (c, this) &&
		  mathVariants.sanitize (c, this));
  }

  inline hb_position_t get_constant (hb_ot_math_constant_t  constant,
				     hb_font_t		   *font) const
  { return (this+mathConstants).get_value (constant, font); }

  inline const MathGlyphInfo &get_math_glyph_info (void) const
  { return this+mathGlyphInfo; }

  inline const MathVariants &get_math_variants (void) const
  { return this+mathVariants; }

  protected:
  FixedVersion<>version;		/* Version of the MATH table
					 * initially set to 0x00010000u */
  OffsetTo<MathConstants> mathConstants;/* MathConstants table */
  OffsetTo<MathGlyphInfo> mathGlyphInfo;/* MathGlyphInfo table */
  OffsetTo<MathVariants>  mathVariants;	/* MathVariants table */

  public:
  DEFINE_SIZE_STATIC (10);
};

} /* namespace OT */


#endif /* HB_OT_MATH_TABLE_HH */
