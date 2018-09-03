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

#include "hb-open-type-private.hh"

#include "hb-ot-layout-private.hh"
#include "hb-ot-math-table.hh"

static inline const OT::MATH&
_get_math (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::MATH);
  hb_ot_layout_t * layout = hb_ot_layout_from_face (face);
  return *(layout->math.get ());
}

/*
 * OT::MATH
 */

/**
 * hb_ot_math_has_data:
 * @face: #hb_face_t to test
 *
 * This function allows to verify the presence of an OpenType MATH table on the
 * face.
 *
 * Return value: true if face has a MATH table, false otherwise
 *
 * Since: 1.3.3
 **/
hb_bool_t
hb_ot_math_has_data (hb_face_t *face)
{
  return &_get_math (face) != &OT::Null(OT::MATH);
}

/**
 * hb_ot_math_get_constant:
 * @font: #hb_font_t from which to retrieve the value
 * @constant: #hb_ot_math_constant_t the constant to retrieve
 *
 * This function returns the requested math constants as a #hb_position_t.
 * If the request constant is HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN,
 * HB_OT_MATH_CONSTANT_SCRIPT_SCRIPT_PERCENT_SCALE_DOWN or
 * HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN then the return value is
 * actually an integer between 0 and 100 representing that percentage.
 *
 * Return value: the requested constant or 0
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_constant (hb_font_t *font,
			 hb_ot_math_constant_t constant)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_constant(constant, font);
}

/**
 * hb_ot_math_get_glyph_italics_correction:
 * @font: #hb_font_t from which to retrieve the value
 * @glyph: glyph index from which to retrieve the value
 *
 * Return value: the italics correction of the glyph or 0
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_italics_correction (hb_font_t *font,
					 hb_codepoint_t glyph)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_glyph_info().get_italics_correction (glyph, font);
}

/**
 * hb_ot_math_get_glyph_top_accent_attachment:
 * @font: #hb_font_t from which to retrieve the value
 * @glyph: glyph index from which to retrieve the value
 *
 * Return value: the top accent attachment of the glyph or 0
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_top_accent_attachment (hb_font_t *font,
					    hb_codepoint_t glyph)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_glyph_info().get_top_accent_attachment (glyph, font);
}

/**
 * hb_ot_math_is_glyph_extended_shape:
 * @face: a #hb_face_t to test
 * @glyph: a glyph index to test
 *
 * Return value: true if the glyph is an extended shape, false otherwise
 *
 * Since: 1.3.3
 **/
hb_bool_t
hb_ot_math_is_glyph_extended_shape (hb_face_t *face,
				    hb_codepoint_t glyph)
{
  const OT::MATH &math = _get_math (face);
  return math.get_math_glyph_info().is_extended_shape (glyph);
}

/**
 * hb_ot_math_get_glyph_kerning:
 * @font: #hb_font_t from which to retrieve the value
 * @glyph: glyph index from which to retrieve the value
 * @kern: the #hb_ot_math_kern_t from which to retrieve the value
 * @correction_height: the correction height to use to determine the kerning.
 *
 * This function tries to retrieve the MathKern table for the specified font,
 * glyph and #hb_ot_math_kern_t. Then it browses the list of heights from the
 * MathKern table to find one value that is greater or equal to specified
 * correction_height. If one is found the corresponding value from the list of
 * kerns is returned and otherwise the last kern value is returned.
 *
 * Return value: requested kerning or 0
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_kerning (hb_font_t *font,
			      hb_codepoint_t glyph,
			      hb_ot_math_kern_t kern,
			      hb_position_t correction_height)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_glyph_info().get_kerning (glyph, kern, correction_height, font);
}

/**
 * hb_ot_math_get_glyph_variants:
 * @font: #hb_font_t from which to retrieve the values
 * @glyph: index of the glyph to stretch
 * @direction: direction of the stretching
 * @start_offset: offset of the first variant to retrieve
 * @variants_count: maximum number of variants to retrieve after start_offset
 * (IN) and actual number of variants retrieved (OUT)
 * @variants: array of size at least @variants_count to store the result
 *
 * This function tries to retrieve the MathGlyphConstruction for the specified
 * font, glyph and direction. Note that only the value of
 * #HB_DIRECTION_IS_HORIZONTAL is considered. It provides the corresponding list
 * of size variants as an array of hb_ot_math_glyph_variant_t structs.
 *
 * Return value: the total number of size variants available or 0
 *
 * Since: 1.3.3
 **/
unsigned int
hb_ot_math_get_glyph_variants (hb_font_t *font,
			       hb_codepoint_t glyph,
			       hb_direction_t direction,
			       unsigned int start_offset,
			       unsigned int *variants_count, /* IN/OUT */
			       hb_ot_math_glyph_variant_t *variants /* OUT */)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_variants().get_glyph_variants (glyph, direction, font,
						      start_offset,
						      variants_count,
						      variants);
}

/**
 * hb_ot_math_get_min_connector_overlap:
 * @font: #hb_font_t from which to retrieve the value
 * @direction: direction of the stretching
 *
 * This function tries to retrieve the MathVariants table for the specified
 * font and returns the minimum overlap of connecting glyphs to draw a glyph
 * assembly in the specified direction. Note that only the value of
 * #HB_DIRECTION_IS_HORIZONTAL is considered.
 *
 * Return value: requested min connector overlap or 0
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_min_connector_overlap (hb_font_t *font,
				      hb_direction_t direction)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_variants().get_min_connector_overlap (direction, font);
}

/**
 * hb_ot_math_get_glyph_assembly:
 * @font: #hb_font_t from which to retrieve the values
 * @glyph: index of the glyph to stretch
 * @direction: direction of the stretching
 * @start_offset: offset of the first glyph part to retrieve
 * @parts_count: maximum number of glyph parts to retrieve after start_offset
 * (IN) and actual number of parts retrieved (OUT)
 * @parts: array of size at least @parts_count to store the result
 * @italics_correction: italic correction of the glyph assembly
 *
 * This function tries to retrieve the GlyphAssembly for the specified font,
 * glyph and direction. Note that only the value of #HB_DIRECTION_IS_HORIZONTAL
 * is considered. It provides the information necessary to draw the glyph
 * assembly as an array of #hb_ot_math_glyph_part_t.
 *
 * Return value: the total number of parts in the glyph assembly
 *
 * Since: 1.3.3
 **/
unsigned int
hb_ot_math_get_glyph_assembly (hb_font_t *font,
			       hb_codepoint_t glyph,
			       hb_direction_t direction,
			       unsigned int start_offset,
			       unsigned int *parts_count, /* IN/OUT */
			       hb_ot_math_glyph_part_t *parts, /* OUT */
			       hb_position_t *italics_correction /* OUT */)
{
  const OT::MATH &math = _get_math (font->face);
  return math.get_math_variants().get_glyph_parts (glyph, direction, font,
						   start_offset,
						   parts_count,
						   parts,
						   italics_correction);
}
