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

#include "hb.hh"

#ifndef HB_NO_MATH

#include "hb-ot-math-table.hh"


/**
 * SECTION:hb-ot-math
 * @title: hb-ot-math
 * @short_description: OpenType Math information
 * @include: hb-ot.h
 *
 * Functions for fetching mathematics layout data from OpenType fonts.
 *
 * HarfBuzz itself does not implement a math layout solution. The
 * functions and types provided can be used by client programs to access
 * the font data necessary for typesetting OpenType Math layout.
 *
 **/


/*
 * OT::MATH
 */

/**
 * hb_ot_math_has_data:
 * @face: #hb_face_t to test
 *
 * Tests whether a face has a `MATH` table.
 *
 * Return value: true if the table is found, false otherwise
 *
 * Since: 1.3.3
 **/
hb_bool_t
hb_ot_math_has_data (hb_face_t *face)
{
  return face->table.MATH->has_data ();
}

/**
 * hb_ot_math_get_constant:
 * @font: #hb_font_t to work upon
 * @constant: #hb_ot_math_constant_t the constant to retrieve
 *
 * Fetches the specified math constant. For most constants, the value returned
 * is an #hb_position_t.
 *
 * However, if the requested constant is #HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN,
 * #HB_OT_MATH_CONSTANT_SCRIPT_SCRIPT_PERCENT_SCALE_DOWN or
 * #HB_OT_MATH_CONSTANT_SCRIPT_PERCENT_SCALE_DOWN, then the return value is
 * an integer between 0 and 100 representing that percentage.
 *
 * Return value: the requested constant or zero
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_constant (hb_font_t *font,
			 hb_ot_math_constant_t constant)
{
  return font->face->table.MATH->get_constant(constant, font);
}

/**
 * hb_ot_math_get_glyph_italics_correction:
 * @font: #hb_font_t to work upon
 * @glyph: The glyph index from which to retrieve the value
 *
 * Fetches an italics-correction value (if one exists) for the specified
 * glyph index.
 *
  * Return value: the italics correction of the glyph or zero
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_italics_correction (hb_font_t *font,
					 hb_codepoint_t glyph)
{
  return font->face->table.MATH->get_glyph_info().get_italics_correction (glyph, font);
}

/**
 * hb_ot_math_get_glyph_top_accent_attachment:
 * @font: #hb_font_t to work upon
 * @glyph: The glyph index from which to retrieve the value
 *
 * Fetches a top-accent-attachment value (if one exists) for the specified
 * glyph index.
 *
 * For any glyph that does not have a top-accent-attachment value - that is,
 * a glyph not covered by the `MathTopAccentAttachment` table (or, when
 * @font has no `MathTopAccentAttachment` table or no `MATH` table, any
 * glyph) - the function synthesizes a value, returning the position at
 * one-half the glyph's advance width.
 *
 * Return value: the top accent attachment of the glyph or 0.5 * the advance
 *               width of @glyph
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_top_accent_attachment (hb_font_t *font,
					    hb_codepoint_t glyph)
{
  return font->face->table.MATH->get_glyph_info().get_top_accent_attachment (glyph, font);
}

/**
 * hb_ot_math_is_glyph_extended_shape:
 * @face: #hb_face_t to work upon
 * @glyph: The glyph index to test
 *
 * Tests whether the given glyph index is an extended shape in the face.
 *
 * Return value: true if the glyph is an extended shape, false otherwise
 *
 * Since: 1.3.3
 **/
hb_bool_t
hb_ot_math_is_glyph_extended_shape (hb_face_t *face,
				    hb_codepoint_t glyph)
{
  return face->table.MATH->get_glyph_info().is_extended_shape (glyph);
}

/**
 * hb_ot_math_get_glyph_kerning:
 * @font: #hb_font_t to work upon
 * @glyph: The glyph index from which to retrieve the value
 * @kern: The #hb_ot_math_kern_t from which to retrieve the value
 * @correction_height: the correction height to use to determine the kerning.
 *
 * Fetches the math kerning (cut-ins) value for the specified font, glyph index, and
 * @kern. 
 *
 * If the MathKern table is found, the function examines it to find a height
 * value that is greater or equal to @correction_height. If such a height
 * value is found, corresponding kerning value from the table is returned. If
 * no such height value is found, the last kerning value is returned.
 *
 * Return value: requested kerning value or zero
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_glyph_kerning (hb_font_t *font,
			      hb_codepoint_t glyph,
			      hb_ot_math_kern_t kern,
			      hb_position_t correction_height)
{
  return font->face->table.MATH->get_glyph_info().get_kerning (glyph,
							       kern,
							       correction_height,
							       font);
}

/**
 * hb_ot_math_get_glyph_variants:
 * @font: #hb_font_t to work upon
 * @glyph: The index of the glyph to stretch
 * @direction: The direction of the stretching (horizontal or vertical)
 * @start_offset: offset of the first variant to retrieve
 * @variants_count: (inout): Input = the maximum number of variants to return;
 *                           Output = the actual number of variants returned
 * @variants: (out) (array length=variants_count): array of variants returned
 *
 * Fetches the MathGlyphConstruction for the specified font, glyph index, and
 * direction. The corresponding list of size variants is returned as a list of
 * #hb_ot_math_glyph_variant_t structs.
 *
 * <note>The @direction parameter is only used to select between horizontal
 * or vertical directions for the construction. Even though all #hb_direction_t
 * values are accepted, only the result of #HB_DIRECTION_IS_HORIZONTAL is
 * considered.</note> 
 *
 * Return value: the total number of size variants available or zero
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
  return font->face->table.MATH->get_variants().get_glyph_variants (glyph, direction, font,
								    start_offset,
								    variants_count,
								    variants);
}

/**
 * hb_ot_math_get_min_connector_overlap:
 * @font: #hb_font_t to work upon
 * @direction: direction of the stretching (horizontal or vertical)
 *
 * Fetches the MathVariants table for the specified font and returns the
 * minimum overlap of connecting glyphs that are required to draw a glyph
 * assembly in the specified direction.
 *
 * <note>The @direction parameter is only used to select between horizontal
 * or vertical directions for the construction. Even though all #hb_direction_t
 * values are accepted, only the result of #HB_DIRECTION_IS_HORIZONTAL is
 * considered.</note> 
 *
 * Return value: requested minimum connector overlap or zero
 *
 * Since: 1.3.3
 **/
hb_position_t
hb_ot_math_get_min_connector_overlap (hb_font_t *font,
				      hb_direction_t direction)
{
  return font->face->table.MATH->get_variants().get_min_connector_overlap (direction, font);
}

/**
 * hb_ot_math_get_glyph_assembly:
 * @font: #hb_font_t to work upon
 * @glyph: The index of the glyph to stretch
 * @direction: direction of the stretching (horizontal or vertical)
 * @start_offset: offset of the first glyph part to retrieve
 * @parts_count: (inout): Input = maximum number of glyph parts to return;
 *               Output = actual number of parts returned
 * @parts: (out) (array length=parts_count): the glyph parts returned
 * @italics_correction: (out): italics correction of the glyph assembly
 *
 * Fetches the GlyphAssembly for the specified font, glyph index, and direction.
 * Returned are a list of #hb_ot_math_glyph_part_t glyph parts that can be
 * used to draw the glyph and an italics-correction value (if one is defined
 * in the font).
 *
 * <note>The @direction parameter is only used to select between horizontal
 * or vertical directions for the construction. Even though all #hb_direction_t
 * values are accepted, only the result of #HB_DIRECTION_IS_HORIZONTAL is
 * considered.</note> 
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
  return font->face->table.MATH->get_variants().get_glyph_parts (glyph,
								 direction,
								 font,
								 start_offset,
								 parts_count,
								 parts,
								 italics_correction);
}


#endif
