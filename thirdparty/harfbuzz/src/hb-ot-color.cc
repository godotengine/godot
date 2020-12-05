/*
 * Copyright © 2016  Google, Inc.
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
 *
 * Google Author(s): Sascha Brawer, Behdad Esfahbod
 */

#include "hb.hh"

#ifndef HB_NO_COLOR

#include "hb-ot.h"

#include "hb-ot-color-cbdt-table.hh"
#include "hb-ot-color-colr-table.hh"
#include "hb-ot-color-cpal-table.hh"
#include "hb-ot-color-sbix-table.hh"
#include "hb-ot-color-svg-table.hh"

#include <stdlib.h>
#include <string.h>


/**
 * SECTION:hb-ot-color
 * @title: hb-ot-color
 * @short_description: OpenType Color Fonts
 * @include: hb-ot.h
 *
 * Functions for fetching color-font information from OpenType font faces.
 *
 * HarfBuzz supports `COLR`/`CPAL`, `sbix`, `CBDT`, and `SVG` color fonts.
 **/


/*
 * CPAL
 */


/**
 * hb_ot_color_has_palettes:
 * @face: #hb_face_t to work upon
 *
 * Tests whether a face includes a `CPAL` color-palette table.
 *
 * Return value: true if data found, false otherwise
 *
 * Since: 2.1.0
 */
hb_bool_t
hb_ot_color_has_palettes (hb_face_t *face)
{
  return face->table.CPAL->has_data ();
}

/**
 * hb_ot_color_palette_get_count:
 * @face: #hb_face_t to work upon
 *
 * Fetches the number of color palettes in a face.
 *
 * Return value: the number of palettes found
 *
 * Since: 2.1.0
 */
unsigned int
hb_ot_color_palette_get_count (hb_face_t *face)
{
  return face->table.CPAL->get_palette_count ();
}

/**
 * hb_ot_color_palette_get_name_id:
 * @face: #hb_face_t to work upon
 * @palette_index: The index of the color palette 
 *
 * Fetches the `name` table Name ID that provides display names for
 * a `CPAL` color palette. 
 *
 * Palette display names can be generic (e.g., "Default") or provide
 * specific, themed names (e.g., "Spring", "Summer", "Fall", and "Winter").
 *
 * Return value: the Named ID found for the palette. 
 * If the requested palette has no name the result is #HB_OT_NAME_ID_INVALID.
 *
 * Since: 2.1.0
 */
hb_ot_name_id_t
hb_ot_color_palette_get_name_id (hb_face_t *face,
				 unsigned int palette_index)
{
  return face->table.CPAL->get_palette_name_id (palette_index);
}

/**
 * hb_ot_color_palette_color_get_name_id:
 * @face: #hb_face_t to work upon
 * @color_index: The index of the color
 *
 * Fetches the `name` table Name ID that provides display names for
 * the specificed color in a face's `CPAL` color palette. 
 *
 * Display names can be generic (e.g., "Background") or specific
 * (e.g., "Eye color").
 *
 * Return value: the Name ID found for the color.
 *
 * Since: 2.1.0
 */
hb_ot_name_id_t
hb_ot_color_palette_color_get_name_id (hb_face_t *face,
				       unsigned int color_index)
{
  return face->table.CPAL->get_color_name_id (color_index);
}

/**
 * hb_ot_color_palette_get_flags:
 * @face: #hb_face_t to work upon
 * @palette_index: The index of the color palette
 *
 * Fetches the flags defined for a color palette.
 *
 * Return value: the #hb_ot_color_palette_flags_t of the requested color palette
 *
 * Since: 2.1.0
 */
hb_ot_color_palette_flags_t
hb_ot_color_palette_get_flags (hb_face_t *face,
			       unsigned int palette_index)
{
  return face->table.CPAL->get_palette_flags (palette_index);
}

/**
 * hb_ot_color_palette_get_colors:
 * @face: #hb_face_t to work upon
 * @palette_index: the index of the color palette to query
 * @start_offset: offset of the first color to retrieve
 * @color_count: (inout) (optional): Input = the maximum number of colors to return;
 *               Output = the actual number of colors returned (may be zero)
 * @colors: (out) (array length=color_count) (nullable): The array of #hb_color_t records found
 *
 * Fetches a list of the colors in a color palette.
 *
 * After calling this function, @colors will be filled with the palette
 * colors. If @colors is NULL, the function will just return the number
 * of total colors without storing any actual colors; this can be used
 * for allocating a buffer of suitable size before calling
 * hb_ot_color_palette_get_colors() a second time.
 *
 * Return value: the total number of colors in the palette
 *
 * Since: 2.1.0
 */
unsigned int
hb_ot_color_palette_get_colors (hb_face_t     *face,
				unsigned int   palette_index,
				unsigned int   start_offset,
				unsigned int  *colors_count  /* IN/OUT.  May be NULL. */,
				hb_color_t    *colors        /* OUT.     May be NULL. */)
{
  return face->table.CPAL->get_palette_colors (palette_index, start_offset, colors_count, colors);
}


/*
 * COLR
 */

/**
 * hb_ot_color_has_layers:
 * @face: #hb_face_t to work upon
 *
 * Tests whether a face includes any `COLR` color layers.
 *
 * Return value: true if data found, false otherwise
 *
 * Since: 2.1.0
 */
hb_bool_t
hb_ot_color_has_layers (hb_face_t *face)
{
  return face->table.COLR->has_data ();
}

/**
 * hb_ot_color_glyph_get_layers:
 * @face: #hb_face_t to work upon
 * @glyph: The glyph index to query
 * @start_offset: offset of the first layer to retrieve
 * @layer_count: (inout) (optional): Input = the maximum number of layers to return;
 *         Output = the actual number of layers returned (may be zero)
 * @layers: (out) (array length=layer_count) (nullable): The array of layers found
 *
 * Fetches a list of all color layers for the specified glyph index in the specified
 * face. The list returned will begin at the offset provided.
 *
 * Return value: Total number of layers available for the glyph index queried
 *
 * Since: 2.1.0
 */
unsigned int
hb_ot_color_glyph_get_layers (hb_face_t           *face,
			      hb_codepoint_t       glyph,
			      unsigned int         start_offset,
			      unsigned int        *layer_count, /* IN/OUT.  May be NULL. */
			      hb_ot_color_layer_t *layers /* OUT.     May be NULL. */)
{
  return face->table.COLR->get_glyph_layers (glyph, start_offset, layer_count, layers);
}


/*
 * SVG
 */

/**
 * hb_ot_color_has_svg:
 * @face: #hb_face_t to work upon.
 *
 * Tests whether a face includes any `SVG` glyph images.
 *
 * Return value: true if data found, false otherwise.
 *
 * Since: 2.1.0
 */
hb_bool_t
hb_ot_color_has_svg (hb_face_t *face)
{
  return face->table.SVG->has_data ();
}

/**
 * hb_ot_color_glyph_reference_svg:
 * @face: #hb_face_t to work upon
 * @glyph: a svg glyph index
 *
 * Fetches the SVG document for a glyph. The blob may be either plain text or gzip-encoded.
 *
 * Return value: (transfer full): An #hb_blob_t containing the SVG document of the glyph, if available
 *
 * Since: 2.1.0
 */
hb_blob_t *
hb_ot_color_glyph_reference_svg (hb_face_t *face, hb_codepoint_t glyph)
{
  return face->table.SVG->reference_blob_for_glyph (glyph);
}


/*
 * PNG: CBDT or sbix
 */

/**
 * hb_ot_color_has_png:
 * @face: #hb_face_t to work upon
 *
 * Tests whether a face has PNG glyph images (either in `CBDT` or `sbix` tables).
 *
 * Return value: true if data found, false otherwise
 *
 * Since: 2.1.0
 */
hb_bool_t
hb_ot_color_has_png (hb_face_t *face)
{
  return face->table.CBDT->has_data () || face->table.sbix->has_data ();
}

/**
 * hb_ot_color_glyph_reference_png:
 * @font: #hb_font_t to work upon
 * @glyph: a glyph index
 *
 * Fetches the PNG image for a glyph. This function takes a font object, not a face object,
 * as input. To get an optimally sized PNG blob, the UPEM value must be set on the @font
 * object. If UPEM is unset, the blob returned will be the largest PNG available.
 *
 * Return value: (transfer full): An #hb_blob_t containing the PNG image for the glyph, if available
 *
 * Since: 2.1.0
 */
hb_blob_t *
hb_ot_color_glyph_reference_png (hb_font_t *font, hb_codepoint_t  glyph)
{
  hb_blob_t *blob = hb_blob_get_empty ();

  if (font->face->table.sbix->has_data ())
    blob = font->face->table.sbix->reference_png (font, glyph, nullptr, nullptr, nullptr);

  if (!blob->length && font->face->table.CBDT->has_data ())
    blob = font->face->table.CBDT->reference_png (font, glyph);

  return blob;
}


#endif
