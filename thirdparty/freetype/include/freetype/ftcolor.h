/****************************************************************************
 *
 * ftcolor.h
 *
 *   FreeType's glyph color management (specification).
 *
 * Copyright (C) 2018-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTCOLOR_H_
#define FTCOLOR_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   color_management
   *
   * @title:
   *   Glyph Color Management
   *
   * @abstract:
   *   Retrieving and manipulating OpenType's 'CPAL' table data.
   *
   * @description:
   *   The functions described here allow access and manipulation of color
   *   palette entries in OpenType's 'CPAL' tables.
   */


  /**************************************************************************
   *
   * @struct:
   *   FT_Color
   *
   * @description:
   *   This structure models a BGRA color value of a 'CPAL' palette entry.
   *
   *   The used color space is sRGB; the colors are not pre-multiplied, and
   *   alpha values must be explicitly set.
   *
   * @fields:
   *   blue ::
   *     Blue value.
   *
   *   green ::
   *     Green value.
   *
   *   red ::
   *     Red value.
   *
   *   alpha ::
   *     Alpha value, giving the red, green, and blue color's opacity.
   *
   * @since:
   *   2.10
   */
  typedef struct  FT_Color_
  {
    FT_Byte  blue;
    FT_Byte  green;
    FT_Byte  red;
    FT_Byte  alpha;

  } FT_Color;


  /**************************************************************************
   *
   * @enum:
   *   FT_PALETTE_XXX
   *
   * @description:
   *   A list of bit field constants used in the `palette_flags` array of the
   *   @FT_Palette_Data structure to indicate for which background a palette
   *   with a given index is usable.
   *
   * @values:
   *   FT_PALETTE_FOR_LIGHT_BACKGROUND ::
   *     The palette is appropriate to use when displaying the font on a
   *     light background such as white.
   *
   *   FT_PALETTE_FOR_DARK_BACKGROUND ::
   *     The palette is appropriate to use when displaying the font on a dark
   *     background such as black.
   *
   * @since:
   *   2.10
   */
#define FT_PALETTE_FOR_LIGHT_BACKGROUND  0x01
#define FT_PALETTE_FOR_DARK_BACKGROUND   0x02


  /**************************************************************************
   *
   * @struct:
   *   FT_Palette_Data
   *
   * @description:
   *   This structure holds the data of the 'CPAL' table.
   *
   * @fields:
   *   num_palettes ::
   *     The number of palettes.
   *
   *   palette_name_ids ::
   *     An optional read-only array of palette name IDs with `num_palettes`
   *     elements, corresponding to entries like 'dark' or 'light' in the
   *     font's 'name' table.
   *
   *     An empty name ID in the 'CPAL' table gets represented as value
   *     0xFFFF.
   *
   *     `NULL` if the font's 'CPAL' table doesn't contain appropriate data.
   *
   *   palette_flags ::
   *     An optional read-only array of palette flags with `num_palettes`
   *     elements.  Possible values are an ORed combination of
   *     @FT_PALETTE_FOR_LIGHT_BACKGROUND and
   *     @FT_PALETTE_FOR_DARK_BACKGROUND.
   *
   *     `NULL` if the font's 'CPAL' table doesn't contain appropriate data.
   *
   *   num_palette_entries ::
   *     The number of entries in a single palette.  All palettes have the
   *     same size.
   *
   *   palette_entry_name_ids ::
   *     An optional read-only array of palette entry name IDs with
   *     `num_palette_entries`.  In each palette, entries with the same index
   *     have the same function.  For example, index~0 might correspond to
   *     string 'outline' in the font's 'name' table to indicate that this
   *     palette entry is used for outlines, index~1 might correspond to
   *     'fill' to indicate the filling color palette entry, etc.
   *
   *     An empty entry name ID in the 'CPAL' table gets represented as value
   *     0xFFFF.
   *
   *     `NULL` if the font's 'CPAL' table doesn't contain appropriate data.
   *
   * @note:
   *   Use function @FT_Get_Sfnt_Name to map name IDs and entry name IDs to
   *   name strings.
   *
   *   Use function @FT_Palette_Select to get the colors associated with a
   *   palette entry.
   *
   * @since:
   *   2.10
   */
  typedef struct  FT_Palette_Data_ {
    FT_UShort         num_palettes;
    const FT_UShort*  palette_name_ids;
    const FT_UShort*  palette_flags;

    FT_UShort         num_palette_entries;
    const FT_UShort*  palette_entry_name_ids;

  } FT_Palette_Data;


  /**************************************************************************
   *
   * @function:
   *   FT_Palette_Data_Get
   *
   * @description:
   *   Retrieve the face's color palette data.
   *
   * @input:
   *   face ::
   *     The source face handle.
   *
   * @output:
   *   apalette ::
   *     A pointer to an @FT_Palette_Data structure.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   All arrays in the returned @FT_Palette_Data structure are read-only.
   *
   *   This function always returns an error if the config macro
   *   `TT_CONFIG_OPTION_COLOR_LAYERS` is not defined in `ftoption.h`.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Palette_Data_Get( FT_Face           face,
                       FT_Palette_Data  *apalette );


  /**************************************************************************
   *
   * @function:
   *   FT_Palette_Select
   *
   * @description:
   *   This function has two purposes.
   *
   *   (1) It activates a palette for rendering color glyphs, and
   *
   *   (2) it retrieves all (unmodified) color entries of this palette.  This
   *       function returns a read-write array, which means that a calling
   *       application can modify the palette entries on demand.
   *
   * A corollary of (2) is that calling the function, then modifying some
   * values, then calling the function again with the same arguments resets
   * all color entries to the original 'CPAL' values; all user modifications
   * are lost.
   *
   * @input:
   *   face ::
   *     The source face handle.
   *
   *   palette_index ::
   *     The palette index.
   *
   * @output:
   *   apalette ::
   *     An array of color entries for a palette with index `palette_index`,
   *     having `num_palette_entries` elements (as found in the
   *     `FT_Palette_Data` structure).  If `apalette` is set to `NULL`, no
   *     array gets returned (and no color entries can be modified).
   *
   *     In case the font doesn't support color palettes, `NULL` is returned.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The array pointed to by `apalette_entries` is owned and managed by
   *   FreeType.
   *
   *   This function always returns an error if the config macro
   *   `TT_CONFIG_OPTION_COLOR_LAYERS` is not defined in `ftoption.h`.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Palette_Select( FT_Face     face,
                     FT_UShort   palette_index,
                     FT_Color*  *apalette );


  /**************************************************************************
   *
   * @function:
   *   FT_Palette_Set_Foreground_Color
   *
   * @description:
   *   'COLR' uses palette index 0xFFFF to indicate a 'text foreground
   *   color'.  This function sets this value.
   *
   * @input:
   *   face ::
   *     The source face handle.
   *
   *   foreground_color ::
   *     An `FT_Color` structure to define the text foreground color.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   If this function isn't called, the text foreground color is set to
   *   white opaque (BGRA value 0xFFFFFFFF) if
   *   @FT_PALETTE_FOR_DARK_BACKGROUND is present for the current palette,
   *   and black opaque (BGRA value 0x000000FF) otherwise, including the case
   *   that no palette types are available in the 'CPAL' table.
   *
   *   This function always returns an error if the config macro
   *   `TT_CONFIG_OPTION_COLOR_LAYERS` is not defined in `ftoption.h`.
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Error )
  FT_Palette_Set_Foreground_Color( FT_Face   face,
                                   FT_Color  foreground_color );

  /* */


FT_END_HEADER

#endif /* FTCOLOR_H_ */


/* END */
