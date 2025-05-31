/****************************************************************************
 *
 * ftcolor.h
 *
 *   FreeType's glyph color management (specification).
 *
 * Copyright (C) 2018-2024 by
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

#include <freetype/freetype.h>

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


  /**************************************************************************
   *
   * @section:
   *   layer_management
   *
   * @title:
   *   Glyph Layer Management
   *
   * @abstract:
   *   Retrieving and manipulating OpenType's 'COLR' table data.
   *
   * @description:
   *   The functions described here allow access of colored glyph layer data
   *   in OpenType's 'COLR' tables.
   */


  /**************************************************************************
   *
   * @struct:
   *   FT_LayerIterator
   *
   * @description:
   *   This iterator object is needed for @FT_Get_Color_Glyph_Layer.
   *
   * @fields:
   *   num_layers ::
   *     The number of glyph layers for the requested glyph index.  Will be
   *     set by @FT_Get_Color_Glyph_Layer.
   *
   *   layer ::
   *     The current layer.  Will be set by @FT_Get_Color_Glyph_Layer.
   *
   *   p ::
   *     An opaque pointer into 'COLR' table data.  The caller must set this
   *     to `NULL` before the first call of @FT_Get_Color_Glyph_Layer.
   */
  typedef struct  FT_LayerIterator_
  {
    FT_UInt   num_layers;
    FT_UInt   layer;
    FT_Byte*  p;

  } FT_LayerIterator;


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Color_Glyph_Layer
   *
   * @description:
   *   This is an interface to the 'COLR' table in OpenType fonts to
   *   iteratively retrieve the colored glyph layers associated with the
   *   current glyph slot.
   *
   *     https://docs.microsoft.com/en-us/typography/opentype/spec/colr
   *
   *   The glyph layer data for a given glyph index, if present, provides an
   *   alternative, multi-color glyph representation: Instead of rendering
   *   the outline or bitmap with the given glyph index, glyphs with the
   *   indices and colors returned by this function are rendered layer by
   *   layer.
   *
   *   The returned elements are ordered in the z~direction from bottom to
   *   top; the 'n'th element should be rendered with the associated palette
   *   color and blended on top of the already rendered layers (elements 0,
   *   1, ..., n-1).
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   *   base_glyph ::
   *     The glyph index the colored glyph layers are associated with.
   *
   * @inout:
   *   iterator ::
   *     An @FT_LayerIterator object.  For the first call you should set
   *     `iterator->p` to `NULL`.  For all following calls, simply use the
   *     same object again.
   *
   * @output:
   *   aglyph_index ::
   *     The glyph index of the current layer.
   *
   *   acolor_index ::
   *     The color index into the font face's color palette of the current
   *     layer.  The value 0xFFFF is special; it doesn't reference a palette
   *     entry but indicates that the text foreground color should be used
   *     instead (to be set up by the application outside of FreeType).
   *
   *     The color palette can be retrieved with @FT_Palette_Select.
   *
   * @return:
   *   Value~1 if everything is OK.  If there are no more layers (or if there
   *   are no layers at all), value~0 gets returned.  In case of an error,
   *   value~0 is returned also.
   *
   * @note:
   *   This function is necessary if you want to handle glyph layers by
   *   yourself.  In particular, functions that operate with @FT_GlyphRec
   *   objects (like @FT_Get_Glyph or @FT_Glyph_To_Bitmap) don't have access
   *   to this information.
   *
   *   Note that @FT_Render_Glyph is able to handle colored glyph layers
   *   automatically if the @FT_LOAD_COLOR flag is passed to a previous call
   *   to @FT_Load_Glyph.  [This is an experimental feature.]
   *
   * @example:
   *   ```
   *     FT_Color*         palette;
   *     FT_LayerIterator  iterator;
   *
   *     FT_Bool  have_layers;
   *     FT_UInt  layer_glyph_index;
   *     FT_UInt  layer_color_index;
   *
   *
   *     error = FT_Palette_Select( face, palette_index, &palette );
   *     if ( error )
   *       palette = NULL;
   *
   *     iterator.p  = NULL;
   *     have_layers = FT_Get_Color_Glyph_Layer( face,
   *                                             glyph_index,
   *                                             &layer_glyph_index,
   *                                             &layer_color_index,
   *                                             &iterator );
   *
   *     if ( palette && have_layers )
   *     {
   *       do
   *       {
   *         FT_Color  layer_color;
   *
   *
   *         if ( layer_color_index == 0xFFFF )
   *           layer_color = text_foreground_color;
   *         else
   *           layer_color = palette[layer_color_index];
   *
   *         // Load and render glyph `layer_glyph_index', then
   *         // blend resulting pixmap (using color `layer_color')
   *         // with previously created pixmaps.
   *
   *       } while ( FT_Get_Color_Glyph_Layer( face,
   *                                           glyph_index,
   *                                           &layer_glyph_index,
   *                                           &layer_color_index,
   *                                           &iterator ) );
   *     }
   *   ```
   *
   * @since:
   *   2.10
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Color_Glyph_Layer( FT_Face            face,
                            FT_UInt            base_glyph,
                            FT_UInt           *aglyph_index,
                            FT_UInt           *acolor_index,
                            FT_LayerIterator*  iterator );


  /**************************************************************************
   *
   * @enum:
   *   FT_PaintFormat
   *
   * @description:
   *   Enumeration describing the different paint format types of the v1
   *   extensions to the 'COLR' table, see
   *   'https://github.com/googlefonts/colr-gradients-spec'.
   *
   *   The enumeration values loosely correspond with the format numbers of
   *   the specification: FreeType always returns a fully specified 'Paint'
   *   structure for the 'Transform', 'Translate', 'Scale', 'Rotate', and
   *   'Skew' table types even though the specification has different formats
   *   depending on whether or not a center is specified, whether the scale
   *   is uniform in x and y~direction or not, etc.  Also, only non-variable
   *   format identifiers are listed in this enumeration; as soon as support
   *   for variable 'COLR' v1 fonts is implemented, interpolation is
   *   performed dependent on axis coordinates, which are configured on the
   *   @FT_Face through @FT_Set_Var_Design_Coordinates.  This implies that
   *   always static, readily interpolated values are returned in the 'Paint'
   *   structures.
   *
   * @since:
   *   2.13
   */
  typedef enum  FT_PaintFormat_
  {
    FT_COLR_PAINTFORMAT_COLR_LAYERS     = 1,
    FT_COLR_PAINTFORMAT_SOLID           = 2,
    FT_COLR_PAINTFORMAT_LINEAR_GRADIENT = 4,
    FT_COLR_PAINTFORMAT_RADIAL_GRADIENT = 6,
    FT_COLR_PAINTFORMAT_SWEEP_GRADIENT  = 8,
    FT_COLR_PAINTFORMAT_GLYPH           = 10,
    FT_COLR_PAINTFORMAT_COLR_GLYPH      = 11,
    FT_COLR_PAINTFORMAT_TRANSFORM       = 12,
    FT_COLR_PAINTFORMAT_TRANSLATE       = 14,
    FT_COLR_PAINTFORMAT_SCALE           = 16,
    FT_COLR_PAINTFORMAT_ROTATE          = 24,
    FT_COLR_PAINTFORMAT_SKEW            = 28,
    FT_COLR_PAINTFORMAT_COMPOSITE       = 32,
    FT_COLR_PAINT_FORMAT_MAX            = 33,
    FT_COLR_PAINTFORMAT_UNSUPPORTED     = 255

  } FT_PaintFormat;


  /**************************************************************************
   *
   * @struct:
   *   FT_ColorStopIterator
   *
   * @description:
   *   This iterator object is needed for @FT_Get_Colorline_Stops.  It keeps
   *   state while iterating over the stops of an @FT_ColorLine, representing
   *   the `ColorLine` struct of the v1 extensions to 'COLR', see
   *   'https://github.com/googlefonts/colr-gradients-spec'.  Do not manually
   *   modify fields of this iterator.
   *
   * @fields:
   *   num_color_stops ::
   *     The number of color stops for the requested glyph index.  Set by
   *     @FT_Get_Paint.
   *
   *   current_color_stop ::
   *     The current color stop.  Set by @FT_Get_Colorline_Stops.
   *
   *   p ::
   *     An opaque pointer into 'COLR' table data.  Set by @FT_Get_Paint.
   *     Updated by @FT_Get_Colorline_Stops.
   *
   *   read_variable ::
   *     A boolean keeping track of whether variable color lines are to be
   *     read.  Set by @FT_Get_Paint.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_ColorStopIterator_
  {
    FT_UInt  num_color_stops;
    FT_UInt  current_color_stop;

    FT_Byte*  p;

    FT_Bool  read_variable;

  } FT_ColorStopIterator;


  /**************************************************************************
   *
   * @struct:
   *   FT_ColorIndex
   *
   * @description:
   *   A structure representing a `ColorIndex` value of the 'COLR' v1
   *   extensions, see 'https://github.com/googlefonts/colr-gradients-spec'.
   *
   * @fields:
   *   palette_index ::
   *     The palette index into a 'CPAL' palette.
   *
   *   alpha ::
   *     Alpha transparency value multiplied with the value from 'CPAL'.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_ColorIndex_
  {
    FT_UInt16   palette_index;
    FT_F2Dot14  alpha;

  } FT_ColorIndex;


  /**************************************************************************
   *
   * @struct:
   *   FT_ColorStop
   *
   * @description:
   *   A structure representing a `ColorStop` value of the 'COLR' v1
   *   extensions, see 'https://github.com/googlefonts/colr-gradients-spec'.
   *
   * @fields:
   *   stop_offset ::
   *     The stop offset along the gradient, expressed as a 16.16 fixed-point
   *     coordinate.
   *
   *   color ::
   *     The color information for this stop, see @FT_ColorIndex.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_ColorStop_
  {
    FT_Fixed       stop_offset;
    FT_ColorIndex  color;

  } FT_ColorStop;


  /**************************************************************************
   *
   * @enum:
   *   FT_PaintExtend
   *
   * @description:
   *   An enumeration representing the 'Extend' mode of the 'COLR' v1
   *   extensions, see 'https://github.com/googlefonts/colr-gradients-spec'.
   *   It describes how the gradient fill continues at the other boundaries.
   *
   * @since:
   *   2.13
   */
  typedef enum  FT_PaintExtend_
  {
    FT_COLR_PAINT_EXTEND_PAD     = 0,
    FT_COLR_PAINT_EXTEND_REPEAT  = 1,
    FT_COLR_PAINT_EXTEND_REFLECT = 2

  } FT_PaintExtend;


  /**************************************************************************
   *
   * @struct:
   *   FT_ColorLine
   *
   * @description:
   *   A structure representing a `ColorLine` value of the 'COLR' v1
   *   extensions, see 'https://github.com/googlefonts/colr-gradients-spec'.
   *   It describes a list of color stops along the defined gradient.
   *
   * @fields:
   *   extend ::
   *     The extend mode at the outer boundaries, see @FT_PaintExtend.
   *
   *   color_stop_iterator ::
   *     The @FT_ColorStopIterator used to enumerate and retrieve the
   *     actual @FT_ColorStop's.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_ColorLine_
  {
    FT_PaintExtend        extend;
    FT_ColorStopIterator  color_stop_iterator;

  } FT_ColorLine;


  /**************************************************************************
   *
   * @struct:
   *   FT_Affine23
   *
   * @description:
   *   A structure used to store a 2x3 matrix.  Coefficients are in
   *   16.16 fixed-point format.  The computation performed is
   *
   *   ```
   *     x' = x*xx + y*xy + dx
   *     y' = x*yx + y*yy + dy
   *   ```
   *
   * @fields:
   *   xx ::
   *     Matrix coefficient.
   *
   *   xy ::
   *     Matrix coefficient.
   *
   *   dx ::
   *     x translation.
   *
   *   yx ::
   *     Matrix coefficient.
   *
   *   yy ::
   *     Matrix coefficient.
   *
   *   dy ::
   *     y translation.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_Affine_23_
  {
    FT_Fixed  xx, xy, dx;
    FT_Fixed  yx, yy, dy;

  } FT_Affine23;


  /**************************************************************************
   *
   * @enum:
   *   FT_Composite_Mode
   *
   * @description:
   *   An enumeration listing the 'COLR' v1 composite modes used in
   *   @FT_PaintComposite.  For more details on each paint mode, see
   *   'https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators'.
   *
   * @since:
   *   2.13
   */
  typedef enum  FT_Composite_Mode_
  {
    FT_COLR_COMPOSITE_CLEAR          = 0,
    FT_COLR_COMPOSITE_SRC            = 1,
    FT_COLR_COMPOSITE_DEST           = 2,
    FT_COLR_COMPOSITE_SRC_OVER       = 3,
    FT_COLR_COMPOSITE_DEST_OVER      = 4,
    FT_COLR_COMPOSITE_SRC_IN         = 5,
    FT_COLR_COMPOSITE_DEST_IN        = 6,
    FT_COLR_COMPOSITE_SRC_OUT        = 7,
    FT_COLR_COMPOSITE_DEST_OUT       = 8,
    FT_COLR_COMPOSITE_SRC_ATOP       = 9,
    FT_COLR_COMPOSITE_DEST_ATOP      = 10,
    FT_COLR_COMPOSITE_XOR            = 11,
    FT_COLR_COMPOSITE_PLUS           = 12,
    FT_COLR_COMPOSITE_SCREEN         = 13,
    FT_COLR_COMPOSITE_OVERLAY        = 14,
    FT_COLR_COMPOSITE_DARKEN         = 15,
    FT_COLR_COMPOSITE_LIGHTEN        = 16,
    FT_COLR_COMPOSITE_COLOR_DODGE    = 17,
    FT_COLR_COMPOSITE_COLOR_BURN     = 18,
    FT_COLR_COMPOSITE_HARD_LIGHT     = 19,
    FT_COLR_COMPOSITE_SOFT_LIGHT     = 20,
    FT_COLR_COMPOSITE_DIFFERENCE     = 21,
    FT_COLR_COMPOSITE_EXCLUSION      = 22,
    FT_COLR_COMPOSITE_MULTIPLY       = 23,
    FT_COLR_COMPOSITE_HSL_HUE        = 24,
    FT_COLR_COMPOSITE_HSL_SATURATION = 25,
    FT_COLR_COMPOSITE_HSL_COLOR      = 26,
    FT_COLR_COMPOSITE_HSL_LUMINOSITY = 27,
    FT_COLR_COMPOSITE_MAX            = 28

  } FT_Composite_Mode;


  /**************************************************************************
   *
   * @struct:
   *   FT_OpaquePaint
   *
   * @description:
   *   A structure representing an offset to a `Paint` value stored in any
   *   of the paint tables of a 'COLR' v1 font.  Compare Offset<24> there.
   *   When 'COLR' v1 paint tables represented by FreeType objects such as
   *   @FT_PaintColrLayers, @FT_PaintComposite, or @FT_PaintTransform
   *   reference downstream nested paint tables, we do not immediately
   *   retrieve them but encapsulate their location in this type.  Use
   *   @FT_Get_Paint to retrieve the actual @FT_COLR_Paint object that
   *   describes the details of the respective paint table.
   *
   * @fields:
   *   p ::
   *     An internal offset to a Paint table, needs to be set to NULL before
   *     passing this struct as an argument to @FT_Get_Paint.
   *
   *   insert_root_transform ::
   *     An internal boolean to track whether an initial root transform is
   *     to be provided.  Do not set this value.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_Opaque_Paint_
  {
    FT_Byte*  p;
    FT_Bool   insert_root_transform;
  } FT_OpaquePaint;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintColrLayers
   *
   * @description:
   *   A structure representing a `PaintColrLayers` table of a 'COLR' v1
   *   font.  This table describes a set of layers that are to be composited
   *   with composite mode `FT_COLR_COMPOSITE_SRC_OVER`.  The return value
   *   of this function is an @FT_LayerIterator initialized so that it can
   *   be used with @FT_Get_Paint_Layers to retrieve the @FT_OpaquePaint
   *   objects as references to each layer.
   *
   * @fields:
   *   layer_iterator ::
   *     The layer iterator that describes the layers of this paint.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintColrLayers_
  {
    FT_LayerIterator  layer_iterator;

  } FT_PaintColrLayers;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintSolid
   *
   * @description:
   *   A structure representing a `PaintSolid` value of the 'COLR' v1
   *   extensions, see 'https://github.com/googlefonts/colr-gradients-spec'.
   *   Using a `PaintSolid` value means that the glyph layer filled with
   *   this paint is solid-colored and does not contain a gradient.
   *
   * @fields:
   *   color ::
   *     The color information for this solid paint, see @FT_ColorIndex.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintSolid_
  {
    FT_ColorIndex  color;

  } FT_PaintSolid;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintLinearGradient
   *
   * @description:
   *   A structure representing a `PaintLinearGradient` value of the 'COLR'
   *   v1 extensions, see
   *   'https://github.com/googlefonts/colr-gradients-spec'.  The glyph
   *   layer filled with this paint is drawn filled with a linear gradient.
   *
   * @fields:
   *   colorline ::
   *     The @FT_ColorLine information for this paint, i.e., the list of
   *     color stops along the gradient.
   *
   *   p0 ::
   *     The starting point of the gradient definition in font units
   *     represented as a 16.16 fixed-point `FT_Vector`.
   *
   *   p1 ::
   *     The end point of the gradient definition in font units
   *     represented as a 16.16 fixed-point `FT_Vector`.
   *
   *   p2 ::
   *     Optional point~p2 to rotate the gradient in font units
   *     represented as a 16.16 fixed-point `FT_Vector`.
   *     Otherwise equal to~p0.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintLinearGradient_
  {
    FT_ColorLine  colorline;

    /* TODO: Potentially expose those as x0, y0 etc. */
    FT_Vector  p0;
    FT_Vector  p1;
    FT_Vector  p2;

  } FT_PaintLinearGradient;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintRadialGradient
   *
   * @description:
   *   A structure representing a `PaintRadialGradient` value of the 'COLR'
   *   v1 extensions, see
   *   'https://github.com/googlefonts/colr-gradients-spec'.  The glyph
   *   layer filled with this paint is drawn filled with a radial gradient.
   *
   * @fields:
   *   colorline ::
   *     The @FT_ColorLine information for this paint, i.e., the list of
   *     color stops along the gradient.
   *
   *   c0 ::
   *     The center of the starting point of the radial gradient in font
   *     units represented as a 16.16 fixed-point `FT_Vector`.
   *
   *   r0 ::
   *     The radius of the starting circle of the radial gradient in font
   *     units represented as a 16.16 fixed-point value.
   *
   *   c1 ::
   *     The center of the end point of the radial gradient in font units
   *     represented as a 16.16 fixed-point `FT_Vector`.
   *
   *   r1 ::
   *     The radius of the end circle of the radial gradient in font
   *     units represented as a 16.16 fixed-point value.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintRadialGradient_
  {
    FT_ColorLine  colorline;

    FT_Vector  c0;
    FT_Pos     r0;
    FT_Vector  c1;
    FT_Pos     r1;

  } FT_PaintRadialGradient;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintSweepGradient
   *
   * @description:
   *   A structure representing a `PaintSweepGradient` value of the 'COLR'
   *   v1 extensions, see
   *   'https://github.com/googlefonts/colr-gradients-spec'.  The glyph
   *   layer filled with this paint is drawn filled with a sweep gradient
   *   from `start_angle` to `end_angle`.
   *
   * @fields:
   *   colorline ::
   *     The @FT_ColorLine information for this paint, i.e., the list of
   *     color stops along the gradient.
   *
   *   center ::
   *     The center of the sweep gradient in font units represented as a
   *     vector of 16.16 fixed-point values.
   *
   *   start_angle ::
   *     The start angle of the sweep gradient in 16.16 fixed-point
   *     format specifying degrees divided by 180.0 (as in the
   *     spec).  Multiply by 180.0f to receive degrees value.  Values are
   *     given counter-clockwise, starting from the (positive) y~axis.
   *
   *   end_angle ::
   *     The end angle of the sweep gradient in 16.16 fixed-point
   *     format specifying degrees divided by 180.0 (as in the
   *     spec).  Multiply by 180.0f to receive degrees value.  Values are
   *     given counter-clockwise, starting from the (positive) y~axis.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintSweepGradient_
  {
    FT_ColorLine  colorline;

    FT_Vector  center;
    FT_Fixed   start_angle;
    FT_Fixed   end_angle;

  } FT_PaintSweepGradient;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintGlyph
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintGlyph` paint table.
   *
   * @fields:
   *   paint ::
   *     An opaque paint object pointing to a `Paint` table that serves as
   *     the fill for the glyph ID.
   *
   *   glyphID ::
   *     The glyph ID from the 'glyf' table, which serves as the contour
   *     information that is filled with paint.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintGlyph_
  {
    FT_OpaquePaint  paint;
    FT_UInt         glyphID;

  } FT_PaintGlyph;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintColrGlyph
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintColorGlyph` paint table.
   *
   * @fields:
   *   glyphID ::
   *     The glyph ID from the `BaseGlyphV1List` table that is drawn for
   *     this paint.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintColrGlyph_
  {
    FT_UInt  glyphID;

  } FT_PaintColrGlyph;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintTransform
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintTransform` paint table.
   *
   * @fields:
   *   paint ::
   *     An opaque paint that is subject to being transformed.
   *
   *   affine ::
   *     A 2x3 transformation matrix in @FT_Affine23 format containing
   *     16.16 fixed-point values.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintTransform_
  {
    FT_OpaquePaint  paint;
    FT_Affine23     affine;

  } FT_PaintTransform;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintTranslate
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintTranslate` paint table.
   *   Used for translating downstream paints by a given x and y~delta.
   *
   * @fields:
   *   paint ::
   *     An @FT_OpaquePaint object referencing the paint that is to be
   *     rotated.
   *
   *   dx ::
   *     Translation in x~direction in font units represented as a
   *     16.16 fixed-point value.
   *
   *   dy ::
   *     Translation in y~direction in font units represented as a
   *     16.16 fixed-point value.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintTranslate_
  {
    FT_OpaquePaint  paint;

    FT_Fixed  dx;
    FT_Fixed  dy;

  } FT_PaintTranslate;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintScale
   *
   * @description:
   *   A structure representing all of the 'COLR' v1 'PaintScale*' paint
   *   tables.  Used for scaling downstream paints by a given x and y~scale,
   *   with a given center.  This structure is used for all 'PaintScale*'
   *   types that are part of specification; fields of this structure are
   *   filled accordingly.  If there is a center, the center values are set,
   *   otherwise they are set to the zero coordinate.  If the source font
   *   file has 'PaintScaleUniform*' set, the scale values are set
   *   accordingly to the same value.
   *
   * @fields:
   *   paint ::
   *     An @FT_OpaquePaint object referencing the paint that is to be
   *     scaled.
   *
   *   scale_x ::
   *     Scale factor in x~direction represented as a
   *     16.16 fixed-point value.
   *
   *   scale_y ::
   *     Scale factor in y~direction represented as a
   *     16.16 fixed-point value.
   *
   *   center_x ::
   *     x~coordinate of center point to scale from represented as a
   *     16.16 fixed-point value.
   *
   *   center_y ::
   *     y~coordinate of center point to scale from represented as a
   *     16.16 fixed-point value.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintScale_
  {
    FT_OpaquePaint  paint;

    FT_Fixed  scale_x;
    FT_Fixed  scale_y;

    FT_Fixed  center_x;
    FT_Fixed  center_y;

  } FT_PaintScale;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintRotate
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintRotate` paint table.  Used
   *   for rotating downstream paints with a given center and angle.
   *
   * @fields:
   *   paint ::
   *     An @FT_OpaquePaint object referencing the paint that is to be
   *     rotated.
   *
   *   angle ::
   *     The rotation angle that is to be applied in degrees divided by
   *     180.0 (as in the spec) represented as a 16.16 fixed-point
   *     value.  Multiply by 180.0f to receive degrees value.
   *
   *   center_x ::
   *     The x~coordinate of the pivot point of the rotation in font
   *     units represented as a 16.16 fixed-point value.
   *
   *   center_y ::
   *     The y~coordinate of the pivot point of the rotation in font
   *     units represented as a 16.16 fixed-point value.
   *
   * @since:
   *   2.13
   */

  typedef struct  FT_PaintRotate_
  {
    FT_OpaquePaint  paint;

    FT_Fixed  angle;

    FT_Fixed  center_x;
    FT_Fixed  center_y;

  } FT_PaintRotate;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintSkew
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintSkew` paint table.  Used
   *   for skewing or shearing downstream paints by a given center and
   *   angle.
   *
   * @fields:
   *   paint ::
   *     An @FT_OpaquePaint object referencing the paint that is to be
   *     skewed.
   *
   *   x_skew_angle ::
   *     The skewing angle in x~direction in degrees divided by 180.0
   *     (as in the spec) represented as a 16.16 fixed-point
   *     value. Multiply by 180.0f to receive degrees.
   *
   *   y_skew_angle ::
   *     The skewing angle in y~direction in degrees divided by 180.0
   *     (as in the spec) represented as a 16.16 fixed-point
   *     value.  Multiply by 180.0f to receive degrees.
   *
   *   center_x ::
   *     The x~coordinate of the pivot point of the skew in font units
   *     represented as a 16.16 fixed-point value.
   *
   *   center_y ::
   *     The y~coordinate of the pivot point of the skew in font units
   *     represented as a 16.16 fixed-point value.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintSkew_
  {
    FT_OpaquePaint  paint;

    FT_Fixed  x_skew_angle;
    FT_Fixed  y_skew_angle;

    FT_Fixed  center_x;
    FT_Fixed  center_y;

  } FT_PaintSkew;


  /**************************************************************************
   *
   * @struct:
   *   FT_PaintComposite
   *
   * @description:
   *   A structure representing a 'COLR' v1 `PaintComposite` paint table.
   *   Used for compositing two paints in a 'COLR' v1 directed acyclic graph.
   *
   * @fields:
   *   source_paint ::
   *     An @FT_OpaquePaint object referencing the source that is to be
   *     composited.
   *
   *   composite_mode ::
   *     An @FT_Composite_Mode enum value determining the composition
   *     operation.
   *
   *   backdrop_paint ::
   *     An @FT_OpaquePaint object referencing the backdrop paint that
   *     `source_paint` is composited onto.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_PaintComposite_
  {
    FT_OpaquePaint     source_paint;
    FT_Composite_Mode  composite_mode;
    FT_OpaquePaint     backdrop_paint;

  } FT_PaintComposite;


  /**************************************************************************
   *
   * @union:
   *   FT_COLR_Paint
   *
   * @description:
   *   A union object representing format and details of a paint table of a
   *   'COLR' v1 font, see
   *   'https://github.com/googlefonts/colr-gradients-spec'.  Use
   *   @FT_Get_Paint to retrieve a @FT_COLR_Paint for an @FT_OpaquePaint
   *   object.
   *
   * @fields:
   *   format ::
   *     The gradient format for this Paint structure.
   *
   *   u ::
   *     Union of all paint table types:
   *
   *       * @FT_PaintColrLayers
   *       * @FT_PaintGlyph
   *       * @FT_PaintSolid
   *       * @FT_PaintLinearGradient
   *       * @FT_PaintRadialGradient
   *       * @FT_PaintSweepGradient
   *       * @FT_PaintTransform
   *       * @FT_PaintTranslate
   *       * @FT_PaintRotate
   *       * @FT_PaintSkew
   *       * @FT_PaintComposite
   *       * @FT_PaintColrGlyph
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_COLR_Paint_
  {
    FT_PaintFormat format;

    union
    {
      FT_PaintColrLayers      colr_layers;
      FT_PaintGlyph           glyph;
      FT_PaintSolid           solid;
      FT_PaintLinearGradient  linear_gradient;
      FT_PaintRadialGradient  radial_gradient;
      FT_PaintSweepGradient   sweep_gradient;
      FT_PaintTransform       transform;
      FT_PaintTranslate       translate;
      FT_PaintScale           scale;
      FT_PaintRotate          rotate;
      FT_PaintSkew            skew;
      FT_PaintComposite       composite;
      FT_PaintColrGlyph       colr_glyph;

    } u;

  } FT_COLR_Paint;


  /**************************************************************************
   *
   * @enum:
   *   FT_Color_Root_Transform
   *
   * @description:
   *   An enumeration to specify whether @FT_Get_Color_Glyph_Paint is to
   *   return a root transform to configure the client's graphics context
   *   matrix.
   *
   * @values:
   *   FT_COLOR_INCLUDE_ROOT_TRANSFORM ::
   *     Do include the root transform as the initial @FT_COLR_Paint object.
   *
   *   FT_COLOR_NO_ROOT_TRANSFORM ::
   *     Do not output an initial root transform.
   *
   * @since:
   *   2.13
   */
  typedef enum  FT_Color_Root_Transform_
  {
    FT_COLOR_INCLUDE_ROOT_TRANSFORM,
    FT_COLOR_NO_ROOT_TRANSFORM,

    FT_COLOR_ROOT_TRANSFORM_MAX

  } FT_Color_Root_Transform;


  /**************************************************************************
   *
   * @struct:
   *   FT_ClipBox
   *
   * @description:
   *   A structure representing a 'COLR' v1 'ClipBox' table.  'COLR' v1
   *   glyphs may optionally define a clip box for aiding allocation or
   *   defining a maximum drawable region.  Use @FT_Get_Color_Glyph_ClipBox
   *   to retrieve it.
   *
   * @fields:
   *   bottom_left ::
   *     The bottom left corner of the clip box as an @FT_Vector with
   *     fixed-point coordinates in 26.6 format.
   *
   *   top_left ::
   *     The top left corner of the clip box as an @FT_Vector with
   *     fixed-point coordinates in 26.6 format.
   *
   *   top_right ::
   *     The top right corner of the clip box as an @FT_Vector with
   *     fixed-point coordinates in 26.6 format.
   *
   *   bottom_right ::
   *     The bottom right corner of the clip box as an @FT_Vector with
   *     fixed-point coordinates in 26.6 format.
   *
   * @since:
   *   2.13
   */
  typedef struct  FT_ClipBox_
  {
    FT_Vector  bottom_left;
    FT_Vector  top_left;
    FT_Vector  top_right;
    FT_Vector  bottom_right;

  } FT_ClipBox;


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Color_Glyph_Paint
   *
   * @description:
   *   This is the starting point and interface to color gradient
   *   information in a 'COLR' v1 table in OpenType fonts to recursively
   *   retrieve the paint tables for the directed acyclic graph of a colored
   *   glyph, given a glyph ID.
   *
   *     https://github.com/googlefonts/colr-gradients-spec
   *
   *   In a 'COLR' v1 font, each color glyph defines a directed acyclic
   *   graph of nested paint tables, such as `PaintGlyph`, `PaintSolid`,
   *   `PaintLinearGradient`, `PaintRadialGradient`, and so on.  Using this
   *   function and specifying a glyph ID, one retrieves the root paint
   *   table for this glyph ID.
   *
   *   This function allows control whether an initial root transform is
   *   returned to configure scaling, transform, and translation correctly
   *   on the client's graphics context.  The initial root transform is
   *   computed and returned according to the values configured for @FT_Size
   *   and @FT_Set_Transform on the @FT_Face object, see below for details
   *   of the `root_transform` parameter.  This has implications for a
   *   client 'COLR' v1 implementation: When this function returns an
   *   initially computed root transform, at the time of executing the
   *   @FT_PaintGlyph operation, the contours should be retrieved using
   *   @FT_Load_Glyph at unscaled, untransformed size.  This is because the
   *   root transform applied to the graphics context will take care of
   *   correct scaling.
   *
   *   Alternatively, to allow hinting of contours, at the time of executing
   *   @FT_Load_Glyph, the current graphics context transformation matrix
   *   can be decomposed into a scaling matrix and a remainder, and
   *   @FT_Load_Glyph can be used to retrieve the contours at scaled size.
   *   Care must then be taken to blit or clip to the graphics context with
   *   taking this remainder transformation into account.
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   *   base_glyph ::
   *     The glyph index for which to retrieve the root paint table.
   *
   *   root_transform ::
   *     Specifies whether an initially computed root is returned by the
   *     @FT_PaintTransform operation to account for the activated size
   *     (see @FT_Activate_Size) and the configured transform and translate
   *     (see @FT_Set_Transform).
   *
   *     This root transform is returned before nodes of the glyph graph of
   *     the font are returned.  Subsequent @FT_COLR_Paint structures
   *     contain unscaled and untransformed values.  The inserted root
   *     transform enables the client application to apply an initial
   *     transform to its graphics context.  When executing subsequent
   *     FT_COLR_Paint operations, values from @FT_COLR_Paint operations
   *     will ultimately be correctly scaled because of the root transform
   *     applied to the graphics context.  Use
   *     @FT_COLOR_INCLUDE_ROOT_TRANSFORM to include the root transform, use
   *     @FT_COLOR_NO_ROOT_TRANSFORM to not include it.  The latter may be
   *     useful when traversing the 'COLR' v1 glyph graph and reaching a
   *     @FT_PaintColrGlyph.  When recursing into @FT_PaintColrGlyph and
   *     painting that inline, no additional root transform is needed as it
   *     has already been applied to the graphics context at the beginning
   *     of drawing this glyph.
   *
   * @output:
   *   paint ::
   *     The @FT_OpaquePaint object that references the actual paint table.
   *
   *     The respective actual @FT_COLR_Paint object is retrieved via
   *     @FT_Get_Paint.
   *
   * @return:
   *   Value~1 if everything is OK.  If no color glyph is found, or the root
   *   paint could not be retrieved, value~0 gets returned.  In case of an
   *   error, value~0 is returned also.
   *
   * @since:
   *   2.13
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Color_Glyph_Paint( FT_Face                  face,
                            FT_UInt                  base_glyph,
                            FT_Color_Root_Transform  root_transform,
                            FT_OpaquePaint*          paint );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Color_Glyph_ClipBox
   *
   * @description:
   *   Search for a 'COLR' v1 clip box for the specified `base_glyph` and
   *   fill the `clip_box` parameter with the 'COLR' v1 'ClipBox' information
   *   if one is found.
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   *   base_glyph ::
   *     The glyph index for which to retrieve the clip box.
   *
   * @output:
   *   clip_box ::
   *     The clip box for the requested `base_glyph` if one is found.  The
   *     clip box is computed taking scale and transformations configured on
   *     the @FT_Face into account.  @FT_ClipBox contains @FT_Vector values
   *     in 26.6 format.
   *
   * @return:
   *   Value~1 if a clip box is found.  If no clip box is found or an error
   *   occured, value~0 is returned.
   *
   * @note:
   *   To retrieve the clip box in font units, reset scale to units-per-em
   *   and remove transforms configured using @FT_Set_Transform.
   *
   * @since:
   *   2.13
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Color_Glyph_ClipBox( FT_Face      face,
                              FT_UInt      base_glyph,
                              FT_ClipBox*  clip_box );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Paint_Layers
   *
   * @description:
   *   Access the layers of a `PaintColrLayers` table.
   *
   *   If the root paint of a color glyph, or a nested paint of a 'COLR'
   *   glyph is a `PaintColrLayers` table, this function retrieves the
   *   layers of the `PaintColrLayers` table.
   *
   *   The @FT_PaintColrLayers object contains an @FT_LayerIterator, which
   *   is used here to iterate over the layers.  Each layer is returned as
   *   an @FT_OpaquePaint object, which then can be used with @FT_Get_Paint
   *   to retrieve the actual paint object.
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   * @inout:
   *   iterator ::
   *     The @FT_LayerIterator from an @FT_PaintColrLayers object, for which
   *     the layers are to be retrieved.  The internal state of the iterator
   *     is incremented after one call to this function for retrieving one
   *     layer.
   *
   * @output:
   *   paint ::
   *     The @FT_OpaquePaint object that references the actual paint table.
   *     The respective actual @FT_COLR_Paint object is retrieved via
   *     @FT_Get_Paint.
   *
   * @return:
   *   Value~1 if everything is OK.  Value~0 gets returned when the paint
   *   object can not be retrieved or any other error occurs.
   *
   * @since:
   *   2.13
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Paint_Layers( FT_Face            face,
                       FT_LayerIterator*  iterator,
                       FT_OpaquePaint*    paint );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Colorline_Stops
   *
   * @description:
   *   This is an interface to color gradient information in a 'COLR' v1
   *   table in OpenType fonts to iteratively retrieve the gradient and
   *   solid fill information for colored glyph layers for a specified glyph
   *   ID.
   *
   *     https://github.com/googlefonts/colr-gradients-spec
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   * @inout:
   *   iterator ::
   *     The retrieved @FT_ColorStopIterator, configured on an @FT_ColorLine,
   *     which in turn got retrieved via paint information in
   *     @FT_PaintLinearGradient or @FT_PaintRadialGradient.
   *
   * @output:
   *   color_stop ::
   *     Color index and alpha value for the retrieved color stop.
   *
   * @return:
   *   Value~1 if everything is OK.  If there are no more color stops,
   *   value~0 gets returned.  In case of an error, value~0 is returned
   *   also.
   *
   * @since:
   *   2.13
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Colorline_Stops( FT_Face                face,
                          FT_ColorStop*          color_stop,
                          FT_ColorStopIterator*  iterator );


  /**************************************************************************
   *
   * @function:
   *  FT_Get_Paint
   *
   * @description:
   *   Access the details of a paint using an @FT_OpaquePaint opaque paint
   *   object, which internally stores the offset to the respective `Paint`
   *   object in the 'COLR' table.
   *
   * @input:
   *   face ::
   *     A handle to the parent face object.
   *
   *   opaque_paint ::
   *     The opaque paint object for which the underlying @FT_COLR_Paint
   *     data is to be retrieved.
   *
   * @output:
   *   paint ::
   *     The specific @FT_COLR_Paint object containing information coming
   *     from one of the font's `Paint*` tables.
   *
   * @return:
   *   Value~1 if everything is OK.  Value~0 if no details can be found for
   *   this paint or any other error occured.
   *
   * @since:
   *   2.13
   */
  FT_EXPORT( FT_Bool )
  FT_Get_Paint( FT_Face         face,
                FT_OpaquePaint  opaque_paint,
                FT_COLR_Paint*  paint );

  /* */


FT_END_HEADER

#endif /* FTCOLOR_H_ */


/* END */
