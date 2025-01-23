/****************************************************************************
 *
 * freetype.h
 *
 *   FreeType high-level API and common types (specification only).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FREETYPE_H_
#define FREETYPE_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/fttypes.h>
#include <freetype/fterrors.h>


FT_BEGIN_HEADER



  /**************************************************************************
   *
   * @section:
   *   preamble
   *
   * @title:
   *   Preamble
   *
   * @abstract:
   *   What FreeType is and isn't
   *
   * @description:
   *   FreeType is a library that provides access to glyphs in font files.  It
   *   scales the glyph images and their metrics to a requested size, and it
   *   rasterizes the glyph images to produce pixel or subpixel alpha coverage
   *   bitmaps.
   *
   *   Note that FreeType is _not_ a text layout engine.  You have to use
   *   higher-level libraries like HarfBuzz, Pango, or ICU for that.
   *
   *   Note also that FreeType does _not_ perform alpha blending or
   *   compositing the resulting bitmaps or pixmaps by itself.  Use your
   *   favourite graphics library (for example, Cairo or Skia) to further
   *   process FreeType's output.
   *
   */


  /**************************************************************************
   *
   * @section:
   *   header_inclusion
   *
   * @title:
   *   FreeType's header inclusion scheme
   *
   * @abstract:
   *   How client applications should include FreeType header files.
   *
   * @description:
   *   To be as flexible as possible (and for historical reasons), you must
   *   load file `ft2build.h` first before other header files, for example
   *
   *   ```
   *     #include <ft2build.h>
   *
   *     #include <freetype/freetype.h>
   *     #include <freetype/ftoutln.h>
   *   ```
   */


  /**************************************************************************
   *
   * @section:
   *   user_allocation
   *
   * @title:
   *   User allocation
   *
   * @abstract:
   *   How client applications should allocate FreeType data structures.
   *
   * @description:
   *   FreeType assumes that structures allocated by the user and passed as
   *   arguments are zeroed out except for the actual data.  In other words,
   *   it is recommended to use `calloc` (or variants of it) instead of
   *   `malloc` for allocation.
   *
   */


  /**************************************************************************
   *
   * @section:
   *   font_testing_macros
   *
   * @title:
   *   Font Testing Macros
   *
   * @abstract:
   *   Macros to test various properties of fonts.
   *
   * @description:
   *   Macros to test the most important font properties.
   *
   *   It is recommended to use these high-level macros instead of directly
   *   testing the corresponding flags, which are scattered over various
   *   structures.
   *
   * @order:
   *   FT_HAS_HORIZONTAL
   *   FT_HAS_VERTICAL
   *   FT_HAS_KERNING
   *   FT_HAS_FIXED_SIZES
   *   FT_HAS_GLYPH_NAMES
   *   FT_HAS_COLOR
   *   FT_HAS_MULTIPLE_MASTERS
   *   FT_HAS_SVG
   *   FT_HAS_SBIX
   *   FT_HAS_SBIX_OVERLAY
   *
   *   FT_IS_SFNT
   *   FT_IS_SCALABLE
   *   FT_IS_FIXED_WIDTH
   *   FT_IS_CID_KEYED
   *   FT_IS_TRICKY
   *   FT_IS_NAMED_INSTANCE
   *   FT_IS_VARIATION
   *
   */


  /**************************************************************************
   *
   * @section:
   *   library_setup
   *
   * @title:
   *   Library Setup
   *
   * @abstract:
   *   Functions to start and end the usage of the FreeType library.
   *
   * @description:
   *   Functions to start and end the usage of the FreeType library.
   *
   *   Note that @FT_Library_Version and @FREETYPE_XXX are of limited use
   *   because even a new release of FreeType with only documentation
   *   changes increases the version number.
   *
   * @order:
   *   FT_Library
   *   FT_Init_FreeType
   *   FT_Done_FreeType
   *
   *   FT_Library_Version
   *   FREETYPE_XXX
   *
   */


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   * @title:
   *   Face Creation
   *
   * @abstract:
   *   Functions to manage fonts.
   *
   * @description:
   *   The functions and structures collected in this section operate on
   *   fonts globally.
   *
   * @order:
   *   FT_Face
   *   FT_FaceRec
   *   FT_FACE_FLAG_XXX
   *   FT_STYLE_FLAG_XXX
   *
   *   FT_New_Face
   *   FT_Done_Face
   *   FT_Reference_Face
   *   FT_New_Memory_Face
   *   FT_Face_Properties
   *   FT_Open_Face
   *   FT_Open_Args
   *   FT_OPEN_XXX
   *   FT_Parameter
   *   FT_Attach_File
   *   FT_Attach_Stream
   *
   */


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   * @title:
   *   Sizing and Scaling
   *
   * @abstract:
   *   Functions to manage font sizes.
   *
   * @description:
   *   The functions and structures collected in this section are related to
   *   selecting and manipulating the size of a font globally.
   *
   * @order:
   *   FT_Size
   *   FT_SizeRec
   *   FT_Size_Metrics
   *
   *   FT_Bitmap_Size
   *
   *   FT_Set_Char_Size
   *   FT_Set_Pixel_Sizes
   *   FT_Request_Size
   *   FT_Select_Size
   *   FT_Size_Request_Type
   *   FT_Size_RequestRec
   *   FT_Size_Request
   *
   *   FT_Set_Transform
   *   FT_Get_Transform
   *
   */


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   * @title:
   *   Glyph Retrieval
   *
   * @abstract:
   *   Functions to manage glyphs.
   *
   * @description:
   *   The functions and structures collected in this section operate on
   *   single glyphs, of which @FT_Load_Glyph is most important.
   *
   * @order:
   *   FT_GlyphSlot
   *   FT_GlyphSlotRec
   *   FT_Glyph_Metrics
   *
   *   FT_Load_Glyph
   *   FT_LOAD_XXX
   *   FT_LOAD_TARGET_MODE
   *   FT_LOAD_TARGET_XXX
   *
   *   FT_Render_Glyph
   *   FT_Render_Mode
   *   FT_Get_Kerning
   *   FT_Kerning_Mode
   *   FT_Get_Track_Kerning
   *
   */


  /**************************************************************************
   *
   * @section:
   *   character_mapping
   *
   * @title:
   *   Character Mapping
   *
   * @abstract:
   *   Functions to manage character-to-glyph maps.
   *
   * @description:
   *   This section holds functions and structures that are related to
   *   mapping character input codes to glyph indices.
   *
   *   Note that for many scripts the simplistic approach used by FreeType
   *   of mapping a single character to a single glyph is not valid or
   *   possible!  In general, a higher-level library like HarfBuzz or ICU
   *   should be used for handling text strings.
   *
   * @order:
   *   FT_CharMap
   *   FT_CharMapRec
   *   FT_Encoding
   *   FT_ENC_TAG
   *
   *   FT_Select_Charmap
   *   FT_Set_Charmap
   *   FT_Get_Charmap_Index
   *
   *   FT_Get_Char_Index
   *   FT_Get_First_Char
   *   FT_Get_Next_Char
   *   FT_Load_Char
   *
   */


  /**************************************************************************
   *
   * @section:
   *   information_retrieval
   *
   * @title:
   *   Information Retrieval
   *
   * @abstract:
   *   Functions to retrieve font and glyph information.
   *
   * @description:
   *   Functions to retrieve font and glyph information.  Only some very
   *   basic data is covered; see also the chapter on the format-specific
   *   API for more.
   *
   *
   * @order:
   *   FT_Get_Name_Index
   *   FT_Get_Glyph_Name
   *   FT_Get_Postscript_Name
   *   FT_Get_FSType_Flags
   *   FT_FSTYPE_XXX
   *   FT_Get_SubGlyph_Info
   *   FT_SUBGLYPH_FLAG_XXX
   *
   */


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   * @title:
   *   Other API Data
   *
   * @abstract:
   *   Other structures, enumerations, and macros.
   *
   * @description:
   *   Other structures, enumerations, and macros.  Deprecated functions are
   *   also listed here.
   *
   * @order:
   *   FT_Face_Internal
   *   FT_Size_Internal
   *   FT_Slot_Internal
   *
   *   FT_SubGlyph
   *
   *   FT_HAS_FAST_GLYPHS
   *   FT_Face_CheckTrueTypePatents
   *   FT_Face_SetUnpatentedHinting
   *
   */


  /*************************************************************************/
  /*************************************************************************/
  /*                                                                       */
  /*                        B A S I C   T Y P E S                          */
  /*                                                                       */
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_Glyph_Metrics
   *
   * @description:
   *   A structure to model the metrics of a single glyph.  The values are
   *   expressed in 26.6 fractional pixel format; if the flag
   *   @FT_LOAD_NO_SCALE has been used while loading the glyph, values are
   *   expressed in font units instead.
   *
   * @fields:
   *   width ::
   *     The glyph's width.
   *
   *   height ::
   *     The glyph's height.
   *
   *   horiBearingX ::
   *     Left side bearing for horizontal layout.
   *
   *   horiBearingY ::
   *     Top side bearing for horizontal layout.
   *
   *   horiAdvance ::
   *     Advance width for horizontal layout.
   *
   *   vertBearingX ::
   *     Left side bearing for vertical layout.
   *
   *   vertBearingY ::
   *     Top side bearing for vertical layout.  Larger positive values mean
   *     further below the vertical glyph origin.
   *
   *   vertAdvance ::
   *     Advance height for vertical layout.  Positive values mean the glyph
   *     has a positive advance downward.
   *
   * @note:
   *   If not disabled with @FT_LOAD_NO_HINTING, the values represent
   *   dimensions of the hinted glyph (in case hinting is applicable).
   *
   *   Stroking a glyph with an outside border does not increase
   *   `horiAdvance` or `vertAdvance`; you have to manually adjust these
   *   values to account for the added width and height.
   *
   *   FreeType doesn't use the 'VORG' table data for CFF fonts because it
   *   doesn't have an interface to quickly retrieve the glyph height.  The
   *   y~coordinate of the vertical origin can be simply computed as
   *   `vertBearingY + height` after loading a glyph.
   */
  typedef struct  FT_Glyph_Metrics_
  {
    FT_Pos  width;
    FT_Pos  height;

    FT_Pos  horiBearingX;
    FT_Pos  horiBearingY;
    FT_Pos  horiAdvance;

    FT_Pos  vertBearingX;
    FT_Pos  vertBearingY;
    FT_Pos  vertAdvance;

  } FT_Glyph_Metrics;


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_Bitmap_Size
   *
   * @description:
   *   This structure models the metrics of a bitmap strike (i.e., a set of
   *   glyphs for a given point size and resolution) in a bitmap font.  It is
   *   used for the `available_sizes` field of @FT_Face.
   *
   * @fields:
   *   height ::
   *     The vertical distance, in pixels, between two consecutive baselines.
   *     It is always positive.
   *
   *   width ::
   *     The average width, in pixels, of all glyphs in the strike.
   *
   *   size ::
   *     The nominal size of the strike in 26.6 fractional points.  This
   *     field is not very useful.
   *
   *   x_ppem ::
   *     The horizontal ppem (nominal width) in 26.6 fractional pixels.
   *
   *   y_ppem ::
   *     The vertical ppem (nominal height) in 26.6 fractional pixels.
   *
   * @note:
   *   Windows FNT:
   *     The nominal size given in a FNT font is not reliable.  If the driver
   *     finds it incorrect, it sets `size` to some calculated values, and
   *     `x_ppem` and `y_ppem` to the pixel width and height given in the
   *     font, respectively.
   *
   *   TrueType embedded bitmaps:
   *     `size`, `width`, and `height` values are not contained in the bitmap
   *     strike itself.  They are computed from the global font parameters.
   */
  typedef struct  FT_Bitmap_Size_
  {
    FT_Short  height;
    FT_Short  width;

    FT_Pos    size;

    FT_Pos    x_ppem;
    FT_Pos    y_ppem;

  } FT_Bitmap_Size;


  /*************************************************************************/
  /*************************************************************************/
  /*                                                                       */
  /*                     O B J E C T   C L A S S E S                       */
  /*                                                                       */
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * @section:
   *   library_setup
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Library
   *
   * @description:
   *   A handle to a FreeType library instance.  Each 'library' is completely
   *   independent from the others; it is the 'root' of a set of objects like
   *   fonts, faces, sizes, etc.
   *
   *   It also embeds a memory manager (see @FT_Memory), as well as a
   *   scan-line converter object (see @FT_Raster).
   *
   *   [Since 2.5.6] In multi-threaded applications it is easiest to use one
   *   `FT_Library` object per thread.  In case this is too cumbersome, a
   *   single `FT_Library` object across threads is possible also, as long as
   *   a mutex lock is used around @FT_New_Face and @FT_Done_Face.
   *
   * @note:
   *   Library objects are normally created by @FT_Init_FreeType, and
   *   destroyed with @FT_Done_FreeType.  If you need reference-counting
   *   (cf. @FT_Reference_Library), use @FT_New_Library and @FT_Done_Library.
   */
  typedef struct FT_LibraryRec_  *FT_Library;


  /**************************************************************************
   *
   * @section:
   *   module_management
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Module
   *
   * @description:
   *   A handle to a given FreeType module object.  A module can be a font
   *   driver, a renderer, or anything else that provides services to the
   *   former.
   */
  typedef struct FT_ModuleRec_*  FT_Module;


  /**************************************************************************
   *
   * @type:
   *   FT_Driver
   *
   * @description:
   *   A handle to a given FreeType font driver object.  A font driver is a
   *   module capable of creating faces from font files.
   */
  typedef struct FT_DriverRec_*  FT_Driver;


  /**************************************************************************
   *
   * @type:
   *   FT_Renderer
   *
   * @description:
   *   A handle to a given FreeType renderer.  A renderer is a module in
   *   charge of converting a glyph's outline image to a bitmap.  It supports
   *   a single glyph image format, and one or more target surface depths.
   */
  typedef struct FT_RendererRec_*  FT_Renderer;


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Face
   *
   * @description:
   *   A handle to a typographic face object.  A face object models a given
   *   typeface, in a given style.
   *
   * @note:
   *   A face object also owns a single @FT_GlyphSlot object, as well as one
   *   or more @FT_Size objects.
   *
   *   Use @FT_New_Face or @FT_Open_Face to create a new face object from a
   *   given filepath or a custom input stream.
   *
   *   Use @FT_Done_Face to destroy it (along with its slot and sizes).
   *
   *   An `FT_Face` object can only be safely used from one thread at a time.
   *   Similarly, creation and destruction of `FT_Face` with the same
   *   @FT_Library object can only be done from one thread at a time.  On the
   *   other hand, functions like @FT_Load_Glyph and its siblings are
   *   thread-safe and do not need the lock to be held as long as the same
   *   `FT_Face` object is not used from multiple threads at the same time.
   *
   * @also:
   *   See @FT_FaceRec for the publicly accessible fields of a given face
   *   object.
   */
  typedef struct FT_FaceRec_*  FT_Face;


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Size
   *
   * @description:
   *   A handle to an object that models a face scaled to a given character
   *   size.
   *
   * @note:
   *   An @FT_Face has one _active_ `FT_Size` object that is used by
   *   functions like @FT_Load_Glyph to determine the scaling transformation
   *   that in turn is used to load and hint glyphs and metrics.
   *
   *   A newly created `FT_Size` object contains only meaningless zero values.
   *   You must use @FT_Set_Char_Size, @FT_Set_Pixel_Sizes, @FT_Request_Size
   *   or even @FT_Select_Size to change the content (i.e., the scaling
   *   values) of the active `FT_Size`.  Otherwise, the scaling and hinting
   *   will not be performed.
   *
   *   You can use @FT_New_Size to create additional size objects for a given
   *   @FT_Face, but they won't be used by other functions until you activate
   *   it through @FT_Activate_Size.  Only one size can be activated at any
   *   given time per face.
   *
   * @also:
   *   See @FT_SizeRec for the publicly accessible fields of a given size
   *   object.
   */
  typedef struct FT_SizeRec_*  FT_Size;


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_GlyphSlot
   *
   * @description:
   *   A handle to a given 'glyph slot'.  A slot is a container that can hold
   *   any of the glyphs contained in its parent face.
   *
   *   In other words, each time you call @FT_Load_Glyph or @FT_Load_Char,
   *   the slot's content is erased by the new glyph data, i.e., the glyph's
   *   metrics, its image (bitmap or outline), and other control information.
   *
   * @also:
   *   See @FT_GlyphSlotRec for the publicly accessible glyph fields.
   */
  typedef struct FT_GlyphSlotRec_*  FT_GlyphSlot;


  /**************************************************************************
   *
   * @section:
   *   character_mapping
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_CharMap
   *
   * @description:
   *   A handle to a character map (usually abbreviated to 'charmap').  A
   *   charmap is used to translate character codes in a given encoding into
   *   glyph indexes for its parent's face.  Some font formats may provide
   *   several charmaps per font.
   *
   *   Each face object owns zero or more charmaps, but only one of them can
   *   be 'active', providing the data used by @FT_Get_Char_Index or
   *   @FT_Load_Char.
   *
   *   The list of available charmaps in a face is available through the
   *   `face->num_charmaps` and `face->charmaps` fields of @FT_FaceRec.
   *
   *   The currently active charmap is available as `face->charmap`.  You
   *   should call @FT_Set_Charmap to change it.
   *
   * @note:
   *   When a new face is created (either through @FT_New_Face or
   *   @FT_Open_Face), the library looks for a Unicode charmap within the
   *   list and automatically activates it.  If there is no Unicode charmap,
   *   FreeType doesn't set an 'active' charmap.
   *
   * @also:
   *   See @FT_CharMapRec for the publicly accessible fields of a given
   *   character map.
   */
  typedef struct FT_CharMapRec_*  FT_CharMap;


  /**************************************************************************
   *
   * @macro:
   *   FT_ENC_TAG
   *
   * @description:
   *   This macro converts four-letter tags into an unsigned long.  It is
   *   used to define 'encoding' identifiers (see @FT_Encoding).
   *
   * @note:
   *   Since many 16-bit compilers don't like 32-bit enumerations, you should
   *   redefine this macro in case of problems to something like this:
   *
   *   ```
   *     #define FT_ENC_TAG( value, a, b, c, d )  value
   *   ```
   *
   *   to get a simple enumeration without assigning special numbers.
   */

#ifndef FT_ENC_TAG

#define FT_ENC_TAG( value, a, b, c, d )                             \
          value = ( ( FT_STATIC_BYTE_CAST( FT_UInt32, a ) << 24 ) | \
                    ( FT_STATIC_BYTE_CAST( FT_UInt32, b ) << 16 ) | \
                    ( FT_STATIC_BYTE_CAST( FT_UInt32, c ) <<  8 ) | \
                      FT_STATIC_BYTE_CAST( FT_UInt32, d )         )

#endif /* FT_ENC_TAG */


  /**************************************************************************
   *
   * @enum:
   *   FT_Encoding
   *
   * @description:
   *   An enumeration to specify character sets supported by charmaps.  Used
   *   in the @FT_Select_Charmap API function.
   *
   * @note:
   *   Despite the name, this enumeration lists specific character
   *   repertoires (i.e., charsets), and not text encoding methods (e.g.,
   *   UTF-8, UTF-16, etc.).
   *
   *   Other encodings might be defined in the future.
   *
   * @values:
   *   FT_ENCODING_NONE ::
   *     The encoding value~0 is reserved for all formats except BDF, PCF,
   *     and Windows FNT; see below for more information.
   *
   *   FT_ENCODING_UNICODE ::
   *     The Unicode character set.  This value covers all versions of the
   *     Unicode repertoire, including ASCII and Latin-1.  Most fonts include
   *     a Unicode charmap, but not all of them.
   *
   *     For example, if you want to access Unicode value U+1F028 (and the
   *     font contains it), use value 0x1F028 as the input value for
   *     @FT_Get_Char_Index.
   *
   *   FT_ENCODING_MS_SYMBOL ::
   *     Microsoft Symbol encoding, used to encode mathematical symbols and
   *     wingdings.  For more information, see
   *     'https://www.microsoft.com/typography/otspec/recom.htm#non-standard-symbol-fonts',
   *     'http://www.kostis.net/charsets/symbol.htm', and
   *     'http://www.kostis.net/charsets/wingding.htm'.
   *
   *     This encoding uses character codes from the PUA (Private Unicode
   *     Area) in the range U+F020-U+F0FF.
   *
   *   FT_ENCODING_SJIS ::
   *     Shift JIS encoding for Japanese.  More info at
   *     'https://en.wikipedia.org/wiki/Shift_JIS'.  See note on multi-byte
   *     encodings below.
   *
   *   FT_ENCODING_PRC ::
   *     Corresponds to encoding systems mainly for Simplified Chinese as
   *     used in People's Republic of China (PRC).  The encoding layout is
   *     based on GB~2312 and its supersets GBK and GB~18030.
   *
   *   FT_ENCODING_BIG5 ::
   *     Corresponds to an encoding system for Traditional Chinese as used in
   *     Taiwan and Hong Kong.
   *
   *   FT_ENCODING_WANSUNG ::
   *     Corresponds to the Korean encoding system known as Extended Wansung
   *     (MS Windows code page 949).  For more information see
   *     'https://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WindowsBestFit/bestfit949.txt'.
   *
   *   FT_ENCODING_JOHAB ::
   *     The Korean standard character set (KS~C 5601-1992), which
   *     corresponds to MS Windows code page 1361.  This character set
   *     includes all possible Hangul character combinations.
   *
   *   FT_ENCODING_ADOBE_LATIN_1 ::
   *     Corresponds to a Latin-1 encoding as defined in a Type~1 PostScript
   *     font.  It is limited to 256 character codes.
   *
   *   FT_ENCODING_ADOBE_STANDARD ::
   *     Adobe Standard encoding, as found in Type~1, CFF, and OpenType/CFF
   *     fonts.  It is limited to 256 character codes.
   *
   *   FT_ENCODING_ADOBE_EXPERT ::
   *     Adobe Expert encoding, as found in Type~1, CFF, and OpenType/CFF
   *     fonts.  It is limited to 256 character codes.
   *
   *   FT_ENCODING_ADOBE_CUSTOM ::
   *     Corresponds to a custom encoding, as found in Type~1, CFF, and
   *     OpenType/CFF fonts.  It is limited to 256 character codes.
   *
   *   FT_ENCODING_APPLE_ROMAN ::
   *     Apple roman encoding.  Many TrueType and OpenType fonts contain a
   *     charmap for this 8-bit encoding, since older versions of Mac OS are
   *     able to use it.
   *
   *   FT_ENCODING_OLD_LATIN_2 ::
   *     This value is deprecated and was neither used nor reported by
   *     FreeType.  Don't use or test for it.
   *
   *   FT_ENCODING_MS_SJIS ::
   *     Same as FT_ENCODING_SJIS.  Deprecated.
   *
   *   FT_ENCODING_MS_GB2312 ::
   *     Same as FT_ENCODING_PRC.  Deprecated.
   *
   *   FT_ENCODING_MS_BIG5 ::
   *     Same as FT_ENCODING_BIG5.  Deprecated.
   *
   *   FT_ENCODING_MS_WANSUNG ::
   *     Same as FT_ENCODING_WANSUNG.  Deprecated.
   *
   *   FT_ENCODING_MS_JOHAB ::
   *     Same as FT_ENCODING_JOHAB.  Deprecated.
   *
   * @note:
   *   When loading a font, FreeType makes a Unicode charmap active if
   *   possible (either if the font provides such a charmap, or if FreeType
   *   can synthesize one from PostScript glyph name dictionaries; in either
   *   case, the charmap is tagged with `FT_ENCODING_UNICODE`).  If such a
   *   charmap is synthesized, it is placed at the first position of the
   *   charmap array.
   *
   *   All other encodings are considered legacy and tagged only if
   *   explicitly defined in the font file.  Otherwise, `FT_ENCODING_NONE` is
   *   used.
   *
   *   `FT_ENCODING_NONE` is set by the BDF and PCF drivers if the charmap is
   *   neither Unicode nor ISO-8859-1 (otherwise it is set to
   *   `FT_ENCODING_UNICODE`).  Use @FT_Get_BDF_Charset_ID to find out which
   *   encoding is really present.  If, for example, the `cs_registry` field
   *   is 'KOI8' and the `cs_encoding` field is 'R', the font is encoded in
   *   KOI8-R.
   *
   *   `FT_ENCODING_NONE` is always set (with a single exception) by the
   *   winfonts driver.  Use @FT_Get_WinFNT_Header and examine the `charset`
   *   field of the @FT_WinFNT_HeaderRec structure to find out which encoding
   *   is really present.  For example, @FT_WinFNT_ID_CP1251 (204) means
   *   Windows code page 1251 (for Russian).
   *
   *   `FT_ENCODING_NONE` is set if `platform_id` is @TT_PLATFORM_MACINTOSH
   *   and `encoding_id` is not `TT_MAC_ID_ROMAN` (otherwise it is set to
   *   `FT_ENCODING_APPLE_ROMAN`).
   *
   *   If `platform_id` is @TT_PLATFORM_MACINTOSH, use the function
   *   @FT_Get_CMap_Language_ID to query the Mac language ID that may be
   *   needed to be able to distinguish Apple encoding variants.  See
   *
   *     https://www.unicode.org/Public/MAPPINGS/VENDORS/APPLE/Readme.txt
   *
   *   to get an idea how to do that.  Basically, if the language ID is~0,
   *   don't use it, otherwise subtract 1 from the language ID.  Then examine
   *   `encoding_id`.  If, for example, `encoding_id` is `TT_MAC_ID_ROMAN`
   *   and the language ID (minus~1) is `TT_MAC_LANGID_GREEK`, it is the
   *   Greek encoding, not Roman.  `TT_MAC_ID_ARABIC` with
   *   `TT_MAC_LANGID_FARSI` means the Farsi variant of the Arabic encoding.
   */
  typedef enum  FT_Encoding_
  {
    FT_ENC_TAG( FT_ENCODING_NONE, 0, 0, 0, 0 ),

    FT_ENC_TAG( FT_ENCODING_MS_SYMBOL, 's', 'y', 'm', 'b' ),
    FT_ENC_TAG( FT_ENCODING_UNICODE,   'u', 'n', 'i', 'c' ),

    FT_ENC_TAG( FT_ENCODING_SJIS,    's', 'j', 'i', 's' ),
    FT_ENC_TAG( FT_ENCODING_PRC,     'g', 'b', ' ', ' ' ),
    FT_ENC_TAG( FT_ENCODING_BIG5,    'b', 'i', 'g', '5' ),
    FT_ENC_TAG( FT_ENCODING_WANSUNG, 'w', 'a', 'n', 's' ),
    FT_ENC_TAG( FT_ENCODING_JOHAB,   'j', 'o', 'h', 'a' ),

    /* for backward compatibility */
    FT_ENCODING_GB2312     = FT_ENCODING_PRC,
    FT_ENCODING_MS_SJIS    = FT_ENCODING_SJIS,
    FT_ENCODING_MS_GB2312  = FT_ENCODING_PRC,
    FT_ENCODING_MS_BIG5    = FT_ENCODING_BIG5,
    FT_ENCODING_MS_WANSUNG = FT_ENCODING_WANSUNG,
    FT_ENCODING_MS_JOHAB   = FT_ENCODING_JOHAB,

    FT_ENC_TAG( FT_ENCODING_ADOBE_STANDARD, 'A', 'D', 'O', 'B' ),
    FT_ENC_TAG( FT_ENCODING_ADOBE_EXPERT,   'A', 'D', 'B', 'E' ),
    FT_ENC_TAG( FT_ENCODING_ADOBE_CUSTOM,   'A', 'D', 'B', 'C' ),
    FT_ENC_TAG( FT_ENCODING_ADOBE_LATIN_1,  'l', 'a', 't', '1' ),

    FT_ENC_TAG( FT_ENCODING_OLD_LATIN_2, 'l', 'a', 't', '2' ),

    FT_ENC_TAG( FT_ENCODING_APPLE_ROMAN, 'a', 'r', 'm', 'n' )

  } FT_Encoding;


  /* these constants are deprecated; use the corresponding `FT_Encoding` */
  /* values instead                                                      */
#define ft_encoding_none            FT_ENCODING_NONE
#define ft_encoding_unicode         FT_ENCODING_UNICODE
#define ft_encoding_symbol          FT_ENCODING_MS_SYMBOL
#define ft_encoding_latin_1         FT_ENCODING_ADOBE_LATIN_1
#define ft_encoding_latin_2         FT_ENCODING_OLD_LATIN_2
#define ft_encoding_sjis            FT_ENCODING_SJIS
#define ft_encoding_gb2312          FT_ENCODING_PRC
#define ft_encoding_big5            FT_ENCODING_BIG5
#define ft_encoding_wansung         FT_ENCODING_WANSUNG
#define ft_encoding_johab           FT_ENCODING_JOHAB

#define ft_encoding_adobe_standard  FT_ENCODING_ADOBE_STANDARD
#define ft_encoding_adobe_expert    FT_ENCODING_ADOBE_EXPERT
#define ft_encoding_adobe_custom    FT_ENCODING_ADOBE_CUSTOM
#define ft_encoding_apple_roman     FT_ENCODING_APPLE_ROMAN


  /**************************************************************************
   *
   * @struct:
   *   FT_CharMapRec
   *
   * @description:
   *   The base charmap structure.
   *
   * @fields:
   *   face ::
   *     A handle to the parent face object.
   *
   *   encoding ::
   *     An @FT_Encoding tag identifying the charmap.  Use this with
   *     @FT_Select_Charmap.
   *
   *   platform_id ::
   *     An ID number describing the platform for the following encoding ID.
   *     This comes directly from the TrueType specification and gets
   *     emulated for other formats.
   *
   *   encoding_id ::
   *     A platform-specific encoding number.  This also comes from the
   *     TrueType specification and gets emulated similarly.
   */
  typedef struct  FT_CharMapRec_
  {
    FT_Face      face;
    FT_Encoding  encoding;
    FT_UShort    platform_id;
    FT_UShort    encoding_id;

  } FT_CharMapRec;


  /*************************************************************************/
  /*************************************************************************/
  /*                                                                       */
  /*                 B A S E   O B J E C T   C L A S S E S                 */
  /*                                                                       */
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Face_Internal
   *
   * @description:
   *   An opaque handle to an `FT_Face_InternalRec` structure that models the
   *   private data of a given @FT_Face object.
   *
   *   This structure might change between releases of FreeType~2 and is not
   *   generally available to client applications.
   */
  typedef struct FT_Face_InternalRec_*  FT_Face_Internal;


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_FaceRec
   *
   * @description:
   *   FreeType root face class structure.  A face object models a typeface
   *   in a font file.
   *
   * @fields:
   *   num_faces ::
   *     The number of faces in the font file.  Some font formats can have
   *     multiple faces in a single font file.
   *
   *   face_index ::
   *     This field holds two different values.  Bits 0-15 are the index of
   *     the face in the font file (starting with value~0).  They are set
   *     to~0 if there is only one face in the font file.
   *
   *     [Since 2.6.1] Bits 16-30 are relevant to GX and OpenType variation
   *     fonts only, holding the named instance index for the current face
   *     index (starting with value~1; value~0 indicates font access without
   *     a named instance).  For non-variation fonts, bits 16-30 are ignored.
   *     If we have the third named instance of face~4, say, `face_index` is
   *     set to 0x00030004.
   *
   *     Bit 31 is always zero (that is, `face_index` is always a positive
   *     value).
   *
   *     [Since 2.9] Changing the design coordinates with
   *     @FT_Set_Var_Design_Coordinates or @FT_Set_Var_Blend_Coordinates does
   *     not influence the named instance index value (only
   *     @FT_Set_Named_Instance does that).
   *
   *   face_flags ::
   *     A set of bit flags that give important information about the face;
   *     see @FT_FACE_FLAG_XXX for the details.
   *
   *   style_flags ::
   *     The lower 16~bits contain a set of bit flags indicating the style of
   *     the face; see @FT_STYLE_FLAG_XXX for the details.
   *
   *     [Since 2.6.1] Bits 16-30 hold the number of named instances
   *     available for the current face if we have a GX or OpenType variation
   *     (sub)font.  Bit 31 is always zero (that is, `style_flags` is always
   *     a positive value).  Note that a variation font has always at least
   *     one named instance, namely the default instance.
   *
   *   num_glyphs ::
   *     The number of glyphs in the face.  If the face is scalable and has
   *     sbits (see `num_fixed_sizes`), it is set to the number of outline
   *     glyphs.
   *
   *     For CID-keyed fonts (not in an SFNT wrapper) this value gives the
   *     highest CID used in the font.
   *
   *   family_name ::
   *     The face's family name.  This is an ASCII string, usually in
   *     English, that describes the typeface's family (like 'Times New
   *     Roman', 'Bodoni', 'Garamond', etc).  This is a least common
   *     denominator used to list fonts.  Some formats (TrueType & OpenType)
   *     provide localized and Unicode versions of this string.  Applications
   *     should use the format-specific interface to access them.  Can be
   *     `NULL` (e.g., in fonts embedded in a PDF file).
   *
   *     In case the font doesn't provide a specific family name entry,
   *     FreeType tries to synthesize one, deriving it from other name
   *     entries.
   *
   *   style_name ::
   *     The face's style name.  This is an ASCII string, usually in English,
   *     that describes the typeface's style (like 'Italic', 'Bold',
   *     'Condensed', etc).  Not all font formats provide a style name, so
   *     this field is optional, and can be set to `NULL`.  As for
   *     `family_name`, some formats provide localized and Unicode versions
   *     of this string.  Applications should use the format-specific
   *     interface to access them.
   *
   *   num_fixed_sizes ::
   *     The number of bitmap strikes in the face.  Even if the face is
   *     scalable, there might still be bitmap strikes, which are called
   *     'sbits' in that case.
   *
   *   available_sizes ::
   *     An array of @FT_Bitmap_Size for all bitmap strikes in the face.  It
   *     is set to `NULL` if there is no bitmap strike.
   *
   *     Note that FreeType tries to sanitize the strike data since they are
   *     sometimes sloppy or incorrect, but this can easily fail.
   *
   *   num_charmaps ::
   *     The number of charmaps in the face.
   *
   *   charmaps ::
   *     An array of the charmaps of the face.
   *
   *   generic ::
   *     A field reserved for client uses.  See the @FT_Generic type
   *     description.
   *
   *   bbox ::
   *     The font bounding box.  Coordinates are expressed in font units (see
   *     `units_per_EM`).  The box is large enough to contain any glyph from
   *     the font.  Thus, `bbox.yMax` can be seen as the 'maximum ascender',
   *     and `bbox.yMin` as the 'minimum descender'.  Only relevant for
   *     scalable formats.
   *
   *     Note that the bounding box might be off by (at least) one pixel for
   *     hinted fonts.  See @FT_Size_Metrics for further discussion.
   *
   *     Note that the bounding box does not vary in OpenType variation fonts
   *     and should only be used in relation to the default instance.
   *
   *   units_per_EM ::
   *     The number of font units per EM square for this face.  This is
   *     typically 2048 for TrueType fonts, and 1000 for Type~1 fonts.  Only
   *     relevant for scalable formats.
   *
   *   ascender ::
   *     The typographic ascender of the face, expressed in font units.  For
   *     font formats not having this information, it is set to `bbox.yMax`.
   *     Only relevant for scalable formats.
   *
   *   descender ::
   *     The typographic descender of the face, expressed in font units.  For
   *     font formats not having this information, it is set to `bbox.yMin`.
   *     Note that this field is negative for values below the baseline.
   *     Only relevant for scalable formats.
   *
   *   height ::
   *     This value is the vertical distance between two consecutive
   *     baselines, expressed in font units.  It is always positive.  Only
   *     relevant for scalable formats.
   *
   *     If you want the global glyph height, use `ascender - descender`.
   *
   *   max_advance_width ::
   *     The maximum advance width, in font units, for all glyphs in this
   *     face.  This can be used to make word wrapping computations faster.
   *     Only relevant for scalable formats.
   *
   *   max_advance_height ::
   *     The maximum advance height, in font units, for all glyphs in this
   *     face.  This is only relevant for vertical layouts, and is set to
   *     `height` for fonts that do not provide vertical metrics.  Only
   *     relevant for scalable formats.
   *
   *   underline_position ::
   *     The position, in font units, of the underline line for this face.
   *     It is the center of the underlining stem.  Only relevant for
   *     scalable formats.
   *
   *   underline_thickness ::
   *     The thickness, in font units, of the underline for this face.  Only
   *     relevant for scalable formats.
   *
   *   glyph ::
   *     The face's associated glyph slot(s).
   *
   *   size ::
   *     The current active size for this face.
   *
   *   charmap ::
   *     The current active charmap for this face.
   *
   * @note:
   *   Fields may be changed after a call to @FT_Attach_File or
   *   @FT_Attach_Stream.
   *
   *   For an OpenType variation font, the values of the following fields can
   *   change after a call to @FT_Set_Var_Design_Coordinates (and friends) if
   *   the font contains an 'MVAR' table: `ascender`, `descender`, `height`,
   *   `underline_position`, and `underline_thickness`.
   *
   *   Especially for TrueType fonts see also the documentation for
   *   @FT_Size_Metrics.
   */
  typedef struct  FT_FaceRec_
  {
    FT_Long           num_faces;
    FT_Long           face_index;

    FT_Long           face_flags;
    FT_Long           style_flags;

    FT_Long           num_glyphs;

    FT_String*        family_name;
    FT_String*        style_name;

    FT_Int            num_fixed_sizes;
    FT_Bitmap_Size*   available_sizes;

    FT_Int            num_charmaps;
    FT_CharMap*       charmaps;

    FT_Generic        generic;

    /* The following member variables (down to `underline_thickness`) */
    /* are only relevant to scalable outlines; cf. @FT_Bitmap_Size    */
    /* for bitmap fonts.                                              */
    FT_BBox           bbox;

    FT_UShort         units_per_EM;
    FT_Short          ascender;
    FT_Short          descender;
    FT_Short          height;

    FT_Short          max_advance_width;
    FT_Short          max_advance_height;

    FT_Short          underline_position;
    FT_Short          underline_thickness;

    FT_GlyphSlot      glyph;
    FT_Size           size;
    FT_CharMap        charmap;

    /* private fields, internal to FreeType */

    FT_Driver         driver;
    FT_Memory         memory;
    FT_Stream         stream;

    FT_ListRec        sizes_list;

    FT_Generic        autohint;   /* face-specific auto-hinter data */
    void*             extensions; /* unused                         */

    FT_Face_Internal  internal;

  } FT_FaceRec;


  /**************************************************************************
   *
   * @enum:
   *   FT_FACE_FLAG_XXX
   *
   * @description:
   *   A list of bit flags used in the `face_flags` field of the @FT_FaceRec
   *   structure.  They inform client applications of properties of the
   *   corresponding face.
   *
   * @values:
   *   FT_FACE_FLAG_SCALABLE ::
   *     The face contains outline glyphs.  Note that a face can contain
   *     bitmap strikes also, i.e., a face can have both this flag and
   *     @FT_FACE_FLAG_FIXED_SIZES set.
   *
   *   FT_FACE_FLAG_FIXED_SIZES ::
   *     The face contains bitmap strikes.  See also the `num_fixed_sizes`
   *     and `available_sizes` fields of @FT_FaceRec.
   *
   *   FT_FACE_FLAG_FIXED_WIDTH ::
   *     The face contains fixed-width characters (like Courier, Lucida,
   *     MonoType, etc.).
   *
   *   FT_FACE_FLAG_SFNT ::
   *     The face uses the SFNT storage scheme.  For now, this means TrueType
   *     and OpenType.
   *
   *   FT_FACE_FLAG_HORIZONTAL ::
   *     The face contains horizontal glyph metrics.  This should be set for
   *     all common formats.
   *
   *   FT_FACE_FLAG_VERTICAL ::
   *     The face contains vertical glyph metrics.  This is only available in
   *     some formats, not all of them.
   *
   *   FT_FACE_FLAG_KERNING ::
   *     The face contains kerning information.  If set, the kerning distance
   *     can be retrieved using the function @FT_Get_Kerning.  Otherwise the
   *     function always returns the vector (0,0).  Note that FreeType
   *     doesn't handle kerning data from the SFNT 'GPOS' table (as present
   *     in many OpenType fonts).
   *
   *   FT_FACE_FLAG_FAST_GLYPHS ::
   *     THIS FLAG IS DEPRECATED.  DO NOT USE OR TEST IT.
   *
   *   FT_FACE_FLAG_MULTIPLE_MASTERS ::
   *     The face contains multiple masters and is capable of interpolating
   *     between them.  Supported formats are Adobe MM, TrueType GX, and
   *     OpenType variation fonts.
   *
   *     See section @multiple_masters for API details.
   *
   *   FT_FACE_FLAG_GLYPH_NAMES ::
   *     The face contains glyph names, which can be retrieved using
   *     @FT_Get_Glyph_Name.  Note that some TrueType fonts contain broken
   *     glyph name tables.  Use the function @FT_Has_PS_Glyph_Names when
   *     needed.
   *
   *   FT_FACE_FLAG_EXTERNAL_STREAM ::
   *     Used internally by FreeType to indicate that a face's stream was
   *     provided by the client application and should not be destroyed when
   *     @FT_Done_Face is called.  Don't read or test this flag.
   *
   *   FT_FACE_FLAG_HINTER ::
   *     The font driver has a hinting machine of its own.  For example, with
   *     TrueType fonts, it makes sense to use data from the SFNT 'gasp'
   *     table only if the native TrueType hinting engine (with the bytecode
   *     interpreter) is available and active.
   *
   *   FT_FACE_FLAG_CID_KEYED ::
   *     The face is CID-keyed.  In that case, the face is not accessed by
   *     glyph indices but by CID values.  For subsetted CID-keyed fonts this
   *     has the consequence that not all index values are a valid argument
   *     to @FT_Load_Glyph.  Only the CID values for which corresponding
   *     glyphs in the subsetted font exist make `FT_Load_Glyph` return
   *     successfully; in all other cases you get an
   *     `FT_Err_Invalid_Argument` error.
   *
   *     Note that CID-keyed fonts that are in an SFNT wrapper (that is, all
   *     OpenType/CFF fonts) don't have this flag set since the glyphs are
   *     accessed in the normal way (using contiguous indices); the
   *     'CID-ness' isn't visible to the application.
   *
   *   FT_FACE_FLAG_TRICKY ::
   *     The face is 'tricky', that is, it always needs the font format's
   *     native hinting engine to get a reasonable result.  A typical example
   *     is the old Chinese font `mingli.ttf` (but not `mingliu.ttc`) that
   *     uses TrueType bytecode instructions to move and scale all of its
   *     subglyphs.
   *
   *     It is not possible to auto-hint such fonts using
   *     @FT_LOAD_FORCE_AUTOHINT; it will also ignore @FT_LOAD_NO_HINTING.
   *     You have to set both @FT_LOAD_NO_HINTING and @FT_LOAD_NO_AUTOHINT to
   *     really disable hinting; however, you probably never want this except
   *     for demonstration purposes.
   *
   *     Currently, there are about a dozen TrueType fonts in the list of
   *     tricky fonts; they are hard-coded in file `ttobjs.c`.
   *
   *   FT_FACE_FLAG_COLOR ::
   *     [Since 2.5.1] The face has color glyph tables.  See @FT_LOAD_COLOR
   *     for more information.
   *
   *   FT_FACE_FLAG_VARIATION ::
   *     [Since 2.9] Set if the current face (or named instance) has been
   *     altered with @FT_Set_MM_Design_Coordinates,
   *     @FT_Set_Var_Design_Coordinates, @FT_Set_Var_Blend_Coordinates, or
   *     @FT_Set_MM_WeightVector to select a non-default instance.
   *
   *   FT_FACE_FLAG_SVG ::
   *     [Since 2.12] The face has an 'SVG~' OpenType table.
   *
   *   FT_FACE_FLAG_SBIX ::
   *     [Since 2.12] The face has an 'sbix' OpenType table *and* outlines.
   *     For such fonts, @FT_FACE_FLAG_SCALABLE is not set by default to
   *     retain backward compatibility.
   *
   *   FT_FACE_FLAG_SBIX_OVERLAY ::
   *     [Since 2.12] The face has an 'sbix' OpenType table where outlines
   *     should be drawn on top of bitmap strikes.
   *
   */
#define FT_FACE_FLAG_SCALABLE          ( 1L <<  0 )
#define FT_FACE_FLAG_FIXED_SIZES       ( 1L <<  1 )
#define FT_FACE_FLAG_FIXED_WIDTH       ( 1L <<  2 )
#define FT_FACE_FLAG_SFNT              ( 1L <<  3 )
#define FT_FACE_FLAG_HORIZONTAL        ( 1L <<  4 )
#define FT_FACE_FLAG_VERTICAL          ( 1L <<  5 )
#define FT_FACE_FLAG_KERNING           ( 1L <<  6 )
#define FT_FACE_FLAG_FAST_GLYPHS       ( 1L <<  7 )
#define FT_FACE_FLAG_MULTIPLE_MASTERS  ( 1L <<  8 )
#define FT_FACE_FLAG_GLYPH_NAMES       ( 1L <<  9 )
#define FT_FACE_FLAG_EXTERNAL_STREAM   ( 1L << 10 )
#define FT_FACE_FLAG_HINTER            ( 1L << 11 )
#define FT_FACE_FLAG_CID_KEYED         ( 1L << 12 )
#define FT_FACE_FLAG_TRICKY            ( 1L << 13 )
#define FT_FACE_FLAG_COLOR             ( 1L << 14 )
#define FT_FACE_FLAG_VARIATION         ( 1L << 15 )
#define FT_FACE_FLAG_SVG               ( 1L << 16 )
#define FT_FACE_FLAG_SBIX              ( 1L << 17 )
#define FT_FACE_FLAG_SBIX_OVERLAY      ( 1L << 18 )


  /**************************************************************************
   *
   * @section:
   *   font_testing_macros
   *
   */

  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_HORIZONTAL
   *
   * @description:
   *   A macro that returns true whenever a face object contains horizontal
   *   metrics (this is true for all font formats though).
   *
   * @also:
   *   @FT_HAS_VERTICAL can be used to check for vertical metrics.
   *
   */
#define FT_HAS_HORIZONTAL( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_HORIZONTAL ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_VERTICAL
   *
   * @description:
   *   A macro that returns true whenever a face object contains real
   *   vertical metrics (and not only synthesized ones).
   *
   */
#define FT_HAS_VERTICAL( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_VERTICAL ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_KERNING
   *
   * @description:
   *   A macro that returns true whenever a face object contains kerning data
   *   that can be accessed with @FT_Get_Kerning.
   *
   */
#define FT_HAS_KERNING( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_KERNING ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_SCALABLE
   *
   * @description:
   *   A macro that returns true whenever a face object contains a scalable
   *   font face (true for TrueType, Type~1, Type~42, CID, OpenType/CFF, and
   *   PFR font formats).
   *
   */
#define FT_IS_SCALABLE( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_SCALABLE ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_SFNT
   *
   * @description:
   *   A macro that returns true whenever a face object contains a font whose
   *   format is based on the SFNT storage scheme.  This usually means:
   *   TrueType fonts, OpenType fonts, as well as SFNT-based embedded bitmap
   *   fonts.
   *
   *   If this macro is true, all functions defined in @FT_SFNT_NAMES_H and
   *   @FT_TRUETYPE_TABLES_H are available.
   *
   */
#define FT_IS_SFNT( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_SFNT ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_FIXED_WIDTH
   *
   * @description:
   *   A macro that returns true whenever a face object contains a font face
   *   that contains fixed-width (or 'monospace', 'fixed-pitch', etc.)
   *   glyphs.
   *
   */
#define FT_IS_FIXED_WIDTH( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_FIXED_WIDTH ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_FIXED_SIZES
   *
   * @description:
   *   A macro that returns true whenever a face object contains some
   *   embedded bitmaps.  See the `available_sizes` field of the @FT_FaceRec
   *   structure.
   *
   */
#define FT_HAS_FIXED_SIZES( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_FIXED_SIZES ) )


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   */

  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_FAST_GLYPHS
   *
   * @description:
   *   Deprecated.
   *
   */
#define FT_HAS_FAST_GLYPHS( face )  0


  /**************************************************************************
   *
   * @section:
   *   font_testing_macros
   *
   */

  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_GLYPH_NAMES
   *
   * @description:
   *   A macro that returns true whenever a face object contains some glyph
   *   names that can be accessed through @FT_Get_Glyph_Name.
   *
   */
#define FT_HAS_GLYPH_NAMES( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_GLYPH_NAMES ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_MULTIPLE_MASTERS
   *
   * @description:
   *   A macro that returns true whenever a face object contains some
   *   multiple masters.  The functions provided by @FT_MULTIPLE_MASTERS_H
   *   are then available to choose the exact design you want.
   *
   */
#define FT_HAS_MULTIPLE_MASTERS( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_NAMED_INSTANCE
   *
   * @description:
   *   A macro that returns true whenever a face object is a named instance
   *   of a GX or OpenType variation font.
   *
   *   [Since 2.9] Changing the design coordinates with
   *   @FT_Set_Var_Design_Coordinates or @FT_Set_Var_Blend_Coordinates does
   *   not influence the return value of this macro (only
   *   @FT_Set_Named_Instance does that).
   *
   * @since:
   *   2.7
   *
   */
#define FT_IS_NAMED_INSTANCE( face ) \
          ( !!( (face)->face_index & 0x7FFF0000L ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_VARIATION
   *
   * @description:
   *   A macro that returns true whenever a face object has been altered by
   *   @FT_Set_MM_Design_Coordinates, @FT_Set_Var_Design_Coordinates,
   *   @FT_Set_Var_Blend_Coordinates, or @FT_Set_MM_WeightVector.
   *
   * @since:
   *   2.9
   *
   */
#define FT_IS_VARIATION( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_VARIATION ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_CID_KEYED
   *
   * @description:
   *   A macro that returns true whenever a face object contains a CID-keyed
   *   font.  See the discussion of @FT_FACE_FLAG_CID_KEYED for more details.
   *
   *   If this macro is true, all functions defined in @FT_CID_H are
   *   available.
   *
   */
#define FT_IS_CID_KEYED( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_CID_KEYED ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_IS_TRICKY
   *
   * @description:
   *   A macro that returns true whenever a face represents a 'tricky' font.
   *   See the discussion of @FT_FACE_FLAG_TRICKY for more details.
   *
   */
#define FT_IS_TRICKY( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_TRICKY ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_COLOR
   *
   * @description:
   *   A macro that returns true whenever a face object contains tables for
   *   color glyphs.
   *
   * @since:
   *   2.5.1
   *
   */
#define FT_HAS_COLOR( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_COLOR ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_SVG
   *
   * @description:
   *   A macro that returns true whenever a face object contains an 'SVG~'
   *   OpenType table.
   *
   * @since:
   *   2.12
   */
#define FT_HAS_SVG( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_SVG ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_SBIX
   *
   * @description:
   *   A macro that returns true whenever a face object contains an 'sbix'
   *   OpenType table *and* outline glyphs.
   *
   *   Currently, FreeType only supports bitmap glyphs in PNG format for this
   *   table (i.e., JPEG and TIFF formats are unsupported, as are
   *   Apple-specific formats not part of the OpenType specification).
   *
   * @note:
   *   For backward compatibility, a font with an 'sbix' table is treated as
   *   a bitmap-only face.  Using @FT_Open_Face with
   *   @FT_PARAM_TAG_IGNORE_SBIX, an application can switch off 'sbix'
   *   handling so that the face is treated as an ordinary outline font with
   *   scalable outlines.
   *
   *   Here is some pseudo code that roughly illustrates how to implement
   *   'sbix' handling according to the OpenType specification.
   *
   * ```
   *   if ( FT_HAS_SBIX( face ) )
   *   {
   *     // open font as a scalable one without sbix handling
   *     FT_Face       face2;
   *     FT_Parameter  param = { FT_PARAM_TAG_IGNORE_SBIX, NULL };
   *     FT_Open_Args  args  = { FT_OPEN_PARAMS | ...,
   *                             ...,
   *                             1, &param };
   *
   *
   *     FT_Open_Face( library, &args, 0, &face2 );
   *
   *     <sort `face->available_size` as necessary into
   *      `preferred_sizes`[*]>
   *
   *     for ( i = 0; i < face->num_fixed_sizes; i++ )
   *     {
   *       size = preferred_sizes[i].size;
   *
   *       error = FT_Set_Pixel_Sizes( face, size, size );
   *       <error handling omitted>
   *
   *       // check whether we have a glyph in a bitmap strike
   *       error = FT_Load_Glyph( face,
   *                              glyph_index,
   *                              FT_LOAD_SBITS_ONLY          |
   *                              FT_LOAD_BITMAP_METRICS_ONLY );
   *       if ( error == FT_Err_Invalid_Argument )
   *         continue;
   *       else if ( error )
   *         <other error handling omitted>
   *       else
   *         break;
   *     }
   *
   *     if ( i != face->num_fixed_sizes )
   *       <load embedded bitmap with `FT_Load_Glyph`,
   *        scale it, display it, etc.>
   *
   *     if ( i == face->num_fixed_sizes  ||
   *          FT_HAS_SBIX_OVERLAY( face ) )
   *       <use `face2` to load outline glyph with `FT_Load_Glyph`,
   *        scale it, display it on top of the bitmap, etc.>
   *   }
   * ```
   *
   * [*] Assuming a target value of 400dpi and available strike sizes 100,
   * 200, 300, and 400dpi, a possible order might be [400, 200, 300, 100]:
   * scaling 200dpi to 400dpi usually gives better results than scaling
   * 300dpi to 400dpi; it is also much faster.  However, scaling 100dpi to
   * 400dpi can yield a too pixelated result, thus the preference might be
   * 300dpi over 100dpi.
   *
   * @since:
   *   2.12
   */
#define FT_HAS_SBIX( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_SBIX ) )


  /**************************************************************************
   *
   * @macro:
   *   FT_HAS_SBIX_OVERLAY
   *
   * @description:
   *   A macro that returns true whenever a face object contains an 'sbix'
   *   OpenType table with bit~1 in its `flags` field set, instructing the
   *   application to overlay the bitmap strike with the corresponding
   *   outline glyph.  See @FT_HAS_SBIX for pseudo code how to use it.
   *
   * @since:
   *   2.12
   */
#define FT_HAS_SBIX_OVERLAY( face ) \
          ( !!( (face)->face_flags & FT_FACE_FLAG_SBIX_OVERLAY ) )


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   */

  /**************************************************************************
   *
   * @enum:
   *   FT_STYLE_FLAG_XXX
   *
   * @description:
   *   A list of bit flags to indicate the style of a given face.  These are
   *   used in the `style_flags` field of @FT_FaceRec.
   *
   * @values:
   *   FT_STYLE_FLAG_ITALIC ::
   *     The face style is italic or oblique.
   *
   *   FT_STYLE_FLAG_BOLD ::
   *     The face is bold.
   *
   * @note:
   *   The style information as provided by FreeType is very basic.  More
   *   details are beyond the scope and should be done on a higher level (for
   *   example, by analyzing various fields of the 'OS/2' table in SFNT based
   *   fonts).
   */
#define FT_STYLE_FLAG_ITALIC  ( 1 << 0 )
#define FT_STYLE_FLAG_BOLD    ( 1 << 1 )


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   */

  /**************************************************************************
   *
   * @type:
   *   FT_Size_Internal
   *
   * @description:
   *   An opaque handle to an `FT_Size_InternalRec` structure, used to model
   *   private data of a given @FT_Size object.
   */
  typedef struct FT_Size_InternalRec_*  FT_Size_Internal;


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_Size_Metrics
   *
   * @description:
   *   The size metrics structure gives the metrics of a size object.
   *
   * @fields:
   *   x_ppem ::
   *     The width of the scaled EM square in pixels, hence the term 'ppem'
   *     (pixels per EM).  It is also referred to as 'nominal width'.
   *
   *   y_ppem ::
   *     The height of the scaled EM square in pixels, hence the term 'ppem'
   *     (pixels per EM).  It is also referred to as 'nominal height'.
   *
   *   x_scale ::
   *     A 16.16 fractional scaling value to convert horizontal metrics from
   *     font units to 26.6 fractional pixels.  Only relevant for scalable
   *     font formats.
   *
   *   y_scale ::
   *     A 16.16 fractional scaling value to convert vertical metrics from
   *     font units to 26.6 fractional pixels.  Only relevant for scalable
   *     font formats.
   *
   *   ascender ::
   *     The ascender in 26.6 fractional pixels, rounded up to an integer
   *     value.  See @FT_FaceRec for the details.
   *
   *   descender ::
   *     The descender in 26.6 fractional pixels, rounded down to an integer
   *     value.  See @FT_FaceRec for the details.
   *
   *   height ::
   *     The height in 26.6 fractional pixels, rounded to an integer value.
   *     See @FT_FaceRec for the details.
   *
   *   max_advance ::
   *     The maximum advance width in 26.6 fractional pixels, rounded to an
   *     integer value.  See @FT_FaceRec for the details.
   *
   * @note:
   *   The scaling values, if relevant, are determined first during a size
   *   changing operation.  The remaining fields are then set by the driver.
   *   For scalable formats, they are usually set to scaled values of the
   *   corresponding fields in @FT_FaceRec.  Some values like ascender or
   *   descender are rounded for historical reasons; more precise values (for
   *   outline fonts) can be derived by scaling the corresponding @FT_FaceRec
   *   values manually, with code similar to the following.
   *
   *   ```
   *     scaled_ascender = FT_MulFix( face->ascender,
   *                                  size_metrics->y_scale );
   *   ```
   *
   *   Note that due to glyph hinting and the selected rendering mode these
   *   values are usually not exact; consequently, they must be treated as
   *   unreliable with an error margin of at least one pixel!
   *
   *   Indeed, the only way to get the exact metrics is to render _all_
   *   glyphs.  As this would be a definite performance hit, it is up to
   *   client applications to perform such computations.
   *
   *   The `FT_Size_Metrics` structure is valid for bitmap fonts also.
   *
   *
   *   **TrueType fonts with native bytecode hinting**
   *
   *   All applications that handle TrueType fonts with native hinting must
   *   be aware that TTFs expect different rounding of vertical font
   *   dimensions.  The application has to cater for this, especially if it
   *   wants to rely on a TTF's vertical data (for example, to properly align
   *   box characters vertically).
   *
   *   Only the application knows _in advance_ that it is going to use native
   *   hinting for TTFs!  FreeType, on the other hand, selects the hinting
   *   mode not at the time of creating an @FT_Size object but much later,
   *   namely while calling @FT_Load_Glyph.
   *
   *   Here is some pseudo code that illustrates a possible solution.
   *
   *   ```
   *     font_format = FT_Get_Font_Format( face );
   *
   *     if ( !strcmp( font_format, "TrueType" ) &&
   *          do_native_bytecode_hinting         )
   *     {
   *       ascender  = ROUND( FT_MulFix( face->ascender,
   *                                     size_metrics->y_scale ) );
   *       descender = ROUND( FT_MulFix( face->descender,
   *                                     size_metrics->y_scale ) );
   *     }
   *     else
   *     {
   *       ascender  = size_metrics->ascender;
   *       descender = size_metrics->descender;
   *     }
   *
   *     height      = size_metrics->height;
   *     max_advance = size_metrics->max_advance;
   *   ```
   */
  typedef struct  FT_Size_Metrics_
  {
    FT_UShort  x_ppem;      /* horizontal pixels per EM               */
    FT_UShort  y_ppem;      /* vertical pixels per EM                 */

    FT_Fixed   x_scale;     /* scaling values used to convert font    */
    FT_Fixed   y_scale;     /* units to 26.6 fractional pixels        */

    FT_Pos     ascender;    /* ascender in 26.6 frac. pixels          */
    FT_Pos     descender;   /* descender in 26.6 frac. pixels         */
    FT_Pos     height;      /* text height in 26.6 frac. pixels       */
    FT_Pos     max_advance; /* max horizontal advance, in 26.6 pixels */

  } FT_Size_Metrics;


  /**************************************************************************
   *
   * @struct:
   *   FT_SizeRec
   *
   * @description:
   *   FreeType root size class structure.  A size object models a face
   *   object at a given size.
   *
   * @fields:
   *   face ::
   *     Handle to the parent face object.
   *
   *   generic ::
   *     A typeless pointer, unused by the FreeType library or any of its
   *     drivers.  It can be used by client applications to link their own
   *     data to each size object.
   *
   *   metrics ::
   *     Metrics for this size object.  This field is read-only.
   */
  typedef struct  FT_SizeRec_
  {
    FT_Face           face;      /* parent face object              */
    FT_Generic        generic;   /* generic pointer for client uses */
    FT_Size_Metrics   metrics;   /* size metrics                    */
    FT_Size_Internal  internal;

  } FT_SizeRec;


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_SubGlyph
   *
   * @description:
   *   The subglyph structure is an internal object used to describe
   *   subglyphs (for example, in the case of composites).
   *
   * @note:
   *   The subglyph implementation is not part of the high-level API, hence
   *   the forward structure declaration.
   *
   *   You can however retrieve subglyph information with
   *   @FT_Get_SubGlyph_Info.
   */
  typedef struct FT_SubGlyphRec_*  FT_SubGlyph;


  /**************************************************************************
   *
   * @type:
   *   FT_Slot_Internal
   *
   * @description:
   *   An opaque handle to an `FT_Slot_InternalRec` structure, used to model
   *   private data of a given @FT_GlyphSlot object.
   */
  typedef struct FT_Slot_InternalRec_*  FT_Slot_Internal;


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @struct:
   *   FT_GlyphSlotRec
   *
   * @description:
   *   FreeType root glyph slot class structure.  A glyph slot is a container
   *   where individual glyphs can be loaded, be they in outline or bitmap
   *   format.
   *
   * @fields:
   *   library ::
   *     A handle to the FreeType library instance this slot belongs to.
   *
   *   face ::
   *     A handle to the parent face object.
   *
   *   next ::
   *     In some cases (like some font tools), several glyph slots per face
   *     object can be a good thing.  As this is rare, the glyph slots are
   *     listed through a direct, single-linked list using its `next` field.
   *
   *   glyph_index ::
   *     [Since 2.10] The glyph index passed as an argument to @FT_Load_Glyph
   *     while initializing the glyph slot.
   *
   *   generic ::
   *     A typeless pointer unused by the FreeType library or any of its
   *     drivers.  It can be used by client applications to link their own
   *     data to each glyph slot object.
   *
   *   metrics ::
   *     The metrics of the last loaded glyph in the slot.  The returned
   *     values depend on the last load flags (see the @FT_Load_Glyph API
   *     function) and can be expressed either in 26.6 fractional pixels or
   *     font units.
   *
   *     Note that even when the glyph image is transformed, the metrics are
   *     not.
   *
   *   linearHoriAdvance ::
   *     The advance width of the unhinted glyph.  Its value is expressed in
   *     16.16 fractional pixels, unless @FT_LOAD_LINEAR_DESIGN is set when
   *     loading the glyph.  This field can be important to perform correct
   *     WYSIWYG layout.  Only relevant for scalable glyphs.
   *
   *   linearVertAdvance ::
   *     The advance height of the unhinted glyph.  Its value is expressed in
   *     16.16 fractional pixels, unless @FT_LOAD_LINEAR_DESIGN is set when
   *     loading the glyph.  This field can be important to perform correct
   *     WYSIWYG layout.  Only relevant for scalable glyphs.
   *
   *   advance ::
   *     This shorthand is, depending on @FT_LOAD_IGNORE_TRANSFORM, the
   *     transformed (hinted) advance width for the glyph, in 26.6 fractional
   *     pixel format.  As specified with @FT_LOAD_VERTICAL_LAYOUT, it uses
   *     either the `horiAdvance` or the `vertAdvance` value of `metrics`
   *     field.
   *
   *   format ::
   *     This field indicates the format of the image contained in the glyph
   *     slot.  Typically @FT_GLYPH_FORMAT_BITMAP, @FT_GLYPH_FORMAT_OUTLINE,
   *     or @FT_GLYPH_FORMAT_COMPOSITE, but other values are possible.
   *
   *   bitmap ::
   *     This field is used as a bitmap descriptor.  Note that the address
   *     and content of the bitmap buffer can change between calls of
   *     @FT_Load_Glyph and a few other functions.
   *
   *   bitmap_left ::
   *     The bitmap's left bearing expressed in integer pixels.
   *
   *   bitmap_top ::
   *     The bitmap's top bearing expressed in integer pixels.  This is the
   *     distance from the baseline to the top-most glyph scanline, upwards
   *     y~coordinates being **positive**.
   *
   *   outline ::
   *     The outline descriptor for the current glyph image if its format is
   *     @FT_GLYPH_FORMAT_OUTLINE.  Once a glyph is loaded, `outline` can be
   *     transformed, distorted, emboldened, etc.  However, it must not be
   *     freed.
   *
   *     [Since 2.10.1] If @FT_LOAD_NO_SCALE is set, outline coordinates of
   *     OpenType variation fonts for a selected instance are internally
   *     handled as 26.6 fractional font units but returned as (rounded)
   *     integers, as expected.  To get unrounded font units, don't use
   *     @FT_LOAD_NO_SCALE but load the glyph with @FT_LOAD_NO_HINTING and
   *     scale it, using the font's `units_per_EM` value as the ppem.
   *
   *   num_subglyphs ::
   *     The number of subglyphs in a composite glyph.  This field is only
   *     valid for the composite glyph format that should normally only be
   *     loaded with the @FT_LOAD_NO_RECURSE flag.
   *
   *   subglyphs ::
   *     An array of subglyph descriptors for composite glyphs.  There are
   *     `num_subglyphs` elements in there.  Currently internal to FreeType.
   *
   *   control_data ::
   *     Certain font drivers can also return the control data for a given
   *     glyph image (e.g.  TrueType bytecode, Type~1 charstrings, etc.).
   *     This field is a pointer to such data; it is currently internal to
   *     FreeType.
   *
   *   control_len ::
   *     This is the length in bytes of the control data.  Currently internal
   *     to FreeType.
   *
   *   other ::
   *     Reserved.
   *
   *   lsb_delta ::
   *     The difference between hinted and unhinted left side bearing while
   *     auto-hinting is active.  Zero otherwise.
   *
   *   rsb_delta ::
   *     The difference between hinted and unhinted right side bearing while
   *     auto-hinting is active.  Zero otherwise.
   *
   * @note:
   *   If @FT_Load_Glyph is called with default flags (see @FT_LOAD_DEFAULT)
   *   the glyph image is loaded in the glyph slot in its native format
   *   (e.g., an outline glyph for TrueType and Type~1 formats).  [Since 2.9]
   *   The prospective bitmap metrics are calculated according to
   *   @FT_LOAD_TARGET_XXX and other flags even for the outline glyph, even
   *   if @FT_LOAD_RENDER is not set.
   *
   *   This image can later be converted into a bitmap by calling
   *   @FT_Render_Glyph.  This function searches the current renderer for the
   *   native image's format, then invokes it.
   *
   *   The renderer is in charge of transforming the native image through the
   *   slot's face transformation fields, then converting it into a bitmap
   *   that is returned in `slot->bitmap`.
   *
   *   Note that `slot->bitmap_left` and `slot->bitmap_top` are also used to
   *   specify the position of the bitmap relative to the current pen
   *   position (e.g., coordinates (0,0) on the baseline).  Of course,
   *   `slot->format` is also changed to @FT_GLYPH_FORMAT_BITMAP.
   *
   *   Here is a small pseudo code fragment that shows how to use `lsb_delta`
   *   and `rsb_delta` to do fractional positioning of glyphs:
   *
   *   ```
   *     FT_GlyphSlot  slot     = face->glyph;
   *     FT_Pos        origin_x = 0;
   *
   *
   *     for all glyphs do
   *       <load glyph with `FT_Load_Glyph'>
   *
   *       FT_Outline_Translate( slot->outline, origin_x & 63, 0 );
   *
   *       <save glyph image, or render glyph, or ...>
   *
   *       <compute kern between current and next glyph
   *        and add it to `origin_x'>
   *
   *       origin_x += slot->advance.x;
   *       origin_x += slot->lsb_delta - slot->rsb_delta;
   *     endfor
   *   ```
   *
   *   Here is another small pseudo code fragment that shows how to use
   *   `lsb_delta` and `rsb_delta` to improve integer positioning of glyphs:
   *
   *   ```
   *     FT_GlyphSlot  slot           = face->glyph;
   *     FT_Pos        origin_x       = 0;
   *     FT_Pos        prev_rsb_delta = 0;
   *
   *
   *     for all glyphs do
   *       <compute kern between current and previous glyph
   *        and add it to `origin_x'>
   *
   *       <load glyph with `FT_Load_Glyph'>
   *
   *       if ( prev_rsb_delta - slot->lsb_delta >  32 )
   *         origin_x -= 64;
   *       else if ( prev_rsb_delta - slot->lsb_delta < -31 )
   *         origin_x += 64;
   *
   *       prev_rsb_delta = slot->rsb_delta;
   *
   *       <save glyph image, or render glyph, or ...>
   *
   *       origin_x += slot->advance.x;
   *     endfor
   *   ```
   *
   *   If you use strong auto-hinting, you **must** apply these delta values!
   *   Otherwise you will experience far too large inter-glyph spacing at
   *   small rendering sizes in most cases.  Note that it doesn't harm to use
   *   the above code for other hinting modes also, since the delta values
   *   are zero then.
   */
  typedef struct  FT_GlyphSlotRec_
  {
    FT_Library        library;
    FT_Face           face;
    FT_GlyphSlot      next;
    FT_UInt           glyph_index; /* new in 2.10; was reserved previously */
    FT_Generic        generic;

    FT_Glyph_Metrics  metrics;
    FT_Fixed          linearHoriAdvance;
    FT_Fixed          linearVertAdvance;
    FT_Vector         advance;

    FT_Glyph_Format   format;

    FT_Bitmap         bitmap;
    FT_Int            bitmap_left;
    FT_Int            bitmap_top;

    FT_Outline        outline;

    FT_UInt           num_subglyphs;
    FT_SubGlyph       subglyphs;

    void*             control_data;
    long              control_len;

    FT_Pos            lsb_delta;
    FT_Pos            rsb_delta;

    void*             other;

    FT_Slot_Internal  internal;

  } FT_GlyphSlotRec;


  /*************************************************************************/
  /*************************************************************************/
  /*                                                                       */
  /*                         F U N C T I O N S                             */
  /*                                                                       */
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @section:
   *   library_setup
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Init_FreeType
   *
   * @description:
   *   Initialize a new FreeType library object.  The set of modules that are
   *   registered by this function is determined at build time.
   *
   * @output:
   *   alibrary ::
   *     A handle to a new library object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   In case you want to provide your own memory allocating routines, use
   *   @FT_New_Library instead, followed by a call to @FT_Add_Default_Modules
   *   (or a series of calls to @FT_Add_Module) and
   *   @FT_Set_Default_Properties.
   *
   *   See the documentation of @FT_Library and @FT_Face for multi-threading
   *   issues.
   *
   *   If you need reference-counting (cf. @FT_Reference_Library), use
   *   @FT_New_Library and @FT_Done_Library.
   *
   *   If compilation option `FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES` is
   *   set, this function reads the `FREETYPE_PROPERTIES` environment
   *   variable to control driver properties.  See section @properties for
   *   more.
   */
  FT_EXPORT( FT_Error )
  FT_Init_FreeType( FT_Library  *alibrary );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_FreeType
   *
   * @description:
   *   Destroy a given FreeType library object and all of its children,
   *   including resources, drivers, faces, sizes, etc.
   *
   * @input:
   *   library ::
   *     A handle to the target library object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Done_FreeType( FT_Library  library );


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   */

  /**************************************************************************
   *
   * @enum:
   *   FT_OPEN_XXX
   *
   * @description:
   *   A list of bit field constants used within the `flags` field of the
   *   @FT_Open_Args structure.
   *
   * @values:
   *   FT_OPEN_MEMORY ::
   *     This is a memory-based stream.
   *
   *   FT_OPEN_STREAM ::
   *     Copy the stream from the `stream` field.
   *
   *   FT_OPEN_PATHNAME ::
   *     Create a new input stream from a C~path name.
   *
   *   FT_OPEN_DRIVER ::
   *     Use the `driver` field.
   *
   *   FT_OPEN_PARAMS ::
   *     Use the `num_params` and `params` fields.
   *
   * @note:
   *   The `FT_OPEN_MEMORY`, `FT_OPEN_STREAM`, and `FT_OPEN_PATHNAME` flags
   *   are mutually exclusive.
   */
#define FT_OPEN_MEMORY    0x1
#define FT_OPEN_STREAM    0x2
#define FT_OPEN_PATHNAME  0x4
#define FT_OPEN_DRIVER    0x8
#define FT_OPEN_PARAMS    0x10


  /* these constants are deprecated; use the corresponding `FT_OPEN_XXX` */
  /* values instead                                                      */
#define ft_open_memory    FT_OPEN_MEMORY
#define ft_open_stream    FT_OPEN_STREAM
#define ft_open_pathname  FT_OPEN_PATHNAME
#define ft_open_driver    FT_OPEN_DRIVER
#define ft_open_params    FT_OPEN_PARAMS


  /**************************************************************************
   *
   * @struct:
   *   FT_Parameter
   *
   * @description:
   *   A simple structure to pass more or less generic parameters to
   *   @FT_Open_Face and @FT_Face_Properties.
   *
   * @fields:
   *   tag ::
   *     A four-byte identification tag.
   *
   *   data ::
   *     A pointer to the parameter data.
   *
   * @note:
   *   The ID and function of parameters are driver-specific.  See section
   *   @parameter_tags for more information.
   */
  typedef struct  FT_Parameter_
  {
    FT_ULong    tag;
    FT_Pointer  data;

  } FT_Parameter;


  /**************************************************************************
   *
   * @struct:
   *   FT_Open_Args
   *
   * @description:
   *   A structure to indicate how to open a new font file or stream.  A
   *   pointer to such a structure can be used as a parameter for the
   *   functions @FT_Open_Face and @FT_Attach_Stream.
   *
   * @fields:
   *   flags ::
   *     A set of bit flags indicating how to use the structure.
   *
   *   memory_base ::
   *     The first byte of the file in memory.
   *
   *   memory_size ::
   *     The size in bytes of the file in memory.
   *
   *   pathname ::
   *     A pointer to an 8-bit file pathname, which must be a C~string (i.e.,
   *     no null bytes except at the very end).  The pointer is not owned by
   *     FreeType.
   *
   *   stream ::
   *     A handle to a source stream object.
   *
   *   driver ::
   *     This field is exclusively used by @FT_Open_Face; it simply specifies
   *     the font driver to use for opening the face.  If set to `NULL`,
   *     FreeType tries to load the face with each one of the drivers in its
   *     list.
   *
   *   num_params ::
   *     The number of extra parameters.
   *
   *   params ::
   *     Extra parameters passed to the font driver when opening a new face.
   *
   * @note:
   *   The stream type is determined by the contents of `flags`:
   *
   *   If the @FT_OPEN_MEMORY bit is set, assume that this is a memory file
   *   of `memory_size` bytes, located at `memory_address`.  The data are not
   *   copied, and the client is responsible for releasing and destroying
   *   them _after_ the corresponding call to @FT_Done_Face.
   *
   *   Otherwise, if the @FT_OPEN_STREAM bit is set, assume that a custom
   *   input stream `stream` is used.
   *
   *   Otherwise, if the @FT_OPEN_PATHNAME bit is set, assume that this is a
   *   normal file and use `pathname` to open it.
   *
   *   If none of the above bits are set or if multiple are set at the same
   *   time, the flags are invalid and @FT_Open_Face fails.
   *
   *   If the @FT_OPEN_DRIVER bit is set, @FT_Open_Face only tries to open
   *   the file with the driver whose handler is in `driver`.
   *
   *   If the @FT_OPEN_PARAMS bit is set, the parameters given by
   *   `num_params` and `params` is used.  They are ignored otherwise.
   *
   *   Ideally, both the `pathname` and `params` fields should be tagged as
   *   'const'; this is missing for API backward compatibility.  In other
   *   words, applications should treat them as read-only.
   */
  typedef struct  FT_Open_Args_
  {
    FT_UInt         flags;
    const FT_Byte*  memory_base;
    FT_Long         memory_size;
    FT_String*      pathname;
    FT_Stream       stream;
    FT_Module       driver;
    FT_Int          num_params;
    FT_Parameter*   params;

  } FT_Open_Args;


  /**************************************************************************
   *
   * @function:
   *   FT_New_Face
   *
   * @description:
   *   Call @FT_Open_Face to open a font by its pathname.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   pathname ::
   *     A path to the font file.
   *
   *   face_index ::
   *     See @FT_Open_Face for a detailed description of this parameter.
   *
   * @output:
   *   aface ::
   *     A handle to a new face object.  If `face_index` is greater than or
   *     equal to zero, it must be non-`NULL`.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The `pathname` string should be recognizable as such by a standard
   *   `fopen` call on your system; in particular, this means that `pathname`
   *   must not contain null bytes.  If that is not sufficient to address all
   *   file name possibilities (for example, to handle wide character file
   *   names on Windows in UTF-16 encoding) you might use @FT_Open_Face to
   *   pass a memory array or a stream object instead.
   *
   *   Use @FT_Done_Face to destroy the created @FT_Face object (along with
   *   its slot and sizes).
   */
  FT_EXPORT( FT_Error )
  FT_New_Face( FT_Library   library,
               const char*  filepathname,
               FT_Long      face_index,
               FT_Face     *aface );


  /**************************************************************************
   *
   * @function:
   *   FT_New_Memory_Face
   *
   * @description:
   *   Call @FT_Open_Face to open a font that has been loaded into memory.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   file_base ::
   *     A pointer to the beginning of the font data.
   *
   *   file_size ::
   *     The size of the memory chunk used by the font data.
   *
   *   face_index ::
   *     See @FT_Open_Face for a detailed description of this parameter.
   *
   * @output:
   *   aface ::
   *     A handle to a new face object.  If `face_index` is greater than or
   *     equal to zero, it must be non-`NULL`.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   You must not deallocate the memory before calling @FT_Done_Face.
   */
  FT_EXPORT( FT_Error )
  FT_New_Memory_Face( FT_Library      library,
                      const FT_Byte*  file_base,
                      FT_Long         file_size,
                      FT_Long         face_index,
                      FT_Face        *aface );


  /**************************************************************************
   *
   * @function:
   *   FT_Open_Face
   *
   * @description:
   *   Create a face object from a given resource described by @FT_Open_Args.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   args ::
   *     A pointer to an `FT_Open_Args` structure that must be filled by the
   *     caller.
   *
   *   face_index ::
   *     This field holds two different values.  Bits 0-15 are the index of
   *     the face in the font file (starting with value~0).  Set it to~0 if
   *     there is only one face in the font file.
   *
   *     [Since 2.6.1] Bits 16-30 are relevant to GX and OpenType variation
   *     fonts only, specifying the named instance index for the current face
   *     index (starting with value~1; value~0 makes FreeType ignore named
   *     instances).  For non-variation fonts, bits 16-30 are ignored.
   *     Assuming that you want to access the third named instance in face~4,
   *     `face_index` should be set to 0x00030004.  If you want to access
   *     face~4 without variation handling, simply set `face_index` to
   *     value~4.
   *
   *     `FT_Open_Face` and its siblings can be used to quickly check whether
   *     the font format of a given font resource is supported by FreeType.
   *     In general, if the `face_index` argument is negative, the function's
   *     return value is~0 if the font format is recognized, or non-zero
   *     otherwise.  The function allocates a more or less empty face handle
   *     in `*aface` (if `aface` isn't `NULL`); the only two useful fields in
   *     this special case are `face->num_faces` and `face->style_flags`.
   *     For any negative value of `face_index`, `face->num_faces` gives the
   *     number of faces within the font file.  For the negative value
   *     '-(N+1)' (with 'N' a non-negative 16-bit value), bits 16-30 in
   *     `face->style_flags` give the number of named instances in face 'N'
   *     if we have a variation font (or zero otherwise).  After examination,
   *     the returned @FT_Face structure should be deallocated with a call to
   *     @FT_Done_Face.
   *
   * @output:
   *   aface ::
   *     A handle to a new face object.  If `face_index` is greater than or
   *     equal to zero, it must be non-`NULL`.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Unlike FreeType 1.x, this function automatically creates a glyph slot
   *   for the face object that can be accessed directly through
   *   `face->glyph`.
   *
   *   Each new face object created with this function also owns a default
   *   @FT_Size object, accessible as `face->size`.
   *
   *   One @FT_Library instance can have multiple face objects, that is,
   *   @FT_Open_Face and its siblings can be called multiple times using the
   *   same `library` argument.
   *
   *   See the discussion of reference counters in the description of
   *   @FT_Reference_Face.
   *
   *   If `FT_OPEN_STREAM` is set in `args->flags`, the stream in
   *   `args->stream` is automatically closed before this function returns
   *   any error (including `FT_Err_Invalid_Argument`).
   *
   * @example:
   *   To loop over all faces, use code similar to the following snippet
   *   (omitting the error handling).
   *
   *   ```
   *     ...
   *     FT_Face  face;
   *     FT_Long  i, num_faces;
   *
   *
   *     error = FT_Open_Face( library, args, -1, &face );
   *     if ( error ) { ... }
   *
   *     num_faces = face->num_faces;
   *     FT_Done_Face( face );
   *
   *     for ( i = 0; i < num_faces; i++ )
   *     {
   *       ...
   *       error = FT_Open_Face( library, args, i, &face );
   *       ...
   *       FT_Done_Face( face );
   *       ...
   *     }
   *   ```
   *
   *   To loop over all valid values for `face_index`, use something similar
   *   to the following snippet, again without error handling.  The code
   *   accesses all faces immediately (thus only a single call of
   *   `FT_Open_Face` within the do-loop), with and without named instances.
   *
   *   ```
   *     ...
   *     FT_Face  face;
   *
   *     FT_Long  num_faces     = 0;
   *     FT_Long  num_instances = 0;
   *
   *     FT_Long  face_idx     = 0;
   *     FT_Long  instance_idx = 0;
   *
   *
   *     do
   *     {
   *       FT_Long  id = ( instance_idx << 16 ) + face_idx;
   *
   *
   *       error = FT_Open_Face( library, args, id, &face );
   *       if ( error ) { ... }
   *
   *       num_faces     = face->num_faces;
   *       num_instances = face->style_flags >> 16;
   *
   *       ...
   *
   *       FT_Done_Face( face );
   *
   *       if ( instance_idx < num_instances )
   *         instance_idx++;
   *       else
   *       {
   *         face_idx++;
   *         instance_idx = 0;
   *       }
   *
   *     } while ( face_idx < num_faces )
   *   ```
   */
  FT_EXPORT( FT_Error )
  FT_Open_Face( FT_Library           library,
                const FT_Open_Args*  args,
                FT_Long              face_index,
                FT_Face             *aface );


  /**************************************************************************
   *
   * @function:
   *   FT_Attach_File
   *
   * @description:
   *   Call @FT_Attach_Stream to attach a file.
   *
   * @inout:
   *   face ::
   *     The target face object.
   *
   * @input:
   *   filepathname ::
   *     The pathname.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_Attach_File( FT_Face      face,
                  const char*  filepathname );


  /**************************************************************************
   *
   * @function:
   *   FT_Attach_Stream
   *
   * @description:
   *   'Attach' data to a face object.  Normally, this is used to read
   *   additional information for the face object.  For example, you can
   *   attach an AFM file that comes with a Type~1 font to get the kerning
   *   values and other metrics.
   *
   * @inout:
   *   face ::
   *     The target face object.
   *
   * @input:
   *   parameters ::
   *     A pointer to @FT_Open_Args that must be filled by the caller.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The meaning of the 'attach' (i.e., what really happens when the new
   *   file is read) is not fixed by FreeType itself.  It really depends on
   *   the font format (and thus the font driver).
   *
   *   Client applications are expected to know what they are doing when
   *   invoking this function.  Most drivers simply do not implement file or
   *   stream attachments.
   */
  FT_EXPORT( FT_Error )
  FT_Attach_Stream( FT_Face              face,
                    const FT_Open_Args*  parameters );


  /**************************************************************************
   *
   * @function:
   *   FT_Reference_Face
   *
   * @description:
   *   A counter gets initialized to~1 at the time an @FT_Face structure is
   *   created.  This function increments the counter.  @FT_Done_Face then
   *   only destroys a face if the counter is~1, otherwise it simply
   *   decrements the counter.
   *
   *   This function helps in managing life-cycles of structures that
   *   reference @FT_Face objects.
   *
   * @input:
   *   face ::
   *     A handle to a target face object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @since:
   *   2.4.2
   *
   */
  FT_EXPORT( FT_Error )
  FT_Reference_Face( FT_Face  face );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_Face
   *
   * @description:
   *   Discard a given face object, as well as all of its child slots and
   *   sizes.
   *
   * @input:
   *   face ::
   *     A handle to a target face object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   See the discussion of reference counters in the description of
   *   @FT_Reference_Face.
   */
  FT_EXPORT( FT_Error )
  FT_Done_Face( FT_Face  face );


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Select_Size
   *
   * @description:
   *   Select a bitmap strike.  To be more precise, this function sets the
   *   scaling factors of the active @FT_Size object in a face so that
   *   bitmaps from this particular strike are taken by @FT_Load_Glyph and
   *   friends.
   *
   * @inout:
   *   face ::
   *     A handle to a target face object.
   *
   * @input:
   *   strike_index ::
   *     The index of the bitmap strike in the `available_sizes` field of
   *     @FT_FaceRec structure.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   For bitmaps embedded in outline fonts it is common that only a subset
   *   of the available glyphs at a given ppem value is available.  FreeType
   *   silently uses outlines if there is no bitmap for a given glyph index.
   *
   *   For GX and OpenType variation fonts, a bitmap strike makes sense only
   *   if the default instance is active (that is, no glyph variation takes
   *   place); otherwise, FreeType simply ignores bitmap strikes.  The same
   *   is true for all named instances that are different from the default
   *   instance.
   *
   *   Don't use this function if you are using the FreeType cache API.
   */
  FT_EXPORT( FT_Error )
  FT_Select_Size( FT_Face  face,
                  FT_Int   strike_index );


  /**************************************************************************
   *
   * @enum:
   *   FT_Size_Request_Type
   *
   * @description:
   *   An enumeration type that lists the supported size request types, i.e.,
   *   what input size (in font units) maps to the requested output size (in
   *   pixels, as computed from the arguments of @FT_Size_Request).
   *
   * @values:
   *   FT_SIZE_REQUEST_TYPE_NOMINAL ::
   *     The nominal size.  The `units_per_EM` field of @FT_FaceRec is used
   *     to determine both scaling values.
   *
   *     This is the standard scaling found in most applications.  In
   *     particular, use this size request type for TrueType fonts if they
   *     provide optical scaling or something similar.  Note, however, that
   *     `units_per_EM` is a rather abstract value which bears no relation to
   *     the actual size of the glyphs in a font.
   *
   *   FT_SIZE_REQUEST_TYPE_REAL_DIM ::
   *     The real dimension.  The sum of the `ascender` and (minus of) the
   *     `descender` fields of @FT_FaceRec is used to determine both scaling
   *     values.
   *
   *   FT_SIZE_REQUEST_TYPE_BBOX ::
   *     The font bounding box.  The width and height of the `bbox` field of
   *     @FT_FaceRec are used to determine the horizontal and vertical
   *     scaling value, respectively.
   *
   *   FT_SIZE_REQUEST_TYPE_CELL ::
   *     The `max_advance_width` field of @FT_FaceRec is used to determine
   *     the horizontal scaling value; the vertical scaling value is
   *     determined the same way as @FT_SIZE_REQUEST_TYPE_REAL_DIM does.
   *     Finally, both scaling values are set to the smaller one.  This type
   *     is useful if you want to specify the font size for, say, a window of
   *     a given dimension and 80x24 cells.
   *
   *   FT_SIZE_REQUEST_TYPE_SCALES ::
   *     Specify the scaling values directly.
   *
   * @note:
   *   The above descriptions only apply to scalable formats.  For bitmap
   *   formats, the behaviour is up to the driver.
   *
   *   See the note section of @FT_Size_Metrics if you wonder how size
   *   requesting relates to scaling values.
   */
  typedef enum  FT_Size_Request_Type_
  {
    FT_SIZE_REQUEST_TYPE_NOMINAL,
    FT_SIZE_REQUEST_TYPE_REAL_DIM,
    FT_SIZE_REQUEST_TYPE_BBOX,
    FT_SIZE_REQUEST_TYPE_CELL,
    FT_SIZE_REQUEST_TYPE_SCALES,

    FT_SIZE_REQUEST_TYPE_MAX

  } FT_Size_Request_Type;


  /**************************************************************************
   *
   * @struct:
   *   FT_Size_RequestRec
   *
   * @description:
   *   A structure to model a size request.
   *
   * @fields:
   *   type ::
   *     See @FT_Size_Request_Type.
   *
   *   width ::
   *     The desired width, given as a 26.6 fractional point value (with 72pt
   *     = 1in).
   *
   *   height ::
   *     The desired height, given as a 26.6 fractional point value (with
   *     72pt = 1in).
   *
   *   horiResolution ::
   *     The horizontal resolution (dpi, i.e., pixels per inch).  If set to
   *     zero, `width` is treated as a 26.6 fractional **pixel** value, which
   *     gets internally rounded to an integer.
   *
   *   vertResolution ::
   *     The vertical resolution (dpi, i.e., pixels per inch).  If set to
   *     zero, `height` is treated as a 26.6 fractional **pixel** value,
   *     which gets internally rounded to an integer.
   *
   * @note:
   *   If `width` is zero, the horizontal scaling value is set equal to the
   *   vertical scaling value, and vice versa.
   *
   *   If `type` is `FT_SIZE_REQUEST_TYPE_SCALES`, `width` and `height` are
   *   interpreted directly as 16.16 fractional scaling values, without any
   *   further modification, and both `horiResolution` and `vertResolution`
   *   are ignored.
   */
  typedef struct  FT_Size_RequestRec_
  {
    FT_Size_Request_Type  type;
    FT_Long               width;
    FT_Long               height;
    FT_UInt               horiResolution;
    FT_UInt               vertResolution;

  } FT_Size_RequestRec;


  /**************************************************************************
   *
   * @struct:
   *   FT_Size_Request
   *
   * @description:
   *   A handle to a size request structure.
   */
  typedef struct FT_Size_RequestRec_  *FT_Size_Request;


  /**************************************************************************
   *
   * @function:
   *   FT_Request_Size
   *
   * @description:
   *   Resize the scale of the active @FT_Size object in a face.
   *
   * @inout:
   *   face ::
   *     A handle to a target face object.
   *
   * @input:
   *   req ::
   *     A pointer to a @FT_Size_RequestRec.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Although drivers may select the bitmap strike matching the request,
   *   you should not rely on this if you intend to select a particular
   *   bitmap strike.  Use @FT_Select_Size instead in that case.
   *
   *   The relation between the requested size and the resulting glyph size
   *   is dependent entirely on how the size is defined in the source face.
   *   The font designer chooses the final size of each glyph relative to
   *   this size.  For more information refer to
   *   'https://www.freetype.org/freetype2/docs/glyphs/glyphs-2.html'.
   *
   *   Contrary to @FT_Set_Char_Size, this function doesn't have special code
   *   to normalize zero-valued widths, heights, or resolutions, which are
   *   treated as @FT_LOAD_NO_SCALE.
   *
   *   Don't use this function if you are using the FreeType cache API.
   */
  FT_EXPORT( FT_Error )
  FT_Request_Size( FT_Face          face,
                   FT_Size_Request  req );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Char_Size
   *
   * @description:
   *   Call @FT_Request_Size to request the nominal size (in points).
   *
   * @inout:
   *   face ::
   *     A handle to a target face object.
   *
   * @input:
   *   char_width ::
   *     The nominal width, in 26.6 fractional points.
   *
   *   char_height ::
   *     The nominal height, in 26.6 fractional points.
   *
   *   horz_resolution ::
   *     The horizontal resolution in dpi.
   *
   *   vert_resolution ::
   *     The vertical resolution in dpi.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   While this function allows fractional points as input values, the
   *   resulting ppem value for the given resolution is always rounded to the
   *   nearest integer.
   *
   *   If either the character width or height is zero, it is set equal to
   *   the other value.
   *
   *   If either the horizontal or vertical resolution is zero, it is set
   *   equal to the other value.
   *
   *   A character width or height smaller than 1pt is set to 1pt; if both
   *   resolution values are zero, they are set to 72dpi.
   *
   *   Don't use this function if you are using the FreeType cache API.
   */
  FT_EXPORT( FT_Error )
  FT_Set_Char_Size( FT_Face     face,
                    FT_F26Dot6  char_width,
                    FT_F26Dot6  char_height,
                    FT_UInt     horz_resolution,
                    FT_UInt     vert_resolution );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Pixel_Sizes
   *
   * @description:
   *   Call @FT_Request_Size to request the nominal size (in pixels).
   *
   * @inout:
   *   face ::
   *     A handle to the target face object.
   *
   * @input:
   *   pixel_width ::
   *     The nominal width, in pixels.
   *
   *   pixel_height ::
   *     The nominal height, in pixels.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   You should not rely on the resulting glyphs matching or being
   *   constrained to this pixel size.  Refer to @FT_Request_Size to
   *   understand how requested sizes relate to actual sizes.
   *
   *   Don't use this function if you are using the FreeType cache API.
   */
  FT_EXPORT( FT_Error )
  FT_Set_Pixel_Sizes( FT_Face  face,
                      FT_UInt  pixel_width,
                      FT_UInt  pixel_height );


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Load_Glyph
   *
   * @description:
   *   Load a glyph into the glyph slot of a face object.
   *
   * @inout:
   *   face ::
   *     A handle to the target face object where the glyph is loaded.
   *
   * @input:
   *   glyph_index ::
   *     The index of the glyph in the font file.  For CID-keyed fonts
   *     (either in PS or in CFF format) this argument specifies the CID
   *     value.
   *
   *   load_flags ::
   *     A flag indicating what to load for this glyph.  The @FT_LOAD_XXX
   *     flags can be used to control the glyph loading process (e.g.,
   *     whether the outline should be scaled, whether to load bitmaps or
   *     not, whether to hint the outline, etc).
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   For proper scaling and hinting, the active @FT_Size object owned by
   *   the face has to be meaningfully initialized by calling
   *   @FT_Set_Char_Size before this function, for example.  The loaded
   *   glyph may be transformed.  See @FT_Set_Transform for the details.
   *
   *   For subsetted CID-keyed fonts, `FT_Err_Invalid_Argument` is returned
   *   for invalid CID values (that is, for CID values that don't have a
   *   corresponding glyph in the font).  See the discussion of the
   *   @FT_FACE_FLAG_CID_KEYED flag for more details.
   *
   *   If you receive `FT_Err_Glyph_Too_Big`, try getting the glyph outline
   *   at EM size, then scale it manually and fill it as a graphics
   *   operation.
   */
  FT_EXPORT( FT_Error )
  FT_Load_Glyph( FT_Face   face,
                 FT_UInt   glyph_index,
                 FT_Int32  load_flags );


  /**************************************************************************
   *
   * @section:
   *   character_mapping
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Load_Char
   *
   * @description:
   *   Load a glyph into the glyph slot of a face object, accessed by its
   *   character code.
   *
   * @inout:
   *   face ::
   *     A handle to a target face object where the glyph is loaded.
   *
   * @input:
   *   char_code ::
   *     The glyph's character code, according to the current charmap used in
   *     the face.
   *
   *   load_flags ::
   *     A flag indicating what to load for this glyph.  The @FT_LOAD_XXX
   *     constants can be used to control the glyph loading process (e.g.,
   *     whether the outline should be scaled, whether to load bitmaps or
   *     not, whether to hint the outline, etc).
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function simply calls @FT_Get_Char_Index and @FT_Load_Glyph.
   *
   *   Many fonts contain glyphs that can't be loaded by this function since
   *   its glyph indices are not listed in any of the font's charmaps.
   *
   *   If no active cmap is set up (i.e., `face->charmap` is zero), the call
   *   to @FT_Get_Char_Index is omitted, and the function behaves identically
   *   to @FT_Load_Glyph.
   */
  FT_EXPORT( FT_Error )
  FT_Load_Char( FT_Face   face,
                FT_ULong  char_code,
                FT_Int32  load_flags );


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @enum:
   *   FT_LOAD_XXX
   *
   * @description:
   *   A list of bit field constants for @FT_Load_Glyph to indicate what kind
   *   of operations to perform during glyph loading.
   *
   * @values:
   *   FT_LOAD_DEFAULT ::
   *     Corresponding to~0, this value is used as the default glyph load
   *     operation.  In this case, the following happens:
   *
   *     1. FreeType looks for a bitmap for the glyph corresponding to the
   *     face's current size.  If one is found, the function returns.  The
   *     bitmap data can be accessed from the glyph slot (see note below).
   *
   *     2. If no embedded bitmap is searched for or found, FreeType looks
   *     for a scalable outline.  If one is found, it is loaded from the font
   *     file, scaled to device pixels, then 'hinted' to the pixel grid in
   *     order to optimize it.  The outline data can be accessed from the
   *     glyph slot (see note below).
   *
   *     Note that by default the glyph loader doesn't render outlines into
   *     bitmaps.  The following flags are used to modify this default
   *     behaviour to more specific and useful cases.
   *
   *   FT_LOAD_NO_SCALE ::
   *     Don't scale the loaded outline glyph but keep it in font units.
   *     This flag is also assumed if @FT_Size owned by the face was not
   *     properly initialized.
   *
   *     This flag implies @FT_LOAD_NO_HINTING and @FT_LOAD_NO_BITMAP, and
   *     unsets @FT_LOAD_RENDER.
   *
   *     If the font is 'tricky' (see @FT_FACE_FLAG_TRICKY for more), using
   *     `FT_LOAD_NO_SCALE` usually yields meaningless outlines because the
   *     subglyphs must be scaled and positioned with hinting instructions.
   *     This can be solved by loading the font without `FT_LOAD_NO_SCALE`
   *     and setting the character size to `font->units_per_EM`.
   *
   *   FT_LOAD_NO_HINTING ::
   *     Disable hinting.  This generally generates 'blurrier' bitmap glyphs
   *     when the glyphs are rendered in any of the anti-aliased modes.  See
   *     also the note below.
   *
   *     This flag is implied by @FT_LOAD_NO_SCALE.
   *
   *   FT_LOAD_RENDER ::
   *     Call @FT_Render_Glyph after the glyph is loaded.  By default, the
   *     glyph is rendered in @FT_RENDER_MODE_NORMAL mode.  This can be
   *     overridden by @FT_LOAD_TARGET_XXX or @FT_LOAD_MONOCHROME.
   *
   *     This flag is unset by @FT_LOAD_NO_SCALE.
   *
   *   FT_LOAD_NO_BITMAP ::
   *     Ignore bitmap strikes when loading.  Bitmap-only fonts ignore this
   *     flag.
   *
   *     @FT_LOAD_NO_SCALE always sets this flag.
   *
   *   FT_LOAD_SBITS_ONLY ::
   *     [Since 2.12] This is the opposite of @FT_LOAD_NO_BITMAP, more or
   *     less: @FT_Load_Glyph returns `FT_Err_Invalid_Argument` if the face
   *     contains a bitmap strike for the given size (or the strike selected
   *     by @FT_Select_Size) but there is no glyph in the strike.
   *
   *     Note that this load flag was part of FreeType since version 2.0.6
   *     but previously tagged as internal.
   *
   *   FT_LOAD_VERTICAL_LAYOUT ::
   *     Load the glyph for vertical text layout.  In particular, the
   *     `advance` value in the @FT_GlyphSlotRec structure is set to the
   *     `vertAdvance` value of the `metrics` field.
   *
   *     In case @FT_HAS_VERTICAL doesn't return true, you shouldn't use this
   *     flag currently.  Reason is that in this case vertical metrics get
   *     synthesized, and those values are not always consistent across
   *     various font formats.
   *
   *   FT_LOAD_FORCE_AUTOHINT ::
   *     Prefer the auto-hinter over the font's native hinter.  See also the
   *     note below.
   *
   *   FT_LOAD_PEDANTIC ::
   *     Make the font driver perform pedantic verifications during glyph
   *     loading and hinting.  This is mostly used to detect broken glyphs in
   *     fonts.  By default, FreeType tries to handle broken fonts also.
   *
   *     In particular, errors from the TrueType bytecode engine are not
   *     passed to the application if this flag is not set; this might result
   *     in partially hinted or distorted glyphs in case a glyph's bytecode
   *     is buggy.
   *
   *   FT_LOAD_NO_RECURSE ::
   *     Don't load composite glyphs recursively.  Instead, the font driver
   *     fills the `num_subglyph` and `subglyphs` values of the glyph slot;
   *     it also sets `glyph->format` to @FT_GLYPH_FORMAT_COMPOSITE.  The
   *     description of subglyphs can then be accessed with
   *     @FT_Get_SubGlyph_Info.
   *
   *     Don't use this flag for retrieving metrics information since some
   *     font drivers only return rudimentary data.
   *
   *     This flag implies @FT_LOAD_NO_SCALE and @FT_LOAD_IGNORE_TRANSFORM.
   *
   *   FT_LOAD_IGNORE_TRANSFORM ::
   *     Ignore the transform matrix set by @FT_Set_Transform.
   *
   *   FT_LOAD_MONOCHROME ::
   *     This flag is used with @FT_LOAD_RENDER to indicate that you want to
   *     render an outline glyph to a 1-bit monochrome bitmap glyph, with
   *     8~pixels packed into each byte of the bitmap data.
   *
   *     Note that this has no effect on the hinting algorithm used.  You
   *     should rather use @FT_LOAD_TARGET_MONO so that the
   *     monochrome-optimized hinting algorithm is used.
   *
   *   FT_LOAD_LINEAR_DESIGN ::
   *     Keep `linearHoriAdvance` and `linearVertAdvance` fields of
   *     @FT_GlyphSlotRec in font units.  See @FT_GlyphSlotRec for details.
   *
   *   FT_LOAD_NO_AUTOHINT ::
   *     Disable the auto-hinter.  See also the note below.
   *
   *   FT_LOAD_COLOR ::
   *     Load colored glyphs.  FreeType searches in the following order;
   *     there are slight differences depending on the font format.
   *
   *     [Since 2.5] Load embedded color bitmap images (provided
   *     @FT_LOAD_NO_BITMAP is not set).  The resulting color bitmaps, if
   *     available, have the @FT_PIXEL_MODE_BGRA format, with pre-multiplied
   *     color channels.  If the flag is not set and color bitmaps are found,
   *     they are converted to 256-level gray bitmaps, using the
   *     @FT_PIXEL_MODE_GRAY format.
   *
   *     [Since 2.12] If the glyph index maps to an entry in the face's
   *     'SVG~' table, load the associated SVG document from this table and
   *     set the `format` field of @FT_GlyphSlotRec to @FT_GLYPH_FORMAT_SVG
   *     ([since 2.13.1] provided @FT_LOAD_NO_SVG is not set).  Note that
   *     FreeType itself can't render SVG documents; however, the library
   *     provides hooks to seamlessly integrate an external renderer.  See
   *     sections @ot_svg_driver and @svg_fonts for more.
   *
   *     [Since 2.10, experimental] If the glyph index maps to an entry in
   *     the face's 'COLR' table with a 'CPAL' palette table (as defined in
   *     the OpenType specification), make @FT_Render_Glyph provide a default
   *     blending of the color glyph layers associated with the glyph index,
   *     using the same bitmap format as embedded color bitmap images.  This
   *     is mainly for convenience and works only for glyphs in 'COLR' v0
   *     tables (or glyphs in 'COLR' v1 tables that exclusively use v0
   *     features).  For full control of color layers use
   *     @FT_Get_Color_Glyph_Layer and FreeType's color functions like
   *     @FT_Palette_Select instead of setting @FT_LOAD_COLOR for rendering
   *     so that the client application can handle blending by itself.
   *
   *   FT_LOAD_NO_SVG ::
   *     [Since 2.13.1] Ignore SVG glyph data when loading.
   *
   *   FT_LOAD_COMPUTE_METRICS ::
   *     [Since 2.6.1] Compute glyph metrics from the glyph data, without the
   *     use of bundled metrics tables (for example, the 'hdmx' table in
   *     TrueType fonts).  This flag is mainly used by font validating or
   *     font editing applications, which need to ignore, verify, or edit
   *     those tables.
   *
   *     Currently, this flag is only implemented for TrueType fonts.
   *
   *   FT_LOAD_BITMAP_METRICS_ONLY ::
   *     [Since 2.7.1] Request loading of the metrics and bitmap image
   *     information of a (possibly embedded) bitmap glyph without allocating
   *     or copying the bitmap image data itself.  No effect if the target
   *     glyph is not a bitmap image.
   *
   *     This flag unsets @FT_LOAD_RENDER.
   *
   *   FT_LOAD_CROP_BITMAP ::
   *     Ignored.  Deprecated.
   *
   *   FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH ::
   *     Ignored.  Deprecated.
   *
   * @note:
   *   By default, hinting is enabled and the font's native hinter (see
   *   @FT_FACE_FLAG_HINTER) is preferred over the auto-hinter.  You can
   *   disable hinting by setting @FT_LOAD_NO_HINTING or change the
   *   precedence by setting @FT_LOAD_FORCE_AUTOHINT.  You can also set
   *   @FT_LOAD_NO_AUTOHINT in case you don't want the auto-hinter to be used
   *   at all.
   *
   *   See the description of @FT_FACE_FLAG_TRICKY for a special exception
   *   (affecting only a handful of Asian fonts).
   *
   *   Besides deciding which hinter to use, you can also decide which
   *   hinting algorithm to use.  See @FT_LOAD_TARGET_XXX for details.
   *
   *   Note that the auto-hinter needs a valid Unicode cmap (either a native
   *   one or synthesized by FreeType) for producing correct results.  If a
   *   font provides an incorrect mapping (for example, assigning the
   *   character code U+005A, LATIN CAPITAL LETTER~Z, to a glyph depicting a
   *   mathematical integral sign), the auto-hinter might produce useless
   *   results.
   *
   */
#define FT_LOAD_DEFAULT                      0x0
#define FT_LOAD_NO_SCALE                     ( 1L << 0  )
#define FT_LOAD_NO_HINTING                   ( 1L << 1  )
#define FT_LOAD_RENDER                       ( 1L << 2  )
#define FT_LOAD_NO_BITMAP                    ( 1L << 3  )
#define FT_LOAD_VERTICAL_LAYOUT              ( 1L << 4  )
#define FT_LOAD_FORCE_AUTOHINT               ( 1L << 5  )
#define FT_LOAD_CROP_BITMAP                  ( 1L << 6  )
#define FT_LOAD_PEDANTIC                     ( 1L << 7  )
#define FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH  ( 1L << 9  )
#define FT_LOAD_NO_RECURSE                   ( 1L << 10 )
#define FT_LOAD_IGNORE_TRANSFORM             ( 1L << 11 )
#define FT_LOAD_MONOCHROME                   ( 1L << 12 )
#define FT_LOAD_LINEAR_DESIGN                ( 1L << 13 )
#define FT_LOAD_SBITS_ONLY                   ( 1L << 14 )
#define FT_LOAD_NO_AUTOHINT                  ( 1L << 15 )
  /* Bits 16-19 are used by `FT_LOAD_TARGET_` */
#define FT_LOAD_COLOR                        ( 1L << 20 )
#define FT_LOAD_COMPUTE_METRICS              ( 1L << 21 )
#define FT_LOAD_BITMAP_METRICS_ONLY          ( 1L << 22 )
#define FT_LOAD_NO_SVG                       ( 1L << 24 )

  /* */

  /* used internally only by certain font drivers */
#define FT_LOAD_ADVANCE_ONLY                 ( 1L << 8  )
#define FT_LOAD_SVG_ONLY                     ( 1L << 23 )


  /**************************************************************************
   *
   * @enum:
   *   FT_LOAD_TARGET_XXX
   *
   * @description:
   *   A list of values to select a specific hinting algorithm for the
   *   hinter.  You should OR one of these values to your `load_flags` when
   *   calling @FT_Load_Glyph.
   *
   *   Note that a font's native hinters may ignore the hinting algorithm you
   *   have specified (e.g., the TrueType bytecode interpreter).  You can set
   *   @FT_LOAD_FORCE_AUTOHINT to ensure that the auto-hinter is used.
   *
   * @values:
   *   FT_LOAD_TARGET_NORMAL ::
   *     The default hinting algorithm, optimized for standard gray-level
   *     rendering.  For monochrome output, use @FT_LOAD_TARGET_MONO instead.
   *
   *   FT_LOAD_TARGET_LIGHT ::
   *     A lighter hinting algorithm for gray-level modes.  Many generated
   *     glyphs are fuzzier but better resemble their original shape.  This
   *     is achieved by snapping glyphs to the pixel grid only vertically
   *     (Y-axis), as is done by FreeType's new CFF engine or Microsoft's
   *     ClearType font renderer.  This preserves inter-glyph spacing in
   *     horizontal text.  The snapping is done either by the native font
   *     driver, if the driver itself and the font support it, or by the
   *     auto-hinter.
   *
   *     Advance widths are rounded to integer values; however, using the
   *     `lsb_delta` and `rsb_delta` fields of @FT_GlyphSlotRec, it is
   *     possible to get fractional advance widths for subpixel positioning
   *     (which is recommended to use).
   *
   *     If configuration option `AF_CONFIG_OPTION_TT_SIZE_METRICS` is
   *     active, TrueType-like metrics are used to make this mode behave
   *     similarly as in unpatched FreeType versions between 2.4.6 and 2.7.1
   *     (inclusive).
   *
   *   FT_LOAD_TARGET_MONO ::
   *     Strong hinting algorithm that should only be used for monochrome
   *     output.  The result is probably unpleasant if the glyph is rendered
   *     in non-monochrome modes.
   *
   *     Note that for outline fonts only the TrueType font driver has proper
   *     monochrome hinting support, provided the TTFs contain hints for B/W
   *     rendering (which most fonts no longer provide).  If these conditions
   *     are not met it is very likely that you get ugly results at smaller
   *     sizes.
   *
   *   FT_LOAD_TARGET_LCD ::
   *     A variant of @FT_LOAD_TARGET_LIGHT optimized for horizontally
   *     decimated LCD displays.
   *
   *   FT_LOAD_TARGET_LCD_V ::
   *     A variant of @FT_LOAD_TARGET_NORMAL optimized for vertically
   *     decimated LCD displays.
   *
   * @note:
   *   You should use only _one_ of the `FT_LOAD_TARGET_XXX` values in your
   *   `load_flags`.  They can't be ORed.
   *
   *   If @FT_LOAD_RENDER is also set, the glyph is rendered in the
   *   corresponding mode (i.e., the mode that matches the used algorithm
   *   best).  An exception is `FT_LOAD_TARGET_MONO` since it implies
   *   @FT_LOAD_MONOCHROME.
   *
   *   You can use a hinting algorithm that doesn't correspond to the same
   *   rendering mode.  As an example, it is possible to use the 'light'
   *   hinting algorithm and have the results rendered in horizontal LCD
   *   pixel mode, with code like
   *
   *   ```
   *     FT_Load_Glyph( face, glyph_index,
   *                    load_flags | FT_LOAD_TARGET_LIGHT );
   *
   *     FT_Render_Glyph( face->glyph, FT_RENDER_MODE_LCD );
   *   ```
   *
   *   In general, you should stick with one rendering mode.  For example,
   *   switching between @FT_LOAD_TARGET_NORMAL and @FT_LOAD_TARGET_MONO
   *   enforces a lot of recomputation for TrueType fonts, which is slow.
   *   Another reason is caching: Selecting a different mode usually causes
   *   changes in both the outlines and the rasterized bitmaps; it is thus
   *   necessary to empty the cache after a mode switch to avoid false hits.
   *
   */
#define FT_LOAD_TARGET_( x )   ( FT_STATIC_CAST( FT_Int32, (x) & 15 ) << 16 )

#define FT_LOAD_TARGET_NORMAL  FT_LOAD_TARGET_( FT_RENDER_MODE_NORMAL )
#define FT_LOAD_TARGET_LIGHT   FT_LOAD_TARGET_( FT_RENDER_MODE_LIGHT  )
#define FT_LOAD_TARGET_MONO    FT_LOAD_TARGET_( FT_RENDER_MODE_MONO   )
#define FT_LOAD_TARGET_LCD     FT_LOAD_TARGET_( FT_RENDER_MODE_LCD    )
#define FT_LOAD_TARGET_LCD_V   FT_LOAD_TARGET_( FT_RENDER_MODE_LCD_V  )


  /**************************************************************************
   *
   * @macro:
   *   FT_LOAD_TARGET_MODE
   *
   * @description:
   *   Return the @FT_Render_Mode corresponding to a given
   *   @FT_LOAD_TARGET_XXX value.
   *
   */
#define FT_LOAD_TARGET_MODE( x )                               \
          FT_STATIC_CAST( FT_Render_Mode, ( (x) >> 16 ) & 15 )


  /**************************************************************************
   *
   * @section:
   *   sizing_and_scaling
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Set_Transform
   *
   * @description:
   *   Set the transformation that is applied to glyph images when they are
   *   loaded into a glyph slot through @FT_Load_Glyph.
   *
   * @inout:
   *   face ::
   *     A handle to the source face object.
   *
   * @input:
   *   matrix ::
   *     A pointer to the transformation's 2x2 matrix.  Use `NULL` for the
   *     identity matrix.
   *   delta ::
   *     A pointer to the translation vector.  Use `NULL` for the null
   *     vector.
   *
   * @note:
   *   This function is provided as a convenience, but keep in mind that
   *   @FT_Matrix coefficients are only 16.16 fixed-point values, which can
   *   limit the accuracy of the results.  Using floating-point computations
   *   to perform the transform directly in client code instead will always
   *   yield better numbers.
   *
   *   The transformation is only applied to scalable image formats after the
   *   glyph has been loaded.  It means that hinting is unaltered by the
   *   transformation and is performed on the character size given in the
   *   last call to @FT_Set_Char_Size or @FT_Set_Pixel_Sizes.
   *
   *   Note that this also transforms the `face.glyph.advance` field, but
   *   **not** the values in `face.glyph.metrics`.
   */
  FT_EXPORT( void )
  FT_Set_Transform( FT_Face     face,
                    FT_Matrix*  matrix,
                    FT_Vector*  delta );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Transform
   *
   * @description:
   *   Return the transformation that is applied to glyph images when they
   *   are loaded into a glyph slot through @FT_Load_Glyph.  See
   *   @FT_Set_Transform for more details.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   * @output:
   *   matrix ::
   *     A pointer to a transformation's 2x2 matrix.  Set this to NULL if you
   *     are not interested in the value.
   *
   *   delta ::
   *     A pointer to a translation vector.  Set this to NULL if you are not
   *     interested in the value.
   *
   * @since:
   *   2.11
   *
   */
  FT_EXPORT( void )
  FT_Get_Transform( FT_Face     face,
                    FT_Matrix*  matrix,
                    FT_Vector*  delta );


  /**************************************************************************
   *
   * @section:
   *   glyph_retrieval
   *
   */

  /**************************************************************************
   *
   * @enum:
   *   FT_Render_Mode
   *
   * @description:
   *   Render modes supported by FreeType~2.  Each mode corresponds to a
   *   specific type of scanline conversion performed on the outline.
   *
   *   For bitmap fonts and embedded bitmaps the `bitmap->pixel_mode` field
   *   in the @FT_GlyphSlotRec structure gives the format of the returned
   *   bitmap.
   *
   *   All modes except @FT_RENDER_MODE_MONO use 256 levels of opacity,
   *   indicating pixel coverage.  Use linear alpha blending and gamma
   *   correction to correctly render non-monochrome glyph bitmaps onto a
   *   surface; see @FT_Render_Glyph.
   *
   *   The @FT_RENDER_MODE_SDF is a special render mode that uses up to 256
   *   distance values, indicating the signed distance from the grid position
   *   to the nearest outline.
   *
   * @values:
   *   FT_RENDER_MODE_NORMAL ::
   *     Default render mode; it corresponds to 8-bit anti-aliased bitmaps.
   *
   *   FT_RENDER_MODE_LIGHT ::
   *     This is equivalent to @FT_RENDER_MODE_NORMAL.  It is only defined as
   *     a separate value because render modes are also used indirectly to
   *     define hinting algorithm selectors.  See @FT_LOAD_TARGET_XXX for
   *     details.
   *
   *   FT_RENDER_MODE_MONO ::
   *     This mode corresponds to 1-bit bitmaps (with 2~levels of opacity).
   *
   *   FT_RENDER_MODE_LCD ::
   *     This mode corresponds to horizontal RGB and BGR subpixel displays
   *     like LCD screens.  It produces 8-bit bitmaps that are 3~times the
   *     width of the original glyph outline in pixels, and which use the
   *     @FT_PIXEL_MODE_LCD mode.
   *
   *   FT_RENDER_MODE_LCD_V ::
   *     This mode corresponds to vertical RGB and BGR subpixel displays
   *     (like PDA screens, rotated LCD displays, etc.).  It produces 8-bit
   *     bitmaps that are 3~times the height of the original glyph outline in
   *     pixels and use the @FT_PIXEL_MODE_LCD_V mode.
   *
   *   FT_RENDER_MODE_SDF ::
   *     This mode corresponds to 8-bit, single-channel signed distance field
   *     (SDF) bitmaps.  Each pixel in the SDF grid is the value from the
   *     pixel's position to the nearest glyph's outline.  The distances are
   *     calculated from the center of the pixel and are positive if they are
   *     filled by the outline (i.e., inside the outline) and negative
   *     otherwise.  Check the note below on how to convert the output values
   *     to usable data.
   *
   * @note:
   *   The selected render mode only affects vector glyphs of a font.
   *   Embedded bitmaps often have a different pixel mode like
   *   @FT_PIXEL_MODE_MONO.  You can use @FT_Bitmap_Convert to transform them
   *   into 8-bit pixmaps.
   *
   *   For @FT_RENDER_MODE_SDF the output bitmap buffer contains normalized
   *   distances that are packed into unsigned 8-bit values.  To get pixel
   *   values in floating point representation use the following pseudo-C
   *   code for the conversion.
   *
   *   ```
   *   // Load glyph and render using FT_RENDER_MODE_SDF,
   *   // then use the output buffer as follows.
   *
   *   ...
   *   FT_Byte  buffer = glyph->bitmap->buffer;
   *
   *
   *   for pixel in buffer
   *   {
   *     // `sd` is the signed distance and `spread` is the current spread;
   *     // the default spread is 2 and can be changed.
   *
   *     float  sd = (float)pixel - 128.0f;
   *
   *
   *     // Convert to pixel values.
   *     sd = ( sd / 128.0f ) * spread;
   *
   *     // Store `sd` in a buffer or use as required.
   *   }
   *
   *   ```
   *
   *   FreeType has two rasterizers for generating SDF, namely:
   *
   *   1. `sdf` for generating SDF directly from glyph's outline, and
   *
   *   2. `bsdf` for generating SDF from rasterized bitmaps.
   *
   *   Depending on the glyph type (i.e., outline or bitmap), one of the two
   *   rasterizers is chosen at runtime and used for generating SDFs.  To
   *   force the use of `bsdf` you should render the glyph with any of the
   *   FreeType's other rendering modes (e.g., `FT_RENDER_MODE_NORMAL`) and
   *   then re-render with `FT_RENDER_MODE_SDF`.
   *
   *   There are some issues with stability and possible failures of the SDF
   *   renderers (specifically `sdf`).
   *
   *   1. The `sdf` rasterizer is sensitive to really small features (e.g.,
   *      sharp turns that are less than 1~pixel) and imperfections in the
   *      glyph's outline, causing artifacts in the final output.
   *
   *   2. The `sdf` rasterizer has limited support for handling intersecting
   *      contours and *cannot* handle self-intersecting contours whatsoever.
   *      Self-intersection happens when a single connected contour
   *      intersects itself at some point; having these in your font
   *      definitely poses a problem to the rasterizer and cause artifacts,
   *      too.
   *
   *   3. Generating SDF for really small glyphs may result in undesirable
   *      output; the pixel grid (which stores distance information) becomes
   *      too coarse.
   *
   *   4. Since the output buffer is normalized, precision at smaller spreads
   *      is greater than precision at larger spread values because the
   *      output range of [0..255] gets mapped to a smaller SDF range.  A
   *      spread of~2 should be sufficient in most cases.
   *
   *   Points (1) and (2) can be avoided by using the `bsdf` rasterizer,
   *   which is more stable than the `sdf` rasterizer in general.
   *
   */
  typedef enum  FT_Render_Mode_
  {
    FT_RENDER_MODE_NORMAL = 0,
    FT_RENDER_MODE_LIGHT,
    FT_RENDER_MODE_MONO,
    FT_RENDER_MODE_LCD,
    FT_RENDER_MODE_LCD_V,
    FT_RENDER_MODE_SDF,

    FT_RENDER_MODE_MAX

  } FT_Render_Mode;


  /* these constants are deprecated; use the corresponding */
  /* `FT_Render_Mode` values instead                       */
#define ft_render_mode_normal  FT_RENDER_MODE_NORMAL
#define ft_render_mode_mono    FT_RENDER_MODE_MONO


  /**************************************************************************
   *
   * @function:
   *   FT_Render_Glyph
   *
   * @description:
   *   Convert a given glyph image to a bitmap.  It does so by inspecting the
   *   glyph image format, finding the relevant renderer, and invoking it.
   *
   * @inout:
   *   slot ::
   *     A handle to the glyph slot containing the image to convert.
   *
   * @input:
   *   render_mode ::
   *     The render mode used to render the glyph image into a bitmap.  See
   *     @FT_Render_Mode for a list of possible values.
   *
   *     If @FT_RENDER_MODE_NORMAL is used, a previous call of @FT_Load_Glyph
   *     with flag @FT_LOAD_COLOR makes `FT_Render_Glyph` provide a default
   *     blending of colored glyph layers associated with the current glyph
   *     slot (provided the font contains such layers) instead of rendering
   *     the glyph slot's outline.  This is an experimental feature; see
   *     @FT_LOAD_COLOR for more information.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   When FreeType outputs a bitmap of a glyph, it really outputs an alpha
   *   coverage map.  If a pixel is completely covered by a filled-in
   *   outline, the bitmap contains 0xFF at that pixel, meaning that
   *   0xFF/0xFF fraction of that pixel is covered, meaning the pixel is 100%
   *   black (or 0% bright).  If a pixel is only 50% covered (value 0x80),
   *   the pixel is made 50% black (50% bright or a middle shade of grey).
   *   0% covered means 0% black (100% bright or white).
   *
   *   On high-DPI screens like on smartphones and tablets, the pixels are so
   *   small that their chance of being completely covered and therefore
   *   completely black are fairly good.  On the low-DPI screens, however,
   *   the situation is different.  The pixels are too large for most of the
   *   details of a glyph and shades of gray are the norm rather than the
   *   exception.
   *
   *   This is relevant because all our screens have a second problem: they
   *   are not linear.  1~+~1 is not~2.  Twice the value does not result in
   *   twice the brightness.  When a pixel is only 50% covered, the coverage
   *   map says 50% black, and this translates to a pixel value of 128 when
   *   you use 8~bits per channel (0-255).  However, this does not translate
   *   to 50% brightness for that pixel on our sRGB and gamma~2.2 screens.
   *   Due to their non-linearity, they dwell longer in the darks and only a
   *   pixel value of about 186 results in 50% brightness -- 128 ends up too
   *   dark on both bright and dark backgrounds.  The net result is that dark
   *   text looks burnt-out, pixely and blotchy on bright background, bright
   *   text too frail on dark backgrounds, and colored text on colored
   *   background (for example, red on green) seems to have dark halos or
   *   'dirt' around it.  The situation is especially ugly for diagonal stems
   *   like in 'w' glyph shapes where the quality of FreeType's anti-aliasing
   *   depends on the correct display of grays.  On high-DPI screens where
   *   smaller, fully black pixels reign supreme, this doesn't matter, but on
   *   our low-DPI screens with all the gray shades, it does.  0% and 100%
   *   brightness are the same things in linear and non-linear space, just
   *   all the shades in-between aren't.
   *
   *   The blending function for placing text over a background is
   *
   *   ```
   *     dst = alpha * src + (1 - alpha) * dst    ,
   *   ```
   *
   *   which is known as the OVER operator.
   *
   *   To correctly composite an anti-aliased pixel of a glyph onto a
   *   surface,
   *
   *   1. take the foreground and background colors (e.g., in sRGB space)
   *      and apply gamma to get them in a linear space,
   *
   *   2. use OVER to blend the two linear colors using the glyph pixel
   *      as the alpha value (remember, the glyph bitmap is an alpha coverage
   *      bitmap), and
   *
   *   3. apply inverse gamma to the blended pixel and write it back to
   *      the image.
   *
   *   Internal testing at Adobe found that a target inverse gamma of~1.8 for
   *   step~3 gives good results across a wide range of displays with an sRGB
   *   gamma curve or a similar one.
   *
   *   This process can cost performance.  There is an approximation that
   *   does not need to know about the background color; see
   *   https://bel.fi/alankila/lcd/ and
   *   https://bel.fi/alankila/lcd/alpcor.html for details.
   *
   *   **ATTENTION**: Linear blending is even more important when dealing
   *   with subpixel-rendered glyphs to prevent color-fringing!  A
   *   subpixel-rendered glyph must first be filtered with a filter that
   *   gives equal weight to the three color primaries and does not exceed a
   *   sum of 0x100, see section @lcd_rendering.  Then the only difference to
   *   gray linear blending is that subpixel-rendered linear blending is done
   *   3~times per pixel: red foreground subpixel to red background subpixel
   *   and so on for green and blue.
   */
  FT_EXPORT( FT_Error )
  FT_Render_Glyph( FT_GlyphSlot    slot,
                   FT_Render_Mode  render_mode );


  /**************************************************************************
   *
   * @enum:
   *   FT_Kerning_Mode
   *
   * @description:
   *   An enumeration to specify the format of kerning values returned by
   *   @FT_Get_Kerning.
   *
   * @values:
   *   FT_KERNING_DEFAULT ::
   *     Return grid-fitted kerning distances in 26.6 fractional pixels.
   *
   *   FT_KERNING_UNFITTED ::
   *     Return un-grid-fitted kerning distances in 26.6 fractional pixels.
   *
   *   FT_KERNING_UNSCALED ::
   *     Return the kerning vector in original font units.
   *
   * @note:
   *   `FT_KERNING_DEFAULT` returns full pixel values; it also makes FreeType
   *   heuristically scale down kerning distances at small ppem values so
   *   that they don't become too big.
   *
   *   Both `FT_KERNING_DEFAULT` and `FT_KERNING_UNFITTED` use the current
   *   horizontal scaling factor (as set e.g. with @FT_Set_Char_Size) to
   *   convert font units to pixels.
   */
  typedef enum  FT_Kerning_Mode_
  {
    FT_KERNING_DEFAULT = 0,
    FT_KERNING_UNFITTED,
    FT_KERNING_UNSCALED

  } FT_Kerning_Mode;


  /* these constants are deprecated; use the corresponding */
  /* `FT_Kerning_Mode` values instead                      */
#define ft_kerning_default   FT_KERNING_DEFAULT
#define ft_kerning_unfitted  FT_KERNING_UNFITTED
#define ft_kerning_unscaled  FT_KERNING_UNSCALED


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Kerning
   *
   * @description:
   *   Return the kerning vector between two glyphs of the same face.
   *
   * @input:
   *   face ::
   *     A handle to a source face object.
   *
   *   left_glyph ::
   *     The index of the left glyph in the kern pair.
   *
   *   right_glyph ::
   *     The index of the right glyph in the kern pair.
   *
   *   kern_mode ::
   *     See @FT_Kerning_Mode for more information.  Determines the scale and
   *     dimension of the returned kerning vector.
   *
   * @output:
   *   akerning ::
   *     The kerning vector.  This is either in font units, fractional pixels
   *     (26.6 format), or pixels for scalable formats, and in pixels for
   *     fixed-sizes formats.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Only horizontal layouts (left-to-right & right-to-left) are supported
   *   by this method.  Other layouts, or more sophisticated kernings, are
   *   out of the scope of this API function -- they can be implemented
   *   through format-specific interfaces.
   *
   *   Kerning for OpenType fonts implemented in a 'GPOS' table is not
   *   supported; use @FT_HAS_KERNING to find out whether a font has data
   *   that can be extracted with `FT_Get_Kerning`.
   */
  FT_EXPORT( FT_Error )
  FT_Get_Kerning( FT_Face     face,
                  FT_UInt     left_glyph,
                  FT_UInt     right_glyph,
                  FT_UInt     kern_mode,
                  FT_Vector  *akerning );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Track_Kerning
   *
   * @description:
   *   Return the track kerning for a given face object at a given size.
   *
   * @input:
   *   face ::
   *     A handle to a source face object.
   *
   *   point_size ::
   *     The point size in 16.16 fractional points.
   *
   *   degree ::
   *     The degree of tightness.  Increasingly negative values represent
   *     tighter track kerning, while increasingly positive values represent
   *     looser track kerning.  Value zero means no track kerning.
   *
   * @output:
   *   akerning ::
   *     The kerning in 16.16 fractional points, to be uniformly applied
   *     between all glyphs.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Currently, only the Type~1 font driver supports track kerning, using
   *   data from AFM files (if attached with @FT_Attach_File or
   *   @FT_Attach_Stream).
   *
   *   Only very few AFM files come with track kerning data; please refer to
   *   Adobe's AFM specification for more details.
   */
  FT_EXPORT( FT_Error )
  FT_Get_Track_Kerning( FT_Face    face,
                        FT_Fixed   point_size,
                        FT_Int     degree,
                        FT_Fixed*  akerning );


  /**************************************************************************
   *
   * @section:
   *   character_mapping
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Select_Charmap
   *
   * @description:
   *   Select a given charmap by its encoding tag (as listed in
   *   `freetype.h`).
   *
   * @inout:
   *   face ::
   *     A handle to the source face object.
   *
   * @input:
   *   encoding ::
   *     A handle to the selected encoding.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function returns an error if no charmap in the face corresponds
   *   to the encoding queried here.
   *
   *   Because many fonts contain more than a single cmap for Unicode
   *   encoding, this function has some special code to select the one that
   *   covers Unicode best ('best' in the sense that a UCS-4 cmap is
   *   preferred to a UCS-2 cmap).  It is thus preferable to @FT_Set_Charmap
   *   in this case.
   */
  FT_EXPORT( FT_Error )
  FT_Select_Charmap( FT_Face      face,
                     FT_Encoding  encoding );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Charmap
   *
   * @description:
   *   Select a given charmap for character code to glyph index mapping.
   *
   * @inout:
   *   face ::
   *     A handle to the source face object.
   *
   * @input:
   *   charmap ::
   *     A handle to the selected charmap.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function returns an error if the charmap is not part of the face
   *   (i.e., if it is not listed in the `face->charmaps` table).
   *
   *   It also fails if an OpenType type~14 charmap is selected (which
   *   doesn't map character codes to glyph indices at all).
   */
  FT_EXPORT( FT_Error )
  FT_Set_Charmap( FT_Face     face,
                  FT_CharMap  charmap );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Charmap_Index
   *
   * @description:
   *   Retrieve index of a given charmap.
   *
   * @input:
   *   charmap ::
   *     A handle to a charmap.
   *
   * @return:
   *   The index into the array of character maps within the face to which
   *   `charmap` belongs.  If an error occurs, -1 is returned.
   *
   */
  FT_EXPORT( FT_Int )
  FT_Get_Charmap_Index( FT_CharMap  charmap );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Char_Index
   *
   * @description:
   *   Return the glyph index of a given character code.  This function uses
   *   the currently selected charmap to do the mapping.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   charcode ::
   *     The character code.
   *
   * @return:
   *   The glyph index.  0~means 'undefined character code'.
   *
   * @note:
   *   If you use FreeType to manipulate the contents of font files directly,
   *   be aware that the glyph index returned by this function doesn't always
   *   correspond to the internal indices used within the file.  This is done
   *   to ensure that value~0 always corresponds to the 'missing glyph'.  If
   *   the first glyph is not named '.notdef', then for Type~1 and Type~42
   *   fonts, '.notdef' will be moved into the glyph ID~0 position, and
   *   whatever was there will be moved to the position '.notdef' had.  For
   *   Type~1 fonts, if there is no '.notdef' glyph at all, then one will be
   *   created at index~0 and whatever was there will be moved to the last
   *   index -- Type~42 fonts are considered invalid under this condition.
   */
  FT_EXPORT( FT_UInt )
  FT_Get_Char_Index( FT_Face   face,
                     FT_ULong  charcode );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_First_Char
   *
   * @description:
   *   Return the first character code in the current charmap of a given
   *   face, together with its corresponding glyph index.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   * @output:
   *   agindex ::
   *     Glyph index of first character code.  0~if charmap is empty.
   *
   * @return:
   *   The charmap's first character code.
   *
   * @note:
   *   You should use this function together with @FT_Get_Next_Char to parse
   *   all character codes available in a given charmap.  The code should
   *   look like this:
   *
   *   ```
   *     FT_ULong  charcode;
   *     FT_UInt   gindex;
   *
   *
   *     charcode = FT_Get_First_Char( face, &gindex );
   *     while ( gindex != 0 )
   *     {
   *       ... do something with (charcode,gindex) pair ...
   *
   *       charcode = FT_Get_Next_Char( face, charcode, &gindex );
   *     }
   *   ```
   *
   *   Be aware that character codes can have values up to 0xFFFFFFFF; this
   *   might happen for non-Unicode or malformed cmaps.  However, even with
   *   regular Unicode encoding, so-called 'last resort fonts' (using SFNT
   *   cmap format 13, see function @FT_Get_CMap_Format) normally have
   *   entries for all Unicode characters up to 0x1FFFFF, which can cause *a
   *   lot* of iterations.
   *
   *   Note that `*agindex` is set to~0 if the charmap is empty.  The result
   *   itself can be~0 in two cases: if the charmap is empty or if the
   *   value~0 is the first valid character code.
   */
  FT_EXPORT( FT_ULong )
  FT_Get_First_Char( FT_Face   face,
                     FT_UInt  *agindex );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Next_Char
   *
   * @description:
   *   Return the next character code in the current charmap of a given face
   *   following the value `char_code`, as well as the corresponding glyph
   *   index.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   char_code ::
   *     The starting character code.
   *
   * @output:
   *   agindex ::
   *     Glyph index of next character code.  0~if charmap is empty.
   *
   * @return:
   *   The charmap's next character code.
   *
   * @note:
   *   You should use this function with @FT_Get_First_Char to walk over all
   *   character codes available in a given charmap.  See the note for that
   *   function for a simple code example.
   *
   *   Note that `*agindex` is set to~0 when there are no more codes in the
   *   charmap.
   */
  FT_EXPORT( FT_ULong )
  FT_Get_Next_Char( FT_Face    face,
                    FT_ULong   char_code,
                    FT_UInt   *agindex );


  /**************************************************************************
   *
   * @section:
   *   face_creation
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Face_Properties
   *
   * @description:
   *   Set or override certain (library or module-wide) properties on a
   *   face-by-face basis.  Useful for finer-grained control and avoiding
   *   locks on shared structures (threads can modify their own faces as they
   *   see fit).
   *
   *   Contrary to @FT_Property_Set, this function uses @FT_Parameter so that
   *   you can pass multiple properties to the target face in one call.  Note
   *   that only a subset of the available properties can be controlled.
   *
   *   * @FT_PARAM_TAG_STEM_DARKENING (stem darkening, corresponding to the
   *     property `no-stem-darkening` provided by the 'autofit', 'cff',
   *     'type1', and 't1cid' modules; see @no-stem-darkening).
   *
   *   * @FT_PARAM_TAG_LCD_FILTER_WEIGHTS (LCD filter weights, corresponding
   *     to function @FT_Library_SetLcdFilterWeights).
   *
   *   * @FT_PARAM_TAG_RANDOM_SEED (seed value for the CFF, Type~1, and CID
   *     'random' operator, corresponding to the `random-seed` property
   *     provided by the 'cff', 'type1', and 't1cid' modules; see
   *     @random-seed).
   *
   *   Pass `NULL` as `data` in @FT_Parameter for a given tag to reset the
   *   option and use the library or module default again.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   num_properties ::
   *     The number of properties that follow.
   *
   *   properties ::
   *     A handle to an @FT_Parameter array with `num_properties` elements.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @example:
   *   Here is an example that sets three properties.  You must define
   *   `FT_CONFIG_OPTION_SUBPIXEL_RENDERING` to make the LCD filter examples
   *   work.
   *
   *   ```
   *     FT_Parameter         property1;
   *     FT_Bool              darken_stems = 1;
   *
   *     FT_Parameter         property2;
   *     FT_LcdFiveTapFilter  custom_weight =
   *                            { 0x11, 0x44, 0x56, 0x44, 0x11 };
   *
   *     FT_Parameter         property3;
   *     FT_Int32             random_seed = 314159265;
   *
   *     FT_Parameter         properties[3] = { property1,
   *                                            property2,
   *                                            property3 };
   *
   *
   *     property1.tag  = FT_PARAM_TAG_STEM_DARKENING;
   *     property1.data = &darken_stems;
   *
   *     property2.tag  = FT_PARAM_TAG_LCD_FILTER_WEIGHTS;
   *     property2.data = custom_weight;
   *
   *     property3.tag  = FT_PARAM_TAG_RANDOM_SEED;
   *     property3.data = &random_seed;
   *
   *     FT_Face_Properties( face, 3, properties );
   *   ```
   *
   *   The next example resets a single property to its default value.
   *
   *   ```
   *     FT_Parameter  property;
   *
   *
   *     property.tag  = FT_PARAM_TAG_LCD_FILTER_WEIGHTS;
   *     property.data = NULL;
   *
   *     FT_Face_Properties( face, 1, &property );
   *   ```
   *
   * @since:
   *   2.8
   *
   */
  FT_EXPORT( FT_Error )
  FT_Face_Properties( FT_Face        face,
                      FT_UInt        num_properties,
                      FT_Parameter*  properties );


  /**************************************************************************
   *
   * @section:
   *   information_retrieval
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Get_Name_Index
   *
   * @description:
   *   Return the glyph index of a given glyph name.  This only works
   *   for those faces where @FT_HAS_GLYPH_NAMES returns true.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   glyph_name ::
   *     The glyph name.
   *
   * @return:
   *   The glyph index.  0~means 'undefined character code'.
   *
   * @note:
   *   Acceptable glyph names might come from the [Adobe Glyph
   *   List](https://github.com/adobe-type-tools/agl-aglfn).  See
   *   @FT_Get_Glyph_Name for the inverse functionality.
   *
   *   This function has limited capabilities if the config macro
   *   `FT_CONFIG_OPTION_POSTSCRIPT_NAMES` is not defined in `ftoption.h`:
   *   It then works only for fonts that actually embed glyph names (which
   *   many recent OpenType fonts do not).
   */
  FT_EXPORT( FT_UInt )
  FT_Get_Name_Index( FT_Face           face,
                     const FT_String*  glyph_name );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Glyph_Name
   *
   * @description:
   *   Retrieve the ASCII name of a given glyph in a face.  This only works
   *   for those faces where @FT_HAS_GLYPH_NAMES returns true.
   *
   * @input:
   *   face ::
   *     A handle to a source face object.
   *
   *   glyph_index ::
   *     The glyph index.
   *
   *   buffer_max ::
   *     The maximum number of bytes available in the buffer.
   *
   * @output:
   *   buffer ::
   *     A pointer to a target buffer where the name is copied to.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   An error is returned if the face doesn't provide glyph names or if the
   *   glyph index is invalid.  In all cases of failure, the first byte of
   *   `buffer` is set to~0 to indicate an empty name.
   *
   *   The glyph name is truncated to fit within the buffer if it is too
   *   long.  The returned string is always zero-terminated.
   *
   *   Be aware that FreeType reorders glyph indices internally so that glyph
   *   index~0 always corresponds to the 'missing glyph' (called '.notdef').
   *
   *   This function has limited capabilities if the config macro
   *   `FT_CONFIG_OPTION_POSTSCRIPT_NAMES` is not defined in `ftoption.h`:
   *   It then works only for fonts that actually embed glyph names (which
   *   many recent OpenType fonts do not).
   */
  FT_EXPORT( FT_Error )
  FT_Get_Glyph_Name( FT_Face     face,
                     FT_UInt     glyph_index,
                     FT_Pointer  buffer,
                     FT_UInt     buffer_max );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Postscript_Name
   *
   * @description:
   *   Retrieve the ASCII PostScript name of a given face, if available.
   *   This only works with PostScript, TrueType, and OpenType fonts.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   * @return:
   *   A pointer to the face's PostScript name.  `NULL` if unavailable.
   *
   * @note:
   *   The returned pointer is owned by the face and is destroyed with it.
   *
   *   For variation fonts, this string changes if you select a different
   *   instance, and you have to call `FT_Get_PostScript_Name` again to
   *   retrieve it.  FreeType follows Adobe TechNote #5902, 'Generating
   *   PostScript Names for Fonts Using OpenType Font Variations'.
   *
   *     https://download.macromedia.com/pub/developer/opentype/tech-notes/5902.AdobePSNameGeneration.html
   *
   *   [Since 2.9] Special PostScript names for named instances are only
   *   returned if the named instance is set with @FT_Set_Named_Instance (and
   *   the font has corresponding entries in its 'fvar' table or is the
   *   default named instance).  If @FT_IS_VARIATION returns true, the
   *   algorithmically derived PostScript name is provided, not looking up
   *   special entries for named instances.
   */
  FT_EXPORT( const char* )
  FT_Get_Postscript_Name( FT_Face  face );


  /**************************************************************************
   *
   * @enum:
   *   FT_SUBGLYPH_FLAG_XXX
   *
   * @description:
   *   A list of constants describing subglyphs.  Please refer to the 'glyf'
   *   table description in the OpenType specification for the meaning of the
   *   various flags (which get synthesized for non-OpenType subglyphs).
   *
   *     https://docs.microsoft.com/en-us/typography/opentype/spec/glyf#composite-glyph-description
   *
   * @values:
   *   FT_SUBGLYPH_FLAG_ARGS_ARE_WORDS ::
   *   FT_SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES ::
   *   FT_SUBGLYPH_FLAG_ROUND_XY_TO_GRID ::
   *   FT_SUBGLYPH_FLAG_SCALE ::
   *   FT_SUBGLYPH_FLAG_XY_SCALE ::
   *   FT_SUBGLYPH_FLAG_2X2 ::
   *   FT_SUBGLYPH_FLAG_USE_MY_METRICS ::
   *
   */
#define FT_SUBGLYPH_FLAG_ARGS_ARE_WORDS          1
#define FT_SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES      2
#define FT_SUBGLYPH_FLAG_ROUND_XY_TO_GRID        4
#define FT_SUBGLYPH_FLAG_SCALE                   8
#define FT_SUBGLYPH_FLAG_XY_SCALE             0x40
#define FT_SUBGLYPH_FLAG_2X2                  0x80
#define FT_SUBGLYPH_FLAG_USE_MY_METRICS      0x200


  /**************************************************************************
   *
   * @function:
   *   FT_Get_SubGlyph_Info
   *
   * @description:
   *   Retrieve a description of a given subglyph.  Only use it if
   *   `glyph->format` is @FT_GLYPH_FORMAT_COMPOSITE; an error is returned
   *   otherwise.
   *
   * @input:
   *   glyph ::
   *     The source glyph slot.
   *
   *   sub_index ::
   *     The index of the subglyph.  Must be less than
   *     `glyph->num_subglyphs`.
   *
   * @output:
   *   p_index ::
   *     The glyph index of the subglyph.
   *
   *   p_flags ::
   *     The subglyph flags, see @FT_SUBGLYPH_FLAG_XXX.
   *
   *   p_arg1 ::
   *     The subglyph's first argument (if any).
   *
   *   p_arg2 ::
   *     The subglyph's second argument (if any).
   *
   *   p_transform ::
   *     The subglyph transformation (if any).
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The values of `*p_arg1`, `*p_arg2`, and `*p_transform` must be
   *   interpreted depending on the flags returned in `*p_flags`.  See the
   *   OpenType specification for details.
   *
   *     https://docs.microsoft.com/en-us/typography/opentype/spec/glyf#composite-glyph-description
   *
   */
  FT_EXPORT( FT_Error )
  FT_Get_SubGlyph_Info( FT_GlyphSlot  glyph,
                        FT_UInt       sub_index,
                        FT_Int       *p_index,
                        FT_UInt      *p_flags,
                        FT_Int       *p_arg1,
                        FT_Int       *p_arg2,
                        FT_Matrix    *p_transform );


  /**************************************************************************
   *
   * @enum:
   *   FT_FSTYPE_XXX
   *
   * @description:
   *   A list of bit flags used in the `fsType` field of the OS/2 table in a
   *   TrueType or OpenType font and the `FSType` entry in a PostScript font.
   *   These bit flags are returned by @FT_Get_FSType_Flags; they inform
   *   client applications of embedding and subsetting restrictions
   *   associated with a font.
   *
   *   See
   *   https://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/FontPolicies.pdf
   *   for more details.
   *
   * @values:
   *   FT_FSTYPE_INSTALLABLE_EMBEDDING ::
   *     Fonts with no fsType bit set may be embedded and permanently
   *     installed on the remote system by an application.
   *
   *   FT_FSTYPE_RESTRICTED_LICENSE_EMBEDDING ::
   *     Fonts that have only this bit set must not be modified, embedded or
   *     exchanged in any manner without first obtaining permission of the
   *     font software copyright owner.
   *
   *   FT_FSTYPE_PREVIEW_AND_PRINT_EMBEDDING ::
   *     The font may be embedded and temporarily loaded on the remote
   *     system.  Documents containing Preview & Print fonts must be opened
   *     'read-only'; no edits can be applied to the document.
   *
   *   FT_FSTYPE_EDITABLE_EMBEDDING ::
   *     The font may be embedded but must only be installed temporarily on
   *     other systems.  In contrast to Preview & Print fonts, documents
   *     containing editable fonts may be opened for reading, editing is
   *     permitted, and changes may be saved.
   *
   *   FT_FSTYPE_NO_SUBSETTING ::
   *     The font may not be subsetted prior to embedding.
   *
   *   FT_FSTYPE_BITMAP_EMBEDDING_ONLY ::
   *     Only bitmaps contained in the font may be embedded; no outline data
   *     may be embedded.  If there are no bitmaps available in the font,
   *     then the font is unembeddable.
   *
   * @note:
   *   The flags are ORed together, thus more than a single value can be
   *   returned.
   *
   *   While the `fsType` flags can indicate that a font may be embedded, a
   *   license with the font vendor may be separately required to use the
   *   font in this way.
   */
#define FT_FSTYPE_INSTALLABLE_EMBEDDING         0x0000
#define FT_FSTYPE_RESTRICTED_LICENSE_EMBEDDING  0x0002
#define FT_FSTYPE_PREVIEW_AND_PRINT_EMBEDDING   0x0004
#define FT_FSTYPE_EDITABLE_EMBEDDING            0x0008
#define FT_FSTYPE_NO_SUBSETTING                 0x0100
#define FT_FSTYPE_BITMAP_EMBEDDING_ONLY         0x0200


  /**************************************************************************
   *
   * @function:
   *   FT_Get_FSType_Flags
   *
   * @description:
   *   Return the `fsType` flags for a font.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   * @return:
   *   The `fsType` flags, see @FT_FSTYPE_XXX.
   *
   * @note:
   *   Use this function rather than directly reading the `fs_type` field in
   *   the @PS_FontInfoRec structure, which is only guaranteed to return the
   *   correct results for Type~1 fonts.
   *
   * @since:
   *   2.3.8
   *
   */
  FT_EXPORT( FT_UShort )
  FT_Get_FSType_Flags( FT_Face  face );


  /**************************************************************************
   *
   * @section:
   *   glyph_variants
   *
   * @title:
   *   Unicode Variation Sequences
   *
   * @abstract:
   *   The FreeType~2 interface to Unicode Variation Sequences (UVS), using
   *   the SFNT cmap format~14.
   *
   * @description:
   *   Many characters, especially for CJK scripts, have variant forms.  They
   *   are a sort of grey area somewhere between being totally irrelevant and
   *   semantically distinct; for this reason, the Unicode consortium decided
   *   to introduce Variation Sequences (VS), consisting of a Unicode base
   *   character and a variation selector instead of further extending the
   *   already huge number of characters.
   *
   *   Unicode maintains two different sets, namely 'Standardized Variation
   *   Sequences' and registered 'Ideographic Variation Sequences' (IVS),
   *   collected in the 'Ideographic Variation Database' (IVD).
   *
   *     https://unicode.org/Public/UCD/latest/ucd/StandardizedVariants.txt
   *     https://unicode.org/reports/tr37/ https://unicode.org/ivd/
   *
   *   To date (January 2017), the character with the most ideographic
   *   variations is U+9089, having 32 such IVS.
   *
   *   Three Mongolian Variation Selectors have the values U+180B-U+180D; 256
   *   generic Variation Selectors are encoded in the ranges U+FE00-U+FE0F
   *   and U+E0100-U+E01EF.  IVS currently use Variation Selectors from the
   *   range U+E0100-U+E01EF only.
   *
   *   A VS consists of the base character value followed by a single
   *   Variation Selector.  For example, to get the first variation of
   *   U+9089, you have to write the character sequence `U+9089 U+E0100`.
   *
   *   Adobe and MS decided to support both standardized and ideographic VS
   *   with a new cmap subtable (format~14).  It is an odd subtable because
   *   it is not a mapping of input code points to glyphs, but contains lists
   *   of all variations supported by the font.
   *
   *   A variation may be either 'default' or 'non-default' for a given font.
   *   A default variation is the one you will get for that code point if you
   *   look it up in the standard Unicode cmap.  A non-default variation is a
   *   different glyph.
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Face_GetCharVariantIndex
   *
   * @description:
   *   Return the glyph index of a given character code as modified by the
   *   variation selector.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   charcode ::
   *     The character code point in Unicode.
   *
   *   variantSelector ::
   *     The Unicode code point of the variation selector.
   *
   * @return:
   *   The glyph index.  0~means either 'undefined character code', or
   *   'undefined selector code', or 'no variation selector cmap subtable',
   *   or 'current CharMap is not Unicode'.
   *
   * @note:
   *   If you use FreeType to manipulate the contents of font files directly,
   *   be aware that the glyph index returned by this function doesn't always
   *   correspond to the internal indices used within the file.  This is done
   *   to ensure that value~0 always corresponds to the 'missing glyph'.
   *
   *   This function is only meaningful if
   *     a) the font has a variation selector cmap sub table, and
   *     b) the current charmap has a Unicode encoding.
   *
   * @since:
   *   2.3.6
   *
   */
  FT_EXPORT( FT_UInt )
  FT_Face_GetCharVariantIndex( FT_Face   face,
                               FT_ULong  charcode,
                               FT_ULong  variantSelector );


  /**************************************************************************
   *
   * @function:
   *   FT_Face_GetCharVariantIsDefault
   *
   * @description:
   *   Check whether this variation of this Unicode character is the one to
   *   be found in the charmap.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   charcode ::
   *     The character codepoint in Unicode.
   *
   *   variantSelector ::
   *     The Unicode codepoint of the variation selector.
   *
   * @return:
   *   1~if found in the standard (Unicode) cmap, 0~if found in the variation
   *   selector cmap, or -1 if it is not a variation.
   *
   * @note:
   *   This function is only meaningful if the font has a variation selector
   *   cmap subtable.
   *
   * @since:
   *   2.3.6
   *
   */
  FT_EXPORT( FT_Int )
  FT_Face_GetCharVariantIsDefault( FT_Face   face,
                                   FT_ULong  charcode,
                                   FT_ULong  variantSelector );


  /**************************************************************************
   *
   * @function:
   *   FT_Face_GetVariantSelectors
   *
   * @description:
   *   Return a zero-terminated list of Unicode variation selectors found in
   *   the font.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   * @return:
   *   A pointer to an array of selector code points, or `NULL` if there is
   *   no valid variation selector cmap subtable.
   *
   * @note:
   *   The last item in the array is~0; the array is owned by the @FT_Face
   *   object but can be overwritten or released on the next call to a
   *   FreeType function.
   *
   * @since:
   *   2.3.6
   *
   */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetVariantSelectors( FT_Face  face );


  /**************************************************************************
   *
   * @function:
   *   FT_Face_GetVariantsOfChar
   *
   * @description:
   *   Return a zero-terminated list of Unicode variation selectors found for
   *   the specified character code.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   charcode ::
   *     The character codepoint in Unicode.
   *
   * @return:
   *   A pointer to an array of variation selector code points that are
   *   active for the given character, or `NULL` if the corresponding list is
   *   empty.
   *
   * @note:
   *   The last item in the array is~0; the array is owned by the @FT_Face
   *   object but can be overwritten or released on the next call to a
   *   FreeType function.
   *
   * @since:
   *   2.3.6
   *
   */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetVariantsOfChar( FT_Face   face,
                             FT_ULong  charcode );


  /**************************************************************************
   *
   * @function:
   *   FT_Face_GetCharsOfVariant
   *
   * @description:
   *   Return a zero-terminated list of Unicode character codes found for the
   *   specified variation selector.
   *
   * @input:
   *   face ::
   *     A handle to the source face object.
   *
   *   variantSelector ::
   *     The variation selector code point in Unicode.
   *
   * @return:
   *   A list of all the code points that are specified by this selector
   *   (both default and non-default codes are returned) or `NULL` if there
   *   is no valid cmap or the variation selector is invalid.
   *
   * @note:
   *   The last item in the array is~0; the array is owned by the @FT_Face
   *   object but can be overwritten or released on the next call to a
   *   FreeType function.
   *
   * @since:
   *   2.3.6
   *
   */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetCharsOfVariant( FT_Face   face,
                             FT_ULong  variantSelector );


  /**************************************************************************
   *
   * @section:
   *   computations
   *
   * @title:
   *   Computations
   *
   * @abstract:
   *   Crunching fixed numbers and vectors.
   *
   * @description:
   *   This section contains various functions used to perform computations
   *   on 16.16 fixed-point numbers or 2D vectors.  FreeType does not use
   *   floating-point data types.
   *
   *   **Attention**: Most arithmetic functions take `FT_Long` as arguments.
   *   For historical reasons, FreeType was designed under the assumption
   *   that `FT_Long` is a 32-bit integer; results can thus be undefined if
   *   the arguments don't fit into 32 bits.
   *
   * @order:
   *   FT_MulDiv
   *   FT_MulFix
   *   FT_DivFix
   *   FT_RoundFix
   *   FT_CeilFix
   *   FT_FloorFix
   *   FT_Vector_Transform
   *   FT_Matrix_Multiply
   *   FT_Matrix_Invert
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_MulDiv
   *
   * @description:
   *   Compute `(a*b)/c` with maximum accuracy, using a 64-bit intermediate
   *   integer whenever necessary.
   *
   *   This function isn't necessarily as fast as some processor-specific
   *   operations, but is at least completely portable.
   *
   * @input:
   *   a ::
   *     The first multiplier.
   *
   *   b ::
   *     The second multiplier.
   *
   *   c ::
   *     The divisor.
   *
   * @return:
   *   The result of `(a*b)/c`.  This function never traps when trying to
   *   divide by zero; it simply returns 'MaxInt' or 'MinInt' depending on
   *   the signs of `a` and `b`.
   */
  FT_EXPORT( FT_Long )
  FT_MulDiv( FT_Long  a,
             FT_Long  b,
             FT_Long  c );


  /**************************************************************************
   *
   * @function:
   *   FT_MulFix
   *
   * @description:
   *   Compute `(a*b)/0x10000` with maximum accuracy.  Its main use is to
   *   multiply a given value by a 16.16 fixed-point factor.
   *
   * @input:
   *   a ::
   *     The first multiplier.
   *
   *   b ::
   *     The second multiplier.  Use a 16.16 factor here whenever possible
   *     (see note below).
   *
   * @return:
   *   The result of `(a*b)/0x10000`.
   *
   * @note:
   *   This function has been optimized for the case where the absolute value
   *   of `a` is less than 2048, and `b` is a 16.16 scaling factor.  As this
   *   happens mainly when scaling from notional units to fractional pixels
   *   in FreeType, it resulted in noticeable speed improvements between
   *   versions 2.x and 1.x.
   *
   *   As a conclusion, always try to place a 16.16 factor as the _second_
   *   argument of this function; this can make a great difference.
   */
  FT_EXPORT( FT_Long )
  FT_MulFix( FT_Long  a,
             FT_Long  b );


  /**************************************************************************
   *
   * @function:
   *   FT_DivFix
   *
   * @description:
   *   Compute `(a*0x10000)/b` with maximum accuracy.  Its main use is to
   *   divide a given value by a 16.16 fixed-point factor.
   *
   * @input:
   *   a ::
   *     The numerator.
   *
   *   b ::
   *     The denominator.  Use a 16.16 factor here.
   *
   * @return:
   *   The result of `(a*0x10000)/b`.
   */
  FT_EXPORT( FT_Long )
  FT_DivFix( FT_Long  a,
             FT_Long  b );


  /**************************************************************************
   *
   * @function:
   *   FT_RoundFix
   *
   * @description:
   *   Round a 16.16 fixed number.
   *
   * @input:
   *   a ::
   *     The number to be rounded.
   *
   * @return:
   *   `a` rounded to the nearest 16.16 fixed integer, halfway cases away
   *   from zero.
   *
   * @note:
   *   The function uses wrap-around arithmetic.
   */
  FT_EXPORT( FT_Fixed )
  FT_RoundFix( FT_Fixed  a );


  /**************************************************************************
   *
   * @function:
   *   FT_CeilFix
   *
   * @description:
   *   Compute the smallest following integer of a 16.16 fixed number.
   *
   * @input:
   *   a ::
   *     The number for which the ceiling function is to be computed.
   *
   * @return:
   *   `a` rounded towards plus infinity.
   *
   * @note:
   *   The function uses wrap-around arithmetic.
   */
  FT_EXPORT( FT_Fixed )
  FT_CeilFix( FT_Fixed  a );


  /**************************************************************************
   *
   * @function:
   *   FT_FloorFix
   *
   * @description:
   *   Compute the largest previous integer of a 16.16 fixed number.
   *
   * @input:
   *   a ::
   *     The number for which the floor function is to be computed.
   *
   * @return:
   *   `a` rounded towards minus infinity.
   */
  FT_EXPORT( FT_Fixed )
  FT_FloorFix( FT_Fixed  a );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_Transform
   *
   * @description:
   *   Transform a single vector through a 2x2 matrix.
   *
   * @inout:
   *   vector ::
   *     The target vector to transform.
   *
   * @input:
   *   matrix ::
   *     A pointer to the source 2x2 matrix.
   *
   * @note:
   *   The result is undefined if either `vector` or `matrix` is invalid.
   */
  FT_EXPORT( void )
  FT_Vector_Transform( FT_Vector*        vector,
                       const FT_Matrix*  matrix );


  /**************************************************************************
   *
   * @section:
   *   library_setup
   *
   */

  /**************************************************************************
   *
   * @enum:
   *   FREETYPE_XXX
   *
   * @description:
   *   These three macros identify the FreeType source code version.  Use
   *   @FT_Library_Version to access them at runtime.
   *
   * @values:
   *   FREETYPE_MAJOR ::
   *     The major version number.
   *   FREETYPE_MINOR ::
   *     The minor version number.
   *   FREETYPE_PATCH ::
   *     The patch level.
   *
   * @note:
   *   The version number of FreeType if built as a dynamic link library with
   *   the 'libtool' package is _not_ controlled by these three macros.
   *
   */
#define FREETYPE_MAJOR  2
#define FREETYPE_MINOR  13
#define FREETYPE_PATCH  2


  /**************************************************************************
   *
   * @function:
   *   FT_Library_Version
   *
   * @description:
   *   Return the version of the FreeType library being used.  This is useful
   *   when dynamically linking to the library, since one cannot use the
   *   macros @FREETYPE_MAJOR, @FREETYPE_MINOR, and @FREETYPE_PATCH.
   *
   * @input:
   *   library ::
   *     A source library handle.
   *
   * @output:
   *   amajor ::
   *     The major version number.
   *
   *   aminor ::
   *     The minor version number.
   *
   *   apatch ::
   *     The patch version number.
   *
   * @note:
   *   The reason why this function takes a `library` argument is because
   *   certain programs implement library initialization in a custom way that
   *   doesn't use @FT_Init_FreeType.
   *
   *   In such cases, the library version might not be available before the
   *   library object has been created.
   */
  FT_EXPORT( void )
  FT_Library_Version( FT_Library   library,
                      FT_Int      *amajor,
                      FT_Int      *aminor,
                      FT_Int      *apatch );


  /**************************************************************************
   *
   * @section:
   *   other_api_data
   *
   */

  /**************************************************************************
   *
   * @function:
   *   FT_Face_CheckTrueTypePatents
   *
   * @description:
   *   Deprecated, does nothing.
   *
   * @input:
   *   face ::
   *     A face handle.
   *
   * @return:
   *   Always returns false.
   *
   * @note:
   *   Since May 2010, TrueType hinting is no longer patented.
   *
   * @since:
   *   2.3.5
   *
   */
  FT_EXPORT( FT_Bool )
  FT_Face_CheckTrueTypePatents( FT_Face  face );


  /**************************************************************************
   *
   * @function:
   *   FT_Face_SetUnpatentedHinting
   *
   * @description:
   *   Deprecated, does nothing.
   *
   * @input:
   *   face ::
   *     A face handle.
   *
   *   value ::
   *     New boolean setting.
   *
   * @return:
   *   Always returns false.
   *
   * @note:
   *   Since May 2010, TrueType hinting is no longer patented.
   *
   * @since:
   *   2.3.5
   *
   */
  FT_EXPORT( FT_Bool )
  FT_Face_SetUnpatentedHinting( FT_Face  face,
                                FT_Bool  value );

  /* */


FT_END_HEADER

#endif /* FREETYPE_H_ */


/* END */
