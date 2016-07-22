/***************************************************************************/
/*                                                                         */
/*  freetype.h                                                             */
/*                                                                         */
/*    FreeType high-level API and common types (specification only).       */
/*                                                                         */
/*  Copyright 1996-2013 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FREETYPE_H__
#define __FREETYPE_H__


#ifndef FT_FREETYPE_H
#error "`ft2build.h' hasn't been included yet!"
#error "Please always use macros to include FreeType header files."
#error "Example:"
#error "  #include <ft2build.h>"
#error "  #include FT_FREETYPE_H"
#endif


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include FT_TYPES_H
#include FT_ERRORS_H


FT_BEGIN_HEADER



  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    user_allocation                                                    */
  /*                                                                       */
  /* <Title>                                                               */
  /*    User allocation                                                    */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    How client applications should allocate FreeType data structures.  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    FreeType assumes that structures allocated by the user and passed  */
  /*    as arguments are zeroed out except for the actual data.  In other  */
  /*    words, it is recommended to use `calloc' (or variants of it)       */
  /*    instead of `malloc' for allocation.                                */
  /*                                                                       */
  /*************************************************************************/



  /*************************************************************************/
  /*************************************************************************/
  /*                                                                       */
  /*                        B A S I C   T Y P E S                          */
  /*                                                                       */
  /*************************************************************************/
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    base_interface                                                     */
  /*                                                                       */
  /* <Title>                                                               */
  /*    Base Interface                                                     */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    The FreeType~2 base font interface.                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This section describes the public high-level API of FreeType~2.    */
  /*                                                                       */
  /* <Order>                                                               */
  /*    FT_Library                                                         */
  /*    FT_Face                                                            */
  /*    FT_Size                                                            */
  /*    FT_GlyphSlot                                                       */
  /*    FT_CharMap                                                         */
  /*    FT_Encoding                                                        */
  /*                                                                       */
  /*    FT_FaceRec                                                         */
  /*                                                                       */
  /*    FT_FACE_FLAG_SCALABLE                                              */
  /*    FT_FACE_FLAG_FIXED_SIZES                                           */
  /*    FT_FACE_FLAG_FIXED_WIDTH                                           */
  /*    FT_FACE_FLAG_HORIZONTAL                                            */
  /*    FT_FACE_FLAG_VERTICAL                                              */
  /*    FT_FACE_FLAG_SFNT                                                  */
  /*    FT_FACE_FLAG_KERNING                                               */
  /*    FT_FACE_FLAG_MULTIPLE_MASTERS                                      */
  /*    FT_FACE_FLAG_GLYPH_NAMES                                           */
  /*    FT_FACE_FLAG_EXTERNAL_STREAM                                       */
  /*    FT_FACE_FLAG_FAST_GLYPHS                                           */
  /*    FT_FACE_FLAG_HINTER                                                */
  /*                                                                       */
  /*    FT_STYLE_FLAG_BOLD                                                 */
  /*    FT_STYLE_FLAG_ITALIC                                               */
  /*                                                                       */
  /*    FT_SizeRec                                                         */
  /*    FT_Size_Metrics                                                    */
  /*                                                                       */
  /*    FT_GlyphSlotRec                                                    */
  /*    FT_Glyph_Metrics                                                   */
  /*    FT_SubGlyph                                                        */
  /*                                                                       */
  /*    FT_Bitmap_Size                                                     */
  /*                                                                       */
  /*    FT_Init_FreeType                                                   */
  /*    FT_Done_FreeType                                                   */
  /*                                                                       */
  /*    FT_New_Face                                                        */
  /*    FT_Done_Face                                                       */
  /*    FT_New_Memory_Face                                                 */
  /*    FT_Open_Face                                                       */
  /*    FT_Open_Args                                                       */
  /*    FT_Parameter                                                       */
  /*    FT_Attach_File                                                     */
  /*    FT_Attach_Stream                                                   */
  /*                                                                       */
  /*    FT_Set_Char_Size                                                   */
  /*    FT_Set_Pixel_Sizes                                                 */
  /*    FT_Request_Size                                                    */
  /*    FT_Select_Size                                                     */
  /*    FT_Size_Request_Type                                               */
  /*    FT_Size_Request                                                    */
  /*    FT_Set_Transform                                                   */
  /*    FT_Load_Glyph                                                      */
  /*    FT_Get_Char_Index                                                  */
  /*    FT_Get_Name_Index                                                  */
  /*    FT_Load_Char                                                       */
  /*                                                                       */
  /*    FT_OPEN_MEMORY                                                     */
  /*    FT_OPEN_STREAM                                                     */
  /*    FT_OPEN_PATHNAME                                                   */
  /*    FT_OPEN_DRIVER                                                     */
  /*    FT_OPEN_PARAMS                                                     */
  /*                                                                       */
  /*    FT_LOAD_DEFAULT                                                    */
  /*    FT_LOAD_RENDER                                                     */
  /*    FT_LOAD_MONOCHROME                                                 */
  /*    FT_LOAD_LINEAR_DESIGN                                              */
  /*    FT_LOAD_NO_SCALE                                                   */
  /*    FT_LOAD_NO_HINTING                                                 */
  /*    FT_LOAD_NO_BITMAP                                                  */
  /*    FT_LOAD_CROP_BITMAP                                                */
  /*                                                                       */
  /*    FT_LOAD_VERTICAL_LAYOUT                                            */
  /*    FT_LOAD_IGNORE_TRANSFORM                                           */
  /*    FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH                                */
  /*    FT_LOAD_FORCE_AUTOHINT                                             */
  /*    FT_LOAD_NO_RECURSE                                                 */
  /*    FT_LOAD_PEDANTIC                                                   */
  /*                                                                       */
  /*    FT_LOAD_TARGET_NORMAL                                              */
  /*    FT_LOAD_TARGET_LIGHT                                               */
  /*    FT_LOAD_TARGET_MONO                                                */
  /*    FT_LOAD_TARGET_LCD                                                 */
  /*    FT_LOAD_TARGET_LCD_V                                               */
  /*                                                                       */
  /*    FT_Render_Glyph                                                    */
  /*    FT_Render_Mode                                                     */
  /*    FT_Get_Kerning                                                     */
  /*    FT_Kerning_Mode                                                    */
  /*    FT_Get_Track_Kerning                                               */
  /*    FT_Get_Glyph_Name                                                  */
  /*    FT_Get_Postscript_Name                                             */
  /*                                                                       */
  /*    FT_CharMapRec                                                      */
  /*    FT_Select_Charmap                                                  */
  /*    FT_Set_Charmap                                                     */
  /*    FT_Get_Charmap_Index                                               */
  /*                                                                       */
  /*    FT_FSTYPE_INSTALLABLE_EMBEDDING                                    */
  /*    FT_FSTYPE_RESTRICTED_LICENSE_EMBEDDING                             */
  /*    FT_FSTYPE_PREVIEW_AND_PRINT_EMBEDDING                              */
  /*    FT_FSTYPE_EDITABLE_EMBEDDING                                       */
  /*    FT_FSTYPE_NO_SUBSETTING                                            */
  /*    FT_FSTYPE_BITMAP_EMBEDDING_ONLY                                    */
  /*                                                                       */
  /*    FT_Get_FSType_Flags                                                */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Glyph_Metrics                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to model the metrics of a single glyph.  The      */
  /*    values are expressed in 26.6 fractional pixel format; if the flag  */
  /*    @FT_LOAD_NO_SCALE has been used while loading the glyph, values    */
  /*    are expressed in font units instead.                               */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    width ::                                                           */
  /*      The glyph's width.                                               */
  /*                                                                       */
  /*    height ::                                                          */
  /*      The glyph's height.                                              */
  /*                                                                       */
  /*    horiBearingX ::                                                    */
  /*      Left side bearing for horizontal layout.                         */
  /*                                                                       */
  /*    horiBearingY ::                                                    */
  /*      Top side bearing for horizontal layout.                          */
  /*                                                                       */
  /*    horiAdvance ::                                                     */
  /*      Advance width for horizontal layout.                             */
  /*                                                                       */
  /*    vertBearingX ::                                                    */
  /*      Left side bearing for vertical layout.                           */
  /*                                                                       */
  /*    vertBearingY ::                                                    */
  /*      Top side bearing for vertical layout.  Larger positive values    */
  /*      mean further below the vertical glyph origin.                    */
  /*                                                                       */
  /*    vertAdvance ::                                                     */
  /*      Advance height for vertical layout.  Positive values mean the    */
  /*      glyph has a positive advance downward.                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If not disabled with @FT_LOAD_NO_HINTING, the values represent     */
  /*    dimensions of the hinted glyph (in case hinting is applicable).    */
  /*                                                                       */
  /*    Stroking a glyph with an outside border does not increase          */
  /*    `horiAdvance' or `vertAdvance'; you have to manually adjust these  */
  /*    values to account for the added width and height.                  */
  /*                                                                       */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Bitmap_Size                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This structure models the metrics of a bitmap strike (i.e., a set  */
  /*    of glyphs for a given point size and resolution) in a bitmap font. */
  /*    It is used for the `available_sizes' field of @FT_Face.            */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    height :: The vertical distance, in pixels, between two            */
  /*              consecutive baselines.  It is always positive.           */
  /*                                                                       */
  /*    width  :: The average width, in pixels, of all glyphs in the       */
  /*              strike.                                                  */
  /*                                                                       */
  /*    size   :: The nominal size of the strike in 26.6 fractional        */
  /*              points.  This field is not very useful.                  */
  /*                                                                       */
  /*    x_ppem :: The horizontal ppem (nominal width) in 26.6 fractional   */
  /*              pixels.                                                  */
  /*                                                                       */
  /*    y_ppem :: The vertical ppem (nominal height) in 26.6 fractional    */
  /*              pixels.                                                  */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Windows FNT:                                                       */
  /*      The nominal size given in a FNT font is not reliable.  Thus when */
  /*      the driver finds it incorrect, it sets `size' to some calculated */
  /*      values and sets `x_ppem' and `y_ppem' to the pixel width and     */
  /*      height given in the font, respectively.                          */
  /*                                                                       */
  /*    TrueType embedded bitmaps:                                         */
  /*      `size', `width', and `height' values are not contained in the    */
  /*      bitmap strike itself.  They are computed from the global font    */
  /*      parameters.                                                      */
  /*                                                                       */
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

  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Library                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a FreeType library instance.  Each `library' is        */
  /*    completely independent from the others; it is the `root' of a set  */
  /*    of objects like fonts, faces, sizes, etc.                          */
  /*                                                                       */
  /*    It also embeds a memory manager (see @FT_Memory), as well as a     */
  /*    scan-line converter object (see @FT_Raster).                       */
  /*                                                                       */
  /*    In multi-threaded applications, make sure that the same FT_Library */
  /*    object or any of its children doesn't get accessed in parallel.    */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Library objects are normally created by @FT_Init_FreeType, and     */
  /*    destroyed with @FT_Done_FreeType.  If you need reference-counting  */
  /*    (cf. @FT_Reference_Library), use @FT_New_Library and               */
  /*    @FT_Done_Library.                                                  */
  /*                                                                       */
  typedef struct FT_LibraryRec_  *FT_Library;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Module                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given FreeType module object.  Each module can be a  */
  /*    font driver, a renderer, or anything else that provides services   */
  /*    to the formers.                                                    */
  /*                                                                       */
  typedef struct FT_ModuleRec_*  FT_Module;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Driver                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given FreeType font driver object.  Each font driver */
  /*    is a special module capable of creating faces from font files.     */
  /*                                                                       */
  typedef struct FT_DriverRec_*  FT_Driver;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Renderer                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given FreeType renderer.  A renderer is a special    */
  /*    module in charge of converting a glyph image to a bitmap, when     */
  /*    necessary.  Each renderer supports a given glyph image format, and */
  /*    one or more target surface depths.                                 */
  /*                                                                       */
  typedef struct FT_RendererRec_*  FT_Renderer;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Face                                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given typographic face object.  A face object models */
  /*    a given typeface, in a given style.                                */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Each face object also owns a single @FT_GlyphSlot object, as well  */
  /*    as one or more @FT_Size objects.                                   */
  /*                                                                       */
  /*    Use @FT_New_Face or @FT_Open_Face to create a new face object from */
  /*    a given filepathname or a custom input stream.                     */
  /*                                                                       */
  /*    Use @FT_Done_Face to destroy it (along with its slot and sizes).   */
  /*                                                                       */
  /* <Also>                                                                */
  /*    See @FT_FaceRec for the publicly accessible fields of a given face */
  /*    object.                                                            */
  /*                                                                       */
  typedef struct FT_FaceRec_*  FT_Face;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Size                                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to an object used to model a face scaled to a given       */
  /*    character size.                                                    */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Each @FT_Face has an _active_ @FT_Size object that is used by      */
  /*    functions like @FT_Load_Glyph to determine the scaling             */
  /*    transformation which is used to load and hint glyphs and metrics.  */
  /*                                                                       */
  /*    You can use @FT_Set_Char_Size, @FT_Set_Pixel_Sizes,                */
  /*    @FT_Request_Size or even @FT_Select_Size to change the content     */
  /*    (i.e., the scaling values) of the active @FT_Size.                 */
  /*                                                                       */
  /*    You can use @FT_New_Size to create additional size objects for a   */
  /*    given @FT_Face, but they won't be used by other functions until    */
  /*    you activate it through @FT_Activate_Size.  Only one size can be   */
  /*    activated at any given time per face.                              */
  /*                                                                       */
  /* <Also>                                                                */
  /*    See @FT_SizeRec for the publicly accessible fields of a given size */
  /*    object.                                                            */
  /*                                                                       */
  typedef struct FT_SizeRec_*  FT_Size;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_GlyphSlot                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given `glyph slot'.  A slot is a container where it  */
  /*    is possible to load any of the glyphs contained in its parent      */
  /*    face.                                                              */
  /*                                                                       */
  /*    In other words, each time you call @FT_Load_Glyph or               */
  /*    @FT_Load_Char, the slot's content is erased by the new glyph data, */
  /*    i.e., the glyph's metrics, its image (bitmap or outline), and      */
  /*    other control information.                                         */
  /*                                                                       */
  /* <Also>                                                                */
  /*    See @FT_GlyphSlotRec for the publicly accessible glyph fields.     */
  /*                                                                       */
  typedef struct FT_GlyphSlotRec_*  FT_GlyphSlot;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_CharMap                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a given character map.  A charmap is used to translate */
  /*    character codes in a given encoding into glyph indexes for its     */
  /*    parent's face.  Some font formats may provide several charmaps per */
  /*    font.                                                              */
  /*                                                                       */
  /*    Each face object owns zero or more charmaps, but only one of them  */
  /*    can be `active' and used by @FT_Get_Char_Index or @FT_Load_Char.   */
  /*                                                                       */
  /*    The list of available charmaps in a face is available through the  */
  /*    `face->num_charmaps' and `face->charmaps' fields of @FT_FaceRec.   */
  /*                                                                       */
  /*    The currently active charmap is available as `face->charmap'.      */
  /*    You should call @FT_Set_Charmap to change it.                      */
  /*                                                                       */
  /* <Note>                                                                */
  /*    When a new face is created (either through @FT_New_Face or         */
  /*    @FT_Open_Face), the library looks for a Unicode charmap within     */
  /*    the list and automatically activates it.                           */
  /*                                                                       */
  /* <Also>                                                                */
  /*    See @FT_CharMapRec for the publicly accessible fields of a given   */
  /*    character map.                                                     */
  /*                                                                       */
  typedef struct FT_CharMapRec_*  FT_CharMap;


  /*************************************************************************/
  /*                                                                       */
  /* <Macro>                                                               */
  /*    FT_ENC_TAG                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This macro converts four-letter tags into an unsigned long.  It is */
  /*    used to define `encoding' identifiers (see @FT_Encoding).          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Since many 16-bit compilers don't like 32-bit enumerations, you    */
  /*    should redefine this macro in case of problems to something like   */
  /*    this:                                                              */
  /*                                                                       */
  /*    {                                                                  */
  /*      #define FT_ENC_TAG( value, a, b, c, d )  value                   */
  /*    }                                                                  */
  /*                                                                       */
  /*    to get a simple enumeration without assigning special numbers.     */
  /*                                                                       */

#ifndef FT_ENC_TAG
#define FT_ENC_TAG( value, a, b, c, d )         \
          value = ( ( (FT_UInt32)(a) << 24 ) |  \
                    ( (FT_UInt32)(b) << 16 ) |  \
                    ( (FT_UInt32)(c) <<  8 ) |  \
                      (FT_UInt32)(d)         )

#endif /* FT_ENC_TAG */


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_Encoding                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An enumeration used to specify character sets supported by         */
  /*    charmaps.  Used in the @FT_Select_Charmap API function.            */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Despite the name, this enumeration lists specific character        */
  /*    repertories (i.e., charsets), and not text encoding methods (e.g., */
  /*    UTF-8, UTF-16, etc.).                                              */
  /*                                                                       */
  /*    Other encodings might be defined in the future.                    */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_ENCODING_NONE ::                                                */
  /*      The encoding value~0 is reserved.                                */
  /*                                                                       */
  /*    FT_ENCODING_UNICODE ::                                             */
  /*      Corresponds to the Unicode character set.  This value covers     */
  /*      all versions of the Unicode repertoire, including ASCII and      */
  /*      Latin-1.  Most fonts include a Unicode charmap, but not all      */
  /*      of them.                                                         */
  /*                                                                       */
  /*      For example, if you want to access Unicode value U+1F028 (and    */
  /*      the font contains it), use value 0x1F028 as the input value for  */
  /*      @FT_Get_Char_Index.                                              */
  /*                                                                       */
  /*    FT_ENCODING_MS_SYMBOL ::                                           */
  /*      Corresponds to the Microsoft Symbol encoding, used to encode     */
  /*      mathematical symbols in the 32..255 character code range.  For   */
  /*      more information, see `http://www.ceviz.net/symbol.htm'.         */
  /*                                                                       */
  /*    FT_ENCODING_SJIS ::                                                */
  /*      Corresponds to Japanese SJIS encoding.  More info at             */
  /*      at `http://langsupport.japanreference.com/encoding.shtml'.       */
  /*      See note on multi-byte encodings below.                          */
  /*                                                                       */
  /*    FT_ENCODING_GB2312 ::                                              */
  /*      Corresponds to an encoding system for Simplified Chinese as used */
  /*      used in mainland China.                                          */
  /*                                                                       */
  /*    FT_ENCODING_BIG5 ::                                                */
  /*      Corresponds to an encoding system for Traditional Chinese as     */
  /*      used in Taiwan and Hong Kong.                                    */
  /*                                                                       */
  /*    FT_ENCODING_WANSUNG ::                                             */
  /*      Corresponds to the Korean encoding system known as Wansung.      */
  /*      For more information see                                         */
  /*      `http://www.microsoft.com/typography/unicode/949.txt'.           */
  /*                                                                       */
  /*    FT_ENCODING_JOHAB ::                                               */
  /*      The Korean standard character set (KS~C 5601-1992), which        */
  /*      corresponds to MS Windows code page 1361.  This character set    */
  /*      includes all possible Hangeul character combinations.            */
  /*                                                                       */
  /*    FT_ENCODING_ADOBE_LATIN_1 ::                                       */
  /*      Corresponds to a Latin-1 encoding as defined in a Type~1         */
  /*      PostScript font.  It is limited to 256 character codes.          */
  /*                                                                       */
  /*    FT_ENCODING_ADOBE_STANDARD ::                                      */
  /*      Corresponds to the Adobe Standard encoding, as found in Type~1,  */
  /*      CFF, and OpenType/CFF fonts.  It is limited to 256 character     */
  /*      codes.                                                           */
  /*                                                                       */
  /*    FT_ENCODING_ADOBE_EXPERT ::                                        */
  /*      Corresponds to the Adobe Expert encoding, as found in Type~1,    */
  /*      CFF, and OpenType/CFF fonts.  It is limited to 256 character     */
  /*      codes.                                                           */
  /*                                                                       */
  /*    FT_ENCODING_ADOBE_CUSTOM ::                                        */
  /*      Corresponds to a custom encoding, as found in Type~1, CFF, and   */
  /*      OpenType/CFF fonts.  It is limited to 256 character codes.       */
  /*                                                                       */
  /*    FT_ENCODING_APPLE_ROMAN ::                                         */
  /*      Corresponds to the 8-bit Apple roman encoding.  Many TrueType    */
  /*      and OpenType fonts contain a charmap for this encoding, since    */
  /*      older versions of Mac OS are able to use it.                     */
  /*                                                                       */
  /*    FT_ENCODING_OLD_LATIN_2 ::                                         */
  /*      This value is deprecated and was never used nor reported by      */
  /*      FreeType.  Don't use or test for it.                             */
  /*                                                                       */
  /*    FT_ENCODING_MS_SJIS ::                                             */
  /*      Same as FT_ENCODING_SJIS.  Deprecated.                           */
  /*                                                                       */
  /*    FT_ENCODING_MS_GB2312 ::                                           */
  /*      Same as FT_ENCODING_GB2312.  Deprecated.                         */
  /*                                                                       */
  /*    FT_ENCODING_MS_BIG5 ::                                             */
  /*      Same as FT_ENCODING_BIG5.  Deprecated.                           */
  /*                                                                       */
  /*    FT_ENCODING_MS_WANSUNG ::                                          */
  /*      Same as FT_ENCODING_WANSUNG.  Deprecated.                        */
  /*                                                                       */
  /*    FT_ENCODING_MS_JOHAB ::                                            */
  /*      Same as FT_ENCODING_JOHAB.  Deprecated.                          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    By default, FreeType automatically synthesizes a Unicode charmap   */
  /*    for PostScript fonts, using their glyph names dictionaries.        */
  /*    However, it also reports the encodings defined explicitly in the   */
  /*    font file, for the cases when they are needed, with the Adobe      */
  /*    values as well.                                                    */
  /*                                                                       */
  /*    FT_ENCODING_NONE is set by the BDF and PCF drivers if the charmap  */
  /*    is neither Unicode nor ISO-8859-1 (otherwise it is set to          */
  /*    FT_ENCODING_UNICODE).  Use @FT_Get_BDF_Charset_ID to find out      */
  /*    which encoding is really present.  If, for example, the            */
  /*    `cs_registry' field is `KOI8' and the `cs_encoding' field is `R',  */
  /*    the font is encoded in KOI8-R.                                     */
  /*                                                                       */
  /*    FT_ENCODING_NONE is always set (with a single exception) by the    */
  /*    winfonts driver.  Use @FT_Get_WinFNT_Header and examine the        */
  /*    `charset' field of the @FT_WinFNT_HeaderRec structure to find out  */
  /*    which encoding is really present.  For example,                    */
  /*    @FT_WinFNT_ID_CP1251 (204) means Windows code page 1251 (for       */
  /*    Russian).                                                          */
  /*                                                                       */
  /*    FT_ENCODING_NONE is set if `platform_id' is @TT_PLATFORM_MACINTOSH */
  /*    and `encoding_id' is not @TT_MAC_ID_ROMAN (otherwise it is set to  */
  /*    FT_ENCODING_APPLE_ROMAN).                                          */
  /*                                                                       */
  /*    If `platform_id' is @TT_PLATFORM_MACINTOSH, use the function       */
  /*    @FT_Get_CMap_Language_ID  to query the Mac language ID which may   */
  /*    be needed to be able to distinguish Apple encoding variants.  See  */
  /*                                                                       */
  /*      http://www.unicode.org/Public/MAPPINGS/VENDORS/APPLE/README.TXT  */
  /*                                                                       */
  /*    to get an idea how to do that.  Basically, if the language ID      */
  /*    is~0, don't use it, otherwise subtract 1 from the language ID.     */
  /*    Then examine `encoding_id'.  If, for example, `encoding_id' is     */
  /*    @TT_MAC_ID_ROMAN and the language ID (minus~1) is                  */
  /*    `TT_MAC_LANGID_GREEK', it is the Greek encoding, not Roman.        */
  /*    @TT_MAC_ID_ARABIC with `TT_MAC_LANGID_FARSI' means the Farsi       */
  /*    variant the Arabic encoding.                                       */
  /*                                                                       */
  typedef enum  FT_Encoding_
  {
    FT_ENC_TAG( FT_ENCODING_NONE, 0, 0, 0, 0 ),

    FT_ENC_TAG( FT_ENCODING_MS_SYMBOL, 's', 'y', 'm', 'b' ),
    FT_ENC_TAG( FT_ENCODING_UNICODE,   'u', 'n', 'i', 'c' ),

    FT_ENC_TAG( FT_ENCODING_SJIS,    's', 'j', 'i', 's' ),
    FT_ENC_TAG( FT_ENCODING_GB2312,  'g', 'b', ' ', ' ' ),
    FT_ENC_TAG( FT_ENCODING_BIG5,    'b', 'i', 'g', '5' ),
    FT_ENC_TAG( FT_ENCODING_WANSUNG, 'w', 'a', 'n', 's' ),
    FT_ENC_TAG( FT_ENCODING_JOHAB,   'j', 'o', 'h', 'a' ),

    /* for backwards compatibility */
    FT_ENCODING_MS_SJIS    = FT_ENCODING_SJIS,
    FT_ENCODING_MS_GB2312  = FT_ENCODING_GB2312,
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


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    ft_encoding_xxx                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    These constants are deprecated; use the corresponding @FT_Encoding */
  /*    values instead.                                                    */
  /*                                                                       */
#define ft_encoding_none            FT_ENCODING_NONE
#define ft_encoding_unicode         FT_ENCODING_UNICODE
#define ft_encoding_symbol          FT_ENCODING_MS_SYMBOL
#define ft_encoding_latin_1         FT_ENCODING_ADOBE_LATIN_1
#define ft_encoding_latin_2         FT_ENCODING_OLD_LATIN_2
#define ft_encoding_sjis            FT_ENCODING_SJIS
#define ft_encoding_gb2312          FT_ENCODING_GB2312
#define ft_encoding_big5            FT_ENCODING_BIG5
#define ft_encoding_wansung         FT_ENCODING_WANSUNG
#define ft_encoding_johab           FT_ENCODING_JOHAB

#define ft_encoding_adobe_standard  FT_ENCODING_ADOBE_STANDARD
#define ft_encoding_adobe_expert    FT_ENCODING_ADOBE_EXPERT
#define ft_encoding_adobe_custom    FT_ENCODING_ADOBE_CUSTOM
#define ft_encoding_apple_roman     FT_ENCODING_APPLE_ROMAN


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_CharMapRec                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The base charmap structure.                                        */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    face        :: A handle to the parent face object.                 */
  /*                                                                       */
  /*    encoding    :: An @FT_Encoding tag identifying the charmap.  Use   */
  /*                   this with @FT_Select_Charmap.                       */
  /*                                                                       */
  /*    platform_id :: An ID number describing the platform for the        */
  /*                   following encoding ID.  This comes directly from    */
  /*                   the TrueType specification and should be emulated   */
  /*                   for other formats.                                  */
  /*                                                                       */
  /*    encoding_id :: A platform specific encoding number.  This also     */
  /*                   comes from the TrueType specification and should be */
  /*                   emulated similarly.                                 */
  /*                                                                       */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Face_Internal                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An opaque handle to an `FT_Face_InternalRec' structure, used to    */
  /*    model private data of a given @FT_Face object.                     */
  /*                                                                       */
  /*    This structure might change between releases of FreeType~2 and is  */
  /*    not generally available to client applications.                    */
  /*                                                                       */
  typedef struct FT_Face_InternalRec_*  FT_Face_Internal;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_FaceRec                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    FreeType root face class structure.  A face object models a        */
  /*    typeface in a font file.                                           */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_faces           :: The number of faces in the font file.  Some */
  /*                           font formats can have multiple faces in     */
  /*                           a font file.                                */
  /*                                                                       */
  /*    face_index          :: The index of the face in the font file.  It */
  /*                           is set to~0 if there is only one face in    */
  /*                           the font file.                              */
  /*                                                                       */
  /*    face_flags          :: A set of bit flags that give important      */
  /*                           information about the face; see             */
  /*                           @FT_FACE_FLAG_XXX for the details.          */
  /*                                                                       */
  /*    style_flags         :: A set of bit flags indicating the style of  */
  /*                           the face; see @FT_STYLE_FLAG_XXX for the    */
  /*                           details.                                    */
  /*                                                                       */
  /*    num_glyphs          :: The number of glyphs in the face.  If the   */
  /*                           face is scalable and has sbits (see         */
  /*                           `num_fixed_sizes'), it is set to the number */
  /*                           of outline glyphs.                          */
  /*                                                                       */
  /*                           For CID-keyed fonts, this value gives the   */
  /*                           highest CID used in the font.               */
  /*                                                                       */
  /*    family_name         :: The face's family name.  This is an ASCII   */
  /*                           string, usually in English, which describes */
  /*                           the typeface's family (like `Times New      */
  /*                           Roman', `Bodoni', `Garamond', etc).  This   */
  /*                           is a least common denominator used to list  */
  /*                           fonts.  Some formats (TrueType & OpenType)  */
  /*                           provide localized and Unicode versions of   */
  /*                           this string.  Applications should use the   */
  /*                           format specific interface to access them.   */
  /*                           Can be NULL (e.g., in fonts embedded in a   */
  /*                           PDF file).                                  */
  /*                                                                       */
  /*    style_name          :: The face's style name.  This is an ASCII    */
  /*                           string, usually in English, which describes */
  /*                           the typeface's style (like `Italic',        */
  /*                           `Bold', `Condensed', etc).  Not all font    */
  /*                           formats provide a style name, so this field */
  /*                           is optional, and can be set to NULL.  As    */
  /*                           for `family_name', some formats provide     */
  /*                           localized and Unicode versions of this      */
  /*                           string.  Applications should use the format */
  /*                           specific interface to access them.          */
  /*                                                                       */
  /*    num_fixed_sizes     :: The number of bitmap strikes in the face.   */
  /*                           Even if the face is scalable, there might   */
  /*                           still be bitmap strikes, which are called   */
  /*                           `sbits' in that case.                       */
  /*                                                                       */
  /*    available_sizes     :: An array of @FT_Bitmap_Size for all bitmap  */
  /*                           strikes in the face.  It is set to NULL if  */
  /*                           there is no bitmap strike.                  */
  /*                                                                       */
  /*    num_charmaps        :: The number of charmaps in the face.         */
  /*                                                                       */
  /*    charmaps            :: An array of the charmaps of the face.       */
  /*                                                                       */
  /*    generic             :: A field reserved for client uses.  See the  */
  /*                           @FT_Generic type description.               */
  /*                                                                       */
  /*    bbox                :: The font bounding box.  Coordinates are     */
  /*                           expressed in font units (see                */
  /*                           `units_per_EM').  The box is large enough   */
  /*                           to contain any glyph from the font.  Thus,  */
  /*                           `bbox.yMax' can be seen as the `maximum     */
  /*                           ascender', and `bbox.yMin' as the `minimum  */
  /*                           descender'.  Only relevant for scalable     */
  /*                           formats.                                    */
  /*                                                                       */
  /*                           Note that the bounding box might be off by  */
  /*                           (at least) one pixel for hinted fonts.  See */
  /*                           @FT_Size_Metrics for further discussion.    */
  /*                                                                       */
  /*    units_per_EM        :: The number of font units per EM square for  */
  /*                           this face.  This is typically 2048 for      */
  /*                           TrueType fonts, and 1000 for Type~1 fonts.  */
  /*                           Only relevant for scalable formats.         */
  /*                                                                       */
  /*    ascender            :: The typographic ascender of the face,       */
  /*                           expressed in font units.  For font formats  */
  /*                           not having this information, it is set to   */
  /*                           `bbox.yMax'.  Only relevant for scalable    */
  /*                           formats.                                    */
  /*                                                                       */
  /*    descender           :: The typographic descender of the face,      */
  /*                           expressed in font units.  For font formats  */
  /*                           not having this information, it is set to   */
  /*                           `bbox.yMin'.  Note that this field is       */
  /*                           usually negative.  Only relevant for        */
  /*                           scalable formats.                           */
  /*                                                                       */
  /*    height              :: This value is the vertical distance         */
  /*                           between two consecutive baselines,          */
  /*                           expressed in font units.  It is always      */
  /*                           positive.  Only relevant for scalable       */
  /*                           formats.                                    */
  /*                                                                       */
  /*                           If you want the global glyph height, use    */
  /*                           `ascender - descender'.                     */
  /*                                                                       */
  /*    max_advance_width   :: The maximum advance width, in font units,   */
  /*                           for all glyphs in this face.  This can be   */
  /*                           used to make word wrapping computations     */
  /*                           faster.  Only relevant for scalable         */
  /*                           formats.                                    */
  /*                                                                       */
  /*    max_advance_height  :: The maximum advance height, in font units,  */
  /*                           for all glyphs in this face.  This is only  */
  /*                           relevant for vertical layouts, and is set   */
  /*                           to `height' for fonts that do not provide   */
  /*                           vertical metrics.  Only relevant for        */
  /*                           scalable formats.                           */
  /*                                                                       */
  /*    underline_position  :: The position, in font units, of the         */
  /*                           underline line for this face.  It is the    */
  /*                           center of the underlining stem.  Only       */
  /*                           relevant for scalable formats.              */
  /*                                                                       */
  /*    underline_thickness :: The thickness, in font units, of the        */
  /*                           underline for this face.  Only relevant for */
  /*                           scalable formats.                           */
  /*                                                                       */
  /*    glyph               :: The face's associated glyph slot(s).        */
  /*                                                                       */
  /*    size                :: The current active size for this face.      */
  /*                                                                       */
  /*    charmap             :: The current active charmap for this face.   */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Fields may be changed after a call to @FT_Attach_File or           */
  /*    @FT_Attach_Stream.                                                 */
  /*                                                                       */
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

    /*# The following member variables (down to `underline_thickness') */
    /*# are only relevant to scalable outlines; cf. @FT_Bitmap_Size    */
    /*# for bitmap fonts.                                              */
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

    /*@private begin */

    FT_Driver         driver;
    FT_Memory         memory;
    FT_Stream         stream;

    FT_ListRec        sizes_list;

    FT_Generic        autohint;   /* face-specific auto-hinter data */
    void*             extensions; /* unused                         */

    FT_Face_Internal  internal;

    /*@private end */

  } FT_FaceRec;


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_FACE_FLAG_XXX                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit flags used in the `face_flags' field of the          */
  /*    @FT_FaceRec structure.  They inform client applications of         */
  /*    properties of the corresponding face.                              */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_FACE_FLAG_SCALABLE ::                                           */
  /*      Indicates that the face contains outline glyphs.  This doesn't   */
  /*      prevent bitmap strikes, i.e., a face can have both this and      */
  /*      and @FT_FACE_FLAG_FIXED_SIZES set.                               */
  /*                                                                       */
  /*    FT_FACE_FLAG_FIXED_SIZES ::                                        */
  /*      Indicates that the face contains bitmap strikes.  See also the   */
  /*      `num_fixed_sizes' and `available_sizes' fields of @FT_FaceRec.   */
  /*                                                                       */
  /*    FT_FACE_FLAG_FIXED_WIDTH ::                                        */
  /*      Indicates that the face contains fixed-width characters (like    */
  /*      Courier, Lucido, MonoType, etc.).                                */
  /*                                                                       */
  /*    FT_FACE_FLAG_SFNT ::                                               */
  /*      Indicates that the face uses the `sfnt' storage scheme.  For     */
  /*      now, this means TrueType and OpenType.                           */
  /*                                                                       */
  /*    FT_FACE_FLAG_HORIZONTAL ::                                         */
  /*      Indicates that the face contains horizontal glyph metrics.  This */
  /*      should be set for all common formats.                            */
  /*                                                                       */
  /*    FT_FACE_FLAG_VERTICAL ::                                           */
  /*      Indicates that the face contains vertical glyph metrics.  This   */
  /*      is only available in some formats, not all of them.              */
  /*                                                                       */
  /*    FT_FACE_FLAG_KERNING ::                                            */
  /*      Indicates that the face contains kerning information.  If set,   */
  /*      the kerning distance can be retrieved through the function       */
  /*      @FT_Get_Kerning.  Otherwise the function always return the       */
  /*      vector (0,0).  Note that FreeType doesn't handle kerning data    */
  /*      from the `GPOS' table (as present in some OpenType fonts).       */
  /*                                                                       */
  /*    FT_FACE_FLAG_FAST_GLYPHS ::                                        */
  /*      THIS FLAG IS DEPRECATED.  DO NOT USE OR TEST IT.                 */
  /*                                                                       */
  /*    FT_FACE_FLAG_MULTIPLE_MASTERS ::                                   */
  /*      Indicates that the font contains multiple masters and is capable */
  /*      of interpolating between them.  See the multiple-masters         */
  /*      specific API for details.                                        */
  /*                                                                       */
  /*    FT_FACE_FLAG_GLYPH_NAMES ::                                        */
  /*      Indicates that the font contains glyph names that can be         */
  /*      retrieved through @FT_Get_Glyph_Name.  Note that some TrueType   */
  /*      fonts contain broken glyph name tables.  Use the function        */
  /*      @FT_Has_PS_Glyph_Names when needed.                              */
  /*                                                                       */
  /*    FT_FACE_FLAG_EXTERNAL_STREAM ::                                    */
  /*      Used internally by FreeType to indicate that a face's stream was */
  /*      provided by the client application and should not be destroyed   */
  /*      when @FT_Done_Face is called.  Don't read or test this flag.     */
  /*                                                                       */
  /*    FT_FACE_FLAG_HINTER ::                                             */
  /*      Set if the font driver has a hinting machine of its own.  For    */
  /*      example, with TrueType fonts, it makes sense to use data from    */
  /*      the SFNT `gasp' table only if the native TrueType hinting engine */
  /*      (with the bytecode interpreter) is available and active.         */
  /*                                                                       */
  /*    FT_FACE_FLAG_CID_KEYED ::                                          */
  /*      Set if the font is CID-keyed.  In that case, the font is not     */
  /*      accessed by glyph indices but by CID values.  For subsetted      */
  /*      CID-keyed fonts this has the consequence that not all index      */
  /*      values are a valid argument to FT_Load_Glyph.  Only the CID      */
  /*      values for which corresponding glyphs in the subsetted font      */
  /*      exist make FT_Load_Glyph return successfully; in all other cases */
  /*      you get an `FT_Err_Invalid_Argument' error.                      */
  /*                                                                       */
  /*      Note that CID-keyed fonts which are in an SFNT wrapper don't     */
  /*      have this flag set since the glyphs are accessed in the normal   */
  /*      way (using contiguous indices); the `CID-ness' isn't visible to  */
  /*      the application.                                                 */
  /*                                                                       */
  /*    FT_FACE_FLAG_TRICKY ::                                             */
  /*      Set if the font is `tricky', this is, it always needs the        */
  /*      font format's native hinting engine to get a reasonable result.  */
  /*      A typical example is the Chinese font `mingli.ttf' which uses    */
  /*      TrueType bytecode instructions to move and scale all of its      */
  /*      subglyphs.                                                       */
  /*                                                                       */
  /*      It is not possible to autohint such fonts using                  */
  /*      @FT_LOAD_FORCE_AUTOHINT; it will also ignore                     */
  /*      @FT_LOAD_NO_HINTING.  You have to set both @FT_LOAD_NO_HINTING   */
  /*      and @FT_LOAD_NO_AUTOHINT to really disable hinting; however, you */
  /*      probably never want this except for demonstration purposes.      */
  /*                                                                       */
  /*      Currently, there are about a dozen TrueType fonts in the list of */
  /*      tricky fonts; they are hard-coded in file `ttobjs.c'.            */
  /*                                                                       */
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


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_HORIZONTAL( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains
   *   horizontal metrics (this is true for all font formats though).
   *
   * @also:
   *   @FT_HAS_VERTICAL can be used to check for vertical metrics.
   *
   */
#define FT_HAS_HORIZONTAL( face ) \
          ( face->face_flags & FT_FACE_FLAG_HORIZONTAL )


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_VERTICAL( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains real
   *   vertical metrics (and not only synthesized ones).
   *
   */
#define FT_HAS_VERTICAL( face ) \
          ( face->face_flags & FT_FACE_FLAG_VERTICAL )


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_KERNING( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains kerning
   *   data that can be accessed with @FT_Get_Kerning.
   *
   */
#define FT_HAS_KERNING( face ) \
          ( face->face_flags & FT_FACE_FLAG_KERNING )


  /*************************************************************************
   *
   * @macro:
   *   FT_IS_SCALABLE( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains a scalable
   *   font face (true for TrueType, Type~1, Type~42, CID, OpenType/CFF,
   *   and PFR font formats.
   *
   */
#define FT_IS_SCALABLE( face ) \
          ( face->face_flags & FT_FACE_FLAG_SCALABLE )


  /*************************************************************************
   *
   * @macro:
   *   FT_IS_SFNT( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains a font
   *   whose format is based on the SFNT storage scheme.  This usually
   *   means: TrueType fonts, OpenType fonts, as well as SFNT-based embedded
   *   bitmap fonts.
   *
   *   If this macro is true, all functions defined in @FT_SFNT_NAMES_H and
   *   @FT_TRUETYPE_TABLES_H are available.
   *
   */
#define FT_IS_SFNT( face ) \
          ( face->face_flags & FT_FACE_FLAG_SFNT )


  /*************************************************************************
   *
   * @macro:
   *   FT_IS_FIXED_WIDTH( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains a font face
   *   that contains fixed-width (or `monospace', `fixed-pitch', etc.)
   *   glyphs.
   *
   */
#define FT_IS_FIXED_WIDTH( face ) \
          ( face->face_flags & FT_FACE_FLAG_FIXED_WIDTH )


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_FIXED_SIZES( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains some
   *   embedded bitmaps.  See the `available_sizes' field of the
   *   @FT_FaceRec structure.
   *
   */
#define FT_HAS_FIXED_SIZES( face ) \
          ( face->face_flags & FT_FACE_FLAG_FIXED_SIZES )


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_FAST_GLYPHS( face )
   *
   * @description:
   *   Deprecated.
   *
   */
#define FT_HAS_FAST_GLYPHS( face )  0


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_GLYPH_NAMES( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains some glyph
   *   names that can be accessed through @FT_Get_Glyph_Name.
   *
   */
#define FT_HAS_GLYPH_NAMES( face ) \
          ( face->face_flags & FT_FACE_FLAG_GLYPH_NAMES )


  /*************************************************************************
   *
   * @macro:
   *   FT_HAS_MULTIPLE_MASTERS( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains some
   *   multiple masters.  The functions provided by @FT_MULTIPLE_MASTERS_H
   *   are then available to choose the exact design you want.
   *
   */
#define FT_HAS_MULTIPLE_MASTERS( face ) \
          ( face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS )


  /*************************************************************************
   *
   * @macro:
   *   FT_IS_CID_KEYED( face )
   *
   * @description:
   *   A macro that returns true whenever a face object contains a CID-keyed
   *   font.  See the discussion of @FT_FACE_FLAG_CID_KEYED for more
   *   details.
   *
   *   If this macro is true, all functions defined in @FT_CID_H are
   *   available.
   *
   */
#define FT_IS_CID_KEYED( face ) \
          ( face->face_flags & FT_FACE_FLAG_CID_KEYED )


  /*************************************************************************
   *
   * @macro:
   *   FT_IS_TRICKY( face )
   *
   * @description:
   *   A macro that returns true whenever a face represents a `tricky' font.
   *   See the discussion of @FT_FACE_FLAG_TRICKY for more details.
   *
   */
#define FT_IS_TRICKY( face ) \
          ( face->face_flags & FT_FACE_FLAG_TRICKY )


  /*************************************************************************/
  /*                                                                       */
  /* <Const>                                                               */
  /*    FT_STYLE_FLAG_XXX                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit-flags used to indicate the style of a given face.    */
  /*    These are used in the `style_flags' field of @FT_FaceRec.          */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_STYLE_FLAG_ITALIC ::                                            */
  /*      Indicates that a given face style is italic or oblique.          */
  /*                                                                       */
  /*    FT_STYLE_FLAG_BOLD ::                                              */
  /*      Indicates that a given face is bold.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The style information as provided by FreeType is very basic.  More */
  /*    details are beyond the scope and should be done on a higher level  */
  /*    (for example, by analyzing various fields of the `OS/2' table in   */
  /*    SFNT based fonts).                                                 */
  /*                                                                       */
#define FT_STYLE_FLAG_ITALIC  ( 1 << 0 )
#define FT_STYLE_FLAG_BOLD    ( 1 << 1 )


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Size_Internal                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An opaque handle to an `FT_Size_InternalRec' structure, used to    */
  /*    model private data of a given @FT_Size object.                     */
  /*                                                                       */
  typedef struct FT_Size_InternalRec_*  FT_Size_Internal;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Size_Metrics                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The size metrics structure gives the metrics of a size object.     */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    x_ppem       :: The width of the scaled EM square in pixels, hence */
  /*                    the term `ppem' (pixels per EM).  It is also       */
  /*                    referred to as `nominal width'.                    */
  /*                                                                       */
  /*    y_ppem       :: The height of the scaled EM square in pixels,      */
  /*                    hence the term `ppem' (pixels per EM).  It is also */
  /*                    referred to as `nominal height'.                   */
  /*                                                                       */
  /*    x_scale      :: A 16.16 fractional scaling value used to convert   */
  /*                    horizontal metrics from font units to 26.6         */
  /*                    fractional pixels.  Only relevant for scalable     */
  /*                    font formats.                                      */
  /*                                                                       */
  /*    y_scale      :: A 16.16 fractional scaling value used to convert   */
  /*                    vertical metrics from font units to 26.6           */
  /*                    fractional pixels.  Only relevant for scalable     */
  /*                    font formats.                                      */
  /*                                                                       */
  /*    ascender     :: The ascender in 26.6 fractional pixels.  See       */
  /*                    @FT_FaceRec for the details.                       */
  /*                                                                       */
  /*    descender    :: The descender in 26.6 fractional pixels.  See      */
  /*                    @FT_FaceRec for the details.                       */
  /*                                                                       */
  /*    height       :: The height in 26.6 fractional pixels.  See         */
  /*                    @FT_FaceRec for the details.                       */
  /*                                                                       */
  /*    max_advance  :: The maximum advance width in 26.6 fractional       */
  /*                    pixels.  See @FT_FaceRec for the details.          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The scaling values, if relevant, are determined first during a     */
  /*    size changing operation.  The remaining fields are then set by the */
  /*    driver.  For scalable formats, they are usually set to scaled      */
  /*    values of the corresponding fields in @FT_FaceRec.                 */
  /*                                                                       */
  /*    Note that due to glyph hinting, these values might not be exact    */
  /*    for certain fonts.  Thus they must be treated as unreliable        */
  /*    with an error margin of at least one pixel!                        */
  /*                                                                       */
  /*    Indeed, the only way to get the exact metrics is to render _all_   */
  /*    glyphs.  As this would be a definite performance hit, it is up to  */
  /*    client applications to perform such computations.                  */
  /*                                                                       */
  /*    The FT_Size_Metrics structure is valid for bitmap fonts also.      */
  /*                                                                       */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_SizeRec                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    FreeType root size class structure.  A size object models a face   */
  /*    object at a given size.                                            */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    face    :: Handle to the parent face object.                       */
  /*                                                                       */
  /*    generic :: A typeless pointer, which is unused by the FreeType     */
  /*               library or any of its drivers.  It can be used by       */
  /*               client applications to link their own data to each size */
  /*               object.                                                 */
  /*                                                                       */
  /*    metrics :: Metrics for this size object.  This field is read-only. */
  /*                                                                       */
  typedef struct  FT_SizeRec_
  {
    FT_Face           face;      /* parent face object              */
    FT_Generic        generic;   /* generic pointer for client uses */
    FT_Size_Metrics   metrics;   /* size metrics                    */
    FT_Size_Internal  internal;

  } FT_SizeRec;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_SubGlyph                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The subglyph structure is an internal object used to describe      */
  /*    subglyphs (for example, in the case of composites).                */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The subglyph implementation is not part of the high-level API,     */
  /*    hence the forward structure declaration.                           */
  /*                                                                       */
  /*    You can however retrieve subglyph information with                 */
  /*    @FT_Get_SubGlyph_Info.                                             */
  /*                                                                       */
  typedef struct FT_SubGlyphRec_*  FT_SubGlyph;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Slot_Internal                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An opaque handle to an `FT_Slot_InternalRec' structure, used to    */
  /*    model private data of a given @FT_GlyphSlot object.                */
  /*                                                                       */
  typedef struct FT_Slot_InternalRec_*  FT_Slot_Internal;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_GlyphSlotRec                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    FreeType root glyph slot class structure.  A glyph slot is a       */
  /*    container where individual glyphs can be loaded, be they in        */
  /*    outline or bitmap format.                                          */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    library           :: A handle to the FreeType library instance     */
  /*                         this slot belongs to.                         */
  /*                                                                       */
  /*    face              :: A handle to the parent face object.           */
  /*                                                                       */
  /*    next              :: In some cases (like some font tools), several */
  /*                         glyph slots per face object can be a good     */
  /*                         thing.  As this is rare, the glyph slots are  */
  /*                         listed through a direct, single-linked list   */
  /*                         using its `next' field.                       */
  /*                                                                       */
  /*    generic           :: A typeless pointer which is unused by the     */
  /*                         FreeType library or any of its drivers.  It   */
  /*                         can be used by client applications to link    */
  /*                         their own data to each glyph slot object.     */
  /*                                                                       */
  /*    metrics           :: The metrics of the last loaded glyph in the   */
  /*                         slot.  The returned values depend on the last */
  /*                         load flags (see the @FT_Load_Glyph API        */
  /*                         function) and can be expressed either in 26.6 */
  /*                         fractional pixels or font units.              */
  /*                                                                       */
  /*                         Note that even when the glyph image is        */
  /*                         transformed, the metrics are not.             */
  /*                                                                       */
  /*    linearHoriAdvance :: The advance width of the unhinted glyph.      */
  /*                         Its value is expressed in 16.16 fractional    */
  /*                         pixels, unless @FT_LOAD_LINEAR_DESIGN is set  */
  /*                         when loading the glyph.  This field can be    */
  /*                         important to perform correct WYSIWYG layout.  */
  /*                         Only relevant for outline glyphs.             */
  /*                                                                       */
  /*    linearVertAdvance :: The advance height of the unhinted glyph.     */
  /*                         Its value is expressed in 16.16 fractional    */
  /*                         pixels, unless @FT_LOAD_LINEAR_DESIGN is set  */
  /*                         when loading the glyph.  This field can be    */
  /*                         important to perform correct WYSIWYG layout.  */
  /*                         Only relevant for outline glyphs.             */
  /*                                                                       */
  /*    advance           :: This shorthand is, depending on               */
  /*                         @FT_LOAD_IGNORE_TRANSFORM, the transformed    */
  /*                         advance width for the glyph (in 26.6          */
  /*                         fractional pixel format).  As specified with  */
  /*                         @FT_LOAD_VERTICAL_LAYOUT, it uses either the  */
  /*                         `horiAdvance' or the `vertAdvance' value of   */
  /*                         `metrics' field.                              */
  /*                                                                       */
  /*    format            :: This field indicates the format of the image  */
  /*                         contained in the glyph slot.  Typically       */
  /*                         @FT_GLYPH_FORMAT_BITMAP,                      */
  /*                         @FT_GLYPH_FORMAT_OUTLINE, or                  */
  /*                         @FT_GLYPH_FORMAT_COMPOSITE, but others are    */
  /*                         possible.                                     */
  /*                                                                       */
  /*    bitmap            :: This field is used as a bitmap descriptor     */
  /*                         when the slot format is                       */
  /*                         @FT_GLYPH_FORMAT_BITMAP.  Note that the       */
  /*                         address and content of the bitmap buffer can  */
  /*                         change between calls of @FT_Load_Glyph and a  */
  /*                         few other functions.                          */
  /*                                                                       */
  /*    bitmap_left       :: This is the bitmap's left bearing expressed   */
  /*                         in integer pixels.  Of course, this is only   */
  /*                         valid if the format is                        */
  /*                         @FT_GLYPH_FORMAT_BITMAP.                      */
  /*                                                                       */
  /*    bitmap_top        :: This is the bitmap's top bearing expressed in */
  /*                         integer pixels.  Remember that this is the    */
  /*                         distance from the baseline to the top-most    */
  /*                         glyph scanline, upwards y~coordinates being   */
  /*                         *positive*.                                   */
  /*                                                                       */
  /*    outline           :: The outline descriptor for the current glyph  */
  /*                         image if its format is                        */
  /*                         @FT_GLYPH_FORMAT_OUTLINE.  Once a glyph is    */
  /*                         loaded, `outline' can be transformed,         */
  /*                         distorted, embolded, etc.  However, it must   */
  /*                         not be freed.                                 */
  /*                                                                       */
  /*    num_subglyphs     :: The number of subglyphs in a composite glyph. */
  /*                         This field is only valid for the composite    */
  /*                         glyph format that should normally only be     */
  /*                         loaded with the @FT_LOAD_NO_RECURSE flag.     */
  /*                         For now this is internal to FreeType.         */
  /*                                                                       */
  /*    subglyphs         :: An array of subglyph descriptors for          */
  /*                         composite glyphs.  There are `num_subglyphs'  */
  /*                         elements in there.  Currently internal to     */
  /*                         FreeType.                                     */
  /*                                                                       */
  /*    control_data      :: Certain font drivers can also return the      */
  /*                         control data for a given glyph image (e.g.    */
  /*                         TrueType bytecode, Type~1 charstrings, etc.). */
  /*                         This field is a pointer to such data.         */
  /*                                                                       */
  /*    control_len       :: This is the length in bytes of the control    */
  /*                         data.                                         */
  /*                                                                       */
  /*    other             :: Really wicked formats can use this pointer to */
  /*                         present their own glyph image to client       */
  /*                         applications.  Note that the application      */
  /*                         needs to know about the image format.         */
  /*                                                                       */
  /*    lsb_delta         :: The difference between hinted and unhinted    */
  /*                         left side bearing while autohinting is        */
  /*                         active.  Zero otherwise.                      */
  /*                                                                       */
  /*    rsb_delta         :: The difference between hinted and unhinted    */
  /*                         right side bearing while autohinting is       */
  /*                         active.  Zero otherwise.                      */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If @FT_Load_Glyph is called with default flags (see                */
  /*    @FT_LOAD_DEFAULT) the glyph image is loaded in the glyph slot in   */
  /*    its native format (e.g., an outline glyph for TrueType and Type~1  */
  /*    formats).                                                          */
  /*                                                                       */
  /*    This image can later be converted into a bitmap by calling         */
  /*    @FT_Render_Glyph.  This function finds the current renderer for    */
  /*    the native image's format, then invokes it.                        */
  /*                                                                       */
  /*    The renderer is in charge of transforming the native image through */
  /*    the slot's face transformation fields, then converting it into a   */
  /*    bitmap that is returned in `slot->bitmap'.                         */
  /*                                                                       */
  /*    Note that `slot->bitmap_left' and `slot->bitmap_top' are also used */
  /*    to specify the position of the bitmap relative to the current pen  */
  /*    position (e.g., coordinates (0,0) on the baseline).  Of course,    */
  /*    `slot->format' is also changed to @FT_GLYPH_FORMAT_BITMAP.         */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Here a small pseudo code fragment which shows how to use           */
  /*    `lsb_delta' and `rsb_delta':                                       */
  /*                                                                       */
  /*    {                                                                  */
  /*      FT_Pos  origin_x       = 0;                                      */
  /*      FT_Pos  prev_rsb_delta = 0;                                      */
  /*                                                                       */
  /*                                                                       */
  /*      for all glyphs do                                                */
  /*        <compute kern between current and previous glyph and add it to */
  /*         `origin_x'>                                                   */
  /*                                                                       */
  /*        <load glyph with `FT_Load_Glyph'>                              */
  /*                                                                       */
  /*        if ( prev_rsb_delta - face->glyph->lsb_delta >= 32 )           */
  /*          origin_x -= 64;                                              */
  /*        else if ( prev_rsb_delta - face->glyph->lsb_delta < -32 )      */
  /*          origin_x += 64;                                              */
  /*                                                                       */
  /*        prev_rsb_delta = face->glyph->rsb_delta;                       */
  /*                                                                       */
  /*        <save glyph image, or render glyph, or ...>                    */
  /*                                                                       */
  /*        origin_x += face->glyph->advance.x;                            */
  /*      endfor                                                           */
  /*    }                                                                  */
  /*                                                                       */
  typedef struct  FT_GlyphSlotRec_
  {
    FT_Library        library;
    FT_Face           face;
    FT_GlyphSlot      next;
    FT_UInt           reserved;       /* retained for binary compatibility */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Init_FreeType                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Initialize a new FreeType library object.  The set of modules      */
  /*    that are registered by this function is determined at build time.  */
  /*                                                                       */
  /* <Output>                                                              */
  /*    alibrary :: A handle to a new library object.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    In case you want to provide your own memory allocating routines,   */
  /*    use @FT_New_Library instead, followed by a call to                 */
  /*    @FT_Add_Default_Modules (or a series of calls to @FT_Add_Module).  */
  /*                                                                       */
  /*    For multi-threading applications each thread should have its own   */
  /*    FT_Library object.                                                 */
  /*                                                                       */
  /*    If you need reference-counting (cf. @FT_Reference_Library), use    */
  /*    @FT_New_Library and @FT_Done_Library.                              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Init_FreeType( FT_Library  *alibrary );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Done_FreeType                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Destroy a given FreeType library object and all of its children,   */
  /*    including resources, drivers, faces, sizes, etc.                   */
  /*                                                                       */
  /* <Input>                                                               */
  /*    library :: A handle to the target library object.                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Done_FreeType( FT_Library  library );


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_OPEN_XXX                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit-field constants used within the `flags' field of the */
  /*    @FT_Open_Args structure.                                           */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_OPEN_MEMORY   :: This is a memory-based stream.                 */
  /*                                                                       */
  /*    FT_OPEN_STREAM   :: Copy the stream from the `stream' field.       */
  /*                                                                       */
  /*    FT_OPEN_PATHNAME :: Create a new input stream from a C~path        */
  /*                        name.                                          */
  /*                                                                       */
  /*    FT_OPEN_DRIVER   :: Use the `driver' field.                        */
  /*                                                                       */
  /*    FT_OPEN_PARAMS   :: Use the `num_params' and `params' fields.      */
  /*                                                                       */
  /*    ft_open_memory   :: Deprecated; use @FT_OPEN_MEMORY instead.       */
  /*                                                                       */
  /*    ft_open_stream   :: Deprecated; use @FT_OPEN_STREAM instead.       */
  /*                                                                       */
  /*    ft_open_pathname :: Deprecated; use @FT_OPEN_PATHNAME instead.     */
  /*                                                                       */
  /*    ft_open_driver   :: Deprecated; use @FT_OPEN_DRIVER instead.       */
  /*                                                                       */
  /*    ft_open_params   :: Deprecated; use @FT_OPEN_PARAMS instead.       */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The `FT_OPEN_MEMORY', `FT_OPEN_STREAM', and `FT_OPEN_PATHNAME'     */
  /*    flags are mutually exclusive.                                      */
  /*                                                                       */
#define FT_OPEN_MEMORY    0x1
#define FT_OPEN_STREAM    0x2
#define FT_OPEN_PATHNAME  0x4
#define FT_OPEN_DRIVER    0x8
#define FT_OPEN_PARAMS    0x10

#define ft_open_memory    FT_OPEN_MEMORY     /* deprecated */
#define ft_open_stream    FT_OPEN_STREAM     /* deprecated */
#define ft_open_pathname  FT_OPEN_PATHNAME   /* deprecated */
#define ft_open_driver    FT_OPEN_DRIVER     /* deprecated */
#define ft_open_params    FT_OPEN_PARAMS     /* deprecated */


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Parameter                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A simple structure used to pass more or less generic parameters to */
  /*    @FT_Open_Face.                                                     */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    tag  :: A four-byte identification tag.                            */
  /*                                                                       */
  /*    data :: A pointer to the parameter data.                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The ID and function of parameters are driver-specific.  See the    */
  /*    various FT_PARAM_TAG_XXX flags for more information.               */
  /*                                                                       */
  typedef struct  FT_Parameter_
  {
    FT_ULong    tag;
    FT_Pointer  data;

  } FT_Parameter;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Open_Args                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to indicate how to open a new font file or        */
  /*    stream.  A pointer to such a structure can be used as a parameter  */
  /*    for the functions @FT_Open_Face and @FT_Attach_Stream.             */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    flags       :: A set of bit flags indicating how to use the        */
  /*                   structure.                                          */
  /*                                                                       */
  /*    memory_base :: The first byte of the file in memory.               */
  /*                                                                       */
  /*    memory_size :: The size in bytes of the file in memory.            */
  /*                                                                       */
  /*    pathname    :: A pointer to an 8-bit file pathname.                */
  /*                                                                       */
  /*    stream      :: A handle to a source stream object.                 */
  /*                                                                       */
  /*    driver      :: This field is exclusively used by @FT_Open_Face;    */
  /*                   it simply specifies the font driver to use to open  */
  /*                   the face.  If set to~0, FreeType tries to load the  */
  /*                   face with each one of the drivers in its list.      */
  /*                                                                       */
  /*    num_params  :: The number of extra parameters.                     */
  /*                                                                       */
  /*    params      :: Extra parameters passed to the font driver when     */
  /*                   opening a new face.                                 */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The stream type is determined by the contents of `flags' which     */
  /*    are tested in the following order by @FT_Open_Face:                */
  /*                                                                       */
  /*    If the `FT_OPEN_MEMORY' bit is set, assume that this is a          */
  /*    memory file of `memory_size' bytes, located at `memory_address'.   */
  /*    The data are are not copied, and the client is responsible for     */
  /*    releasing and destroying them _after_ the corresponding call to    */
  /*    @FT_Done_Face.                                                     */
  /*                                                                       */
  /*    Otherwise, if the `FT_OPEN_STREAM' bit is set, assume that a       */
  /*    custom input stream `stream' is used.                              */
  /*                                                                       */
  /*    Otherwise, if the `FT_OPEN_PATHNAME' bit is set, assume that this  */
  /*    is a normal file and use `pathname' to open it.                    */
  /*                                                                       */
  /*    If the `FT_OPEN_DRIVER' bit is set, @FT_Open_Face only tries to    */
  /*    open the file with the driver whose handler is in `driver'.        */
  /*                                                                       */
  /*    If the `FT_OPEN_PARAMS' bit is set, the parameters given by        */
  /*    `num_params' and `params' is used.  They are ignored otherwise.    */
  /*                                                                       */
  /*    Ideally, both the `pathname' and `params' fields should be tagged  */
  /*    as `const'; this is missing for API backwards compatibility.  In   */
  /*    other words, applications should treat them as read-only.          */
  /*                                                                       */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_New_Face                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function calls @FT_Open_Face to open a font by its pathname.  */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    library    :: A handle to the library resource.                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    pathname   :: A path to the font file.                             */
  /*                                                                       */
  /*    face_index :: The index of the face within the font.  The first    */
  /*                  face has index~0.                                    */
  /*                                                                       */
  /* <Output>                                                              */
  /*    aface      :: A handle to a new face object.  If `face_index' is   */
  /*                  greater than or equal to zero, it must be non-NULL.  */
  /*                  See @FT_Open_Face for more details.                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Use @FT_Done_Face to destroy the created @FT_Face object (along    */
  /*    with its slot and sizes).                                          */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_New_Face( FT_Library   library,
               const char*  filepathname,
               FT_Long      face_index,
               FT_Face     *aface );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_New_Memory_Face                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function calls @FT_Open_Face to open a font which has been    */
  /*    loaded into memory.                                                */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    library    :: A handle to the library resource.                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    file_base  :: A pointer to the beginning of the font data.         */
  /*                                                                       */
  /*    file_size  :: The size of the memory chunk used by the font data.  */
  /*                                                                       */
  /*    face_index :: The index of the face within the font.  The first    */
  /*                  face has index~0.                                    */
  /*                                                                       */
  /* <Output>                                                              */
  /*    aface      :: A handle to a new face object.  If `face_index' is   */
  /*                  greater than or equal to zero, it must be non-NULL.  */
  /*                  See @FT_Open_Face for more details.                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    You must not deallocate the memory before calling @FT_Done_Face.   */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_New_Memory_Face( FT_Library      library,
                      const FT_Byte*  file_base,
                      FT_Long         file_size,
                      FT_Long         face_index,
                      FT_Face        *aface );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Open_Face                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Create a face object from a given resource described by            */
  /*    @FT_Open_Args.                                                     */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    library    :: A handle to the library resource.                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    args       :: A pointer to an `FT_Open_Args' structure which must  */
  /*                  be filled by the caller.                             */
  /*                                                                       */
  /*    face_index :: The index of the face within the font.  The first    */
  /*                  face has index~0.                                    */
  /*                                                                       */
  /* <Output>                                                              */
  /*    aface      :: A handle to a new face object.  If `face_index' is   */
  /*                  greater than or equal to zero, it must be non-NULL.  */
  /*                  See note below.                                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Unlike FreeType 1.x, this function automatically creates a glyph   */
  /*    slot for the face object which can be accessed directly through    */
  /*    `face->glyph'.                                                     */
  /*                                                                       */
  /*    FT_Open_Face can be used to quickly check whether the font         */
  /*    format of a given font resource is supported by FreeType.  If the  */
  /*    `face_index' field is negative, the function's return value is~0   */
  /*    if the font format is recognized, or non-zero otherwise;           */
  /*    the function returns a more or less empty face handle in `*aface'  */
  /*    (if `aface' isn't NULL).  The only useful field in this special    */
  /*    case is `face->num_faces' which gives the number of faces within   */
  /*    the font file.  After examination, the returned @FT_Face structure */
  /*    should be deallocated with a call to @FT_Done_Face.                */
  /*                                                                       */
  /*    Each new face object created with this function also owns a        */
  /*    default @FT_Size object, accessible as `face->size'.               */
  /*                                                                       */
  /*    One @FT_Library instance can have multiple face objects, this is,  */
  /*    @FT_Open_Face and its siblings can be called multiple times using  */
  /*    the same `library' argument.                                       */
  /*                                                                       */
  /*    See the discussion of reference counters in the description of     */
  /*    @FT_Reference_Face.                                                */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Open_Face( FT_Library           library,
                const FT_Open_Args*  args,
                FT_Long              face_index,
                FT_Face             *aface );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Attach_File                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function calls @FT_Attach_Stream to attach a file.            */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face         :: The target face object.                            */
  /*                                                                       */
  /* <Input>                                                               */
  /*    filepathname :: The pathname.                                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Attach_File( FT_Face      face,
                  const char*  filepathname );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Attach_Stream                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    `Attach' data to a face object.  Normally, this is used to read    */
  /*    additional information for the face object.  For example, you can  */
  /*    attach an AFM file that comes with a Type~1 font to get the        */
  /*    kerning values and other metrics.                                  */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: The target face object.                              */
  /*                                                                       */
  /* <Input>                                                               */
  /*    parameters :: A pointer to @FT_Open_Args which must be filled by   */
  /*                  the caller.                                          */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The meaning of the `attach' (i.e., what really happens when the    */
  /*    new file is read) is not fixed by FreeType itself.  It really      */
  /*    depends on the font format (and thus the font driver).             */
  /*                                                                       */
  /*    Client applications are expected to know what they are doing       */
  /*    when invoking this function.  Most drivers simply do not implement */
  /*    file attachments.                                                  */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Attach_Stream( FT_Face        face,
                    FT_Open_Args*  parameters );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Reference_Face                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A counter gets initialized to~1 at the time an @FT_Face structure  */
  /*    is created.  This function increments the counter.  @FT_Done_Face  */
  /*    then only destroys a face if the counter is~1, otherwise it simply */
  /*    decrements the counter.                                            */
  /*                                                                       */
  /*    This function helps in managing life-cycles of structures which    */
  /*    reference @FT_Face objects.                                        */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A handle to a target face object.                          */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.4.2                                                              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Reference_Face( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Done_Face                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Discard a given face object, as well as all of its child slots and */
  /*    sizes.                                                             */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A handle to a target face object.                          */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    See the discussion of reference counters in the description of     */
  /*    @FT_Reference_Face.                                                */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Done_Face( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Select_Size                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Select a bitmap strike.                                            */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face         :: A handle to a target face object.                  */
  /*                                                                       */
  /* <Input>                                                               */
  /*    strike_index :: The index of the bitmap strike in the              */
  /*                    `available_sizes' field of @FT_FaceRec structure.  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Select_Size( FT_Face  face,
                  FT_Int   strike_index );


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_Size_Request_Type                                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An enumeration type that lists the supported size request types.   */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_SIZE_REQUEST_TYPE_NOMINAL ::                                    */
  /*      The nominal size.  The `units_per_EM' field of @FT_FaceRec is    */
  /*      used to determine both scaling values.                           */
  /*                                                                       */
  /*    FT_SIZE_REQUEST_TYPE_REAL_DIM ::                                   */
  /*      The real dimension.  The sum of the the `ascender' and (minus    */
  /*      of) the `descender' fields of @FT_FaceRec are used to determine  */
  /*      both scaling values.                                             */
  /*                                                                       */
  /*    FT_SIZE_REQUEST_TYPE_BBOX ::                                       */
  /*      The font bounding box.  The width and height of the `bbox' field */
  /*      of @FT_FaceRec are used to determine the horizontal and vertical */
  /*      scaling value, respectively.                                     */
  /*                                                                       */
  /*    FT_SIZE_REQUEST_TYPE_CELL ::                                       */
  /*      The `max_advance_width' field of @FT_FaceRec is used to          */
  /*      determine the horizontal scaling value; the vertical scaling     */
  /*      value is determined the same way as                              */
  /*      @FT_SIZE_REQUEST_TYPE_REAL_DIM does.  Finally, both scaling      */
  /*      values are set to the smaller one.  This type is useful if you   */
  /*      want to specify the font size for, say, a window of a given      */
  /*      dimension and 80x24 cells.                                       */
  /*                                                                       */
  /*    FT_SIZE_REQUEST_TYPE_SCALES ::                                     */
  /*      Specify the scaling values directly.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The above descriptions only apply to scalable formats.  For bitmap */
  /*    formats, the behaviour is up to the driver.                        */
  /*                                                                       */
  /*    See the note section of @FT_Size_Metrics if you wonder how size    */
  /*    requesting relates to scaling values.                              */
  /*                                                                       */
  typedef enum  FT_Size_Request_Type_
  {
    FT_SIZE_REQUEST_TYPE_NOMINAL,
    FT_SIZE_REQUEST_TYPE_REAL_DIM,
    FT_SIZE_REQUEST_TYPE_BBOX,
    FT_SIZE_REQUEST_TYPE_CELL,
    FT_SIZE_REQUEST_TYPE_SCALES,

    FT_SIZE_REQUEST_TYPE_MAX

  } FT_Size_Request_Type;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Size_RequestRec                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to model a size request.                          */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    type           :: See @FT_Size_Request_Type.                       */
  /*                                                                       */
  /*    width          :: The desired width.                               */
  /*                                                                       */
  /*    height         :: The desired height.                              */
  /*                                                                       */
  /*    horiResolution :: The horizontal resolution.  If set to zero,      */
  /*                      `width' is treated as a 26.6 fractional pixel    */
  /*                      value.                                           */
  /*                                                                       */
  /*    vertResolution :: The vertical resolution.  If set to zero,        */
  /*                      `height' is treated as a 26.6 fractional pixel   */
  /*                      value.                                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If `width' is zero, then the horizontal scaling value is set equal */
  /*    to the vertical scaling value, and vice versa.                     */
  /*                                                                       */
  typedef struct  FT_Size_RequestRec_
  {
    FT_Size_Request_Type  type;
    FT_Long               width;
    FT_Long               height;
    FT_UInt               horiResolution;
    FT_UInt               vertResolution;

  } FT_Size_RequestRec;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_Size_Request                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle to a size request structure.                              */
  /*                                                                       */
  typedef struct FT_Size_RequestRec_  *FT_Size_Request;


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Request_Size                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Resize the scale of the active @FT_Size object in a face.          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: A handle to a target face object.                          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    req  :: A pointer to a @FT_Size_RequestRec.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Although drivers may select the bitmap strike matching the         */
  /*    request, you should not rely on this if you intend to select a     */
  /*    particular bitmap strike.  Use @FT_Select_Size instead in that     */
  /*    case.                                                              */
  /*                                                                       */
  /*    The relation between the requested size and the resulting glyph    */
  /*    size is dependent entirely on how the size is defined in the       */
  /*    source face.  The font designer chooses the final size of each     */
  /*    glyph relative to this size.  For more information refer to        */
  /*    `http://www.freetype.org/freetype2/docs/glyphs/glyphs-2.html'      */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Request_Size( FT_Face          face,
                   FT_Size_Request  req );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Char_Size                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function calls @FT_Request_Size to request the nominal size   */
  /*    (in points).                                                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face            :: A handle to a target face object.               */
  /*                                                                       */
  /* <Input>                                                               */
  /*    char_width      :: The nominal width, in 26.6 fractional points.   */
  /*                                                                       */
  /*    char_height     :: The nominal height, in 26.6 fractional points.  */
  /*                                                                       */
  /*    horz_resolution :: The horizontal resolution in dpi.               */
  /*                                                                       */
  /*    vert_resolution :: The vertical resolution in dpi.                 */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If either the character width or height is zero, it is set equal   */
  /*    to the other value.                                                */
  /*                                                                       */
  /*    If either the horizontal or vertical resolution is zero, it is set */
  /*    equal to the other value.                                          */
  /*                                                                       */
  /*    A character width or height smaller than 1pt is set to 1pt; if     */
  /*    both resolution values are zero, they are set to 72dpi.            */
  /*                                                                       */
  /*    Don't use this function if you are using the FreeType cache API.   */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Char_Size( FT_Face     face,
                    FT_F26Dot6  char_width,
                    FT_F26Dot6  char_height,
                    FT_UInt     horz_resolution,
                    FT_UInt     vert_resolution );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Pixel_Sizes                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function calls @FT_Request_Size to request the nominal size   */
  /*    (in pixels).                                                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face         :: A handle to the target face object.                */
  /*                                                                       */
  /* <Input>                                                               */
  /*    pixel_width  :: The nominal width, in pixels.                      */
  /*                                                                       */
  /*    pixel_height :: The nominal height, in pixels.                     */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    You should not rely on the resulting glyphs matching, or being     */
  /*    constrained, to this pixel size.  Refer to @FT_Request_Size to     */
  /*    understand how requested sizes relate to actual sizes.             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Pixel_Sizes( FT_Face  face,
                      FT_UInt  pixel_width,
                      FT_UInt  pixel_height );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Load_Glyph                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used to load a single glyph into the glyph slot of a    */
  /*    face object.                                                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face        :: A handle to the target face object where the glyph  */
  /*                   is loaded.                                          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    glyph_index :: The index of the glyph in the font file.  For       */
  /*                   CID-keyed fonts (either in PS or in CFF format)     */
  /*                   this argument specifies the CID value.              */
  /*                                                                       */
  /*    load_flags  :: A flag indicating what to load for this glyph.  The */
  /*                   @FT_LOAD_XXX constants can be used to control the   */
  /*                   glyph loading process (e.g., whether the outline    */
  /*                   should be scaled, whether to load bitmaps or not,   */
  /*                   whether to hint the outline, etc).                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The loaded glyph may be transformed.  See @FT_Set_Transform for    */
  /*    the details.                                                       */
  /*                                                                       */
  /*    For subsetted CID-keyed fonts, `FT_Err_Invalid_Argument' is        */
  /*    returned for invalid CID values (this is, for CID values which     */
  /*    don't have a corresponding glyph in the font).  See the discussion */
  /*    of the @FT_FACE_FLAG_CID_KEYED flag for more details.              */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Load_Glyph( FT_Face   face,
                 FT_UInt   glyph_index,
                 FT_Int32  load_flags );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Load_Char                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used to load a single glyph into the glyph slot of a    */
  /*    face object, according to its character code.                      */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face        :: A handle to a target face object where the glyph    */
  /*                   is loaded.                                          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    char_code   :: The glyph's character code, according to the        */
  /*                   current charmap used in the face.                   */
  /*                                                                       */
  /*    load_flags  :: A flag indicating what to load for this glyph.  The */
  /*                   @FT_LOAD_XXX constants can be used to control the   */
  /*                   glyph loading process (e.g., whether the outline    */
  /*                   should be scaled, whether to load bitmaps or not,   */
  /*                   whether to hint the outline, etc).                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function simply calls @FT_Get_Char_Index and @FT_Load_Glyph.  */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Load_Char( FT_Face   face,
                FT_ULong  char_code,
                FT_Int32  load_flags );


  /*************************************************************************
   *
   * @enum:
   *   FT_LOAD_XXX
   *
   * @description:
   *   A list of bit-field constants used with @FT_Load_Glyph to indicate
   *   what kind of operations to perform during glyph loading.
   *
   * @values:
   *   FT_LOAD_DEFAULT ::
   *     Corresponding to~0, this value is used as the default glyph load
   *     operation.  In this case, the following happens:
   *
   *     1. FreeType looks for a bitmap for the glyph corresponding to the
   *        face's current size.  If one is found, the function returns.
   *        The bitmap data can be accessed from the glyph slot (see note
   *        below).
   *
   *     2. If no embedded bitmap is searched or found, FreeType looks for a
   *        scalable outline.  If one is found, it is loaded from the font
   *        file, scaled to device pixels, then `hinted' to the pixel grid
   *        in order to optimize it.  The outline data can be accessed from
   *        the glyph slot (see note below).
   *
   *     Note that by default, the glyph loader doesn't render outlines into
   *     bitmaps.  The following flags are used to modify this default
   *     behaviour to more specific and useful cases.
   *
   *   FT_LOAD_NO_SCALE ::
   *     Don't scale the loaded outline glyph but keep it in font units.
   *
   *     This flag implies @FT_LOAD_NO_HINTING and @FT_LOAD_NO_BITMAP, and
   *     unsets @FT_LOAD_RENDER.
   *
   *     If the font is `tricky' (see @FT_FACE_FLAG_TRICKY for more), using
   *     FT_LOAD_NO_SCALE usually yields meaningless outlines because the
   *     subglyphs must be scaled and positioned with hinting instructions.
   *     This can be solved by loading the font without FT_LOAD_NO_SCALE and
   *     setting the character size to `font->units_per_EM'.
   *
   *   FT_LOAD_NO_HINTING ::
   *     Disable hinting.  This generally generates `blurrier' bitmap glyphs
   *     when the glyph are rendered in any of the anti-aliased modes.  See
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
   *   FT_LOAD_VERTICAL_LAYOUT ::
   *     Load the glyph for vertical text layout.  In particular, the
   *     `advance' value in the @FT_GlyphSlotRec structure is set to the
   *     `vertAdvance' value of the `metrics' field.
   *
   *     In case @FT_HAS_VERTICAL doesn't return true, you shouldn't use
   *     this flag currently.  Reason is that in this case vertical metrics
   *     get synthesized, and those values are not always consistent across
   *     various font formats.
   *
   *   FT_LOAD_FORCE_AUTOHINT ::
   *     Indicates that the auto-hinter is preferred over the font's native
   *     hinter.  See also the note below.
   *
   *   FT_LOAD_CROP_BITMAP ::
   *     Indicates that the font driver should crop the loaded bitmap glyph
   *     (i.e., remove all space around its black bits).  Not all drivers
   *     implement this.
   *
   *   FT_LOAD_PEDANTIC ::
   *     Indicates that the font driver should perform pedantic verifications
   *     during glyph loading.  This is mostly used to detect broken glyphs
   *     in fonts.  By default, FreeType tries to handle broken fonts also.
   *
   *     In particular, errors from the TrueType bytecode engine are not
   *     passed to the application if this flag is not set; this might
   *     result in partially hinted or distorted glyphs in case a glyph's
   *     bytecode is buggy.
   *
   *   FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH ::
   *     Ignored.  Deprecated.
   *
   *   FT_LOAD_NO_RECURSE ::
   *     This flag is only used internally.  It merely indicates that the
   *     font driver should not load composite glyphs recursively.  Instead,
   *     it should set the `num_subglyph' and `subglyphs' values of the
   *     glyph slot accordingly, and set `glyph->format' to
   *     @FT_GLYPH_FORMAT_COMPOSITE.
   *
   *     The description of sub-glyphs is not available to client
   *     applications for now.
   *
   *     This flag implies @FT_LOAD_NO_SCALE and @FT_LOAD_IGNORE_TRANSFORM.
   *
   *   FT_LOAD_IGNORE_TRANSFORM ::
   *     Indicates that the transform matrix set by @FT_Set_Transform should
   *     be ignored.
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
   *     Indicates that the `linearHoriAdvance' and `linearVertAdvance'
   *     fields of @FT_GlyphSlotRec should be kept in font units.  See
   *     @FT_GlyphSlotRec for details.
   *
   *   FT_LOAD_NO_AUTOHINT ::
   *     Disable auto-hinter.  See also the note below.
   *
   *   FT_LOAD_COLOR ::
   *     This flag is used to request loading of color embedded-bitmap
   *     images.  The resulting color bitmaps, if available, will have the
   *     @FT_PIXEL_MODE_BGRA format.  When the flag is not used and color
   *     bitmaps are found, they will be converted to 256-level gray
   *     bitmaps transparently.  Those bitmaps will be in the
   *     @FT_PIXEL_MODE_GRAY format.
   *
   * @note:
   *   By default, hinting is enabled and the font's native hinter (see
   *   @FT_FACE_FLAG_HINTER) is preferred over the auto-hinter.  You can
   *   disable hinting by setting @FT_LOAD_NO_HINTING or change the
   *   precedence by setting @FT_LOAD_FORCE_AUTOHINT.  You can also set
   *   @FT_LOAD_NO_AUTOHINT in case you don't want the auto-hinter to be
   *   used at all.
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
   *   character code U+005A, LATIN CAPITAL LETTER Z, to a glyph depicting a
   *   mathematical integral sign), the auto-hinter might produce useless
   *   results.
   *
   */
#define FT_LOAD_DEFAULT                      0x0
#define FT_LOAD_NO_SCALE                     ( 1L << 0 )
#define FT_LOAD_NO_HINTING                   ( 1L << 1 )
#define FT_LOAD_RENDER                       ( 1L << 2 )
#define FT_LOAD_NO_BITMAP                    ( 1L << 3 )
#define FT_LOAD_VERTICAL_LAYOUT              ( 1L << 4 )
#define FT_LOAD_FORCE_AUTOHINT               ( 1L << 5 )
#define FT_LOAD_CROP_BITMAP                  ( 1L << 6 )
#define FT_LOAD_PEDANTIC                     ( 1L << 7 )
#define FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH  ( 1L << 9 )
#define FT_LOAD_NO_RECURSE                   ( 1L << 10 )
#define FT_LOAD_IGNORE_TRANSFORM             ( 1L << 11 )
#define FT_LOAD_MONOCHROME                   ( 1L << 12 )
#define FT_LOAD_LINEAR_DESIGN                ( 1L << 13 )
#define FT_LOAD_NO_AUTOHINT                  ( 1L << 15 )
  /* Bits 16..19 are used by `FT_LOAD_TARGET_' */
#define FT_LOAD_COLOR                        ( 1L << 20 )

  /* */

  /* used internally only by certain font drivers! */
#define FT_LOAD_ADVANCE_ONLY                 ( 1L << 8 )
#define FT_LOAD_SBITS_ONLY                   ( 1L << 14 )


  /**************************************************************************
   *
   * @enum:
   *   FT_LOAD_TARGET_XXX
   *
   * @description:
   *   A list of values that are used to select a specific hinting algorithm
   *   to use by the hinter.  You should OR one of these values to your
   *   `load_flags' when calling @FT_Load_Glyph.
   *
   *   Note that font's native hinters may ignore the hinting algorithm you
   *   have specified (e.g., the TrueType bytecode interpreter).  You can set
   *   @FT_LOAD_FORCE_AUTOHINT to ensure that the auto-hinter is used.
   *
   *   Also note that @FT_LOAD_TARGET_LIGHT is an exception, in that it
   *   always implies @FT_LOAD_FORCE_AUTOHINT.
   *
   * @values:
   *   FT_LOAD_TARGET_NORMAL ::
   *     This corresponds to the default hinting algorithm, optimized for
   *     standard gray-level rendering.  For monochrome output, use
   *     @FT_LOAD_TARGET_MONO instead.
   *
   *   FT_LOAD_TARGET_LIGHT ::
   *     A lighter hinting algorithm for non-monochrome modes.  Many
   *     generated glyphs are more fuzzy but better resemble its original
   *     shape.  A bit like rendering on Mac OS~X.
   *
   *     As a special exception, this target implies @FT_LOAD_FORCE_AUTOHINT.
   *
   *   FT_LOAD_TARGET_MONO ::
   *     Strong hinting algorithm that should only be used for monochrome
   *     output.  The result is probably unpleasant if the glyph is rendered
   *     in non-monochrome modes.
   *
   *   FT_LOAD_TARGET_LCD ::
   *     A variant of @FT_LOAD_TARGET_NORMAL optimized for horizontally
   *     decimated LCD displays.
   *
   *   FT_LOAD_TARGET_LCD_V ::
   *     A variant of @FT_LOAD_TARGET_NORMAL optimized for vertically
   *     decimated LCD displays.
   *
   * @note:
   *   You should use only _one_ of the FT_LOAD_TARGET_XXX values in your
   *   `load_flags'.  They can't be ORed.
   *
   *   If @FT_LOAD_RENDER is also set, the glyph is rendered in the
   *   corresponding mode (i.e., the mode which matches the used algorithm
   *   best).  An exeption is FT_LOAD_TARGET_MONO since it implies
   *   @FT_LOAD_MONOCHROME.
   *
   *   You can use a hinting algorithm that doesn't correspond to the same
   *   rendering mode.  As an example, it is possible to use the `light'
   *   hinting algorithm and have the results rendered in horizontal LCD
   *   pixel mode, with code like
   *
   *     {
   *       FT_Load_Glyph( face, glyph_index,
   *                      load_flags | FT_LOAD_TARGET_LIGHT );
   *
   *       FT_Render_Glyph( face->glyph, FT_RENDER_MODE_LCD );
   *     }
   *
   */
#define FT_LOAD_TARGET_( x )   ( (FT_Int32)( (x) & 15 ) << 16 )

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
#define FT_LOAD_TARGET_MODE( x )  ( (FT_Render_Mode)( ( (x) >> 16 ) & 15 ) )


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Transform                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used to set the transformation that is applied to glyph */
  /*    images when they are loaded into a glyph slot through              */
  /*    @FT_Load_Glyph.                                                    */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face   :: A handle to the source face object.                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    matrix :: A pointer to the transformation's 2x2 matrix.  Use~0 for */
  /*              the identity matrix.                                     */
  /*    delta  :: A pointer to the translation vector.  Use~0 for the null */
  /*              vector.                                                  */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The transformation is only applied to scalable image formats after */
  /*    the glyph has been loaded.  It means that hinting is unaltered by  */
  /*    the transformation and is performed on the character size given in */
  /*    the last call to @FT_Set_Char_Size or @FT_Set_Pixel_Sizes.         */
  /*                                                                       */
  /*    Note that this also transforms the `face.glyph.advance' field, but */
  /*    *not* the values in `face.glyph.metrics'.                          */
  /*                                                                       */
  FT_EXPORT( void )
  FT_Set_Transform( FT_Face     face,
                    FT_Matrix*  matrix,
                    FT_Vector*  delta );


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_Render_Mode                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An enumeration type that lists the render modes supported by       */
  /*    FreeType~2.  Each mode corresponds to a specific type of scanline  */
  /*    conversion performed on the outline.                               */
  /*                                                                       */
  /*    For bitmap fonts and embedded bitmaps the `bitmap->pixel_mode'     */
  /*    field in the @FT_GlyphSlotRec structure gives the format of the    */
  /*    returned bitmap.                                                   */
  /*                                                                       */
  /*    All modes except @FT_RENDER_MODE_MONO use 256 levels of opacity.   */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_RENDER_MODE_NORMAL ::                                           */
  /*      This is the default render mode; it corresponds to 8-bit         */
  /*      anti-aliased bitmaps.                                            */
  /*                                                                       */
  /*    FT_RENDER_MODE_LIGHT ::                                            */
  /*      This is equivalent to @FT_RENDER_MODE_NORMAL.  It is only        */
  /*      defined as a separate value because render modes are also used   */
  /*      indirectly to define hinting algorithm selectors.  See           */
  /*      @FT_LOAD_TARGET_XXX for details.                                 */
  /*                                                                       */
  /*    FT_RENDER_MODE_MONO ::                                             */
  /*      This mode corresponds to 1-bit bitmaps (with 2~levels of         */
  /*      opacity).                                                        */
  /*                                                                       */
  /*    FT_RENDER_MODE_LCD ::                                              */
  /*      This mode corresponds to horizontal RGB and BGR sub-pixel        */
  /*      displays like LCD screens.  It produces 8-bit bitmaps that are   */
  /*      3~times the width of the original glyph outline in pixels, and   */
  /*      which use the @FT_PIXEL_MODE_LCD mode.                           */
  /*                                                                       */
  /*    FT_RENDER_MODE_LCD_V ::                                            */
  /*      This mode corresponds to vertical RGB and BGR sub-pixel displays */
  /*      (like PDA screens, rotated LCD displays, etc.).  It produces     */
  /*      8-bit bitmaps that are 3~times the height of the original        */
  /*      glyph outline in pixels and use the @FT_PIXEL_MODE_LCD_V mode.   */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The LCD-optimized glyph bitmaps produced by FT_Render_Glyph can be */
  /*    filtered to reduce color-fringes by using @FT_Library_SetLcdFilter */
  /*    (not active in the default builds).  It is up to the caller to     */
  /*    either call @FT_Library_SetLcdFilter (if available) or do the      */
  /*    filtering itself.                                                  */
  /*                                                                       */
  /*    The selected render mode only affects vector glyphs of a font.     */
  /*    Embedded bitmaps often have a different pixel mode like            */
  /*    @FT_PIXEL_MODE_MONO.  You can use @FT_Bitmap_Convert to transform  */
  /*    them into 8-bit pixmaps.                                           */
  /*                                                                       */
  typedef enum  FT_Render_Mode_
  {
    FT_RENDER_MODE_NORMAL = 0,
    FT_RENDER_MODE_LIGHT,
    FT_RENDER_MODE_MONO,
    FT_RENDER_MODE_LCD,
    FT_RENDER_MODE_LCD_V,

    FT_RENDER_MODE_MAX

  } FT_Render_Mode;


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    ft_render_mode_xxx                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    These constants are deprecated.  Use the corresponding             */
  /*    @FT_Render_Mode values instead.                                    */
  /*                                                                       */
  /* <Values>                                                              */
  /*    ft_render_mode_normal :: see @FT_RENDER_MODE_NORMAL                */
  /*    ft_render_mode_mono   :: see @FT_RENDER_MODE_MONO                  */
  /*                                                                       */
#define ft_render_mode_normal  FT_RENDER_MODE_NORMAL
#define ft_render_mode_mono    FT_RENDER_MODE_MONO


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Render_Glyph                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Convert a given glyph image to a bitmap.  It does so by inspecting */
  /*    the glyph image format, finding the relevant renderer, and         */
  /*    invoking it.                                                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    slot        :: A handle to the glyph slot containing the image to  */
  /*                   convert.                                            */
  /*                                                                       */
  /* <Input>                                                               */
  /*    render_mode :: This is the render mode used to render the glyph    */
  /*                   image into a bitmap.  See @FT_Render_Mode for a     */
  /*                   list of possible values.                            */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Render_Glyph( FT_GlyphSlot    slot,
                   FT_Render_Mode  render_mode );


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_Kerning_Mode                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An enumeration used to specify which kerning values to return in   */
  /*    @FT_Get_Kerning.                                                   */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_KERNING_DEFAULT  :: Return scaled and grid-fitted kerning       */
  /*                           distances (value is~0).                     */
  /*                                                                       */
  /*    FT_KERNING_UNFITTED :: Return scaled but un-grid-fitted kerning    */
  /*                           distances.                                  */
  /*                                                                       */
  /*    FT_KERNING_UNSCALED :: Return the kerning vector in original font  */
  /*                           units.                                      */
  /*                                                                       */
  typedef enum  FT_Kerning_Mode_
  {
    FT_KERNING_DEFAULT  = 0,
    FT_KERNING_UNFITTED,
    FT_KERNING_UNSCALED

  } FT_Kerning_Mode;


  /*************************************************************************/
  /*                                                                       */
  /* <Const>                                                               */
  /*    ft_kerning_default                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This constant is deprecated.  Please use @FT_KERNING_DEFAULT       */
  /*    instead.                                                           */
  /*                                                                       */
#define ft_kerning_default   FT_KERNING_DEFAULT


  /*************************************************************************/
  /*                                                                       */
  /* <Const>                                                               */
  /*    ft_kerning_unfitted                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This constant is deprecated.  Please use @FT_KERNING_UNFITTED      */
  /*    instead.                                                           */
  /*                                                                       */
#define ft_kerning_unfitted  FT_KERNING_UNFITTED


  /*************************************************************************/
  /*                                                                       */
  /* <Const>                                                               */
  /*    ft_kerning_unscaled                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This constant is deprecated.  Please use @FT_KERNING_UNSCALED      */
  /*    instead.                                                           */
  /*                                                                       */
#define ft_kerning_unscaled  FT_KERNING_UNSCALED


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Kerning                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the kerning vector between two glyphs of a same face.       */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face        :: A handle to a source face object.                   */
  /*                                                                       */
  /*    left_glyph  :: The index of the left glyph in the kern pair.       */
  /*                                                                       */
  /*    right_glyph :: The index of the right glyph in the kern pair.      */
  /*                                                                       */
  /*    kern_mode   :: See @FT_Kerning_Mode for more information.          */
  /*                   Determines the scale and dimension of the returned  */
  /*                   kerning vector.                                     */
  /*                                                                       */
  /* <Output>                                                              */
  /*    akerning    :: The kerning vector.  This is either in font units   */
  /*                   or in pixels (26.6 format) for scalable formats,    */
  /*                   and in pixels for fixed-sizes formats.              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Only horizontal layouts (left-to-right & right-to-left) are        */
  /*    supported by this method.  Other layouts, or more sophisticated    */
  /*    kernings, are out of the scope of this API function -- they can be */
  /*    implemented through format-specific interfaces.                    */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Kerning( FT_Face     face,
                  FT_UInt     left_glyph,
                  FT_UInt     right_glyph,
                  FT_UInt     kern_mode,
                  FT_Vector  *akerning );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Track_Kerning                                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the track kerning for a given face object at a given size.  */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face       :: A handle to a source face object.                    */
  /*                                                                       */
  /*    point_size :: The point size in 16.16 fractional points.           */
  /*                                                                       */
  /*    degree     :: The degree of tightness.  Increasingly negative      */
  /*                  values represent tighter track kerning, while        */
  /*                  increasingly positive values represent looser track  */
  /*                  kerning.  Value zero means no track kerning.         */
  /*                                                                       */
  /* <Output>                                                              */
  /*    akerning   :: The kerning in 16.16 fractional points, to be        */
  /*                  uniformly applied between all glyphs.                */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Currently, only the Type~1 font driver supports track kerning,     */
  /*    using data from AFM files (if attached with @FT_Attach_File or     */
  /*    @FT_Attach_Stream).                                                */
  /*                                                                       */
  /*    Only very few AFM files come with track kerning data; please refer */
  /*    to the Adobe's AFM specification for more details.                 */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Track_Kerning( FT_Face    face,
                        FT_Fixed   point_size,
                        FT_Int     degree,
                        FT_Fixed*  akerning );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Glyph_Name                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Retrieve the ASCII name of a given glyph in a face.  This only     */
  /*    works for those faces where @FT_HAS_GLYPH_NAMES(face) returns~1.   */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face        :: A handle to a source face object.                   */
  /*                                                                       */
  /*    glyph_index :: The glyph index.                                    */
  /*                                                                       */
  /*    buffer_max  :: The maximum number of bytes available in the        */
  /*                   buffer.                                             */
  /*                                                                       */
  /* <Output>                                                              */
  /*    buffer      :: A pointer to a target buffer where the name is      */
  /*                   copied to.                                          */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    An error is returned if the face doesn't provide glyph names or if */
  /*    the glyph index is invalid.  In all cases of failure, the first    */
  /*    byte of `buffer' is set to~0 to indicate an empty name.            */
  /*                                                                       */
  /*    The glyph name is truncated to fit within the buffer if it is too  */
  /*    long.  The returned string is always zero-terminated.              */
  /*                                                                       */
  /*    Be aware that FreeType reorders glyph indices internally so that   */
  /*    glyph index~0 always corresponds to the `missing glyph' (called    */
  /*    `.notdef').                                                        */
  /*                                                                       */
  /*    This function is not compiled within the library if the config     */
  /*    macro `FT_CONFIG_OPTION_NO_GLYPH_NAMES' is defined in              */
  /*    `include/freetype/config/ftoptions.h'.                             */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Get_Glyph_Name( FT_Face     face,
                     FT_UInt     glyph_index,
                     FT_Pointer  buffer,
                     FT_UInt     buffer_max );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Postscript_Name                                             */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Retrieve the ASCII PostScript name of a given face, if available.  */
  /*    This only works with PostScript and TrueType fonts.                */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A handle to the source face object.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    A pointer to the face's PostScript name.  NULL if unavailable.     */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The returned pointer is owned by the face and is destroyed with    */
  /*    it.                                                                */
  /*                                                                       */
  FT_EXPORT( const char* )
  FT_Get_Postscript_Name( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Select_Charmap                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Select a given charmap by its encoding tag (as listed in           */
  /*    `freetype.h').                                                     */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face     :: A handle to the source face object.                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    encoding :: A handle to the selected encoding.                     */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function returns an error if no charmap in the face           */
  /*    corresponds to the encoding queried here.                          */
  /*                                                                       */
  /*    Because many fonts contain more than a single cmap for Unicode     */
  /*    encoding, this function has some special code to select the one    */
  /*    which covers Unicode best (`best' in the sense that a UCS-4 cmap   */
  /*    is preferred to a UCS-2 cmap).  It is thus preferable to           */
  /*    @FT_Set_Charmap in this case.                                      */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Select_Charmap( FT_Face      face,
                     FT_Encoding  encoding );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Set_Charmap                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Select a given charmap for character code to glyph index mapping.  */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face    :: A handle to the source face object.                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    charmap :: A handle to the selected charmap.                       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function returns an error if the charmap is not part of       */
  /*    the face (i.e., if it is not listed in the `face->charmaps'        */
  /*    table).                                                            */
  /*                                                                       */
  /*    It also fails if a type~14 charmap is selected.                    */
  /*                                                                       */
  FT_EXPORT( FT_Error )
  FT_Set_Charmap( FT_Face     face,
                  FT_CharMap  charmap );


  /*************************************************************************
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
   *   `charmap' belongs.  If an error occurs, -1 is returned.
   *
   */
  FT_EXPORT( FT_Int )
  FT_Get_Charmap_Index( FT_CharMap  charmap );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Char_Index                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the glyph index of a given character code.  This function   */
  /*    uses a charmap object to do the mapping.                           */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face     :: A handle to the source face object.                    */
  /*                                                                       */
  /*    charcode :: The character code.                                    */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The glyph index.  0~means `undefined character code'.              */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If you use FreeType to manipulate the contents of font files       */
  /*    directly, be aware that the glyph index returned by this function  */
  /*    doesn't always correspond to the internal indices used within the  */
  /*    file.  This is done to ensure that value~0 always corresponds to   */
  /*    the `missing glyph'.  If the first glyph is not named `.notdef',   */
  /*    then for Type~1 and Type~42 fonts, `.notdef' will be moved into    */
  /*    the glyph ID~0 position, and whatever was there will be moved to   */
  /*    the position `.notdef' had.  For Type~1 fonts, if there is no      */
  /*    `.notdef' glyph at all, then one will be created at index~0 and    */
  /*    whatever was there will be moved to the last index -- Type~42      */
  /*    fonts are considered invalid under this condition.                 */
  /*                                                                       */
  FT_EXPORT( FT_UInt )
  FT_Get_Char_Index( FT_Face   face,
                     FT_ULong  charcode );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_First_Char                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function is used to return the first character code in the    */
  /*    current charmap of a given face.  It also returns the              */
  /*    corresponding glyph index.                                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face    :: A handle to the source face object.                     */
  /*                                                                       */
  /* <Output>                                                              */
  /*    agindex :: Glyph index of first character code.  0~if charmap is   */
  /*               empty.                                                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The charmap's first character code.                                */
  /*                                                                       */
  /* <Note>                                                                */
  /*    You should use this function with @FT_Get_Next_Char to be able to  */
  /*    parse all character codes available in a given charmap.  The code  */
  /*    should look like this:                                             */
  /*                                                                       */
  /*    {                                                                  */
  /*      FT_ULong  charcode;                                              */
  /*      FT_UInt   gindex;                                                */
  /*                                                                       */
  /*                                                                       */
  /*      charcode = FT_Get_First_Char( face, &gindex );                   */
  /*      while ( gindex != 0 )                                            */
  /*      {                                                                */
  /*        ... do something with (charcode,gindex) pair ...               */
  /*                                                                       */
  /*        charcode = FT_Get_Next_Char( face, charcode, &gindex );        */
  /*      }                                                                */
  /*    }                                                                  */
  /*                                                                       */
  /*    Note that `*agindex' is set to~0 if the charmap is empty.  The     */
  /*    result itself can be~0 in two cases: if the charmap is empty or    */
  /*    if the value~0 is the first valid character code.                  */
  /*                                                                       */
  FT_EXPORT( FT_ULong )
  FT_Get_First_Char( FT_Face   face,
                     FT_UInt  *agindex );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Next_Char                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function is used to return the next character code in the     */
  /*    current charmap of a given face following the value `char_code',   */
  /*    as well as the corresponding glyph index.                          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face      :: A handle to the source face object.                   */
  /*    char_code :: The starting character code.                          */
  /*                                                                       */
  /* <Output>                                                              */
  /*    agindex   :: Glyph index of next character code.  0~if charmap     */
  /*                 is empty.                                             */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The charmap's next character code.                                 */
  /*                                                                       */
  /* <Note>                                                                */
  /*    You should use this function with @FT_Get_First_Char to walk       */
  /*    over all character codes available in a given charmap.  See the    */
  /*    note for this function for a simple code example.                  */
  /*                                                                       */
  /*    Note that `*agindex' is set to~0 when there are no more codes in   */
  /*    the charmap.                                                       */
  /*                                                                       */
  FT_EXPORT( FT_ULong )
  FT_Get_Next_Char( FT_Face    face,
                    FT_ULong   char_code,
                    FT_UInt   *agindex );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_Name_Index                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the glyph index of a given glyph name.  This function uses  */
  /*    driver specific objects to do the translation.                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face       :: A handle to the source face object.                  */
  /*                                                                       */
  /*    glyph_name :: The glyph name.                                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The glyph index.  0~means `undefined character code'.              */
  /*                                                                       */
  FT_EXPORT( FT_UInt )
  FT_Get_Name_Index( FT_Face     face,
                     FT_String*  glyph_name );


  /*************************************************************************
   *
   * @macro:
   *   FT_SUBGLYPH_FLAG_XXX
   *
   * @description:
   *   A list of constants used to describe subglyphs.  Please refer to the
   *   TrueType specification for the meaning of the various flags.
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


  /*************************************************************************
   *
   * @func:
   *   FT_Get_SubGlyph_Info
   *
   * @description:
   *   Retrieve a description of a given subglyph.  Only use it if
   *   `glyph->format' is @FT_GLYPH_FORMAT_COMPOSITE; an error is
   *   returned otherwise.
   *
   * @input:
   *   glyph ::
   *     The source glyph slot.
   *
   *   sub_index ::
   *     The index of the subglyph.  Must be less than
   *     `glyph->num_subglyphs'.
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
   *   The values of `*p_arg1', `*p_arg2', and `*p_transform' must be
   *   interpreted depending on the flags returned in `*p_flags'.  See the
   *   TrueType specification for details.
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


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    FT_FSTYPE_XXX                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit flags used in the `fsType' field of the OS/2 table   */
  /*    in a TrueType or OpenType font and the `FSType' entry in a         */
  /*    PostScript font.  These bit flags are returned by                  */
  /*    @FT_Get_FSType_Flags; they inform client applications of embedding */
  /*    and subsetting restrictions associated with a font.                */
  /*                                                                       */
  /*    See http://www.adobe.com/devnet/acrobat/pdfs/FontPolicies.pdf for  */
  /*    more details.                                                      */
  /*                                                                       */
  /* <Values>                                                              */
  /*    FT_FSTYPE_INSTALLABLE_EMBEDDING ::                                 */
  /*      Fonts with no fsType bit set may be embedded and permanently     */
  /*      installed on the remote system by an application.                */
  /*                                                                       */
  /*    FT_FSTYPE_RESTRICTED_LICENSE_EMBEDDING ::                          */
  /*      Fonts that have only this bit set must not be modified, embedded */
  /*      or exchanged in any manner without first obtaining permission of */
  /*      the font software copyright owner.                               */
  /*                                                                       */
  /*    FT_FSTYPE_PREVIEW_AND_PRINT_EMBEDDING ::                           */
  /*      If this bit is set, the font may be embedded and temporarily     */
  /*      loaded on the remote system.  Documents containing Preview &     */
  /*      Print fonts must be opened `read-only'; no edits can be applied  */
  /*      to the document.                                                 */
  /*                                                                       */
  /*    FT_FSTYPE_EDITABLE_EMBEDDING ::                                    */
  /*      If this bit is set, the font may be embedded but must only be    */
  /*      installed temporarily on other systems.  In contrast to Preview  */
  /*      & Print fonts, documents containing editable fonts may be opened */
  /*      for reading, editing is permitted, and changes may be saved.     */
  /*                                                                       */
  /*    FT_FSTYPE_NO_SUBSETTING ::                                         */
  /*      If this bit is set, the font may not be subsetted prior to       */
  /*      embedding.                                                       */
  /*                                                                       */
  /*    FT_FSTYPE_BITMAP_EMBEDDING_ONLY ::                                 */
  /*      If this bit is set, only bitmaps contained in the font may be    */
  /*      embedded; no outline data may be embedded.  If there are no      */
  /*      bitmaps available in the font, then the font is unembeddable.    */
  /*                                                                       */
  /* <Note>                                                                */
  /*    While the fsType flags can indicate that a font may be embedded, a */
  /*    license with the font vendor may be separately required to use the */
  /*    font in this way.                                                  */
  /*                                                                       */
#define FT_FSTYPE_INSTALLABLE_EMBEDDING         0x0000
#define FT_FSTYPE_RESTRICTED_LICENSE_EMBEDDING  0x0002
#define FT_FSTYPE_PREVIEW_AND_PRINT_EMBEDDING   0x0004
#define FT_FSTYPE_EDITABLE_EMBEDDING            0x0008
#define FT_FSTYPE_NO_SUBSETTING                 0x0100
#define FT_FSTYPE_BITMAP_EMBEDDING_ONLY         0x0200


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Get_FSType_Flags                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the fsType flags for a font.                                */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A handle to the source face object.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The fsType flags, @FT_FSTYPE_XXX.                                  */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Use this function rather than directly reading the `fs_type' field */
  /*    in the @PS_FontInfoRec structure which is only guaranteed to       */
  /*    return the correct results for Type~1 fonts.                       */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.8                                                              */
  /*                                                                       */
  FT_EXPORT( FT_UShort )
  FT_Get_FSType_Flags( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    glyph_variants                                                     */
  /*                                                                       */
  /* <Title>                                                               */
  /*    Glyph Variants                                                     */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    The FreeType~2 interface to Unicode Ideographic Variation          */
  /*    Sequences (IVS), using the SFNT cmap format~14.                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Many CJK characters have variant forms.  They are a sort of grey   */
  /*    area somewhere between being totally irrelevant and semantically   */
  /*    distinct; for this reason, the Unicode consortium decided to       */
  /*    introduce Ideographic Variation Sequences (IVS), consisting of a   */
  /*    Unicode base character and one of 240 variant selectors            */
  /*    (U+E0100-U+E01EF), instead of further extending the already huge   */
  /*    code range for CJK characters.                                     */
  /*                                                                       */
  /*    An IVS is registered and unique; for further details please refer  */
  /*    to Unicode Technical Standard #37, the Ideographic Variation       */
  /*    Database:                                                          */
  /*                                                                       */
  /*      http://www.unicode.org/reports/tr37/                             */
  /*                                                                       */
  /*    To date (November 2012), the character with the most variants is   */
  /*    U+9089, having 31 such IVS.                                        */
  /*                                                                       */
  /*    Adobe and MS decided to support IVS with a new cmap subtable       */
  /*    (format~14).  It is an odd subtable because it is not a mapping of */
  /*    input code points to glyphs, but contains lists of all variants    */
  /*    supported by the font.                                             */
  /*                                                                       */
  /*    A variant may be either `default' or `non-default'.  A default     */
  /*    variant is the one you will get for that code point if you look it */
  /*    up in the standard Unicode cmap.  A non-default variant is a       */
  /*    different glyph.                                                   */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_GetCharVariantIndex                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the glyph index of a given character code as modified by    */
  /*    the variation selector.                                            */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face ::                                                            */
  /*      A handle to the source face object.                              */
  /*                                                                       */
  /*    charcode ::                                                        */
  /*      The character code point in Unicode.                             */
  /*                                                                       */
  /*    variantSelector ::                                                 */
  /*      The Unicode code point of the variation selector.                */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The glyph index.  0~means either `undefined character code', or    */
  /*    `undefined selector code', or `no variation selector cmap          */
  /*    subtable', or `current CharMap is not Unicode'.                    */
  /*                                                                       */
  /* <Note>                                                                */
  /*    If you use FreeType to manipulate the contents of font files       */
  /*    directly, be aware that the glyph index returned by this function  */
  /*    doesn't always correspond to the internal indices used within      */
  /*    the file.  This is done to ensure that value~0 always corresponds  */
  /*    to the `missing glyph'.                                            */
  /*                                                                       */
  /*    This function is only meaningful if                                */
  /*      a) the font has a variation selector cmap sub table,             */
  /*    and                                                                */
  /*      b) the current charmap has a Unicode encoding.                   */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.6                                                              */
  /*                                                                       */
  FT_EXPORT( FT_UInt )
  FT_Face_GetCharVariantIndex( FT_Face   face,
                               FT_ULong  charcode,
                               FT_ULong  variantSelector );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_GetCharVariantIsDefault                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Check whether this variant of this Unicode character is the one to */
  /*    be found in the `cmap'.                                            */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face ::                                                            */
  /*      A handle to the source face object.                              */
  /*                                                                       */
  /*    charcode ::                                                        */
  /*      The character codepoint in Unicode.                              */
  /*                                                                       */
  /*    variantSelector ::                                                 */
  /*      The Unicode codepoint of the variation selector.                 */
  /*                                                                       */
  /* <Return>                                                              */
  /*    1~if found in the standard (Unicode) cmap, 0~if found in the       */
  /*    variation selector cmap, or -1 if it is not a variant.             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function is only meaningful if the font has a variation       */
  /*    selector cmap subtable.                                            */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.6                                                              */
  /*                                                                       */
  FT_EXPORT( FT_Int )
  FT_Face_GetCharVariantIsDefault( FT_Face   face,
                                   FT_ULong  charcode,
                                   FT_ULong  variantSelector );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_GetVariantSelectors                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return a zero-terminated list of Unicode variant selectors found   */
  /*    in the font.                                                       */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face ::                                                            */
  /*      A handle to the source face object.                              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    A pointer to an array of selector code points, or NULL if there is */
  /*    no valid variant selector cmap subtable.                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The last item in the array is~0; the array is owned by the         */
  /*    @FT_Face object but can be overwritten or released on the next     */
  /*    call to a FreeType function.                                       */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.6                                                              */
  /*                                                                       */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetVariantSelectors( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_GetVariantsOfChar                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return a zero-terminated list of Unicode variant selectors found   */
  /*    for the specified character code.                                  */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face ::                                                            */
  /*      A handle to the source face object.                              */
  /*                                                                       */
  /*    charcode ::                                                        */
  /*      The character codepoint in Unicode.                              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    A pointer to an array of variant selector code points which are    */
  /*    active for the given character, or NULL if the corresponding list  */
  /*    is empty.                                                          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The last item in the array is~0; the array is owned by the         */
  /*    @FT_Face object but can be overwritten or released on the next     */
  /*    call to a FreeType function.                                       */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.6                                                              */
  /*                                                                       */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetVariantsOfChar( FT_Face   face,
                             FT_ULong  charcode );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_GetCharsOfVariant                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return a zero-terminated list of Unicode character codes found for */
  /*    the specified variant selector.                                    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face ::                                                            */
  /*      A handle to the source face object.                              */
  /*                                                                       */
  /*    variantSelector ::                                                 */
  /*      The variant selector code point in Unicode.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    A list of all the code points which are specified by this selector */
  /*    (both default and non-default codes are returned) or NULL if there */
  /*    is no valid cmap or the variant selector is invalid.               */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The last item in the array is~0; the array is owned by the         */
  /*    @FT_Face object but can be overwritten or released on the next     */
  /*    call to a FreeType function.                                       */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.6                                                              */
  /*                                                                       */
  FT_EXPORT( FT_UInt32* )
  FT_Face_GetCharsOfVariant( FT_Face   face,
                             FT_ULong  variantSelector );


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    computations                                                       */
  /*                                                                       */
  /* <Title>                                                               */
  /*    Computations                                                       */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    Crunching fixed numbers and vectors.                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This section contains various functions used to perform            */
  /*    computations on 16.16 fixed-float numbers or 2d vectors.           */
  /*                                                                       */
  /* <Order>                                                               */
  /*    FT_MulDiv                                                          */
  /*    FT_MulFix                                                          */
  /*    FT_DivFix                                                          */
  /*    FT_RoundFix                                                        */
  /*    FT_CeilFix                                                         */
  /*    FT_FloorFix                                                        */
  /*    FT_Vector_Transform                                                */
  /*    FT_Matrix_Multiply                                                 */
  /*    FT_Matrix_Invert                                                   */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_MulDiv                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to perform the computation `(a*b)/c'   */
  /*    with maximum accuracy (it uses a 64-bit intermediate integer       */
  /*    whenever necessary).                                               */
  /*                                                                       */
  /*    This function isn't necessarily as fast as some processor specific */
  /*    operations, but is at least completely portable.                   */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The first multiplier.                                         */
  /*    b :: The second multiplier.                                        */
  /*    c :: The divisor.                                                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `(a*b)/c'.  This function never traps when trying to */
  /*    divide by zero; it simply returns `MaxInt' or `MinInt' depending   */
  /*    on the signs of `a' and `b'.                                       */
  /*                                                                       */
  FT_EXPORT( FT_Long )
  FT_MulDiv( FT_Long  a,
             FT_Long  b,
             FT_Long  c );


  /* */

  /* The following #if 0 ... #endif is for the documentation formatter, */
  /* hiding the internal `FT_MULFIX_INLINED' macro.                     */

#if 0
  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_MulFix                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to perform the computation             */
  /*    `(a*b)/0x10000' with maximum accuracy.  Most of the time this is   */
  /*    used to multiply a given value by a 16.16 fixed-point factor.      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The first multiplier.                                         */
  /*    b :: The second multiplier.  Use a 16.16 factor here whenever      */
  /*         possible (see note below).                                    */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `(a*b)/0x10000'.                                     */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function has been optimized for the case where the absolute   */
  /*    value of `a' is less than 2048, and `b' is a 16.16 scaling factor. */
  /*    As this happens mainly when scaling from notional units to         */
  /*    fractional pixels in FreeType, it resulted in noticeable speed     */
  /*    improvements between versions 2.x and 1.x.                         */
  /*                                                                       */
  /*    As a conclusion, always try to place a 16.16 factor as the         */
  /*    _second_ argument of this function; this can make a great          */
  /*    difference.                                                        */
  /*                                                                       */
  FT_EXPORT( FT_Long )
  FT_MulFix( FT_Long  a,
             FT_Long  b );

  /* */
#endif

#ifdef FT_MULFIX_INLINED
#define FT_MulFix( a, b )  FT_MULFIX_INLINED( a, b )
#else
  FT_EXPORT( FT_Long )
  FT_MulFix( FT_Long  a,
             FT_Long  b );
#endif


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_DivFix                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to perform the computation             */
  /*    `(a*0x10000)/b' with maximum accuracy.  Most of the time, this is  */
  /*    used to divide a given value by a 16.16 fixed-point factor.        */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The first multiplier.                                         */
  /*    b :: The second multiplier.  Use a 16.16 factor here whenever      */
  /*         possible (see note below).                                    */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `(a*0x10000)/b'.                                     */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The optimization for FT_DivFix() is simple: If (a~<<~16) fits in   */
  /*    32~bits, then the division is computed directly.  Otherwise, we    */
  /*    use a specialized version of @FT_MulDiv.                           */
  /*                                                                       */
  FT_EXPORT( FT_Long )
  FT_DivFix( FT_Long  a,
             FT_Long  b );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_RoundFix                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to round a 16.16 fixed number.         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The number to be rounded.                                     */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `(a + 0x8000) & -0x10000'.                           */
  /*                                                                       */
  FT_EXPORT( FT_Fixed )
  FT_RoundFix( FT_Fixed  a );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_CeilFix                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to compute the ceiling function of a   */
  /*    16.16 fixed number.                                                */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The number for which the ceiling function is to be computed.  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `(a + 0x10000 - 1) & -0x10000'.                      */
  /*                                                                       */
  FT_EXPORT( FT_Fixed )
  FT_CeilFix( FT_Fixed  a );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_FloorFix                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to compute the floor function of a     */
  /*    16.16 fixed number.                                                */
  /*                                                                       */
  /* <Input>                                                               */
  /*    a :: The number for which the floor function is to be computed.    */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `a & -0x10000'.                                      */
  /*                                                                       */
  FT_EXPORT( FT_Fixed )
  FT_FloorFix( FT_Fixed  a );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Vector_Transform                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Transform a single vector through a 2x2 matrix.                    */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    vector :: The target vector to transform.                          */
  /*                                                                       */
  /* <Input>                                                               */
  /*    matrix :: A pointer to the source 2x2 matrix.                      */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The result is undefined if either `vector' or `matrix' is invalid. */
  /*                                                                       */
  FT_EXPORT( void )
  FT_Vector_Transform( FT_Vector*        vec,
                       const FT_Matrix*  matrix );


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    version                                                            */
  /*                                                                       */
  /* <Title>                                                               */
  /*    FreeType Version                                                   */
  /*                                                                       */
  /* <Abstract>                                                            */
  /*    Functions and macros related to FreeType versions.                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Note that those functions and macros are of limited use because    */
  /*    even a new release of FreeType with only documentation changes     */
  /*    increases the version number.                                      */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************
   *
   * @enum:
   *   FREETYPE_XXX
   *
   * @description:
   *   These three macros identify the FreeType source code version.
   *   Use @FT_Library_Version to access them at runtime.
   *
   * @values:
   *   FREETYPE_MAJOR :: The major version number.
   *   FREETYPE_MINOR :: The minor version number.
   *   FREETYPE_PATCH :: The patch level.
   *
   * @note:
   *   The version number of FreeType if built as a dynamic link library
   *   with the `libtool' package is _not_ controlled by these three
   *   macros.
   *
   */
#define FREETYPE_MAJOR  2
#define FREETYPE_MINOR  5
#define FREETYPE_PATCH  0


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Library_Version                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Return the version of the FreeType library being used.  This is    */
  /*    useful when dynamically linking to the library, since one cannot   */
  /*    use the macros @FREETYPE_MAJOR, @FREETYPE_MINOR, and               */
  /*    @FREETYPE_PATCH.                                                   */
  /*                                                                       */
  /* <Input>                                                               */
  /*    library :: A source library handle.                                */
  /*                                                                       */
  /* <Output>                                                              */
  /*    amajor  :: The major version number.                               */
  /*                                                                       */
  /*    aminor  :: The minor version number.                               */
  /*                                                                       */
  /*    apatch  :: The patch version number.                               */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The reason why this function takes a `library' argument is because */
  /*    certain programs implement library initialization in a custom way  */
  /*    that doesn't use @FT_Init_FreeType.                                */
  /*                                                                       */
  /*    In such cases, the library version might not be available before   */
  /*    the library object has been created.                               */
  /*                                                                       */
  FT_EXPORT( void )
  FT_Library_Version( FT_Library   library,
                      FT_Int      *amajor,
                      FT_Int      *aminor,
                      FT_Int      *apatch );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_CheckTrueTypePatents                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Parse all bytecode instructions of a TrueType font file to check   */
  /*    whether any of the patented opcodes are used.  This is only useful */
  /*    if you want to be able to use the unpatented hinter with           */
  /*    fonts that do *not* use these opcodes.                             */
  /*                                                                       */
  /*    Note that this function parses *all* glyph instructions in the     */
  /*    font file, which may be slow.                                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face :: A face handle.                                             */
  /*                                                                       */
  /* <Return>                                                              */
  /*    1~if this is a TrueType font that uses one of the patented         */
  /*    opcodes, 0~otherwise.                                              */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Since May 2010, TrueType hinting is no longer patented.            */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.5                                                              */
  /*                                                                       */
  FT_EXPORT( FT_Bool )
  FT_Face_CheckTrueTypePatents( FT_Face  face );


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_Face_SetUnpatentedHinting                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Enable or disable the unpatented hinter for a given face.          */
  /*    Only enable it if you have determined that the face doesn't        */
  /*    use any patented opcodes (see @FT_Face_CheckTrueTypePatents).      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face  :: A face handle.                                            */
  /*                                                                       */
  /*    value :: New boolean setting.                                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The old setting value.  This will always be false if this is not   */
  /*    an SFNT font, or if the unpatented hinter is not compiled in this  */
  /*    instance of the library.                                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Since May 2010, TrueType hinting is no longer patented.            */
  /*                                                                       */
  /* <Since>                                                               */
  /*    2.3.5                                                              */
  /*                                                                       */
  FT_EXPORT( FT_Bool )
  FT_Face_SetUnpatentedHinting( FT_Face  face,
                                FT_Bool  value );

  /* */


FT_END_HEADER

#endif /* __FREETYPE_H__ */


/* END */
