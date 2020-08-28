/****************************************************************************
 *
 * ftobjs.h
 *
 *   The FreeType private base classes (specification).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * This file contains the definition of all internal FreeType classes.
   *
   */


#ifndef FTOBJS_H_
#define FTOBJS_H_

#include <ft2build.h>
#include FT_RENDER_H
#include FT_SIZES_H
#include FT_LCD_FILTER_H
#include FT_INTERNAL_MEMORY_H
#include FT_INTERNAL_GLYPH_LOADER_H
#include FT_INTERNAL_DRIVER_H
#include FT_INTERNAL_AUTOHINT_H
#include FT_INTERNAL_SERVICE_H
#include FT_INTERNAL_CALC_H

#ifdef FT_CONFIG_OPTION_INCREMENTAL
#include FT_INCREMENTAL_H
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * Some generic definitions.
   */
#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE  0
#endif

#ifndef NULL
#define NULL  (void*)0
#endif


  /**************************************************************************
   *
   * The min and max functions missing in C.  As usual, be careful not to
   * write things like FT_MIN( a++, b++ ) to avoid side effects.
   */
#define FT_MIN( a, b )  ( (a) < (b) ? (a) : (b) )
#define FT_MAX( a, b )  ( (a) > (b) ? (a) : (b) )

#define FT_ABS( a )     ( (a) < 0 ? -(a) : (a) )

  /*
   * Approximate sqrt(x*x+y*y) using the `alpha max plus beta min' algorithm.
   * We use alpha = 1, beta = 3/8, giving us results with a largest error
   * less than 7% compared to the exact value.
   */
#define FT_HYPOT( x, y )                 \
          ( x = FT_ABS( x ),             \
            y = FT_ABS( y ),             \
            x > y ? x + ( 3 * y >> 3 )   \
                  : y + ( 3 * x >> 3 ) )

  /* we use FT_TYPEOF to suppress signedness compilation warnings */
#define FT_PAD_FLOOR( x, n )  ( (x) & ~FT_TYPEOF( x )( (n) - 1 ) )
#define FT_PAD_ROUND( x, n )  FT_PAD_FLOOR( (x) + (n) / 2, n )
#define FT_PAD_CEIL( x, n )   FT_PAD_FLOOR( (x) + (n) - 1, n )

#define FT_PIX_FLOOR( x )     ( (x) & ~FT_TYPEOF( x )63 )
#define FT_PIX_ROUND( x )     FT_PIX_FLOOR( (x) + 32 )
#define FT_PIX_CEIL( x )      FT_PIX_FLOOR( (x) + 63 )

  /* specialized versions (for signed values)                   */
  /* that don't produce run-time errors due to integer overflow */
#define FT_PAD_ROUND_LONG( x, n )  FT_PAD_FLOOR( ADD_LONG( (x), (n) / 2 ), \
                                                 n )
#define FT_PAD_CEIL_LONG( x, n )   FT_PAD_FLOOR( ADD_LONG( (x), (n) - 1 ), \
                                                 n )
#define FT_PIX_ROUND_LONG( x )     FT_PIX_FLOOR( ADD_LONG( (x), 32 ) )
#define FT_PIX_CEIL_LONG( x )      FT_PIX_FLOOR( ADD_LONG( (x), 63 ) )

#define FT_PAD_ROUND_INT32( x, n )  FT_PAD_FLOOR( ADD_INT32( (x), (n) / 2 ), \
                                                  n )
#define FT_PAD_CEIL_INT32( x, n )   FT_PAD_FLOOR( ADD_INT32( (x), (n) - 1 ), \
                                                  n )
#define FT_PIX_ROUND_INT32( x )     FT_PIX_FLOOR( ADD_INT32( (x), 32 ) )
#define FT_PIX_CEIL_INT32( x )      FT_PIX_FLOOR( ADD_INT32( (x), 63 ) )


  /*
   * character classification functions -- since these are used to parse font
   * files, we must not use those in <ctypes.h> which are locale-dependent
   */
#define  ft_isdigit( x )   ( ( (unsigned)(x) - '0' ) < 10U )

#define  ft_isxdigit( x )  ( ( (unsigned)(x) - '0' ) < 10U || \
                             ( (unsigned)(x) - 'a' ) < 6U  || \
                             ( (unsigned)(x) - 'A' ) < 6U  )

  /* the next two macros assume ASCII representation */
#define  ft_isupper( x )  ( ( (unsigned)(x) - 'A' ) < 26U )
#define  ft_islower( x )  ( ( (unsigned)(x) - 'a' ) < 26U )

#define  ft_isalpha( x )  ( ft_isupper( x ) || ft_islower( x ) )
#define  ft_isalnum( x )  ( ft_isdigit( x ) || ft_isalpha( x ) )


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                       C H A R M A P S                           ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

  /* handle to internal charmap object */
  typedef struct FT_CMapRec_*              FT_CMap;

  /* handle to charmap class structure */
  typedef const struct FT_CMap_ClassRec_*  FT_CMap_Class;

  /* internal charmap object structure */
  typedef struct  FT_CMapRec_
  {
    FT_CharMapRec  charmap;
    FT_CMap_Class  clazz;

  } FT_CMapRec;

  /* typecast any pointer to a charmap handle */
#define FT_CMAP( x )  ( (FT_CMap)( x ) )

  /* obvious macros */
#define FT_CMAP_PLATFORM_ID( x )  FT_CMAP( x )->charmap.platform_id
#define FT_CMAP_ENCODING_ID( x )  FT_CMAP( x )->charmap.encoding_id
#define FT_CMAP_ENCODING( x )     FT_CMAP( x )->charmap.encoding
#define FT_CMAP_FACE( x )         FT_CMAP( x )->charmap.face


  /* class method definitions */
  typedef FT_Error
  (*FT_CMap_InitFunc)( FT_CMap     cmap,
                       FT_Pointer  init_data );

  typedef void
  (*FT_CMap_DoneFunc)( FT_CMap  cmap );

  typedef FT_UInt
  (*FT_CMap_CharIndexFunc)( FT_CMap    cmap,
                            FT_UInt32  char_code );

  typedef FT_UInt
  (*FT_CMap_CharNextFunc)( FT_CMap     cmap,
                           FT_UInt32  *achar_code );

  typedef FT_UInt
  (*FT_CMap_CharVarIndexFunc)( FT_CMap    cmap,
                               FT_CMap    unicode_cmap,
                               FT_UInt32  char_code,
                               FT_UInt32  variant_selector );

  typedef FT_Int
  (*FT_CMap_CharVarIsDefaultFunc)( FT_CMap    cmap,
                                   FT_UInt32  char_code,
                                   FT_UInt32  variant_selector );

  typedef FT_UInt32 *
  (*FT_CMap_VariantListFunc)( FT_CMap    cmap,
                              FT_Memory  mem );

  typedef FT_UInt32 *
  (*FT_CMap_CharVariantListFunc)( FT_CMap    cmap,
                                  FT_Memory  mem,
                                  FT_UInt32  char_code );

  typedef FT_UInt32 *
  (*FT_CMap_VariantCharListFunc)( FT_CMap    cmap,
                                  FT_Memory  mem,
                                  FT_UInt32  variant_selector );


  typedef struct  FT_CMap_ClassRec_
  {
    FT_ULong               size;

    FT_CMap_InitFunc       init;
    FT_CMap_DoneFunc       done;
    FT_CMap_CharIndexFunc  char_index;
    FT_CMap_CharNextFunc   char_next;

    /* Subsequent entries are special ones for format 14 -- the variant */
    /* selector subtable which behaves like no other                    */

    FT_CMap_CharVarIndexFunc      char_var_index;
    FT_CMap_CharVarIsDefaultFunc  char_var_default;
    FT_CMap_VariantListFunc       variant_list;
    FT_CMap_CharVariantListFunc   charvariant_list;
    FT_CMap_VariantCharListFunc   variantchar_list;

  } FT_CMap_ClassRec;


#define FT_DECLARE_CMAP_CLASS( class_ )              \
  FT_CALLBACK_TABLE const  FT_CMap_ClassRec class_;

#define FT_DEFINE_CMAP_CLASS(       \
          class_,                   \
          size_,                    \
          init_,                    \
          done_,                    \
          char_index_,              \
          char_next_,               \
          char_var_index_,          \
          char_var_default_,        \
          variant_list_,            \
          charvariant_list_,        \
          variantchar_list_ )       \
  FT_CALLBACK_TABLE_DEF             \
  const FT_CMap_ClassRec  class_ =  \
  {                                 \
    size_,                          \
    init_,                          \
    done_,                          \
    char_index_,                    \
    char_next_,                     \
    char_var_index_,                \
    char_var_default_,              \
    variant_list_,                  \
    charvariant_list_,              \
    variantchar_list_               \
  };


  /* create a new charmap and add it to charmap->face */
  FT_BASE( FT_Error )
  FT_CMap_New( FT_CMap_Class  clazz,
               FT_Pointer     init_data,
               FT_CharMap     charmap,
               FT_CMap       *acmap );

  /* destroy a charmap and remove it from face's list */
  FT_BASE( void )
  FT_CMap_Done( FT_CMap  cmap );


  /* add LCD padding to CBox */
  FT_BASE( void )
  ft_lcd_padding( FT_BBox*        cbox,
                  FT_GlyphSlot    slot,
                  FT_Render_Mode  mode );

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

  typedef void  (*FT_Bitmap_LcdFilterFunc)( FT_Bitmap*      bitmap,
                                            FT_Byte*        weights );


  /* This is the default LCD filter, an in-place, 5-tap FIR filter. */
  FT_BASE( void )
  ft_lcd_filter_fir( FT_Bitmap*           bitmap,
                     FT_LcdFiveTapFilter  weights );

#endif /* FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

  /**************************************************************************
   *
   * @struct:
   *   FT_Face_InternalRec
   *
   * @description:
   *   This structure contains the internal fields of each FT_Face object.
   *   These fields may change between different releases of FreeType.
   *
   * @fields:
   *   max_points ::
   *     The maximum number of points used to store the vectorial outline of
   *     any glyph in this face.  If this value cannot be known in advance,
   *     or if the face isn't scalable, this should be set to 0.  Only
   *     relevant for scalable formats.
   *
   *   max_contours ::
   *     The maximum number of contours used to store the vectorial outline
   *     of any glyph in this face.  If this value cannot be known in
   *     advance, or if the face isn't scalable, this should be set to 0.
   *     Only relevant for scalable formats.
   *
   *   transform_matrix ::
   *     A 2x2 matrix of 16.16 coefficients used to transform glyph outlines
   *     after they are loaded from the font.  Only used by the convenience
   *     functions.
   *
   *   transform_delta ::
   *     A translation vector used to transform glyph outlines after they are
   *     loaded from the font.  Only used by the convenience functions.
   *
   *   transform_flags ::
   *     Some flags used to classify the transform.  Only used by the
   *     convenience functions.
   *
   *   services ::
   *     A cache for frequently used services.  It should be only accessed
   *     with the macro `FT_FACE_LOOKUP_SERVICE`.
   *
   *   incremental_interface ::
   *     If non-null, the interface through which glyph data and metrics are
   *     loaded incrementally for faces that do not provide all of this data
   *     when first opened.  This field exists only if
   *     @FT_CONFIG_OPTION_INCREMENTAL is defined.
   *
   *   no_stem_darkening ::
   *     Overrides the module-level default, see @stem-darkening[cff], for
   *     example.  FALSE and TRUE toggle stem darkening on and off,
   *     respectively, value~-1 means to use the module/driver default.
   *
   *   random_seed ::
   *     If positive, override the seed value for the CFF 'random' operator.
   *     Value~0 means to use the font's value.  Value~-1 means to use the
   *     CFF driver's default.
   *
   *   lcd_weights ::
   *   lcd_filter_func ::
   *     These fields specify the LCD filtering weights and callback function
   *     for ClearType-style subpixel rendering.
   *
   *   refcount ::
   *     A counter initialized to~1 at the time an @FT_Face structure is
   *     created.  @FT_Reference_Face increments this counter, and
   *     @FT_Done_Face only destroys a face if the counter is~1, otherwise it
   *     simply decrements it.
   */
  typedef struct  FT_Face_InternalRec_
  {
    FT_Matrix  transform_matrix;
    FT_Vector  transform_delta;
    FT_Int     transform_flags;

    FT_ServiceCacheRec  services;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    FT_Incremental_InterfaceRec*  incremental_interface;
#endif

    FT_Char              no_stem_darkening;
    FT_Int32             random_seed;

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    FT_LcdFiveTapFilter      lcd_weights;      /* filter weights, if any */
    FT_Bitmap_LcdFilterFunc  lcd_filter_func;  /* filtering callback     */
#endif

    FT_Int  refcount;

  } FT_Face_InternalRec;


  /**************************************************************************
   *
   * @struct:
   *   FT_Slot_InternalRec
   *
   * @description:
   *   This structure contains the internal fields of each FT_GlyphSlot
   *   object.  These fields may change between different releases of
   *   FreeType.
   *
   * @fields:
   *   loader ::
   *     The glyph loader object used to load outlines into the glyph slot.
   *
   *   flags ::
   *     Possible values are zero or FT_GLYPH_OWN_BITMAP.  The latter
   *     indicates that the FT_GlyphSlot structure owns the bitmap buffer.
   *
   *   glyph_transformed ::
   *     Boolean.  Set to TRUE when the loaded glyph must be transformed
   *     through a specific font transformation.  This is _not_ the same as
   *     the face transform set through FT_Set_Transform().
   *
   *   glyph_matrix ::
   *     The 2x2 matrix corresponding to the glyph transformation, if
   *     necessary.
   *
   *   glyph_delta ::
   *     The 2d translation vector corresponding to the glyph transformation,
   *     if necessary.
   *
   *   glyph_hints ::
   *     Format-specific glyph hints management.
   *
   *   load_flags ::
   *     The load flags passed as an argument to @FT_Load_Glyph while
   *     initializing the glyph slot.
   */

#define FT_GLYPH_OWN_BITMAP  0x1U

  typedef struct  FT_Slot_InternalRec_
  {
    FT_GlyphLoader  loader;
    FT_UInt         flags;
    FT_Bool         glyph_transformed;
    FT_Matrix       glyph_matrix;
    FT_Vector       glyph_delta;
    void*           glyph_hints;

    FT_Int32        load_flags;

  } FT_GlyphSlot_InternalRec;


  /**************************************************************************
   *
   * @struct:
   *   FT_Size_InternalRec
   *
   * @description:
   *   This structure contains the internal fields of each FT_Size object.
   *
   * @fields:
   *   module_data ::
   *     Data specific to a driver module.
   *
   *   autohint_mode ::
   *     The used auto-hinting mode.
   *
   *   autohint_metrics ::
   *     Metrics used by the auto-hinter.
   *
   */

  typedef struct  FT_Size_InternalRec_
  {
    void*  module_data;

    FT_Render_Mode   autohint_mode;
    FT_Size_Metrics  autohint_metrics;

  } FT_Size_InternalRec;


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                         M O D U L E S                           ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @struct:
   *   FT_ModuleRec
   *
   * @description:
   *   A module object instance.
   *
   * @fields:
   *   clazz ::
   *     A pointer to the module's class.
   *
   *   library ::
   *     A handle to the parent library object.
   *
   *   memory ::
   *     A handle to the memory manager.
   */
  typedef struct  FT_ModuleRec_
  {
    FT_Module_Class*  clazz;
    FT_Library        library;
    FT_Memory         memory;

  } FT_ModuleRec;


  /* typecast an object to an FT_Module */
#define FT_MODULE( x )  ( (FT_Module)(x) )

#define FT_MODULE_CLASS( x )    FT_MODULE( x )->clazz
#define FT_MODULE_LIBRARY( x )  FT_MODULE( x )->library
#define FT_MODULE_MEMORY( x )   FT_MODULE( x )->memory


#define FT_MODULE_IS_DRIVER( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                    FT_MODULE_FONT_DRIVER )

#define FT_MODULE_IS_RENDERER( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                      FT_MODULE_RENDERER )

#define FT_MODULE_IS_HINTER( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                    FT_MODULE_HINTER )

#define FT_MODULE_IS_STYLER( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                    FT_MODULE_STYLER )

#define FT_DRIVER_IS_SCALABLE( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                      FT_MODULE_DRIVER_SCALABLE )

#define FT_DRIVER_USES_OUTLINES( x )  !( FT_MODULE_CLASS( x )->module_flags & \
                                         FT_MODULE_DRIVER_NO_OUTLINES )

#define FT_DRIVER_HAS_HINTER( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                     FT_MODULE_DRIVER_HAS_HINTER )

#define FT_DRIVER_HINTS_LIGHTLY( x )  ( FT_MODULE_CLASS( x )->module_flags & \
                                        FT_MODULE_DRIVER_HINTS_LIGHTLY )


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Module_Interface
   *
   * @description:
   *   Finds a module and returns its specific interface as a typeless
   *   pointer.
   *
   * @input:
   *   library ::
   *     A handle to the library object.
   *
   *   module_name ::
   *     The module's name (as an ASCII string).
   *
   * @return:
   *   A module-specific interface if available, 0 otherwise.
   *
   * @note:
   *   You should better be familiar with FreeType internals to know which
   *   module to look for, and what its interface is :-)
   */
  FT_BASE( const void* )
  FT_Get_Module_Interface( FT_Library   library,
                           const char*  mod_name );

  FT_BASE( FT_Pointer )
  ft_module_get_service( FT_Module    module,
                         const char*  service_id,
                         FT_Bool      global );

#ifdef FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES
  FT_BASE( FT_Error )
  ft_property_string_set( FT_Library        library,
                          const FT_String*  module_name,
                          const FT_String*  property_name,
                          FT_String*        value );
#endif

  /* */


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****   F A C E,   S I Z E   &   G L Y P H   S L O T   O B J E C T S  ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

  /* a few macros used to perform easy typecasts with minimal brain damage */

#define FT_FACE( x )          ( (FT_Face)(x) )
#define FT_SIZE( x )          ( (FT_Size)(x) )
#define FT_SLOT( x )          ( (FT_GlyphSlot)(x) )

#define FT_FACE_DRIVER( x )   FT_FACE( x )->driver
#define FT_FACE_LIBRARY( x )  FT_FACE_DRIVER( x )->root.library
#define FT_FACE_MEMORY( x )   FT_FACE( x )->memory
#define FT_FACE_STREAM( x )   FT_FACE( x )->stream

#define FT_SIZE_FACE( x )     FT_SIZE( x )->face
#define FT_SLOT_FACE( x )     FT_SLOT( x )->face

#define FT_FACE_SLOT( x )     FT_FACE( x )->glyph
#define FT_FACE_SIZE( x )     FT_FACE( x )->size


  /**************************************************************************
   *
   * @function:
   *   FT_New_GlyphSlot
   *
   * @description:
   *   It is sometimes useful to have more than one glyph slot for a given
   *   face object.  This function is used to create additional slots.  All
   *   of them are automatically discarded when the face is destroyed.
   *
   * @input:
   *   face ::
   *     A handle to a parent face object.
   *
   * @output:
   *   aslot ::
   *     A handle to a new glyph slot object.
   *
   * @return:
   *   FreeType error code.  0 means success.
   */
  FT_BASE( FT_Error )
  FT_New_GlyphSlot( FT_Face        face,
                    FT_GlyphSlot  *aslot );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_GlyphSlot
   *
   * @description:
   *   Destroys a given glyph slot.  Remember however that all slots are
   *   automatically destroyed with its parent.  Using this function is not
   *   always mandatory.
   *
   * @input:
   *   slot ::
   *     A handle to a target glyph slot.
   */
  FT_BASE( void )
  FT_Done_GlyphSlot( FT_GlyphSlot  slot );

 /* */

#define FT_REQUEST_WIDTH( req )                                            \
          ( (req)->horiResolution                                          \
              ? ( (req)->width * (FT_Pos)(req)->horiResolution + 36 ) / 72 \
              : (req)->width )

#define FT_REQUEST_HEIGHT( req )                                            \
          ( (req)->vertResolution                                           \
              ? ( (req)->height * (FT_Pos)(req)->vertResolution + 36 ) / 72 \
              : (req)->height )


  /* Set the metrics according to a bitmap strike. */
  FT_BASE( void )
  FT_Select_Metrics( FT_Face   face,
                     FT_ULong  strike_index );


  /* Set the metrics according to a size request. */
  FT_BASE( void )
  FT_Request_Metrics( FT_Face          face,
                      FT_Size_Request  req );


  /* Match a size request against `available_sizes'. */
  FT_BASE( FT_Error )
  FT_Match_Size( FT_Face          face,
                 FT_Size_Request  req,
                 FT_Bool          ignore_width,
                 FT_ULong*        size_index );


  /* Use the horizontal metrics to synthesize the vertical metrics. */
  /* If `advance' is zero, it is also synthesized.                  */
  FT_BASE( void )
  ft_synthesize_vertical_metrics( FT_Glyph_Metrics*  metrics,
                                  FT_Pos             advance );


  /* Free the bitmap of a given glyphslot when needed (i.e., only when it */
  /* was allocated with ft_glyphslot_alloc_bitmap).                       */
  FT_BASE( void )
  ft_glyphslot_free_bitmap( FT_GlyphSlot  slot );


  /* Preset bitmap metrics of an outline glyphslot prior to rendering */
  /* and check whether the truncated bbox is too large for rendering. */
  FT_BASE( FT_Bool )
  ft_glyphslot_preset_bitmap( FT_GlyphSlot      slot,
                              FT_Render_Mode    mode,
                              const FT_Vector*  origin );

  /* Allocate a new bitmap buffer in a glyph slot. */
  FT_BASE( FT_Error )
  ft_glyphslot_alloc_bitmap( FT_GlyphSlot  slot,
                             FT_ULong      size );


  /* Set the bitmap buffer in a glyph slot to a given pointer.  The buffer */
  /* will not be freed by a later call to ft_glyphslot_free_bitmap.        */
  FT_BASE( void )
  ft_glyphslot_set_bitmap( FT_GlyphSlot  slot,
                           FT_Byte*      buffer );


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                        R E N D E R E R S                        ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


#define FT_RENDERER( x )       ( (FT_Renderer)(x) )
#define FT_GLYPH( x )          ( (FT_Glyph)(x) )
#define FT_BITMAP_GLYPH( x )   ( (FT_BitmapGlyph)(x) )
#define FT_OUTLINE_GLYPH( x )  ( (FT_OutlineGlyph)(x) )


  typedef struct  FT_RendererRec_
  {
    FT_ModuleRec            root;
    FT_Renderer_Class*      clazz;
    FT_Glyph_Format         glyph_format;
    FT_Glyph_Class          glyph_class;

    FT_Raster               raster;
    FT_Raster_Render_Func   raster_render;
    FT_Renderer_RenderFunc  render;

  } FT_RendererRec;


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                    F O N T   D R I V E R S                      ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /* typecast a module into a driver easily */
#define FT_DRIVER( x )  ( (FT_Driver)(x) )

  /* typecast a module as a driver, and get its driver class */
#define FT_DRIVER_CLASS( x )  FT_DRIVER( x )->clazz


  /**************************************************************************
   *
   * @struct:
   *   FT_DriverRec
   *
   * @description:
   *   The root font driver class.  A font driver is responsible for managing
   *   and loading font files of a given format.
   *
   * @fields:
   *   root ::
   *     Contains the fields of the root module class.
   *
   *   clazz ::
   *     A pointer to the font driver's class.  Note that this is NOT
   *     root.clazz.  'class' wasn't used as it is a reserved word in C++.
   *
   *   faces_list ::
   *     The list of faces currently opened by this driver.
   *
   *   glyph_loader ::
   *     Unused.  Used to be glyph loader for all faces managed by this
   *     driver.
   */
  typedef struct  FT_DriverRec_
  {
    FT_ModuleRec     root;
    FT_Driver_Class  clazz;
    FT_ListRec       faces_list;
    FT_GlyphLoader   glyph_loader;

  } FT_DriverRec;


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                       L I B R A R I E S                         ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @struct:
   *   FT_LibraryRec
   *
   * @description:
   *   The FreeType library class.  This is the root of all FreeType data.
   *   Use FT_New_Library() to create a library object, and FT_Done_Library()
   *   to discard it and all child objects.
   *
   * @fields:
   *   memory ::
   *     The library's memory object.  Manages memory allocation.
   *
   *   version_major ::
   *     The major version number of the library.
   *
   *   version_minor ::
   *     The minor version number of the library.
   *
   *   version_patch ::
   *     The current patch level of the library.
   *
   *   num_modules ::
   *     The number of modules currently registered within this library.
   *     This is set to 0 for new libraries.  New modules are added through
   *     the FT_Add_Module() API function.
   *
   *   modules ::
   *     A table used to store handles to the currently registered
   *     modules. Note that each font driver contains a list of its opened
   *     faces.
   *
   *   renderers ::
   *     The list of renderers currently registered within the library.
   *
   *   cur_renderer ::
   *     The current outline renderer.  This is a shortcut used to avoid
   *     parsing the list on each call to FT_Outline_Render().  It is a
   *     handle to the current renderer for the FT_GLYPH_FORMAT_OUTLINE
   *     format.
   *
   *   auto_hinter ::
   *     The auto-hinter module interface.
   *
   *   debug_hooks ::
   *     An array of four function pointers that allow debuggers to hook into
   *     a font format's interpreter.  Currently, only the TrueType bytecode
   *     debugger uses this.
   *
   *   lcd_weights ::
   *     The LCD filter weights for ClearType-style subpixel rendering.
   *
   *   lcd_filter_func ::
   *     The LCD filtering callback function for for ClearType-style subpixel
   *     rendering.
   *
   *   lcd_geometry ::
   *     This array specifies LCD subpixel geometry and controls Harmony LCD
   *     rendering technique, alternative to ClearType.
   *
   *   pic_container ::
   *     Contains global structs and tables, instead of defining them
   *     globally.
   *
   *   refcount ::
   *     A counter initialized to~1 at the time an @FT_Library structure is
   *     created.  @FT_Reference_Library increments this counter, and
   *     @FT_Done_Library only destroys a library if the counter is~1,
   *     otherwise it simply decrements it.
   */
  typedef struct  FT_LibraryRec_
  {
    FT_Memory          memory;           /* library's memory manager */

    FT_Int             version_major;
    FT_Int             version_minor;
    FT_Int             version_patch;

    FT_UInt            num_modules;
    FT_Module          modules[FT_MAX_MODULES];  /* module objects  */

    FT_ListRec         renderers;        /* list of renderers        */
    FT_Renderer        cur_renderer;     /* current outline renderer */
    FT_Module          auto_hinter;

    FT_DebugHook_Func  debug_hooks[4];

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    FT_LcdFiveTapFilter      lcd_weights;      /* filter weights, if any */
    FT_Bitmap_LcdFilterFunc  lcd_filter_func;  /* filtering callback     */
#else
    FT_Vector                lcd_geometry[3];  /* RGB subpixel positions */
#endif

    FT_Int             refcount;

  } FT_LibraryRec;


  FT_BASE( FT_Renderer )
  FT_Lookup_Renderer( FT_Library       library,
                      FT_Glyph_Format  format,
                      FT_ListNode*     node );

  FT_BASE( FT_Error )
  FT_Render_Glyph_Internal( FT_Library      library,
                            FT_GlyphSlot    slot,
                            FT_Render_Mode  render_mode );

  typedef const char*
  (*FT_Face_GetPostscriptNameFunc)( FT_Face  face );

  typedef FT_Error
  (*FT_Face_GetGlyphNameFunc)( FT_Face     face,
                               FT_UInt     glyph_index,
                               FT_Pointer  buffer,
                               FT_UInt     buffer_max );

  typedef FT_UInt
  (*FT_Face_GetGlyphNameIndexFunc)( FT_Face           face,
                                    const FT_String*  glyph_name );


#ifndef FT_CONFIG_OPTION_NO_DEFAULT_SYSTEM

  /**************************************************************************
   *
   * @function:
   *   FT_New_Memory
   *
   * @description:
   *   Creates a new memory object.
   *
   * @return:
   *   A pointer to the new memory object.  0 in case of error.
   */
  FT_BASE( FT_Memory )
  FT_New_Memory( void );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_Memory
   *
   * @description:
   *   Discards memory manager.
   *
   * @input:
   *   memory ::
   *     A handle to the memory manager.
   */
  FT_BASE( void )
  FT_Done_Memory( FT_Memory  memory );

#endif /* !FT_CONFIG_OPTION_NO_DEFAULT_SYSTEM */


  /* Define default raster's interface.  The default raster is located in  */
  /* `src/base/ftraster.c'.                                                */
  /*                                                                       */
  /* Client applications can register new rasters through the              */
  /* FT_Set_Raster() API.                                                  */

#ifndef FT_NO_DEFAULT_RASTER
  FT_EXPORT_VAR( FT_Raster_Funcs )  ft_default_raster;
#endif


  /**************************************************************************
   *
   * @macro:
   *   FT_DEFINE_OUTLINE_FUNCS
   *
   * @description:
   *   Used to initialize an instance of FT_Outline_Funcs struct.  The struct
   *   will be allocated in the global scope (or the scope where the macro is
   *   used).
   */
#define FT_DEFINE_OUTLINE_FUNCS(           \
          class_,                          \
          move_to_,                        \
          line_to_,                        \
          conic_to_,                       \
          cubic_to_,                       \
          shift_,                          \
          delta_ )                         \
  static const  FT_Outline_Funcs class_ =  \
  {                                        \
    move_to_,                              \
    line_to_,                              \
    conic_to_,                             \
    cubic_to_,                             \
    shift_,                                \
    delta_                                 \
  };


  /**************************************************************************
   *
   * @macro:
   *   FT_DEFINE_RASTER_FUNCS
   *
   * @description:
   *   Used to initialize an instance of FT_Raster_Funcs struct.  The struct
   *   will be allocated in the global scope (or the scope where the macro is
   *   used).
   */
#define FT_DEFINE_RASTER_FUNCS(    \
          class_,                  \
          glyph_format_,           \
          raster_new_,             \
          raster_reset_,           \
          raster_set_mode_,        \
          raster_render_,          \
          raster_done_ )           \
  const FT_Raster_Funcs  class_ =  \
  {                                \
    glyph_format_,                 \
    raster_new_,                   \
    raster_reset_,                 \
    raster_set_mode_,              \
    raster_render_,                \
    raster_done_                   \
  };



  /**************************************************************************
   *
   * @macro:
   *   FT_DEFINE_GLYPH
   *
   * @description:
   *   The struct will be allocated in the global scope (or the scope where
   *   the macro is used).
   */
#define FT_DEFINE_GLYPH(          \
          class_,                 \
          size_,                  \
          format_,                \
          init_,                  \
          done_,                  \
          copy_,                  \
          transform_,             \
          bbox_,                  \
          prepare_ )              \
  FT_CALLBACK_TABLE_DEF           \
  const FT_Glyph_Class  class_ =  \
  {                               \
    size_,                        \
    format_,                      \
    init_,                        \
    done_,                        \
    copy_,                        \
    transform_,                   \
    bbox_,                        \
    prepare_                      \
  };


  /**************************************************************************
   *
   * @macro:
   *   FT_DECLARE_RENDERER
   *
   * @description:
   *   Used to create a forward declaration of a FT_Renderer_Class struct
   *   instance.
   *
   * @macro:
   *   FT_DEFINE_RENDERER
   *
   * @description:
   *   Used to initialize an instance of FT_Renderer_Class struct.
   *
   *   The struct will be allocated in the global scope (or the scope where
   *   the macro is used).
   */
#define FT_DECLARE_RENDERER( class_ )               \
  FT_EXPORT_VAR( const FT_Renderer_Class ) class_;

#define FT_DEFINE_RENDERER(                  \
          class_,                            \
          flags_,                            \
          size_,                             \
          name_,                             \
          version_,                          \
          requires_,                         \
          interface_,                        \
          init_,                             \
          done_,                             \
          get_interface_,                    \
          glyph_format_,                     \
          render_glyph_,                     \
          transform_glyph_,                  \
          get_glyph_cbox_,                   \
          set_mode_,                         \
          raster_class_ )                    \
  FT_CALLBACK_TABLE_DEF                      \
  const FT_Renderer_Class  class_ =          \
  {                                          \
    FT_DEFINE_ROOT_MODULE( flags_,           \
                           size_,            \
                           name_,            \
                           version_,         \
                           requires_,        \
                           interface_,       \
                           init_,            \
                           done_,            \
                           get_interface_ )  \
    glyph_format_,                           \
                                             \
    render_glyph_,                           \
    transform_glyph_,                        \
    get_glyph_cbox_,                         \
    set_mode_,                               \
                                             \
    raster_class_                            \
  };


  /**************************************************************************
   *
   * @macro:
   *   FT_DECLARE_MODULE
   *
   * @description:
   *   Used to create a forward declaration of a FT_Module_Class struct
   *   instance.
   *
   * @macro:
   *   FT_DEFINE_MODULE
   *
   * @description:
   *   Used to initialize an instance of an FT_Module_Class struct.
   *
   *   The struct will be allocated in the global scope (or the scope where
   *   the macro is used).
   *
   * @macro:
   *   FT_DEFINE_ROOT_MODULE
   *
   * @description:
   *   Used to initialize an instance of an FT_Module_Class struct inside
   *   another struct that contains it or in a function that initializes that
   *   containing struct.
   */
#define FT_DECLARE_MODULE( class_ )  \
  FT_CALLBACK_TABLE                  \
  const FT_Module_Class  class_;

#define FT_DEFINE_ROOT_MODULE(  \
          flags_,               \
          size_,                \
          name_,                \
          version_,             \
          requires_,            \
          interface_,           \
          init_,                \
          done_,                \
          get_interface_ )      \
  {                             \
    flags_,                     \
    size_,                      \
                                \
    name_,                      \
    version_,                   \
    requires_,                  \
                                \
    interface_,                 \
                                \
    init_,                      \
    done_,                      \
    get_interface_,             \
  },

#define FT_DEFINE_MODULE(         \
          class_,                 \
          flags_,                 \
          size_,                  \
          name_,                  \
          version_,               \
          requires_,              \
          interface_,             \
          init_,                  \
          done_,                  \
          get_interface_ )        \
  FT_CALLBACK_TABLE_DEF           \
  const FT_Module_Class class_ =  \
  {                               \
    flags_,                       \
    size_,                        \
                                  \
    name_,                        \
    version_,                     \
    requires_,                    \
                                  \
    interface_,                   \
                                  \
    init_,                        \
    done_,                        \
    get_interface_,               \
  };


FT_END_HEADER

#endif /* FTOBJS_H_ */


/* END */
