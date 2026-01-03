/*  pcfdrivr.c

    FreeType font driver for pcf files

    Copyright (C) 2000-2004, 2006-2011, 2013, 2014 by
    Francesco Zappa Nardelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/



#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/ftgzip.h>
#include <freetype/ftlzw.h>
#include <freetype/ftbzip2.h>
#include <freetype/fterrors.h>
#include <freetype/ftbdf.h>
#include <freetype/ttnameid.h>

#include "pcf.h"
#include "pcfdrivr.h"
#include "pcfread.h"

#include "pcferror.h"
#include "pcfutil.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  pcfread

#include <freetype/internal/services/svbdf.h>
#include <freetype/internal/services/svfntfmt.h>
#include <freetype/internal/services/svprop.h>
#include <freetype/ftdriver.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  pcfdriver


  /*
   * This file uses X11 terminology for PCF data; an `encoding' in X11 speak
   * is the same as a `character code' in FreeType speak.
   */
  typedef struct  PCF_CMapRec_
  {
    FT_CMapRec  root;
    PCF_Enc     enc;

  } PCF_CMapRec, *PCF_CMap;


  FT_CALLBACK_DEF( FT_Error )
  pcf_cmap_init( FT_CMap     cmap,       /* PCF_CMap */
                 FT_Pointer  init_data )
  {
    PCF_CMap  pcfcmap = (PCF_CMap)cmap;
    PCF_Face  face    = (PCF_Face)FT_CMAP_FACE( cmap );

    FT_UNUSED( init_data );


    pcfcmap->enc = &face->enc;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  pcf_cmap_done( FT_CMap  cmap )         /* PCF_CMap */
  {
    PCF_CMap  pcfcmap = (PCF_CMap)cmap;


    pcfcmap->enc = NULL;
  }


  FT_CALLBACK_DEF( FT_UInt )
  pcf_cmap_char_index( FT_CMap    cmap,      /* PCF_CMap */
                       FT_UInt32  charcode )
  {
    PCF_Enc  enc = ( (PCF_CMap)cmap )->enc;

    FT_UInt32  i = ( charcode >> 8   ) - enc->firstRow;
    FT_UInt32  j = ( charcode & 0xFF ) - enc->firstCol;
    FT_UInt32  h = enc->lastRow - enc->firstRow + 1;
    FT_UInt32  w = enc->lastCol - enc->firstCol + 1;


    /* wrapped around "negative" values are also rejected */
    if ( i >= h || j >= w )
      return 0;

    return (FT_UInt)enc->offset[i * w + j];
  }


  FT_CALLBACK_DEF( FT_UInt )
  pcf_cmap_char_next( FT_CMap     cmap,       /* PCF_CMap */
                      FT_UInt32  *acharcode )
  {
    PCF_Enc    enc = ( (PCF_CMap)cmap )->enc;
    FT_UInt32  charcode = *acharcode + 1;

    FT_UInt32  i = ( charcode >> 8   ) - enc->firstRow;
    FT_UInt32  j = ( charcode & 0xFF ) - enc->firstCol;
    FT_UInt32  h = enc->lastRow - enc->firstRow + 1;
    FT_UInt32  w = enc->lastCol - enc->firstCol + 1;

    FT_UInt  result = 0;


    /* adjust wrapped around "negative" values */
    if ( (FT_Int32)i < 0 )
      i = 0;
    if ( (FT_Int32)j < 0 )
      j = 0;

    for ( ; i < h; i++, j = 0 )
      for ( ; j < w; j++ )
      {
        result = (FT_UInt)enc->offset[i * w + j];
        if ( result != 0xFFFFU )
          goto Exit;
      }

  Exit:
    *acharcode = ( ( i + enc->firstRow ) << 8 ) | ( j + enc->firstCol );

    return result;
  }


  static
  const FT_CMap_ClassRec  pcf_cmap_class =
  {
    sizeof ( PCF_CMapRec ),
    pcf_cmap_init,
    pcf_cmap_done,
    pcf_cmap_char_index,
    pcf_cmap_char_next,

    NULL, NULL, NULL, NULL, NULL
  };


  FT_CALLBACK_DEF( void )
  PCF_Face_Done( FT_Face  face )    /* PCF_Face */
  {
    PCF_Face   pcfface = (PCF_Face)face;
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    FT_FREE( pcfface->metrics );
    FT_FREE( pcfface->enc.offset );

    /* free properties */
    if ( pcfface->properties )
    {
      FT_Int  i;


      for ( i = 0; i < pcfface->nprops; i++ )
      {
        PCF_Property  prop = &pcfface->properties[i];


        if ( prop )
        {
          FT_FREE( prop->name );
          if ( prop->isString )
            FT_FREE( prop->value.atom );
        }
      }

      FT_FREE( pcfface->properties );
    }

    FT_FREE( pcfface->toc.tables );
    FT_FREE( face->family_name );
    FT_FREE( face->style_name );
    FT_FREE( face->available_sizes );
    FT_FREE( pcfface->charset_encoding );
    FT_FREE( pcfface->charset_registry );

    /* close compressed stream if any */
    if ( face->stream == &pcfface->comp_stream )
    {
      FT_Stream_Close( &pcfface->comp_stream );
      face->stream = pcfface->comp_source;
    }
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Face_Init( FT_Stream      stream,
                 FT_Face        face,       /* PCF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    PCF_Face  pcfface = (PCF_Face)face;
    FT_Error  error;

    FT_UNUSED( num_params );
    FT_UNUSED( params );


    FT_TRACE2(( "PCF driver\n" ));

    error = pcf_load_font( stream, pcfface, face_index );
    if ( error )
    {
      PCF_Face_Done( face );

#if defined( FT_CONFIG_OPTION_USE_ZLIB )  || \
    defined( FT_CONFIG_OPTION_USE_LZW )   || \
    defined( FT_CONFIG_OPTION_USE_BZIP2 )

#ifdef FT_CONFIG_OPTION_USE_ZLIB
      {
        FT_Error  error2;


        /* this didn't work, try gzip support! */
        FT_TRACE2(( "  ... try gzip stream\n" ));
        error2 = FT_Stream_OpenGzip( &pcfface->comp_stream, stream );
        if ( FT_ERR_EQ( error2, Unimplemented_Feature ) )
          goto Fail;

        error = error2;
      }
#endif /* FT_CONFIG_OPTION_USE_ZLIB */

#ifdef FT_CONFIG_OPTION_USE_LZW
      if ( error )
      {
        FT_Error  error3;


        /* this didn't work, try LZW support! */
        FT_TRACE2(( "  ... try LZW stream\n" ));
        error3 = FT_Stream_OpenLZW( &pcfface->comp_stream, stream );
        if ( FT_ERR_EQ( error3, Unimplemented_Feature ) )
          goto Fail;

        error = error3;
      }
#endif /* FT_CONFIG_OPTION_USE_LZW */

#ifdef FT_CONFIG_OPTION_USE_BZIP2
      if ( error )
      {
        FT_Error  error4;


        /* this didn't work, try Bzip2 support! */
        FT_TRACE2(( "  ... try Bzip2 stream\n" ));
        error4 = FT_Stream_OpenBzip2( &pcfface->comp_stream, stream );
        if ( FT_ERR_EQ( error4, Unimplemented_Feature ) )
          goto Fail;

        error = error4;
      }
#endif /* FT_CONFIG_OPTION_USE_BZIP2 */

      if ( error )
        goto Fail;

      pcfface->comp_source = stream;
      face->stream         = &pcfface->comp_stream;

      stream = face->stream;

      error = pcf_load_font( stream, pcfface, face_index );
      if ( error )
        goto Fail;

#else /* !(FT_CONFIG_OPTION_USE_ZLIB ||
           FT_CONFIG_OPTION_USE_LZW ||
           FT_CONFIG_OPTION_USE_BZIP2) */

      goto Fail;

#endif
    }

    /* PCF cannot have multiple faces in a single font file.
     * XXX: A non-zero face_index is already an invalid argument, but
     *      Type1, Type42 drivers have a convention to return
     *      an invalid argument error when the font could be
     *      opened by the specified driver.
     */
    if ( face_index < 0 )
      goto Exit;
    else if ( face_index > 0 && ( face_index & 0xFFFF ) > 0 )
    {
      FT_ERROR(( "PCF_Face_Init: invalid face index\n" ));
      PCF_Face_Done( face );
      return FT_THROW( Invalid_Argument );
    }

    /* set up charmap */
    {
      FT_String  *charset_registry = pcfface->charset_registry;
      FT_String  *charset_encoding = pcfface->charset_encoding;
      FT_Bool     unicode_charmap  = 0;


      if ( charset_registry && charset_encoding )
      {
        char*  s = charset_registry;


        /* Uh, oh, compare first letters manually to avoid dependency
           on locales. */
        if ( ( s[0] == 'i' || s[0] == 'I' ) &&
             ( s[1] == 's' || s[1] == 'S' ) &&
             ( s[2] == 'o' || s[2] == 'O' ) )
        {
          s += 3;
          if ( !ft_strcmp( s, "10646" )                         ||
               ( !ft_strcmp( s, "8859" )                      &&
                 !ft_strcmp( pcfface->charset_encoding, "1" ) ) )
            unicode_charmap = 1;
          /* another name for ASCII */
          else if ( !ft_strcmp( s, "646.1991" )                    &&
                    !ft_strcmp( pcfface->charset_encoding, "IRV" ) )
            unicode_charmap = 1;
        }
      }

      {
        FT_CharMapRec  charmap;


        charmap.face        = face;
        charmap.encoding    = FT_ENCODING_NONE;
        /* initial platform/encoding should indicate unset status? */
        charmap.platform_id = TT_PLATFORM_APPLE_UNICODE;
        charmap.encoding_id = TT_APPLE_ID_DEFAULT;

        if ( unicode_charmap )
        {
          charmap.encoding    = FT_ENCODING_UNICODE;
          charmap.platform_id = TT_PLATFORM_MICROSOFT;
          charmap.encoding_id = TT_MS_ID_UNICODE_CS;
        }

        error = FT_CMap_New( &pcf_cmap_class, NULL, &charmap, NULL );
      }
    }

  Exit:
    return error;

  Fail:
    FT_TRACE2(( "  not a PCF file\n" ));
    PCF_Face_Done( face );
    error = FT_THROW( Unknown_File_Format );  /* error */
    goto Exit;
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Size_Select( FT_Size   size,
                   FT_ULong  strike_index )
  {
    PCF_Accel  accel = &( (PCF_Face)size->face )->accel;


    FT_Select_Metrics( size->face, strike_index );

    size->metrics.ascender    =  accel->fontAscent * 64;
    size->metrics.descender   = -accel->fontDescent * 64;
    size->metrics.max_advance =  accel->maxbounds.characterWidth * 64;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Size_Request( FT_Size          size,
                    FT_Size_Request  req )
  {
    PCF_Face         face  = (PCF_Face)size->face;
    FT_Bitmap_Size*  bsize = size->face->available_sizes;
    FT_Error         error = FT_ERR( Invalid_Pixel_Size );
    FT_Long          height;


    height = FT_REQUEST_HEIGHT( req );
    height = ( height + 32 ) >> 6;

    switch ( req->type )
    {
    case FT_SIZE_REQUEST_TYPE_NOMINAL:
      if ( height == ( ( bsize->y_ppem + 32 ) >> 6 ) )
        error = FT_Err_Ok;
      break;

    case FT_SIZE_REQUEST_TYPE_REAL_DIM:
      if ( height == ( face->accel.fontAscent +
                       face->accel.fontDescent ) )
        error = FT_Err_Ok;
      break;

    default:
      error = FT_THROW( Unimplemented_Feature );
      break;
    }

    if ( error )
      return error;
    else
      return PCF_Size_Select( size, 0 );
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Glyph_Load( FT_GlyphSlot  slot,
                  FT_Size       size,
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    PCF_Face    face   = (PCF_Face)size->face;
    FT_Stream   stream;
    FT_Error    error  = FT_Err_Ok;
    FT_Bitmap*  bitmap = &slot->bitmap;
    PCF_Metric  metric;
    FT_ULong    bytes;


    FT_TRACE1(( "PCF_Glyph_Load: glyph index %u\n", glyph_index ));

    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    if ( glyph_index >= (FT_UInt)face->root.num_glyphs )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    stream = face->root.stream;

    metric = face->metrics + glyph_index;

    bitmap->rows       = (unsigned int)( metric->ascent +
                                         metric->descent );
    bitmap->width      = (unsigned int)( metric->rightSideBearing -
                                         metric->leftSideBearing );
    bitmap->num_grays  = 1;
    bitmap->pixel_mode = FT_PIXEL_MODE_MONO;

    switch ( PCF_GLYPH_PAD( face->bitmapsFormat ) )
    {
    case 1:
      bitmap->pitch = (int)( ( bitmap->width + 7 ) >> 3 );
      break;

    case 2:
      bitmap->pitch = (int)( ( ( bitmap->width + 15 ) >> 4 ) << 1 );
      break;

    case 4:
      bitmap->pitch = (int)( ( ( bitmap->width + 31 ) >> 5 ) << 2 );
      break;

    case 8:
      bitmap->pitch = (int)( ( ( bitmap->width + 63 ) >> 6 ) << 3 );
      break;

    default:
      return FT_THROW( Invalid_File_Format );
    }

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = metric->leftSideBearing;
    slot->bitmap_top  = metric->ascent;

    slot->metrics.horiAdvance  = (FT_Pos)( metric->characterWidth * 64 );
    slot->metrics.horiBearingX = (FT_Pos)( metric->leftSideBearing * 64 );
    slot->metrics.horiBearingY = (FT_Pos)( metric->ascent * 64 );
    slot->metrics.width        = (FT_Pos)( ( metric->rightSideBearing -
                                             metric->leftSideBearing ) * 64 );
    slot->metrics.height       = (FT_Pos)( bitmap->rows * 64 );

    ft_synthesize_vertical_metrics( &slot->metrics,
                                    ( face->accel.fontAscent +
                                      face->accel.fontDescent ) * 64 );

    if ( load_flags & FT_LOAD_BITMAP_METRICS_ONLY )
      goto Exit;

    /* XXX: to do: are there cases that need repadding the bitmap? */
    bytes = (FT_ULong)bitmap->pitch * bitmap->rows;

    error = ft_glyphslot_alloc_bitmap( slot, (FT_ULong)bytes );
    if ( error )
      goto Exit;

    if ( FT_STREAM_SEEK( metric->bits )          ||
         FT_STREAM_READ( bitmap->buffer, bytes ) )
      goto Exit;

    if ( PCF_BIT_ORDER( face->bitmapsFormat ) != MSBFirst )
      BitOrderInvert( bitmap->buffer, bytes );

    if ( ( PCF_BYTE_ORDER( face->bitmapsFormat ) !=
           PCF_BIT_ORDER( face->bitmapsFormat )  ) )
    {
      switch ( PCF_SCAN_UNIT( face->bitmapsFormat ) )
      {
      case 1:
        break;

      case 2:
        TwoByteSwap( bitmap->buffer, bytes );
        break;

      case 4:
        FourByteSwap( bitmap->buffer, bytes );
        break;
      }
    }

  Exit:
    return error;
  }


  /*
   *
   * BDF SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  pcf_get_bdf_property( FT_Face           face,       /* PCF_Face */
                        const char*       prop_name,
                        BDF_PropertyRec  *aproperty )
  {
    PCF_Face      pcfface = (PCF_Face)face;
    PCF_Property  prop;


    prop = pcf_find_property( pcfface, prop_name );
    if ( prop )
    {
      if ( prop->isString )
      {
        aproperty->type   = BDF_PROPERTY_TYPE_ATOM;
        aproperty->u.atom = prop->value.atom;
      }
      else
      {
        if ( prop->value.l > 0x7FFFFFFFL          ||
             prop->value.l < ( -1 - 0x7FFFFFFFL ) )
        {
          FT_TRACE2(( "pcf_get_bdf_property:"
                      " too large integer 0x%lx is truncated\n",
                      prop->value.l ));
        }

        /*
         * The PCF driver loads all properties as signed integers.
         * This really doesn't seem to be a problem, because this is
         * sufficient for any meaningful values.
         */
        aproperty->type      = BDF_PROPERTY_TYPE_INTEGER;
        aproperty->u.integer = (FT_Int32)prop->value.l;
      }

      return FT_Err_Ok;
    }

    return FT_THROW( Invalid_Argument );
  }


  FT_CALLBACK_DEF( FT_Error )
  pcf_get_charset_id( FT_Face       face,               /* PCF_Face */
                      const char*  *acharset_encoding,
                      const char*  *acharset_registry )
  {
    PCF_Face  pcfface = (PCF_Face)face;


    *acharset_encoding = pcfface->charset_encoding;
    *acharset_registry = pcfface->charset_registry;

    return FT_Err_Ok;
  }


  static const FT_Service_BDFRec  pcf_service_bdf =
  {
    (FT_BDF_GetCharsetIdFunc)pcf_get_charset_id,     /* get_charset_id */
    (FT_BDF_GetPropertyFunc) pcf_get_bdf_property    /* get_property   */
  };


  /*
   * PROPERTY SERVICE
   *
   */
  FT_CALLBACK_DEF( FT_Error )
  pcf_property_set( FT_Module    module,         /* PCF_Driver */
                    const char*  property_name,
                    const void*  value,
                    FT_Bool      value_is_string )
  {
#ifdef PCF_CONFIG_OPTION_LONG_FAMILY_NAMES

    FT_Error    error  = FT_Err_Ok;
    PCF_Driver  driver = (PCF_Driver)module;

#ifndef FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES
    FT_UNUSED( value_is_string );
#endif


    if ( !ft_strcmp( property_name, "no-long-family-names" ) )
    {
#ifdef FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES
      if ( value_is_string )
      {
        const char*  s   = (const char*)value;
        long         lfn = ft_strtol( s, NULL, 10 );


        if ( lfn == 0 )
          driver->no_long_family_names = 0;
        else if ( lfn == 1 )
          driver->no_long_family_names = 1;
        else
          return FT_THROW( Invalid_Argument );
      }
      else
#endif
      {
        FT_Bool*  no_long_family_names = (FT_Bool*)value;


        driver->no_long_family_names = *no_long_family_names;
      }

      return error;
    }

#else /* !PCF_CONFIG_OPTION_LONG_FAMILY_NAMES */

    FT_UNUSED( module );
    FT_UNUSED( value );
    FT_UNUSED( value_is_string );
#ifndef FT_DEBUG_LEVEL_TRACE
    FT_UNUSED( property_name );
#endif

#endif /* !PCF_CONFIG_OPTION_LONG_FAMILY_NAMES */

    FT_TRACE2(( "pcf_property_set: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_CALLBACK_DEF( FT_Error )
  pcf_property_get( FT_Module    module,         /* PCF_Driver */
                    const char*  property_name,
                    void*        value )
  {
#ifdef PCF_CONFIG_OPTION_LONG_FAMILY_NAMES

    FT_Error    error  = FT_Err_Ok;
    PCF_Driver  driver = (PCF_Driver)module;


    if ( !ft_strcmp( property_name, "no-long-family-names" ) )
    {
      FT_Bool   no_long_family_names = driver->no_long_family_names;
      FT_Bool*  val                  = (FT_Bool*)value;


      *val = no_long_family_names;

      return error;
    }

#else /* !PCF_CONFIG_OPTION_LONG_FAMILY_NAMES */

    FT_UNUSED( module );
    FT_UNUSED( value );
#ifndef FT_DEBUG_LEVEL_TRACE
    FT_UNUSED( property_name );
#endif

#endif /* !PCF_CONFIG_OPTION_LONG_FAMILY_NAMES */

    FT_TRACE2(( "pcf_property_get: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    pcf_service_properties,

    (FT_Properties_SetFunc)pcf_property_set,      /* set_property */
    (FT_Properties_GetFunc)pcf_property_get )     /* get_property */


  /*
   *
   * SERVICE LIST
   *
   */

  static const FT_ServiceDescRec  pcf_services[] =
  {
    { FT_SERVICE_ID_BDF,         &pcf_service_bdf },
    { FT_SERVICE_ID_FONT_FORMAT, FT_FONT_FORMAT_PCF },
    { FT_SERVICE_ID_PROPERTIES,  &pcf_service_properties },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  pcf_driver_requester( FT_Module    module,
                        const char*  name )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( pcf_services, name );
  }


  FT_CALLBACK_DEF( FT_Error )
  pcf_driver_init( FT_Module  module )      /* PCF_Driver */
  {
#ifdef PCF_CONFIG_OPTION_LONG_FAMILY_NAMES
    PCF_Driver  driver = (PCF_Driver)module;


    driver->no_long_family_names = 0;
#else
    FT_UNUSED( module );
#endif

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  pcf_driver_done( FT_Module  module )      /* PCF_Driver */
  {
    FT_UNUSED( module );
  }


  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  pcf_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER        |
      FT_MODULE_DRIVER_NO_OUTLINES,

      sizeof ( PCF_DriverRec ),
      "pcf",
      0x10000L,
      0x20000L,

      NULL,   /* module-specific interface */

      pcf_driver_init,          /* FT_Module_Constructor  module_init   */
      pcf_driver_done,          /* FT_Module_Destructor   module_done   */
      pcf_driver_requester      /* FT_Module_Requester    get_interface */
    },

    sizeof ( PCF_FaceRec ),
    sizeof ( FT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    PCF_Face_Init,              /* FT_Face_InitFunc  init_face */
    PCF_Face_Done,              /* FT_Face_DoneFunc  done_face */
    NULL,                       /* FT_Size_InitFunc  init_size */
    NULL,                       /* FT_Size_DoneFunc  done_size */
    NULL,                       /* FT_Slot_InitFunc  init_slot */
    NULL,                       /* FT_Slot_DoneFunc  done_slot */

    PCF_Glyph_Load,             /* FT_Slot_LoadFunc  load_glyph */

    NULL,                       /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    NULL,                       /* FT_Face_GetAdvancesFunc  get_advances */

    PCF_Size_Request,           /* FT_Size_RequestFunc  request_size */
    PCF_Size_Select             /* FT_Size_SelectFunc   select_size  */
  };


/* END */
