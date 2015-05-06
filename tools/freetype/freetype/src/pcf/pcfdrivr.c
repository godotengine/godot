/*  pcfdrivr.c

    FreeType font driver for pcf files

    Copyright (C) 2000-2004, 2006-2011, 2013 by
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


#include <ft2build.h>

#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_OBJECTS_H
#include FT_GZIP_H
#include FT_LZW_H
#include FT_BZIP2_H
#include FT_ERRORS_H
#include FT_BDF_H
#include FT_TRUETYPE_IDS_H

#include "pcf.h"
#include "pcfdrivr.h"
#include "pcfread.h"

#include "pcferror.h"
#include "pcfutil.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  trace_pcfread

#include FT_SERVICE_BDF_H
#include FT_SERVICE_XFREE86_NAME_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_pcfdriver


  typedef struct  PCF_CMapRec_
  {
    FT_CMapRec    root;
    FT_UInt       num_encodings;
    PCF_Encoding  encodings;

  } PCF_CMapRec, *PCF_CMap;


  FT_CALLBACK_DEF( FT_Error )
  pcf_cmap_init( FT_CMap     pcfcmap,   /* PCF_CMap */
                 FT_Pointer  init_data )
  {
    PCF_CMap  cmap = (PCF_CMap)pcfcmap;
    PCF_Face  face = (PCF_Face)FT_CMAP_FACE( pcfcmap );

    FT_UNUSED( init_data );


    cmap->num_encodings = (FT_UInt)face->nencodings;
    cmap->encodings     = face->encodings;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  pcf_cmap_done( FT_CMap  pcfcmap )         /* PCF_CMap */
  {
    PCF_CMap  cmap = (PCF_CMap)pcfcmap;


    cmap->encodings     = NULL;
    cmap->num_encodings = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  pcf_cmap_char_index( FT_CMap    pcfcmap,  /* PCF_CMap */
                       FT_UInt32  charcode )
  {
    PCF_CMap      cmap      = (PCF_CMap)pcfcmap;
    PCF_Encoding  encodings = cmap->encodings;
    FT_UInt       min, max, mid;
    FT_UInt       result    = 0;


    min = 0;
    max = cmap->num_encodings;

    while ( min < max )
    {
      FT_ULong  code;


      mid  = ( min + max ) >> 1;
      code = encodings[mid].enc;

      if ( charcode == code )
      {
        result = encodings[mid].glyph + 1;
        break;
      }

      if ( charcode < code )
        max = mid;
      else
        min = mid + 1;
    }

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt )
  pcf_cmap_char_next( FT_CMap    pcfcmap,   /* PCF_CMap */
                      FT_UInt32  *acharcode )
  {
    PCF_CMap      cmap      = (PCF_CMap)pcfcmap;
    PCF_Encoding  encodings = cmap->encodings;
    FT_UInt       min, max, mid;
    FT_ULong      charcode  = *acharcode + 1;
    FT_UInt       result    = 0;


    min = 0;
    max = cmap->num_encodings;

    while ( min < max )
    {
      FT_ULong  code;


      mid  = ( min + max ) >> 1;
      code = encodings[mid].enc;

      if ( charcode == code )
      {
        result = encodings[mid].glyph + 1;
        goto Exit;
      }

      if ( charcode < code )
        max = mid;
      else
        min = mid + 1;
    }

    charcode = 0;
    if ( min < cmap->num_encodings )
    {
      charcode = encodings[min].enc;
      result   = encodings[min].glyph + 1;
    }

  Exit:
    if ( charcode > 0xFFFFFFFFUL )
    {
      FT_TRACE1(( "pcf_cmap_char_next: charcode 0x%x > 32bit API" ));
      *acharcode = 0;
      /* XXX: result should be changed to indicate an overflow error */
    }
    else
      *acharcode = (FT_UInt32)charcode;
    return result;
  }


  FT_CALLBACK_TABLE_DEF
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
  PCF_Face_Done( FT_Face  pcfface )         /* PCF_Face */
  {
    PCF_Face   face = (PCF_Face)pcfface;
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    FT_FREE( face->encodings );
    FT_FREE( face->metrics );

    /* free properties */
    {
      PCF_Property  prop;
      FT_Int        i;


      if ( face->properties )
      {
        for ( i = 0; i < face->nprops; i++ )
        {
          prop = &face->properties[i];

          if ( prop )
          {
            FT_FREE( prop->name );
            if ( prop->isString )
              FT_FREE( prop->value.atom );
          }
        }
      }
      FT_FREE( face->properties );
    }

    FT_FREE( face->toc.tables );
    FT_FREE( pcfface->family_name );
    FT_FREE( pcfface->style_name );
    FT_FREE( pcfface->available_sizes );
    FT_FREE( face->charset_encoding );
    FT_FREE( face->charset_registry );

    /* close compressed stream if any */
    if ( pcfface->stream == &face->comp_stream )
    {
      FT_Stream_Close( &face->comp_stream );
      pcfface->stream = face->comp_source;
    }
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Face_Init( FT_Stream      stream,
                 FT_Face        pcfface,        /* PCF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    PCF_Face  face  = (PCF_Face)pcfface;
    FT_Error  error = FT_Err_Ok;

    FT_UNUSED( num_params );
    FT_UNUSED( params );
    FT_UNUSED( face_index );


    FT_TRACE2(( "PCF driver\n" ));

    error = pcf_load_font( stream, face );
    if ( error )
    {
      PCF_Face_Done( pcfface );

#if defined( FT_CONFIG_OPTION_USE_ZLIB )  || \
    defined( FT_CONFIG_OPTION_USE_LZW )   || \
    defined( FT_CONFIG_OPTION_USE_BZIP2 )

#ifdef FT_CONFIG_OPTION_USE_ZLIB
      {
        FT_Error  error2;


        /* this didn't work, try gzip support! */
        error2 = FT_Stream_OpenGzip( &face->comp_stream, stream );
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
        error3 = FT_Stream_OpenLZW( &face->comp_stream, stream );
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
        error4 = FT_Stream_OpenBzip2( &face->comp_stream, stream );
        if ( FT_ERR_EQ( error4, Unimplemented_Feature ) )
          goto Fail;

        error = error4;
      }
#endif /* FT_CONFIG_OPTION_USE_BZIP2 */

      if ( error )
        goto Fail;

      face->comp_source = stream;
      pcfface->stream   = &face->comp_stream;

      stream = pcfface->stream;

      error = pcf_load_font( stream, face );
      if ( error )
        goto Fail;

#else /* !(FT_CONFIG_OPTION_USE_ZLIB ||
           FT_CONFIG_OPTION_USE_LZW ||
           FT_CONFIG_OPTION_USE_BZIP2) */

      goto Fail;

#endif
    }

    /* set up charmap */
    {
      FT_String  *charset_registry = face->charset_registry;
      FT_String  *charset_encoding = face->charset_encoding;
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
          if ( !ft_strcmp( s, "10646" )                      ||
               ( !ft_strcmp( s, "8859" ) &&
                 !ft_strcmp( face->charset_encoding, "1" ) ) )
          unicode_charmap = 1;
        }
      }

      {
        FT_CharMapRec  charmap;


        charmap.face        = FT_FACE( face );
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

#if 0
        /* Select default charmap */
        if ( pcfface->num_charmaps )
          pcfface->charmap = pcfface->charmaps[0];
#endif
      }
    }

  Exit:
    return error;

  Fail:
    FT_TRACE2(( "  not a PCF file\n" ));
    PCF_Face_Done( pcfface );
    error = FT_THROW( Unknown_File_Format );  /* error */
    goto Exit;
  }


  FT_CALLBACK_DEF( FT_Error )
  PCF_Size_Select( FT_Size   size,
                   FT_ULong  strike_index )
  {
    PCF_Accel  accel = &( (PCF_Face)size->face )->accel;


    FT_Select_Metrics( size->face, strike_index );

    size->metrics.ascender    =  accel->fontAscent << 6;
    size->metrics.descender   = -accel->fontDescent << 6;
    size->metrics.max_advance =  accel->maxbounds.characterWidth << 6;

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
    PCF_Face    face   = (PCF_Face)FT_SIZE_FACE( size );
    FT_Stream   stream;
    FT_Error    error  = FT_Err_Ok;
    FT_Bitmap*  bitmap = &slot->bitmap;
    PCF_Metric  metric;
    FT_Offset   bytes;

    FT_UNUSED( load_flags );


    FT_TRACE4(( "load_glyph %d ---", glyph_index ));

    if ( !face || glyph_index >= (FT_UInt)face->root.num_glyphs )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    stream = face->root.stream;

    if ( glyph_index > 0 )
      glyph_index--;

    metric = face->metrics + glyph_index;

    bitmap->rows       = metric->ascent + metric->descent;
    bitmap->width      = metric->rightSideBearing - metric->leftSideBearing;
    bitmap->num_grays  = 1;
    bitmap->pixel_mode = FT_PIXEL_MODE_MONO;

    FT_TRACE6(( "BIT_ORDER %d ; BYTE_ORDER %d ; GLYPH_PAD %d\n",
                  PCF_BIT_ORDER( face->bitmapsFormat ),
                  PCF_BYTE_ORDER( face->bitmapsFormat ),
                  PCF_GLYPH_PAD( face->bitmapsFormat ) ));

    switch ( PCF_GLYPH_PAD( face->bitmapsFormat ) )
    {
    case 1:
      bitmap->pitch = ( bitmap->width + 7 ) >> 3;
      break;

    case 2:
      bitmap->pitch = ( ( bitmap->width + 15 ) >> 4 ) << 1;
      break;

    case 4:
      bitmap->pitch = ( ( bitmap->width + 31 ) >> 5 ) << 2;
      break;

    case 8:
      bitmap->pitch = ( ( bitmap->width + 63 ) >> 6 ) << 3;
      break;

    default:
      return FT_THROW( Invalid_File_Format );
    }

    /* XXX: to do: are there cases that need repadding the bitmap? */
    bytes = bitmap->pitch * bitmap->rows;

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

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = metric->leftSideBearing;
    slot->bitmap_top  = metric->ascent;

    slot->metrics.horiAdvance  = metric->characterWidth << 6;
    slot->metrics.horiBearingX = metric->leftSideBearing << 6;
    slot->metrics.horiBearingY = metric->ascent << 6;
    slot->metrics.width        = ( metric->rightSideBearing -
                                   metric->leftSideBearing ) << 6;
    slot->metrics.height       = bitmap->rows << 6;

    ft_synthesize_vertical_metrics( &slot->metrics,
                                    ( face->accel.fontAscent +
                                      face->accel.fontDescent ) << 6 );

    FT_TRACE4(( " --- ok\n" ));

  Exit:
    return error;
  }


 /*
  *
  *  BDF SERVICE
  *
  */

  static FT_Error
  pcf_get_bdf_property( PCF_Face          face,
                        const char*       prop_name,
                        BDF_PropertyRec  *aproperty )
  {
    PCF_Property  prop;


    prop = pcf_find_property( face, prop_name );
    if ( prop != NULL )
    {
      if ( prop->isString )
      {
        aproperty->type   = BDF_PROPERTY_TYPE_ATOM;
        aproperty->u.atom = prop->value.atom;
      }
      else
      {
        if ( prop->value.l > 0x7FFFFFFFL || prop->value.l < ( -1 - 0x7FFFFFFFL ) )
        {
          FT_TRACE1(( "pcf_get_bdf_property: " ));
          FT_TRACE1(( "too large integer 0x%x is truncated\n" ));
        }
        /* Apparently, the PCF driver loads all properties as signed integers!
         * This really doesn't seem to be a problem, because this is
         * sufficient for any meaningful values.
         */
        aproperty->type      = BDF_PROPERTY_TYPE_INTEGER;
        aproperty->u.integer = (FT_Int32)prop->value.l;
      }
      return 0;
    }

    return FT_THROW( Invalid_Argument );
  }


  static FT_Error
  pcf_get_charset_id( PCF_Face      face,
                      const char*  *acharset_encoding,
                      const char*  *acharset_registry )
  {
    *acharset_encoding = face->charset_encoding;
    *acharset_registry = face->charset_registry;

    return 0;
  }


  static const FT_Service_BDFRec  pcf_service_bdf =
  {
    (FT_BDF_GetCharsetIdFunc)pcf_get_charset_id,
    (FT_BDF_GetPropertyFunc) pcf_get_bdf_property
  };


 /*
  *
  *  SERVICE LIST
  *
  */

  static const FT_ServiceDescRec  pcf_services[] =
  {
    { FT_SERVICE_ID_BDF,       &pcf_service_bdf },
    { FT_SERVICE_ID_XF86_NAME, FT_XF86_FORMAT_PCF },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  pcf_driver_requester( FT_Module    module,
                        const char*  name )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( pcf_services, name );
  }


  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  pcf_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER        |
      FT_MODULE_DRIVER_NO_OUTLINES,
      sizeof ( FT_DriverRec ),

      "pcf",
      0x10000L,
      0x20000L,

      0,

      0,                    /* FT_Module_Constructor */
      0,                    /* FT_Module_Destructor  */
      pcf_driver_requester
    },

    sizeof ( PCF_FaceRec ),
    sizeof ( FT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    PCF_Face_Init,
    PCF_Face_Done,
    0,                      /* FT_Size_InitFunc */
    0,                      /* FT_Size_DoneFunc */
    0,                      /* FT_Slot_InitFunc */
    0,                      /* FT_Slot_DoneFunc */

    PCF_Glyph_Load,

    0,                      /* FT_Face_GetKerningFunc  */
    0,                      /* FT_Face_AttachFunc      */
    0,                      /* FT_Face_GetAdvancesFunc */

    PCF_Size_Request,
    PCF_Size_Select
  };


/* END */
