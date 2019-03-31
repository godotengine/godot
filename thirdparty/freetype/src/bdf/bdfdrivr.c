/*  bdfdrivr.c

    FreeType font driver for bdf files

    Copyright (C) 2001-2008, 2011, 2013, 2014 by
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
#include FT_BDF_H
#include FT_TRUETYPE_IDS_H

#include FT_SERVICE_BDF_H
#include FT_SERVICE_FONT_FORMAT_H

#include "bdf.h"
#include "bdfdrivr.h"

#include "bdferror.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  bdfdriver


  typedef struct  BDF_CMapRec_
  {
    FT_CMapRec        cmap;
    FT_ULong          num_encodings; /* ftobjs.h: FT_CMap->clazz->size */
    BDF_encoding_el*  encodings;

  } BDF_CMapRec, *BDF_CMap;


  FT_CALLBACK_DEF( FT_Error )
  bdf_cmap_init( FT_CMap     bdfcmap,
                 FT_Pointer  init_data )
  {
    BDF_CMap  cmap = (BDF_CMap)bdfcmap;
    BDF_Face  face = (BDF_Face)FT_CMAP_FACE( cmap );
    FT_UNUSED( init_data );


    cmap->num_encodings = face->bdffont->glyphs_used;
    cmap->encodings     = face->en_table;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  bdf_cmap_done( FT_CMap  bdfcmap )
  {
    BDF_CMap  cmap = (BDF_CMap)bdfcmap;


    cmap->encodings     = NULL;
    cmap->num_encodings = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  bdf_cmap_char_index( FT_CMap    bdfcmap,
                       FT_UInt32  charcode )
  {
    BDF_CMap          cmap      = (BDF_CMap)bdfcmap;
    BDF_encoding_el*  encodings = cmap->encodings;
    FT_ULong          min, max, mid; /* num_encodings */
    FT_UShort         result    = 0; /* encodings->glyph */


    min = 0;
    max = cmap->num_encodings;
    mid = ( min + max ) >> 1;

    while ( min < max )
    {
      FT_ULong  code;


      if ( mid > max || mid < min )
        mid = ( min + max ) >> 1;

      code = encodings[mid].enc;

      if ( charcode == code )
      {
        /* increase glyph index by 1 --              */
        /* we reserve slot 0 for the undefined glyph */
        result = encodings[mid].glyph + 1;
        break;
      }

      if ( charcode < code )
        max = mid;
      else
        min = mid + 1;

      /* prediction in a continuous block */
      mid += charcode - code;
    }

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt )
  bdf_cmap_char_next( FT_CMap     bdfcmap,
                      FT_UInt32  *acharcode )
  {
    BDF_CMap          cmap      = (BDF_CMap)bdfcmap;
    BDF_encoding_el*  encodings = cmap->encodings;
    FT_ULong          min, max, mid; /* num_encodings */
    FT_UShort         result   = 0;  /* encodings->glyph */
    FT_ULong          charcode = *acharcode + 1;


    min = 0;
    max = cmap->num_encodings;
    mid = ( min + max ) >> 1;

    while ( min < max )
    {
      FT_ULong  code; /* same as BDF_encoding_el.enc */


      if ( mid > max || mid < min )
        mid = ( min + max ) >> 1;

      code = encodings[mid].enc;

      if ( charcode == code )
      {
        /* increase glyph index by 1 --              */
        /* we reserve slot 0 for the undefined glyph */
        result = encodings[mid].glyph + 1;
        goto Exit;
      }

      if ( charcode < code )
        max = mid;
      else
        min = mid + 1;

      /* prediction in a continuous block */
      mid += charcode - code;
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
      FT_TRACE1(( "bdf_cmap_char_next: charcode 0x%x > 32bit API" ));
      *acharcode = 0;
      /* XXX: result should be changed to indicate an overflow error */
    }
    else
      *acharcode = (FT_UInt32)charcode;
    return result;
  }


  static
  const FT_CMap_ClassRec  bdf_cmap_class =
  {
    sizeof ( BDF_CMapRec ),
    bdf_cmap_init,
    bdf_cmap_done,
    bdf_cmap_char_index,
    bdf_cmap_char_next,

    NULL, NULL, NULL, NULL, NULL
  };


  static FT_Error
  bdf_interpret_style( BDF_Face  bdf )
  {
    FT_Error         error  = FT_Err_Ok;
    FT_Face          face   = FT_FACE( bdf );
    FT_Memory        memory = face->memory;
    bdf_font_t*      font   = bdf->bdffont;
    bdf_property_t*  prop;

    char*   strings[4] = { NULL, NULL, NULL, NULL };
    size_t  nn, len, lengths[4];


    face->style_flags = 0;

    prop = bdf_get_font_property( font, (char *)"SLANT" );
    if ( prop && prop->format == BDF_ATOM                             &&
         prop->value.atom                                             &&
         ( *(prop->value.atom) == 'O' || *(prop->value.atom) == 'o' ||
           *(prop->value.atom) == 'I' || *(prop->value.atom) == 'i' ) )
    {
      face->style_flags |= FT_STYLE_FLAG_ITALIC;
      strings[2] = ( *(prop->value.atom) == 'O' || *(prop->value.atom) == 'o' )
                   ? (char *)"Oblique"
                   : (char *)"Italic";
    }

    prop = bdf_get_font_property( font, (char *)"WEIGHT_NAME" );
    if ( prop && prop->format == BDF_ATOM                             &&
         prop->value.atom                                             &&
         ( *(prop->value.atom) == 'B' || *(prop->value.atom) == 'b' ) )
    {
      face->style_flags |= FT_STYLE_FLAG_BOLD;
      strings[1] = (char *)"Bold";
    }

    prop = bdf_get_font_property( font, (char *)"SETWIDTH_NAME" );
    if ( prop && prop->format == BDF_ATOM                              &&
         prop->value.atom && *(prop->value.atom)                       &&
         !( *(prop->value.atom) == 'N' || *(prop->value.atom) == 'n' ) )
      strings[3] = (char *)(prop->value.atom);

    prop = bdf_get_font_property( font, (char *)"ADD_STYLE_NAME" );
    if ( prop && prop->format == BDF_ATOM                              &&
         prop->value.atom && *(prop->value.atom)                       &&
         !( *(prop->value.atom) == 'N' || *(prop->value.atom) == 'n' ) )
      strings[0] = (char *)(prop->value.atom);

    for ( len = 0, nn = 0; nn < 4; nn++ )
    {
      lengths[nn] = 0;
      if ( strings[nn] )
      {
        lengths[nn] = ft_strlen( strings[nn] );
        len        += lengths[nn] + 1;
      }
    }

    if ( len == 0 )
    {
      strings[0] = (char *)"Regular";
      lengths[0] = ft_strlen( strings[0] );
      len        = lengths[0] + 1;
    }

    {
      char*  s;


      if ( FT_ALLOC( face->style_name, len ) )
        return error;

      s = face->style_name;

      for ( nn = 0; nn < 4; nn++ )
      {
        char*  src = strings[nn];


        len = lengths[nn];

        if ( !src )
          continue;

        /* separate elements with a space */
        if ( s != face->style_name )
          *s++ = ' ';

        ft_memcpy( s, src, len );

        /* need to convert spaces to dashes for */
        /* add_style_name and setwidth_name     */
        if ( nn == 0 || nn == 3 )
        {
          size_t  mm;


          for ( mm = 0; mm < len; mm++ )
            if ( s[mm] == ' ' )
              s[mm] = '-';
        }

        s += len;
      }
      *s = 0;
    }

    return error;
  }


  FT_CALLBACK_DEF( void )
  BDF_Face_Done( FT_Face  bdfface )         /* BDF_Face */
  {
    BDF_Face   face = (BDF_Face)bdfface;
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    bdf_free_font( face->bdffont );

    FT_FREE( face->en_table );

    FT_FREE( face->charset_encoding );
    FT_FREE( face->charset_registry );
    FT_FREE( bdfface->family_name );
    FT_FREE( bdfface->style_name );

    FT_FREE( bdfface->available_sizes );

    FT_FREE( face->bdffont );
  }


  FT_CALLBACK_DEF( FT_Error )
  BDF_Face_Init( FT_Stream      stream,
                 FT_Face        bdfface,        /* BDF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    FT_Error       error  = FT_Err_Ok;
    BDF_Face       face   = (BDF_Face)bdfface;
    FT_Memory      memory = FT_FACE_MEMORY( face );

    bdf_font_t*    font = NULL;
    bdf_options_t  options;

    FT_UNUSED( num_params );
    FT_UNUSED( params );


    FT_TRACE2(( "BDF driver\n" ));

    if ( FT_STREAM_SEEK( 0 ) )
      goto Exit;

    options.correct_metrics = 1;   /* FZ XXX: options semantics */
    options.keep_unencoded  = 1;
    options.keep_comments   = 0;
    options.font_spacing    = BDF_PROPORTIONAL;

    error = bdf_load_font( stream, memory, &options, &font );
    if ( FT_ERR_EQ( error, Missing_Startfont_Field ) )
    {
      FT_TRACE2(( "  not a BDF file\n" ));
      goto Fail;
    }
    else if ( error )
      goto Exit;

    /* we have a bdf font: let's construct the face object */
    face->bdffont = font;

    /* BDF cannot have multiple faces in a single font file.
     * XXX: non-zero face_index is already invalid argument, but
     *      Type1, Type42 driver has a convention to return
     *      an invalid argument error when the font could be
     *      opened by the specified driver.
     */
    if ( face_index > 0 && ( face_index & 0xFFFF ) > 0 )
    {
      FT_ERROR(( "BDF_Face_Init: invalid face index\n" ));
      BDF_Face_Done( bdfface );
      return FT_THROW( Invalid_Argument );
    }

    {
      bdf_property_t*  prop = NULL;


      FT_TRACE4(( "  number of glyphs: allocated %d (used %d)\n",
                  font->glyphs_size,
                  font->glyphs_used ));
      FT_TRACE4(( "  number of unencoded glyphs: allocated %d (used %d)\n",
                  font->unencoded_size,
                  font->unencoded_used ));

      bdfface->num_faces  = 1;
      bdfface->face_index = 0;

      bdfface->face_flags |= FT_FACE_FLAG_FIXED_SIZES |
                             FT_FACE_FLAG_HORIZONTAL;

      prop = bdf_get_font_property( font, "SPACING" );
      if ( prop && prop->format == BDF_ATOM                             &&
           prop->value.atom                                             &&
           ( *(prop->value.atom) == 'M' || *(prop->value.atom) == 'm' ||
             *(prop->value.atom) == 'C' || *(prop->value.atom) == 'c' ) )
        bdfface->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      /* FZ XXX: TO DO: FT_FACE_FLAGS_VERTICAL   */
      /* FZ XXX: I need a font to implement this */

      prop = bdf_get_font_property( font, "FAMILY_NAME" );
      if ( prop && prop->value.atom )
      {
        if ( FT_STRDUP( bdfface->family_name, prop->value.atom ) )
          goto Exit;
      }
      else
        bdfface->family_name = NULL;

      if ( FT_SET_ERROR( bdf_interpret_style( face ) ) )
        goto Exit;

      /* the number of glyphs (with one slot for the undefined glyph */
      /* at position 0 and all unencoded glyphs)                     */
      bdfface->num_glyphs = (FT_Long)( font->glyphs_size + 1 );

      bdfface->num_fixed_sizes = 1;
      if ( FT_NEW_ARRAY( bdfface->available_sizes, 1 ) )
        goto Exit;

      {
        FT_Bitmap_Size*  bsize = bdfface->available_sizes;
        FT_Short         resolution_x = 0, resolution_y = 0;
        long             value;


        FT_ZERO( bsize );

        /* sanity checks */
        if ( font->font_ascent > 0x7FFF || font->font_ascent < -0x7FFF )
        {
          font->font_ascent = font->font_ascent < 0 ? -0x7FFF : 0x7FFF;
          FT_TRACE0(( "BDF_Face_Init: clamping font ascent to value %d\n",
                      font->font_ascent ));
        }
        if ( font->font_descent > 0x7FFF || font->font_descent < -0x7FFF )
        {
          font->font_descent = font->font_descent < 0 ? -0x7FFF : 0x7FFF;
          FT_TRACE0(( "BDF_Face_Init: clamping font descent to value %d\n",
                      font->font_descent ));
        }

        bsize->height = (FT_Short)( font->font_ascent + font->font_descent );

        prop = bdf_get_font_property( font, "AVERAGE_WIDTH" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "BDF_Face_Init: negative average width\n" ));
#endif
          if ( prop->value.l >    0x7FFFL * 10 - 5   ||
               prop->value.l < -( 0x7FFFL * 10 - 5 ) )
          {
            bsize->width = 0x7FFF;
            FT_TRACE0(( "BDF_Face_Init: clamping average width to value %d\n",
                        bsize->width ));
          }
          else
            bsize->width = FT_ABS( (FT_Short)( ( prop->value.l + 5 ) / 10 ) );
        }
        else
        {
          /* this is a heuristical value */
          bsize->width = (FT_Short)FT_MulDiv( bsize->height, 2, 3 );
        }

        prop = bdf_get_font_property( font, "POINT_SIZE" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "BDF_Face_Init: negative point size\n" ));
#endif
          /* convert from 722.7 decipoints to 72 points per inch */
          if ( prop->value.l >  0x504C2L || /* 0x7FFF * 72270/7200 */
               prop->value.l < -0x504C2L )
          {
            bsize->size = 0x7FFF;
            FT_TRACE0(( "BDF_Face_Init: clamping point size to value %d\n",
                        bsize->size ));
          }
          else
            bsize->size = FT_MulDiv( FT_ABS( prop->value.l ),
                                     64 * 7200,
                                     72270L );
        }
        else if ( font->point_size )
        {
          if ( font->point_size > 0x7FFF )
          {
            bsize->size = 0x7FFF;
            FT_TRACE0(( "BDF_Face_Init: clamping point size to value %d\n",
                        bsize->size ));
          }
          else
            bsize->size = (FT_Pos)font->point_size << 6;
        }
        else
        {
          /* this is a heuristical value */
          bsize->size = bsize->width * 64;
        }

        prop = bdf_get_font_property( font, "PIXEL_SIZE" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "BDF_Face_Init: negative pixel size\n" ));
#endif
          if ( prop->value.l > 0x7FFF || prop->value.l < -0x7FFF )
          {
            bsize->y_ppem = 0x7FFF << 6;
            FT_TRACE0(( "BDF_Face_Init: clamping pixel size to value %d\n",
                        bsize->y_ppem ));
          }
          else
            bsize->y_ppem = FT_ABS( (FT_Short)prop->value.l ) << 6;
        }

        prop = bdf_get_font_property( font, "RESOLUTION_X" );
        if ( prop )
          value = prop->value.l;
        else
          value = (long)font->resolution_x;
        if ( value )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( value < 0 )
            FT_TRACE0(( "BDF_Face_Init: negative X resolution\n" ));
#endif
          if ( value > 0x7FFF || value < -0x7FFF )
          {
            resolution_x = 0x7FFF;
            FT_TRACE0(( "BDF_Face_Init: clamping X resolution to value %d\n",
                        resolution_x ));
          }
          else
            resolution_x = FT_ABS( (FT_Short)value );
        }

        prop = bdf_get_font_property( font, "RESOLUTION_Y" );
        if ( prop )
          value = prop->value.l;
        else
          value = (long)font->resolution_y;
        if ( value )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( value < 0 )
            FT_TRACE0(( "BDF_Face_Init: negative Y resolution\n" ));
#endif
          if ( value > 0x7FFF || value < -0x7FFF )
          {
            resolution_y = 0x7FFF;
            FT_TRACE0(( "BDF_Face_Init: clamping Y resolution to value %d\n",
                        resolution_y ));
          }
          else
            resolution_y = FT_ABS( (FT_Short)value );
        }

        if ( bsize->y_ppem == 0 )
        {
          bsize->y_ppem = bsize->size;
          if ( resolution_y )
            bsize->y_ppem = FT_MulDiv( bsize->y_ppem, resolution_y, 72 );
        }
        if ( resolution_x && resolution_y )
          bsize->x_ppem = FT_MulDiv( bsize->y_ppem,
                                     resolution_x,
                                     resolution_y );
        else
          bsize->x_ppem = bsize->y_ppem;
      }

      /* encoding table */
      {
        bdf_glyph_t*   cur = font->glyphs;
        unsigned long  n;


        if ( FT_NEW_ARRAY( face->en_table, font->glyphs_size ) )
          goto Exit;

        face->default_glyph = 0;
        for ( n = 0; n < font->glyphs_size; n++ )
        {
          (face->en_table[n]).enc = cur[n].encoding;
          FT_TRACE4(( "  idx %d, val 0x%lX\n", n, cur[n].encoding ));
          (face->en_table[n]).glyph = (FT_UShort)n;

          if ( cur[n].encoding == font->default_char )
          {
            if ( n < FT_UINT_MAX )
              face->default_glyph = (FT_UInt)n;
            else
              FT_TRACE1(( "BDF_Face_Init:"
                          " idx %d is too large for this system\n", n ));
          }
        }
      }

      /* charmaps */
      {
        bdf_property_t  *charset_registry, *charset_encoding;
        FT_Bool          unicode_charmap  = 0;


        charset_registry =
          bdf_get_font_property( font, "CHARSET_REGISTRY" );
        charset_encoding =
          bdf_get_font_property( font, "CHARSET_ENCODING" );
        if ( charset_registry && charset_encoding )
        {
          if ( charset_registry->format == BDF_ATOM &&
               charset_encoding->format == BDF_ATOM &&
               charset_registry->value.atom         &&
               charset_encoding->value.atom         )
          {
            const char*  s;


            if ( FT_STRDUP( face->charset_encoding,
                            charset_encoding->value.atom ) ||
                 FT_STRDUP( face->charset_registry,
                            charset_registry->value.atom ) )
              goto Exit;

            /* Uh, oh, compare first letters manually to avoid dependency */
            /* on locales.                                                */
            s = face->charset_registry;
            if ( ( s[0] == 'i' || s[0] == 'I' ) &&
                 ( s[1] == 's' || s[1] == 'S' ) &&
                 ( s[2] == 'o' || s[2] == 'O' ) )
            {
              s += 3;
              if ( !ft_strcmp( s, "10646" )                      ||
                   ( !ft_strcmp( s, "8859" ) &&
                     !ft_strcmp( face->charset_encoding, "1" ) ) )
                unicode_charmap = 1;
              /* another name for ASCII */
              else if ( !ft_strcmp( s, "646.1991" )                 &&
                        !ft_strcmp( face->charset_encoding, "IRV" ) )
                unicode_charmap = 1;
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

              error = FT_CMap_New( &bdf_cmap_class, NULL, &charmap, NULL );
            }

            goto Exit;
          }
        }

        /* otherwise assume Adobe standard encoding */

        {
          FT_CharMapRec  charmap;


          charmap.face        = FT_FACE( face );
          charmap.encoding    = FT_ENCODING_ADOBE_STANDARD;
          charmap.platform_id = TT_PLATFORM_ADOBE;
          charmap.encoding_id = TT_ADOBE_ID_STANDARD;

          error = FT_CMap_New( &bdf_cmap_class, NULL, &charmap, NULL );

          /* Select default charmap */
          if ( bdfface->num_charmaps )
            bdfface->charmap = bdfface->charmaps[0];
        }
      }
    }

  Exit:
    return error;

  Fail:
    BDF_Face_Done( bdfface );
    return FT_THROW( Unknown_File_Format );
  }


  FT_CALLBACK_DEF( FT_Error )
  BDF_Size_Select( FT_Size   size,
                   FT_ULong  strike_index )
  {
    bdf_font_t*  bdffont = ( (BDF_Face)size->face )->bdffont;


    FT_Select_Metrics( size->face, strike_index );

    size->metrics.ascender    = bdffont->font_ascent * 64;
    size->metrics.descender   = -bdffont->font_descent * 64;
    size->metrics.max_advance = bdffont->bbx.width * 64;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  BDF_Size_Request( FT_Size          size,
                    FT_Size_Request  req )
  {
    FT_Face          face    = size->face;
    FT_Bitmap_Size*  bsize   = face->available_sizes;
    bdf_font_t*      bdffont = ( (BDF_Face)face )->bdffont;
    FT_Error         error   = FT_ERR( Invalid_Pixel_Size );
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
      if ( height == ( bdffont->font_ascent +
                       bdffont->font_descent ) )
        error = FT_Err_Ok;
      break;

    default:
      error = FT_THROW( Unimplemented_Feature );
      break;
    }

    if ( error )
      return error;
    else
      return BDF_Size_Select( size, 0 );
  }



  FT_CALLBACK_DEF( FT_Error )
  BDF_Glyph_Load( FT_GlyphSlot  slot,
                  FT_Size       size,
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    BDF_Face     bdf    = (BDF_Face)FT_SIZE_FACE( size );
    FT_Face      face   = FT_FACE( bdf );
    FT_Error     error  = FT_Err_Ok;
    FT_Bitmap*   bitmap = &slot->bitmap;
    bdf_glyph_t  glyph;
    int          bpp    = bdf->bdffont->bpp;

    FT_UNUSED( load_flags );


    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    if ( glyph_index >= (FT_UInt)face->num_glyphs )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_TRACE1(( "BDF_Glyph_Load: glyph index %d\n", glyph_index ));

    /* index 0 is the undefined glyph */
    if ( glyph_index == 0 )
      glyph_index = bdf->default_glyph;
    else
      glyph_index--;

    /* slot, bitmap => freetype, glyph => bdflib */
    glyph = bdf->bdffont->glyphs[glyph_index];

    bitmap->rows  = glyph.bbx.height;
    bitmap->width = glyph.bbx.width;
    if ( glyph.bpr > FT_INT_MAX )
      FT_TRACE1(( "BDF_Glyph_Load: too large pitch %d is truncated\n",
                   glyph.bpr ));
    bitmap->pitch = (int)glyph.bpr; /* same as FT_Bitmap.pitch */

    /* note: we don't allocate a new array to hold the bitmap; */
    /*       we can simply point to it                         */
    ft_glyphslot_set_bitmap( slot, glyph.bitmap );

    switch ( bpp )
    {
    case 1:
      bitmap->pixel_mode = FT_PIXEL_MODE_MONO;
      break;
    case 2:
      bitmap->pixel_mode = FT_PIXEL_MODE_GRAY2;
      break;
    case 4:
      bitmap->pixel_mode = FT_PIXEL_MODE_GRAY4;
      break;
    case 8:
      bitmap->pixel_mode = FT_PIXEL_MODE_GRAY;
      bitmap->num_grays  = 256;
      break;
    }

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = glyph.bbx.x_offset;
    slot->bitmap_top  = glyph.bbx.ascent;

    slot->metrics.horiAdvance  = (FT_Pos)( glyph.dwidth * 64 );
    slot->metrics.horiBearingX = (FT_Pos)( glyph.bbx.x_offset * 64 );
    slot->metrics.horiBearingY = (FT_Pos)( glyph.bbx.ascent * 64 );
    slot->metrics.width        = (FT_Pos)( bitmap->width * 64 );
    slot->metrics.height       = (FT_Pos)( bitmap->rows * 64 );

    /*
     * XXX DWIDTH1 and VVECTOR should be parsed and
     * used here, provided such fonts do exist.
     */
    ft_synthesize_vertical_metrics( &slot->metrics,
                                    bdf->bdffont->bbx.height * 64 );

  Exit:
    return error;
  }


 /*
  *
  * BDF SERVICE
  *
  */

  static FT_Error
  bdf_get_bdf_property( BDF_Face          face,
                        const char*       prop_name,
                        BDF_PropertyRec  *aproperty )
  {
    bdf_property_t*  prop;


    FT_ASSERT( face && face->bdffont );

    prop = bdf_get_font_property( face->bdffont, prop_name );
    if ( prop )
    {
      switch ( prop->format )
      {
      case BDF_ATOM:
        aproperty->type   = BDF_PROPERTY_TYPE_ATOM;
        aproperty->u.atom = prop->value.atom;
        break;

      case BDF_INTEGER:
        if ( prop->value.l > 0x7FFFFFFFL || prop->value.l < ( -1 - 0x7FFFFFFFL ) )
        {
          FT_TRACE1(( "bdf_get_bdf_property:"
                      " too large integer 0x%x is truncated\n" ));
        }
        aproperty->type      = BDF_PROPERTY_TYPE_INTEGER;
        aproperty->u.integer = (FT_Int32)prop->value.l;
        break;

      case BDF_CARDINAL:
        if ( prop->value.ul > 0xFFFFFFFFUL )
        {
          FT_TRACE1(( "bdf_get_bdf_property:"
                      " too large cardinal 0x%x is truncated\n" ));
        }
        aproperty->type       = BDF_PROPERTY_TYPE_CARDINAL;
        aproperty->u.cardinal = (FT_UInt32)prop->value.ul;
        break;

      default:
        goto Fail;
      }
      return 0;
    }

  Fail:
    return FT_THROW( Invalid_Argument );
  }


  static FT_Error
  bdf_get_charset_id( BDF_Face      face,
                      const char*  *acharset_encoding,
                      const char*  *acharset_registry )
  {
    *acharset_encoding = face->charset_encoding;
    *acharset_registry = face->charset_registry;

    return 0;
  }


  static const FT_Service_BDFRec  bdf_service_bdf =
  {
    (FT_BDF_GetCharsetIdFunc)bdf_get_charset_id,       /* get_charset_id */
    (FT_BDF_GetPropertyFunc) bdf_get_bdf_property      /* get_property   */
  };


 /*
  *
  * SERVICES LIST
  *
  */

  static const FT_ServiceDescRec  bdf_services[] =
  {
    { FT_SERVICE_ID_BDF,         &bdf_service_bdf },
    { FT_SERVICE_ID_FONT_FORMAT, FT_FONT_FORMAT_BDF },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  bdf_driver_requester( FT_Module    module,
                        const char*  name )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( bdf_services, name );
  }



  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  bdf_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER         |
      FT_MODULE_DRIVER_NO_OUTLINES,
      sizeof ( FT_DriverRec ),

      "bdf",
      0x10000L,
      0x20000L,

      NULL,    /* module-specific interface */

      NULL,                     /* FT_Module_Constructor  module_init   */
      NULL,                     /* FT_Module_Destructor   module_done   */
      bdf_driver_requester      /* FT_Module_Requester    get_interface */
    },

    sizeof ( BDF_FaceRec ),
    sizeof ( FT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    BDF_Face_Init,              /* FT_Face_InitFunc  init_face */
    BDF_Face_Done,              /* FT_Face_DoneFunc  done_face */
    NULL,                       /* FT_Size_InitFunc  init_size */
    NULL,                       /* FT_Size_DoneFunc  done_size */
    NULL,                       /* FT_Slot_InitFunc  init_slot */
    NULL,                       /* FT_Slot_DoneFunc  done_slot */

    BDF_Glyph_Load,             /* FT_Slot_LoadFunc  load_glyph */

    NULL,                       /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    NULL,                       /* FT_Face_GetAdvancesFunc  get_advances */

    BDF_Size_Request,           /* FT_Size_RequestFunc  request_size */
    BDF_Size_Select             /* FT_Size_SelectFunc   select_size  */
  };


/* END */
