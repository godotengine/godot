/***************************************************************************/
/*                                                                         */
/*  ttbdf.c                                                                */
/*                                                                         */
/*    TrueType and OpenType embedded BDF properties (body).                */
/*                                                                         */
/*  Copyright 2005-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H
#include "ttbdf.h"

#include "sferrors.h"


#ifdef TT_CONFIG_OPTION_BDF

  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_ttbdf


  FT_LOCAL_DEF( void )
  tt_face_free_bdf_props( TT_Face  face )
  {
    TT_BDF  bdf = &face->bdf;


    if ( bdf->loaded )
    {
      FT_Stream  stream = FT_FACE(face)->stream;


      if ( bdf->table )
        FT_FRAME_RELEASE( bdf->table );

      bdf->table_end    = NULL;
      bdf->strings      = NULL;
      bdf->strings_size = 0;
    }
  }


  static FT_Error
  tt_face_load_bdf_props( TT_Face    face,
                          FT_Stream  stream )
  {
    TT_BDF    bdf = &face->bdf;
    FT_ULong  length;
    FT_Error  error;


    FT_ZERO( bdf );

    error = tt_face_goto_table( face, TTAG_BDF, stream, &length );
    if ( error                                  ||
         length < 8                             ||
         FT_FRAME_EXTRACT( length, bdf->table ) )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    bdf->table_end = bdf->table + length;

    {
      FT_Byte*   p           = bdf->table;
      FT_UInt    version     = FT_NEXT_USHORT( p );
      FT_UInt    num_strikes = FT_NEXT_USHORT( p );
      FT_ULong   strings     = FT_NEXT_ULONG ( p );
      FT_UInt    count;
      FT_Byte*   strike;


      if ( version != 0x0001                 ||
           strings < 8                       ||
           ( strings - 8 ) / 4 < num_strikes ||
           strings + 1 > length              )
      {
        goto BadTable;
      }

      bdf->num_strikes  = num_strikes;
      bdf->strings      = bdf->table + strings;
      bdf->strings_size = length - strings;

      count  = bdf->num_strikes;
      p      = bdf->table + 8;
      strike = p + count * 4;


      for ( ; count > 0; count-- )
      {
        FT_UInt  num_items = FT_PEEK_USHORT( p + 2 );

        /*
         *  We don't need to check the value sets themselves, since this
         *  is done later.
         */
        strike += 10 * num_items;

        p += 4;
      }

      if ( strike > bdf->strings )
        goto BadTable;
    }

    bdf->loaded = 1;

  Exit:
    return error;

  BadTable:
    FT_FRAME_RELEASE( bdf->table );
    FT_ZERO( bdf );
    error = FT_THROW( Invalid_Table );
    goto Exit;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_find_bdf_prop( TT_Face           face,
                         const char*       property_name,
                         BDF_PropertyRec  *aprop )
  {
    TT_BDF     bdf   = &face->bdf;
    FT_Size    size  = FT_FACE(face)->size;
    FT_Error   error = FT_Err_Ok;
    FT_Byte*   p;
    FT_UInt    count;
    FT_Byte*   strike;
    FT_Offset  property_len;


    aprop->type = BDF_PROPERTY_TYPE_NONE;

    if ( bdf->loaded == 0 )
    {
      error = tt_face_load_bdf_props( face, FT_FACE( face )->stream );
      if ( error )
        goto Exit;
    }

    count  = bdf->num_strikes;
    p      = bdf->table + 8;
    strike = p + 4 * count;

    error = FT_ERR( Invalid_Argument );

    if ( !size || !property_name )
      goto Exit;

    property_len = ft_strlen( property_name );
    if ( property_len == 0 )
      goto Exit;

    for ( ; count > 0; count-- )
    {
      FT_UInt  _ppem  = FT_NEXT_USHORT( p );
      FT_UInt  _count = FT_NEXT_USHORT( p );


      if ( _ppem == size->metrics.y_ppem )
      {
        count = _count;
        goto FoundStrike;
      }

      strike += 10 * _count;
    }
    goto Exit;

  FoundStrike:
    p = strike;
    for ( ; count > 0; count-- )
    {
      FT_UInt  type = FT_PEEK_USHORT( p + 4 );


      if ( ( type & 0x10 ) != 0 )
      {
        FT_UInt32  name_offset = FT_PEEK_ULONG( p     );
        FT_UInt32  value       = FT_PEEK_ULONG( p + 6 );

        /* be a bit paranoid for invalid entries here */
        if ( name_offset < bdf->strings_size                    &&
             property_len < bdf->strings_size - name_offset     &&
             ft_strncmp( property_name,
                         (const char*)bdf->strings + name_offset,
                         bdf->strings_size - name_offset ) == 0 )
        {
          switch ( type & 0x0F )
          {
          case 0x00:  /* string */
          case 0x01:  /* atoms */
            /* check that the content is really 0-terminated */
            if ( value < bdf->strings_size &&
                 ft_memchr( bdf->strings + value, 0, bdf->strings_size ) )
            {
              aprop->type   = BDF_PROPERTY_TYPE_ATOM;
              aprop->u.atom = (const char*)bdf->strings + value;
              error         = FT_Err_Ok;
              goto Exit;
            }
            break;

          case 0x02:
            aprop->type      = BDF_PROPERTY_TYPE_INTEGER;
            aprop->u.integer = (FT_Int32)value;
            error            = FT_Err_Ok;
            goto Exit;

          case 0x03:
            aprop->type       = BDF_PROPERTY_TYPE_CARDINAL;
            aprop->u.cardinal = value;
            error             = FT_Err_Ok;
            goto Exit;

          default:
            ;
          }
        }
      }
      p += 10;
    }

  Exit:
    return error;
  }

#else /* !TT_CONFIG_OPTION_BDF */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_bdf_dummy;

#endif /* !TT_CONFIG_OPTION_BDF */


/* END */
