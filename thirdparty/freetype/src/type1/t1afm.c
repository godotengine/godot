/****************************************************************************
 *
 * t1afm.c
 *
 *   AFM support for Type 1 fonts (body).
 *
 * Copyright (C) 1996-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "t1afm.h"
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/psaux.h>
#include "t1errors.h"


#ifndef T1_CONFIG_OPTION_NO_AFM

  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  t1afm


  FT_LOCAL_DEF( void )
  T1_Done_Metrics( FT_Memory     memory,
                   AFM_FontInfo  fi )
  {
    FT_FREE( fi->KernPairs );
    fi->NumKernPair = 0;

    FT_FREE( fi->TrackKerns );
    fi->NumTrackKern = 0;

    FT_FREE( fi );
  }


  /* read a glyph name and return the equivalent glyph index */
  static FT_Int
  t1_get_index( const char*  name,
                FT_Offset    len,
                void*        user_data )
  {
    T1_Font  type1 = (T1_Font)user_data;
    FT_Int   n;


    /* PS string/name length must be < 16-bit */
    if ( len > 0xFFFFU )
      return 0;

    for ( n = 0; n < type1->num_glyphs; n++ )
    {
      char*  gname = (char*)type1->glyph_names[n];


      if ( gname && gname[0] == name[0]        &&
           ft_strlen( gname ) == len           &&
           ft_strncmp( gname, name, len ) == 0 )
        return n;
    }

    return 0;
  }


#undef  KERN_INDEX
#define KERN_INDEX( g1, g2 )  ( ( (FT_ULong)(g1) << 16 ) | (g2) )


  /* compare two kerning pairs */
  FT_COMPARE_DEF( int )
  compare_kern_pairs( const void*  a,
                      const void*  b )
  {
    AFM_KernPair  pair1 = (AFM_KernPair)a;
    AFM_KernPair  pair2 = (AFM_KernPair)b;

    FT_ULong  index1 = KERN_INDEX( pair1->index1, pair1->index2 );
    FT_ULong  index2 = KERN_INDEX( pair2->index1, pair2->index2 );


    if ( index1 > index2 )
      return 1;
    else if ( index1 < index2 )
      return -1;
    else
      return 0;
  }


  /* parse a PFM file -- for now, only read the kerning pairs */
  static FT_Error
  T1_Read_PFM( FT_Face       t1_face,
               FT_Stream     stream,
               AFM_FontInfo  fi )
  {
    FT_Error      error  = FT_Err_Ok;
    FT_Memory     memory = stream->memory;
    FT_Byte*      start;
    FT_Byte*      limit;
    FT_Byte*      p;
    AFM_KernPair  kp;
    FT_Int        width_table_length;
    FT_CharMap    oldcharmap;
    FT_CharMap    charmap;
    FT_Int        n;


    start = (FT_Byte*)stream->cursor;
    limit = (FT_Byte*)stream->limit;

    /* Figure out how long the width table is.          */
    /* This info is a little-endian short at offset 99. */
    p = start + 99;
    if ( p + 2 > limit )
    {
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }
    width_table_length = FT_PEEK_USHORT_LE( p );

    p += 18 + width_table_length;
    if ( p + 0x12 > limit || FT_PEEK_USHORT_LE( p ) < 0x12 )
      /* extension table is probably optional */
      goto Exit;

    /* Kerning offset is 14 bytes from start of extensions table. */
    p += 14;
    p = start + FT_PEEK_ULONG_LE( p );

    if ( p == start )
      /* zero offset means no table */
      goto Exit;

    if ( p + 2 > limit )
    {
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    fi->NumKernPair = FT_PEEK_USHORT_LE( p );
    p += 2;
    if ( p + 4 * fi->NumKernPair > limit )
    {
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* Actually, kerning pairs are simply optional! */
    if ( fi->NumKernPair == 0 )
      goto Exit;

    /* allocate the pairs */
    if ( FT_QNEW_ARRAY( fi->KernPairs, fi->NumKernPair ) )
      goto Exit;

    /* now, read each kern pair */
    kp    = fi->KernPairs;
    limit = p + 4 * fi->NumKernPair;

    /* PFM kerning data are stored by encoding rather than glyph index, */
    /* so find the PostScript charmap of this font and install it       */
    /* temporarily.  If we find no PostScript charmap, then just use    */
    /* the default and hope it is the right one.                        */
    oldcharmap = t1_face->charmap;
    charmap    = NULL;

    for ( n = 0; n < t1_face->num_charmaps; n++ )
    {
      charmap = t1_face->charmaps[n];
      /* check against PostScript pseudo platform */
      if ( charmap->platform_id == 7 )
      {
        error = FT_Set_Charmap( t1_face, charmap );
        if ( error )
          goto Exit;
        break;
      }
    }

    /* Kerning info is stored as:             */
    /*                                        */
    /*   encoding of first glyph (1 byte)     */
    /*   encoding of second glyph (1 byte)    */
    /*   offset (little-endian short)         */
    for ( ; p < limit; p += 4 )
    {
      kp->index1 = FT_Get_Char_Index( t1_face, p[0] );
      kp->index2 = FT_Get_Char_Index( t1_face, p[1] );

      kp->x = (FT_Int)FT_PEEK_SHORT_LE( p + 2 );
      kp->y = 0;

      kp++;
    }

    if ( oldcharmap )
      error = FT_Set_Charmap( t1_face, oldcharmap );
    if ( error )
      goto Exit;

    /* now, sort the kern pairs according to their glyph indices */
    ft_qsort( fi->KernPairs, fi->NumKernPair, sizeof ( AFM_KernPairRec ),
              compare_kern_pairs );

  Exit:
    if ( error )
    {
      FT_FREE( fi->KernPairs );
      fi->NumKernPair = 0;
    }

    return error;
  }


  /* parse a metrics file -- either AFM or PFM depending on what */
  /* it turns out to be                                          */
  FT_LOCAL_DEF( FT_Error )
  T1_Read_Metrics( FT_Face    t1_face,
                   FT_Stream  stream )
  {
    PSAux_Service  psaux;
    FT_Memory      memory  = stream->memory;
    AFM_ParserRec  parser;
    AFM_FontInfo   fi      = NULL;
    FT_Error       error   = FT_ERR( Unknown_File_Format );
    T1_Face        face    = (T1_Face)t1_face;
    T1_Font        t1_font = &face->type1;


    if ( face->afm_data )
    {
      FT_TRACE1(( "T1_Read_Metrics:"
                  " Freeing previously attached metrics data.\n" ));
      T1_Done_Metrics( memory, (AFM_FontInfo)face->afm_data );

      face->afm_data = NULL;
    }

    if ( FT_NEW( fi )                   ||
         FT_FRAME_ENTER( stream->size ) )
      goto Exit;

    fi->FontBBox  = t1_font->font_bbox;
    fi->Ascender  = t1_font->font_bbox.yMax;
    fi->Descender = t1_font->font_bbox.yMin;

    psaux = (PSAux_Service)face->psaux;
    if ( psaux->afm_parser_funcs )
    {
      error = psaux->afm_parser_funcs->init( &parser,
                                             stream->memory,
                                             stream->cursor,
                                             stream->limit );

      if ( !error )
      {
        parser.FontInfo  = fi;
        parser.get_index = t1_get_index;
        parser.user_data = t1_font;

        error = psaux->afm_parser_funcs->parse( &parser );
        psaux->afm_parser_funcs->done( &parser );
      }
    }

    if ( FT_ERR_EQ( error, Unknown_File_Format ) )
    {
      FT_Byte*  start = stream->cursor;


      /* MS Windows allows versions up to 0x3FF without complaining */
      if ( stream->size > 6                              &&
           start[1] < 4                                  &&
           FT_PEEK_ULONG_LE( start + 2 ) == stream->size )
        error = T1_Read_PFM( t1_face, stream, fi );
    }

    if ( !error )
    {
      t1_font->font_bbox = fi->FontBBox;

      t1_face->bbox.xMin =   fi->FontBBox.xMin            >> 16;
      t1_face->bbox.yMin =   fi->FontBBox.yMin            >> 16;
      /* no `U' suffix here to 0xFFFF! */
      t1_face->bbox.xMax = ( fi->FontBBox.xMax + 0xFFFF ) >> 16;
      t1_face->bbox.yMax = ( fi->FontBBox.yMax + 0xFFFF ) >> 16;

      /* no `U' suffix here to 0x8000! */
      t1_face->ascender  = (FT_Short)( ( fi->Ascender  + 0x8000 ) >> 16 );
      t1_face->descender = (FT_Short)( ( fi->Descender + 0x8000 ) >> 16 );

      if ( fi->NumKernPair )
      {
        t1_face->face_flags |= FT_FACE_FLAG_KERNING;
        face->afm_data       = fi;
        fi                   = NULL;
      }
    }

    FT_FRAME_EXIT();

  Exit:
    if ( fi )
      T1_Done_Metrics( memory, fi );

    return error;
  }


  /* find the kerning for a given glyph pair */
  FT_LOCAL_DEF( void )
  T1_Get_Kerning( AFM_FontInfo  fi,
                  FT_UInt       glyph1,
                  FT_UInt       glyph2,
                  FT_Vector*    kerning )
  {
    AFM_KernPair  min, mid, max;
    FT_ULong      idx = KERN_INDEX( glyph1, glyph2 );


    /* simple binary search */
    min = fi->KernPairs;
    max = min + fi->NumKernPair - 1;

    while ( min <= max )
    {
      FT_ULong  midi;


      mid  = min + ( max - min ) / 2;
      midi = KERN_INDEX( mid->index1, mid->index2 );

      if ( midi == idx )
      {
        kerning->x = mid->x;
        kerning->y = mid->y;

        return;
      }

      if ( midi < idx )
        min = mid + 1;
      else
        max = mid - 1;
    }

    kerning->x = 0;
    kerning->y = 0;
  }


  FT_LOCAL_DEF( FT_Error )
  T1_Get_Track_Kerning( FT_Face    face,
                        FT_Fixed   ptsize,
                        FT_Int     degree,
                        FT_Fixed*  kerning )
  {
    AFM_FontInfo  fi = (AFM_FontInfo)( (T1_Face)face )->afm_data;
    FT_UInt       i;


    if ( !fi )
      return FT_THROW( Invalid_Argument );

    for ( i = 0; i < fi->NumTrackKern; i++ )
    {
      AFM_TrackKern  tk = fi->TrackKerns + i;


      if ( tk->degree != degree )
        continue;

      if ( ptsize < tk->min_ptsize )
        *kerning = tk->min_kern;
      else if ( ptsize > tk->max_ptsize )
        *kerning = tk->max_kern;
      else
      {
        *kerning = FT_MulDiv( ptsize - tk->min_ptsize,
                              tk->max_kern - tk->min_kern,
                              tk->max_ptsize - tk->min_ptsize ) +
                   tk->min_kern;
      }
    }

    return FT_Err_Ok;
  }

#else /* T1_CONFIG_OPTION_NO_AFM */

  /* ANSI C doesn't like empty source files */
  typedef int  _t1_afm_dummy;

#endif /* T1_CONFIG_OPTION_NO_AFM */


/* END */
