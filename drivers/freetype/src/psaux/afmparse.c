/***************************************************************************/
/*                                                                         */
/*  afmparse.c                                                             */
/*                                                                         */
/*    AFM parser (body).                                                   */
/*                                                                         */
/*  Copyright 2006-2016 by                                                 */
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
#include FT_FREETYPE_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H

#include "afmparse.h"
#include "psconv.h"

#include "psauxerr.h"


/***************************************************************************/
/*                                                                         */
/*    AFM_Stream                                                           */
/*                                                                         */
/* The use of AFM_Stream is largely inspired by parseAFM.[ch] from t1lib.  */
/*                                                                         */
/*                                                                         */

  enum
  {
    AFM_STREAM_STATUS_NORMAL,
    AFM_STREAM_STATUS_EOC,
    AFM_STREAM_STATUS_EOL,
    AFM_STREAM_STATUS_EOF
  };


  typedef struct  AFM_StreamRec_
  {
    FT_Byte*  cursor;
    FT_Byte*  base;
    FT_Byte*  limit;

    FT_Int    status;

  } AFM_StreamRec;


#ifndef EOF
#define EOF -1
#endif


  /* this works because empty lines are ignored */
#define AFM_IS_NEWLINE( ch )  ( (ch) == '\r' || (ch) == '\n' )

#define AFM_IS_EOF( ch )      ( (ch) == EOF  || (ch) == '\x1a' )
#define AFM_IS_SPACE( ch )    ( (ch) == ' '  || (ch) == '\t' )

  /* column separator; there is no `column' in the spec actually */
#define AFM_IS_SEP( ch )      ( (ch) == ';' )

#define AFM_GETC()                                                       \
          ( ( (stream)->cursor < (stream)->limit ) ? *(stream)->cursor++ \
                                                   : EOF )

#define AFM_STREAM_KEY_BEGIN( stream )    \
          (char*)( (stream)->cursor - 1 )

#define AFM_STREAM_KEY_LEN( stream, key )           \
          (FT_Offset)( (char*)(stream)->cursor - key - 1 )

#define AFM_STATUS_EOC( stream ) \
          ( (stream)->status >= AFM_STREAM_STATUS_EOC )

#define AFM_STATUS_EOL( stream ) \
          ( (stream)->status >= AFM_STREAM_STATUS_EOL )

#define AFM_STATUS_EOF( stream ) \
          ( (stream)->status >= AFM_STREAM_STATUS_EOF )


  static int
  afm_stream_skip_spaces( AFM_Stream  stream )
  {
    int  ch = 0;  /* make stupid compiler happy */


    if ( AFM_STATUS_EOC( stream ) )
      return ';';

    while ( 1 )
    {
      ch = AFM_GETC();
      if ( !AFM_IS_SPACE( ch ) )
        break;
    }

    if ( AFM_IS_NEWLINE( ch ) )
      stream->status = AFM_STREAM_STATUS_EOL;
    else if ( AFM_IS_SEP( ch ) )
      stream->status = AFM_STREAM_STATUS_EOC;
    else if ( AFM_IS_EOF( ch ) )
      stream->status = AFM_STREAM_STATUS_EOF;

    return ch;
  }


  /* read a key or value in current column */
  static char*
  afm_stream_read_one( AFM_Stream  stream )
  {
    char*  str;


    afm_stream_skip_spaces( stream );
    if ( AFM_STATUS_EOC( stream ) )
      return NULL;

    str = AFM_STREAM_KEY_BEGIN( stream );

    while ( 1 )
    {
      int  ch = AFM_GETC();


      if ( AFM_IS_SPACE( ch ) )
        break;
      else if ( AFM_IS_NEWLINE( ch ) )
      {
        stream->status = AFM_STREAM_STATUS_EOL;
        break;
      }
      else if ( AFM_IS_SEP( ch ) )
      {
        stream->status = AFM_STREAM_STATUS_EOC;
        break;
      }
      else if ( AFM_IS_EOF( ch ) )
      {
        stream->status = AFM_STREAM_STATUS_EOF;
        break;
      }
    }

    return str;
  }


  /* read a string (i.e., read to EOL) */
  static char*
  afm_stream_read_string( AFM_Stream  stream )
  {
    char*  str;


    afm_stream_skip_spaces( stream );
    if ( AFM_STATUS_EOL( stream ) )
      return NULL;

    str = AFM_STREAM_KEY_BEGIN( stream );

    /* scan to eol */
    while ( 1 )
    {
      int  ch = AFM_GETC();


      if ( AFM_IS_NEWLINE( ch ) )
      {
        stream->status = AFM_STREAM_STATUS_EOL;
        break;
      }
      else if ( AFM_IS_EOF( ch ) )
      {
        stream->status = AFM_STREAM_STATUS_EOF;
        break;
      }
    }

    return str;
  }


  /*************************************************************************/
  /*                                                                       */
  /*    AFM_Parser                                                         */
  /*                                                                       */
  /*                                                                       */

  /* all keys defined in Ch. 7-10 of 5004.AFM_Spec.pdf */
  typedef enum  AFM_Token_
  {
    AFM_TOKEN_ASCENDER,
    AFM_TOKEN_AXISLABEL,
    AFM_TOKEN_AXISTYPE,
    AFM_TOKEN_B,
    AFM_TOKEN_BLENDAXISTYPES,
    AFM_TOKEN_BLENDDESIGNMAP,
    AFM_TOKEN_BLENDDESIGNPOSITIONS,
    AFM_TOKEN_C,
    AFM_TOKEN_CC,
    AFM_TOKEN_CH,
    AFM_TOKEN_CAPHEIGHT,
    AFM_TOKEN_CHARWIDTH,
    AFM_TOKEN_CHARACTERSET,
    AFM_TOKEN_CHARACTERS,
    AFM_TOKEN_DESCENDER,
    AFM_TOKEN_ENCODINGSCHEME,
    AFM_TOKEN_ENDAXIS,
    AFM_TOKEN_ENDCHARMETRICS,
    AFM_TOKEN_ENDCOMPOSITES,
    AFM_TOKEN_ENDDIRECTION,
    AFM_TOKEN_ENDFONTMETRICS,
    AFM_TOKEN_ENDKERNDATA,
    AFM_TOKEN_ENDKERNPAIRS,
    AFM_TOKEN_ENDTRACKKERN,
    AFM_TOKEN_ESCCHAR,
    AFM_TOKEN_FAMILYNAME,
    AFM_TOKEN_FONTBBOX,
    AFM_TOKEN_FONTNAME,
    AFM_TOKEN_FULLNAME,
    AFM_TOKEN_ISBASEFONT,
    AFM_TOKEN_ISCIDFONT,
    AFM_TOKEN_ISFIXEDPITCH,
    AFM_TOKEN_ISFIXEDV,
    AFM_TOKEN_ITALICANGLE,
    AFM_TOKEN_KP,
    AFM_TOKEN_KPH,
    AFM_TOKEN_KPX,
    AFM_TOKEN_KPY,
    AFM_TOKEN_L,
    AFM_TOKEN_MAPPINGSCHEME,
    AFM_TOKEN_METRICSSETS,
    AFM_TOKEN_N,
    AFM_TOKEN_NOTICE,
    AFM_TOKEN_PCC,
    AFM_TOKEN_STARTAXIS,
    AFM_TOKEN_STARTCHARMETRICS,
    AFM_TOKEN_STARTCOMPOSITES,
    AFM_TOKEN_STARTDIRECTION,
    AFM_TOKEN_STARTFONTMETRICS,
    AFM_TOKEN_STARTKERNDATA,
    AFM_TOKEN_STARTKERNPAIRS,
    AFM_TOKEN_STARTKERNPAIRS0,
    AFM_TOKEN_STARTKERNPAIRS1,
    AFM_TOKEN_STARTTRACKKERN,
    AFM_TOKEN_STDHW,
    AFM_TOKEN_STDVW,
    AFM_TOKEN_TRACKKERN,
    AFM_TOKEN_UNDERLINEPOSITION,
    AFM_TOKEN_UNDERLINETHICKNESS,
    AFM_TOKEN_VV,
    AFM_TOKEN_VVECTOR,
    AFM_TOKEN_VERSION,
    AFM_TOKEN_W,
    AFM_TOKEN_W0,
    AFM_TOKEN_W0X,
    AFM_TOKEN_W0Y,
    AFM_TOKEN_W1,
    AFM_TOKEN_W1X,
    AFM_TOKEN_W1Y,
    AFM_TOKEN_WX,
    AFM_TOKEN_WY,
    AFM_TOKEN_WEIGHT,
    AFM_TOKEN_WEIGHTVECTOR,
    AFM_TOKEN_XHEIGHT,
    N_AFM_TOKENS,
    AFM_TOKEN_UNKNOWN

  } AFM_Token;


  static const char*  const afm_key_table[N_AFM_TOKENS] =
  {
    "Ascender",
    "AxisLabel",
    "AxisType",
    "B",
    "BlendAxisTypes",
    "BlendDesignMap",
    "BlendDesignPositions",
    "C",
    "CC",
    "CH",
    "CapHeight",
    "CharWidth",
    "CharacterSet",
    "Characters",
    "Descender",
    "EncodingScheme",
    "EndAxis",
    "EndCharMetrics",
    "EndComposites",
    "EndDirection",
    "EndFontMetrics",
    "EndKernData",
    "EndKernPairs",
    "EndTrackKern",
    "EscChar",
    "FamilyName",
    "FontBBox",
    "FontName",
    "FullName",
    "IsBaseFont",
    "IsCIDFont",
    "IsFixedPitch",
    "IsFixedV",
    "ItalicAngle",
    "KP",
    "KPH",
    "KPX",
    "KPY",
    "L",
    "MappingScheme",
    "MetricsSets",
    "N",
    "Notice",
    "PCC",
    "StartAxis",
    "StartCharMetrics",
    "StartComposites",
    "StartDirection",
    "StartFontMetrics",
    "StartKernData",
    "StartKernPairs",
    "StartKernPairs0",
    "StartKernPairs1",
    "StartTrackKern",
    "StdHW",
    "StdVW",
    "TrackKern",
    "UnderlinePosition",
    "UnderlineThickness",
    "VV",
    "VVector",
    "Version",
    "W",
    "W0",
    "W0X",
    "W0Y",
    "W1",
    "W1X",
    "W1Y",
    "WX",
    "WY",
    "Weight",
    "WeightVector",
    "XHeight"
  };


  /*
   * `afm_parser_read_vals' and `afm_parser_next_key' provide
   * high-level operations to an AFM_Stream.  The rest of the
   * parser functions should use them without accessing the
   * AFM_Stream directly.
   */

  FT_LOCAL_DEF( FT_Int )
  afm_parser_read_vals( AFM_Parser  parser,
                        AFM_Value   vals,
                        FT_Int      n )
  {
    AFM_Stream  stream = parser->stream;
    char*       str;
    FT_Int      i;


    if ( n > AFM_MAX_ARGUMENTS )
      return 0;

    for ( i = 0; i < n; i++ )
    {
      FT_Offset  len;
      AFM_Value  val = vals + i;


      if ( val->type == AFM_VALUE_TYPE_STRING )
        str = afm_stream_read_string( stream );
      else
        str = afm_stream_read_one( stream );

      if ( !str )
        break;

      len = AFM_STREAM_KEY_LEN( stream, str );

      switch ( val->type )
      {
      case AFM_VALUE_TYPE_STRING:
      case AFM_VALUE_TYPE_NAME:
        {
          FT_Memory  memory = parser->memory;
          FT_Error   error;


          if ( !FT_QALLOC( val->u.s, len + 1 ) )
          {
            ft_memcpy( val->u.s, str, len );
            val->u.s[len] = '\0';
          }
        }
        break;

      case AFM_VALUE_TYPE_FIXED:
        val->u.f = PS_Conv_ToFixed( (FT_Byte**)(void*)&str,
                                    (FT_Byte*)str + len, 0 );
        break;

      case AFM_VALUE_TYPE_INTEGER:
        val->u.i = PS_Conv_ToInt( (FT_Byte**)(void*)&str,
                                  (FT_Byte*)str + len );
        break;

      case AFM_VALUE_TYPE_BOOL:
        val->u.b = FT_BOOL( len == 4                      &&
                            !ft_strncmp( str, "true", 4 ) );
        break;

      case AFM_VALUE_TYPE_INDEX:
        if ( parser->get_index )
          val->u.i = parser->get_index( str, len, parser->user_data );
        else
          val->u.i = 0;
        break;
      }
    }

    return i;
  }


  FT_LOCAL_DEF( char* )
  afm_parser_next_key( AFM_Parser  parser,
                       FT_Bool     line,
                       FT_Offset*  len )
  {
    AFM_Stream  stream = parser->stream;
    char*       key    = NULL;  /* make stupid compiler happy */


    if ( line )
    {
      while ( 1 )
      {
        /* skip current line */
        if ( !AFM_STATUS_EOL( stream ) )
          afm_stream_read_string( stream );

        stream->status = AFM_STREAM_STATUS_NORMAL;
        key = afm_stream_read_one( stream );

        /* skip empty line */
        if ( !key                      &&
             !AFM_STATUS_EOF( stream ) &&
             AFM_STATUS_EOL( stream )  )
          continue;

        break;
      }
    }
    else
    {
      while ( 1 )
      {
        /* skip current column */
        while ( !AFM_STATUS_EOC( stream ) )
          afm_stream_read_one( stream );

        stream->status = AFM_STREAM_STATUS_NORMAL;
        key = afm_stream_read_one( stream );

        /* skip empty column */
        if ( !key                      &&
             !AFM_STATUS_EOF( stream ) &&
             AFM_STATUS_EOC( stream )  )
          continue;

        break;
      }
    }

    if ( len )
      *len = ( key ) ? (FT_Offset)AFM_STREAM_KEY_LEN( stream, key )
                     : 0;

    return key;
  }


  static AFM_Token
  afm_tokenize( const char*  key,
                FT_Offset    len )
  {
    int  n;


    for ( n = 0; n < N_AFM_TOKENS; n++ )
    {
      if ( *( afm_key_table[n] ) == *key )
      {
        for ( ; n < N_AFM_TOKENS; n++ )
        {
          if ( *( afm_key_table[n] ) != *key )
            return AFM_TOKEN_UNKNOWN;

          if ( ft_strncmp( afm_key_table[n], key, len ) == 0 )
            return (AFM_Token) n;
        }
      }
    }

    return AFM_TOKEN_UNKNOWN;
  }


  FT_LOCAL_DEF( FT_Error )
  afm_parser_init( AFM_Parser  parser,
                   FT_Memory   memory,
                   FT_Byte*    base,
                   FT_Byte*    limit )
  {
    AFM_Stream  stream = NULL;
    FT_Error    error;


    if ( FT_NEW( stream ) )
      return error;

    stream->cursor = stream->base = base;
    stream->limit  = limit;

    /* don't skip the first line during the first call */
    stream->status = AFM_STREAM_STATUS_EOL;

    parser->memory    = memory;
    parser->stream    = stream;
    parser->FontInfo  = NULL;
    parser->get_index = NULL;

    return FT_Err_Ok;
  }


  FT_LOCAL( void )
  afm_parser_done( AFM_Parser  parser )
  {
    FT_Memory  memory = parser->memory;


    FT_FREE( parser->stream );
  }


  static FT_Error
  afm_parser_read_int( AFM_Parser  parser,
                       FT_Int*     aint )
  {
    AFM_ValueRec  val;


    val.type = AFM_VALUE_TYPE_INTEGER;

    if ( afm_parser_read_vals( parser, &val, 1 ) == 1 )
    {
      *aint = val.u.i;

      return FT_Err_Ok;
    }
    else
      return FT_THROW( Syntax_Error );
  }


  static FT_Error
  afm_parse_track_kern( AFM_Parser  parser )
  {
    AFM_FontInfo   fi = parser->FontInfo;
    AFM_TrackKern  tk;
    char*          key;
    FT_Offset      len;
    int            n = -1;
    FT_Int         tmp;


    if ( afm_parser_read_int( parser, &tmp ) )
        goto Fail;

    if ( tmp < 0 )
      goto Fail;

    fi->NumTrackKern = (FT_UInt)tmp;

    if ( fi->NumTrackKern )
    {
      FT_Memory  memory = parser->memory;
      FT_Error   error;


      if ( FT_QNEW_ARRAY( fi->TrackKerns, fi->NumTrackKern ) )
        return error;
    }

    while ( ( key = afm_parser_next_key( parser, 1, &len ) ) != 0 )
    {
      AFM_ValueRec  shared_vals[5];


      switch ( afm_tokenize( key, len ) )
      {
      case AFM_TOKEN_TRACKKERN:
        n++;

        if ( n >= (int)fi->NumTrackKern )
          goto Fail;

        tk = fi->TrackKerns + n;

        shared_vals[0].type = AFM_VALUE_TYPE_INTEGER;
        shared_vals[1].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[2].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[3].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[4].type = AFM_VALUE_TYPE_FIXED;
        if ( afm_parser_read_vals( parser, shared_vals, 5 ) != 5 )
          goto Fail;

        tk->degree     = shared_vals[0].u.i;
        tk->min_ptsize = shared_vals[1].u.f;
        tk->min_kern   = shared_vals[2].u.f;
        tk->max_ptsize = shared_vals[3].u.f;
        tk->max_kern   = shared_vals[4].u.f;

        break;

      case AFM_TOKEN_ENDTRACKKERN:
      case AFM_TOKEN_ENDKERNDATA:
      case AFM_TOKEN_ENDFONTMETRICS:
        fi->NumTrackKern = (FT_UInt)( n + 1 );
        return FT_Err_Ok;

      case AFM_TOKEN_UNKNOWN:
        break;

      default:
        goto Fail;
      }
    }

  Fail:
    return FT_THROW( Syntax_Error );
  }


#undef  KERN_INDEX
#define KERN_INDEX( g1, g2 )  ( ( (FT_ULong)g1 << 16 ) | g2 )


  /* compare two kerning pairs */
  FT_CALLBACK_DEF( int )
  afm_compare_kern_pairs( const void*  a,
                          const void*  b )
  {
    AFM_KernPair  kp1 = (AFM_KernPair)a;
    AFM_KernPair  kp2 = (AFM_KernPair)b;

    FT_ULong  index1 = KERN_INDEX( kp1->index1, kp1->index2 );
    FT_ULong  index2 = KERN_INDEX( kp2->index1, kp2->index2 );


    if ( index1 > index2 )
      return 1;
    else if ( index1 < index2 )
      return -1;
    else
      return 0;
  }


  static FT_Error
  afm_parse_kern_pairs( AFM_Parser  parser )
  {
    AFM_FontInfo  fi = parser->FontInfo;
    AFM_KernPair  kp;
    char*         key;
    FT_Offset     len;
    int           n = -1;
    FT_Int        tmp;


    if ( afm_parser_read_int( parser, &tmp ) )
      goto Fail;

    if ( tmp < 0 )
      goto Fail;

    fi->NumKernPair = (FT_UInt)tmp;

    if ( fi->NumKernPair )
    {
      FT_Memory  memory = parser->memory;
      FT_Error   error;


      if ( FT_QNEW_ARRAY( fi->KernPairs, fi->NumKernPair ) )
        return error;
    }

    while ( ( key = afm_parser_next_key( parser, 1, &len ) ) != 0 )
    {
      AFM_Token  token = afm_tokenize( key, len );


      switch ( token )
      {
      case AFM_TOKEN_KP:
      case AFM_TOKEN_KPX:
      case AFM_TOKEN_KPY:
        {
          FT_Int        r;
          AFM_ValueRec  shared_vals[4];


          n++;

          if ( n >= (int)fi->NumKernPair )
            goto Fail;

          kp = fi->KernPairs + n;

          shared_vals[0].type = AFM_VALUE_TYPE_INDEX;
          shared_vals[1].type = AFM_VALUE_TYPE_INDEX;
          shared_vals[2].type = AFM_VALUE_TYPE_INTEGER;
          shared_vals[3].type = AFM_VALUE_TYPE_INTEGER;
          r = afm_parser_read_vals( parser, shared_vals, 4 );
          if ( r < 3 )
            goto Fail;

          /* index values can't be negative */
          kp->index1 = shared_vals[0].u.u;
          kp->index2 = shared_vals[1].u.u;
          if ( token == AFM_TOKEN_KPY )
          {
            kp->x = 0;
            kp->y = shared_vals[2].u.i;
          }
          else
          {
            kp->x = shared_vals[2].u.i;
            kp->y = ( token == AFM_TOKEN_KP && r == 4 )
                      ? shared_vals[3].u.i : 0;
          }
        }
        break;

      case AFM_TOKEN_ENDKERNPAIRS:
      case AFM_TOKEN_ENDKERNDATA:
      case AFM_TOKEN_ENDFONTMETRICS:
        fi->NumKernPair = (FT_UInt)( n + 1 );
        ft_qsort( fi->KernPairs, fi->NumKernPair,
                  sizeof ( AFM_KernPairRec ),
                  afm_compare_kern_pairs );
        return FT_Err_Ok;

      case AFM_TOKEN_UNKNOWN:
        break;

      default:
        goto Fail;
      }
    }

  Fail:
    return FT_THROW( Syntax_Error );
  }


  static FT_Error
  afm_parse_kern_data( AFM_Parser  parser )
  {
    FT_Error   error;
    char*      key;
    FT_Offset  len;


    while ( ( key = afm_parser_next_key( parser, 1, &len ) ) != 0 )
    {
      switch ( afm_tokenize( key, len ) )
      {
      case AFM_TOKEN_STARTTRACKKERN:
        error = afm_parse_track_kern( parser );
        if ( error )
          return error;
        break;

      case AFM_TOKEN_STARTKERNPAIRS:
      case AFM_TOKEN_STARTKERNPAIRS0:
        error = afm_parse_kern_pairs( parser );
        if ( error )
          return error;
        break;

      case AFM_TOKEN_ENDKERNDATA:
      case AFM_TOKEN_ENDFONTMETRICS:
        return FT_Err_Ok;

      case AFM_TOKEN_UNKNOWN:
        break;

      default:
        goto Fail;
      }
    }

  Fail:
    return FT_THROW( Syntax_Error );
  }


  static FT_Error
  afm_parser_skip_section( AFM_Parser  parser,
                           FT_Int      n,
                           AFM_Token   end_section )
  {
    char*      key;
    FT_Offset  len;


    while ( n-- > 0 )
    {
      key = afm_parser_next_key( parser, 1, NULL );
      if ( !key )
        goto Fail;
    }

    while ( ( key = afm_parser_next_key( parser, 1, &len ) ) != 0 )
    {
      AFM_Token  token = afm_tokenize( key, len );


      if ( token == end_section || token == AFM_TOKEN_ENDFONTMETRICS )
        return FT_Err_Ok;
    }

  Fail:
    return FT_THROW( Syntax_Error );
  }


  FT_LOCAL_DEF( FT_Error )
  afm_parser_parse( AFM_Parser  parser )
  {
    FT_Memory     memory = parser->memory;
    AFM_FontInfo  fi     = parser->FontInfo;
    FT_Error      error  = FT_ERR( Syntax_Error );
    char*         key;
    FT_Offset     len;
    FT_Int        metrics_sets = 0;


    if ( !fi )
      return FT_THROW( Invalid_Argument );

    key = afm_parser_next_key( parser, 1, &len );
    if ( !key || len != 16                              ||
         ft_strncmp( key, "StartFontMetrics", 16 ) != 0 )
      return FT_THROW( Unknown_File_Format );

    while ( ( key = afm_parser_next_key( parser, 1, &len ) ) != 0 )
    {
      AFM_ValueRec  shared_vals[4];


      switch ( afm_tokenize( key, len ) )
      {
      case AFM_TOKEN_METRICSSETS:
        if ( afm_parser_read_int( parser, &metrics_sets ) )
          goto Fail;

        if ( metrics_sets != 0 && metrics_sets != 2 )
        {
          error = FT_THROW( Unimplemented_Feature );

          goto Fail;
        }
        break;

      case AFM_TOKEN_ISCIDFONT:
        shared_vals[0].type = AFM_VALUE_TYPE_BOOL;
        if ( afm_parser_read_vals( parser, shared_vals, 1 ) != 1 )
          goto Fail;

        fi->IsCIDFont = shared_vals[0].u.b;
        break;

      case AFM_TOKEN_FONTBBOX:
        shared_vals[0].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[1].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[2].type = AFM_VALUE_TYPE_FIXED;
        shared_vals[3].type = AFM_VALUE_TYPE_FIXED;
        if ( afm_parser_read_vals( parser, shared_vals, 4 ) != 4 )
          goto Fail;

        fi->FontBBox.xMin = shared_vals[0].u.f;
        fi->FontBBox.yMin = shared_vals[1].u.f;
        fi->FontBBox.xMax = shared_vals[2].u.f;
        fi->FontBBox.yMax = shared_vals[3].u.f;
        break;

      case AFM_TOKEN_ASCENDER:
        shared_vals[0].type = AFM_VALUE_TYPE_FIXED;
        if ( afm_parser_read_vals( parser, shared_vals, 1 ) != 1 )
          goto Fail;

        fi->Ascender = shared_vals[0].u.f;
        break;

      case AFM_TOKEN_DESCENDER:
        shared_vals[0].type = AFM_VALUE_TYPE_FIXED;
        if ( afm_parser_read_vals( parser, shared_vals, 1 ) != 1 )
          goto Fail;

        fi->Descender = shared_vals[0].u.f;
        break;

      case AFM_TOKEN_STARTCHARMETRICS:
        {
          FT_Int  n = 0;


          if ( afm_parser_read_int( parser, &n ) )
            goto Fail;

          error = afm_parser_skip_section( parser, n,
                                           AFM_TOKEN_ENDCHARMETRICS );
          if ( error )
            return error;
        }
        break;

      case AFM_TOKEN_STARTKERNDATA:
        error = afm_parse_kern_data( parser );
        if ( error )
          goto Fail;
        /* fall through since we only support kern data */

      case AFM_TOKEN_ENDFONTMETRICS:
        return FT_Err_Ok;

      default:
        break;
      }
    }

  Fail:
    FT_FREE( fi->TrackKerns );
    fi->NumTrackKern = 0;

    FT_FREE( fi->KernPairs );
    fi->NumKernPair = 0;

    fi->IsCIDFont = 0;

    return error;
  }


/* END */
