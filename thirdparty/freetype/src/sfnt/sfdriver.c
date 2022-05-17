/****************************************************************************
 *
 * sfdriver.c
 *
 *   High-level SFNT driver interface (body).
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


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/sfnt.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/ttnameid.h>

#include "sfdriver.h"
#include "ttload.h"
#include "sfobjs.h"

#include "sferrors.h"

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#include "ttsbit.h"
#endif

#ifdef TT_CONFIG_OPTION_COLOR_LAYERS
#include "ttcolr.h"
#include "ttcpal.h"
#endif

#ifdef FT_CONFIG_OPTION_SVG
#include "ttsvg.h"
#endif

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
#include "ttpost.h"
#endif

#ifdef TT_CONFIG_OPTION_BDF
#include "ttbdf.h"
#include <freetype/internal/services/svbdf.h>
#endif

#include "ttcmap.h"
#include "ttkern.h"
#include "ttmtx.h"

#include <freetype/internal/services/svgldict.h>
#include <freetype/internal/services/svpostnm.h>
#include <freetype/internal/services/svsfnt.h>
#include <freetype/internal/services/svttcmap.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/ftmm.h>
#include <freetype/internal/services/svmm.h>
#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sfdriver


  /*
   * SFNT TABLE SERVICE
   *
   */

  static void*
  get_sfnt_table( TT_Face      face,
                  FT_Sfnt_Tag  tag )
  {
    void*  table;


    switch ( tag )
    {
    case FT_SFNT_HEAD:
      table = &face->header;
      break;

    case FT_SFNT_HHEA:
      table = &face->horizontal;
      break;

    case FT_SFNT_VHEA:
      table = face->vertical_info ? &face->vertical : NULL;
      break;

    case FT_SFNT_OS2:
      table = ( face->os2.version == 0xFFFFU ) ? NULL : &face->os2;
      break;

    case FT_SFNT_POST:
      table = &face->postscript;
      break;

    case FT_SFNT_MAXP:
      table = &face->max_profile;
      break;

    case FT_SFNT_PCLT:
      table = face->pclt.Version ? &face->pclt : NULL;
      break;

    default:
      table = NULL;
    }

    return table;
  }


  static FT_Error
  sfnt_table_info( TT_Face    face,
                   FT_UInt    idx,
                   FT_ULong  *tag,
                   FT_ULong  *offset,
                   FT_ULong  *length )
  {
    if ( !offset || !length )
      return FT_THROW( Invalid_Argument );

    if ( !tag )
      *length = face->num_tables;
    else
    {
      if ( idx >= face->num_tables )
        return FT_THROW( Table_Missing );

      *tag    = face->dir_tables[idx].Tag;
      *offset = face->dir_tables[idx].Offset;
      *length = face->dir_tables[idx].Length;
    }

    return FT_Err_Ok;
  }


  FT_DEFINE_SERVICE_SFNT_TABLEREC(
    sfnt_service_sfnt_table,

    (FT_SFNT_TableLoadFunc)tt_face_load_any,     /* load_table */
    (FT_SFNT_TableGetFunc) get_sfnt_table,       /* get_table  */
    (FT_SFNT_TableInfoFunc)sfnt_table_info       /* table_info */
  )


#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES

  /*
   * GLYPH DICT SERVICE
   *
   */

  static FT_Error
  sfnt_get_glyph_name( FT_Face     face,
                       FT_UInt     glyph_index,
                       FT_Pointer  buffer,
                       FT_UInt     buffer_max )
  {
    FT_String*  gname;
    FT_Error    error;


    error = tt_face_get_ps_name( (TT_Face)face, glyph_index, &gname );
    if ( !error )
      FT_STRCPYN( buffer, gname, buffer_max );

    return error;
  }


  static FT_UInt
  sfnt_get_name_index( FT_Face           face,
                       const FT_String*  glyph_name )
  {
    TT_Face  ttface = (TT_Face)face;

    FT_UInt  i, max_gid = FT_UINT_MAX;


    if ( face->num_glyphs < 0 )
      return 0;
    else if ( (FT_ULong)face->num_glyphs < FT_UINT_MAX )
      max_gid = (FT_UInt)face->num_glyphs;
    else
      FT_TRACE0(( "Ignore glyph names for invalid GID 0x%08x - 0x%08lx\n",
                  FT_UINT_MAX, face->num_glyphs ));

    for ( i = 0; i < max_gid; i++ )
    {
      FT_String*  gname;
      FT_Error    error = tt_face_get_ps_name( ttface, i, &gname );


      if ( error )
        continue;

      if ( !ft_strcmp( glyph_name, gname ) )
        return i;
    }

    return 0;
  }


  FT_DEFINE_SERVICE_GLYPHDICTREC(
    sfnt_service_glyph_dict,

    (FT_GlyphDict_GetNameFunc)  sfnt_get_glyph_name,    /* get_name   */
    (FT_GlyphDict_NameIndexFunc)sfnt_get_name_index     /* name_index */
  )

#endif /* TT_CONFIG_OPTION_POSTSCRIPT_NAMES */


  /*
   * POSTSCRIPT NAME SERVICE
   *
   */

  /* an array representing allowed ASCII characters in a PS string */
  static const unsigned char sfnt_ps_map[16] =
  {
                /*             4        0        C        8 */
    0x00, 0x00, /* 0x00: 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 */
    0x00, 0x00, /* 0x10: 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 */
    0xDE, 0x7C, /* 0x20: 1 1 0 1  1 1 1 0  0 1 1 1  1 1 0 0 */
    0xFF, 0xAF, /* 0x30: 1 1 1 1  1 1 1 1  1 0 1 0  1 1 1 1 */
    0xFF, 0xFF, /* 0x40: 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1 */
    0xFF, 0xD7, /* 0x50: 1 1 1 1  1 1 1 1  1 1 0 1  0 1 1 1 */
    0xFF, 0xFF, /* 0x60: 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1 */
    0xFF, 0x57  /* 0x70: 1 1 1 1  1 1 1 1  0 1 0 1  0 1 1 1 */
  };


  static int
  sfnt_is_postscript( int  c )
  {
    unsigned int  cc;


    if ( c < 0 || c >= 0x80 )
      return 0;

    cc = (unsigned int)c;

    return sfnt_ps_map[cc >> 3] & ( 1 << ( cc & 0x07 ) );
  }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /* Only ASCII letters and digits are taken for a variation font */
  /* instance's PostScript name.                                  */
  /*                                                              */
  /* `ft_isalnum' is a macro, but we need a function here, thus   */
  /* this definition.                                             */
  static int
  sfnt_is_alphanumeric( int  c )
  {
    return ft_isalnum( c );
  }


  /* the implementation of MurmurHash3 is taken and adapted from          */
  /* https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp */

#define ROTL32( x, r )  ( x << r ) | ( x >> ( 32 - r ) )


  static FT_UInt32
  fmix32( FT_UInt32  h )
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
  }


  static void
  murmur_hash_3_128( const void*         key,
                     const unsigned int  len,
                     FT_UInt32           seed,
                     void*               out )
  {
    const FT_Byte*  data    = (const FT_Byte*)key;
    const int       nblocks = (int)len / 16;

    FT_UInt32  h1 = seed;
    FT_UInt32  h2 = seed;
    FT_UInt32  h3 = seed;
    FT_UInt32  h4 = seed;

    const FT_UInt32  c1 = 0x239b961b;
    const FT_UInt32  c2 = 0xab0e9789;
    const FT_UInt32  c3 = 0x38b34ae5;
    const FT_UInt32  c4 = 0xa1e38b93;

    const FT_UInt32*  blocks = (const FT_UInt32*)( data + nblocks * 16 );

    int  i;


    for( i = -nblocks; i; i++ )
    {
      FT_UInt32  k1 = blocks[i * 4 + 0];
      FT_UInt32  k2 = blocks[i * 4 + 1];
      FT_UInt32  k3 = blocks[i * 4 + 2];
      FT_UInt32  k4 = blocks[i * 4 + 3];


      k1 *= c1;
      k1  = ROTL32( k1, 15 );
      k1 *= c2;
      h1 ^= k1;

      h1  = ROTL32( h1, 19 );
      h1 += h2;
      h1  = h1 * 5 + 0x561ccd1b;

      k2 *= c2;
      k2  = ROTL32( k2, 16 );
      k2 *= c3;
      h2 ^= k2;

      h2  = ROTL32( h2, 17 );
      h2 += h3;
      h2  = h2 * 5 + 0x0bcaa747;

      k3 *= c3;
      k3  = ROTL32( k3, 17 );
      k3 *= c4;
      h3 ^= k3;

      h3  = ROTL32( h3, 15 );
      h3 += h4;
      h3  = h3 * 5 + 0x96cd1c35;

      k4 *= c4;
      k4  = ROTL32( k4, 18 );
      k4 *= c1;
      h4 ^= k4;

      h4  = ROTL32( h4, 13 );
      h4 += h1;
      h4  = h4 * 5 + 0x32ac3b17;
    }

    {
      const FT_Byte*  tail = (const FT_Byte*)( data + nblocks * 16 );

      FT_UInt32  k1 = 0;
      FT_UInt32  k2 = 0;
      FT_UInt32  k3 = 0;
      FT_UInt32  k4 = 0;


      switch ( len & 15 )
      {
      case 15:
        k4 ^= (FT_UInt32)tail[14] << 16;
        /* fall through */
      case 14:
        k4 ^= (FT_UInt32)tail[13] << 8;
        /* fall through */
      case 13:
        k4 ^= (FT_UInt32)tail[12];
        k4 *= c4;
        k4  = ROTL32( k4, 18 );
        k4 *= c1;
        h4 ^= k4;
        /* fall through */

      case 12:
        k3 ^= (FT_UInt32)tail[11] << 24;
        /* fall through */
      case 11:
        k3 ^= (FT_UInt32)tail[10] << 16;
        /* fall through */
      case 10:
        k3 ^= (FT_UInt32)tail[9] << 8;
        /* fall through */
      case 9:
        k3 ^= (FT_UInt32)tail[8];
        k3 *= c3;
        k3  = ROTL32( k3, 17 );
        k3 *= c4;
        h3 ^= k3;
        /* fall through */

      case 8:
        k2 ^= (FT_UInt32)tail[7] << 24;
        /* fall through */
      case 7:
        k2 ^= (FT_UInt32)tail[6] << 16;
        /* fall through */
      case 6:
        k2 ^= (FT_UInt32)tail[5] << 8;
        /* fall through */
      case 5:
        k2 ^= (FT_UInt32)tail[4];
        k2 *= c2;
        k2  = ROTL32( k2, 16 );
        k2 *= c3;
        h2 ^= k2;
        /* fall through */

      case 4:
        k1 ^= (FT_UInt32)tail[3] << 24;
        /* fall through */
      case 3:
        k1 ^= (FT_UInt32)tail[2] << 16;
        /* fall through */
      case 2:
        k1 ^= (FT_UInt32)tail[1] << 8;
        /* fall through */
      case 1:
        k1 ^= (FT_UInt32)tail[0];
        k1 *= c1;
        k1  = ROTL32( k1, 15 );
        k1 *= c2;
        h1 ^= k1;
      }
    }

    h1 ^= len;
    h2 ^= len;
    h3 ^= len;
    h4 ^= len;

    h1 += h2;
    h1 += h3;
    h1 += h4;

    h2 += h1;
    h3 += h1;
    h4 += h1;

    h1 = fmix32( h1 );
    h2 = fmix32( h2 );
    h3 = fmix32( h3 );
    h4 = fmix32( h4 );

    h1 += h2;
    h1 += h3;
    h1 += h4;

    h2 += h1;
    h3 += h1;
    h4 += h1;

    ((FT_UInt32*)out)[0] = h1;
    ((FT_UInt32*)out)[1] = h2;
    ((FT_UInt32*)out)[2] = h3;
    ((FT_UInt32*)out)[3] = h4;
  }


#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  typedef int (*char_type_func)( int  c );


  /* Handling of PID/EID 3/0 and 3/1 is the same. */
#define IS_WIN( n )  ( (n)->platformID == 3                             && \
                       ( (n)->encodingID == 1 || (n)->encodingID == 0 ) )

#define IS_APPLE( n )  ( (n)->platformID == 1 && \
                         (n)->encodingID == 0 )

  static char*
  get_win_string( FT_Memory       memory,
                  FT_Stream       stream,
                  TT_Name         entry,
                  char_type_func  char_type,
                  FT_Bool         report_invalid_characters )
  {
    FT_Error  error;

    char*       result = NULL;
    FT_String*  r;
    FT_Char*    p;
    FT_UInt     len;


    if ( FT_QALLOC( result, entry->stringLength / 2 + 1 ) )
      return NULL;

    if ( FT_STREAM_SEEK( entry->stringOffset ) ||
         FT_FRAME_ENTER( entry->stringLength ) )
      goto get_win_string_error;

    r = (FT_String*)result;
    p = (FT_Char*)stream->cursor;

    for ( len = entry->stringLength / 2; len > 0; len--, p += 2 )
    {
      if ( p[0] == 0 && char_type( p[1] ) )
        *r++ = p[1];
      else
      {
        if ( report_invalid_characters )
          FT_TRACE0(( "get_win_string:"
                      " Character 0x%X invalid in PS name string\n",
                      ((unsigned)p[0])*256 + (unsigned)p[1] ));
        break;
      }
    }
    if ( !len )
      *r = '\0';

    FT_FRAME_EXIT();

    if ( !len )
      return result;

  get_win_string_error:
    FT_FREE( result );

    entry->stringLength = 0;
    entry->stringOffset = 0;
    FT_FREE( entry->string );

    return NULL;
  }


  static char*
  get_apple_string( FT_Memory       memory,
                    FT_Stream       stream,
                    TT_Name         entry,
                    char_type_func  char_type,
                    FT_Bool         report_invalid_characters )
  {
    FT_Error  error;

    char*       result = NULL;
    FT_String*  r;
    FT_Char*    p;
    FT_UInt     len;


    if ( FT_QALLOC( result, entry->stringLength + 1 ) )
      return NULL;

    if ( FT_STREAM_SEEK( entry->stringOffset ) ||
         FT_FRAME_ENTER( entry->stringLength ) )
      goto get_apple_string_error;

    r = (FT_String*)result;
    p = (FT_Char*)stream->cursor;

    for ( len = entry->stringLength; len > 0; len--, p++ )
    {
      if ( char_type( *p ) )
        *r++ = *p;
      else
      {
        if ( report_invalid_characters )
          FT_TRACE0(( "get_apple_string:"
                      " Character `%c' (0x%X) invalid in PS name string\n",
                      *p, *p ));
        break;
      }
    }
    if ( !len )
      *r = '\0';

    FT_FRAME_EXIT();

    if ( !len )
      return result;

  get_apple_string_error:
    FT_FREE( result );

    entry->stringOffset = 0;
    entry->stringLength = 0;
    FT_FREE( entry->string );

    return NULL;
  }


  static FT_Bool
  sfnt_get_name_id( TT_Face    face,
                    FT_UShort  id,
                    FT_Int    *win,
                    FT_Int    *apple )
  {
    FT_Int  n;


    *win   = -1;
    *apple = -1;

    for ( n = 0; n < face->num_names; n++ )
    {
      TT_Name  name = face->name_table.names + n;


      if ( name->nameID == id && name->stringLength > 0 )
      {
        if ( IS_WIN( name ) && ( name->languageID == 0x409 || *win == -1 ) )
          *win = n;

        if ( IS_APPLE( name ) && ( name->languageID == 0 || *apple == -1 ) )
          *apple = n;
      }
    }

    return ( *win >= 0 ) || ( *apple >= 0 );
  }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /*
      The maximum length of an axis value descriptor.

      We need 65536 different values for the decimal fraction; this fits
      nicely into five decimal places.  Consequently, it consists of

        . the minus sign if the number is negative,
        . up to five characters for the digits before the decimal point,
        . the decimal point if there is a fractional part, and
        . up to five characters for the digits after the decimal point.

      We also need one byte for the leading `_' character and up to four
      bytes for the axis tag.
   */
#define MAX_VALUE_DESCRIPTOR_LEN  ( 1 + 5 + 1 + 5 + 1 + 4 )


  /* the maximum length of PostScript font names */
#define MAX_PS_NAME_LEN  127


  /*
   * Find the shortest decimal representation of a 16.16 fixed point
   * number.  The function fills `buf' with the result, returning a pointer
   * to the position after the representation's last byte.
   */

  static char*
  fixed2float( FT_Int  fixed,
               char*   buf )
  {
    char*  p;
    char*  q;
    char   tmp[5];

    FT_Int  int_part;
    FT_Int  frac_part;

    FT_Int  i;


    p = buf;

    if ( fixed == 0 )
    {
      *p++ = '0';
      return p;
    }

    if ( fixed < 0 )
    {
      *p++ = '-';
      fixed = NEG_INT( fixed );
    }

    int_part  = ( fixed >> 16 ) & 0xFFFF;
    frac_part = fixed & 0xFFFF;

    /* get digits of integer part (in reverse order) */
    q = tmp;
    while ( int_part > 0 )
    {
      *q++      = '0' + int_part % 10;
      int_part /= 10;
    }

    /* copy digits in correct order to buffer */
    while ( q > tmp )
      *p++ = *--q;

    if ( !frac_part )
      return p;

    /* save position of point */
    q    = p;
    *p++ = '.';

    /* apply rounding */
    frac_part = frac_part * 10 + 5;

    /* get digits of fractional part */
    for ( i = 0; i < 5; i++ )
    {
      *p++ = '0' + (char)( frac_part / 0x10000L );

      frac_part %= 0x10000L;
      if ( !frac_part )
        break;

      frac_part *= 10;
    }

    /*
        If the remainder stored in `frac_part' (after the last FOR loop) is
        smaller than 34480*10, the resulting decimal value minus 0.00001 is
        an equivalent representation of `fixed'.

        The above FOR loop always finds the larger of the two values; I
        verified this by iterating over all possible fixed point numbers.

        If the remainder is 17232*10, both values are equally good, and we
        take the next even number (following IEEE 754's `round to nearest,
        ties to even' rounding rule).

        If the remainder is smaller than 17232*10, the lower of the two
        numbers is nearer to the exact result (values 17232 and 34480 were
        also found by testing all possible fixed point values).

        We use this to find a shorter decimal representation.  If not ending
        with digit zero, we take the representation with less error.
     */
    p--;
    if ( p - q == 5 )  /* five digits? */
    {
      /* take the representation that has zero as the last digit */
      if ( frac_part < 34480 * 10 &&
           *p == '1'              )
        *p = '0';

      /* otherwise use the one with less error */
      else if ( frac_part == 17232 * 10 &&
                *p & 1                  )
        *p -= 1;

      else if ( frac_part < 17232 * 10 &&
                *p != '0'              )
        *p -= 1;
    }

    /* remove trailing zeros */
    while ( *p == '0' )
      *p-- = '\0';

    return p + 1;
  }


  static const char  hexdigits[16] =
  {
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'
  };


  static const char*
  sfnt_get_var_ps_name( TT_Face  face )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;

    FT_UInt     num_coords;
    FT_Fixed*   coords;
    FT_MM_Var*  mm_var;

    FT_Int   found, win, apple;
    FT_UInt  i, j;

    char*  result = NULL;
    char*  p;


    if ( !face->var_postscript_prefix )
    {
      FT_UInt  len;


      /* check whether we have a Variations PostScript Name Prefix */
      found = sfnt_get_name_id( face,
                                TT_NAME_ID_VARIATIONS_PREFIX,
                                &win,
                                &apple );
      if ( !found )
      {
        /* otherwise use the typographic family name */
        found = sfnt_get_name_id( face,
                                  TT_NAME_ID_TYPOGRAPHIC_FAMILY,
                                  &win,
                                  &apple );
      }

      if ( !found )
      {
        /* as a last resort we try the family name; note that this is */
        /* not in the Adobe TechNote, but GX fonts (which predate the */
        /* TechNote) benefit from this behaviour                      */
        found = sfnt_get_name_id( face,
                                  TT_NAME_ID_FONT_FAMILY,
                                  &win,
                                  &apple );
      }

      if ( !found )
      {
        FT_TRACE0(( "sfnt_get_var_ps_name:"
                    " Can't construct PS name prefix for font instances\n" ));
        return NULL;
      }

      /* prefer Windows entries over Apple */
      if ( win != -1 )
        result = get_win_string( face->root.memory,
                                 face->name_table.stream,
                                 face->name_table.names + win,
                                 sfnt_is_alphanumeric,
                                 0 );
      if ( !result && apple != -1 )
        result = get_apple_string( face->root.memory,
                                   face->name_table.stream,
                                   face->name_table.names + apple,
                                   sfnt_is_alphanumeric,
                                   0 );

      if ( !result )
      {
        FT_TRACE0(( "sfnt_get_var_ps_name:"
                    " No valid PS name prefix for font instances found\n" ));
        return NULL;
      }

      len = ft_strlen( result );

      /* sanitize if necessary; we reserve space for 36 bytes (a 128bit  */
      /* checksum as a hex number, preceded by `-' and followed by three */
      /* ASCII dots, to be used if the constructed PS name would be too  */
      /* long); this is also sufficient for a single instance            */
      if ( len > MAX_PS_NAME_LEN - ( 1 + 32 + 3 ) )
      {
        len         = MAX_PS_NAME_LEN - ( 1 + 32 + 3 );
        result[len] = '\0';

        FT_TRACE0(( "sfnt_get_var_ps_name:"
                    " Shortening variation PS name prefix\n" ));
        FT_TRACE0(( "                     "
                    " to %d characters\n", len ));
      }

      face->var_postscript_prefix     = result;
      face->var_postscript_prefix_len = len;
    }

    mm->get_var_blend( FT_FACE( face ),
                       &num_coords,
                       &coords,
                       NULL,
                       &mm_var );

    if ( FT_IS_NAMED_INSTANCE( FT_FACE( face ) ) &&
         !FT_IS_VARIATION( FT_FACE( face ) )     )
    {
      SFNT_Service  sfnt = (SFNT_Service)face->sfnt;

      FT_Long  instance = ( ( face->root.face_index & 0x7FFF0000L ) >> 16 ) - 1;
      FT_UInt  psid     = mm_var->namedstyle[instance].psid;

      char*  ps_name = NULL;


      /* try first to load the name string with index `postScriptNameID' */
      if ( psid == 6                      ||
           ( psid > 255 && psid < 32768 ) )
        (void)sfnt->get_name( face, (FT_UShort)psid, &ps_name );

      if ( ps_name )
      {
        result = ps_name;
        p      = result + ft_strlen( result ) + 1;

        goto check_length;
      }
      else
      {
        /* otherwise construct a name using `subfamilyNameID' */
        FT_UInt  strid = mm_var->namedstyle[instance].strid;

        char*  subfamily_name;
        char*  s;


        (void)sfnt->get_name( face, (FT_UShort)strid, &subfamily_name );

        if ( !subfamily_name )
        {
          FT_TRACE1(( "sfnt_get_var_ps_name:"
                      " can't construct named instance PS name;\n" ));
          FT_TRACE1(( "                     "
                      " trying to construct normal instance PS name\n" ));
          goto construct_instance_name;
        }

        /* after the prefix we have character `-' followed by the   */
        /* subfamily name (using only characters a-z, A-Z, and 0-9) */
        if ( FT_QALLOC( result, face->var_postscript_prefix_len +
                                1 + ft_strlen( subfamily_name ) + 1 ) )
          return NULL;

        ft_strcpy( result, face->var_postscript_prefix );

        p = result + face->var_postscript_prefix_len;
        *p++ = '-';

        s = subfamily_name;
        while ( *s )
        {
          if ( ft_isalnum( *s ) )
            *p++ = *s;
          s++;
        }
        *p++ = '\0';

        FT_FREE( subfamily_name );
      }
    }
    else
    {
      FT_Var_Axis*  axis;


    construct_instance_name:
      axis = mm_var->axis;

      if ( FT_QALLOC( result,
                      face->var_postscript_prefix_len +
                        num_coords * MAX_VALUE_DESCRIPTOR_LEN + 1 ) )
        return NULL;

      p = result;

      ft_strcpy( p, face->var_postscript_prefix );
      p += face->var_postscript_prefix_len;

      for ( i = 0; i < num_coords; i++, coords++, axis++ )
      {
        char  t;


        /* omit axis value descriptor if it is identical */
        /* to the default axis value                     */
        if ( *coords == axis->def )
          continue;

        *p++ = '_';
        p    = fixed2float( *coords, p );

        t = (char)( axis->tag >> 24 );
        if ( t != ' ' && ft_isalnum( t ) )
          *p++ = t;
        t = (char)( axis->tag >> 16 );
        if ( t != ' ' && ft_isalnum( t ) )
          *p++ = t;
        t = (char)( axis->tag >> 8 );
        if ( t != ' ' && ft_isalnum( t ) )
          *p++ = t;
        t = (char)axis->tag;
        if ( t != ' ' && ft_isalnum( t ) )
          *p++ = t;
      }
      *p++ = '\0';
    }

  check_length:
    if ( p - result > MAX_PS_NAME_LEN )
    {
      /* the PS name is too long; replace the part after the prefix with */
      /* a checksum; we use MurmurHash 3 with a hash length of 128 bit   */

      FT_UInt32  seed = 123456789;

      FT_UInt32   hash[4];
      FT_UInt32*  h;


      murmur_hash_3_128( result, p - result, seed, hash );

      p = result + face->var_postscript_prefix_len;
      *p++ = '-';

      /* we convert the hash value to hex digits from back to front */
      p += 32 + 3;
      h  = hash + 3;

      *p-- = '\0';
      *p-- = '.';
      *p-- = '.';
      *p-- = '.';

      for ( i = 0; i < 4; i++, h-- )
      {
        FT_UInt32  v = *h;


        for ( j = 0; j < 8; j++ )
        {
          *p--   = hexdigits[v & 0xF];
          v    >>= 4;
        }
      }
    }

    return result;
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  static const char*
  sfnt_get_ps_name( TT_Face  face )
  {
    FT_Int       found, win, apple;
    const char*  result = NULL;


    if ( face->postscript_name )
      return face->postscript_name;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    if ( face->blend                                 &&
         ( FT_IS_NAMED_INSTANCE( FT_FACE( face ) ) ||
           FT_IS_VARIATION( FT_FACE( face ) )      ) )
    {
      face->postscript_name = sfnt_get_var_ps_name( face );
      return face->postscript_name;
    }
#endif

    /* scan the name table to see whether we have a Postscript name here, */
    /* either in Macintosh or Windows platform encodings                  */
    found = sfnt_get_name_id( face, TT_NAME_ID_PS_NAME, &win, &apple );
    if ( !found )
      return NULL;

    /* prefer Windows entries over Apple */
    if ( win != -1 )
      result = get_win_string( face->root.memory,
                               face->name_table.stream,
                               face->name_table.names + win,
                               sfnt_is_postscript,
                               1 );
    if ( !result && apple != -1 )
      result = get_apple_string( face->root.memory,
                                 face->name_table.stream,
                                 face->name_table.names + apple,
                                 sfnt_is_postscript,
                                 1 );

    face->postscript_name = result;

    return result;
  }


  FT_DEFINE_SERVICE_PSFONTNAMEREC(
    sfnt_service_ps_name,

    (FT_PsName_GetFunc)sfnt_get_ps_name       /* get_ps_font_name */
  )


  /*
   * TT CMAP INFO
   */
  FT_DEFINE_SERVICE_TTCMAPSREC(
    tt_service_get_cmap_info,

    (TT_CMap_Info_GetFunc)tt_get_cmap_info    /* get_cmap_info */
  )


#ifdef TT_CONFIG_OPTION_BDF

  static FT_Error
  sfnt_get_charset_id( TT_Face       face,
                       const char*  *acharset_encoding,
                       const char*  *acharset_registry )
  {
    BDF_PropertyRec  encoding, registry;
    FT_Error         error;


    /* XXX: I don't know whether this is correct, since
     *      tt_face_find_bdf_prop only returns something correct if we have
     *      previously selected a size that is listed in the BDF table.
     *      Should we change the BDF table format to include single offsets
     *      for `CHARSET_REGISTRY' and `CHARSET_ENCODING'?
     */
    error = tt_face_find_bdf_prop( face, "CHARSET_REGISTRY", &registry );
    if ( !error )
    {
      error = tt_face_find_bdf_prop( face, "CHARSET_ENCODING", &encoding );
      if ( !error )
      {
        if ( registry.type == BDF_PROPERTY_TYPE_ATOM &&
             encoding.type == BDF_PROPERTY_TYPE_ATOM )
        {
          *acharset_encoding = encoding.u.atom;
          *acharset_registry = registry.u.atom;
        }
        else
          error = FT_THROW( Invalid_Argument );
      }
    }

    return error;
  }


  FT_DEFINE_SERVICE_BDFRec(
    sfnt_service_bdf,

    (FT_BDF_GetCharsetIdFunc)sfnt_get_charset_id,     /* get_charset_id */
    (FT_BDF_GetPropertyFunc) tt_face_find_bdf_prop    /* get_property   */
  )


#endif /* TT_CONFIG_OPTION_BDF */


  /*
   * SERVICE LIST
   */

#if defined TT_CONFIG_OPTION_POSTSCRIPT_NAMES && defined TT_CONFIG_OPTION_BDF
  FT_DEFINE_SERVICEDESCREC5(
    sfnt_services,

    FT_SERVICE_ID_SFNT_TABLE,           &sfnt_service_sfnt_table,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &sfnt_service_ps_name,
    FT_SERVICE_ID_GLYPH_DICT,           &sfnt_service_glyph_dict,
    FT_SERVICE_ID_BDF,                  &sfnt_service_bdf,
    FT_SERVICE_ID_TT_CMAP,              &tt_service_get_cmap_info )
#elif defined TT_CONFIG_OPTION_POSTSCRIPT_NAMES
  FT_DEFINE_SERVICEDESCREC4(
    sfnt_services,

    FT_SERVICE_ID_SFNT_TABLE,           &sfnt_service_sfnt_table,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &sfnt_service_ps_name,
    FT_SERVICE_ID_GLYPH_DICT,           &sfnt_service_glyph_dict,
    FT_SERVICE_ID_TT_CMAP,              &tt_service_get_cmap_info )
#elif defined TT_CONFIG_OPTION_BDF
  FT_DEFINE_SERVICEDESCREC4(
    sfnt_services,

    FT_SERVICE_ID_SFNT_TABLE,           &sfnt_service_sfnt_table,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &sfnt_service_ps_name,
    FT_SERVICE_ID_BDF,                  &sfnt_service_bdf,
    FT_SERVICE_ID_TT_CMAP,              &tt_service_get_cmap_info )
#else
  FT_DEFINE_SERVICEDESCREC3(
    sfnt_services,

    FT_SERVICE_ID_SFNT_TABLE,           &sfnt_service_sfnt_table,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &sfnt_service_ps_name,
    FT_SERVICE_ID_TT_CMAP,              &tt_service_get_cmap_info )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  sfnt_get_interface( FT_Module    module,
                      const char*  module_interface )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( sfnt_services, module_interface );
  }


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#define PUT_EMBEDDED_BITMAPS( a )  a
#else
#define PUT_EMBEDDED_BITMAPS( a )  NULL
#endif

#ifdef TT_CONFIG_OPTION_COLOR_LAYERS
#define PUT_COLOR_LAYERS( a )  a
#else
#define PUT_COLOR_LAYERS( a )  NULL
#endif

#ifdef FT_CONFIG_OPTION_SVG
#define PUT_SVG_SUPPORT( a )  a
#else
#define PUT_SVG_SUPPORT( a )  NULL
#endif

#define PUT_COLOR_LAYERS_V1( a )  PUT_COLOR_LAYERS( a )

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
#define PUT_PS_NAMES( a )  a
#else
#define PUT_PS_NAMES( a )  NULL
#endif

  FT_DEFINE_SFNT_INTERFACE(
    sfnt_interface,

    tt_face_goto_table,     /* TT_Loader_GotoTableFunc goto_table      */

    sfnt_init_face,         /* TT_Init_Face_Func       init_face       */
    sfnt_load_face,         /* TT_Load_Face_Func       load_face       */
    sfnt_done_face,         /* TT_Done_Face_Func       done_face       */
    sfnt_get_interface,     /* FT_Module_Requester     get_interface   */

    tt_face_load_any,       /* TT_Load_Any_Func        load_any        */

    tt_face_load_head,      /* TT_Load_Table_Func      load_head       */
    tt_face_load_hhea,      /* TT_Load_Metrics_Func    load_hhea       */
    tt_face_load_cmap,      /* TT_Load_Table_Func      load_cmap       */
    tt_face_load_maxp,      /* TT_Load_Table_Func      load_maxp       */
    tt_face_load_os2,       /* TT_Load_Table_Func      load_os2        */
    tt_face_load_post,      /* TT_Load_Table_Func      load_post       */

    tt_face_load_name,      /* TT_Load_Table_Func      load_name       */
    tt_face_free_name,      /* TT_Free_Table_Func      free_name       */

    tt_face_load_kern,      /* TT_Load_Table_Func      load_kern       */
    tt_face_load_gasp,      /* TT_Load_Table_Func      load_gasp       */
    tt_face_load_pclt,      /* TT_Load_Table_Func      load_init       */

    /* see `ttload.h' */
    PUT_EMBEDDED_BITMAPS( tt_face_load_bhed ),
                            /* TT_Load_Table_Func      load_bhed       */
    PUT_EMBEDDED_BITMAPS( tt_face_load_sbit_image ),
                            /* TT_Load_SBit_Image_Func load_sbit_image */

    /* see `ttpost.h' */
    PUT_PS_NAMES( tt_face_get_ps_name   ),
                            /* TT_Get_PS_Name_Func     get_psname      */
    PUT_PS_NAMES( tt_face_free_ps_names ),
                            /* TT_Free_Table_Func      free_psnames    */

    /* since version 2.1.8 */
    tt_face_get_kerning,    /* TT_Face_GetKerningFunc  get_kerning     */

    /* since version 2.2 */
    tt_face_load_font_dir,  /* TT_Load_Table_Func      load_font_dir   */
    tt_face_load_hmtx,      /* TT_Load_Metrics_Func    load_hmtx       */

    /* see `ttsbit.h' and `sfnt.h' */
    PUT_EMBEDDED_BITMAPS( tt_face_load_sbit ),
                            /* TT_Load_Table_Func      load_eblc       */
    PUT_EMBEDDED_BITMAPS( tt_face_free_sbit ),
                            /* TT_Free_Table_Func      free_eblc       */

    PUT_EMBEDDED_BITMAPS( tt_face_set_sbit_strike     ),
                  /* TT_Set_SBit_Strike_Func      set_sbit_strike      */
    PUT_EMBEDDED_BITMAPS( tt_face_load_strike_metrics ),
                  /* TT_Load_Strike_Metrics_Func  load_strike_metrics  */

    PUT_COLOR_LAYERS( tt_face_load_cpal ),
                            /* TT_Load_Table_Func      load_cpal       */
    PUT_COLOR_LAYERS( tt_face_load_colr ),
                            /* TT_Load_Table_Func      load_colr       */
    PUT_COLOR_LAYERS( tt_face_free_cpal ),
                            /* TT_Free_Table_Func      free_cpal       */
    PUT_COLOR_LAYERS( tt_face_free_colr ),
                            /* TT_Free_Table_Func      free_colr       */
    PUT_COLOR_LAYERS( tt_face_palette_set ),
                            /* TT_Set_Palette_Func     set_palette     */
    PUT_COLOR_LAYERS( tt_face_get_colr_layer ),
                            /* TT_Get_Colr_Layer_Func  get_colr_layer  */

    PUT_COLOR_LAYERS_V1( tt_face_get_colr_glyph_paint ),
              /* TT_Get_Color_Glyph_Paint_Func    get_colr_glyph_paint */
    PUT_COLOR_LAYERS_V1( tt_face_get_color_glyph_clipbox ),
              /* TT_Get_Color_Glyph_ClipBox_Func  get_clipbox          */
    PUT_COLOR_LAYERS_V1( tt_face_get_paint_layers ),
              /* TT_Get_Paint_Layers_Func         get_paint_layers     */
    PUT_COLOR_LAYERS_V1( tt_face_get_colorline_stops ),
              /* TT_Get_Paint                     get_paint            */
    PUT_COLOR_LAYERS_V1( tt_face_get_paint ),
              /* TT_Get_Colorline_Stops_Func      get_colorline_stops  */

    PUT_COLOR_LAYERS( tt_face_colr_blend_layer ),
                            /* TT_Blend_Colr_Func      colr_blend      */

    tt_face_get_metrics,    /* TT_Get_Metrics_Func     get_metrics     */

    tt_face_get_name,       /* TT_Get_Name_Func        get_name        */
    sfnt_get_name_id,       /* TT_Get_Name_ID_Func     get_name_id     */

    PUT_SVG_SUPPORT( tt_face_load_svg ),
                            /* TT_Load_Table_Func      load_svg        */
    PUT_SVG_SUPPORT( tt_face_free_svg ),
                            /* TT_Free_Table_Func      free_svg        */
    PUT_SVG_SUPPORT( tt_face_load_svg_doc )
                            /* TT_Load_Svg_Doc_Func    load_svg_doc    */
  )


  FT_DEFINE_MODULE(
    sfnt_module_class,

    0,  /* not a font driver or renderer */
    sizeof ( FT_ModuleRec ),

    "sfnt",     /* driver name                            */
    0x10000L,   /* driver version 1.0                     */
    0x20000L,   /* driver requires FreeType 2.0 or higher */

    (const void*)&sfnt_interface,  /* module specific interface */

    (FT_Module_Constructor)NULL,               /* module_init   */
    (FT_Module_Destructor) NULL,               /* module_done   */
    (FT_Module_Requester)  sfnt_get_interface  /* get_interface */
  )


/* END */
