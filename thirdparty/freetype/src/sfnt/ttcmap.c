/****************************************************************************
 *
 * ttcmap.c
 *
 *   TrueType character mapping table (cmap) support (body).
 *
 * Copyright (C) 2002-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H

#include "sferrors.h"           /* must come before FT_INTERNAL_VALIDATE_H */

#include FT_INTERNAL_VALIDATE_H
#include FT_INTERNAL_STREAM_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include "ttload.h"
#include "ttcmap.h"
#include "ttpost.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttcmap


#define TT_PEEK_SHORT   FT_PEEK_SHORT
#define TT_PEEK_USHORT  FT_PEEK_USHORT
#define TT_PEEK_UINT24  FT_PEEK_UOFF3
#define TT_PEEK_LONG    FT_PEEK_LONG
#define TT_PEEK_ULONG   FT_PEEK_ULONG

#define TT_NEXT_SHORT   FT_NEXT_SHORT
#define TT_NEXT_USHORT  FT_NEXT_USHORT
#define TT_NEXT_UINT24  FT_NEXT_UOFF3
#define TT_NEXT_LONG    FT_NEXT_LONG
#define TT_NEXT_ULONG   FT_NEXT_ULONG


  /* Too large glyph index return values are caught in `FT_Get_Char_Index' */
  /* and `FT_Get_Next_Char' (the latter calls the internal `next' function */
  /* again in this case).  To mark character code return values as invalid */
  /* it is sufficient to set the corresponding glyph index return value to */
  /* zero.                                                                 */


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap_init( TT_CMap   cmap,
                FT_Byte*  table )
  {
    cmap->data = table;
    return FT_Err_Ok;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           FORMAT 0                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET         TYPE          DESCRIPTION
   *
   *   format      0              USHORT        must be 0
   *   length      2              USHORT        table length in bytes
   *   language    4              USHORT        Mac language code
   *   glyph_ids   6              BYTE[256]     array of glyph indices
   *               262
   */

#ifdef TT_CONFIG_CMAP_FORMAT_0

  FT_CALLBACK_DEF( FT_Error )
  tt_cmap0_validate( FT_Byte*      table,
                     FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_UInt   length;


    if ( table + 2 + 2 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 2;           /* skip format */
    length = TT_NEXT_USHORT( p );

    if ( table + length > valid->limit || length < 262 )
      FT_INVALID_TOO_SHORT;

    /* check glyph indices whenever necessary */
    if ( valid->level >= FT_VALIDATE_TIGHT )
    {
      FT_UInt  n, idx;


      p = table + 6;
      for ( n = 0; n < 256; n++ )
      {
        idx = *p++;
        if ( idx >= TT_VALID_GLYPH_COUNT( valid ) )
          FT_INVALID_GLYPH_ID;
      }
    }

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap0_char_index( TT_CMap    cmap,
                       FT_UInt32  char_code )
  {
    FT_Byte*  table = cmap->data;


    return char_code < 256 ? table[6 + char_code] : 0;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap0_char_next( TT_CMap     cmap,
                      FT_UInt32  *pchar_code )
  {
    FT_Byte*   table    = cmap->data;
    FT_UInt32  charcode = *pchar_code;
    FT_UInt32  result   = 0;
    FT_UInt    gindex   = 0;


    table += 6;  /* go to glyph IDs */
    while ( ++charcode < 256 )
    {
      gindex = table[charcode];
      if ( gindex != 0 )
      {
        result = charcode;
        break;
      }
    }

    *pchar_code = result;
    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap0_get_info( TT_CMap       cmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 4;


    cmap_info->format   = 0;
    cmap_info->language = (FT_ULong)TT_PEEK_USHORT( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap0_class_rec,

      sizeof ( TT_CMapRec ),

      (FT_CMap_InitFunc)     tt_cmap_init,         /* init       */
      (FT_CMap_DoneFunc)     NULL,                 /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap0_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap0_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    0,
    (TT_CMap_ValidateFunc)tt_cmap0_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap0_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_0 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 2                             *****/
  /*****                                                               *****/
  /***** This is used for certain CJK encodings that encode text in a  *****/
  /***** mixed 8/16 bits encoding along the following lines.           *****/
  /*****                                                               *****/
  /***** * Certain byte values correspond to an 8-bit character code   *****/
  /*****   (typically in the range 0..127 for ASCII compatibility).    *****/
  /*****                                                               *****/
  /***** * Certain byte values signal the first byte of a 2-byte       *****/
  /*****   character code (but these values are also valid as the      *****/
  /*****   second byte of a 2-byte character).                         *****/
  /*****                                                               *****/
  /***** The following charmap lookup and iteration functions all      *****/
  /***** assume that the value `charcode' fulfills the following.      *****/
  /*****                                                               *****/
  /*****   - For one-byte characters, `charcode' is simply the         *****/
  /*****     character code.                                           *****/
  /*****                                                               *****/
  /*****   - For two-byte characters, `charcode' is the 2-byte         *****/
  /*****     character code in big endian format.  More precisely:     *****/
  /*****                                                               *****/
  /*****       (charcode >> 8)    is the first byte value              *****/
  /*****       (charcode & 0xFF)  is the second byte value             *****/
  /*****                                                               *****/
  /***** Note that not all values of `charcode' are valid according    *****/
  /***** to these rules, and the function moderately checks the        *****/
  /***** arguments.                                                    *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET         TYPE            DESCRIPTION
   *
   *   format      0              USHORT          must be 2
   *   length      2              USHORT          table length in bytes
   *   language    4              USHORT          Mac language code
   *   keys        6              USHORT[256]     sub-header keys
   *   subs        518            SUBHEAD[NSUBS]  sub-headers array
   *   glyph_ids   518+NSUB*8     USHORT[]        glyph ID array
   *
   * The `keys' table is used to map charcode high bytes to sub-headers.
   * The value of `NSUBS' is the number of sub-headers defined in the
   * table and is computed by finding the maximum of the `keys' table.
   *
   * Note that for any `n', `keys[n]' is a byte offset within the `subs'
   * table, i.e., it is the corresponding sub-header index multiplied
   * by 8.
   *
   * Each sub-header has the following format.
   *
   *   NAME        OFFSET      TYPE            DESCRIPTION
   *
   *   first       0           USHORT          first valid low-byte
   *   count       2           USHORT          number of valid low-bytes
   *   delta       4           SHORT           see below
   *   offset      6           USHORT          see below
   *
   * A sub-header defines, for each high byte, the range of valid
   * low bytes within the charmap.  Note that the range defined by `first'
   * and `count' must be completely included in the interval [0..255]
   * according to the specification.
   *
   * If a character code is contained within a given sub-header, then
   * mapping it to a glyph index is done as follows.
   *
   * - The value of `offset' is read.  This is a _byte_ distance from the
   *   location of the `offset' field itself into a slice of the
   *   `glyph_ids' table.  Let's call it `slice' (it is a USHORT[], too).
   *
   * - The value `slice[char.lo - first]' is read.  If it is 0, there is
   *   no glyph for the charcode.  Otherwise, the value of `delta' is
   *   added to it (modulo 65536) to form a new glyph index.
   *
   * It is up to the validation routine to check that all offsets fall
   * within the glyph IDs table (and not within the `subs' table itself or
   * outside of the CMap).
   */

#ifdef TT_CONFIG_CMAP_FORMAT_2

  FT_CALLBACK_DEF( FT_Error )
  tt_cmap2_validate( FT_Byte*      table,
                     FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_UInt   length;

    FT_UInt   n, max_subs;
    FT_Byte*  keys;        /* keys table     */
    FT_Byte*  subs;        /* sub-headers    */
    FT_Byte*  glyph_ids;   /* glyph ID array */


    if ( table + 2 + 2 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 2;           /* skip format */
    length = TT_NEXT_USHORT( p );

    if ( table + length > valid->limit || length < 6 + 512 )
      FT_INVALID_TOO_SHORT;

    keys = table + 6;

    /* parse keys to compute sub-headers count */
    p        = keys;
    max_subs = 0;
    for ( n = 0; n < 256; n++ )
    {
      FT_UInt  idx = TT_NEXT_USHORT( p );


      /* value must be multiple of 8 */
      if ( valid->level >= FT_VALIDATE_PARANOID && ( idx & 7 ) != 0 )
        FT_INVALID_DATA;

      idx >>= 3;

      if ( idx > max_subs )
        max_subs = idx;
    }

    FT_ASSERT( p == table + 518 );

    subs      = p;
    glyph_ids = subs + ( max_subs + 1 ) * 8;
    if ( glyph_ids > valid->limit )
      FT_INVALID_TOO_SHORT;

    /* parse sub-headers */
    for ( n = 0; n <= max_subs; n++ )
    {
      FT_UInt  first_code, code_count, offset;
      FT_Int   delta;


      first_code = TT_NEXT_USHORT( p );
      code_count = TT_NEXT_USHORT( p );
      delta      = TT_NEXT_SHORT( p );
      offset     = TT_NEXT_USHORT( p );

      /* many Dynalab fonts have empty sub-headers */
      if ( code_count == 0 )
        continue;

      /* check range within 0..255 */
      if ( valid->level >= FT_VALIDATE_PARANOID )
      {
        if ( first_code >= 256 || code_count > 256 - first_code )
          FT_INVALID_DATA;
      }

      /* check offset */
      if ( offset != 0 )
      {
        FT_Byte*  ids;


        ids = p - 2 + offset;
        if ( ids < glyph_ids || ids + code_count * 2 > table + length )
          FT_INVALID_OFFSET;

        /* check glyph IDs */
        if ( valid->level >= FT_VALIDATE_TIGHT )
        {
          FT_Byte*  limit = p + code_count * 2;
          FT_UInt   idx;


          for ( ; p < limit; )
          {
            idx = TT_NEXT_USHORT( p );
            if ( idx != 0 )
            {
              idx = (FT_UInt)( (FT_Int)idx + delta ) & 0xFFFFU;
              if ( idx >= TT_VALID_GLYPH_COUNT( valid ) )
                FT_INVALID_GLYPH_ID;
            }
          }
        }
      }
    }

    return FT_Err_Ok;
  }


  /* return sub header corresponding to a given character code */
  /* NULL on invalid charcode                                  */
  static FT_Byte*
  tt_cmap2_get_subheader( FT_Byte*   table,
                          FT_UInt32  char_code )
  {
    FT_Byte*  result = NULL;


    if ( char_code < 0x10000UL )
    {
      FT_UInt   char_lo = (FT_UInt)( char_code & 0xFF );
      FT_UInt   char_hi = (FT_UInt)( char_code >> 8 );
      FT_Byte*  p       = table + 6;    /* keys table       */
      FT_Byte*  subs    = table + 518;  /* subheaders table */
      FT_Byte*  sub;


      if ( char_hi == 0 )
      {
        /* an 8-bit character code -- we use subHeader 0 in this case */
        /* to test whether the character code is in the charmap       */
        /*                                                            */
        sub = subs;  /* jump to first sub-header */

        /* check that the sub-header for this byte is 0, which */
        /* indicates that it is really a valid one-byte value; */
        /* otherwise, return 0                                 */
        /*                                                     */
        p += char_lo * 2;
        if ( TT_PEEK_USHORT( p ) != 0 )
          goto Exit;
      }
      else
      {
        /* a 16-bit character code */

        /* jump to key entry  */
        p  += char_hi * 2;
        /* jump to sub-header */
        sub = subs + ( FT_PAD_FLOOR( TT_PEEK_USHORT( p ), 8 ) );

        /* check that the high byte isn't a valid one-byte value */
        if ( sub == subs )
          goto Exit;
      }

      result = sub;
    }

  Exit:
    return result;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap2_char_index( TT_CMap    cmap,
                       FT_UInt32  char_code )
  {
    FT_Byte*  table   = cmap->data;
    FT_UInt   result  = 0;
    FT_Byte*  subheader;


    subheader = tt_cmap2_get_subheader( table, char_code );
    if ( subheader )
    {
      FT_Byte*  p   = subheader;
      FT_UInt   idx = (FT_UInt)(char_code & 0xFF);
      FT_UInt   start, count;
      FT_Int    delta;
      FT_UInt   offset;


      start  = TT_NEXT_USHORT( p );
      count  = TT_NEXT_USHORT( p );
      delta  = TT_NEXT_SHORT ( p );
      offset = TT_PEEK_USHORT( p );

      idx -= start;
      if ( idx < count && offset != 0 )
      {
        p  += offset + 2 * idx;
        idx = TT_PEEK_USHORT( p );

        if ( idx != 0 )
          result = (FT_UInt)( (FT_Int)idx + delta ) & 0xFFFFU;
      }
    }

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap2_char_next( TT_CMap     cmap,
                      FT_UInt32  *pcharcode )
  {
    FT_Byte*   table    = cmap->data;
    FT_UInt    gindex   = 0;
    FT_UInt32  result   = 0;
    FT_UInt32  charcode = *pcharcode + 1;
    FT_Byte*   subheader;


    while ( charcode < 0x10000UL )
    {
      subheader = tt_cmap2_get_subheader( table, charcode );
      if ( subheader )
      {
        FT_Byte*  p       = subheader;
        FT_UInt   start   = TT_NEXT_USHORT( p );
        FT_UInt   count   = TT_NEXT_USHORT( p );
        FT_Int    delta   = TT_NEXT_SHORT ( p );
        FT_UInt   offset  = TT_PEEK_USHORT( p );
        FT_UInt   char_lo = (FT_UInt)( charcode & 0xFF );
        FT_UInt   pos, idx;


        if ( char_lo >= start + count && charcode <= 0xFF )
        {
          /* this happens only for a malformed cmap */
          charcode = 0x100;
          continue;
        }

        if ( offset == 0 )
        {
          if ( charcode == 0x100 )
            goto Exit; /* this happens only for a malformed cmap */
          goto Next_SubHeader;
        }

        if ( char_lo < start )
        {
          char_lo = start;
          pos     = 0;
        }
        else
          pos = (FT_UInt)( char_lo - start );

        p       += offset + pos * 2;
        charcode = FT_PAD_FLOOR( charcode, 256 ) + char_lo;

        for ( ; pos < count; pos++, charcode++ )
        {
          idx = TT_NEXT_USHORT( p );

          if ( idx != 0 )
          {
            gindex = (FT_UInt)( (FT_Int)idx + delta ) & 0xFFFFU;
            if ( gindex != 0 )
            {
              result = charcode;
              goto Exit;
            }
          }
        }

        /* if unsuccessful, avoid `charcode' leaving */
        /* the current 256-character block           */
        if ( count )
          charcode--;
      }

      /* If `charcode' is <= 0xFF, retry with `charcode + 1'.      */
      /* Otherwise jump to the next 256-character block and retry. */
    Next_SubHeader:
      if ( charcode <= 0xFF )
        charcode++;
      else
        charcode = FT_PAD_FLOOR( charcode, 0x100 ) + 0x100;
    }

  Exit:
    *pcharcode = result;

    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap2_get_info( TT_CMap       cmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 4;


    cmap_info->format   = 2;
    cmap_info->language = (FT_ULong)TT_PEEK_USHORT( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap2_class_rec,

      sizeof ( TT_CMapRec ),

      (FT_CMap_InitFunc)     tt_cmap_init,         /* init       */
      (FT_CMap_DoneFunc)     NULL,                 /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap2_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap2_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    2,
    (TT_CMap_ValidateFunc)tt_cmap2_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap2_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_2 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           FORMAT 4                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME          OFFSET         TYPE              DESCRIPTION
   *
   *   format        0              USHORT            must be 4
   *   length        2              USHORT            table length
   *                                                  in bytes
   *   language      4              USHORT            Mac language code
   *
   *   segCountX2    6              USHORT            2*NUM_SEGS
   *   searchRange   8              USHORT            2*(1 << LOG_SEGS)
   *   entrySelector 10             USHORT            LOG_SEGS
   *   rangeShift    12             USHORT            segCountX2 -
   *                                                    searchRange
   *
   *   endCount      14             USHORT[NUM_SEGS]  end charcode for
   *                                                  each segment; last
   *                                                  is 0xFFFF
   *
   *   pad           14+NUM_SEGS*2  USHORT            padding
   *
   *   startCount    16+NUM_SEGS*2  USHORT[NUM_SEGS]  first charcode for
   *                                                  each segment
   *
   *   idDelta       16+NUM_SEGS*4  SHORT[NUM_SEGS]   delta for each
   *                                                  segment
   *   idOffset      16+NUM_SEGS*6  SHORT[NUM_SEGS]   range offset for
   *                                                  each segment; can be
   *                                                  zero
   *
   *   glyphIds      16+NUM_SEGS*8  USHORT[]          array of glyph ID
   *                                                  ranges
   *
   * Character codes are modelled by a series of ordered (increasing)
   * intervals called segments.  Each segment has start and end codes,
   * provided by the `startCount' and `endCount' arrays.  Segments must
   * not overlap, and the last segment should always contain the value
   * 0xFFFF for `endCount'.
   *
   * The fields `searchRange', `entrySelector' and `rangeShift' are better
   * ignored (they are traces of over-engineering in the TrueType
   * specification).
   *
   * Each segment also has a signed `delta', as well as an optional offset
   * within the `glyphIds' table.
   *
   * If a segment's idOffset is 0, the glyph index corresponding to any
   * charcode within the segment is obtained by adding the value of
   * `idDelta' directly to the charcode, modulo 65536.
   *
   * Otherwise, a glyph index is taken from the glyph IDs sub-array for
   * the segment, and the value of `idDelta' is added to it.
   *
   *
   * Finally, note that a lot of fonts contain an invalid last segment,
   * where `start' and `end' are correctly set to 0xFFFF but both `delta'
   * and `offset' are incorrect (e.g., `opens___.ttf' which comes with
   * OpenOffice.org).  We need special code to deal with them correctly.
   */

#ifdef TT_CONFIG_CMAP_FORMAT_4

  typedef struct  TT_CMap4Rec_
  {
    TT_CMapRec  cmap;
    FT_UInt32   cur_charcode;   /* current charcode */
    FT_UInt     cur_gindex;     /* current glyph index */

    FT_UInt     num_ranges;
    FT_UInt     cur_range;
    FT_UInt     cur_start;
    FT_UInt     cur_end;
    FT_Int      cur_delta;
    FT_Byte*    cur_values;

  } TT_CMap4Rec, *TT_CMap4;


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap4_init( TT_CMap4  cmap,
                 FT_Byte*  table )
  {
    FT_Byte*  p;


    cmap->cmap.data    = table;

    p                  = table + 6;
    cmap->num_ranges   = FT_PEEK_USHORT( p ) >> 1;
    cmap->cur_charcode = (FT_UInt32)0xFFFFFFFFUL;
    cmap->cur_gindex   = 0;

    return FT_Err_Ok;
  }


  static FT_Int
  tt_cmap4_set_range( TT_CMap4  cmap,
                      FT_UInt   range_index )
  {
    FT_Byte*  table = cmap->cmap.data;
    FT_Byte*  p;
    FT_UInt   num_ranges = cmap->num_ranges;


    while ( range_index < num_ranges )
    {
      FT_UInt  offset;


      p             = table + 14 + range_index * 2;
      cmap->cur_end = FT_PEEK_USHORT( p );

      p              += 2 + num_ranges * 2;
      cmap->cur_start = FT_PEEK_USHORT( p );

      p              += num_ranges * 2;
      cmap->cur_delta = FT_PEEK_SHORT( p );

      p     += num_ranges * 2;
      offset = FT_PEEK_USHORT( p );

      /* some fonts have an incorrect last segment; */
      /* we have to catch it                        */
      if ( range_index     >= num_ranges - 1 &&
           cmap->cur_start == 0xFFFFU        &&
           cmap->cur_end   == 0xFFFFU        )
      {
        TT_Face   face  = (TT_Face)cmap->cmap.cmap.charmap.face;
        FT_Byte*  limit = face->cmap_table + face->cmap_size;


        if ( offset && p + offset + 2 > limit )
        {
          cmap->cur_delta = 1;
          offset          = 0;
        }
      }

      if ( offset != 0xFFFFU )
      {
        cmap->cur_values = offset ? p + offset : NULL;
        cmap->cur_range  = range_index;
        return 0;
      }

      /* we skip empty segments */
      range_index++;
    }

    return -1;
  }


  /* search the index of the charcode next to cmap->cur_charcode; */
  /* caller should call tt_cmap4_set_range with proper range      */
  /* before calling this function                                 */
  /*                                                              */
  static void
  tt_cmap4_next( TT_CMap4  cmap )
  {
    TT_Face   face  = (TT_Face)cmap->cmap.cmap.charmap.face;
    FT_Byte*  limit = face->cmap_table + face->cmap_size;

    FT_UInt  charcode;


    if ( cmap->cur_charcode >= 0xFFFFUL )
      goto Fail;

    charcode = (FT_UInt)cmap->cur_charcode + 1;

    if ( charcode < cmap->cur_start )
      charcode = cmap->cur_start;

    for (;;)
    {
      FT_Byte*  values = cmap->cur_values;
      FT_UInt   end    = cmap->cur_end;
      FT_Int    delta  = cmap->cur_delta;


      if ( charcode <= end )
      {
        if ( values )
        {
          FT_Byte*  p = values + 2 * ( charcode - cmap->cur_start );


          /* if p > limit, the whole segment is invalid */
          if ( p > limit )
            goto Next_Segment;

          do
          {
            FT_UInt  gindex = FT_NEXT_USHORT( p );


            if ( gindex )
            {
              gindex = (FT_UInt)( (FT_Int)gindex + delta ) & 0xFFFFU;
              if ( gindex )
              {
                cmap->cur_charcode = charcode;
                cmap->cur_gindex   = gindex;
                return;
              }
            }
          } while ( ++charcode <= end );
        }
        else
        {
          do
          {
            FT_UInt  gindex = (FT_UInt)( (FT_Int)charcode + delta ) & 0xFFFFU;


            if ( gindex >= (FT_UInt)face->root.num_glyphs )
            {
              /* we have an invalid glyph index; if there is an overflow, */
              /* we can adjust `charcode', otherwise the whole segment is */
              /* invalid                                                  */
              gindex = 0;

              if ( (FT_Int)charcode + delta < 0 &&
                   (FT_Int)end + delta >= 0     )
                charcode = (FT_UInt)( -delta );

              else if ( (FT_Int)charcode + delta < 0x10000L &&
                        (FT_Int)end + delta >= 0x10000L     )
                charcode = (FT_UInt)( 0x10000L - delta );

              else
                goto Next_Segment;
            }

            if ( gindex )
            {
              cmap->cur_charcode = charcode;
              cmap->cur_gindex   = gindex;
              return;
            }
          } while ( ++charcode <= end );
        }
      }

    Next_Segment:
      /* we need to find another range */
      if ( tt_cmap4_set_range( cmap, cmap->cur_range + 1 ) < 0 )
        break;

      if ( charcode < cmap->cur_start )
        charcode = cmap->cur_start;
    }

  Fail:
    cmap->cur_charcode = (FT_UInt32)0xFFFFFFFFUL;
    cmap->cur_gindex   = 0;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap4_validate( FT_Byte*      table,
                     FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_UInt   length;

    FT_Byte   *ends, *starts, *offsets, *deltas, *glyph_ids;
    FT_UInt   num_segs;
    FT_Error  error = FT_Err_Ok;


    if ( table + 2 + 2 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 2;           /* skip format */
    length = TT_NEXT_USHORT( p );

    /* in certain fonts, the `length' field is invalid and goes */
    /* out of bound.  We try to correct this here...            */
    if ( table + length > valid->limit )
    {
      if ( valid->level >= FT_VALIDATE_TIGHT )
        FT_INVALID_TOO_SHORT;

      length = (FT_UInt)( valid->limit - table );
    }

    if ( length < 16 )
      FT_INVALID_TOO_SHORT;

    p        = table + 6;
    num_segs = TT_NEXT_USHORT( p );   /* read segCountX2 */

    if ( valid->level >= FT_VALIDATE_PARANOID )
    {
      /* check that we have an even value here */
      if ( num_segs & 1 )
        FT_INVALID_DATA;
    }

    num_segs /= 2;

    if ( length < 16 + num_segs * 2 * 4 )
      FT_INVALID_TOO_SHORT;

    /* check the search parameters - even though we never use them */
    /*                                                             */
    if ( valid->level >= FT_VALIDATE_PARANOID )
    {
      /* check the values of `searchRange', `entrySelector', `rangeShift' */
      FT_UInt  search_range   = TT_NEXT_USHORT( p );
      FT_UInt  entry_selector = TT_NEXT_USHORT( p );
      FT_UInt  range_shift    = TT_NEXT_USHORT( p );


      if ( ( search_range | range_shift ) & 1 )  /* must be even values */
        FT_INVALID_DATA;

      search_range /= 2;
      range_shift  /= 2;

      /* `search range' is the greatest power of 2 that is <= num_segs */

      if ( search_range                > num_segs                 ||
           search_range * 2            < num_segs                 ||
           search_range + range_shift != num_segs                 ||
           search_range               != ( 1U << entry_selector ) )
        FT_INVALID_DATA;
    }

    ends      = table   + 14;
    starts    = table   + 16 + num_segs * 2;
    deltas    = starts  + num_segs * 2;
    offsets   = deltas  + num_segs * 2;
    glyph_ids = offsets + num_segs * 2;

    /* check last segment; its end count value must be 0xFFFF */
    if ( valid->level >= FT_VALIDATE_PARANOID )
    {
      p = ends + ( num_segs - 1 ) * 2;
      if ( TT_PEEK_USHORT( p ) != 0xFFFFU )
        FT_INVALID_DATA;
    }

    {
      FT_UInt   start, end, offset, n;
      FT_UInt   last_start = 0, last_end = 0;
      FT_Int    delta;
      FT_Byte*  p_start   = starts;
      FT_Byte*  p_end     = ends;
      FT_Byte*  p_delta   = deltas;
      FT_Byte*  p_offset  = offsets;


      for ( n = 0; n < num_segs; n++ )
      {
        p      = p_offset;
        start  = TT_NEXT_USHORT( p_start );
        end    = TT_NEXT_USHORT( p_end );
        delta  = TT_NEXT_SHORT( p_delta );
        offset = TT_NEXT_USHORT( p_offset );

        if ( start > end )
          FT_INVALID_DATA;

        /* this test should be performed at default validation level; */
        /* unfortunately, some popular Asian fonts have overlapping   */
        /* ranges in their charmaps                                   */
        /*                                                            */
        if ( start <= last_end && n > 0 )
        {
          if ( valid->level >= FT_VALIDATE_TIGHT )
            FT_INVALID_DATA;
          else
          {
            /* allow overlapping segments, provided their start points */
            /* and end points, respectively, are in ascending order    */
            /*                                                         */
            if ( last_start > start || last_end > end )
              error |= TT_CMAP_FLAG_UNSORTED;
            else
              error |= TT_CMAP_FLAG_OVERLAPPING;
          }
        }

        if ( offset && offset != 0xFFFFU )
        {
          p += offset;  /* start of glyph ID array */

          /* check that we point within the glyph IDs table only */
          if ( valid->level >= FT_VALIDATE_TIGHT )
          {
            if ( p < glyph_ids                                ||
                 p + ( end - start + 1 ) * 2 > table + length )
              FT_INVALID_DATA;
          }
          /* Some fonts handle the last segment incorrectly.  In */
          /* theory, 0xFFFF might point to an ordinary glyph --  */
          /* a cmap 4 is versatile and could be used for any     */
          /* encoding, not only Unicode.  However, reality shows */
          /* that far too many fonts are sloppy and incorrectly  */
          /* set all fields but `start' and `end' for the last   */
          /* segment if it contains only a single character.     */
          /*                                                     */
          /* We thus omit the test here, delaying it to the      */
          /* routines that actually access the cmap.             */
          else if ( n != num_segs - 1                       ||
                    !( start == 0xFFFFU && end == 0xFFFFU ) )
          {
            if ( p < glyph_ids                              ||
                 p + ( end - start + 1 ) * 2 > valid->limit )
              FT_INVALID_DATA;
          }

          /* check glyph indices within the segment range */
          if ( valid->level >= FT_VALIDATE_TIGHT )
          {
            FT_UInt  i, idx;


            for ( i = start; i < end; i++ )
            {
              idx = FT_NEXT_USHORT( p );
              if ( idx != 0 )
              {
                idx = (FT_UInt)( (FT_Int)idx + delta ) & 0xFFFFU;

                if ( idx >= TT_VALID_GLYPH_COUNT( valid ) )
                  FT_INVALID_GLYPH_ID;
              }
            }
          }
        }
        else if ( offset == 0xFFFFU )
        {
          /* some fonts (erroneously?) use a range offset of 0xFFFF */
          /* to mean missing glyph in cmap table                    */
          /*                                                        */
          if ( valid->level >= FT_VALIDATE_PARANOID    ||
               n != num_segs - 1                       ||
               !( start == 0xFFFFU && end == 0xFFFFU ) )
            FT_INVALID_DATA;
        }

        last_start = start;
        last_end   = end;
      }
    }

    return error;
  }


  static FT_UInt
  tt_cmap4_char_map_linear( TT_CMap     cmap,
                            FT_UInt32*  pcharcode,
                            FT_Bool     next )
  {
    TT_Face   face  = (TT_Face)cmap->cmap.charmap.face;
    FT_Byte*  limit = face->cmap_table + face->cmap_size;


    FT_UInt    num_segs2, start, end, offset;
    FT_Int     delta;
    FT_UInt    i, num_segs;
    FT_UInt32  charcode = *pcharcode;
    FT_UInt    gindex   = 0;
    FT_Byte*   p;
    FT_Byte*   q;


    p = cmap->data + 6;
    num_segs2 = FT_PAD_FLOOR( TT_PEEK_USHORT( p ), 2 );

    num_segs = num_segs2 >> 1;

    if ( !num_segs )
      return 0;

    if ( next )
      charcode++;

    if ( charcode > 0xFFFFU )
      return 0;

    /* linear search */
    p = cmap->data + 14;               /* ends table   */
    q = cmap->data + 16 + num_segs2;   /* starts table */

    for ( i = 0; i < num_segs; i++ )
    {
      end   = TT_NEXT_USHORT( p );
      start = TT_NEXT_USHORT( q );

      if ( charcode < start )
      {
        if ( next )
          charcode = start;
        else
          break;
      }

    Again:
      if ( charcode <= end )
      {
        FT_Byte*  r;


        r       = q - 2 + num_segs2;
        delta   = TT_PEEK_SHORT( r );
        r      += num_segs2;
        offset  = TT_PEEK_USHORT( r );

        /* some fonts have an incorrect last segment; */
        /* we have to catch it                        */
        if ( i >= num_segs - 1                  &&
             start == 0xFFFFU && end == 0xFFFFU )
        {
          if ( offset && r + offset + 2 > limit )
          {
            delta  = 1;
            offset = 0;
          }
        }

        if ( offset == 0xFFFFU )
          continue;

        if ( offset )
        {
          r += offset + ( charcode - start ) * 2;

          /* if r > limit, the whole segment is invalid */
          if ( next && r > limit )
            continue;

          gindex = TT_PEEK_USHORT( r );
          if ( gindex )
          {
            gindex = (FT_UInt)( (FT_Int)gindex + delta ) & 0xFFFFU;
            if ( gindex >= (FT_UInt)face->root.num_glyphs )
              gindex = 0;
          }
        }
        else
        {
          gindex = (FT_UInt)( (FT_Int)charcode + delta ) & 0xFFFFU;

          if ( next && gindex >= (FT_UInt)face->root.num_glyphs )
          {
            /* we have an invalid glyph index; if there is an overflow, */
            /* we can adjust `charcode', otherwise the whole segment is */
            /* invalid                                                  */
            gindex = 0;

            if ( (FT_Int)charcode + delta < 0 &&
                 (FT_Int)end + delta >= 0     )
              charcode = (FT_UInt)( -delta );

            else if ( (FT_Int)charcode + delta < 0x10000L &&
                      (FT_Int)end + delta >= 0x10000L     )
              charcode = (FT_UInt)( 0x10000L - delta );

            else
              continue;
          }
        }

        if ( next && !gindex )
        {
          if ( charcode >= 0xFFFFU )
            break;

          charcode++;
          goto Again;
        }

        break;
      }
    }

    if ( next )
      *pcharcode = charcode;

    return gindex;
  }


  static FT_UInt
  tt_cmap4_char_map_binary( TT_CMap     cmap,
                            FT_UInt32*  pcharcode,
                            FT_Bool     next )
  {
    TT_Face   face  = (TT_Face)cmap->cmap.charmap.face;
    FT_Byte*  limit = face->cmap_table + face->cmap_size;

    FT_UInt   num_segs2, start, end, offset;
    FT_Int    delta;
    FT_UInt   max, min, mid, num_segs;
    FT_UInt   charcode = (FT_UInt)*pcharcode;
    FT_UInt   gindex   = 0;
    FT_Byte*  p;


    p = cmap->data + 6;
    num_segs2 = FT_PAD_FLOOR( TT_PEEK_USHORT( p ), 2 );

    if ( !num_segs2 )
      return 0;

    num_segs = num_segs2 >> 1;

    /* make compiler happy */
    mid = num_segs;
    end = 0xFFFFU;

    if ( next )
      charcode++;

    min = 0;
    max = num_segs;

    /* binary search */
    while ( min < max )
    {
      mid    = ( min + max ) >> 1;
      p      = cmap->data + 14 + mid * 2;
      end    = TT_PEEK_USHORT( p );
      p     += 2 + num_segs2;
      start  = TT_PEEK_USHORT( p );

      if ( charcode < start )
        max = mid;
      else if ( charcode > end )
        min = mid + 1;
      else
      {
        p     += num_segs2;
        delta  = TT_PEEK_SHORT( p );
        p     += num_segs2;
        offset = TT_PEEK_USHORT( p );

        /* some fonts have an incorrect last segment; */
        /* we have to catch it                        */
        if ( mid >= num_segs - 1                &&
             start == 0xFFFFU && end == 0xFFFFU )
        {
          if ( offset && p + offset + 2 > limit )
          {
            delta  = 1;
            offset = 0;
          }
        }

        /* search the first segment containing `charcode' */
        if ( cmap->flags & TT_CMAP_FLAG_OVERLAPPING )
        {
          FT_UInt  i;


          /* call the current segment `max' */
          max = mid;

          if ( offset == 0xFFFFU )
            mid = max + 1;

          /* search in segments before the current segment */
          for ( i = max; i > 0; i-- )
          {
            FT_UInt   prev_end;
            FT_Byte*  old_p;


            old_p    = p;
            p        = cmap->data + 14 + ( i - 1 ) * 2;
            prev_end = TT_PEEK_USHORT( p );

            if ( charcode > prev_end )
            {
              p = old_p;
              break;
            }

            end    = prev_end;
            p     += 2 + num_segs2;
            start  = TT_PEEK_USHORT( p );
            p     += num_segs2;
            delta  = TT_PEEK_SHORT( p );
            p     += num_segs2;
            offset = TT_PEEK_USHORT( p );

            if ( offset != 0xFFFFU )
              mid = i - 1;
          }

          /* no luck */
          if ( mid == max + 1 )
          {
            if ( i != max )
            {
              p      = cmap->data + 14 + max * 2;
              end    = TT_PEEK_USHORT( p );
              p     += 2 + num_segs2;
              start  = TT_PEEK_USHORT( p );
              p     += num_segs2;
              delta  = TT_PEEK_SHORT( p );
              p     += num_segs2;
              offset = TT_PEEK_USHORT( p );
            }

            mid = max;

            /* search in segments after the current segment */
            for ( i = max + 1; i < num_segs; i++ )
            {
              FT_UInt  next_end, next_start;


              p          = cmap->data + 14 + i * 2;
              next_end   = TT_PEEK_USHORT( p );
              p         += 2 + num_segs2;
              next_start = TT_PEEK_USHORT( p );

              if ( charcode < next_start )
                break;

              end    = next_end;
              start  = next_start;
              p     += num_segs2;
              delta  = TT_PEEK_SHORT( p );
              p     += num_segs2;
              offset = TT_PEEK_USHORT( p );

              if ( offset != 0xFFFFU )
                mid = i;
            }
            i--;

            /* still no luck */
            if ( mid == max )
            {
              mid = i;

              break;
            }
          }

          /* end, start, delta, and offset are for the i'th segment */
          if ( mid != i )
          {
            p      = cmap->data + 14 + mid * 2;
            end    = TT_PEEK_USHORT( p );
            p     += 2 + num_segs2;
            start  = TT_PEEK_USHORT( p );
            p     += num_segs2;
            delta  = TT_PEEK_SHORT( p );
            p     += num_segs2;
            offset = TT_PEEK_USHORT( p );
          }
        }
        else
        {
          if ( offset == 0xFFFFU )
            break;
        }

        if ( offset )
        {
          p += offset + ( charcode - start ) * 2;

          /* if p > limit, the whole segment is invalid */
          if ( next && p > limit )
            break;

          gindex = TT_PEEK_USHORT( p );
          if ( gindex )
          {
            gindex = (FT_UInt)( (FT_Int)gindex + delta ) & 0xFFFFU;
            if ( gindex >= (FT_UInt)face->root.num_glyphs )
              gindex = 0;
          }
        }
        else
        {
          gindex = (FT_UInt)( (FT_Int)charcode + delta ) & 0xFFFFU;

          if ( next && gindex >= (FT_UInt)face->root.num_glyphs )
          {
            /* we have an invalid glyph index; if there is an overflow, */
            /* we can adjust `charcode', otherwise the whole segment is */
            /* invalid                                                  */
            gindex = 0;

            if ( (FT_Int)charcode + delta < 0 &&
                 (FT_Int)end + delta >= 0     )
              charcode = (FT_UInt)( -delta );

            else if ( (FT_Int)charcode + delta < 0x10000L &&
                      (FT_Int)end + delta >= 0x10000L     )
              charcode = (FT_UInt)( 0x10000L - delta );
          }
        }

        break;
      }
    }

    if ( next )
    {
      TT_CMap4  cmap4 = (TT_CMap4)cmap;


      /* if `charcode' is not in any segment, then `mid' is */
      /* the segment nearest to `charcode'                  */

      if ( charcode > end )
      {
        mid++;
        if ( mid == num_segs )
          return 0;
      }

      if ( tt_cmap4_set_range( cmap4, mid ) )
      {
        if ( gindex )
          *pcharcode = charcode;
      }
      else
      {
        cmap4->cur_charcode = charcode;

        if ( gindex )
          cmap4->cur_gindex = gindex;
        else
        {
          cmap4->cur_charcode = charcode;
          tt_cmap4_next( cmap4 );
          gindex = cmap4->cur_gindex;
        }

        if ( gindex )
          *pcharcode = cmap4->cur_charcode;
      }
    }

    return gindex;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap4_char_index( TT_CMap    cmap,
                       FT_UInt32  char_code )
  {
    if ( char_code >= 0x10000UL )
      return 0;

    if ( cmap->flags & TT_CMAP_FLAG_UNSORTED )
      return tt_cmap4_char_map_linear( cmap, &char_code, 0 );
    else
      return tt_cmap4_char_map_binary( cmap, &char_code, 0 );
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap4_char_next( TT_CMap     cmap,
                      FT_UInt32  *pchar_code )
  {
    FT_UInt  gindex;


    if ( *pchar_code >= 0xFFFFU )
      return 0;

    if ( cmap->flags & TT_CMAP_FLAG_UNSORTED )
      gindex = tt_cmap4_char_map_linear( cmap, pchar_code, 1 );
    else
    {
      TT_CMap4  cmap4 = (TT_CMap4)cmap;


      /* no need to search */
      if ( *pchar_code == cmap4->cur_charcode )
      {
        tt_cmap4_next( cmap4 );
        gindex = cmap4->cur_gindex;
        if ( gindex )
          *pchar_code = cmap4->cur_charcode;
      }
      else
        gindex = tt_cmap4_char_map_binary( cmap, pchar_code, 1 );
    }

    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap4_get_info( TT_CMap       cmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 4;


    cmap_info->format   = 4;
    cmap_info->language = (FT_ULong)TT_PEEK_USHORT( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap4_class_rec,

      sizeof ( TT_CMap4Rec ),

      (FT_CMap_InitFunc)     tt_cmap4_init,        /* init       */
      (FT_CMap_DoneFunc)     NULL,                 /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap4_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap4_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    4,
    (TT_CMap_ValidateFunc)tt_cmap4_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap4_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_4 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 6                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET          TYPE             DESCRIPTION
   *
   *   format       0              USHORT           must be 6
   *   length       2              USHORT           table length in bytes
   *   language     4              USHORT           Mac language code
   *
   *   first        6              USHORT           first segment code
   *   count        8              USHORT           segment size in chars
   *   glyphIds     10             USHORT[count]    glyph IDs
   *
   * A very simplified segment mapping.
   */

#ifdef TT_CONFIG_CMAP_FORMAT_6

  FT_CALLBACK_DEF( FT_Error )
  tt_cmap6_validate( FT_Byte*      table,
                     FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_UInt   length, count;


    if ( table + 10 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 2;
    length = TT_NEXT_USHORT( p );

    p      = table + 8;             /* skip language and start index */
    count  = TT_NEXT_USHORT( p );

    if ( table + length > valid->limit || length < 10 + count * 2 )
      FT_INVALID_TOO_SHORT;

    /* check glyph indices */
    if ( valid->level >= FT_VALIDATE_TIGHT )
    {
      FT_UInt  gindex;


      for ( ; count > 0; count-- )
      {
        gindex = TT_NEXT_USHORT( p );
        if ( gindex >= TT_VALID_GLYPH_COUNT( valid ) )
          FT_INVALID_GLYPH_ID;
      }
    }

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap6_char_index( TT_CMap    cmap,
                       FT_UInt32  char_code )
  {
    FT_Byte*  table  = cmap->data;
    FT_UInt   result = 0;
    FT_Byte*  p      = table + 6;
    FT_UInt   start  = TT_NEXT_USHORT( p );
    FT_UInt   count  = TT_NEXT_USHORT( p );
    FT_UInt   idx    = (FT_UInt)( char_code - start );


    if ( idx < count )
    {
      p += 2 * idx;
      result = TT_PEEK_USHORT( p );
    }

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap6_char_next( TT_CMap     cmap,
                      FT_UInt32  *pchar_code )
  {
    FT_Byte*   table     = cmap->data;
    FT_UInt32  result    = 0;
    FT_UInt32  char_code = *pchar_code + 1;
    FT_UInt    gindex    = 0;

    FT_Byte*   p         = table + 6;
    FT_UInt    start     = TT_NEXT_USHORT( p );
    FT_UInt    count     = TT_NEXT_USHORT( p );
    FT_UInt    idx;


    if ( char_code >= 0x10000UL )
      return 0;

    if ( char_code < start )
      char_code = start;

    idx = (FT_UInt)( char_code - start );
    p  += 2 * idx;

    for ( ; idx < count; idx++ )
    {
      gindex = TT_NEXT_USHORT( p );
      if ( gindex != 0 )
      {
        result = char_code;
        break;
      }

      if ( char_code >= 0xFFFFU )
        return 0;

      char_code++;
    }

    *pchar_code = result;
    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap6_get_info( TT_CMap       cmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 4;


    cmap_info->format   = 6;
    cmap_info->language = (FT_ULong)TT_PEEK_USHORT( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap6_class_rec,

      sizeof ( TT_CMapRec ),

      (FT_CMap_InitFunc)     tt_cmap_init,         /* init       */
      (FT_CMap_DoneFunc)     NULL,                 /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap6_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap6_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    6,
    (TT_CMap_ValidateFunc)tt_cmap6_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap6_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_6 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 8                             *****/
  /*****                                                               *****/
  /***** It is hard to completely understand what the OpenType spec    *****/
  /***** says about this format, but here is my conclusion.            *****/
  /*****                                                               *****/
  /***** The purpose of this format is to easily map UTF-16 text to    *****/
  /***** glyph indices.  Basically, the `char_code' must be in one of  *****/
  /***** the following formats.                                        *****/
  /*****                                                               *****/
  /*****   - A 16-bit value that isn't part of the Unicode Surrogates  *****/
  /*****     Area (i.e. U+D800-U+DFFF).                                *****/
  /*****                                                               *****/
  /*****   - A 32-bit value, made of two surrogate values, i.e.. if    *****/
  /*****     `char_code = (char_hi << 16) | char_lo', then both        *****/
  /*****     `char_hi' and `char_lo' must be in the Surrogates Area.   *****/
  /*****      Area.                                                    *****/
  /*****                                                               *****/
  /***** The `is32' table embedded in the charmap indicates whether a  *****/
  /***** given 16-bit value is in the surrogates area or not.          *****/
  /*****                                                               *****/
  /***** So, for any given `char_code', we can assert the following.   *****/
  /*****                                                               *****/
  /*****   If `char_hi == 0' then we must have `is32[char_lo] == 0'.   *****/
  /*****                                                               *****/
  /*****   If `char_hi != 0' then we must have both                    *****/
  /*****   `is32[char_hi] != 0' and `is32[char_lo] != 0'.              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET         TYPE        DESCRIPTION
   *
   *   format      0              USHORT      must be 8
   *   reserved    2              USHORT      reserved
   *   length      4              ULONG       length in bytes
   *   language    8              ULONG       Mac language code
   *   is32        12             BYTE[8192]  32-bitness bitmap
   *   count       8204           ULONG       number of groups
   *
   * This header is followed by `count' groups of the following format:
   *
   *   start       0              ULONG       first charcode
   *   end         4              ULONG       last charcode
   *   startId     8              ULONG       start glyph ID for the group
   */

#ifdef TT_CONFIG_CMAP_FORMAT_8

  FT_CALLBACK_DEF( FT_Error )
  tt_cmap8_validate( FT_Byte*      table,
                     FT_Validator  valid )
  {
    FT_Byte*   p = table + 4;
    FT_Byte*   is32;
    FT_UInt32  length;
    FT_UInt32  num_groups;


    if ( table + 16 + 8192 > valid->limit )
      FT_INVALID_TOO_SHORT;

    length = TT_NEXT_ULONG( p );
    if ( length > (FT_UInt32)( valid->limit - table ) || length < 8192 + 16 )
      FT_INVALID_TOO_SHORT;

    is32       = table + 12;
    p          = is32  + 8192;          /* skip `is32' array */
    num_groups = TT_NEXT_ULONG( p );

    /* p + num_groups * 12 > valid->limit ? */
    if ( num_groups > (FT_UInt32)( valid->limit - p ) / 12 )
      FT_INVALID_TOO_SHORT;

    /* check groups, they must be in increasing order */
    {
      FT_UInt32  n, start, end, start_id, count, last = 0;


      for ( n = 0; n < num_groups; n++ )
      {
        FT_UInt   hi, lo;


        start    = TT_NEXT_ULONG( p );
        end      = TT_NEXT_ULONG( p );
        start_id = TT_NEXT_ULONG( p );

        if ( start > end )
          FT_INVALID_DATA;

        if ( n > 0 && start <= last )
          FT_INVALID_DATA;

        if ( valid->level >= FT_VALIDATE_TIGHT )
        {
          FT_UInt32  d = end - start;


          /* start_id + end - start >= TT_VALID_GLYPH_COUNT( valid ) ? */
          if ( d > TT_VALID_GLYPH_COUNT( valid )             ||
               start_id >= TT_VALID_GLYPH_COUNT( valid ) - d )
            FT_INVALID_GLYPH_ID;

          count = (FT_UInt32)( end - start + 1 );

          if ( start & ~0xFFFFU )
          {
            /* start_hi != 0; check that is32[i] is 1 for each i in */
            /* the `hi' and `lo' of the range [start..end]          */
            for ( ; count > 0; count--, start++ )
            {
              hi = (FT_UInt)( start >> 16 );
              lo = (FT_UInt)( start & 0xFFFFU );

              if ( (is32[hi >> 3] & ( 0x80 >> ( hi & 7 ) ) ) == 0 )
                FT_INVALID_DATA;

              if ( (is32[lo >> 3] & ( 0x80 >> ( lo & 7 ) ) ) == 0 )
                FT_INVALID_DATA;
            }
          }
          else
          {
            /* start_hi == 0; check that is32[i] is 0 for each i in */
            /* the range [start..end]                               */

            /* end_hi cannot be != 0! */
            if ( end & ~0xFFFFU )
              FT_INVALID_DATA;

            for ( ; count > 0; count--, start++ )
            {
              lo = (FT_UInt)( start & 0xFFFFU );

              if ( (is32[lo >> 3] & ( 0x80 >> ( lo & 7 ) ) ) != 0 )
                FT_INVALID_DATA;
            }
          }
        }

        last = end;
      }
    }

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap8_char_index( TT_CMap    cmap,
                       FT_UInt32  char_code )
  {
    FT_Byte*   table      = cmap->data;
    FT_UInt    result     = 0;
    FT_Byte*   p          = table + 8204;
    FT_UInt32  num_groups = TT_NEXT_ULONG( p );
    FT_UInt32  start, end, start_id;


    for ( ; num_groups > 0; num_groups-- )
    {
      start    = TT_NEXT_ULONG( p );
      end      = TT_NEXT_ULONG( p );
      start_id = TT_NEXT_ULONG( p );

      if ( char_code < start )
        break;

      if ( char_code <= end )
      {
        if ( start_id > 0xFFFFFFFFUL - ( char_code - start ) )
          return 0;

        result = (FT_UInt)( start_id + ( char_code - start ) );
        break;
      }
    }
    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap8_char_next( TT_CMap     cmap,
                      FT_UInt32  *pchar_code )
  {
    FT_Face    face       = cmap->cmap.charmap.face;
    FT_UInt32  result     = 0;
    FT_UInt32  char_code;
    FT_UInt    gindex     = 0;
    FT_Byte*   table      = cmap->data;
    FT_Byte*   p          = table + 8204;
    FT_UInt32  num_groups = TT_NEXT_ULONG( p );
    FT_UInt32  start, end, start_id;


    if ( *pchar_code >= 0xFFFFFFFFUL )
      return 0;

    char_code = *pchar_code + 1;

    p = table + 8208;

    for ( ; num_groups > 0; num_groups-- )
    {
      start    = TT_NEXT_ULONG( p );
      end      = TT_NEXT_ULONG( p );
      start_id = TT_NEXT_ULONG( p );

      if ( char_code < start )
        char_code = start;

    Again:
      if ( char_code <= end )
      {
        /* ignore invalid group */
        if ( start_id > 0xFFFFFFFFUL - ( char_code - start ) )
          continue;

        gindex = (FT_UInt)( start_id + ( char_code - start ) );

        /* does first element of group point to `.notdef' glyph? */
        if ( gindex == 0 )
        {
          if ( char_code >= 0xFFFFFFFFUL )
            break;

          char_code++;
          goto Again;
        }

        /* if `gindex' is invalid, the remaining values */
        /* in this group are invalid, too               */
        if ( gindex >= (FT_UInt)face->num_glyphs )
        {
          gindex = 0;
          continue;
        }

        result = char_code;
        break;
      }
    }

    *pchar_code = result;
    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap8_get_info( TT_CMap       cmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 8;


    cmap_info->format   = 8;
    cmap_info->language = (FT_ULong)TT_PEEK_ULONG( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap8_class_rec,

      sizeof ( TT_CMapRec ),

      (FT_CMap_InitFunc)     tt_cmap_init,         /* init       */
      (FT_CMap_DoneFunc)     NULL,                 /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap8_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap8_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    8,
    (TT_CMap_ValidateFunc)tt_cmap8_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap8_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_8 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 10                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME      OFFSET  TYPE               DESCRIPTION
   *
   *   format     0      USHORT             must be 10
   *   reserved   2      USHORT             reserved
   *   length     4      ULONG              length in bytes
   *   language   8      ULONG              Mac language code
   *
   *   start     12      ULONG              first char in range
   *   count     16      ULONG              number of chars in range
   *   glyphIds  20      USHORT[count]      glyph indices covered
   */

#ifdef TT_CONFIG_CMAP_FORMAT_10

  FT_CALLBACK_DEF( FT_Error )
  tt_cmap10_validate( FT_Byte*      table,
                      FT_Validator  valid )
  {
    FT_Byte*  p = table + 4;
    FT_ULong  length, count;


    if ( table + 20 > valid->limit )
      FT_INVALID_TOO_SHORT;

    length = TT_NEXT_ULONG( p );
    p      = table + 16;
    count  = TT_NEXT_ULONG( p );

    if ( length > (FT_ULong)( valid->limit - table ) ||
         /* length < 20 + count * 2 ? */
         length < 20                                 ||
         ( length - 20 ) / 2 < count                 )
      FT_INVALID_TOO_SHORT;

    /* check glyph indices */
    if ( valid->level >= FT_VALIDATE_TIGHT )
    {
      FT_UInt  gindex;


      for ( ; count > 0; count-- )
      {
        gindex = TT_NEXT_USHORT( p );
        if ( gindex >= TT_VALID_GLYPH_COUNT( valid ) )
          FT_INVALID_GLYPH_ID;
      }
    }

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap10_char_index( TT_CMap    cmap,
                        FT_UInt32  char_code )
  {
    FT_Byte*   table  = cmap->data;
    FT_UInt    result = 0;
    FT_Byte*   p      = table + 12;
    FT_UInt32  start  = TT_NEXT_ULONG( p );
    FT_UInt32  count  = TT_NEXT_ULONG( p );
    FT_UInt32  idx;


    if ( char_code < start )
      return 0;

    idx = char_code - start;

    if ( idx < count )
    {
      p     += 2 * idx;
      result = TT_PEEK_USHORT( p );
    }

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap10_char_next( TT_CMap     cmap,
                       FT_UInt32  *pchar_code )
  {
    FT_Byte*   table     = cmap->data;
    FT_UInt32  char_code;
    FT_UInt    gindex    = 0;
    FT_Byte*   p         = table + 12;
    FT_UInt32  start     = TT_NEXT_ULONG( p );
    FT_UInt32  count     = TT_NEXT_ULONG( p );
    FT_UInt32  idx;


    if ( *pchar_code >= 0xFFFFFFFFUL )
      return 0;

    char_code = *pchar_code + 1;

    if ( char_code < start )
      char_code = start;

    idx = char_code - start;
    p  += 2 * idx;

    for ( ; idx < count; idx++ )
    {
      gindex = TT_NEXT_USHORT( p );
      if ( gindex != 0 )
        break;

      if ( char_code >= 0xFFFFFFFFUL )
        return 0;

      char_code++;
    }

    *pchar_code = char_code;
    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap10_get_info( TT_CMap       cmap,
                      TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 8;


    cmap_info->format   = 10;
    cmap_info->language = (FT_ULong)TT_PEEK_ULONG( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap10_class_rec,

      sizeof ( TT_CMapRec ),

      (FT_CMap_InitFunc)     tt_cmap_init,          /* init       */
      (FT_CMap_DoneFunc)     NULL,                  /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap10_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap10_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    10,
    (TT_CMap_ValidateFunc)tt_cmap10_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap10_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_10 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 12                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET     TYPE       DESCRIPTION
   *
   *   format      0          USHORT     must be 12
   *   reserved    2          USHORT     reserved
   *   length      4          ULONG      length in bytes
   *   language    8          ULONG      Mac language code
   *   count       12         ULONG      number of groups
   *               16
   *
   * This header is followed by `count' groups of the following format:
   *
   *   start       0          ULONG      first charcode
   *   end         4          ULONG      last charcode
   *   startId     8          ULONG      start glyph ID for the group
   */

#ifdef TT_CONFIG_CMAP_FORMAT_12

  typedef struct  TT_CMap12Rec_
  {
    TT_CMapRec  cmap;
    FT_Bool     valid;
    FT_ULong    cur_charcode;
    FT_UInt     cur_gindex;
    FT_ULong    cur_group;
    FT_ULong    num_groups;

  } TT_CMap12Rec, *TT_CMap12;


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap12_init( TT_CMap12  cmap,
                  FT_Byte*   table )
  {
    cmap->cmap.data  = table;

    table           += 12;
    cmap->num_groups = FT_PEEK_ULONG( table );

    cmap->valid      = 0;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap12_validate( FT_Byte*      table,
                      FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_ULong  length;
    FT_ULong  num_groups;


    if ( table + 16 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 4;
    length = TT_NEXT_ULONG( p );

    p          = table + 12;
    num_groups = TT_NEXT_ULONG( p );

    if ( length > (FT_ULong)( valid->limit - table ) ||
         /* length < 16 + 12 * num_groups ? */
         length < 16                                 ||
         ( length - 16 ) / 12 < num_groups           )
      FT_INVALID_TOO_SHORT;

    /* check groups, they must be in increasing order */
    {
      FT_ULong  n, start, end, start_id, last = 0;


      for ( n = 0; n < num_groups; n++ )
      {
        start    = TT_NEXT_ULONG( p );
        end      = TT_NEXT_ULONG( p );
        start_id = TT_NEXT_ULONG( p );

        if ( start > end )
          FT_INVALID_DATA;

        if ( n > 0 && start <= last )
          FT_INVALID_DATA;

        if ( valid->level >= FT_VALIDATE_TIGHT )
        {
          FT_UInt32  d = end - start;


          /* start_id + end - start >= TT_VALID_GLYPH_COUNT( valid ) ? */
          if ( d > TT_VALID_GLYPH_COUNT( valid )             ||
               start_id >= TT_VALID_GLYPH_COUNT( valid ) - d )
            FT_INVALID_GLYPH_ID;
        }

        last = end;
      }
    }

    return FT_Err_Ok;
  }


  /* search the index of the charcode next to cmap->cur_charcode */
  /* cmap->cur_group should be set up properly by caller         */
  /*                                                             */
  static void
  tt_cmap12_next( TT_CMap12  cmap )
  {
    FT_Face   face = cmap->cmap.cmap.charmap.face;
    FT_Byte*  p;
    FT_ULong  start, end, start_id, char_code;
    FT_ULong  n;
    FT_UInt   gindex;


    if ( cmap->cur_charcode >= 0xFFFFFFFFUL )
      goto Fail;

    char_code = cmap->cur_charcode + 1;

    for ( n = cmap->cur_group; n < cmap->num_groups; n++ )
    {
      p        = cmap->cmap.data + 16 + 12 * n;
      start    = TT_NEXT_ULONG( p );
      end      = TT_NEXT_ULONG( p );
      start_id = TT_PEEK_ULONG( p );

      if ( char_code < start )
        char_code = start;

    Again:
      if ( char_code <= end )
      {
        /* ignore invalid group */
        if ( start_id > 0xFFFFFFFFUL - ( char_code - start ) )
          continue;

        gindex = (FT_UInt)( start_id + ( char_code - start ) );

        /* does first element of group point to `.notdef' glyph? */
        if ( gindex == 0 )
        {
          if ( char_code >= 0xFFFFFFFFUL )
            goto Fail;

          char_code++;
          goto Again;
        }

        /* if `gindex' is invalid, the remaining values */
        /* in this group are invalid, too               */
        if ( gindex >= (FT_UInt)face->num_glyphs )
        {
          gindex = 0;
          continue;
        }

        cmap->cur_charcode = char_code;
        cmap->cur_gindex   = gindex;
        cmap->cur_group    = n;

        return;
      }
    }

  Fail:
    cmap->valid = 0;
  }


  static FT_UInt
  tt_cmap12_char_map_binary( TT_CMap     cmap,
                             FT_UInt32*  pchar_code,
                             FT_Bool     next )
  {
    FT_UInt    gindex     = 0;
    FT_Byte*   p          = cmap->data + 12;
    FT_UInt32  num_groups = TT_PEEK_ULONG( p );
    FT_UInt32  char_code  = *pchar_code;
    FT_UInt32  start, end, start_id;
    FT_UInt32  max, min, mid;


    if ( !num_groups )
      return 0;

    /* make compiler happy */
    mid = num_groups;
    end = 0xFFFFFFFFUL;

    if ( next )
    {
      if ( char_code >= 0xFFFFFFFFUL )
        return 0;

      char_code++;
    }

    min = 0;
    max = num_groups;

    /* binary search */
    while ( min < max )
    {
      mid = ( min + max ) >> 1;
      p   = cmap->data + 16 + 12 * mid;

      start = TT_NEXT_ULONG( p );
      end   = TT_NEXT_ULONG( p );

      if ( char_code < start )
        max = mid;
      else if ( char_code > end )
        min = mid + 1;
      else
      {
        start_id = TT_PEEK_ULONG( p );

        /* reject invalid glyph index */
        if ( start_id > 0xFFFFFFFFUL - ( char_code - start ) )
          gindex = 0;
        else
          gindex = (FT_UInt)( start_id + ( char_code - start ) );
        break;
      }
    }

    if ( next )
    {
      FT_Face    face   = cmap->cmap.charmap.face;
      TT_CMap12  cmap12 = (TT_CMap12)cmap;


      /* if `char_code' is not in any group, then `mid' is */
      /* the group nearest to `char_code'                  */

      if ( char_code > end )
      {
        mid++;
        if ( mid == num_groups )
          return 0;
      }

      cmap12->valid        = 1;
      cmap12->cur_charcode = char_code;
      cmap12->cur_group    = mid;

      if ( gindex >= (FT_UInt)face->num_glyphs )
        gindex = 0;

      if ( !gindex )
      {
        tt_cmap12_next( cmap12 );

        if ( cmap12->valid )
          gindex = cmap12->cur_gindex;
      }
      else
        cmap12->cur_gindex = gindex;

      *pchar_code = cmap12->cur_charcode;
    }

    return gindex;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap12_char_index( TT_CMap    cmap,
                        FT_UInt32  char_code )
  {
    return tt_cmap12_char_map_binary( cmap, &char_code, 0 );
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap12_char_next( TT_CMap     cmap,
                       FT_UInt32  *pchar_code )
  {
    TT_CMap12  cmap12 = (TT_CMap12)cmap;
    FT_UInt    gindex;


    /* no need to search */
    if ( cmap12->valid && cmap12->cur_charcode == *pchar_code )
    {
      tt_cmap12_next( cmap12 );
      if ( cmap12->valid )
      {
        gindex      = cmap12->cur_gindex;
        *pchar_code = (FT_UInt32)cmap12->cur_charcode;
      }
      else
        gindex = 0;
    }
    else
      gindex = tt_cmap12_char_map_binary( cmap, pchar_code, 1 );

    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap12_get_info( TT_CMap       cmap,
                      TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 8;


    cmap_info->format   = 12;
    cmap_info->language = (FT_ULong)TT_PEEK_ULONG( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap12_class_rec,

      sizeof ( TT_CMap12Rec ),

      (FT_CMap_InitFunc)     tt_cmap12_init,        /* init       */
      (FT_CMap_DoneFunc)     NULL,                  /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap12_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap12_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    12,
    (TT_CMap_ValidateFunc)tt_cmap12_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap12_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_12 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          FORMAT 13                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME        OFFSET     TYPE       DESCRIPTION
   *
   *   format      0          USHORT     must be 13
   *   reserved    2          USHORT     reserved
   *   length      4          ULONG      length in bytes
   *   language    8          ULONG      Mac language code
   *   count       12         ULONG      number of groups
   *               16
   *
   * This header is followed by `count' groups of the following format:
   *
   *   start       0          ULONG      first charcode
   *   end         4          ULONG      last charcode
   *   glyphId     8          ULONG      glyph ID for the whole group
   */

#ifdef TT_CONFIG_CMAP_FORMAT_13

  typedef struct  TT_CMap13Rec_
  {
    TT_CMapRec  cmap;
    FT_Bool     valid;
    FT_ULong    cur_charcode;
    FT_UInt     cur_gindex;
    FT_ULong    cur_group;
    FT_ULong    num_groups;

  } TT_CMap13Rec, *TT_CMap13;


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap13_init( TT_CMap13  cmap,
                  FT_Byte*   table )
  {
    cmap->cmap.data  = table;

    table           += 12;
    cmap->num_groups = FT_PEEK_ULONG( table );

    cmap->valid      = 0;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap13_validate( FT_Byte*      table,
                      FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_ULong  length;
    FT_ULong  num_groups;


    if ( table + 16 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p      = table + 4;
    length = TT_NEXT_ULONG( p );

    p          = table + 12;
    num_groups = TT_NEXT_ULONG( p );

    if ( length > (FT_ULong)( valid->limit - table ) ||
         /* length < 16 + 12 * num_groups ? */
         length < 16                                 ||
         ( length - 16 ) / 12 < num_groups           )
      FT_INVALID_TOO_SHORT;

    /* check groups, they must be in increasing order */
    {
      FT_ULong  n, start, end, glyph_id, last = 0;


      for ( n = 0; n < num_groups; n++ )
      {
        start    = TT_NEXT_ULONG( p );
        end      = TT_NEXT_ULONG( p );
        glyph_id = TT_NEXT_ULONG( p );

        if ( start > end )
          FT_INVALID_DATA;

        if ( n > 0 && start <= last )
          FT_INVALID_DATA;

        if ( valid->level >= FT_VALIDATE_TIGHT )
        {
          if ( glyph_id >= TT_VALID_GLYPH_COUNT( valid ) )
            FT_INVALID_GLYPH_ID;
        }

        last = end;
      }
    }

    return FT_Err_Ok;
  }


  /* search the index of the charcode next to cmap->cur_charcode */
  /* cmap->cur_group should be set up properly by caller         */
  /*                                                             */
  static void
  tt_cmap13_next( TT_CMap13  cmap )
  {
    FT_Face   face = cmap->cmap.cmap.charmap.face;
    FT_Byte*  p;
    FT_ULong  start, end, glyph_id, char_code;
    FT_ULong  n;
    FT_UInt   gindex;


    if ( cmap->cur_charcode >= 0xFFFFFFFFUL )
      goto Fail;

    char_code = cmap->cur_charcode + 1;

    for ( n = cmap->cur_group; n < cmap->num_groups; n++ )
    {
      p        = cmap->cmap.data + 16 + 12 * n;
      start    = TT_NEXT_ULONG( p );
      end      = TT_NEXT_ULONG( p );
      glyph_id = TT_PEEK_ULONG( p );

      if ( char_code < start )
        char_code = start;

      if ( char_code <= end )
      {
        gindex = (FT_UInt)glyph_id;

        if ( gindex && gindex < (FT_UInt)face->num_glyphs )
        {
          cmap->cur_charcode = char_code;
          cmap->cur_gindex   = gindex;
          cmap->cur_group    = n;

          return;
        }
      }
    }

  Fail:
    cmap->valid = 0;
  }


  static FT_UInt
  tt_cmap13_char_map_binary( TT_CMap     cmap,
                             FT_UInt32*  pchar_code,
                             FT_Bool     next )
  {
    FT_UInt    gindex     = 0;
    FT_Byte*   p          = cmap->data + 12;
    FT_UInt32  num_groups = TT_PEEK_ULONG( p );
    FT_UInt32  char_code  = *pchar_code;
    FT_UInt32  start, end;
    FT_UInt32  max, min, mid;


    if ( !num_groups )
      return 0;

    /* make compiler happy */
    mid = num_groups;
    end = 0xFFFFFFFFUL;

    if ( next )
    {
      if ( char_code >= 0xFFFFFFFFUL )
        return 0;

      char_code++;
    }

    min = 0;
    max = num_groups;

    /* binary search */
    while ( min < max )
    {
      mid = ( min + max ) >> 1;
      p   = cmap->data + 16 + 12 * mid;

      start = TT_NEXT_ULONG( p );
      end   = TT_NEXT_ULONG( p );

      if ( char_code < start )
        max = mid;
      else if ( char_code > end )
        min = mid + 1;
      else
      {
        gindex = (FT_UInt)TT_PEEK_ULONG( p );

        break;
      }
    }

    if ( next )
    {
      FT_Face    face   = cmap->cmap.charmap.face;
      TT_CMap13  cmap13 = (TT_CMap13)cmap;


      /* if `char_code' is not in any group, then `mid' is */
      /* the group nearest to `char_code'                  */

      if ( char_code > end )
      {
        mid++;
        if ( mid == num_groups )
          return 0;
      }

      cmap13->valid        = 1;
      cmap13->cur_charcode = char_code;
      cmap13->cur_group    = mid;

      if ( gindex >= (FT_UInt)face->num_glyphs )
        gindex = 0;

      if ( !gindex )
      {
        tt_cmap13_next( cmap13 );

        if ( cmap13->valid )
          gindex = cmap13->cur_gindex;
      }
      else
        cmap13->cur_gindex = gindex;

      *pchar_code = cmap13->cur_charcode;
    }

    return gindex;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap13_char_index( TT_CMap    cmap,
                        FT_UInt32  char_code )
  {
    return tt_cmap13_char_map_binary( cmap, &char_code, 0 );
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap13_char_next( TT_CMap     cmap,
                       FT_UInt32  *pchar_code )
  {
    TT_CMap13  cmap13 = (TT_CMap13)cmap;
    FT_UInt    gindex;


    /* no need to search */
    if ( cmap13->valid && cmap13->cur_charcode == *pchar_code )
    {
      tt_cmap13_next( cmap13 );
      if ( cmap13->valid )
      {
        gindex      = cmap13->cur_gindex;
        *pchar_code = cmap13->cur_charcode;
      }
      else
        gindex = 0;
    }
    else
      gindex = tt_cmap13_char_map_binary( cmap, pchar_code, 1 );

    return gindex;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap13_get_info( TT_CMap       cmap,
                      TT_CMapInfo  *cmap_info )
  {
    FT_Byte*  p = cmap->data + 8;


    cmap_info->format   = 13;
    cmap_info->language = (FT_ULong)TT_PEEK_ULONG( p );

    return FT_Err_Ok;
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap13_class_rec,

      sizeof ( TT_CMap13Rec ),

      (FT_CMap_InitFunc)     tt_cmap13_init,        /* init       */
      (FT_CMap_DoneFunc)     NULL,                  /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap13_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap13_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    13,
    (TT_CMap_ValidateFunc)tt_cmap13_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap13_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_13 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           FORMAT 14                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * TABLE OVERVIEW
   * --------------
   *
   *   NAME         OFFSET  TYPE    DESCRIPTION
   *
   *   format         0     USHORT  must be 14
   *   length         2     ULONG   table length in bytes
   *   numSelector    6     ULONG   number of variation sel. records
   *
   * Followed by numSelector records, each of which looks like
   *
   *   varSelector    0     UINT24  Unicode codepoint of sel.
   *   defaultOff     3     ULONG   offset to a default UVS table
   *                                describing any variants to be found in
   *                                the normal Unicode subtable.
   *   nonDefOff      7     ULONG   offset to a non-default UVS table
   *                                describing any variants not in the
   *                                standard cmap, with GIDs here
   * (either offset may be 0 NULL)
   *
   * Selectors are sorted by code point.
   *
   * A default Unicode Variation Selector (UVS) subtable is just a list of
   * ranges of code points which are to be found in the standard cmap.  No
   * glyph IDs (GIDs) here.
   *
   *   numRanges      0     ULONG   number of ranges following
   *
   * A range looks like
   *
   *   uniStart       0     UINT24  code point of the first character in
   *                                this range
   *   additionalCnt  3     UBYTE   count of additional characters in this
   *                                range (zero means a range of a single
   *                                character)
   *
   * Ranges are sorted by `uniStart'.
   *
   * A non-default Unicode Variation Selector (UVS) subtable is a list of
   * mappings from codepoint to GID.
   *
   *   numMappings    0     ULONG   number of mappings
   *
   * A range looks like
   *
   *   uniStart       0     UINT24  code point of the first character in
   *                                this range
   *   GID            3     USHORT  and its GID
   *
   * Ranges are sorted by `uniStart'.
   */

#ifdef TT_CONFIG_CMAP_FORMAT_14

  typedef struct  TT_CMap14Rec_
  {
    TT_CMapRec  cmap;
    FT_ULong    num_selectors;

    /* This array is used to store the results of various
     * cmap 14 query functions.  The data is overwritten
     * on each call to these functions.
     */
    FT_UInt32   max_results;
    FT_UInt32*  results;
    FT_Memory   memory;

  } TT_CMap14Rec, *TT_CMap14;


  FT_CALLBACK_DEF( void )
  tt_cmap14_done( TT_CMap14  cmap )
  {
    FT_Memory  memory = cmap->memory;


    cmap->max_results = 0;
    if ( memory && cmap->results )
      FT_FREE( cmap->results );
  }


  static FT_Error
  tt_cmap14_ensure( TT_CMap14  cmap,
                    FT_UInt32  num_results,
                    FT_Memory  memory )
  {
    FT_UInt32  old_max = cmap->max_results;
    FT_Error   error   = FT_Err_Ok;


    if ( num_results > cmap->max_results )
    {
       cmap->memory = memory;

       if ( FT_QRENEW_ARRAY( cmap->results, old_max, num_results ) )
         return error;

       cmap->max_results = num_results;
    }

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap14_init( TT_CMap14  cmap,
                  FT_Byte*   table )
  {
    cmap->cmap.data = table;

    table               += 6;
    cmap->num_selectors  = FT_PEEK_ULONG( table );
    cmap->max_results    = 0;
    cmap->results        = NULL;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap14_validate( FT_Byte*      table,
                      FT_Validator  valid )
  {
    FT_Byte*  p;
    FT_ULong  length;
    FT_ULong  num_selectors;


    if ( table + 2 + 4 + 4 > valid->limit )
      FT_INVALID_TOO_SHORT;

    p             = table + 2;
    length        = TT_NEXT_ULONG( p );
    num_selectors = TT_NEXT_ULONG( p );

    if ( length > (FT_ULong)( valid->limit - table ) ||
         /* length < 10 + 11 * num_selectors ? */
         length < 10                                 ||
         ( length - 10 ) / 11 < num_selectors        )
      FT_INVALID_TOO_SHORT;

    /* check selectors, they must be in increasing order */
    {
      /* we start lastVarSel at 1 because a variant selector value of 0
       * isn't valid.
       */
      FT_ULong  n, lastVarSel = 1;


      for ( n = 0; n < num_selectors; n++ )
      {
        FT_ULong  varSel    = TT_NEXT_UINT24( p );
        FT_ULong  defOff    = TT_NEXT_ULONG( p );
        FT_ULong  nondefOff = TT_NEXT_ULONG( p );


        if ( defOff >= length || nondefOff >= length )
          FT_INVALID_TOO_SHORT;

        if ( varSel < lastVarSel )
          FT_INVALID_DATA;

        lastVarSel = varSel + 1;

        /* check the default table (these glyphs should be reached     */
        /* through the normal Unicode cmap, no GIDs, just check order) */
        if ( defOff != 0 )
        {
          FT_Byte*  defp     = table + defOff;
          FT_ULong  numRanges;
          FT_ULong  i;
          FT_ULong  lastBase = 0;


          if ( defp + 4 > valid->limit )
            FT_INVALID_TOO_SHORT;

          numRanges = TT_NEXT_ULONG( defp );

          /* defp + numRanges * 4 > valid->limit ? */
          if ( numRanges > (FT_ULong)( valid->limit - defp ) / 4 )
            FT_INVALID_TOO_SHORT;

          for ( i = 0; i < numRanges; i++ )
          {
            FT_ULong  base = TT_NEXT_UINT24( defp );
            FT_ULong  cnt  = FT_NEXT_BYTE( defp );


            if ( base + cnt >= 0x110000UL )              /* end of Unicode */
              FT_INVALID_DATA;

            if ( base < lastBase )
              FT_INVALID_DATA;

            lastBase = base + cnt + 1U;
          }
        }

        /* and the non-default table (these glyphs are specified here) */
        if ( nondefOff != 0 )
        {
          FT_Byte*  ndp        = table + nondefOff;
          FT_ULong  numMappings;
          FT_ULong  i, lastUni = 0;


          if ( ndp + 4 > valid->limit )
            FT_INVALID_TOO_SHORT;

          numMappings = TT_NEXT_ULONG( ndp );

          /* numMappings * 5 > (FT_ULong)( valid->limit - ndp ) ? */
          if ( numMappings > ( (FT_ULong)( valid->limit - ndp ) ) / 5 )
            FT_INVALID_TOO_SHORT;

          for ( i = 0; i < numMappings; i++ )
          {
            FT_ULong  uni = TT_NEXT_UINT24( ndp );
            FT_ULong  gid = TT_NEXT_USHORT( ndp );


            if ( uni >= 0x110000UL )                     /* end of Unicode */
              FT_INVALID_DATA;

            if ( uni < lastUni )
              FT_INVALID_DATA;

            lastUni = uni + 1U;

            if ( valid->level >= FT_VALIDATE_TIGHT    &&
                 gid >= TT_VALID_GLYPH_COUNT( valid ) )
              FT_INVALID_GLYPH_ID;
          }
        }
      }
    }

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap14_char_index( TT_CMap    cmap,
                        FT_UInt32  char_code )
  {
    FT_UNUSED( cmap );
    FT_UNUSED( char_code );

    /* This can't happen */
    return 0;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap14_char_next( TT_CMap     cmap,
                       FT_UInt32  *pchar_code )
  {
    FT_UNUSED( cmap );

    /* This can't happen */
    *pchar_code = 0;
    return 0;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap14_get_info( TT_CMap       cmap,
                      TT_CMapInfo  *cmap_info )
  {
    FT_UNUSED( cmap );

    cmap_info->format   = 14;
    /* subtable 14 does not define a language field */
    cmap_info->language = 0xFFFFFFFFUL;

    return FT_Err_Ok;
  }


  static FT_UInt
  tt_cmap14_char_map_def_binary( FT_Byte    *base,
                                 FT_UInt32   char_code )
  {
    FT_UInt32  numRanges = TT_PEEK_ULONG( base );
    FT_UInt32  max, min;


    min = 0;
    max = numRanges;

    base += 4;

    /* binary search */
    while ( min < max )
    {
      FT_UInt32  mid   = ( min + max ) >> 1;
      FT_Byte*   p     = base + 4 * mid;
      FT_ULong   start = TT_NEXT_UINT24( p );
      FT_UInt    cnt   = FT_NEXT_BYTE( p );


      if ( char_code < start )
        max = mid;
      else if ( char_code > start + cnt )
        min = mid + 1;
      else
        return TRUE;
    }

    return FALSE;
  }


  static FT_UInt
  tt_cmap14_char_map_nondef_binary( FT_Byte    *base,
                                    FT_UInt32   char_code )
  {
    FT_UInt32  numMappings = TT_PEEK_ULONG( base );
    FT_UInt32  max, min;


    min = 0;
    max = numMappings;

    base += 4;

    /* binary search */
    while ( min < max )
    {
      FT_UInt32  mid = ( min + max ) >> 1;
      FT_Byte*   p   = base + 5 * mid;
      FT_UInt32  uni = (FT_UInt32)TT_NEXT_UINT24( p );


      if ( char_code < uni )
        max = mid;
      else if ( char_code > uni )
        min = mid + 1;
      else
        return TT_PEEK_USHORT( p );
    }

    return 0;
  }


  static FT_Byte*
  tt_cmap14_find_variant( FT_Byte    *base,
                          FT_UInt32   variantCode )
  {
    FT_UInt32  numVar = TT_PEEK_ULONG( base );
    FT_UInt32  max, min;


    min = 0;
    max = numVar;

    base += 4;

    /* binary search */
    while ( min < max )
    {
      FT_UInt32  mid    = ( min + max ) >> 1;
      FT_Byte*   p      = base + 11 * mid;
      FT_ULong   varSel = TT_NEXT_UINT24( p );


      if ( variantCode < varSel )
        max = mid;
      else if ( variantCode > varSel )
        min = mid + 1;
      else
        return p;
    }

    return NULL;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap14_char_var_index( TT_CMap    cmap,
                            TT_CMap    ucmap,
                            FT_UInt32  charcode,
                            FT_UInt32  variantSelector )
  {
    FT_Byte*  p = tt_cmap14_find_variant( cmap->data + 6, variantSelector );
    FT_ULong  defOff;
    FT_ULong  nondefOff;


    if ( !p )
      return 0;

    defOff    = TT_NEXT_ULONG( p );
    nondefOff = TT_PEEK_ULONG( p );

    if ( defOff != 0                                                    &&
         tt_cmap14_char_map_def_binary( cmap->data + defOff, charcode ) )
    {
      /* This is the default variant of this charcode.  GID not stored */
      /* here; stored in the normal Unicode charmap instead.           */
      return ucmap->cmap.clazz->char_index( &ucmap->cmap, charcode );
    }

    if ( nondefOff != 0 )
      return tt_cmap14_char_map_nondef_binary( cmap->data + nondefOff,
                                               charcode );

    return 0;
  }


  FT_CALLBACK_DEF( FT_Int )
  tt_cmap14_char_var_isdefault( TT_CMap    cmap,
                                FT_UInt32  charcode,
                                FT_UInt32  variantSelector )
  {
    FT_Byte*  p = tt_cmap14_find_variant( cmap->data + 6, variantSelector );
    FT_ULong  defOff;
    FT_ULong  nondefOff;


    if ( !p )
      return -1;

    defOff    = TT_NEXT_ULONG( p );
    nondefOff = TT_NEXT_ULONG( p );

    if ( defOff != 0                                                    &&
         tt_cmap14_char_map_def_binary( cmap->data + defOff, charcode ) )
      return 1;

    if ( nondefOff != 0                                            &&
         tt_cmap14_char_map_nondef_binary( cmap->data + nondefOff,
                                           charcode ) != 0         )
      return 0;

    return -1;
  }


  FT_CALLBACK_DEF( FT_UInt32* )
  tt_cmap14_variants( TT_CMap    cmap,
                      FT_Memory  memory )
  {
    TT_CMap14   cmap14 = (TT_CMap14)cmap;
    FT_UInt32   count  = cmap14->num_selectors;
    FT_Byte*    p      = cmap->data + 10;
    FT_UInt32*  result;
    FT_UInt32   i;


    if ( tt_cmap14_ensure( cmap14, ( count + 1 ), memory ) )
      return NULL;

    result = cmap14->results;
    for ( i = 0; i < count; i++ )
    {
      result[i] = (FT_UInt32)TT_NEXT_UINT24( p );
      p        += 8;
    }
    result[i] = 0;

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 * )
  tt_cmap14_char_variants( TT_CMap    cmap,
                           FT_Memory  memory,
                           FT_UInt32  charCode )
  {
    TT_CMap14   cmap14 = (TT_CMap14)  cmap;
    FT_UInt32   count  = cmap14->num_selectors;
    FT_Byte*    p      = cmap->data + 10;
    FT_UInt32*  q;


    if ( tt_cmap14_ensure( cmap14, ( count + 1 ), memory ) )
      return NULL;

    for ( q = cmap14->results; count > 0; count-- )
    {
      FT_UInt32  varSel    = TT_NEXT_UINT24( p );
      FT_ULong   defOff    = TT_NEXT_ULONG( p );
      FT_ULong   nondefOff = TT_NEXT_ULONG( p );


      if ( ( defOff != 0                                               &&
             tt_cmap14_char_map_def_binary( cmap->data + defOff,
                                            charCode )                 ) ||
           ( nondefOff != 0                                            &&
             tt_cmap14_char_map_nondef_binary( cmap->data + nondefOff,
                                               charCode ) != 0         ) )
      {
        q[0] = varSel;
        q++;
      }
    }
    q[0] = 0;

    return cmap14->results;
  }


  static FT_UInt
  tt_cmap14_def_char_count( FT_Byte  *p )
  {
    FT_UInt32  numRanges = (FT_UInt32)TT_NEXT_ULONG( p );
    FT_UInt    tot       = 0;


    p += 3;  /* point to the first `cnt' field */
    for ( ; numRanges > 0; numRanges-- )
    {
      tot += 1 + p[0];
      p   += 4;
    }

    return tot;
  }


  static FT_UInt32*
  tt_cmap14_get_def_chars( TT_CMap    cmap,
                           FT_Byte*   p,
                           FT_Memory  memory )
  {
    TT_CMap14   cmap14 = (TT_CMap14) cmap;
    FT_UInt32   numRanges;
    FT_UInt     cnt;
    FT_UInt32*  q;


    cnt       = tt_cmap14_def_char_count( p );
    numRanges = (FT_UInt32)TT_NEXT_ULONG( p );

    if ( tt_cmap14_ensure( cmap14, ( cnt + 1 ), memory ) )
      return NULL;

    for ( q = cmap14->results; numRanges > 0; numRanges-- )
    {
      FT_UInt32  uni = (FT_UInt32)TT_NEXT_UINT24( p );


      cnt = FT_NEXT_BYTE( p ) + 1;
      do
      {
        q[0]  = uni;
        uni  += 1;
        q    += 1;

      } while ( --cnt != 0 );
    }
    q[0] = 0;

    return cmap14->results;
  }


  static FT_UInt32*
  tt_cmap14_get_nondef_chars( TT_CMap     cmap,
                              FT_Byte    *p,
                              FT_Memory   memory )
  {
    TT_CMap14   cmap14 = (TT_CMap14) cmap;
    FT_UInt32   numMappings;
    FT_UInt     i;
    FT_UInt32  *ret;


    numMappings = (FT_UInt32)TT_NEXT_ULONG( p );

    if ( tt_cmap14_ensure( cmap14, ( numMappings + 1 ), memory ) )
      return NULL;

    ret = cmap14->results;
    for ( i = 0; i < numMappings; i++ )
    {
      ret[i] = (FT_UInt32)TT_NEXT_UINT24( p );
      p += 2;
    }
    ret[i] = 0;

    return ret;
  }


  FT_CALLBACK_DEF( FT_UInt32 * )
  tt_cmap14_variant_chars( TT_CMap    cmap,
                           FT_Memory  memory,
                           FT_UInt32  variantSelector )
  {
    FT_Byte    *p  = tt_cmap14_find_variant( cmap->data + 6,
                                             variantSelector );
    FT_Int      i;
    FT_ULong    defOff;
    FT_ULong    nondefOff;


    if ( !p )
      return NULL;

    defOff    = TT_NEXT_ULONG( p );
    nondefOff = TT_NEXT_ULONG( p );

    if ( defOff == 0 && nondefOff == 0 )
      return NULL;

    if ( defOff == 0 )
      return tt_cmap14_get_nondef_chars( cmap, cmap->data + nondefOff,
                                         memory );
    else if ( nondefOff == 0 )
      return tt_cmap14_get_def_chars( cmap, cmap->data + defOff,
                                      memory );
    else
    {
      /* Both a default and a non-default glyph set?  That's probably not */
      /* good font design, but the spec allows for it...                  */
      TT_CMap14  cmap14 = (TT_CMap14) cmap;
      FT_UInt32  numRanges;
      FT_UInt32  numMappings;
      FT_UInt32  duni;
      FT_UInt32  dcnt;
      FT_UInt32  nuni;
      FT_Byte*   dp;
      FT_UInt    di, ni, k;

      FT_UInt32  *ret;


      p  = cmap->data + nondefOff;
      dp = cmap->data + defOff;

      numMappings = (FT_UInt32)TT_NEXT_ULONG( p );
      dcnt        = tt_cmap14_def_char_count( dp );
      numRanges   = (FT_UInt32)TT_NEXT_ULONG( dp );

      if ( numMappings == 0 )
        return tt_cmap14_get_def_chars( cmap, cmap->data + defOff,
                                        memory );
      if ( dcnt == 0 )
        return tt_cmap14_get_nondef_chars( cmap, cmap->data + nondefOff,
                                           memory );

      if ( tt_cmap14_ensure( cmap14, ( dcnt + numMappings + 1 ), memory ) )
        return NULL;

      ret  = cmap14->results;
      duni = (FT_UInt32)TT_NEXT_UINT24( dp );
      dcnt = FT_NEXT_BYTE( dp );
      di   = 1;
      nuni = (FT_UInt32)TT_NEXT_UINT24( p );
      p   += 2;
      ni   = 1;
      i    = 0;

      for (;;)
      {
        if ( nuni > duni + dcnt )
        {
          for ( k = 0; k <= dcnt; k++ )
            ret[i++] = duni + k;

          di++;

          if ( di > numRanges )
            break;

          duni = (FT_UInt32)TT_NEXT_UINT24( dp );
          dcnt = FT_NEXT_BYTE( dp );
        }
        else
        {
          if ( nuni < duni )
            ret[i++] = nuni;
          /* If it is within the default range then ignore it -- */
          /* that should not have happened                       */
          ni++;
          if ( ni > numMappings )
            break;

          nuni = (FT_UInt32)TT_NEXT_UINT24( p );
          p += 2;
        }
      }

      if ( ni <= numMappings )
      {
        /* If we get here then we have run out of all default ranges.   */
        /* We have read one non-default mapping which we haven't stored */
        /* and there may be others that need to be read.                */
        ret[i++] = nuni;
        while ( ni < numMappings )
        {
          ret[i++] = (FT_UInt32)TT_NEXT_UINT24( p );
          p += 2;
          ni++;
        }
      }
      else if ( di <= numRanges )
      {
        /* If we get here then we have run out of all non-default     */
        /* mappings.  We have read one default range which we haven't */
        /* stored and there may be others that need to be read.       */
        for ( k = 0; k <= dcnt; k++ )
          ret[i++] = duni + k;

        while ( di < numRanges )
        {
          duni = (FT_UInt32)TT_NEXT_UINT24( dp );
          dcnt = FT_NEXT_BYTE( dp );

          for ( k = 0; k <= dcnt; k++ )
            ret[i++] = duni + k;
          di++;
        }
      }

      ret[i] = 0;

      return ret;
    }
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap14_class_rec,

      sizeof ( TT_CMap14Rec ),

      (FT_CMap_InitFunc)     tt_cmap14_init,        /* init       */
      (FT_CMap_DoneFunc)     tt_cmap14_done,        /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap14_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap14_char_next,   /* char_next  */

      /* Format 14 extension functions */
      (FT_CMap_CharVarIndexFunc)    tt_cmap14_char_var_index,
      (FT_CMap_CharVarIsDefaultFunc)tt_cmap14_char_var_isdefault,
      (FT_CMap_VariantListFunc)     tt_cmap14_variants,
      (FT_CMap_CharVariantListFunc) tt_cmap14_char_variants,
      (FT_CMap_VariantCharListFunc) tt_cmap14_variant_chars,

    14,
    (TT_CMap_ValidateFunc)tt_cmap14_validate,  /* validate      */
    (TT_CMap_Info_GetFunc)tt_cmap14_get_info   /* get_cmap_info */
  )

#endif /* TT_CONFIG_CMAP_FORMAT_14 */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       SYNTHETIC UNICODE                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*        This charmap is generated using postscript glyph names.        */

#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES

  FT_CALLBACK_DEF( const char * )
  tt_get_glyph_name( TT_Face  face,
                     FT_UInt  idx )
  {
    FT_String*  PSname;


    tt_face_get_ps_name( face, idx, &PSname );

    return PSname;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_cmap_unicode_init( PS_Unicodes  unicodes,
                        FT_Pointer   pointer )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    FT_Memory           memory  = FT_FACE_MEMORY( face );
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)face->psnames;

    FT_UNUSED( pointer );


    if ( !psnames->unicodes_init )
      return FT_THROW( Unimplemented_Feature );

    return psnames->unicodes_init( memory,
                                   unicodes,
                                   face->root.num_glyphs,
                                   (PS_GetGlyphNameFunc)&tt_get_glyph_name,
                                   (PS_FreeGlyphNameFunc)NULL,
                                   (FT_Pointer)face );
  }


  FT_CALLBACK_DEF( void )
  tt_cmap_unicode_done( PS_Unicodes  unicodes )
  {
    FT_Face    face   = FT_CMAP_FACE( unicodes );
    FT_Memory  memory = FT_FACE_MEMORY( face );


    FT_FREE( unicodes->maps );
    unicodes->num_maps = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  tt_cmap_unicode_char_index( PS_Unicodes  unicodes,
                              FT_UInt32    char_code )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)face->psnames;


    return psnames->unicodes_char_index( unicodes, char_code );
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  tt_cmap_unicode_char_next( PS_Unicodes  unicodes,
                             FT_UInt32   *pchar_code )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)face->psnames;


    return psnames->unicodes_char_next( unicodes, pchar_code );
  }


  FT_DEFINE_TT_CMAP(
    tt_cmap_unicode_class_rec,

      sizeof ( PS_UnicodesRec ),

      (FT_CMap_InitFunc)     tt_cmap_unicode_init,        /* init       */
      (FT_CMap_DoneFunc)     tt_cmap_unicode_done,        /* done       */
      (FT_CMap_CharIndexFunc)tt_cmap_unicode_char_index,  /* char_index */
      (FT_CMap_CharNextFunc) tt_cmap_unicode_char_next,   /* char_next  */

      (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
      (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
      (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
      (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
      (FT_CMap_VariantCharListFunc) NULL,  /* variantchar_list */

    ~0U,
    (TT_CMap_ValidateFunc)NULL,  /* validate      */
    (TT_CMap_Info_GetFunc)NULL   /* get_cmap_info */
  )

#endif /* FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


  static const TT_CMap_Class  tt_cmap_classes[] =
  {
#define TTCMAPCITEM( a )  &a,
#include "ttcmapc.h"
    NULL,
  };


  /* parse the `cmap' table and build the corresponding TT_CMap objects */
  /* in the current face                                                */
  /*                                                                    */
  FT_LOCAL_DEF( FT_Error )
  tt_face_build_cmaps( TT_Face  face )
  {
    FT_Byte*           table = face->cmap_table;
    FT_Byte*           limit = table + face->cmap_size;
    FT_UInt volatile   num_cmaps;
    FT_Byte* volatile  p     = table;
    FT_Library         library = FT_FACE_LIBRARY( face );

    FT_UNUSED( library );


    if ( !p || p + 4 > limit )
      return FT_THROW( Invalid_Table );

    /* only recognize format 0 */
    if ( TT_NEXT_USHORT( p ) != 0 )
    {
      FT_ERROR(( "tt_face_build_cmaps:"
                 " unsupported `cmap' table format = %d\n",
                 TT_PEEK_USHORT( p - 2 ) ));
      return FT_THROW( Invalid_Table );
    }

    num_cmaps = TT_NEXT_USHORT( p );

    for ( ; num_cmaps > 0 && p + 8 <= limit; num_cmaps-- )
    {
      FT_CharMapRec  charmap;
      FT_UInt32      offset;


      charmap.platform_id = TT_NEXT_USHORT( p );
      charmap.encoding_id = TT_NEXT_USHORT( p );
      charmap.face        = FT_FACE( face );
      charmap.encoding    = FT_ENCODING_NONE;  /* will be filled later */
      offset              = TT_NEXT_ULONG( p );

      if ( offset && offset <= face->cmap_size - 2 )
      {
        FT_Byte* volatile              cmap   = table + offset;
        volatile FT_UInt               format = TT_PEEK_USHORT( cmap );
        const TT_CMap_Class* volatile  pclazz = tt_cmap_classes;
        TT_CMap_Class volatile         clazz;


        for ( ; *pclazz; pclazz++ )
        {
          clazz = *pclazz;
          if ( clazz->format == format )
          {
            volatile TT_ValidatorRec  valid;
            volatile FT_Error         error = FT_Err_Ok;


            ft_validator_init( FT_VALIDATOR( &valid ), cmap, limit,
                               FT_VALIDATE_DEFAULT );

            valid.num_glyphs = (FT_UInt)face->max_profile.numGlyphs;

            if ( ft_setjmp( FT_VALIDATOR( &valid )->jump_buffer) == 0 )
            {
              /* validate this cmap sub-table */
              error = clazz->validate( cmap, FT_VALIDATOR( &valid ) );
            }

            if ( !valid.validator.error )
            {
              FT_CMap  ttcmap;


              /* It might make sense to store the single variation         */
              /* selector cmap somewhere special.  But it would have to be */
              /* in the public FT_FaceRec, and we can't change that.       */

              if ( !FT_CMap_New( (FT_CMap_Class)clazz,
                                 cmap, &charmap, &ttcmap ) )
              {
                /* it is simpler to directly set `flags' than adding */
                /* a parameter to FT_CMap_New                        */
                ((TT_CMap)ttcmap)->flags = (FT_Int)error;
              }
            }
            else
            {
              FT_TRACE0(( "tt_face_build_cmaps:"
                          " broken cmap sub-table ignored\n" ));
            }
            break;
          }
        }

        if ( !*pclazz )
        {
          FT_TRACE0(( "tt_face_build_cmaps:"
                      " unsupported cmap sub-table ignored\n" ));
        }
      }
    }

    return FT_Err_Ok;
  }


  FT_LOCAL( FT_Error )
  tt_get_cmap_info( FT_CharMap    charmap,
                    TT_CMapInfo  *cmap_info )
  {
    FT_CMap        cmap  = (FT_CMap)charmap;
    TT_CMap_Class  clazz = (TT_CMap_Class)cmap->clazz;

    if ( clazz->get_cmap_info )
      return clazz->get_cmap_info( charmap, cmap_info );
    else
      return FT_THROW( Invalid_CharMap_Format );
  }


/* END */
