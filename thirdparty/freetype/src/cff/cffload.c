/****************************************************************************
 *
 * cffload.c
 *
 *   OpenType and CFF data/program tables loader (body).
 *
 * Copyright (C) 1996-2023 by
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
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include <freetype/t1tables.h>
#include <freetype/internal/psaux.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/ftmm.h>
#include <freetype/internal/services/svmm.h>
#endif

#include "cffload.h"
#include "cffparse.h"

#include "cfferrs.h"


#define FT_FIXED_ONE  ( (FT_Fixed)0x10000 )


#if 1

  static const FT_UShort  cff_isoadobe_charset[229] =
  {
      0,   1,   2,   3,   4,   5,   6,   7,
      8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,
     24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,
     40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,
     56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,
     72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,
     88,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215,
    216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228
  };

  static const FT_UShort  cff_expert_charset[166] =
  {
      0,   1, 229, 230, 231, 232, 233, 234,
    235, 236, 237, 238,  13,  14,  15,  99,
    239, 240, 241, 242, 243, 244, 245, 246,
    247, 248,  27,  28, 249, 250, 251, 252,
    253, 254, 255, 256, 257, 258, 259, 260,
    261, 262, 263, 264, 265, 266, 109, 110,
    267, 268, 269, 270, 271, 272, 273, 274,
    275, 276, 277, 278, 279, 280, 281, 282,
    283, 284, 285, 286, 287, 288, 289, 290,
    291, 292, 293, 294, 295, 296, 297, 298,
    299, 300, 301, 302, 303, 304, 305, 306,
    307, 308, 309, 310, 311, 312, 313, 314,
    315, 316, 317, 318, 158, 155, 163, 319,
    320, 321, 322, 323, 324, 325, 326, 150,
    164, 169, 327, 328, 329, 330, 331, 332,
    333, 334, 335, 336, 337, 338, 339, 340,
    341, 342, 343, 344, 345, 346, 347, 348,
    349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364,
    365, 366, 367, 368, 369, 370, 371, 372,
    373, 374, 375, 376, 377, 378
  };

  static const FT_UShort  cff_expertsubset_charset[87] =
  {
      0,   1, 231, 232, 235, 236, 237, 238,
     13,  14,  15,  99, 239, 240, 241, 242,
    243, 244, 245, 246, 247, 248,  27,  28,
    249, 250, 251, 253, 254, 255, 256, 257,
    258, 259, 260, 261, 262, 263, 264, 265,
    266, 109, 110, 267, 268, 269, 270, 272,
    300, 301, 302, 305, 314, 315, 158, 155,
    163, 320, 321, 322, 323, 324, 325, 326,
    150, 164, 169, 327, 328, 329, 330, 331,
    332, 333, 334, 335, 336, 337, 338, 339,
    340, 341, 342, 343, 344, 345, 346
  };

  static const FT_UShort  cff_standard_encoding[256] =
  {
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      1,   2,   3,   4,   5,   6,   7,   8,
      9,  10,  11,  12,  13,  14,  15,  16,
     17,  18,  19,  20,  21,  22,  23,  24,
     25,  26,  27,  28,  29,  30,  31,  32,
     33,  34,  35,  36,  37,  38,  39,  40,
     41,  42,  43,  44,  45,  46,  47,  48,
     49,  50,  51,  52,  53,  54,  55,  56,
     57,  58,  59,  60,  61,  62,  63,  64,
     65,  66,  67,  68,  69,  70,  71,  72,
     73,  74,  75,  76,  77,  78,  79,  80,
     81,  82,  83,  84,  85,  86,  87,  88,
     89,  90,  91,  92,  93,  94,  95,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,  96,  97,  98,  99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110,
      0, 111, 112, 113, 114,   0, 115, 116,
    117, 118, 119, 120, 121, 122,   0, 123,
      0, 124, 125, 126, 127, 128, 129, 130,
    131,   0, 132, 133,   0, 134, 135, 136,
    137,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0, 138,   0, 139,   0,   0,   0,   0,
    140, 141, 142, 143,   0,   0,   0,   0,
      0, 144,   0,   0,   0, 145,   0,   0,
    146, 147, 148, 149,   0,   0,   0,   0
  };

  static const FT_UShort  cff_expert_encoding[256] =
  {
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      1, 229, 230,   0, 231, 232, 233, 234,
    235, 236, 237, 238,  13,  14,  15,  99,
    239, 240, 241, 242, 243, 244, 245, 246,
    247, 248,  27,  28, 249, 250, 251, 252,
      0, 253, 254, 255, 256, 257,   0,   0,
      0, 258,   0,   0, 259, 260, 261, 262,
      0,   0, 263, 264, 265,   0, 266, 109,
    110, 267, 268, 269,   0, 270, 271, 272,
    273, 274, 275, 276, 277, 278, 279, 280,
    281, 282, 283, 284, 285, 286, 287, 288,
    289, 290, 291, 292, 293, 294, 295, 296,
    297, 298, 299, 300, 301, 302, 303,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0, 304, 305, 306,   0,   0, 307, 308,
    309, 310, 311,   0, 312,   0,   0, 312,
      0,   0, 314, 315,   0,   0, 316, 317,
    318,   0,   0,   0, 158, 155, 163, 319,
    320, 321, 322, 323, 324, 325,   0,   0,
    326, 150, 164, 169, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338,
    339, 340, 341, 342, 343, 344, 345, 346,
    347, 348, 349, 350, 351, 352, 353, 354,
    355, 356, 357, 358, 359, 360, 361, 362,
    363, 364, 365, 366, 367, 368, 369, 370,
    371, 372, 373, 374, 375, 376, 377, 378
  };

#endif /* 1 */


  FT_LOCAL_DEF( FT_UShort )
  cff_get_standard_encoding( FT_UInt  charcode )
  {
    return (FT_UShort)( charcode < 256 ? cff_standard_encoding[charcode]
                                       : 0 );
  }


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffload


  /* read an offset from the index's stream current position */
  static FT_ULong
  cff_index_read_offset( CFF_Index  idx,
                         FT_Error  *errorp )
  {
    FT_Error   error;
    FT_Stream  stream = idx->stream;
    FT_Byte    tmp[4];
    FT_ULong   result = 0;


    if ( !FT_STREAM_READ( tmp, idx->off_size ) )
    {
      FT_Int  nn;


      for ( nn = 0; nn < idx->off_size; nn++ )
        result = ( result << 8 ) | tmp[nn];
    }

    *errorp = error;
    return result;
  }


  static FT_Error
  cff_index_init( CFF_Index  idx,
                  FT_Stream  stream,
                  FT_Bool    load,
                  FT_Bool    cff2 )
  {
    FT_Error   error;
    FT_Memory  memory = stream->memory;
    FT_UInt    count;


    FT_ZERO( idx );

    idx->stream = stream;
    idx->start  = FT_STREAM_POS();

    if ( cff2 )
    {
      if ( FT_READ_ULONG( count ) )
        goto Exit;
      idx->hdr_size = 5;
    }
    else
    {
      if ( FT_READ_USHORT( count ) )
        goto Exit;
      idx->hdr_size = 3;
    }

    if ( count > 0 )
    {
      FT_Byte   offsize;
      FT_ULong  size;


      /* there is at least one element; read the offset size,           */
      /* then access the offset table to compute the index's total size */
      if ( FT_READ_BYTE( offsize ) )
        goto Exit;

      if ( offsize < 1 || offsize > 4 )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      idx->count    = count;
      idx->off_size = offsize;
      size          = (FT_ULong)( count + 1 ) * offsize;

      idx->data_offset = idx->start + idx->hdr_size + size;

      if ( FT_STREAM_SKIP( size - offsize ) )
        goto Exit;

      size = cff_index_read_offset( idx, &error );
      if ( error )
        goto Exit;

      if ( size == 0 )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      idx->data_size = --size;

      if ( load )
      {
        /* load the data */
        if ( FT_FRAME_EXTRACT( size, idx->bytes ) )
          goto Exit;
      }
      else
      {
        /* skip the data */
        if ( FT_STREAM_SKIP( size ) )
          goto Exit;
      }
    }

  Exit:
    if ( error )
      FT_FREE( idx->offsets );

    return error;
  }


  static void
  cff_index_done( CFF_Index  idx )
  {
    if ( idx->stream )
    {
      FT_Stream  stream = idx->stream;
      FT_Memory  memory = stream->memory;


      if ( idx->bytes )
        FT_FRAME_RELEASE( idx->bytes );

      FT_FREE( idx->offsets );
      FT_ZERO( idx );
    }
  }


  static FT_Error
  cff_index_load_offsets( CFF_Index  idx )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Stream  stream = idx->stream;
    FT_Memory  memory = stream->memory;


    if ( idx->count > 0 && !idx->offsets )
    {
      FT_Byte    offsize = idx->off_size;
      FT_ULong   data_size;
      FT_Byte*   p;
      FT_Byte*   p_end;
      FT_ULong*  poff;


      data_size = (FT_ULong)( idx->count + 1 ) * offsize;

      if ( FT_QNEW_ARRAY( idx->offsets, idx->count + 1 ) ||
           FT_STREAM_SEEK( idx->start + idx->hdr_size )  ||
           FT_FRAME_ENTER( data_size )                   )
        goto Exit;

      poff   = idx->offsets;
      p      = (FT_Byte*)stream->cursor;
      p_end  = p + data_size;

      switch ( offsize )
      {
      case 1:
        for ( ; p < p_end; p++, poff++ )
          poff[0] = p[0];
        break;

      case 2:
        for ( ; p < p_end; p += 2, poff++ )
          poff[0] = FT_PEEK_USHORT( p );
        break;

      case 3:
        for ( ; p < p_end; p += 3, poff++ )
          poff[0] = FT_PEEK_UOFF3( p );
        break;

      default:
        for ( ; p < p_end; p += 4, poff++ )
          poff[0] = FT_PEEK_ULONG( p );
      }

      FT_FRAME_EXIT();
    }

  Exit:
    if ( error )
      FT_FREE( idx->offsets );

    return error;
  }


  /* Allocate a table containing pointers to an index's elements. */
  /* The `pool' argument makes this function convert the index    */
  /* entries to C-style strings (this is, null-terminated).       */
  static FT_Error
  cff_index_get_pointers( CFF_Index   idx,
                          FT_Byte***  table,
                          FT_Byte**   pool,
                          FT_ULong*   pool_size )
  {
    FT_Error   error     = FT_Err_Ok;
    FT_Memory  memory    = idx->stream->memory;

    FT_Byte**  tbl       = NULL;
    FT_Byte*   new_bytes = NULL;
    FT_ULong   new_size;


    *table = NULL;

    if ( !idx->offsets )
    {
      error = cff_index_load_offsets( idx );
      if ( error )
        goto Exit;
    }

    new_size = idx->data_size + idx->count;

    if ( idx->count > 0                                &&
         !FT_QNEW_ARRAY( tbl, idx->count + 1 )         &&
         ( !pool || !FT_ALLOC( new_bytes, new_size ) ) )
    {
      FT_ULong  n, cur_offset;
      FT_ULong  extra     = 0;
      FT_Byte*  org_bytes = idx->bytes;


      /* at this point, `idx->offsets' can't be NULL */
      cur_offset = idx->offsets[0] - 1;

      /* sanity check */
      if ( cur_offset != 0 )
      {
        FT_TRACE0(( "cff_index_get_pointers:"
                    " invalid first offset value %ld set to zero\n",
                    cur_offset ));
        cur_offset = 0;
      }

      if ( !pool )
        tbl[0] = org_bytes + cur_offset;
      else
        tbl[0] = new_bytes + cur_offset;

      for ( n = 1; n <= idx->count; n++ )
      {
        FT_ULong  next_offset = idx->offsets[n] - 1;


        /* two sanity checks for invalid offset tables */
        if ( next_offset < cur_offset )
          next_offset = cur_offset;
        else if ( next_offset > idx->data_size )
          next_offset = idx->data_size;

        if ( !pool )
          tbl[n] = org_bytes + next_offset;
        else
        {
          tbl[n] = new_bytes + next_offset + extra;

          if ( next_offset != cur_offset )
          {
            FT_MEM_COPY( tbl[n - 1],
                         org_bytes + cur_offset,
                         tbl[n] - tbl[n - 1] );
            tbl[n][0] = '\0';
            tbl[n]   += 1;
            extra++;
          }
        }

        cur_offset = next_offset;
      }
      *table = tbl;

      if ( pool )
        *pool = new_bytes;
      if ( pool_size )
        *pool_size = new_size;
    }

  Exit:
    if ( error && new_bytes )
      FT_FREE( new_bytes );
    if ( error && tbl )
      FT_FREE( tbl );

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  cff_index_access_element( CFF_Index  idx,
                            FT_UInt    element,
                            FT_Byte**  pbytes,
                            FT_ULong*  pbyte_len )
  {
    FT_Error  error = FT_Err_Ok;


    if ( idx && idx->count > element )
    {
      /* compute start and end offsets */
      FT_Stream  stream = idx->stream;
      FT_ULong   off1, off2 = 0;


      /* load offsets from file or the offset table */
      if ( !idx->offsets )
      {
        FT_ULong  pos = element * idx->off_size;


        if ( FT_STREAM_SEEK( idx->start + idx->hdr_size + pos ) )
          goto Exit;

        off1 = cff_index_read_offset( idx, &error );
        if ( error )
          goto Exit;

        if ( off1 != 0 )
        {
          do
          {
            element++;
            off2 = cff_index_read_offset( idx, &error );

          } while ( off2 == 0 && element < idx->count );
        }
      }
      else   /* use offsets table */
      {
        off1 = idx->offsets[element];
        if ( off1 )
        {
          do
          {
            element++;
            off2 = idx->offsets[element];

          } while ( off2 == 0 && element < idx->count );
        }
      }

      /* XXX: should check off2 does not exceed the end of this entry; */
      /*      at present, only truncate off2 at the end of this stream */
      if ( off2 > stream->size + 1                    ||
           idx->data_offset > stream->size - off2 + 1 )
      {
        FT_ERROR(( "cff_index_access_element:"
                   " offset to next entry (%ld)"
                   " exceeds the end of stream (%ld)\n",
                   off2, stream->size - idx->data_offset + 1 ));
        off2 = stream->size - idx->data_offset + 1;
      }

      /* access element */
      if ( off1 && off2 > off1 )
      {
        *pbyte_len = off2 - off1;

        if ( idx->bytes )
        {
          /* this index was completely loaded in memory, that's easy */
          *pbytes = idx->bytes + off1 - 1;
        }
        else
        {
          /* this index is still on disk/file, access it through a frame */
          if ( FT_STREAM_SEEK( idx->data_offset + off1 - 1 ) ||
               FT_FRAME_EXTRACT( off2 - off1, *pbytes )      )
            goto Exit;
        }
      }
      else
      {
        /* empty index element */
        *pbytes    = 0;
        *pbyte_len = 0;
      }
    }
    else
      error = FT_THROW( Invalid_Argument );

  Exit:
    return error;
  }


  FT_LOCAL_DEF( void )
  cff_index_forget_element( CFF_Index  idx,
                            FT_Byte**  pbytes )
  {
    if ( idx->bytes == 0 )
    {
      FT_Stream  stream = idx->stream;


      FT_FRAME_RELEASE( *pbytes );
    }
  }


  /* get an entry from Name INDEX */
  FT_LOCAL_DEF( FT_String* )
  cff_index_get_name( CFF_Font  font,
                      FT_UInt   element )
  {
    CFF_Index   idx = &font->name_index;
    FT_Memory   memory;
    FT_Byte*    bytes;
    FT_ULong    byte_len;
    FT_Error    error;
    FT_String*  name = NULL;


    if ( !idx->stream )  /* CFF2 does not include a name index */
      goto Exit;

    memory = idx->stream->memory;

    error = cff_index_access_element( idx, element, &bytes, &byte_len );
    if ( error )
      goto Exit;

    if ( !FT_QALLOC( name, byte_len + 1 ) )
    {
      FT_MEM_COPY( name, bytes, byte_len );
      name[byte_len] = 0;
    }
    cff_index_forget_element( idx, &bytes );

  Exit:
    return name;
  }


  /* get an entry from String INDEX */
  FT_LOCAL_DEF( FT_String* )
  cff_index_get_string( CFF_Font  font,
                        FT_UInt   element )
  {
    return ( element < font->num_strings )
             ? (FT_String*)font->strings[element]
             : NULL;
  }


  FT_LOCAL_DEF( FT_String* )
  cff_index_get_sid_string( CFF_Font  font,
                            FT_UInt   sid )
  {
    /* value 0xFFFFU indicates a missing dictionary entry */
    if ( sid == 0xFFFFU )
      return NULL;

    /* if it is not a standard string, return it */
    if ( sid > 390 )
      return cff_index_get_string( font, sid - 391 );

    /* CID-keyed CFF fonts don't have glyph names */
    if ( !font->psnames )
      return NULL;

    /* this is a standard string */
    return (FT_String *)font->psnames->adobe_std_strings( sid );
  }


  /*************************************************************************/
  /*************************************************************************/
  /***                                                                   ***/
  /***   FD Select table support                                         ***/
  /***                                                                   ***/
  /*************************************************************************/
  /*************************************************************************/


  static void
  CFF_Done_FD_Select( CFF_FDSelect  fdselect,
                      FT_Stream     stream )
  {
    if ( fdselect->data )
      FT_FRAME_RELEASE( fdselect->data );

    fdselect->data_size   = 0;
    fdselect->format      = 0;
    fdselect->range_count = 0;
  }


  static FT_Error
  CFF_Load_FD_Select( CFF_FDSelect  fdselect,
                      FT_UInt       num_glyphs,
                      FT_Stream     stream,
                      FT_ULong      offset )
  {
    FT_Error  error;
    FT_Byte   format;
    FT_UInt   num_ranges;


    /* read format */
    if ( FT_STREAM_SEEK( offset ) || FT_READ_BYTE( format ) )
      goto Exit;

    fdselect->format      = format;
    fdselect->cache_count = 0;   /* clear cache */

    switch ( format )
    {
    case 0:     /* format 0, that's simple */
      fdselect->data_size = num_glyphs;
      goto Load_Data;

    case 3:     /* format 3, a tad more complex */
      if ( FT_READ_USHORT( num_ranges ) )
        goto Exit;

      if ( !num_ranges )
      {
        FT_TRACE0(( "CFF_Load_FD_Select: empty FDSelect array\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      fdselect->data_size = num_ranges * 3 + 2;

    Load_Data:
      if ( FT_FRAME_EXTRACT( fdselect->data_size, fdselect->data ) )
        goto Exit;
      break;

    default:    /* hmm... that's wrong */
      error = FT_THROW( Invalid_File_Format );
    }

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Byte )
  cff_fd_select_get( CFF_FDSelect  fdselect,
                     FT_UInt       glyph_index )
  {
    FT_Byte  fd = 0;


    /* if there is no FDSelect, return zero               */
    /* Note: CFF2 with just one Font Dict has no FDSelect */
    if ( !fdselect->data )
      goto Exit;

    switch ( fdselect->format )
    {
    case 0:
      fd = fdselect->data[glyph_index];
      break;

    case 3:
      /* first, compare to the cache */
      if ( glyph_index - fdselect->cache_first < fdselect->cache_count )
      {
        fd = fdselect->cache_fd;
        break;
      }

      /* then, look up the ranges array */
      {
        FT_Byte*  p       = fdselect->data;
        FT_Byte*  p_limit = p + fdselect->data_size;
        FT_Byte   fd2;
        FT_UInt   first, limit;


        first = FT_NEXT_USHORT( p );
        do
        {
          if ( glyph_index < first )
            break;

          fd2   = *p++;
          limit = FT_NEXT_USHORT( p );

          if ( glyph_index < limit )
          {
            fd = fd2;

            /* update cache */
            fdselect->cache_first = first;
            fdselect->cache_count = limit - first;
            fdselect->cache_fd    = fd2;
            break;
          }
          first = limit;

        } while ( p < p_limit );
      }
      break;

    default:
      ;
    }

  Exit:
    return fd;
  }


  /*************************************************************************/
  /*************************************************************************/
  /***                                                                   ***/
  /***   CFF font support                                                ***/
  /***                                                                   ***/
  /*************************************************************************/
  /*************************************************************************/

  static FT_Error
  cff_charset_compute_cids( CFF_Charset  charset,
                            FT_UInt      num_glyphs,
                            FT_Memory    memory )
  {
    FT_Error   error   = FT_Err_Ok;
    FT_UInt    i;
    FT_UShort  max_cid = 0;


    if ( charset->max_cid > 0 )
      goto Exit;

    for ( i = 0; i < num_glyphs; i++ )
    {
      if ( charset->sids[i] > max_cid )
        max_cid = charset->sids[i];
    }

    if ( FT_NEW_ARRAY( charset->cids, (FT_ULong)max_cid + 1 ) )
      goto Exit;

    /* When multiple GIDs map to the same CID, we choose the lowest */
    /* GID.  This is not described in any spec, but it matches the  */
    /* behaviour of recent Acroread versions.  The loop stops when  */
    /* the unsigned index wraps around after reaching zero.         */
    for ( i = num_glyphs - 1; i < num_glyphs; i-- )
      charset->cids[charset->sids[i]] = (FT_UShort)i;

    charset->max_cid    = max_cid;
    charset->num_glyphs = num_glyphs;

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_UInt )
  cff_charset_cid_to_gindex( CFF_Charset  charset,
                             FT_UInt      cid )
  {
    FT_UInt  result = 0;


    if ( cid <= charset->max_cid )
      result = charset->cids[cid];

    return result;
  }


  static void
  cff_charset_free_cids( CFF_Charset  charset,
                         FT_Memory    memory )
  {
    FT_FREE( charset->cids );
    charset->max_cid = 0;
  }


  static void
  cff_charset_done( CFF_Charset  charset,
                    FT_Stream    stream )
  {
    FT_Memory  memory = stream->memory;


    cff_charset_free_cids( charset, memory );

    FT_FREE( charset->sids );
    charset->format = 0;
    charset->offset = 0;
  }


  static FT_Error
  cff_charset_load( CFF_Charset  charset,
                    FT_UInt      num_glyphs,
                    FT_Stream    stream,
                    FT_ULong     base_offset,
                    FT_ULong     offset,
                    FT_Bool      invert )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;
    FT_UShort  glyph_sid;


    /* If the offset is greater than 2, we have to parse the charset */
    /* table.                                                        */
    if ( offset > 2 )
    {
      FT_UInt  j;


      charset->offset = base_offset + offset;

      /* Get the format of the table. */
      if ( FT_STREAM_SEEK( charset->offset ) ||
           FT_READ_BYTE( charset->format )   )
        goto Exit;

      /* Allocate memory for sids. */
      if ( FT_QNEW_ARRAY( charset->sids, num_glyphs ) )
        goto Exit;

      /* assign the .notdef glyph */
      charset->sids[0] = 0;

      switch ( charset->format )
      {
      case 0:
        if ( num_glyphs > 0 )
        {
          if ( FT_FRAME_ENTER( ( num_glyphs - 1 ) * 2 ) )
            goto Exit;

          for ( j = 1; j < num_glyphs; j++ )
            charset->sids[j] = FT_GET_USHORT();

          FT_FRAME_EXIT();
        }
        break;

      case 1:
      case 2:
        {
          FT_UInt  nleft;
          FT_UInt  i;


          j = 1;

          while ( j < num_glyphs )
          {
            /* Read the first glyph sid of the range. */
            if ( FT_READ_USHORT( glyph_sid ) )
              goto Exit;

            /* Read the number of glyphs in the range.  */
            if ( charset->format == 2 )
            {
              if ( FT_READ_USHORT( nleft ) )
                goto Exit;
            }
            else
            {
              if ( FT_READ_BYTE( nleft ) )
                goto Exit;
            }

            /* try to rescue some of the SIDs if `nleft' is too large */
            if ( glyph_sid > 0xFFFFL - nleft )
            {
              FT_ERROR(( "cff_charset_load: invalid SID range trimmed"
                         " nleft=%d -> %ld\n", nleft, 0xFFFFL - glyph_sid ));
              nleft = ( FT_UInt )( 0xFFFFL - glyph_sid );
            }

            /* Fill in the range of sids -- `nleft + 1' glyphs. */
            for ( i = 0; j < num_glyphs && i <= nleft; i++, j++, glyph_sid++ )
              charset->sids[j] = glyph_sid;
          }
        }
        break;

      default:
        FT_ERROR(( "cff_charset_load: invalid table format\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }
    else
    {
      /* Parse default tables corresponding to offset == 0, 1, or 2.  */
      /* CFF specification intimates the following:                   */
      /*                                                              */
      /* In order to use a predefined charset, the following must be  */
      /* true: The charset constructed for the glyphs in the font's   */
      /* charstrings dictionary must match the predefined charset in  */
      /* the first num_glyphs.                                        */

      charset->offset = offset;  /* record charset type */

      switch ( (FT_UInt)offset )
      {
      case 0:
        if ( num_glyphs > 229 )
        {
          FT_ERROR(( "cff_charset_load: implicit charset larger than\n" ));
          FT_ERROR(( "predefined charset (Adobe ISO-Latin)\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /* Allocate memory for sids. */
        if ( FT_QNEW_ARRAY( charset->sids, num_glyphs ) )
          goto Exit;

        /* Copy the predefined charset into the allocated memory. */
        FT_ARRAY_COPY( charset->sids, cff_isoadobe_charset, num_glyphs );

        break;

      case 1:
        if ( num_glyphs > 166 )
        {
          FT_ERROR(( "cff_charset_load: implicit charset larger than\n" ));
          FT_ERROR(( "predefined charset (Adobe Expert)\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /* Allocate memory for sids. */
        if ( FT_QNEW_ARRAY( charset->sids, num_glyphs ) )
          goto Exit;

        /* Copy the predefined charset into the allocated memory.     */
        FT_ARRAY_COPY( charset->sids, cff_expert_charset, num_glyphs );

        break;

      case 2:
        if ( num_glyphs > 87 )
        {
          FT_ERROR(( "cff_charset_load: implicit charset larger than\n" ));
          FT_ERROR(( "predefined charset (Adobe Expert Subset)\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /* Allocate memory for sids. */
        if ( FT_QNEW_ARRAY( charset->sids, num_glyphs ) )
          goto Exit;

        /* Copy the predefined charset into the allocated memory.     */
        FT_ARRAY_COPY( charset->sids, cff_expertsubset_charset, num_glyphs );

        break;

      default:
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }

    /* we have to invert the `sids' array for subsetted CID-keyed fonts */
    if ( invert )
      error = cff_charset_compute_cids( charset, num_glyphs, memory );

  Exit:
    /* Clean up if there was an error. */
    if ( error )
    {
      FT_FREE( charset->sids );
      FT_FREE( charset->cids );
      charset->format = 0;
      charset->offset = 0;
    }

    return error;
  }


  static void
  cff_vstore_done( CFF_VStoreRec*  vstore,
                   FT_Memory       memory )
  {
    FT_UInt  i;


    /* free regionList and axisLists */
    if ( vstore->varRegionList )
    {
      for ( i = 0; i < vstore->regionCount; i++ )
        FT_FREE( vstore->varRegionList[i].axisList );
    }
    FT_FREE( vstore->varRegionList );

    /* free varData and indices */
    if ( vstore->varData )
    {
      for ( i = 0; i < vstore->dataCount; i++ )
        FT_FREE( vstore->varData[i].regionIndices );
    }
    FT_FREE( vstore->varData );
  }


  /* convert 2.14 to Fixed */
  #define FT_fdot14ToFixed( x )  ( (FT_Fixed)( (FT_ULong)(x) << 2 ) )


  static FT_Error
  cff_vstore_load( CFF_VStoreRec*  vstore,
                   FT_Stream       stream,
                   FT_ULong        base_offset,
                   FT_ULong        offset )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_ERR( Invalid_File_Format );

    FT_ULong*  dataOffsetArray = NULL;
    FT_UInt    i, j;


    /* no offset means no vstore to parse */
    if ( offset )
    {
      FT_UInt   vsOffset;
      FT_UInt   format;
      FT_UInt   dataCount;
      FT_UInt   regionCount;
      FT_ULong  regionListOffset;


      /* we need to parse the table to determine its size; */
      /* skip table length                                 */
      if ( FT_STREAM_SEEK( base_offset + offset ) ||
           FT_STREAM_SKIP( 2 )                    )
        goto Exit;

      /* actual variation store begins after the length */
      vsOffset = FT_STREAM_POS();

      /* check the header */
      if ( FT_READ_USHORT( format ) )
        goto Exit;
      if ( format != 1 )
      {
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* read top level fields */
      if ( FT_READ_ULONG( regionListOffset ) ||
           FT_READ_USHORT( dataCount )       )
        goto Exit;

      /* make temporary copy of item variation data offsets; */
      /* we'll parse region list first, then come back       */
      if ( FT_QNEW_ARRAY( dataOffsetArray, dataCount ) )
        goto Exit;

      for ( i = 0; i < dataCount; i++ )
      {
        if ( FT_READ_ULONG( dataOffsetArray[i] ) )
          goto Exit;
      }

      /* parse regionList and axisLists */
      if ( FT_STREAM_SEEK( vsOffset + regionListOffset ) ||
           FT_READ_USHORT( vstore->axisCount )           ||
           FT_READ_USHORT( regionCount )                 )
        goto Exit;

      vstore->regionCount = 0;
      if ( FT_QNEW_ARRAY( vstore->varRegionList, regionCount ) )
        goto Exit;

      for ( i = 0; i < regionCount; i++ )
      {
        CFF_VarRegion*  region = &vstore->varRegionList[i];


        if ( FT_QNEW_ARRAY( region->axisList, vstore->axisCount ) )
          goto Exit;

        /* keep track of how many axisList to deallocate on error */
        vstore->regionCount++;

        for ( j = 0; j < vstore->axisCount; j++ )
        {
          CFF_AxisCoords*  axis = &region->axisList[j];

          FT_Int16  start14, peak14, end14;


          if ( FT_READ_SHORT( start14 ) ||
               FT_READ_SHORT( peak14 )  ||
               FT_READ_SHORT( end14 )   )
            goto Exit;

          axis->startCoord = FT_fdot14ToFixed( start14 );
          axis->peakCoord  = FT_fdot14ToFixed( peak14 );
          axis->endCoord   = FT_fdot14ToFixed( end14 );
        }
      }

      /* use dataOffsetArray now to parse varData items */
      vstore->dataCount = 0;
      if ( FT_QNEW_ARRAY( vstore->varData, dataCount ) )
        goto Exit;

      for ( i = 0; i < dataCount; i++ )
      {
        CFF_VarData*  data = &vstore->varData[i];


        if ( FT_STREAM_SEEK( vsOffset + dataOffsetArray[i] ) )
          goto Exit;

        /* ignore `itemCount' and `shortDeltaCount' */
        /* because CFF2 has no delta sets           */
        if ( FT_STREAM_SKIP( 4 ) )
          goto Exit;

        /* Note: just record values; consistency is checked later    */
        /*       by cff_blend_build_vector when it consumes `vstore' */

        if ( FT_READ_USHORT( data->regionIdxCount ) )
          goto Exit;

        if ( FT_QNEW_ARRAY( data->regionIndices, data->regionIdxCount ) )
          goto Exit;

        /* keep track of how many regionIndices to deallocate on error */
        vstore->dataCount++;

        for ( j = 0; j < data->regionIdxCount; j++ )
        {
          if ( FT_READ_USHORT( data->regionIndices[j] ) )
            goto Exit;
        }
      }
    }

    error = FT_Err_Ok;

  Exit:
    FT_FREE( dataOffsetArray );
    if ( error )
      cff_vstore_done( vstore, memory );

    return error;
  }


  /* Clear blend stack (after blend values are consumed). */
  /*                                                      */
  /* TODO: Should do this in cff_run_parse, but subFont   */
  /*       ref is not available there.                    */
  /*                                                      */
  /* Allocation is not changed when stack is cleared.     */
  FT_LOCAL_DEF( void )
  cff_blend_clear( CFF_SubFont  subFont )
  {
    subFont->blend_top  = subFont->blend_stack;
    subFont->blend_used = 0;
  }


  /* Blend numOperands on the stack,                       */
  /* store results into the first numBlends values,        */
  /* then pop remaining arguments.                         */
  /*                                                       */
  /* This is comparable to `cf2_doBlend' but               */
  /* the cffparse stack is different and can't be written. */
  /* Blended values are written to a different buffer,     */
  /* using reserved operator 255.                          */
  /*                                                       */
  /* Blend calculation is done in 16.16 fixed-point.       */
  FT_LOCAL_DEF( FT_Error )
  cff_blend_doBlend( CFF_SubFont  subFont,
                     CFF_Parser   parser,
                     FT_UInt      numBlends )
  {
    FT_UInt  delta;
    FT_UInt  base;
    FT_UInt  i, j;
    FT_UInt  size;

    CFF_Blend  blend = &subFont->blend;

    FT_Memory  memory = subFont->blend.font->memory; /* for FT_REALLOC */
    FT_Error   error  = FT_Err_Ok;                   /* for FT_REALLOC */

    /* compute expected number of operands for this blend */
    FT_UInt  numOperands = (FT_UInt)( numBlends * blend->lenBV );
    FT_UInt  count       = (FT_UInt)( parser->top - 1 - parser->stack );


    if ( numOperands > count )
    {
      FT_TRACE4(( " cff_blend_doBlend: Stack underflow %d argument%s\n",
                  count,
                  count == 1 ? "" : "s" ));

      error = FT_THROW( Stack_Underflow );
      goto Exit;
    }

    /* check whether we have room for `numBlends' values at `blend_top' */
    size = 5 * numBlends;           /* add 5 bytes per entry    */
    if ( subFont->blend_used + size > subFont->blend_alloc )
    {
      FT_Byte*  blend_stack_old = subFont->blend_stack;
      FT_Byte*  blend_top_old   = subFont->blend_top;


      /* increase or allocate `blend_stack' and reset `blend_top'; */
      /* prepare to append `numBlends' values to the buffer        */
      if ( FT_QREALLOC( subFont->blend_stack,
                        subFont->blend_alloc,
                        subFont->blend_alloc + size ) )
        goto Exit;

      subFont->blend_top    = subFont->blend_stack + subFont->blend_used;
      subFont->blend_alloc += size;

      /* iterate over the parser stack and adjust pointers */
      /* if the reallocated buffer has a different address */
      if ( blend_stack_old                         &&
           subFont->blend_stack != blend_stack_old )
      {
        FT_PtrDist  offset = subFont->blend_stack - blend_stack_old;
        FT_Byte**   p;


        for ( p = parser->stack; p < parser->top; p++ )
        {
          if ( *p >= blend_stack_old && *p < blend_top_old )
            *p += offset;
        }
      }
    }
    subFont->blend_used += size;

    base  = count - numOperands;     /* index of first blend arg */
    delta = base + numBlends;        /* index of first delta arg */

    for ( i = 0; i < numBlends; i++ )
    {
      const FT_Int32*  weight = &blend->BV[1];
      FT_UInt32        sum;


      /* convert inputs to 16.16 fixed-point */
      sum = cff_parse_num( parser, &parser->stack[i + base] ) * 0x10000;

      for ( j = 1; j < blend->lenBV; j++ )
        sum += cff_parse_num( parser, &parser->stack[delta++] ) * *weight++;

      /* point parser stack to new value on blend_stack */
      parser->stack[i + base] = subFont->blend_top;

      /* Push blended result as Type 2 5-byte fixed-point number.  This */
      /* will not conflict with actual DICTs because 255 is a reserved  */
      /* opcode in both CFF and CFF2 DICTs.  See `cff_parse_num' for    */
      /* decode of this, which rounds to an integer.                    */
      *subFont->blend_top++ = 255;
      *subFont->blend_top++ = (FT_Byte)( sum >> 24 );
      *subFont->blend_top++ = (FT_Byte)( sum >> 16 );
      *subFont->blend_top++ = (FT_Byte)( sum >>  8 );
      *subFont->blend_top++ = (FT_Byte)sum;
    }

    /* leave only numBlends results on parser stack */
    parser->top = &parser->stack[base + numBlends];

  Exit:
    return error;
  }


  /* Compute a blend vector from variation store index and normalized  */
  /* vector based on pseudo-code in OpenType Font Variations Overview. */
  /*                                                                   */
  /* Note: lenNDV == 0 produces a default blend vector, (1,0,0,...).   */
  FT_LOCAL_DEF( FT_Error )
  cff_blend_build_vector( CFF_Blend  blend,
                          FT_UInt    vsindex,
                          FT_UInt    lenNDV,
                          FT_Fixed*  NDV )
  {
    FT_Error   error  = FT_Err_Ok;            /* for FT_REALLOC */
    FT_Memory  memory = blend->font->memory;  /* for FT_REALLOC */

    FT_UInt       len;
    CFF_VStore    vs;
    CFF_VarData*  varData;
    FT_UInt       master;


    /* protect against malformed fonts */
    if ( !( lenNDV == 0 || NDV ) )
    {
      FT_TRACE4(( " cff_blend_build_vector:"
                  " Malformed Normalize Design Vector data\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    blend->builtBV = FALSE;

    vs = &blend->font->vstore;

    /* VStore and fvar must be consistent */
    if ( lenNDV != 0 && lenNDV != vs->axisCount )
    {
      FT_TRACE4(( " cff_blend_build_vector: Axis count mismatch\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( vsindex >= vs->dataCount )
    {
      FT_TRACE4(( " cff_blend_build_vector: vsindex out of range\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* select the item variation data structure */
    varData = &vs->varData[vsindex];

    /* prepare buffer for the blend vector */
    len = varData->regionIdxCount + 1;    /* add 1 for default component */
    if ( FT_QRENEW_ARRAY( blend->BV, blend->lenBV, len ) )
      goto Exit;

    blend->lenBV = len;

    /* outer loop steps through master designs to be blended */
    for ( master = 0; master < len; master++ )
    {
      FT_UInt         j;
      FT_UInt         idx;
      CFF_VarRegion*  varRegion;


      /* default factor is always one */
      if ( master == 0 )
      {
        blend->BV[master] = FT_FIXED_ONE;
        FT_TRACE4(( "   build blend vector len %d\n", len ));
        FT_TRACE4(( "   [ %f ", blend->BV[master] / 65536.0 ));
        continue;
      }

      /* VStore array does not include default master, so subtract one */
      idx       = varData->regionIndices[master - 1];
      varRegion = &vs->varRegionList[idx];

      if ( idx >= vs->regionCount )
      {
        FT_TRACE4(( " cff_blend_build_vector:"
                    " region index out of range\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* Note: `lenNDV' could be zero.                              */
      /*       In that case, build default blend vector (1,0,0...). */
      if ( !lenNDV )
      {
        blend->BV[master] = 0;
        continue;
      }

      /* In the normal case, initialize each component to 1 */
      /* before inner loop.                                 */
      blend->BV[master] = FT_FIXED_ONE; /* default */

      /* inner loop steps through axes in this region */
      for ( j = 0; j < lenNDV; j++ )
      {
        CFF_AxisCoords*  axis = &varRegion->axisList[j];
        FT_Fixed         axisScalar;


        /* compute the scalar contribution of this axis; */
        /* ignore invalid ranges                         */
        if ( axis->startCoord > axis->peakCoord ||
             axis->peakCoord > axis->endCoord   )
          axisScalar = FT_FIXED_ONE;

        else if ( axis->startCoord < 0 &&
                  axis->endCoord > 0   &&
                  axis->peakCoord != 0 )
          axisScalar = FT_FIXED_ONE;

        /* peak of 0 means ignore this axis */
        else if ( axis->peakCoord == 0 )
          axisScalar = FT_FIXED_ONE;

        /* ignore this region if coords are out of range */
        else if ( NDV[j] < axis->startCoord ||
                  NDV[j] > axis->endCoord   )
          axisScalar = 0;

        /* calculate a proportional factor */
        else
        {
          if ( NDV[j] == axis->peakCoord )
            axisScalar = FT_FIXED_ONE;
          else if ( NDV[j] < axis->peakCoord )
            axisScalar = FT_DivFix( NDV[j] - axis->startCoord,
                                    axis->peakCoord - axis->startCoord );
          else
            axisScalar = FT_DivFix( axis->endCoord - NDV[j],
                                    axis->endCoord - axis->peakCoord );
        }

        /* take product of all the axis scalars */
        blend->BV[master] = FT_MulFix( blend->BV[master], axisScalar );
      }

      FT_TRACE4(( ", %f ",
                  blend->BV[master] / 65536.0 ));
    }

    FT_TRACE4(( "]\n" ));

    /* record the parameters used to build the blend vector */
    blend->lastVsindex = vsindex;

    if ( lenNDV != 0 )
    {
      /* user has set a normalized vector */
      if ( FT_QRENEW_ARRAY( blend->lastNDV, blend->lenNDV, lenNDV ) )
        goto Exit;

      FT_MEM_COPY( blend->lastNDV,
                   NDV,
                   lenNDV * sizeof ( *NDV ) );
    }

    blend->lenNDV  = lenNDV;
    blend->builtBV = TRUE;

  Exit:
    return error;
  }


  /* `lenNDV' is zero for default vector;           */
  /* return TRUE if blend vector needs to be built. */
  FT_LOCAL_DEF( FT_Bool )
  cff_blend_check_vector( CFF_Blend  blend,
                          FT_UInt    vsindex,
                          FT_UInt    lenNDV,
                          FT_Fixed*  NDV )
  {
    if ( !blend->builtBV                                ||
         blend->lastVsindex != vsindex                  ||
         blend->lenNDV != lenNDV                        ||
         ( lenNDV                                     &&
           ft_memcmp( NDV,
                      blend->lastNDV,
                      lenNDV * sizeof ( *NDV ) ) != 0 ) )
    {
      /* need to build blend vector */
      return TRUE;
    }

    return FALSE;
  }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  FT_LOCAL_DEF( FT_Error )
  cff_get_var_blend( CFF_Face     face,
                     FT_UInt     *num_coords,
                     FT_Fixed*   *coords,
                     FT_Fixed*   *normalizedcoords,
                     FT_MM_Var*  *mm_var )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    return mm->get_var_blend( FT_FACE( face ),
                              num_coords,
                              coords,
                              normalizedcoords,
                              mm_var );
  }


  FT_LOCAL_DEF( void )
  cff_done_blend( CFF_Face  face )
  {
    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


    if (mm)
      mm->done_blend( FT_FACE( face ) );
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  static void
  cff_encoding_done( CFF_Encoding  encoding )
  {
    encoding->format = 0;
    encoding->offset = 0;
    encoding->count  = 0;
  }


  static FT_Error
  cff_encoding_load( CFF_Encoding  encoding,
                     CFF_Charset   charset,
                     FT_UInt       num_glyphs,
                     FT_Stream     stream,
                     FT_ULong      base_offset,
                     FT_ULong      offset )
  {
    FT_Error   error = FT_Err_Ok;
    FT_UInt    count;
    FT_UInt    j;
    FT_UShort  glyph_sid;
    FT_UInt    glyph_code;


    /* Check for charset->sids.  If we do not have this, we fail. */
    if ( !charset->sids )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* Zero out the code to gid/sid mappings. */
    for ( j = 0; j < 256; j++ )
    {
      encoding->sids [j] = 0;
      encoding->codes[j] = 0;
    }

    /* Note: The encoding table in a CFF font is indexed by glyph index;  */
    /* the first encoded glyph index is 1.  Hence, we read the character  */
    /* code (`glyph_code') at index j and make the assignment:            */
    /*                                                                    */
    /*    encoding->codes[glyph_code] = j + 1                             */
    /*                                                                    */
    /* We also make the assignment:                                       */
    /*                                                                    */
    /*    encoding->sids[glyph_code] = charset->sids[j + 1]               */
    /*                                                                    */
    /* This gives us both a code to GID and a code to SID mapping.        */

    if ( offset > 1 )
    {
      encoding->offset = base_offset + offset;

      /* we need to parse the table to determine its size */
      if ( FT_STREAM_SEEK( encoding->offset ) ||
           FT_READ_BYTE( encoding->format )   ||
           FT_READ_BYTE( count )              )
        goto Exit;

      switch ( encoding->format & 0x7F )
      {
      case 0:
        {
          FT_Byte*  p;


          /* By convention, GID 0 is always ".notdef" and is never */
          /* coded in the font.  Hence, the number of codes found  */
          /* in the table is `count+1'.                            */
          /*                                                       */
          encoding->count = count + 1;

          if ( FT_FRAME_ENTER( count ) )
            goto Exit;

          p = (FT_Byte*)stream->cursor;

          for ( j = 1; j <= count; j++ )
          {
            glyph_code = *p++;

            /* Make sure j is not too big. */
            if ( j < num_glyphs )
            {
              /* Assign code to GID mapping. */
              encoding->codes[glyph_code] = (FT_UShort)j;

              /* Assign code to SID mapping. */
              encoding->sids[glyph_code] = charset->sids[j];
            }
          }

          FT_FRAME_EXIT();
        }
        break;

      case 1:
        {
          FT_UInt  nleft;
          FT_UInt  i = 1;
          FT_UInt  k;


          encoding->count = 0;

          /* Parse the Format1 ranges. */
          for ( j = 0;  j < count; j++, i += nleft )
          {
            /* Read the first glyph code of the range. */
            if ( FT_READ_BYTE( glyph_code ) )
              goto Exit;

            /* Read the number of codes in the range. */
            if ( FT_READ_BYTE( nleft ) )
              goto Exit;

            /* Increment nleft, so we read `nleft + 1' codes/sids. */
            nleft++;

            /* compute max number of character codes */
            if ( (FT_UInt)nleft > encoding->count )
              encoding->count = nleft;

            /* Fill in the range of codes/sids. */
            for ( k = i; k < nleft + i; k++, glyph_code++ )
            {
              /* Make sure k is not too big. */
              if ( k < num_glyphs && glyph_code < 256 )
              {
                /* Assign code to GID mapping. */
                encoding->codes[glyph_code] = (FT_UShort)k;

                /* Assign code to SID mapping. */
                encoding->sids[glyph_code] = charset->sids[k];
              }
            }
          }

          /* simple check; one never knows what can be found in a font */
          if ( encoding->count > 256 )
            encoding->count = 256;
        }
        break;

      default:
        FT_ERROR(( "cff_encoding_load: invalid table format\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* Parse supplemental encodings, if any. */
      if ( encoding->format & 0x80 )
      {
        FT_UInt  gindex;


        /* count supplements */
        if ( FT_READ_BYTE( count ) )
          goto Exit;

        for ( j = 0; j < count; j++ )
        {
          /* Read supplemental glyph code. */
          if ( FT_READ_BYTE( glyph_code ) )
            goto Exit;

          /* Read the SID associated with this glyph code. */
          if ( FT_READ_USHORT( glyph_sid ) )
            goto Exit;

          /* Assign code to SID mapping. */
          encoding->sids[glyph_code] = glyph_sid;

          /* First, look up GID which has been assigned to */
          /* SID glyph_sid.                                */
          for ( gindex = 0; gindex < num_glyphs; gindex++ )
          {
            if ( charset->sids[gindex] == glyph_sid )
            {
              encoding->codes[glyph_code] = (FT_UShort)gindex;
              break;
            }
          }
        }
      }
    }
    else
    {
      /* We take into account the fact a CFF font can use a predefined */
      /* encoding without containing all of the glyphs encoded by this */
      /* encoding (see the note at the end of section 12 in the CFF    */
      /* specification).                                               */

      switch ( (FT_UInt)offset )
      {
      case 0:
        /* First, copy the code to SID mapping. */
        FT_ARRAY_COPY( encoding->sids, cff_standard_encoding, 256 );
        goto Populate;

      case 1:
        /* First, copy the code to SID mapping. */
        FT_ARRAY_COPY( encoding->sids, cff_expert_encoding, 256 );

      Populate:
        /* Construct code to GID mapping from code to SID mapping */
        /* and charset.                                           */

        encoding->offset = offset; /* used in cff_face_init */
        encoding->count  = 0;

        error = cff_charset_compute_cids( charset, num_glyphs,
                                          stream->memory );
        if ( error )
          goto Exit;

        for ( j = 0; j < 256; j++ )
        {
          FT_UInt  sid = encoding->sids[j];
          FT_UInt  gid = 0;


          if ( sid )
            gid = cff_charset_cid_to_gindex( charset, sid );

          if ( gid != 0 )
          {
            encoding->codes[j] = (FT_UShort)gid;
            encoding->count    = j + 1;
          }
          else
          {
            encoding->codes[j] = 0;
            encoding->sids [j] = 0;
          }
        }
        break;

      default:
        FT_ERROR(( "cff_encoding_load: invalid table format\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }

  Exit:

    /* Clean up if there was an error. */
    return error;
  }


  /* Parse private dictionary; first call is always from `cff_face_init', */
  /* so NDV has not been set for CFF2 variation.                          */
  /*                                                                      */
  /* `cff_slot_load' must call this function each time NDV changes.       */
  FT_LOCAL_DEF( FT_Error )
  cff_load_private_dict( CFF_Font     font,
                         CFF_SubFont  subfont,
                         FT_UInt      lenNDV,
                         FT_Fixed*    NDV )
  {
    FT_Error         error  = FT_Err_Ok;
    CFF_ParserRec    parser;
    CFF_FontRecDict  top    = &subfont->font_dict;
    CFF_Private      priv   = &subfont->private_dict;
    FT_Stream        stream = font->stream;
    FT_UInt          stackSize;


    /* store handle needed to access memory, vstore for blend;    */
    /* we need this for clean-up even if there is no private DICT */
    subfont->blend.font   = font;
    subfont->blend.usedBV = FALSE;  /* clear state */

    if ( !top->private_offset || !top->private_size )
      goto Exit2;       /* no private DICT, do nothing */

    /* set defaults */
    FT_ZERO( priv );

    priv->blue_shift       = 7;
    priv->blue_fuzz        = 1;
    priv->lenIV            = -1;
    priv->expansion_factor = (FT_Fixed)( 0.06 * 0x10000L );
    priv->blue_scale       = (FT_Fixed)( 0.039625 * 0x10000L * 1000 );

    /* provide inputs for blend calculations */
    priv->subfont   = subfont;
    subfont->lenNDV = lenNDV;
    subfont->NDV    = NDV;

    /* add 1 for the operator */
    stackSize = font->cff2 ? font->top_font.font_dict.maxstack + 1
                           : CFF_MAX_STACK_DEPTH + 1;

    if ( cff_parser_init( &parser,
                          font->cff2 ? CFF2_CODE_PRIVATE : CFF_CODE_PRIVATE,
                          priv,
                          font->library,
                          stackSize,
                          top->num_designs,
                          top->num_axes ) )
      goto Exit;

    if ( FT_STREAM_SEEK( font->base_offset + top->private_offset ) ||
         FT_FRAME_ENTER( top->private_size )                       )
      goto Exit;

    FT_TRACE4(( " private dictionary:\n" ));
    error = cff_parser_run( &parser,
                            (FT_Byte*)stream->cursor,
                            (FT_Byte*)stream->limit );
    FT_FRAME_EXIT();

    if ( error )
      goto Exit;

    /* ensure that `num_blue_values' is even */
    priv->num_blue_values &= ~1;

    /* sanitize `initialRandomSeed' to be a positive value, if necessary;  */
    /* this is not mandated by the specification but by our implementation */
    if ( priv->initial_random_seed < 0 )
      priv->initial_random_seed = -priv->initial_random_seed;
    else if ( priv->initial_random_seed == 0 )
      priv->initial_random_seed = 987654321;

    /* some sanitizing to avoid overflows later on; */
    /* the upper limits are ad-hoc values           */
    if ( priv->blue_shift > 1000 || priv->blue_shift < 0 )
    {
      FT_TRACE2(( "cff_load_private_dict:"
                  " setting unlikely BlueShift value %ld to default (7)\n",
                  priv->blue_shift ));
      priv->blue_shift = 7;
    }

    if ( priv->blue_fuzz > 1000 || priv->blue_fuzz < 0 )
    {
      FT_TRACE2(( "cff_load_private_dict:"
                  " setting unlikely BlueFuzz value %ld to default (1)\n",
                  priv->blue_fuzz ));
      priv->blue_fuzz = 1;
    }

  Exit:
    /* clean up */
    cff_blend_clear( subfont ); /* clear blend stack */
    cff_parser_done( &parser ); /* free parser stack */

  Exit2:
    /* no clean up (parser not initialized) */
    return error;
  }


  /* There are 3 ways to call this function, distinguished by code.  */
  /*                                                                 */
  /* . CFF_CODE_TOPDICT for either a CFF Top DICT or a CFF Font DICT */
  /* . CFF2_CODE_TOPDICT for CFF2 Top DICT                           */
  /* . CFF2_CODE_FONTDICT for CFF2 Font DICT                         */

  static FT_Error
  cff_subfont_load( CFF_SubFont  subfont,
                    CFF_Index    idx,
                    FT_UInt      font_index,
                    FT_Stream    stream,
                    FT_ULong     base_offset,
                    FT_UInt      code,
                    CFF_Font     font,
                    CFF_Face     face )
  {
    FT_Error         error;
    CFF_ParserRec    parser;
    FT_Byte*         dict = NULL;
    FT_ULong         dict_len;
    CFF_FontRecDict  top  = &subfont->font_dict;
    CFF_Private      priv = &subfont->private_dict;

    PSAux_Service  psaux = (PSAux_Service)face->psaux;

    FT_Bool  cff2      = FT_BOOL( code == CFF2_CODE_TOPDICT  ||
                                  code == CFF2_CODE_FONTDICT );
    FT_UInt  stackSize = cff2 ? CFF2_DEFAULT_STACK
                              : CFF_MAX_STACK_DEPTH;


    /* Note: We use default stack size for CFF2 Font DICT because        */
    /*       Top and Font DICTs are not allowed to have blend operators. */
    error = cff_parser_init( &parser,
                             code,
                             &subfont->font_dict,
                             font->library,
                             stackSize,
                             0,
                             0 );
    if ( error )
      goto Exit;

    /* set defaults */
    FT_ZERO( top );

    top->underline_position  = -( 100L << 16 );
    top->underline_thickness = 50L << 16;
    top->charstring_type     = 2;
    top->font_matrix.xx      = 0x10000L;
    top->font_matrix.yy      = 0x10000L;
    top->cid_count           = 8720;

    /* we use the implementation specific SID value 0xFFFF to indicate */
    /* missing entries                                                 */
    top->version             = 0xFFFFU;
    top->notice              = 0xFFFFU;
    top->copyright           = 0xFFFFU;
    top->full_name           = 0xFFFFU;
    top->family_name         = 0xFFFFU;
    top->weight              = 0xFFFFU;
    top->embedded_postscript = 0xFFFFU;

    top->cid_registry        = 0xFFFFU;
    top->cid_ordering        = 0xFFFFU;
    top->cid_font_name       = 0xFFFFU;

    /* set default stack size */
    top->maxstack            = cff2 ? CFF2_DEFAULT_STACK : 48;

    if ( idx->count )   /* count is nonzero for a real index */
      error = cff_index_access_element( idx, font_index, &dict, &dict_len );
    else
    {
      /* CFF2 has a fake top dict index;     */
      /* simulate `cff_index_access_element' */

      /* Note: macros implicitly use `stream' and set `error' */
      if ( FT_STREAM_SEEK( idx->data_offset )       ||
           FT_FRAME_EXTRACT( idx->data_size, dict ) )
        goto Exit;

      dict_len = idx->data_size;
    }

    if ( !error )
    {
      FT_TRACE4(( " top dictionary:\n" ));
      error = cff_parser_run( &parser, dict, FT_OFFSET( dict, dict_len ) );
    }

    /* clean up regardless of error */
    if ( idx->count )
      cff_index_forget_element( idx, &dict );
    else
      FT_FRAME_RELEASE( dict );

    if ( error )
      goto Exit;

    /* if it is a CID font, we stop there */
    if ( top->cid_registry != 0xFFFFU )
      goto Exit;

    /* Parse the private dictionary, if any.                   */
    /*                                                         */
    /* CFF2 does not have a private dictionary in the Top DICT */
    /* but may have one in a Font DICT.  We need to parse      */
    /* the latter here in order to load any local subrs.       */
    error = cff_load_private_dict( font, subfont, 0, 0 );
    if ( error )
      goto Exit;

    if ( !cff2 )
    {
      /*
       * Initialize the random number generator.
       *
       * - If we have a face-specific seed, use it.
       *   If non-zero, update it to a positive value.
       *
       * - Otherwise, use the seed from the CFF driver.
       *   If non-zero, update it to a positive value.
       *
       * - If the random value is zero, use the seed given by the subfont's
       *   `initialRandomSeed' value.
       *
       */
      if ( face->root.internal->random_seed == -1 )
      {
        PS_Driver  driver = (PS_Driver)FT_FACE_DRIVER( face );


        subfont->random = (FT_UInt32)driver->random_seed;
        if ( driver->random_seed )
        {
          do
          {
            driver->random_seed =
              (FT_Int32)psaux->cff_random( (FT_UInt32)driver->random_seed );

          } while ( driver->random_seed < 0 );
        }
      }
      else
      {
        subfont->random = (FT_UInt32)face->root.internal->random_seed;
        if ( face->root.internal->random_seed )
        {
          do
          {
            face->root.internal->random_seed =
              (FT_Int32)psaux->cff_random(
                (FT_UInt32)face->root.internal->random_seed );

          } while ( face->root.internal->random_seed < 0 );
        }
      }

      if ( !subfont->random )
        subfont->random = (FT_UInt32)priv->initial_random_seed;
    }

    /* read the local subrs, if any */
    if ( priv->local_subrs_offset )
    {
      if ( FT_STREAM_SEEK( base_offset + top->private_offset +
                           priv->local_subrs_offset ) )
        goto Exit;

      error = cff_index_init( &subfont->local_subrs_index, stream, 1, cff2 );
      if ( error )
        goto Exit;

      error = cff_index_get_pointers( &subfont->local_subrs_index,
                                      &subfont->local_subrs, NULL, NULL );
      if ( error )
        goto Exit;
    }

  Exit:
    cff_parser_done( &parser ); /* free parser stack */

    return error;
  }


  static void
  cff_subfont_done( FT_Memory    memory,
                    CFF_SubFont  subfont )
  {
    if ( subfont )
    {
      cff_index_done( &subfont->local_subrs_index );
      FT_FREE( subfont->local_subrs );

      FT_FREE( subfont->blend.lastNDV );
      FT_FREE( subfont->blend.BV );
      FT_FREE( subfont->blend_stack );
    }
  }


  FT_LOCAL_DEF( FT_Error )
  cff_font_load( FT_Library library,
                 FT_Stream  stream,
                 FT_Int     face_index,
                 CFF_Font   font,
                 CFF_Face   face,
                 FT_Bool    pure_cff,
                 FT_Bool    cff2 )
  {
    static const FT_Frame_Field  cff_header_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  CFF_FontRec

      FT_FRAME_START( 3 ),
        FT_FRAME_BYTE( version_major ),
        FT_FRAME_BYTE( version_minor ),
        FT_FRAME_BYTE( header_size ),
      FT_FRAME_END
    };

    FT_Error         error;
    FT_Memory        memory = stream->memory;
    FT_ULong         base_offset;
    CFF_FontRecDict  dict;
    CFF_IndexRec     string_index;
    FT_UInt          subfont_index;


    FT_ZERO( font );
    FT_ZERO( &string_index );

    dict        = &font->top_font.font_dict;
    base_offset = FT_STREAM_POS();

    font->library     = library;
    font->stream      = stream;
    font->memory      = memory;
    font->cff2        = cff2;
    font->base_offset = base_offset;

    /* read CFF font header */
    if ( FT_STREAM_READ_FIELDS( cff_header_fields, font ) )
      goto Exit;

    if ( cff2 )
    {
      if ( font->version_major != 2 ||
           font->header_size < 5    )
      {
        FT_TRACE2(( "  not a CFF2 font header\n" ));
        error = FT_THROW( Unknown_File_Format );
        goto Exit;
      }

      if ( FT_READ_USHORT( font->top_dict_length ) )
        goto Exit;
    }
    else
    {
      FT_Byte  absolute_offset;


      if ( FT_READ_BYTE( absolute_offset ) )
        goto Exit;

      if ( font->version_major != 1 ||
           font->header_size < 4    ||
           absolute_offset > 4      )
      {
        FT_TRACE2(( "  not a CFF font header\n" ));
        error = FT_THROW( Unknown_File_Format );
        goto Exit;
      }
    }

    /* skip the rest of the header */
    if ( FT_STREAM_SEEK( base_offset + font->header_size ) )
    {
      /* For pure CFFs we have read only four bytes so far.  Contrary to */
      /* other formats like SFNT those bytes doesn't define a signature; */
      /* it is thus possible that the font isn't a CFF at all.           */
      if ( pure_cff )
      {
        FT_TRACE2(( "  not a CFF file\n" ));
        error = FT_THROW( Unknown_File_Format );
      }
      goto Exit;
    }

    if ( cff2 )
    {
      /* For CFF2, the top dict data immediately follow the header    */
      /* and the length is stored in the header `offSize' field;      */
      /* there is no index for it.                                    */
      /*                                                              */
      /* Use the `font_dict_index' to save the current position       */
      /* and length of data, but leave count at zero as an indicator. */
      FT_ZERO( &font->font_dict_index );

      font->font_dict_index.data_offset = FT_STREAM_POS();
      font->font_dict_index.data_size   = font->top_dict_length;

      /* skip the top dict data for now, we will parse it later */
      if ( FT_STREAM_SKIP( font->top_dict_length ) )
        goto Exit;

      /* next, read the global subrs index */
      if ( FT_SET_ERROR( cff_index_init( &font->global_subrs_index,
                                         stream, 1, cff2 ) ) )
        goto Exit;
    }
    else
    {
      /* for CFF, read the name, top dict, string and global subrs index */
      if ( FT_SET_ERROR( cff_index_init( &font->name_index,
                                         stream, 0, cff2 ) ) )
      {
        if ( pure_cff )
        {
          FT_TRACE2(( "  not a CFF file\n" ));
          error = FT_THROW( Unknown_File_Format );
        }
        goto Exit;
      }

      /* if we have an empty font name,      */
      /* it must be the only font in the CFF */
      if ( font->name_index.count > 1                          &&
           font->name_index.data_size < font->name_index.count )
      {
        /* for pure CFFs, we still haven't checked enough bytes */
        /* to be sure that it is a CFF at all                   */
        error = pure_cff ? FT_THROW( Unknown_File_Format )
                         : FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( FT_SET_ERROR( cff_index_init( &font->font_dict_index,
                                         stream, 0, cff2 ) )                 ||
           FT_SET_ERROR( cff_index_init( &string_index,
                                         stream, 1, cff2 ) )                 ||
           FT_SET_ERROR( cff_index_init( &font->global_subrs_index,
                                         stream, 1, cff2 ) )                 ||
           FT_SET_ERROR( cff_index_get_pointers( &string_index,
                                                 &font->strings,
                                                 &font->string_pool,
                                                 &font->string_pool_size ) ) )
        goto Exit;

      /* there must be a Top DICT index entry for each name index entry */
      if ( font->name_index.count > font->font_dict_index.count )
      {
        FT_ERROR(( "cff_font_load:"
                   " not enough entries in Top DICT index\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }

    font->num_strings = string_index.count;

    if ( pure_cff )
    {
      /* well, we don't really forget the `disabled' fonts... */
      subfont_index = (FT_UInt)( face_index & 0xFFFF );

      if ( face_index > 0 && subfont_index >= font->name_index.count )
      {
        FT_ERROR(( "cff_font_load:"
                   " invalid subfont index for pure CFF font (%d)\n",
                   subfont_index ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      font->num_faces = font->name_index.count;
    }
    else
    {
      subfont_index = 0;

      if ( font->name_index.count > 1 )
      {
        FT_ERROR(( "cff_font_load:"
                   " invalid CFF font with multiple subfonts\n" ));
        FT_ERROR(( "              "
                   " in SFNT wrapper\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }

    /* in case of a font format check, simply exit now */
    if ( face_index < 0 )
      goto Exit;

    /* now, parse the top-level font dictionary */
    FT_TRACE4(( "parsing top-level\n" ));
    error = cff_subfont_load( &font->top_font,
                              &font->font_dict_index,
                              subfont_index,
                              stream,
                              base_offset,
                              cff2 ? CFF2_CODE_TOPDICT : CFF_CODE_TOPDICT,
                              font,
                              face );
    if ( error )
      goto Exit;

    if ( FT_STREAM_SEEK( base_offset + dict->charstrings_offset ) )
      goto Exit;

    error = cff_index_init( &font->charstrings_index, stream, 0, cff2 );
    if ( error )
      goto Exit;

    /* now, check for a CID or CFF2 font */
    if ( dict->cid_registry != 0xFFFFU ||
         cff2                          )
    {
      CFF_IndexRec  fd_index;
      CFF_SubFont   sub = NULL;
      FT_UInt       idx;


      /* for CFF2, read the Variation Store if available;                 */
      /* this must follow the Top DICT parse and precede any Private DICT */
      error = cff_vstore_load( &font->vstore,
                               stream,
                               base_offset,
                               dict->vstore_offset );
      if ( error )
        goto Exit;

      /* this is a CID-keyed font, we must now allocate a table of */
      /* sub-fonts, then load each of them separately              */
      if ( FT_STREAM_SEEK( base_offset + dict->cid_fd_array_offset ) )
        goto Exit;

      error = cff_index_init( &fd_index, stream, 0, cff2 );
      if ( error )
        goto Exit;

      /* Font Dicts are not limited to 256 for CFF2. */
      /* TODO: support this for CFF2                 */
      if ( fd_index.count > CFF_MAX_CID_FONTS )
      {
        FT_TRACE0(( "cff_font_load: FD array too large in CID font\n" ));
        goto Fail_CID;
      }

      /* allocate & read each font dict independently */
      font->num_subfonts = fd_index.count;
      if ( FT_NEW_ARRAY( sub, fd_index.count ) )
        goto Fail_CID;

      /* set up pointer table */
      for ( idx = 0; idx < fd_index.count; idx++ )
        font->subfonts[idx] = sub + idx;

      /* now load each subfont independently */
      for ( idx = 0; idx < fd_index.count; idx++ )
      {
        sub = font->subfonts[idx];
        FT_TRACE4(( "parsing subfont %u\n", idx ));
        error = cff_subfont_load( sub,
                                  &fd_index,
                                  idx,
                                  stream,
                                  base_offset,
                                  cff2 ? CFF2_CODE_FONTDICT
                                       : CFF_CODE_TOPDICT,
                                  font,
                                  face );
        if ( error )
          goto Fail_CID;
      }

      /* now load the FD Select array;               */
      /* CFF2 omits FDSelect if there is only one FD */
      if ( !cff2 || fd_index.count > 1 )
        error = CFF_Load_FD_Select( &font->fd_select,
                                    font->charstrings_index.count,
                                    stream,
                                    base_offset + dict->cid_fd_select_offset );

    Fail_CID:
      cff_index_done( &fd_index );

      if ( error )
        goto Exit;
    }
    else
      font->num_subfonts = 0;

    /* read the charstrings index now */
    if ( dict->charstrings_offset == 0 )
    {
      FT_ERROR(( "cff_font_load: no charstrings offset\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    font->num_glyphs = font->charstrings_index.count;

    error = cff_index_get_pointers( &font->global_subrs_index,
                                    &font->global_subrs, NULL, NULL );

    if ( error )
      goto Exit;

    /* read the Charset and Encoding tables if available */
    if ( !cff2 && font->num_glyphs > 0 )
    {
      FT_Bool  invert = FT_BOOL( dict->cid_registry != 0xFFFFU && pure_cff );


      error = cff_charset_load( &font->charset, font->num_glyphs, stream,
                                base_offset, dict->charset_offset, invert );
      if ( error )
        goto Exit;

      /* CID-keyed CFFs don't have an encoding */
      if ( dict->cid_registry == 0xFFFFU )
      {
        error = cff_encoding_load( &font->encoding,
                                   &font->charset,
                                   font->num_glyphs,
                                   stream,
                                   base_offset,
                                   dict->encoding_offset );
        if ( error )
          goto Exit;
      }
    }

    /* get the font name (/CIDFontName for CID-keyed fonts, */
    /* /FontName otherwise)                                 */
    font->font_name = cff_index_get_name( font, subfont_index );

  Exit:
    cff_index_done( &string_index );

    return error;
  }


  FT_LOCAL_DEF( void )
  cff_font_done( CFF_Font  font )
  {
    FT_Memory  memory = font->memory;
    FT_UInt    idx;


    cff_index_done( &font->global_subrs_index );
    cff_index_done( &font->font_dict_index );
    cff_index_done( &font->name_index );
    cff_index_done( &font->charstrings_index );

    /* release font dictionaries, but only if working with */
    /* a CID keyed CFF font or a CFF2 font                 */
    if ( font->num_subfonts > 0 )
    {
      for ( idx = 0; idx < font->num_subfonts; idx++ )
        cff_subfont_done( memory, font->subfonts[idx] );

      /* the subfonts array has been allocated as a single block */
      FT_FREE( font->subfonts[0] );
    }

    cff_encoding_done( &font->encoding );
    cff_charset_done( &font->charset, font->stream );
    cff_vstore_done( &font->vstore, memory );

    cff_subfont_done( memory, &font->top_font );

    CFF_Done_FD_Select( &font->fd_select, font->stream );

    FT_FREE( font->font_info );

    FT_FREE( font->font_name );
    FT_FREE( font->global_subrs );
    FT_FREE( font->strings );
    FT_FREE( font->string_pool );

    if ( font->cf2_instance.finalizer )
    {
      font->cf2_instance.finalizer( font->cf2_instance.data );
      FT_FREE( font->cf2_instance.data );
    }

    FT_FREE( font->font_extra );
  }


/* END */
