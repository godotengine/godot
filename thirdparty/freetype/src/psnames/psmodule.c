/****************************************************************************
 *
 * psmodule.c
 *
 *   psnames module implementation (body).
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
#include <freetype/internal/services/svpscmap.h>

#include "psmodule.h"

  /*
   * The file `pstables.h' with its arrays and its function
   * `ft_get_adobe_glyph_index' is useful for other projects also (for
   * example, `pdfium' is using it).  However, if used as a C++ header,
   * including it in two different source files makes it necessary to use
   * `extern const' for the declaration of its arrays, otherwise the data
   * would be duplicated as mandated by the C++ standard.
   *
   * For this reason, we use `DEFINE_PS_TABLES' to guard the function
   * definitions, and `DEFINE_PS_TABLES_DATA' to provide both proper array
   * declarations and definitions.
   */
#include "pstables.h"
#define  DEFINE_PS_TABLES
#define  DEFINE_PS_TABLES_DATA
#include "pstables.h"

#include "psnamerr.h"


#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES


#ifdef FT_CONFIG_OPTION_ADOBE_GLYPH_LIST


#define VARIANT_BIT         0x80000000UL
#define BASE_GLYPH( code )  ( (FT_UInt32)( (code) & ~VARIANT_BIT ) )


  /* Return the Unicode value corresponding to a given glyph.  Note that */
  /* we do deal with glyph variants by detecting a non-initial dot in    */
  /* the name, as in `A.swash' or `e.final'; in this case, the           */
  /* VARIANT_BIT is set in the return value.                             */
  /*                                                                     */
  static FT_UInt32
  ps_unicode_value( const char*  glyph_name )
  {
    /* If the name begins with `uni', then the glyph name may be a */
    /* hard-coded unicode character code.                          */
    if ( glyph_name[0] == 'u' &&
         glyph_name[1] == 'n' &&
         glyph_name[2] == 'i' )
    {
      /* determine whether the next four characters following are */
      /* hexadecimal.                                             */

      /* XXX: Add code to deal with ligatures, i.e. glyph names like */
      /*      `uniXXXXYYYYZZZZ'...                                   */

      FT_Int       count;
      FT_UInt32    value = 0;
      const char*  p     = glyph_name + 3;


      for ( count = 4; count > 0; count--, p++ )
      {
        char          c = *p;
        unsigned int  d;


        d = (unsigned char)c - '0';
        if ( d >= 10 )
        {
          d = (unsigned char)c - 'A';
          if ( d >= 6 )
            d = 16;
          else
            d += 10;
        }

        /* Exit if a non-uppercase hexadecimal character was found   */
        /* -- this also catches character codes below `0' since such */
        /* negative numbers cast to `unsigned int' are far too big.  */
        if ( d >= 16 )
          break;

        value = ( value << 4 ) + d;
      }

      /* there must be exactly four hex digits */
      if ( count == 0 )
      {
        if ( *p == '\0' )
          return value;
        if ( *p == '.' )
          return (FT_UInt32)( value | VARIANT_BIT );
      }
    }

    /* If the name begins with `u', followed by four to six uppercase */
    /* hexadecimal digits, it is a hard-coded unicode character code. */
    if ( glyph_name[0] == 'u' )
    {
      FT_Int       count;
      FT_UInt32    value = 0;
      const char*  p     = glyph_name + 1;


      for ( count = 6; count > 0; count--, p++ )
      {
        char          c = *p;
        unsigned int  d;


        d = (unsigned char)c - '0';
        if ( d >= 10 )
        {
          d = (unsigned char)c - 'A';
          if ( d >= 6 )
            d = 16;
          else
            d += 10;
        }

        if ( d >= 16 )
          break;

        value = ( value << 4 ) + d;
      }

      if ( count <= 2 )
      {
        if ( *p == '\0' )
          return value;
        if ( *p == '.' )
          return (FT_UInt32)( value | VARIANT_BIT );
      }
    }

    /* Look for a non-initial dot in the glyph name in order to */
    /* find variants like `A.swash', `e.final', etc.            */
    {
      FT_UInt32    value = 0;
      const char*  p     = glyph_name;


      for ( ; *p && *p != '.'; p++ )
        ;

      /* now look up the glyph in the Adobe Glyph List;      */
      /* `.notdef', `.null' and the empty name are short cut */
      if ( p > glyph_name )
      {
        value = (FT_UInt32)ft_get_adobe_glyph_index( glyph_name, p );

        if ( *p == '.' )
          value |= (FT_UInt32)VARIANT_BIT;
      }

      return value;
    }
  }


  /* ft_qsort callback to sort the unicode map */
  FT_COMPARE_DEF( int )
  compare_uni_maps( const void*  a,
                    const void*  b )
  {
    PS_UniMap*  map1 = (PS_UniMap*)a;
    PS_UniMap*  map2 = (PS_UniMap*)b;
    FT_UInt32   unicode1 = BASE_GLYPH( map1->unicode );
    FT_UInt32   unicode2 = BASE_GLYPH( map2->unicode );


    /* sort base glyphs before glyph variants */
    if ( unicode1 == unicode2 )
    {
      if ( map1->unicode > map2->unicode )
        return 1;
      else if ( map1->unicode < map2->unicode )
        return -1;
      else
        return 0;
    }
    else
    {
      if ( unicode1 > unicode2 )
        return 1;
      else if ( unicode1 < unicode2 )
        return -1;
      else
        return 0;
    }
  }


  /* support for extra glyphs not handled (well) in AGL; */
  /* we add extra mappings for them if necessary         */

#define EXTRA_GLYPH_LIST_SIZE  10

  static const FT_UInt32  ft_extra_glyph_unicodes[EXTRA_GLYPH_LIST_SIZE] =
  {
    /* WGL 4 */
    0x0394,
    0x03A9,
    0x2215,
    0x00AD,
    0x02C9,
    0x03BC,
    0x2219,
    0x00A0,
    /* Romanian */
    0x021A,
    0x021B
  };

  static const char  ft_extra_glyph_names[] =
  {
    'D','e','l','t','a',0,
    'O','m','e','g','a',0,
    'f','r','a','c','t','i','o','n',0,
    'h','y','p','h','e','n',0,
    'm','a','c','r','o','n',0,
    'm','u',0,
    'p','e','r','i','o','d','c','e','n','t','e','r','e','d',0,
    's','p','a','c','e',0,
    'T','c','o','m','m','a','a','c','c','e','n','t',0,
    't','c','o','m','m','a','a','c','c','e','n','t',0
  };

  static const FT_Int
  ft_extra_glyph_name_offsets[EXTRA_GLYPH_LIST_SIZE] =
  {
     0,
     6,
    12,
    21,
    28,
    35,
    38,
    53,
    59,
    72
  };


  static void
  ps_check_extra_glyph_name( const char*  gname,
                             FT_UInt      glyph,
                             FT_UInt*     extra_glyphs,
                             FT_UInt     *states )
  {
    FT_UInt  n;


    for ( n = 0; n < EXTRA_GLYPH_LIST_SIZE; n++ )
    {
      if ( ft_strcmp( ft_extra_glyph_names +
                        ft_extra_glyph_name_offsets[n], gname ) == 0 )
      {
        if ( states[n] == 0 )
        {
          /* mark this extra glyph as a candidate for the cmap */
          states[n]     = 1;
          extra_glyphs[n] = glyph;
        }

        return;
      }
    }
  }


  static void
  ps_check_extra_glyph_unicode( FT_UInt32  uni_char,
                                FT_UInt   *states )
  {
    FT_UInt  n;


    for ( n = 0; n < EXTRA_GLYPH_LIST_SIZE; n++ )
    {
      if ( uni_char == ft_extra_glyph_unicodes[n] )
      {
        /* disable this extra glyph from being added to the cmap */
        states[n] = 2;

        return;
      }
    }
  }


  /* Build a table that maps Unicode values to glyph indices. */
  static FT_Error
  ps_unicodes_init( FT_Memory             memory,
                    PS_Unicodes           table,
                    FT_UInt               num_glyphs,
                    PS_GetGlyphNameFunc   get_glyph_name,
                    PS_FreeGlyphNameFunc  free_glyph_name,
                    FT_Pointer            glyph_data )
  {
    FT_Error  error;

    FT_UInt  extra_glyph_list_states[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    FT_UInt  extra_glyphs[EXTRA_GLYPH_LIST_SIZE];


    /* we first allocate the table */
    table->num_maps = 0;

    if ( !FT_QNEW_ARRAY( table->maps, num_glyphs + EXTRA_GLYPH_LIST_SIZE ) )
    {
      FT_UInt     n;
      FT_UInt     count;
      PS_UniMap*  map;
      FT_UInt32   uni_char;


      map = table->maps;

      for ( n = 0; n < num_glyphs; n++ )
      {
        const char*  gname = get_glyph_name( glyph_data, n );


        if ( gname && *gname )
        {
          ps_check_extra_glyph_name( gname, n,
                                     extra_glyphs, extra_glyph_list_states );
          uni_char = ps_unicode_value( gname );

          if ( BASE_GLYPH( uni_char ) != 0 )
          {
            ps_check_extra_glyph_unicode( uni_char,
                                          extra_glyph_list_states );
            map->unicode     = uni_char;
            map->glyph_index = n;
            map++;
          }

          if ( free_glyph_name )
            free_glyph_name( glyph_data, gname );
        }
      }

      for ( n = 0; n < EXTRA_GLYPH_LIST_SIZE; n++ )
      {
        if ( extra_glyph_list_states[n] == 1 )
        {
          /* This glyph name has an additional representation. */
          /* Add it to the cmap.                               */

          map->unicode     = ft_extra_glyph_unicodes[n];
          map->glyph_index = extra_glyphs[n];
          map++;
        }
      }

      /* now compress the table a bit */
      count = (FT_UInt)( map - table->maps );

      if ( count == 0 )
      {
        /* No unicode chars here! */
        FT_FREE( table->maps );
        if ( !error )
          error = FT_THROW( No_Unicode_Glyph_Name );
      }
      else
      {
        /* Reallocate if the number of used entries is much smaller. */
        if ( count < num_glyphs / 2 )
        {
          FT_MEM_QRENEW_ARRAY( table->maps,
                               num_glyphs + EXTRA_GLYPH_LIST_SIZE,
                               count );
          error = FT_Err_Ok;
        }

        /* Sort the table in increasing order of unicode values, */
        /* taking care of glyph variants.                        */
        ft_qsort( table->maps, count, sizeof ( PS_UniMap ),
                  compare_uni_maps );
      }

      table->num_maps = count;
    }

    return error;
  }


  static FT_UInt
  ps_unicodes_char_index( PS_Unicodes  table,
                          FT_UInt32    unicode )
  {
    PS_UniMap  *result = NULL;
    PS_UniMap  *min = table->maps;
    PS_UniMap  *max = min + table->num_maps;
    PS_UniMap  *mid = min + ( ( max - min ) >> 1 );


    /* Perform a binary search on the table. */
    while ( min < max )
    {
      FT_UInt32  base_glyph;


      if ( mid->unicode == unicode )
      {
        result = mid;
        break;
      }

      base_glyph = BASE_GLYPH( mid->unicode );

      if ( base_glyph == unicode )
        result = mid; /* remember match but continue search for base glyph */

      if ( base_glyph < unicode )
        min = mid + 1;
      else
        max = mid;

      /* reasonable prediction in a continuous block */
      mid += unicode - base_glyph;
      if ( mid >= max || mid < min )
        mid = min + ( ( max - min ) >> 1 );
    }

    if ( result )
      return result->glyph_index;
    else
      return 0;
  }


  static FT_UInt32
  ps_unicodes_char_next( PS_Unicodes  table,
                         FT_UInt32   *unicode )
  {
    FT_UInt    result    = 0;
    FT_UInt32  char_code = *unicode + 1;


    {
      FT_UInt     min = 0;
      FT_UInt     max = table->num_maps;
      FT_UInt     mid = min + ( ( max - min ) >> 1 );
      PS_UniMap*  map;
      FT_UInt32   base_glyph;


      while ( min < max )
      {
        map = table->maps + mid;

        if ( map->unicode == char_code )
        {
          result = map->glyph_index;
          goto Exit;
        }

        base_glyph = BASE_GLYPH( map->unicode );

        if ( base_glyph == char_code )
          result = map->glyph_index;

        if ( base_glyph < char_code )
          min = mid + 1;
        else
          max = mid;

        /* reasonable prediction in a continuous block */
        mid += char_code - base_glyph;
        if ( mid >= max || mid < min )
          mid = min + ( max - min ) / 2;
      }

      if ( result )
        goto Exit;               /* we have a variant glyph */

      /* we didn't find it; check whether we have a map just above it */
      char_code = 0;

      if ( min < table->num_maps )
      {
        map       = table->maps + min;
        result    = map->glyph_index;
        char_code = BASE_GLYPH( map->unicode );
      }
    }

  Exit:
    *unicode = char_code;
    return result;
  }


#endif /* FT_CONFIG_OPTION_ADOBE_GLYPH_LIST */


  static const char*
  ps_get_macintosh_name( FT_UInt  name_index )
  {
    if ( name_index >= FT_NUM_MAC_NAMES )
      name_index = 0;

    return ft_standard_glyph_names + ft_mac_names[name_index];
  }


  static const char*
  ps_get_standard_strings( FT_UInt  sid )
  {
    if ( sid >= FT_NUM_SID_NAMES )
      return 0;

    return ft_standard_glyph_names + ft_sid_names[sid];
  }


#ifdef FT_CONFIG_OPTION_ADOBE_GLYPH_LIST

  FT_DEFINE_SERVICE_PSCMAPSREC(
    pscmaps_interface,

    (PS_Unicode_ValueFunc)     ps_unicode_value,        /* unicode_value         */
    (PS_Unicodes_InitFunc)     ps_unicodes_init,        /* unicodes_init         */
    (PS_Unicodes_CharIndexFunc)ps_unicodes_char_index,  /* unicodes_char_index   */
    (PS_Unicodes_CharNextFunc) ps_unicodes_char_next,   /* unicodes_char_next    */

    (PS_Macintosh_NameFunc)    ps_get_macintosh_name,   /* macintosh_name        */
    (PS_Adobe_Std_StringsFunc) ps_get_standard_strings, /* adobe_std_strings     */

    t1_standard_encoding,                               /* adobe_std_encoding    */
    t1_expert_encoding                                  /* adobe_expert_encoding */
  )

#else

  FT_DEFINE_SERVICE_PSCMAPSREC(
    pscmaps_interface,

    NULL,                                               /* unicode_value         */
    NULL,                                               /* unicodes_init         */
    NULL,                                               /* unicodes_char_index   */
    NULL,                                               /* unicodes_char_next    */

    (PS_Macintosh_NameFunc)    ps_get_macintosh_name,   /* macintosh_name        */
    (PS_Adobe_Std_StringsFunc) ps_get_standard_strings, /* adobe_std_strings     */

    t1_standard_encoding,                               /* adobe_std_encoding    */
    t1_expert_encoding                                  /* adobe_expert_encoding */
  )

#endif /* FT_CONFIG_OPTION_ADOBE_GLYPH_LIST */


  FT_DEFINE_SERVICEDESCREC1(
    pscmaps_services,

    FT_SERVICE_ID_POSTSCRIPT_CMAPS, &pscmaps_interface )


  static FT_Pointer
  psnames_get_service( FT_Module    module,
                       const char*  service_id )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( pscmaps_services, service_id );
  }

#endif /* FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


#ifndef FT_CONFIG_OPTION_POSTSCRIPT_NAMES
#define PUT_PS_NAMES_SERVICE( a )  NULL
#else
#define PUT_PS_NAMES_SERVICE( a )  a
#endif

  FT_DEFINE_MODULE(
    psnames_module_class,

    0,  /* this is not a font driver, nor a renderer */
    sizeof ( FT_ModuleRec ),

    "psnames",  /* driver name                         */
    0x10000L,   /* driver version                      */
    0x20000L,   /* driver requires FreeType 2 or above */

    PUT_PS_NAMES_SERVICE(
      (void*)&pscmaps_interface ),   /* module specific interface */

    (FT_Module_Constructor)NULL,                                       /* module_init   */
    (FT_Module_Destructor) NULL,                                       /* module_done   */
    (FT_Module_Requester)  PUT_PS_NAMES_SERVICE( psnames_get_service ) /* get_interface */
  )


/* END */
