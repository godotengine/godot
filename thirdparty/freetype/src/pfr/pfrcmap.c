/****************************************************************************
 *
 * pfrcmap.c
 *
 *   FreeType PFR cmap handling (body).
 *
 * Copyright (C) 2002-2024 by
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
#include "pfrcmap.h"
#include "pfrobjs.h"

#include "pfrerror.h"


  FT_CALLBACK_DEF( FT_Error )
  pfr_cmap_init( FT_CMap     cmap,     /* PFR_CMap */
                 FT_Pointer  pointer )
  {
    PFR_CMap  pfrcmap = (PFR_CMap)cmap;
    FT_Error  error   = FT_Err_Ok;
    PFR_Face  face    = (PFR_Face)FT_CMAP_FACE( cmap );

    FT_UNUSED( pointer );


    pfrcmap->num_chars = face->phy_font.num_chars;
    pfrcmap->chars     = face->phy_font.chars;

    /* just for safety, check that the character entries are correctly */
    /* sorted in increasing character code order                       */
    {
      FT_UInt  n;


      for ( n = 1; n < pfrcmap->num_chars; n++ )
      {
        if ( pfrcmap->chars[n - 1].char_code >= pfrcmap->chars[n].char_code )
        {
          error = FT_THROW( Invalid_Table );
          goto Exit;
        }
      }
    }

  Exit:
    return error;
  }


  FT_CALLBACK_DEF( void )
  pfr_cmap_done( FT_CMap  cmap )    /* PFR_CMap */
  {
    PFR_CMap  pfrcmap = (PFR_CMap)cmap;


    pfrcmap->chars     = NULL;
    pfrcmap->num_chars = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  pfr_cmap_char_index( FT_CMap    cmap,       /* PFR_CMap */
                       FT_UInt32  char_code )
  {
    PFR_CMap  pfrcmap = (PFR_CMap)cmap;
    FT_UInt   min     = 0;
    FT_UInt   max     = pfrcmap->num_chars;
    FT_UInt   mid     = min + ( max - min ) / 2;
    PFR_Char  gchar;


    while ( min < max )
    {
      gchar = pfrcmap->chars + mid;

      if ( gchar->char_code == char_code )
        return mid + 1;

      if ( gchar->char_code < char_code )
        min = mid + 1;
      else
        max = mid;

      /* reasonable prediction in a continuous block */
      mid += char_code - gchar->char_code;
      if ( mid >= max || mid < min )
        mid = min + ( max - min ) / 2;
    }
    return 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  pfr_cmap_char_next( FT_CMap     cmap,        /* PFR_CMap */
                      FT_UInt32  *pchar_code )
  {
    PFR_CMap   pfrcmap   = (PFR_CMap)cmap;
    FT_UInt    result    = 0;
    FT_UInt32  char_code = *pchar_code + 1;


  Restart:
    {
      FT_UInt   min = 0;
      FT_UInt   max = pfrcmap->num_chars;
      FT_UInt   mid = min + ( max - min ) / 2;
      PFR_Char  gchar;


      while ( min < max )
      {
        gchar = pfrcmap->chars + mid;

        if ( gchar->char_code == char_code )
        {
          result = mid;
          if ( result != 0 )
          {
            result++;
            goto Exit;
          }

          char_code++;
          goto Restart;
        }

        if ( gchar->char_code < char_code )
          min = mid + 1;
        else
          max = mid;

        /* reasonable prediction in a continuous block */
        mid += char_code - gchar->char_code;
        if ( mid >= max || mid < min )
          mid = min + ( max - min ) / 2;
      }

      /* we didn't find it, but we have a pair just above it */
      char_code = 0;

      if ( min < pfrcmap->num_chars )
      {
        gchar  = pfrcmap->chars + min;
        result = min;
        if ( result != 0 )
        {
          result++;
          char_code = gchar->char_code;
        }
      }
    }

  Exit:
    *pchar_code = char_code;
    return result;
  }


  FT_CALLBACK_TABLE_DEF const FT_CMap_ClassRec
  pfr_cmap_class_rec =
  {
    sizeof ( PFR_CMapRec ),

    (FT_CMap_InitFunc)     pfr_cmap_init,        /* init       */
    (FT_CMap_DoneFunc)     pfr_cmap_done,        /* done       */
    (FT_CMap_CharIndexFunc)pfr_cmap_char_index,  /* char_index */
    (FT_CMap_CharNextFunc) pfr_cmap_char_next,   /* char_next  */

    (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
    (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
    (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
    (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
    (FT_CMap_VariantCharListFunc) NULL   /* variantchar_list */
  };


/* END */
