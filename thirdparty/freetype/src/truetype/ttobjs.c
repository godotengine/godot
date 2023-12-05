/****************************************************************************
 *
 * ttobjs.c
 *
 *   Objects manager (body).
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
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include <freetype/internal/sfnt.h>
#include <freetype/ftdriver.h>

#include "ttgload.h"
#include "ttpload.h"

#include "tterrors.h"

#ifdef TT_USE_BYTECODE_INTERPRETER
#include "ttinterp.h"
#endif

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif

  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttobjs


#ifdef TT_USE_BYTECODE_INTERPRETER

  /**************************************************************************
   *
   *                      GLYPH ZONE FUNCTIONS
   *
   */


  /**************************************************************************
   *
   * @Function:
   *   tt_glyphzone_done
   *
   * @Description:
   *   Deallocate a glyph zone.
   *
   * @Input:
   *   zone ::
   *     A pointer to the target glyph zone.
   */
  FT_LOCAL_DEF( void )
  tt_glyphzone_done( TT_GlyphZone  zone )
  {
    FT_Memory  memory = zone->memory;


    if ( memory )
    {
      FT_FREE( zone->contours );
      FT_FREE( zone->tags );
      FT_FREE( zone->cur );
      FT_FREE( zone->org );
      FT_FREE( zone->orus );

      zone->max_points   = zone->n_points   = 0;
      zone->max_contours = zone->n_contours = 0;
      zone->memory       = NULL;
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_glyphzone_new
   *
   * @Description:
   *   Allocate a new glyph zone.
   *
   * @Input:
   *   memory ::
   *     A handle to the current memory object.
   *
   *   maxPoints ::
   *     The capacity of glyph zone in points.
   *
   *   maxContours ::
   *     The capacity of glyph zone in contours.
   *
   * @Output:
   *   zone ::
   *     A pointer to the target glyph zone record.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_glyphzone_new( FT_Memory     memory,
                    FT_UShort     maxPoints,
                    FT_Short      maxContours,
                    TT_GlyphZone  zone )
  {
    FT_Error  error;


    FT_ZERO( zone );
    zone->memory = memory;

    if ( FT_NEW_ARRAY( zone->org,      maxPoints   ) ||
         FT_NEW_ARRAY( zone->cur,      maxPoints   ) ||
         FT_NEW_ARRAY( zone->orus,     maxPoints   ) ||
         FT_NEW_ARRAY( zone->tags,     maxPoints   ) ||
         FT_NEW_ARRAY( zone->contours, maxContours ) )
    {
      tt_glyphzone_done( zone );
    }
    else
    {
      zone->max_points   = maxPoints;
      zone->max_contours = maxContours;
    }

    return error;
  }


  /*
   * Fonts embedded in PDFs are made unique by prepending randomization
   * prefixes to their names: as defined in Section 5.5.3, 'Font Subsets',
   * of the PDF Reference, they consist of 6 uppercase letters followed by
   * the `+` sign.  For safety, we do not skip prefixes violating this rule.
   */

  static const FT_String*
  tt_skip_pdffont_random_tag( const FT_String*  name )
  {
    unsigned int  i;


    if ( ft_strlen( name ) < 8 || name[6] != '+' )
      return name;

    for ( i = 0; i < 6; i++ )
      if ( !ft_isupper( name[i] ) )
        return name;

    FT_TRACE7(( "name without randomization tag: %s\n", name + 7 ));
    return name + 7;
  }


  /* Compare the face with a list of well-known `tricky' fonts. */
  /* This list shall be expanded as we find more of them.       */

  static FT_Bool
  tt_check_trickyness_family( const FT_String*  name )
  {

#define TRICK_NAMES_MAX_CHARACTERS  19
#define TRICK_NAMES_COUNT           20

    static const char trick_names[TRICK_NAMES_COUNT]
                                 [TRICK_NAMES_MAX_CHARACTERS + 1] =
    {
      /*
         PostScript names are given in brackets if they differ from the
         family name.  The version numbers, together with the copyright or
         release year data, are taken from fonts available to the
         developers.

         Note that later versions of the fonts might be no longer tricky;
         for example, `MingLiU' version 7.00 (file `mingliu.ttc' from
         Windows 7) is an ordinary TTC with non-tricky subfonts.
       */

      "cpop",               /* dftt-p7.ttf; version 1.00, 1992 [DLJGyShoMedium] */
      "DFGirl-W6-WIN-BF",   /* dftt-h6.ttf; version 1.00, 1993 */
      "DFGothic-EB",        /* DynaLab Inc. 1992-1995 */
      "DFGyoSho-Lt",        /* DynaLab Inc. 1992-1995 */
      "DFHei",              /* DynaLab Inc. 1992-1995 [DFHei-Bd-WIN-HK-BF] */
                            /* covers "DFHei-Md-HK-BF", maybe DynaLab Inc. */

      "DFHSGothic-W5",      /* DynaLab Inc. 1992-1995 */
      "DFHSMincho-W3",      /* DynaLab Inc. 1992-1995 */
      "DFHSMincho-W7",      /* DynaLab Inc. 1992-1995 */
      "DFKaiSho-SB",        /* dfkaisb.ttf */
      "DFKaiShu",           /* covers "DFKaiShu-Md-HK-BF", maybe DynaLab Inc. */
      "DFKai-SB",           /* kaiu.ttf; version 3.00, 1998 [DFKaiShu-SB-Estd-BF] */

      "DFMing",             /* DynaLab Inc. 1992-1995 [DFMing-Md-WIN-HK-BF] */
                            /* covers "DFMing-Bd-HK-BF", maybe DynaLab Inc. */

      "DLC",                /* dftt-m7.ttf; version 1.00, 1993 [DLCMingBold] */
                            /* dftt-f5.ttf; version 1.00, 1993 [DLCFongSung] */
                            /* covers following */
                            /* "DLCHayMedium", dftt-b5.ttf; version 1.00, 1993 */
                            /* "DLCHayBold",   dftt-b7.ttf; version 1.00, 1993 */
                            /* "DLCKaiMedium", dftt-k5.ttf; version 1.00, 1992 */
                            /* "DLCLiShu",     dftt-l5.ttf; version 1.00, 1992 */
                            /* "DLCRoundBold", dftt-r7.ttf; version 1.00, 1993 */

      "HuaTianKaiTi?",      /* htkt2.ttf */
      "HuaTianSongTi?",     /* htst3.ttf */
      "Ming(for ISO10646)", /* hkscsiic.ttf; version 0.12, 2007 [Ming] */
                            /* iicore.ttf; version 0.07, 2007 [Ming] */
      "MingLiU",            /* mingliu.ttf */
                            /* mingliu.ttc; version 3.21, 2001 */
      "MingMedium",         /* dftt-m5.ttf; version 1.00, 1993 [DLCMingMedium] */
      "PMingLiU",           /* mingliu.ttc; version 3.21, 2001 */
      "MingLi43",           /* mingli.ttf; version 1.00, 1992 */
    };

    int  nn;
    const FT_String*  name_without_tag;


    name_without_tag = tt_skip_pdffont_random_tag( name );
    for ( nn = 0; nn < TRICK_NAMES_COUNT; nn++ )
      if ( ft_strstr( name_without_tag, trick_names[nn] ) )
        return TRUE;

    return FALSE;
  }


  /* XXX: This function should be in the `sfnt' module. */

  /* Some PDF generators clear the checksums in the TrueType header table. */
  /* For example, Quartz ContextPDF clears all entries, or Bullzip PDF     */
  /* Printer clears the entries for subsetted subtables.  We thus have to  */
  /* recalculate the checksums  where necessary.                           */

  static FT_UInt32
  tt_synth_sfnt_checksum( FT_Stream  stream,
                          FT_ULong   length )
  {
    FT_Error   error;
    FT_UInt32  checksum = 0;
    FT_UInt    i;


    if ( FT_FRAME_ENTER( length ) )
      return 0;

    for ( ; length > 3; length -= 4 )
      checksum += (FT_UInt32)FT_GET_ULONG();

    for ( i = 3; length > 0; length--, i-- )
      checksum += (FT_UInt32)FT_GET_BYTE() << ( i * 8 );

    FT_FRAME_EXIT();

    return checksum;
  }


  /* XXX: This function should be in the `sfnt' module. */

  static FT_ULong
  tt_get_sfnt_checksum( TT_Face    face,
                        FT_UShort  i )
  {
#if 0 /* if we believe the written value, use following part. */
    if ( face->dir_tables[i].CheckSum )
      return face->dir_tables[i].CheckSum;
#endif

    if ( !face->goto_table )
      return 0;

    if ( face->goto_table( face,
                           face->dir_tables[i].Tag,
                           face->root.stream,
                           NULL ) )
      return 0;

    return (FT_ULong)tt_synth_sfnt_checksum( face->root.stream,
                                             face->dir_tables[i].Length );
  }


  typedef struct tt_sfnt_id_rec_
  {
    FT_ULong  CheckSum;
    FT_ULong  Length;

  } tt_sfnt_id_rec;


  static FT_Bool
  tt_check_trickyness_sfnt_ids( TT_Face  face )
  {
#define TRICK_SFNT_IDS_PER_FACE   3
#define TRICK_SFNT_IDS_NUM_FACES  31

    static const tt_sfnt_id_rec sfnt_id[TRICK_SFNT_IDS_NUM_FACES]
                                       [TRICK_SFNT_IDS_PER_FACE] =
    {

#define TRICK_SFNT_ID_cvt   0
#define TRICK_SFNT_ID_fpgm  1
#define TRICK_SFNT_ID_prep  2

      { /* MingLiU 1995 */
        { 0x05BCF058UL, 0x000002E4UL }, /* cvt  */
        { 0x28233BF1UL, 0x000087C4UL }, /* fpgm */
        { 0xA344A1EAUL, 0x000001E1UL }  /* prep */
      },
      { /* MingLiU 1996- */
        { 0x05BCF058UL, 0x000002E4UL }, /* cvt  */
        { 0x28233BF1UL, 0x000087C4UL }, /* fpgm */
        { 0xA344A1EBUL, 0x000001E1UL }  /* prep */
      },
      { /* DFGothic-EB */
        { 0x12C3EBB2UL, 0x00000350UL }, /* cvt  */
        { 0xB680EE64UL, 0x000087A7UL }, /* fpgm */
        { 0xCE939563UL, 0x00000758UL }  /* prep */
      },
      { /* DFGyoSho-Lt */
        { 0x11E5EAD4UL, 0x00000350UL }, /* cvt  */
        { 0xCE5956E9UL, 0x0000BC85UL }, /* fpgm */
        { 0x8272F416UL, 0x00000045UL }  /* prep */
      },
      { /* DFHei-Md-HK-BF */
        { 0x1257EB46UL, 0x00000350UL }, /* cvt  */
        { 0xF699D160UL, 0x0000715FUL }, /* fpgm */
        { 0xD222F568UL, 0x000003BCUL }  /* prep */
      },
      { /* DFHSGothic-W5 */
        { 0x1262EB4EUL, 0x00000350UL }, /* cvt  */
        { 0xE86A5D64UL, 0x00007940UL }, /* fpgm */
        { 0x7850F729UL, 0x000005FFUL }  /* prep */
      },
      { /* DFHSMincho-W3 */
        { 0x122DEB0AUL, 0x00000350UL }, /* cvt  */
        { 0x3D16328AUL, 0x0000859BUL }, /* fpgm */
        { 0xA93FC33BUL, 0x000002CBUL }  /* prep */
      },
      { /* DFHSMincho-W7 */
        { 0x125FEB26UL, 0x00000350UL }, /* cvt  */
        { 0xA5ACC982UL, 0x00007EE1UL }, /* fpgm */
        { 0x90999196UL, 0x0000041FUL }  /* prep */
      },
      { /* DFKaiShu */
        { 0x11E5EAD4UL, 0x00000350UL }, /* cvt  */
        { 0x5A30CA3BUL, 0x00009063UL }, /* fpgm */
        { 0x13A42602UL, 0x0000007EUL }  /* prep */
      },
      { /* DFKaiShu, variant */
        { 0x11E5EAD4UL, 0x00000350UL }, /* cvt  */
        { 0xA6E78C01UL, 0x00008998UL }, /* fpgm */
        { 0x13A42602UL, 0x0000007EUL }  /* prep */
      },
      { /* DFKaiShu-Md-HK-BF */
        { 0x11E5EAD4UL, 0x00000360UL }, /* cvt  */
        { 0x9DB282B2UL, 0x0000C06EUL }, /* fpgm */
        { 0x53E6D7CAUL, 0x00000082UL }  /* prep */
      },
      { /* DFMing-Bd-HK-BF */
        { 0x1243EB18UL, 0x00000350UL }, /* cvt  */
        { 0xBA0A8C30UL, 0x000074ADUL }, /* fpgm */
        { 0xF3D83409UL, 0x0000037BUL }  /* prep */
      },
      { /* DLCLiShu */
        { 0x07DCF546UL, 0x00000308UL }, /* cvt  */
        { 0x40FE7C90UL, 0x00008E2AUL }, /* fpgm */
        { 0x608174B5UL, 0x0000007AUL }  /* prep */
      },
      { /* DLCHayBold */
        { 0xEB891238UL, 0x00000308UL }, /* cvt  */
        { 0xD2E4DCD4UL, 0x0000676FUL }, /* fpgm */
        { 0x8EA5F293UL, 0x000003B8UL }  /* prep */
      },
      { /* HuaTianKaiTi */
        { 0xFFFBFFFCUL, 0x00000008UL }, /* cvt  */
        { 0x9C9E48B8UL, 0x0000BEA2UL }, /* fpgm */
        { 0x70020112UL, 0x00000008UL }  /* prep */
      },
      { /* HuaTianSongTi */
        { 0xFFFBFFFCUL, 0x00000008UL }, /* cvt  */
        { 0x0A5A0483UL, 0x00017C39UL }, /* fpgm */
        { 0x70020112UL, 0x00000008UL }  /* prep */
      },
      { /* NEC fadpop7.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x40C92555UL, 0x000000E5UL }, /* fpgm */
        { 0xA39B58E3UL, 0x0000117CUL }  /* prep */
      },
      { /* NEC fadrei5.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x33C41652UL, 0x000000E5UL }, /* fpgm */
        { 0x26D6C52AUL, 0x00000F6AUL }  /* prep */
      },
      { /* NEC fangot7.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x6DB1651DUL, 0x0000019DUL }, /* fpgm */
        { 0x6C6E4B03UL, 0x00002492UL }  /* prep */
      },
      { /* NEC fangyo5.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x40C92555UL, 0x000000E5UL }, /* fpgm */
        { 0xDE51FAD0UL, 0x0000117CUL }  /* prep */
      },
      { /* NEC fankyo5.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x85E47664UL, 0x000000E5UL }, /* fpgm */
        { 0xA6C62831UL, 0x00001CAAUL }  /* prep */
      },
      { /* NEC fanrgo5.ttf */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x2D891CFDUL, 0x0000019DUL }, /* fpgm */
        { 0xA0604633UL, 0x00001DE8UL }  /* prep */
      },
      { /* NEC fangot5.ttc */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x40AA774CUL, 0x000001CBUL }, /* fpgm */
        { 0x9B5CAA96UL, 0x00001F9AUL }  /* prep */
      },
      { /* NEC fanmin3.ttc */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x0D3DE9CBUL, 0x00000141UL }, /* fpgm */
        { 0xD4127766UL, 0x00002280UL }  /* prep */
      },
      { /* NEC FA-Gothic, 1996 */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x4A692698UL, 0x000001F0UL }, /* fpgm */
        { 0x340D4346UL, 0x00001FCAUL }  /* prep */
      },
      { /* NEC FA-Minchou, 1996 */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0xCD34C604UL, 0x00000166UL }, /* fpgm */
        { 0x6CF31046UL, 0x000022B0UL }  /* prep */
      },
      { /* NEC FA-RoundGothicB, 1996 */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0x5DA75315UL, 0x0000019DUL }, /* fpgm */
        { 0x40745A5FUL, 0x000022E0UL }  /* prep */
      },
      { /* NEC FA-RoundGothicM, 1996 */
        { 0x00000000UL, 0x00000000UL }, /* cvt  */
        { 0xF055FC48UL, 0x000001C2UL }, /* fpgm */
        { 0x3900DED3UL, 0x00001E18UL }  /* prep */
      },
        { /* MINGLI.TTF, 1992 */
        { 0x00170003UL, 0x00000060UL }, /* cvt  */
        { 0xDBB4306EUL, 0x000058AAUL }, /* fpgm */
        { 0xD643482AUL, 0x00000035UL }  /* prep */
      },
        { /* DFHei-Bd-WIN-HK-BF, issue #1087 */
        { 0x1269EB58UL, 0x00000350UL }, /* cvt  */
        { 0x5CD5957AUL, 0x00006A4EUL }, /* fpgm */
        { 0xF758323AUL, 0x00000380UL }  /* prep */
      },
        { /* DFMing-Md-WIN-HK-BF, issue #1087 */
        { 0x122FEB0BUL, 0x00000350UL }, /* cvt  */
        { 0x7F10919AUL, 0x000070A9UL }, /* fpgm */
        { 0x7CD7E7B7UL, 0x0000025CUL }  /* prep */
      }
    };

    FT_ULong   checksum;
    int        num_matched_ids[TRICK_SFNT_IDS_NUM_FACES];
    FT_Bool    has_cvt, has_fpgm, has_prep;
    FT_UShort  i;
    int        j, k;


    FT_MEM_SET( num_matched_ids, 0,
                sizeof ( int ) * TRICK_SFNT_IDS_NUM_FACES );
    has_cvt  = FALSE;
    has_fpgm = FALSE;
    has_prep = FALSE;

    for ( i = 0; i < face->num_tables; i++ )
    {
      checksum = 0;

      switch( face->dir_tables[i].Tag )
      {
      case TTAG_cvt:
        k = TRICK_SFNT_ID_cvt;
        has_cvt  = TRUE;
        break;

      case TTAG_fpgm:
        k = TRICK_SFNT_ID_fpgm;
        has_fpgm = TRUE;
        break;

      case TTAG_prep:
        k = TRICK_SFNT_ID_prep;
        has_prep = TRUE;
        break;

      default:
        continue;
      }

      for ( j = 0; j < TRICK_SFNT_IDS_NUM_FACES; j++ )
        if ( face->dir_tables[i].Length == sfnt_id[j][k].Length )
        {
          if ( !checksum )
            checksum = tt_get_sfnt_checksum( face, i );

          if ( sfnt_id[j][k].CheckSum == checksum )
            num_matched_ids[j]++;

          if ( num_matched_ids[j] == TRICK_SFNT_IDS_PER_FACE )
            return TRUE;
        }
    }

    for ( j = 0; j < TRICK_SFNT_IDS_NUM_FACES; j++ )
    {
      if ( !has_cvt  && !sfnt_id[j][TRICK_SFNT_ID_cvt].Length )
        num_matched_ids[j]++;
      if ( !has_fpgm && !sfnt_id[j][TRICK_SFNT_ID_fpgm].Length )
        num_matched_ids[j]++;
      if ( !has_prep && !sfnt_id[j][TRICK_SFNT_ID_prep].Length )
        num_matched_ids[j]++;
      if ( num_matched_ids[j] == TRICK_SFNT_IDS_PER_FACE )
        return TRUE;
    }

    return FALSE;
  }


  static FT_Bool
  tt_check_trickyness( FT_Face  face )
  {
    if ( !face )
      return FALSE;

    /* For first, check the face name for quick check. */
    if ( face->family_name                               &&
         tt_check_trickyness_family( face->family_name ) )
    {
      FT_TRACE3(( "found as a tricky font"
                  " by its family name: %s\n", face->family_name ));
      return TRUE;
    }

    /* Type42 fonts may lack `name' tables, we thus try to identify */
    /* tricky fonts by checking the checksums of Type42-persistent  */
    /* sfnt tables (`cvt', `fpgm', and `prep').                     */
    if ( tt_check_trickyness_sfnt_ids( (TT_Face)face ) )
    {
      FT_TRACE3(( "found as a tricky font"
                  " by its cvt/fpgm/prep table checksum\n" ));
      return TRUE;
    }

    return FALSE;
  }

#endif /* TT_USE_BYTECODE_INTERPRETER */


  /* Check whether `.notdef' is the only glyph in the `loca' table. */
  static FT_Bool
  tt_check_single_notdef( FT_Face  ttface )
  {
    FT_Bool   result = FALSE;

    TT_Face   face = (TT_Face)ttface;
    FT_ULong  asize;
    FT_ULong  i;
    FT_ULong  glyph_index = 0;
    FT_UInt   count       = 0;


    for( i = 0; i < face->num_locations; i++ )
    {
      tt_face_get_location( ttface, i, &asize );
      if ( asize > 0 )
      {
        count += 1;
        if ( count > 1 )
          break;
        glyph_index = i;
      }
    }

    /* Only have a single outline. */
    if ( count == 1 )
    {
      if ( glyph_index == 0 )
        result = TRUE;
      else
      {
        /* FIXME: Need to test glyphname == .notdef ? */
        FT_Error error;
        char buf[8];


        error = FT_Get_Glyph_Name( ttface, glyph_index, buf, 8 );
        if ( !error                                            &&
             buf[0] == '.' && !ft_strncmp( buf, ".notdef", 8 ) )
          result = TRUE;
      }
    }

    return result;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_init
   *
   * @Description:
   *   Initialize a given TrueType face object.
   *
   * @Input:
   *   stream ::
   *     The source font stream.
   *
   *   face_index ::
   *     The index of the TrueType font, if we are opening a
   *     collection, in bits 0-15.  The numbered instance
   *     index~+~1 of a GX (sub)font, if applicable, in bits
   *     16-30.
   *
   *   num_params ::
   *     Number of additional generic parameters.  Ignored.
   *
   *   params ::
   *     Additional generic parameters.  Ignored.
   *
   * @InOut:
   *   face ::
   *     The newly built face object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_init( FT_Stream      stream,
                FT_Face        ttface,      /* TT_Face */
                FT_Int         face_index,
                FT_Int         num_params,
                FT_Parameter*  params )
  {
    FT_Error      error;
    FT_Library    library;
    SFNT_Service  sfnt;
    TT_Face       face = (TT_Face)ttface;


    FT_TRACE2(( "TTF driver\n" ));

    library = ttface->driver->root.library;

    sfnt = (SFNT_Service)FT_Get_Module_Interface( library, "sfnt" );
    if ( !sfnt )
    {
      FT_ERROR(( "tt_face_init: cannot access `sfnt' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    /* create input stream from resource */
    if ( FT_STREAM_SEEK( 0 ) )
      goto Exit;

    /* check that we have a valid TrueType file */
    FT_TRACE2(( "  " ));
    error = sfnt->init_face( stream, face, face_index, num_params, params );

    /* Stream may have changed. */
    stream = face->root.stream;

    if ( error )
      goto Exit;

    /* We must also be able to accept Mac/GX fonts, as well as OT ones. */
    /* The 0x00020000 tag is completely undocumented; some fonts from   */
    /* Arphic made for Chinese Windows 3.1 have this.                   */
    if ( face->format_tag != 0x00010000L  && /* MS fonts                             */
         face->format_tag != 0x00020000L  && /* CJK fonts for Win 3.1                */
         face->format_tag != TTAG_true    && /* Mac fonts                            */
         face->format_tag != TTAG_0xA5kbd && /* `Keyboard.dfont' (legacy Mac OS X)   */
         face->format_tag != TTAG_0xA5lst )  /* `LastResort.dfont' (legacy Mac OS X) */
    {
      FT_TRACE2(( "  not a TTF font\n" ));
      goto Bad_Format;
    }

#ifdef TT_USE_BYTECODE_INTERPRETER
    ttface->face_flags |= FT_FACE_FLAG_HINTER;
#endif

    /* If we are performing a simple font format check, exit immediately. */
    if ( face_index < 0 )
      return FT_Err_Ok;

    /* Load font directory */
    error = sfnt->load_face( stream, face, face_index, num_params, params );
    if ( error )
      goto Exit;

#ifdef TT_USE_BYTECODE_INTERPRETER
    if ( tt_check_trickyness( ttface ) )
      ttface->face_flags |= FT_FACE_FLAG_TRICKY;
#endif

    error = tt_face_load_hdmx( face, stream );
    if ( error )
      goto Exit;

    if ( FT_IS_SCALABLE( ttface ) ||
         FT_HAS_SBIX( ttface )    )
    {
#ifdef FT_CONFIG_OPTION_INCREMENTAL
      if ( !ttface->internal->incremental_interface )
#endif
      {
        error = tt_face_load_loca( face, stream );

        /* having a (non-zero) `glyf' table without */
        /* a `loca' table is not valid              */
        if ( face->glyf_len && FT_ERR_EQ( error, Table_Missing ) )
          goto Exit;
        if ( error )
          goto Exit;
      }

      /* `fpgm', `cvt', and `prep' are optional */
      error = tt_face_load_cvt( face, stream );
      if ( error && FT_ERR_NEQ( error, Table_Missing ) )
        goto Exit;

      error = tt_face_load_fpgm( face, stream );
      if ( error && FT_ERR_NEQ( error, Table_Missing ) )
        goto Exit;

      error = tt_face_load_prep( face, stream );
      if ( error && FT_ERR_NEQ( error, Table_Missing ) )
        goto Exit;

      /* Check the scalable flag based on `loca'. */
#ifdef FT_CONFIG_OPTION_INCREMENTAL
      if ( !ttface->internal->incremental_interface )
#endif
      {
        if ( ttface->num_fixed_sizes          &&
             face->glyph_locations            &&
             tt_check_single_notdef( ttface ) )
        {
          FT_TRACE5(( "tt_face_init:"
                      " Only the `.notdef' glyph has an outline.\n" ));
          FT_TRACE5(( "             "
                      " Resetting scalable flag to FALSE.\n" ));

          ttface->face_flags &= ~FT_FACE_FLAG_SCALABLE;
        }
      }
    }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    {
      FT_UInt  instance_index = (FT_UInt)face_index >> 16;


      if ( FT_HAS_MULTIPLE_MASTERS( ttface ) &&
           instance_index > 0                )
      {
        error = FT_Set_Named_Instance( ttface, instance_index );
        if ( error )
          goto Exit;
      }
    }
#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    /* initialize standard glyph loading routines */
    TT_Init_Glyph_Loading( face );

  Exit:
    return error;

  Bad_Format:
    error = FT_THROW( Unknown_File_Format );
    goto Exit;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_done
   *
   * @Description:
   *   Finalize a given face object.
   *
   * @Input:
   *   face ::
   *     A pointer to the face object to destroy.
   */
  FT_LOCAL_DEF( void )
  tt_face_done( FT_Face  ttface )           /* TT_Face */
  {
    TT_Face       face = (TT_Face)ttface;
    FT_Memory     memory;
    FT_Stream     stream;
    SFNT_Service  sfnt;


    if ( !face )
      return;

    memory = ttface->memory;
    stream = ttface->stream;
    sfnt   = (SFNT_Service)face->sfnt;

    /* for `extended TrueType formats' (i.e. compressed versions) */
    if ( face->extra.finalizer )
      face->extra.finalizer( face->extra.data );

    if ( sfnt )
      sfnt->done_face( face );

    /* freeing the locations table */
    tt_face_done_loca( face );

    tt_face_free_hdmx( face );

    /* freeing the CVT */
    FT_FREE( face->cvt );
    face->cvt_size = 0;

    /* freeing the programs */
    FT_FRAME_RELEASE( face->font_program );
    FT_FRAME_RELEASE( face->cvt_program );
    face->font_program_size = 0;
    face->cvt_program_size  = 0;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    tt_done_blend( ttface );
    face->blend = NULL;
#endif
  }


  /**************************************************************************
   *
   *                          SIZE  FUNCTIONS
   *
   */

#ifdef TT_USE_BYTECODE_INTERPRETER

  /**************************************************************************
   *
   * @Function:
   *   tt_size_run_fpgm
   *
   * @Description:
   *   Run the font program.
   *
   * @Input:
   *   size ::
   *     A handle to the size object.
   *
   *   pedantic ::
   *     Set if bytecode execution should be pedantic.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_size_run_fpgm( TT_Size  size,
                    FT_Bool  pedantic )
  {
    TT_Face         face = (TT_Face)size->root.face;
    TT_ExecContext  exec;
    FT_Error        error;


    exec = size->context;

    error = TT_Load_Context( exec, face, size );
    if ( error )
      return error;

    exec->callTop = 0;
    exec->top     = 0;

    exec->period    = 64;
    exec->phase     = 0;
    exec->threshold = 0;

    exec->instruction_trap = FALSE;
    exec->F_dot_P          = 0x4000L;

    exec->pedantic_hinting = pedantic;

    {
      FT_Size_Metrics*  size_metrics = &exec->metrics;
      TT_Size_Metrics*  tt_metrics   = &exec->tt_metrics;


      size_metrics->x_ppem   = 0;
      size_metrics->y_ppem   = 0;
      size_metrics->x_scale  = 0;
      size_metrics->y_scale  = 0;

      tt_metrics->ppem  = 0;
      tt_metrics->scale = 0;
      tt_metrics->ratio = 0x10000L;
    }

    /* allow font program execution */
    TT_Set_CodeRange( exec,
                      tt_coderange_font,
                      face->font_program,
                      (FT_Long)face->font_program_size );

    /* disable CVT and glyph programs coderange */
    TT_Clear_CodeRange( exec, tt_coderange_cvt );
    TT_Clear_CodeRange( exec, tt_coderange_glyph );

    if ( face->font_program_size > 0 )
    {
      TT_Goto_CodeRange( exec, tt_coderange_font, 0 );

      FT_TRACE4(( "Executing `fpgm' table.\n" ));
      error = face->interpreter( exec );
#ifdef FT_DEBUG_LEVEL_TRACE
      if ( error )
        FT_TRACE4(( "  interpretation failed with error code 0x%x\n",
                    error ));
#endif
    }
    else
      error = FT_Err_Ok;

    size->bytecode_ready = error;

    if ( !error )
      TT_Save_Context( exec, size );

    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_size_run_prep
   *
   * @Description:
   *   Run the control value program.
   *
   * @Input:
   *   size ::
   *     A handle to the size object.
   *
   *   pedantic ::
   *     Set if bytecode execution should be pedantic.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_size_run_prep( TT_Size  size,
                    FT_Bool  pedantic )
  {
    TT_Face         face = (TT_Face)size->root.face;
    TT_ExecContext  exec;
    FT_Error        error;
    FT_UInt         i;

    /* unscaled CVT values are already stored in 26.6 format */
    FT_Fixed  scale = size->ttmetrics.scale >> 6;


    /* Scale the cvt values to the new ppem.            */
    /* By default, we use the y ppem value for scaling. */
    FT_TRACE6(( "CVT values:\n" ));
    for ( i = 0; i < size->cvt_size; i++ )
    {
      size->cvt[i] = FT_MulFix( face->cvt[i], scale );
      FT_TRACE6(( "  %3d: %f (%f)\n",
                  i, (double)face->cvt[i] / 64, (double)size->cvt[i] / 64 ));
    }
    FT_TRACE6(( "\n" ));

    exec = size->context;

    error = TT_Load_Context( exec, face, size );
    if ( error )
      return error;

    exec->callTop = 0;
    exec->top     = 0;

    exec->instruction_trap = FALSE;

    exec->pedantic_hinting = pedantic;

    TT_Set_CodeRange( exec,
                      tt_coderange_cvt,
                      face->cvt_program,
                      (FT_Long)face->cvt_program_size );

    TT_Clear_CodeRange( exec, tt_coderange_glyph );

    if ( face->cvt_program_size > 0 )
    {
      TT_Goto_CodeRange( exec, tt_coderange_cvt, 0 );

      FT_TRACE4(( "Executing `prep' table.\n" ));
      error = face->interpreter( exec );
#ifdef FT_DEBUG_LEVEL_TRACE
      if ( error )
        FT_TRACE4(( "  interpretation failed with error code 0x%x\n",
                    error ));
#endif
    }
    else
      error = FT_Err_Ok;

    size->cvt_ready = error;

    /* UNDOCUMENTED!  The MS rasterizer doesn't allow the following */
    /* graphics state variables to be modified by the CVT program.  */

    exec->GS.dualVector.x = 0x4000;
    exec->GS.dualVector.y = 0;
    exec->GS.projVector.x = 0x4000;
    exec->GS.projVector.y = 0x0;
    exec->GS.freeVector.x = 0x4000;
    exec->GS.freeVector.y = 0x0;

    exec->GS.rp0 = 0;
    exec->GS.rp1 = 0;
    exec->GS.rp2 = 0;

    exec->GS.gep0 = 1;
    exec->GS.gep1 = 1;
    exec->GS.gep2 = 1;

    exec->GS.loop = 1;

    /* save as default graphics state */
    size->GS = exec->GS;

    TT_Save_Context( exec, size );

    return error;
  }


  static void
  tt_size_done_bytecode( FT_Size  ftsize )
  {
    TT_Size    size   = (TT_Size)ftsize;
    TT_Face    face   = (TT_Face)ftsize->face;
    FT_Memory  memory = face->root.memory;

    if ( size->context )
    {
      TT_Done_Context( size->context );
      size->context = NULL;
    }

    FT_FREE( size->cvt );
    size->cvt_size = 0;

    /* free storage area */
    FT_FREE( size->storage );
    size->storage_size = 0;

    /* twilight zone */
    tt_glyphzone_done( &size->twilight );

    FT_FREE( size->function_defs );
    FT_FREE( size->instruction_defs );

    size->num_function_defs    = 0;
    size->max_function_defs    = 0;
    size->num_instruction_defs = 0;
    size->max_instruction_defs = 0;

    size->max_func = 0;
    size->max_ins  = 0;

    size->bytecode_ready = -1;
    size->cvt_ready      = -1;
  }


  /* Initialize bytecode-related fields in the size object.       */
  /* We do this only if bytecode interpretation is really needed. */
  static FT_Error
  tt_size_init_bytecode( FT_Size  ftsize,
                         FT_Bool  pedantic )
  {
    FT_Error   error;
    TT_Size    size = (TT_Size)ftsize;
    TT_Face    face = (TT_Face)ftsize->face;
    FT_Memory  memory = face->root.memory;

    FT_UShort       n_twilight;
    TT_MaxProfile*  maxp = &face->max_profile;


    /* clean up bytecode related data */
    FT_FREE( size->function_defs );
    FT_FREE( size->instruction_defs );
    FT_FREE( size->cvt );
    FT_FREE( size->storage );

    if ( size->context )
      TT_Done_Context( size->context );
    tt_glyphzone_done( &size->twilight );

    size->bytecode_ready = -1;
    size->cvt_ready      = -1;

    size->context = TT_New_Context( (TT_Driver)face->root.driver );

    size->max_function_defs    = maxp->maxFunctionDefs;
    size->max_instruction_defs = maxp->maxInstructionDefs;

    size->num_function_defs    = 0;
    size->num_instruction_defs = 0;

    size->max_func = 0;
    size->max_ins  = 0;

    size->cvt_size     = face->cvt_size;
    size->storage_size = maxp->maxStorage;

    /* Set default metrics */
    {
      TT_Size_Metrics*  tt_metrics = &size->ttmetrics;


      tt_metrics->rotated   = FALSE;
      tt_metrics->stretched = FALSE;

      /* Set default engine compensation.  Value 3 is not described */
      /* in the OpenType specification (as of Mai 2019), but Greg   */
      /* says that MS handles it the same as `gray'.                */
      /*                                                            */
      /* The Apple specification says that the compensation for     */
      /* `gray' is always zero.  FreeType doesn't do any            */
      /* compensation at all.                                       */
      tt_metrics->compensations[0] = 0;   /* gray  */
      tt_metrics->compensations[1] = 0;   /* black */
      tt_metrics->compensations[2] = 0;   /* white */
      tt_metrics->compensations[3] = 0;   /* zero  */
    }

    /* allocate function defs, instruction defs, cvt, and storage area */
    if ( FT_NEW_ARRAY( size->function_defs,    size->max_function_defs    ) ||
         FT_NEW_ARRAY( size->instruction_defs, size->max_instruction_defs ) ||
         FT_NEW_ARRAY( size->cvt,              size->cvt_size             ) ||
         FT_NEW_ARRAY( size->storage,          size->storage_size         ) )
      goto Exit;

    /* reserve twilight zone */
    n_twilight = maxp->maxTwilightPoints;

    /* there are 4 phantom points (do we need this?) */
    n_twilight += 4;

    error = tt_glyphzone_new( memory, n_twilight, 0, &size->twilight );
    if ( error )
      goto Exit;

    size->twilight.n_points = n_twilight;

    size->GS = tt_default_graphics_state;

    /* set `face->interpreter' according to the debug hook present */
    {
      FT_Library  library = face->root.driver->root.library;


      face->interpreter = (TT_Interpreter)
                            library->debug_hooks[FT_DEBUG_HOOK_TRUETYPE];
      if ( !face->interpreter )
        face->interpreter = (TT_Interpreter)TT_RunIns;
    }

    /* Fine, now run the font program! */

    /* In case of an error while executing `fpgm', we intentionally don't */
    /* clean up immediately – bugs in the `fpgm' are so fundamental that  */
    /* all following hinting calls should fail.  Additionally, `fpgm' is  */
    /* to be executed just once; calling it again is completely useless   */
    /* and might even lead to extremely slow behaviour if it is malformed */
    /* (containing an infinite loop, for example).                        */
    error = tt_size_run_fpgm( size, pedantic );
    return error;

  Exit:
    if ( error )
      tt_size_done_bytecode( ftsize );

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_size_ready_bytecode( TT_Size  size,
                          FT_Bool  pedantic )
  {
    FT_Error  error = FT_Err_Ok;


    if ( size->bytecode_ready < 0 )
      error = tt_size_init_bytecode( (FT_Size)size, pedantic );
    else
      error = size->bytecode_ready;

    if ( error )
      goto Exit;

    /* rescale CVT when needed */
    if ( size->cvt_ready < 0 )
    {
      FT_UShort  i;


      /* all twilight points are originally zero */
      for ( i = 0; i < size->twilight.n_points; i++ )
      {
        size->twilight.org[i].x = 0;
        size->twilight.org[i].y = 0;
        size->twilight.cur[i].x = 0;
        size->twilight.cur[i].y = 0;
      }

      /* clear storage area */
      for ( i = 0; i < size->storage_size; i++ )
        size->storage[i] = 0;

      size->GS = tt_default_graphics_state;

      error = tt_size_run_prep( size, pedantic );
    }
    else
      error = size->cvt_ready;

  Exit:
    return error;
  }

#endif /* TT_USE_BYTECODE_INTERPRETER */


  /**************************************************************************
   *
   * @Function:
   *   tt_size_init
   *
   * @Description:
   *   Initialize a new TrueType size object.
   *
   * @InOut:
   *   size ::
   *     A handle to the size object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_size_init( FT_Size  ttsize )           /* TT_Size */
  {
    TT_Size   size  = (TT_Size)ttsize;
    FT_Error  error = FT_Err_Ok;


#ifdef TT_USE_BYTECODE_INTERPRETER
    size->bytecode_ready = -1;
    size->cvt_ready      = -1;
#endif

    size->ttmetrics.valid = FALSE;
    size->strike_index    = 0xFFFFFFFFUL;

    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_size_done
   *
   * @Description:
   *   The TrueType size object finalizer.
   *
   * @Input:
   *   size ::
   *     A handle to the target size object.
   */
  FT_LOCAL_DEF( void )
  tt_size_done( FT_Size  ttsize )           /* TT_Size */
  {
    TT_Size  size = (TT_Size)ttsize;


#ifdef TT_USE_BYTECODE_INTERPRETER
    tt_size_done_bytecode( ttsize );
#endif

    size->ttmetrics.valid = FALSE;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_size_reset_height
   *
   * @Description:
   *   Recompute a TrueType size's ascender, descender, and height
   *   when resolutions and character dimensions have been changed.
   *   Used for variation fonts as an iterator function.
   *
   * @Input:
   *   ft_size ::
   *     A handle to the target TT_Size object. This function will be called
   *     through a `FT_Size_Reset_Func` pointer which takes `FT_Size`. This
   *     function must take `FT_Size` as a result. The passed `FT_Size` is
   *     expected to point to a `TT_Size`.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_size_reset_height( FT_Size  ft_size )
  {
    TT_Size           size         = (TT_Size)ft_size;
    TT_Face           face         = (TT_Face)size->root.face;
    FT_Size_Metrics*  size_metrics = &size->hinted_metrics;

    size->ttmetrics.valid = FALSE;

    /* copy the result from base layer */
    *size_metrics = size->root.metrics;

    if ( size_metrics->x_ppem < 1 || size_metrics->y_ppem < 1 )
      return FT_THROW( Invalid_PPem );

    /* This bit flag, if set, indicates that the ppems must be       */
    /* rounded to integers.  Nearly all TrueType fonts have this bit */
    /* set, as hinting won't work really well otherwise.             */
    /*                                                               */
    if ( face->header.Flags & 8 )
    {
      /* the TT spec always asks for ROUND, not FLOOR or CEIL */
      size_metrics->ascender = FT_PIX_ROUND(
                                 FT_MulFix( face->root.ascender,
                                            size_metrics->y_scale ) );
      size_metrics->descender = FT_PIX_ROUND(
                                 FT_MulFix( face->root.descender,
                                            size_metrics->y_scale ) );
      size_metrics->height = FT_PIX_ROUND(
                               FT_MulFix( face->root.height,
                                          size_metrics->y_scale ) );
    }

    size->ttmetrics.valid = TRUE;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_size_reset
   *
   * @Description:
   *   Reset a TrueType size when resolutions and character dimensions
   *   have been changed.
   *
   * @Input:
   *   size ::
   *     A handle to the target size object.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_size_reset( TT_Size  size )
  {
    FT_Error          error;
    TT_Face           face         = (TT_Face)size->root.face;
    FT_Size_Metrics*  size_metrics = &size->hinted_metrics;


    error = tt_size_reset_height( (FT_Size)size );
    if ( error )
      return error;

    if ( face->header.Flags & 8 )
    {
      /* base scaling values on integer ppem values, */
      /* as mandated by the TrueType specification   */
      size_metrics->x_scale = FT_DivFix( size_metrics->x_ppem << 6,
                                         face->root.units_per_EM );
      size_metrics->y_scale = FT_DivFix( size_metrics->y_ppem << 6,
                                         face->root.units_per_EM );

      size_metrics->max_advance = FT_PIX_ROUND(
                                    FT_MulFix( face->root.max_advance_width,
                                               size_metrics->x_scale ) );
    }

    /* compute new transformation */
    if ( size_metrics->x_ppem >= size_metrics->y_ppem )
    {
      size->ttmetrics.scale   = size_metrics->x_scale;
      size->ttmetrics.ppem    = size_metrics->x_ppem;
      size->ttmetrics.x_ratio = 0x10000L;
      size->ttmetrics.y_ratio = FT_DivFix( size_metrics->y_ppem,
                                           size_metrics->x_ppem );
    }
    else
    {
      size->ttmetrics.scale   = size_metrics->y_scale;
      size->ttmetrics.ppem    = size_metrics->y_ppem;
      size->ttmetrics.x_ratio = FT_DivFix( size_metrics->x_ppem,
                                           size_metrics->y_ppem );
      size->ttmetrics.y_ratio = 0x10000L;
    }

    size->widthp = tt_face_get_device_metrics( face, size_metrics->x_ppem, 0 );

    size->metrics = size_metrics;

#ifdef TT_USE_BYTECODE_INTERPRETER
    size->cvt_ready = -1;
#endif /* TT_USE_BYTECODE_INTERPRETER */

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_driver_init
   *
   * @Description:
   *   Initialize a given TrueType driver object.
   *
   * @Input:
   *   driver ::
   *     A handle to the target driver object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_driver_init( FT_Module  ttdriver )     /* TT_Driver */
  {

#ifdef TT_USE_BYTECODE_INTERPRETER

    TT_Driver  driver = (TT_Driver)ttdriver;

    driver->interpreter_version = TT_INTERPRETER_VERSION_35;
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    driver->interpreter_version = TT_INTERPRETER_VERSION_40;
#endif

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_UNUSED( ttdriver );

#endif /* !TT_USE_BYTECODE_INTERPRETER */

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_driver_done
   *
   * @Description:
   *   Finalize a given TrueType driver.
   *
   * @Input:
   *   driver ::
   *     A handle to the target TrueType driver.
   */
  FT_LOCAL_DEF( void )
  tt_driver_done( FT_Module  ttdriver )     /* TT_Driver */
  {
    FT_UNUSED( ttdriver );
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_slot_init
   *
   * @Description:
   *   Initialize a new slot object.
   *
   * @InOut:
   *   slot ::
   *     A handle to the slot object.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_slot_init( FT_GlyphSlot  slot )
  {
    return FT_GlyphLoader_CreateExtra( slot->internal->loader );
  }


/* END */
