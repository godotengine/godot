/****************************************************************************
 *
 * afadjust.c
 *
 *   Auto-fitter routines to adjust components based on charcode (body).
 *
 * Copyright (C) 2023-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Craig White <gerzytet@gmail.com>.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#include "afadjust.h"
#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
#  include "afgsub.h"
#endif

#include <freetype/freetype.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftmemory.h>
#include <freetype/internal/ftdebug.h>

#define AF_ADJUSTMENT_DATABASE_LENGTH           \
          ( sizeof ( adjustment_database ) /    \
            sizeof ( adjustment_database[0] ) )

#undef  FT_COMPONENT
#define FT_COMPONENT  afadjust


  typedef struct  AF_AdjustmentDatabaseEntry_
  {
    FT_UInt32  codepoint;
    FT_UInt32  flags;

  } AF_AdjustmentDatabaseEntry;


  /*
    All entries in this list must be sorted by ascending Unicode code
    points.  The table entries are 3 numbers consisting of:

    - Unicode code point.
    - The vertical adjustment type.  This should be a combination of the
      AF_ADJUST_XXX and AF_IGNORE_XXX macros.
  */
  static AF_AdjustmentDatabaseEntry  adjustment_database[] =
  {
    /* C0 Controls and Basic Latin */
    { 0x21,  AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ! */
    { 0x51,  AF_IGNORE_CAPITAL_BOTTOM } , /* Q */
    { 0x3F,  AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ? */
    { 0x69,  AF_ADJUST_UP }, /* i */
    { 0x6A,  AF_ADJUST_UP }, /* j */
#if 0
    /* XXX TODO */
    { 0x7E,  AF_ADJUST_TILDE_TOP }, /* ~ */
#endif

    /* C1 Controls and Latin-1 Supplement */
    { 0xA1,  AF_ADJUST_UP }, /* ¡ */
    { 0xA6,  AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ¦ */
    { 0xAA,  AF_ADJUST_UP }, /* ª */
    { 0xBA,  AF_ADJUST_UP }, /* º */
    { 0xBF,  AF_ADJUST_UP }, /* ¿ */

    { 0xC0,  AF_ADJUST_UP }, /* À */
    { 0xC1,  AF_ADJUST_UP }, /* Á */
    { 0xC2,  AF_ADJUST_UP }, /* Â */
    { 0xC3,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ã */
    { 0xC4,  AF_ADJUST_UP }, /* Ä */
    { 0xC5,  AF_ADJUST_UP }, /* Å */
    { 0xC7,  AF_IGNORE_CAPITAL_BOTTOM }, /* Ç */
    { 0xC8,  AF_ADJUST_UP }, /* È */
    { 0xC9,  AF_ADJUST_UP }, /* É */
    { 0xCA,  AF_ADJUST_UP }, /* Ê */
    { 0xCB,  AF_ADJUST_UP }, /* Ë */
    { 0xCC,  AF_ADJUST_UP }, /* Ì */
    { 0xCD,  AF_ADJUST_UP }, /* Í */
    { 0xCE,  AF_ADJUST_UP }, /* Î */
    { 0xCF,  AF_ADJUST_UP }, /* Ï */

    { 0xD1,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ñ */
    { 0xD2,  AF_ADJUST_UP }, /* Ò */
    { 0xD3,  AF_ADJUST_UP }, /* Ó */
    { 0xD4,  AF_ADJUST_UP }, /* Ô */
    { 0xD5,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Õ */
    { 0xD6,  AF_ADJUST_UP }, /* Ö */
    { 0xD8,  AF_IGNORE_CAPITAL_TOP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ø */
    { 0xD9,  AF_ADJUST_UP }, /* Ù */
    { 0xDA,  AF_ADJUST_UP }, /* Ú */
    { 0xDB,  AF_ADJUST_UP }, /* Û */
    { 0xDC,  AF_ADJUST_UP }, /* Ü */
    { 0xDD,  AF_ADJUST_UP }, /* Ý */

    { 0xE0,  AF_ADJUST_UP }, /* à */
    { 0xE1,  AF_ADJUST_UP }, /* á */
    { 0xE2,  AF_ADJUST_UP }, /* â */
    { 0xE3,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ã */
    { 0xE4,  AF_ADJUST_UP }, /* ä */
    { 0xE5,  AF_ADJUST_UP }, /* å */
    { 0xE7,  AF_IGNORE_SMALL_BOTTOM }, /* ç */
    { 0xE8,  AF_ADJUST_UP }, /* è */
    { 0xE9,  AF_ADJUST_UP }, /* é */
    { 0xEA,  AF_ADJUST_UP }, /* ê */
    { 0xEB,  AF_ADJUST_UP }, /* ë */
    { 0xEC,  AF_ADJUST_UP }, /* ì */
    { 0xED,  AF_ADJUST_UP }, /* í */
    { 0xEE,  AF_ADJUST_UP }, /* î */
    { 0xEF,  AF_ADJUST_UP }, /* ï */

    { 0xF1,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ñ */
    { 0xF2,  AF_ADJUST_UP }, /* ò */
    { 0xF3,  AF_ADJUST_UP }, /* ó */
    { 0xF4,  AF_ADJUST_UP }, /* ô */
    { 0xF5,  AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* õ */
    { 0xF6,  AF_ADJUST_UP }, /* ö */
    { 0xF8,  AF_IGNORE_SMALL_TOP | AF_IGNORE_SMALL_BOTTOM }, /* ø */
    { 0xF9,  AF_ADJUST_UP }, /* ù */
    { 0xFA,  AF_ADJUST_UP }, /* ú */
    { 0xFB,  AF_ADJUST_UP }, /* û */
    { 0xFC,  AF_ADJUST_UP }, /* ü */
    { 0xFD,  AF_ADJUST_UP }, /* ý */
    { 0xFF,  AF_ADJUST_UP }, /* ÿ */

    /* Latin Extended-A */
    { 0x100, AF_ADJUST_UP }, /* Ā */
    { 0x101, AF_ADJUST_UP }, /* ā */
    { 0x102, AF_ADJUST_UP }, /* Ă */
    { 0x103, AF_ADJUST_UP }, /* ă */
    { 0x104, AF_IGNORE_CAPITAL_BOTTOM }, /* Ą */
    { 0x105, AF_IGNORE_SMALL_BOTTOM }, /* ą */
    { 0x106, AF_ADJUST_UP }, /* Ć */
    { 0x107, AF_ADJUST_UP }, /* ć */
    { 0x108, AF_ADJUST_UP }, /* Ĉ */
    { 0x109, AF_ADJUST_UP }, /* ĉ */
    { 0x10A, AF_ADJUST_UP }, /* Ċ */
    { 0x10B, AF_ADJUST_UP }, /* ċ */
    { 0x10C, AF_ADJUST_UP }, /* Č */
    { 0x10D, AF_ADJUST_UP }, /* č */
    { 0x10E, AF_ADJUST_UP }, /* Ď */

    { 0x112, AF_ADJUST_UP }, /* Ē */
    { 0x113, AF_ADJUST_UP }, /* ē */
    { 0x114, AF_ADJUST_UP }, /* Ĕ */
    { 0x115, AF_ADJUST_UP }, /* ĕ */
    { 0x116, AF_ADJUST_UP }, /* Ė */
    { 0x117, AF_ADJUST_UP }, /* ė */
    { 0x118, AF_IGNORE_CAPITAL_BOTTOM }, /* Ę */
    { 0x119, AF_IGNORE_SMALL_BOTTOM }, /* ę */
    { 0x11A, AF_ADJUST_UP }, /* Ě */
    { 0x11B, AF_ADJUST_UP }, /* ě */
    { 0x11C, AF_ADJUST_UP }, /* Ĝ */
    { 0x11D, AF_ADJUST_UP }, /* ĝ */
    { 0x11E, AF_ADJUST_UP }, /* Ğ */
    { 0x11F, AF_ADJUST_UP }, /* ğ */

    { 0x120, AF_ADJUST_UP }, /* Ġ */
    { 0x121, AF_ADJUST_UP }, /* ġ */
    { 0x122, AF_ADJUST_DOWN }, /* Ģ */
    { 0x123, AF_ADJUST_UP }, /* ģ */
    { 0x124, AF_ADJUST_UP }, /* Ĥ */
    { 0x125, AF_ADJUST_UP }, /* ĥ */
    { 0x128, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ĩ */
    { 0x129, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ĩ */
    { 0x12A, AF_ADJUST_UP }, /* Ī */
    { 0x12B, AF_ADJUST_UP }, /* ī */
    { 0x12C, AF_ADJUST_UP }, /* Ĭ */
    { 0x12D, AF_ADJUST_UP }, /* ĭ */
    { 0x12E, AF_IGNORE_CAPITAL_BOTTOM }, /* Į */
    { 0x12F, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* į */

    { 0x130, AF_ADJUST_UP }, /* İ */
    { 0x133, AF_ADJUST_UP }, /* ĳ */
    { 0x134, AF_ADJUST_UP }, /* Ĵ */
    { 0x135, AF_ADJUST_UP }, /* ĵ */
    { 0x136, AF_ADJUST_DOWN }, /* Ķ */
    { 0x137, AF_ADJUST_DOWN }, /* ķ */
    { 0x139, AF_ADJUST_UP }, /* Ĺ */
    { 0x13A, AF_ADJUST_UP }, /* ĺ */
    { 0x13B, AF_ADJUST_DOWN }, /* Ļ */
    { 0x13C, AF_ADJUST_DOWN }, /* ļ */

    { 0x143, AF_ADJUST_UP }, /* Ń */
    { 0x144, AF_ADJUST_UP }, /* ń */
    { 0x145, AF_ADJUST_DOWN }, /* Ņ */
    { 0x146, AF_ADJUST_DOWN }, /* ņ */
    { 0x147, AF_ADJUST_UP }, /* Ň */
    { 0x148, AF_ADJUST_UP }, /* ň */
    { 0x14C, AF_ADJUST_UP }, /* Ō */
    { 0x14D, AF_ADJUST_UP }, /* ō */
    { 0x14E, AF_ADJUST_UP }, /* Ŏ */
    { 0x14F, AF_ADJUST_UP }, /* ŏ */

    { 0x150, AF_ADJUST_UP }, /* Ő */
    { 0x151, AF_ADJUST_UP }, /* ő */
    { 0x154, AF_ADJUST_UP }, /* Ŕ */
    { 0x155, AF_ADJUST_UP }, /* ŕ */
    { 0x156, AF_ADJUST_DOWN }, /* Ŗ */
    { 0x157, AF_ADJUST_DOWN }, /* ŗ */
    { 0x158, AF_ADJUST_UP }, /* Ř */
    { 0x159, AF_ADJUST_UP }, /* ř */
    { 0x15A, AF_ADJUST_UP }, /* Ś */
    { 0x15B, AF_ADJUST_UP }, /* ś */
    { 0x15C, AF_ADJUST_UP }, /* Ŝ */
    { 0x15D, AF_ADJUST_UP }, /* ŝ */
    { 0x15E, AF_IGNORE_CAPITAL_BOTTOM }, /* Ş */
    { 0x15F, AF_IGNORE_SMALL_BOTTOM }, /* ş */

    { 0x160, AF_ADJUST_UP }, /* Š */
    { 0x161, AF_ADJUST_UP }, /* š */
    { 0x162, AF_IGNORE_CAPITAL_BOTTOM }, /* Ţ */
    { 0x163, AF_IGNORE_SMALL_BOTTOM }, /* ţ */
    { 0x164, AF_ADJUST_UP }, /* Ť */
    { 0x168, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ũ */
    { 0x169, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ũ */
    { 0x16A, AF_ADJUST_UP }, /* Ū */
    { 0x16B, AF_ADJUST_UP }, /* ū */
    { 0x16C, AF_ADJUST_UP }, /* Ŭ */
    { 0x16D, AF_ADJUST_UP }, /* ŭ */
    { 0x16E, AF_ADJUST_UP }, /* Ů */
    { 0x16F, AF_ADJUST_UP }, /* ů */

    { 0x170, AF_ADJUST_UP }, /* Ű */
    { 0x171, AF_ADJUST_UP }, /* ű */
    { 0x172, AF_IGNORE_CAPITAL_BOTTOM }, /* Ų */
    { 0x173, AF_IGNORE_SMALL_BOTTOM }, /* ų */
    { 0x174, AF_ADJUST_UP }, /* Ŵ */
    { 0x175, AF_ADJUST_UP }, /* ŵ */
    { 0x176, AF_ADJUST_UP }, /* Ŷ */
    { 0x177, AF_ADJUST_UP }, /* ŷ */
    { 0x178, AF_ADJUST_UP }, /* Ÿ */
    { 0x179, AF_ADJUST_UP }, /* Ź */
    { 0x17A, AF_ADJUST_UP }, /* ź */
    { 0x17B, AF_ADJUST_UP }, /* Ż */
    { 0x17C, AF_ADJUST_UP }, /* ż */
    { 0x17D, AF_ADJUST_UP }, /* Ž */
    { 0x17E, AF_ADJUST_UP }, /* ž */

    /* Latin Extended-B */
    { 0x187, AF_IGNORE_CAPITAL_TOP }, /* Ƈ */
    { 0x188, AF_IGNORE_SMALL_TOP }, /* ƈ */

    { 0x1A0, AF_IGNORE_CAPITAL_TOP }, /* Ơ */
    { 0x1A1, AF_IGNORE_SMALL_TOP }, /* ơ */
    { 0x1A5, AF_IGNORE_SMALL_TOP }, /* ƥ */
    { 0x1AB, AF_IGNORE_SMALL_BOTTOM }, /* ƫ */
    { 0x1AE, AF_IGNORE_CAPITAL_BOTTOM }, /* Ʈ */
    { 0x1AF, AF_IGNORE_CAPITAL_TOP }, /* Ư */

    { 0x1B0, AF_IGNORE_SMALL_TOP }, /* ư */
    { 0x1B4, AF_IGNORE_SMALL_TOP }, /* ƴ */

    { 0x1C3, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ǃ */
    { 0x1C4, AF_ADJUST_UP }, /* Ǆ */
#if 0
    { 0x1C5, AF_ADJUST_UP }, /* ǅ */
    { 0x1C6, AF_ADJUST_UP }, /* ǆ */
    { 0x1C8, AF_ADJUST_UP }, /* ǈ */
    { 0x1C9, AF_ADJUST_UP }, /* ǉ */
    { 0x1CB, AF_ADJUST_UP }, /* ǋ */
#endif
    { 0x1CC, AF_ADJUST_UP }, /* ǌ */
    { 0x1CD, AF_ADJUST_UP }, /* Ǎ */
    { 0x1CE, AF_ADJUST_UP }, /* ǎ */
    { 0x1CF, AF_ADJUST_UP }, /* Ǐ */

    { 0x1D0, AF_ADJUST_UP }, /* ǐ */
    { 0x1D1, AF_ADJUST_UP }, /* Ǒ */
    { 0x1D2, AF_ADJUST_UP }, /* ǒ */
    { 0x1D3, AF_ADJUST_UP }, /* Ǔ */
    { 0x1D4, AF_ADJUST_UP }, /* ǔ */
    { 0x1D5, AF_ADJUST_UP2 }, /* Ǖ */
    { 0x1D6, AF_ADJUST_UP2 }, /* ǖ */
    { 0x1D7, AF_ADJUST_UP2 }, /* Ǘ */
    { 0x1D8, AF_ADJUST_UP2 }, /* ǘ */
    { 0x1D9, AF_ADJUST_UP2 }, /* Ǚ */
    { 0x1DA, AF_ADJUST_UP2 }, /* ǚ */
    { 0x1DB, AF_ADJUST_UP2 }, /* Ǜ */
    { 0x1DC, AF_ADJUST_UP2 }, /* ǜ */
    { 0x1DE, AF_ADJUST_UP2 }, /* Ǟ */
    { 0x1DF, AF_ADJUST_UP2 }, /* ǟ */

    { 0x1E0, AF_ADJUST_UP2 }, /* Ǡ */
    { 0x1E1, AF_ADJUST_UP2 }, /* ǡ */
    { 0x1E2, AF_ADJUST_UP }, /* Ǣ */
    { 0x1E3, AF_ADJUST_UP }, /* ǣ */
    { 0x1E6, AF_ADJUST_UP }, /* Ǧ */
    { 0x1E7, AF_ADJUST_UP }, /* ǧ */
    { 0x1E8, AF_ADJUST_UP }, /* Ǩ */
    { 0x1E9, AF_ADJUST_UP }, /* ǩ */
    { 0x1EA, AF_IGNORE_CAPITAL_BOTTOM }, /* Ǫ */
    { 0x1EB, AF_IGNORE_SMALL_BOTTOM }, /* ǫ */
    { 0x1EC, AF_ADJUST_UP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ǭ */
    { 0x1ED, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* ǭ */
    { 0x1EE, AF_ADJUST_UP }, /* Ǯ */
    { 0x1EF, AF_ADJUST_UP }, /* ǯ */

    { 0x1F0, AF_ADJUST_UP }, /* ǰ */
    { 0x1F4, AF_ADJUST_UP }, /* Ǵ */
    { 0x1F5, AF_ADJUST_UP }, /* ǵ */
    { 0x1F8, AF_ADJUST_UP }, /* Ǹ */
    { 0x1F9, AF_ADJUST_UP }, /* ǹ */
    { 0x1FA, AF_ADJUST_UP2 }, /* Ǻ */
    { 0x1FB, AF_ADJUST_UP2 }, /* ǻ */
    { 0x1FC, AF_ADJUST_UP }, /* Ǽ */
    { 0x1FD, AF_ADJUST_UP }, /* ǽ */
    { 0x1FE, AF_ADJUST_UP }, /* Ǿ */
    { 0x1FF, AF_ADJUST_UP }, /* ǿ */

    { 0x200, AF_ADJUST_UP }, /* Ȁ */
    { 0x201, AF_ADJUST_UP }, /* ȁ */
    { 0x202, AF_ADJUST_UP }, /* Ȃ */
    { 0x203, AF_ADJUST_UP }, /* ȃ */
    { 0x204, AF_ADJUST_UP }, /* Ȅ */
    { 0x205, AF_ADJUST_UP }, /* ȅ */
    { 0x206, AF_ADJUST_UP }, /* Ȇ */
    { 0x207, AF_ADJUST_UP }, /* ȇ */
    { 0x208, AF_ADJUST_UP }, /* Ȉ */
    { 0x209, AF_ADJUST_UP }, /* ȉ */
    { 0x20A, AF_ADJUST_UP }, /* Ȋ */
    { 0x20B, AF_ADJUST_UP }, /* ȋ */
    { 0x20C, AF_ADJUST_UP }, /* Ȍ */
    { 0x20D, AF_ADJUST_UP }, /* ȍ */
    { 0x20E, AF_ADJUST_UP }, /* Ȏ */
    { 0x20F, AF_ADJUST_UP }, /* ȏ */

    { 0x210, AF_ADJUST_UP }, /* Ȑ */
    { 0x211, AF_ADJUST_UP }, /* ȑ */
    { 0x212, AF_ADJUST_UP }, /* Ȓ */
    { 0x213, AF_ADJUST_UP }, /* ȓ */
    { 0x214, AF_ADJUST_UP }, /* Ȕ */
    { 0x215, AF_ADJUST_UP }, /* ȕ */
    { 0x216, AF_ADJUST_UP }, /* Ȗ */
    { 0x217, AF_ADJUST_UP }, /* ȗ */
    { 0x218, AF_ADJUST_DOWN }, /* Ș */
    { 0x219, AF_ADJUST_DOWN }, /* ș */
    { 0x21A, AF_ADJUST_DOWN }, /* Ț */
    { 0x21B, AF_ADJUST_DOWN }, /* ț */
    { 0x21E, AF_ADJUST_UP }, /* Ȟ */
    { 0x21F, AF_ADJUST_UP }, /* ȟ */

    { 0x224, AF_IGNORE_CAPITAL_BOTTOM }, /* Ȥ */
    { 0x225, AF_IGNORE_SMALL_BOTTOM }, /* ȥ */
    { 0x226, AF_ADJUST_UP }, /* Ȧ */
    { 0x227, AF_ADJUST_UP }, /* ȧ */
    { 0x228, AF_IGNORE_CAPITAL_BOTTOM }, /* Ȩ */
    { 0x229, AF_IGNORE_SMALL_BOTTOM }, /* ȩ */
    { 0x22A, AF_ADJUST_UP2 }, /* Ȫ */
    { 0x22B, AF_ADJUST_UP2 }, /* ȫ */
    { 0x22C, AF_ADJUST_UP2 }, /* Ȭ */
    { 0x22D, AF_ADJUST_UP2 }, /* ȭ */
    { 0x22E, AF_ADJUST_UP }, /* Ȯ */
    { 0x22F, AF_ADJUST_UP }, /* ȯ */

    { 0x230, AF_ADJUST_UP2 }, /* Ȱ */
    { 0x231, AF_ADJUST_UP2 }, /* ȱ */
    { 0x232, AF_ADJUST_UP }, /* Ȳ */
    { 0x233, AF_ADJUST_UP }, /* ȳ */
    { 0x23A, AF_IGNORE_CAPITAL_TOP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ⱥ */
    { 0x23B, AF_IGNORE_CAPITAL_TOP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ȼ */
    { 0x23F, AF_IGNORE_SMALL_BOTTOM }, /* ȿ */

    { 0x240, AF_IGNORE_SMALL_BOTTOM }, /* ɀ */
    { 0x249, AF_ADJUST_UP }, /* ɉ */

    /* IPA Extensions */
    { 0x256, AF_IGNORE_SMALL_BOTTOM }, /* ɖ */

    { 0x260, AF_IGNORE_SMALL_TOP }, /* ɠ */
    { 0x267, AF_IGNORE_SMALL_BOTTOM }, /* ɧ */
    { 0x268, AF_ADJUST_UP }, /* ɨ */

    { 0x272, AF_IGNORE_SMALL_BOTTOM }, /* ɲ */
    { 0x273, AF_IGNORE_SMALL_BOTTOM }, /* ɳ */
    { 0x27B, AF_IGNORE_SMALL_BOTTOM }, /* ɻ */
    { 0x27D, AF_IGNORE_SMALL_BOTTOM }, /* ɽ */

    { 0x282, AF_IGNORE_SMALL_BOTTOM }, /* ʂ */
    { 0x288, AF_IGNORE_SMALL_BOTTOM }, /* ʈ */

    { 0x290, AF_IGNORE_SMALL_BOTTOM }, /* ʐ */
    { 0x29B, AF_IGNORE_SMALL_TOP }, /* ʛ */

    { 0x2A0, AF_IGNORE_SMALL_TOP }, /* ʠ */

    /* Spacing Modifier Letters */
    { 0x2B2, AF_ADJUST_UP }, /* ʲ */
    { 0x2B5, AF_IGNORE_SMALL_BOTTOM }, /* ʵ */

    /* Greek and Coptic */
    { 0x390, AF_ADJUST_UP2 }, /* ΐ */

    { 0x3AA, AF_ADJUST_UP }, /* Ϊ */
    { 0x3AB, AF_ADJUST_UP }, /* Ϋ */
    { 0x3AC, AF_ADJUST_UP }, /* ά */
    { 0x3AD, AF_ADJUST_UP }, /* έ */
    { 0x3AE, AF_ADJUST_UP }, /* ή */
    { 0x3AF, AF_ADJUST_UP }, /* ί */

    { 0x3B0, AF_ADJUST_UP2 }, /* ΰ */

    { 0x3CA, AF_ADJUST_UP }, /* ϊ */
    { 0x3CB, AF_ADJUST_UP }, /* ϋ */
    { 0x3CC, AF_ADJUST_UP }, /* ό */
    { 0x3CD, AF_ADJUST_UP }, /* ύ */
    { 0x3CE, AF_ADJUST_UP }, /* ώ */
    { 0x3CF, AF_IGNORE_CAPITAL_BOTTOM }, /* Ϗ */

    { 0x3D4, AF_ADJUST_UP }, /* ϔ */
    { 0x3D7, AF_IGNORE_SMALL_BOTTOM }, /* ϗ */
    { 0x3D9, AF_IGNORE_SMALL_BOTTOM }, /* ϙ */

    { 0x3E2, AF_IGNORE_CAPITAL_BOTTOM }, /* Ϣ */
    { 0x3E3, AF_IGNORE_SMALL_BOTTOM }, /* ϣ */

    { 0x3F3, AF_ADJUST_UP }, /* ϳ */

    /* Cyrillic */
    { 0x400, AF_ADJUST_UP }, /* Ѐ */
    { 0x401, AF_ADJUST_UP }, /* Ё */
    { 0x403, AF_ADJUST_UP }, /* Ѓ */
    { 0x407, AF_ADJUST_UP }, /* Ї */
    { 0x40C, AF_ADJUST_UP }, /* Ќ */
    { 0x40D, AF_ADJUST_UP }, /* Ѝ */
    { 0x40E, AF_ADJUST_UP }, /* Ў */
    { 0x40F, AF_IGNORE_CAPITAL_BOTTOM }, /* Џ */

    { 0x419, AF_ADJUST_UP }, /* Й */

    { 0x426, AF_IGNORE_CAPITAL_BOTTOM }, /* Ц */
    { 0x429, AF_IGNORE_CAPITAL_BOTTOM }, /* Щ */

    { 0x439, AF_ADJUST_UP }, /* й */

    { 0x446, AF_IGNORE_SMALL_BOTTOM }, /* ц */
    { 0x449, AF_IGNORE_SMALL_BOTTOM }, /* щ */

    { 0x450, AF_ADJUST_UP }, /* ѐ */
    { 0x451, AF_ADJUST_UP }, /* ё */
    { 0x453, AF_ADJUST_UP }, /* ѓ */
    { 0x456, AF_ADJUST_UP }, /* і */
    { 0x457, AF_ADJUST_UP }, /* ї */
    { 0x458, AF_ADJUST_UP }, /* ј */
    { 0x45C, AF_ADJUST_UP }, /* ќ */
    { 0x45D, AF_ADJUST_UP }, /* ѝ */
    { 0x45E, AF_ADJUST_UP }, /* ў */
    { 0x45F, AF_IGNORE_SMALL_BOTTOM }, /* џ */

    { 0x476, AF_ADJUST_UP }, /* Ѷ */
    { 0x477, AF_ADJUST_UP }, /* ѷ */
    { 0x47C, AF_ADJUST_UP2 }, /* Ѽ */
    { 0x47D, AF_ADJUST_UP2 }, /* ѽ */
    { 0x47E, AF_ADJUST_UP }, /* Ѿ */
    { 0x47F, AF_ADJUST_UP }, /* ѿ */

    { 0x480, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҁ */
    { 0x481, AF_IGNORE_SMALL_BOTTOM }, /* ҁ */
    { 0x48A, AF_ADJUST_UP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ҋ */
    { 0x48B, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* ҋ */

    { 0x490, AF_IGNORE_CAPITAL_TOP }, /* Ґ */
    { 0x491, AF_IGNORE_SMALL_TOP }, /* ґ */
    { 0x496, AF_IGNORE_CAPITAL_BOTTOM }, /* Җ */
    { 0x497, AF_IGNORE_SMALL_BOTTOM }, /* җ */
    { 0x498, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҙ */
    { 0x499, AF_IGNORE_SMALL_BOTTOM }, /* ҙ */
    { 0x49A, AF_IGNORE_CAPITAL_BOTTOM }, /* Қ */
    { 0x49B, AF_IGNORE_SMALL_BOTTOM }, /* қ */

    { 0x4A2, AF_IGNORE_CAPITAL_BOTTOM }, /* Ң */
    { 0x4A3, AF_IGNORE_SMALL_BOTTOM }, /* ң */
    { 0x4AA, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҫ */
    { 0x4AB, AF_IGNORE_SMALL_BOTTOM }, /* ҫ */
    { 0x4AC, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҭ */
    { 0x4AD, AF_IGNORE_SMALL_BOTTOM }, /* ҭ */

    { 0x4B2, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҳ */
    { 0x4B3, AF_IGNORE_SMALL_BOTTOM }, /* ҳ */
    { 0x4B4, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҵ */
    { 0x4B5, AF_IGNORE_SMALL_BOTTOM }, /* ҵ */
    { 0x4B6, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҷ */
    { 0x4B7, AF_IGNORE_SMALL_BOTTOM }, /* ҷ */
    { 0x4BE, AF_IGNORE_CAPITAL_BOTTOM }, /* Ҿ */
    { 0x4BF, AF_IGNORE_SMALL_BOTTOM }, /* ҿ */

    { 0x4C1, AF_ADJUST_UP }, /* Ӂ */
    { 0x4C2, AF_ADJUST_UP }, /* ӂ */
    { 0x4C5, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӆ */
    { 0x4C6, AF_IGNORE_SMALL_BOTTOM }, /* ӆ */
    { 0x4C9, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӊ */
    { 0x4CA, AF_IGNORE_SMALL_BOTTOM }, /* ӊ */
    { 0x4CB, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӌ */
    { 0x4CC, AF_IGNORE_SMALL_BOTTOM }, /* ӌ */
    { 0x4CD, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӎ */
    { 0x4CE, AF_IGNORE_SMALL_BOTTOM }, /* ӎ */

    { 0x4D0, AF_ADJUST_UP }, /* Ӑ */
    { 0x4D1, AF_ADJUST_UP }, /* ӑ */
    { 0x4D2, AF_ADJUST_UP }, /* Ӓ */
    { 0x4D3, AF_ADJUST_UP }, /* ӓ */
    { 0x4D6, AF_ADJUST_UP }, /* Ӗ */
    { 0x4D7, AF_ADJUST_UP }, /* ӗ */
    { 0x4DA, AF_ADJUST_UP }, /* Ӛ */
    { 0x4DB, AF_ADJUST_UP }, /* ӛ */
    { 0x4DC, AF_ADJUST_UP }, /* Ӝ */
    { 0x4DD, AF_ADJUST_UP }, /* ӝ */
    { 0x4DE, AF_ADJUST_UP }, /* Ӟ */
    { 0x4DF, AF_ADJUST_UP }, /* ӟ */

    { 0x4E2, AF_ADJUST_UP }, /* Ӣ */
    { 0x4E3, AF_ADJUST_UP }, /* ӣ */
    { 0x4E4, AF_ADJUST_UP }, /* Ӥ */
    { 0x4E5, AF_ADJUST_UP }, /* ӥ */
    { 0x4E6, AF_ADJUST_UP }, /* Ӧ */
    { 0x4E7, AF_ADJUST_UP }, /* ӧ */
    { 0x4EA, AF_ADJUST_UP }, /* Ӫ */
    { 0x4EB, AF_ADJUST_UP }, /* ӫ */
    { 0x4EC, AF_ADJUST_UP }, /* Ӭ */
    { 0x4ED, AF_ADJUST_UP }, /* ӭ */
    { 0x4EE, AF_ADJUST_UP }, /* Ӯ */
    { 0x4EF, AF_ADJUST_UP }, /* ӯ */

    { 0x4F0, AF_ADJUST_UP }, /* Ӱ */
    { 0x4F1, AF_ADJUST_UP }, /* ӱ */
    { 0x4F2, AF_ADJUST_UP }, /* Ӳ */
    { 0x4F3, AF_ADJUST_UP }, /* ӳ */
    { 0x4F4, AF_ADJUST_UP }, /* Ӵ */
    { 0x4F5, AF_ADJUST_UP }, /* ӵ */
    { 0x4F6, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӷ */
    { 0x4F7, AF_IGNORE_SMALL_BOTTOM }, /* ӷ */
    { 0x4F8, AF_ADJUST_UP }, /* Ӹ */
    { 0x4F9, AF_ADJUST_UP }, /* ӹ */
    { 0x4FA, AF_IGNORE_CAPITAL_BOTTOM }, /* Ӻ */
    { 0x4FB, AF_IGNORE_SMALL_BOTTOM }, /* ӻ */

    /* Cyrillic Supplement */
    { 0x506, AF_IGNORE_CAPITAL_BOTTOM }, /* Ԇ */
    { 0x507, AF_IGNORE_SMALL_BOTTOM }, /* ԇ */

    { 0x524, AF_IGNORE_CAPITAL_BOTTOM }, /* Ԥ */
    { 0x525, AF_IGNORE_SMALL_BOTTOM }, /* ԥ */
    { 0x526, AF_IGNORE_CAPITAL_BOTTOM }, /* Ԧ */
    { 0x527, AF_IGNORE_SMALL_BOTTOM }, /* ԧ */
    { 0x52E, AF_IGNORE_CAPITAL_BOTTOM }, /* Ԯ */
    { 0x52F, AF_IGNORE_SMALL_BOTTOM }, /* ԯ */

    /* Cherokee */
    { 0x13A5, AF_ADJUST_UP }, /* Ꭵ */

    /* Phonetic Extensions */
    { 0x1D09, AF_ADJUST_DOWN }, /* ᴉ */

    { 0x1D4E, AF_ADJUST_DOWN }, /* ᵎ */

    { 0x1D51, AF_IGNORE_SMALL_BOTTOM }, /* ᵑ */

    { 0x1D62, AF_ADJUST_UP }, /* ᵢ */

    /* Phonetic Extensions Supplement */
    { 0x1D80, AF_IGNORE_SMALL_BOTTOM }, /* ᶀ */
    { 0x1D81, AF_IGNORE_SMALL_BOTTOM }, /* ᶁ */
    { 0x1D82, AF_IGNORE_SMALL_BOTTOM }, /* ᶂ */
    { 0x1D84, AF_IGNORE_SMALL_BOTTOM }, /* ᶄ */
    { 0x1D85, AF_IGNORE_SMALL_BOTTOM }, /* ᶅ */
    { 0x1D86, AF_IGNORE_SMALL_BOTTOM }, /* ᶆ */
    { 0x1D87, AF_IGNORE_SMALL_BOTTOM }, /* ᶇ */
    { 0x1D89, AF_IGNORE_SMALL_BOTTOM }, /* ᶉ */
    { 0x1D8A, AF_IGNORE_SMALL_BOTTOM }, /* ᶊ */
    { 0x1D8C, AF_IGNORE_SMALL_BOTTOM }, /* ᶌ */
    { 0x1D8D, AF_IGNORE_SMALL_BOTTOM }, /* ᶍ */
    { 0x1D8E, AF_IGNORE_SMALL_BOTTOM }, /* ᶎ */
    { 0x1D8F, AF_IGNORE_SMALL_BOTTOM }, /* ᶏ */

    { 0x1D90, AF_IGNORE_SMALL_BOTTOM }, /* ᶐ */
    { 0x1D91, AF_IGNORE_SMALL_BOTTOM }, /* ᶑ */
    { 0x1D92, AF_IGNORE_SMALL_BOTTOM }, /* ᶒ */
    { 0x1D93, AF_IGNORE_SMALL_BOTTOM }, /* ᶓ */
    { 0x1D94, AF_IGNORE_SMALL_BOTTOM }, /* ᶔ */
    { 0x1D95, AF_IGNORE_SMALL_BOTTOM }, /* ᶕ */
    { 0x1D96, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* ᶖ */
    { 0x1D97, AF_IGNORE_SMALL_BOTTOM }, /* ᶗ */
    { 0x1D98, AF_IGNORE_SMALL_BOTTOM }, /* ᶘ */
    { 0x1D99, AF_IGNORE_SMALL_BOTTOM }, /* ᶙ */
    { 0x1D9A, AF_IGNORE_SMALL_BOTTOM }, /* ᶚ */

    { 0x1DA4, AF_ADJUST_UP }, /* ᶤ */
    { 0x1DA8, AF_ADJUST_UP }, /* ᶨ */
    { 0x1DA9, AF_IGNORE_SMALL_BOTTOM }, /* ᶩ */
    { 0x1DAA, AF_IGNORE_SMALL_BOTTOM }, /* ᶪ */
    { 0x1DAC, AF_IGNORE_SMALL_BOTTOM }, /* ᶬ */
    { 0x1DAE, AF_IGNORE_SMALL_BOTTOM }, /* ᶮ */
    { 0x1DAF, AF_IGNORE_SMALL_BOTTOM }, /* ᶯ */

    { 0x1DB3, AF_IGNORE_SMALL_BOTTOM }, /* ᶳ */
    { 0x1DB5, AF_IGNORE_SMALL_BOTTOM }, /* ᶵ */
    { 0x1DBC, AF_IGNORE_SMALL_BOTTOM }, /* ᶼ */

    /* Latin Extended Additional */
    { 0x1E00, AF_ADJUST_DOWN }, /* Ḁ */
    { 0x1E01, AF_ADJUST_DOWN }, /* ḁ */
    { 0x1E02, AF_ADJUST_UP }, /* Ḃ */
    { 0x1E03, AF_ADJUST_UP }, /* ḃ */
    { 0x1E04, AF_ADJUST_DOWN }, /* Ḅ */
    { 0x1E05, AF_ADJUST_DOWN }, /* ḅ */
    { 0x1E06, AF_ADJUST_DOWN }, /* Ḇ */
    { 0x1E07, AF_ADJUST_DOWN }, /* ḇ */
    { 0x1E08, AF_ADJUST_UP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ḉ */
    { 0x1E09, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* ḉ */
    { 0x1E0A, AF_ADJUST_UP }, /* Ḋ */
    { 0x1E0B, AF_ADJUST_UP }, /* ḋ */
    { 0x1E0C, AF_ADJUST_DOWN }, /* Ḍ */
    { 0x1E0D, AF_ADJUST_DOWN }, /* ḍ */
    { 0x1E0E, AF_ADJUST_DOWN }, /* Ḏ */
    { 0x1E0F, AF_ADJUST_DOWN }, /* ḏ */

    { 0x1E10, AF_ADJUST_DOWN }, /* Ḑ */
    { 0x1E11, AF_ADJUST_DOWN }, /* ḑ */
    { 0x1E12, AF_ADJUST_DOWN }, /* Ḓ */
    { 0x1E13, AF_ADJUST_DOWN }, /* ḓ */
    { 0x1E14, AF_ADJUST_UP2 }, /* Ḕ */
    { 0x1E15, AF_ADJUST_UP2 }, /* ḕ */
    { 0x1E16, AF_ADJUST_UP2 }, /* Ḗ */
    { 0x1E17, AF_ADJUST_UP2 }, /* ḗ */
    { 0x1E18, AF_ADJUST_DOWN }, /* Ḙ */
    { 0x1E19, AF_ADJUST_DOWN }, /* ḙ */
    { 0x1E1A, AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* Ḛ */
    { 0x1E1B, AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* ḛ */
    { 0x1E1C, AF_ADJUST_UP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ḝ */
    { 0x1E1D, AF_ADJUST_UP | AF_IGNORE_SMALL_BOTTOM }, /* ḝ */
    { 0x1E1E, AF_ADJUST_UP }, /* Ḟ */
    { 0x1E1F, AF_ADJUST_UP }, /* ḟ */

    { 0x1E20, AF_ADJUST_UP }, /* Ḡ */
    { 0x1E21, AF_ADJUST_UP }, /* ḡ */
    { 0x1E22, AF_ADJUST_UP }, /* Ḣ */
    { 0x1E23, AF_ADJUST_UP }, /* ḣ */
    { 0x1E24, AF_ADJUST_DOWN }, /* Ḥ */
    { 0x1E25, AF_ADJUST_DOWN }, /* ḥ */
    { 0x1E26, AF_ADJUST_UP }, /* Ḧ */
    { 0x1E27, AF_ADJUST_UP }, /* ḧ */
    { 0x1E28, AF_IGNORE_CAPITAL_BOTTOM }, /* Ḩ */
    { 0x1E29, AF_IGNORE_SMALL_BOTTOM }, /* ḩ */
    { 0x1E2A, AF_ADJUST_DOWN }, /* Ḫ */
    { 0x1E2B, AF_ADJUST_DOWN }, /* ḫ */
    { 0x1E2C, AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* Ḭ */
    { 0x1E2D, AF_ADJUST_UP | AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* ḭ */
    { 0x1E2E, AF_ADJUST_UP2 }, /* Ḯ */
    { 0x1E2F, AF_ADJUST_UP2 }, /* ḯ */

    { 0x1E30, AF_ADJUST_UP }, /* Ḱ */
    { 0x1E31, AF_ADJUST_UP }, /* ḱ */
    { 0x1E32, AF_ADJUST_DOWN }, /* Ḳ */
    { 0x1E33, AF_ADJUST_DOWN }, /* ḳ */
    { 0x1E34, AF_ADJUST_DOWN }, /* Ḵ */
    { 0x1E35, AF_ADJUST_DOWN }, /* ḵ */
    { 0x1E36, AF_ADJUST_DOWN }, /* Ḷ */
    { 0x1E37, AF_ADJUST_DOWN }, /* ḷ */
    { 0x1E38, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ḹ */
    { 0x1E39, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ḹ */
    { 0x1E3A, AF_ADJUST_DOWN }, /* Ḻ */
    { 0x1E3B, AF_ADJUST_DOWN }, /* ḻ */
    { 0x1E3C, AF_ADJUST_DOWN }, /* Ḽ */
    { 0x1E3D, AF_ADJUST_DOWN }, /* ḽ */
    { 0x1E3E, AF_ADJUST_UP }, /* Ḿ */
    { 0x1E3F, AF_ADJUST_UP }, /* ḿ */

    { 0x1E40, AF_ADJUST_UP }, /* Ṁ */
    { 0x1E41, AF_ADJUST_UP }, /* ṁ */
    { 0x1E42, AF_ADJUST_DOWN }, /* Ṃ */
    { 0x1E43, AF_ADJUST_DOWN }, /* ṃ */
    { 0x1E44, AF_ADJUST_UP }, /* Ṅ */
    { 0x1E45, AF_ADJUST_UP }, /* ṅ */
    { 0x1E46, AF_ADJUST_DOWN }, /* Ṇ */
    { 0x1E47, AF_ADJUST_DOWN }, /* ṇ */
    { 0x1E48, AF_ADJUST_DOWN }, /* Ṉ */
    { 0x1E49, AF_ADJUST_DOWN }, /* ṉ */
    { 0x1E4A, AF_ADJUST_DOWN }, /* Ṋ */
    { 0x1E4B, AF_ADJUST_DOWN }, /* ṋ */
    { 0x1E4C, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* Ṍ */
    { 0x1E4D, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* ṍ */
    { 0x1E4E, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* Ṏ */
    { 0x1E4F, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* ṏ */

    { 0x1E50, AF_ADJUST_UP2 }, /* Ṑ */
    { 0x1E51, AF_ADJUST_UP2 }, /* ṑ */
    { 0x1E52, AF_ADJUST_UP2 }, /* Ṓ */
    { 0x1E53, AF_ADJUST_UP2 }, /* ṓ */
    { 0x1E54, AF_ADJUST_UP }, /* Ṕ */
    { 0x1E55, AF_ADJUST_UP }, /* ṕ */
    { 0x1E56, AF_ADJUST_UP }, /* Ṗ */
    { 0x1E57, AF_ADJUST_UP }, /* ṗ */
    { 0x1E58, AF_ADJUST_UP }, /* Ṙ */
    { 0x1E59, AF_ADJUST_UP }, /* ṙ */
    { 0x1E5A, AF_ADJUST_DOWN }, /* Ṛ */
    { 0x1E5B, AF_ADJUST_DOWN }, /* ṛ */
    { 0x1E5C, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ṝ */
    { 0x1E5D, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ṝ */
    { 0x1E5E, AF_ADJUST_DOWN }, /* Ṟ */
    { 0x1E5F, AF_ADJUST_DOWN }, /* ṟ */

    { 0x1E60, AF_ADJUST_UP }, /* Ṡ */
    { 0x1E61, AF_ADJUST_UP }, /* ṡ */
    { 0x1E62, AF_ADJUST_DOWN }, /* Ṣ */
    { 0x1E63, AF_ADJUST_DOWN }, /* ṣ */
    { 0x1E64, AF_ADJUST_UP }, /* Ṥ */
    { 0x1E65, AF_ADJUST_UP }, /* ṥ */
    { 0x1E66, AF_ADJUST_UP }, /* Ṧ */
    { 0x1E67, AF_ADJUST_UP }, /* ṧ */
    { 0x1E68, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ṩ */
    { 0x1E69, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ṩ */
    { 0x1E6A, AF_ADJUST_UP }, /* Ṫ */
    { 0x1E6B, AF_ADJUST_UP }, /* ṫ */
    { 0x1E6C, AF_ADJUST_DOWN }, /* Ṭ */
    { 0x1E6D, AF_ADJUST_DOWN }, /* ṭ */
    { 0x1E6E, AF_ADJUST_DOWN }, /* Ṯ */
    { 0x1E6F, AF_ADJUST_DOWN }, /* ṯ */

    { 0x1E70, AF_ADJUST_DOWN }, /* Ṱ */
    { 0x1E71, AF_ADJUST_DOWN }, /* ṱ */
    { 0x1E72, AF_ADJUST_DOWN }, /* Ṳ */
    { 0x1E73, AF_ADJUST_DOWN }, /* ṳ */
    { 0x1E74, AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* Ṵ */
    { 0x1E75, AF_ADJUST_DOWN | AF_ADJUST_TILDE_BOTTOM }, /* ṵ */
    { 0x1E76, AF_ADJUST_DOWN }, /* Ṷ */
    { 0x1E77, AF_ADJUST_DOWN }, /* ṷ */
    { 0x1E78, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* Ṹ */
    { 0x1E79, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP2 }, /* ṹ */
    { 0x1E7A, AF_ADJUST_UP2 }, /* Ṻ */
    { 0x1E7B, AF_ADJUST_UP2 }, /* ṻ */
    { 0x1E7C, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ṽ */
    { 0x1E7D, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ṽ */
    { 0x1E7E, AF_ADJUST_DOWN }, /* Ṿ */
    { 0x1E7F, AF_ADJUST_DOWN }, /* ṿ */

    { 0x1E80, AF_ADJUST_UP }, /* Ẁ */
    { 0x1E81, AF_ADJUST_UP }, /* ẁ */
    { 0x1E82, AF_ADJUST_UP }, /* Ẃ */
    { 0x1E83, AF_ADJUST_UP }, /* ẃ */
    { 0x1E84, AF_ADJUST_UP }, /* Ẅ */
    { 0x1E85, AF_ADJUST_UP }, /* ẅ */
    { 0x1E86, AF_ADJUST_UP }, /* Ẇ */
    { 0x1E87, AF_ADJUST_UP }, /* ẇ */
    { 0x1E88, AF_ADJUST_DOWN }, /* Ẉ */
    { 0x1E89, AF_ADJUST_DOWN }, /* ẉ */
    { 0x1E8A, AF_ADJUST_UP }, /* Ẋ */
    { 0x1E8B, AF_ADJUST_UP }, /* ẋ */
    { 0x1E8C, AF_ADJUST_UP }, /* Ẍ */
    { 0x1E8D, AF_ADJUST_UP }, /* ẍ */
    { 0x1E8E, AF_ADJUST_UP }, /* Ẏ */
    { 0x1E8F, AF_ADJUST_UP }, /* ẏ */

    { 0x1E90, AF_ADJUST_UP }, /* Ẑ */
    { 0x1E91, AF_ADJUST_UP }, /* ẑ */
    { 0x1E92, AF_ADJUST_DOWN }, /* Ẓ */
    { 0x1E93, AF_ADJUST_DOWN }, /* ẓ */
    { 0x1E94, AF_ADJUST_DOWN }, /* Ẕ */
    { 0x1E95, AF_ADJUST_DOWN }, /* ẕ */
    { 0x1E96, AF_ADJUST_DOWN }, /* ẖ */
    { 0x1E97, AF_ADJUST_UP }, /* ẗ */
    { 0x1E98, AF_ADJUST_UP }, /* ẘ */
    { 0x1E99, AF_ADJUST_UP }, /* ẙ */
    { 0x1E9A, AF_ADJUST_UP }, /* ẚ */
    { 0x1E9B, AF_ADJUST_UP }, /* ẛ */

    { 0x1EA0, AF_ADJUST_DOWN }, /* Ạ */
    { 0x1EA1, AF_ADJUST_DOWN }, /* ạ */
    { 0x1EA2, AF_ADJUST_UP }, /* Ả */
    { 0x1EA3, AF_ADJUST_UP }, /* ả */
    { 0x1EA4, AF_ADJUST_UP2 }, /* Ấ */
    { 0x1EA5, AF_ADJUST_UP2 }, /* ấ */
    { 0x1EA6, AF_ADJUST_UP2 }, /* Ầ */
    { 0x1EA7, AF_ADJUST_UP2 }, /* ầ */
    { 0x1EA8, AF_ADJUST_UP2 }, /* Ẩ */
    { 0x1EA9, AF_ADJUST_UP2 }, /* ẩ */
    { 0x1EAA, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* Ẫ */
    { 0x1EAB, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ẫ */
    { 0x1EAC, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ậ */
    { 0x1EAD, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ậ */
    { 0x1EAE, AF_ADJUST_UP2 }, /* Ắ */
    { 0x1EAF, AF_ADJUST_UP2 }, /* ắ */

    { 0x1EB0, AF_ADJUST_UP2 }, /* Ằ */
    { 0x1EB1, AF_ADJUST_UP2 }, /* ằ */
    { 0x1EB2, AF_ADJUST_UP2 }, /* Ẳ */
    { 0x1EB3, AF_ADJUST_UP2 }, /* ẳ */
    { 0x1EB4, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* Ẵ */
    { 0x1EB5, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ẵ */
    { 0x1EB6, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ặ */
    { 0x1EB7, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ặ */
    { 0x1EB8, AF_ADJUST_DOWN }, /* Ẹ */
    { 0x1EB9, AF_ADJUST_DOWN }, /* ẹ */
    { 0x1EBA, AF_ADJUST_UP }, /* Ẻ */
    { 0x1EBB, AF_ADJUST_UP }, /* ẻ */
    { 0x1EBC, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ẽ */
    { 0x1EBD, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ẽ */
    { 0x1EBE, AF_ADJUST_UP2 }, /* Ế */
    { 0x1EBF, AF_ADJUST_UP2 }, /* ế */

    { 0x1EC0, AF_ADJUST_UP2 }, /* Ề */
    { 0x1EC1, AF_ADJUST_UP2 }, /* ề */
    { 0x1EC2, AF_ADJUST_UP2 }, /* Ể */
    { 0x1EC3, AF_ADJUST_UP2 }, /* ể */
    { 0x1EC4, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* Ễ */
    { 0x1EC5, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ễ */
    { 0x1EC6, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ệ */
    { 0x1EC7, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ệ */
    { 0x1EC8, AF_ADJUST_UP }, /* Ỉ */
    { 0x1EC9, AF_ADJUST_UP }, /* ỉ */
    { 0x1ECA, AF_ADJUST_DOWN }, /* Ị */
    { 0x1ECB, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ị */
    { 0x1ECC, AF_ADJUST_DOWN }, /* Ọ */
    { 0x1ECD, AF_ADJUST_DOWN }, /* ọ */
    { 0x1ECE, AF_ADJUST_UP }, /* Ỏ */
    { 0x1ECF, AF_ADJUST_UP }, /* ỏ */

    { 0x1ED0, AF_ADJUST_UP2 }, /* Ố */
    { 0x1ED1, AF_ADJUST_UP2 }, /* ố */
    { 0x1ED2, AF_ADJUST_UP2 }, /* Ồ */
    { 0x1ED3, AF_ADJUST_UP2 }, /* ồ */
    { 0x1ED4, AF_ADJUST_UP2 }, /* Ổ */
    { 0x1ED5, AF_ADJUST_UP2 }, /* ổ */
    { 0x1ED6, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* Ỗ */
    { 0x1ED7, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ỗ */
    { 0x1ED8, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* Ộ */
    { 0x1ED9, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ộ */
    { 0x1EDA, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ớ */
    { 0x1EDB, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ớ */
    { 0x1EDC, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ờ */
    { 0x1EDD, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ờ */
    { 0x1EDE, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ở */
    { 0x1EDF, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ở */

    { 0x1EE0, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_IGNORE_CAPITAL_TOP }, /* Ỡ */
    { 0x1EE1, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_IGNORE_SMALL_TOP }, /* ỡ */
    { 0x1EE2, AF_ADJUST_DOWN | AF_IGNORE_CAPITAL_TOP }, /* Ợ */
    { 0x1EE3, AF_ADJUST_DOWN | AF_IGNORE_SMALL_TOP }, /* ợ */
    { 0x1EE4, AF_ADJUST_DOWN }, /* Ụ */
    { 0x1EE5, AF_ADJUST_DOWN }, /* ụ */
    { 0x1EE6, AF_ADJUST_UP }, /* Ủ */
    { 0x1EE7, AF_ADJUST_UP }, /* ủ */
    { 0x1EE8, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ứ */
    { 0x1EE9, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ứ */
    { 0x1EEA, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ừ */
    { 0x1EEB, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ừ */
    { 0x1EEC, AF_ADJUST_UP | AF_IGNORE_CAPITAL_TOP }, /* Ử */
    { 0x1EED, AF_ADJUST_UP | AF_IGNORE_SMALL_TOP }, /* ử */
    { 0x1EEE, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_IGNORE_CAPITAL_TOP }, /* Ữ */
    { 0x1EEF, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_IGNORE_SMALL_TOP }, /* ữ */

    { 0x1EF0, AF_ADJUST_DOWN | AF_IGNORE_CAPITAL_TOP }, /* Ự */
    { 0x1EF1, AF_ADJUST_DOWN | AF_IGNORE_SMALL_TOP }, /* ự */
    { 0x1EF2, AF_ADJUST_UP }, /* Ỳ */
    { 0x1EF3, AF_ADJUST_UP }, /* ỳ */
    { 0x1EF4, AF_ADJUST_DOWN }, /* Ỵ */
    { 0x1EF5, AF_ADJUST_DOWN }, /* ỵ */
    { 0x1EF6, AF_ADJUST_UP }, /* Ỷ */
    { 0x1EF7, AF_ADJUST_UP }, /* ỷ */
    { 0x1EF8, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* Ỹ */
    { 0x1EF9, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ỹ */

    /* Greek Extended */
    { 0x1F00, AF_ADJUST_UP }, /* ἀ */
    { 0x1F01, AF_ADJUST_UP }, /* ἁ */
    { 0x1F02, AF_ADJUST_UP }, /* ἂ */
    { 0x1F03, AF_ADJUST_UP }, /* ἃ */
    { 0x1F04, AF_ADJUST_UP }, /* ἄ */
    { 0x1F05, AF_ADJUST_UP }, /* ἅ */
    { 0x1F06, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἆ */
    { 0x1F07, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἇ */

    { 0x1F10, AF_ADJUST_UP }, /* ἐ */
    { 0x1F11, AF_ADJUST_UP }, /* ἑ */
    { 0x1F12, AF_ADJUST_UP }, /* ἒ */
    { 0x1F13, AF_ADJUST_UP }, /* ἓ */
    { 0x1F14, AF_ADJUST_UP }, /* ἔ */
    { 0x1F15, AF_ADJUST_UP }, /* ἕ */

    { 0x1F20, AF_ADJUST_UP }, /* ἠ */
    { 0x1F21, AF_ADJUST_UP }, /* ἡ */
    { 0x1F22, AF_ADJUST_UP }, /* ἢ */
    { 0x1F23, AF_ADJUST_UP }, /* ἣ */
    { 0x1F24, AF_ADJUST_UP }, /* ἤ */
    { 0x1F25, AF_ADJUST_UP }, /* ἥ */
    { 0x1F26, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἦ */
    { 0x1F27, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἧ */

    { 0x1F30, AF_ADJUST_UP }, /* ἰ */
    { 0x1F31, AF_ADJUST_UP }, /* ἱ */
    { 0x1F32, AF_ADJUST_UP }, /* ἲ */
    { 0x1F33, AF_ADJUST_UP }, /* ἳ */
    { 0x1F34, AF_ADJUST_UP }, /* ἴ */
    { 0x1F35, AF_ADJUST_UP }, /* ἵ */
    { 0x1F36, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἶ */
    { 0x1F37, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ἷ */

    { 0x1F40, AF_ADJUST_UP }, /* ὀ */
    { 0x1F41, AF_ADJUST_UP }, /* ὁ */
    { 0x1F42, AF_ADJUST_UP }, /* ὂ */
    { 0x1F43, AF_ADJUST_UP }, /* ὃ */
    { 0x1F44, AF_ADJUST_UP }, /* ὄ */
    { 0x1F45, AF_ADJUST_UP }, /* ὅ */

    { 0x1F50, AF_ADJUST_UP }, /* ὐ */
    { 0x1F51, AF_ADJUST_UP }, /* ὑ */
    { 0x1F52, AF_ADJUST_UP }, /* ὒ */
    { 0x1F53, AF_ADJUST_UP }, /* ὓ */
    { 0x1F54, AF_ADJUST_UP }, /* ὔ */
    { 0x1F55, AF_ADJUST_UP }, /* ὕ */
    { 0x1F56, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ὖ */
    { 0x1F57, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ὗ */

    { 0x1F60, AF_ADJUST_UP }, /* ὠ */
    { 0x1F61, AF_ADJUST_UP }, /* ὡ */
    { 0x1F62, AF_ADJUST_UP }, /* ὢ */
    { 0x1F63, AF_ADJUST_UP }, /* ὣ */
    { 0x1F64, AF_ADJUST_UP }, /* ὤ */
    { 0x1F65, AF_ADJUST_UP }, /* ὥ */
    { 0x1F66, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ὦ */
    { 0x1F67, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ὧ */

    { 0x1F70, AF_ADJUST_UP }, /* ὰ */
    { 0x1F71, AF_ADJUST_UP }, /* ά */
    { 0x1F72, AF_ADJUST_UP }, /* ὲ */
    { 0x1F73, AF_ADJUST_UP }, /* έ */
    { 0x1F74, AF_ADJUST_UP }, /* ὴ */
    { 0x1F75, AF_ADJUST_UP }, /* ή */
    { 0x1F76, AF_ADJUST_UP }, /* ὶ */
    { 0x1F77, AF_ADJUST_UP }, /* ί */
    { 0x1F78, AF_ADJUST_UP }, /* ὸ */
    { 0x1F79, AF_ADJUST_UP }, /* ό */
    { 0x1F7A, AF_ADJUST_UP }, /* ὺ */
    { 0x1F7B, AF_ADJUST_UP }, /* ύ */
    { 0x1F7C, AF_ADJUST_UP }, /* ὼ */
    { 0x1F7D, AF_ADJUST_UP }, /* ώ */

    { 0x1F80, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾀ */
    { 0x1F81, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾁ */
    { 0x1F82, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾂ */
    { 0x1F83, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾃ */
    { 0x1F84, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾄ */
    { 0x1F85, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾅ */
    { 0x1F86, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾆ */
    { 0x1F87, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾇ */
    { 0x1F88, AF_ADJUST_DOWN }, /* ᾈ */
    { 0x1F89, AF_ADJUST_DOWN }, /* ᾉ */
    { 0x1F8A, AF_ADJUST_DOWN }, /* ᾊ */
    { 0x1F8B, AF_ADJUST_DOWN }, /* ᾋ */
    { 0x1F8C, AF_ADJUST_DOWN }, /* ᾌ */
    { 0x1F8D, AF_ADJUST_DOWN }, /* ᾍ */
    { 0x1F8E, AF_ADJUST_DOWN }, /* ᾎ */
    { 0x1F8F, AF_ADJUST_DOWN }, /* ᾏ */

    { 0x1F90, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾐ */
    { 0x1F91, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾑ */
    { 0x1F92, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾒ */
    { 0x1F93, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾓ */
    { 0x1F94, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾔ */
    { 0x1F95, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾕ */
    { 0x1F96, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾖ */
    { 0x1F97, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾗ */
    { 0x1F98, AF_ADJUST_DOWN }, /* ᾘ */
    { 0x1F99, AF_ADJUST_DOWN }, /* ᾙ */
    { 0x1F9A, AF_ADJUST_DOWN }, /* ᾚ */
    { 0x1F9B, AF_ADJUST_DOWN }, /* ᾛ */
    { 0x1F9C, AF_ADJUST_DOWN }, /* ᾜ */
    { 0x1F9D, AF_ADJUST_DOWN }, /* ᾝ */
    { 0x1F9E, AF_ADJUST_DOWN }, /* ᾞ */
    { 0x1F9F, AF_ADJUST_DOWN }, /* ᾟ */

    { 0x1FA0, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾠ */
    { 0x1FA1, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾡ */
    { 0x1FA2, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾢ */
    { 0x1FA3, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾣ */
    { 0x1FA4, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾤ */
    { 0x1FA5, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾥ */
    { 0x1FA6, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾦ */
    { 0x1FA7, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾧ */
    { 0x1FA8, AF_ADJUST_DOWN }, /* ᾨ */
    { 0x1FA9, AF_ADJUST_DOWN }, /* ᾩ */
    { 0x1FAA, AF_ADJUST_DOWN }, /* ᾪ */
    { 0x1FAB, AF_ADJUST_DOWN }, /* ᾫ */
    { 0x1FAC, AF_ADJUST_DOWN }, /* ᾬ */
    { 0x1FAD, AF_ADJUST_DOWN }, /* ᾭ */
    { 0x1FAE, AF_ADJUST_DOWN }, /* ᾮ */
    { 0x1FAF, AF_ADJUST_DOWN }, /* ᾯ */

    { 0x1FB0, AF_ADJUST_UP }, /* ᾰ */
    { 0x1FB1, AF_ADJUST_UP }, /* ᾱ */
    { 0x1FB2, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾲ */
    { 0x1FB3, AF_ADJUST_DOWN }, /* ᾳ */
    { 0x1FB4, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ᾴ */
    { 0x1FB6, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ᾶ */
    { 0x1FB7, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ᾷ */
    { 0x1FB8, AF_ADJUST_UP }, /* Ᾰ */
    { 0x1FB9, AF_ADJUST_UP }, /* Ᾱ */
    { 0x1FBC, AF_ADJUST_DOWN }, /* ᾼ */

    { 0x1FC2, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ῂ */
    { 0x1FC3, AF_ADJUST_DOWN }, /* ῃ */
    { 0x1FC4, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ῄ */
    { 0x1FC6, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ῆ */
    { 0x1FC7, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ῇ */
    { 0x1FCC, AF_ADJUST_DOWN }, /* ῌ */

    { 0x1FD0, AF_ADJUST_UP }, /* ῐ */
    { 0x1FD1, AF_ADJUST_UP }, /* ῑ */
    { 0x1FD2, AF_ADJUST_UP2 }, /* ῒ */
    { 0x1FD3, AF_ADJUST_UP2 }, /* ΐ */
    { 0x1FD6, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ῖ */
    { 0x1FD7, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ῗ */
    { 0x1FD8, AF_ADJUST_UP }, /* Ῐ */
    { 0x1FD9, AF_ADJUST_UP }, /* Ῑ */

    { 0x1FE0, AF_ADJUST_UP }, /* ῠ */
    { 0x1FE1, AF_ADJUST_UP }, /* ῡ */
    { 0x1FE2, AF_ADJUST_UP2 }, /* ῢ */
    { 0x1FE3, AF_ADJUST_UP2 }, /* ΰ */
    { 0x1FE4, AF_ADJUST_UP }, /* ῤ */
    { 0x1FE5, AF_ADJUST_UP }, /* ῥ */
    { 0x1FE6, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ῦ */
    { 0x1FE7, AF_ADJUST_UP2 | AF_ADJUST_TILDE_TOP }, /* ῧ */
    { 0x1FE8, AF_ADJUST_UP }, /* Ῠ */
    { 0x1FE9, AF_ADJUST_UP }, /* Ῡ */
    { 0x1FF2, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ῲ */
    { 0x1FF3, AF_ADJUST_DOWN }, /* ῳ */
    { 0x1FF4, AF_ADJUST_UP | AF_ADJUST_DOWN }, /* ῴ */
    { 0x1FF6, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP }, /* ῶ */
    { 0x1FF7, AF_ADJUST_UP | AF_ADJUST_TILDE_TOP | AF_ADJUST_DOWN }, /* ῷ */
    { 0x1FFC, AF_ADJUST_DOWN }, /* ῼ */

    /* General Punctuation */
    { 0x203C, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ‼ */
    { 0x203D, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ‽ */

    { 0x2047, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ⁇ */
    { 0x2048, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ⁈ */
    { 0x2049, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ⁉ */

    /* Superscripts and Subscripts */
    { 0x2071, AF_ADJUST_UP }, /* ⁱ */

    /* Currency Symbols */
    { 0x20AB, AF_ADJUST_DOWN }, /* ₫ */

    { 0x20C0, AF_ADJUST_DOWN }, /* ⃀ */

    /* Number Forms */
    { 0x2170, AF_ADJUST_UP }, /* ⅰ */
    { 0x2171, AF_ADJUST_UP }, /* ⅱ */
    { 0x2172, AF_ADJUST_UP }, /* ⅲ */
    { 0x2173, AF_ADJUST_UP }, /* ⅳ */
    { 0x2175, AF_ADJUST_UP }, /* ⅵ */
    { 0x2176, AF_ADJUST_UP }, /* ⅶ */
    { 0x2177, AF_ADJUST_UP }, /* ⅷ */
    { 0x2178, AF_ADJUST_UP }, /* ⅸ */
    { 0x217A, AF_ADJUST_UP }, /* ⅺ */
    { 0x217B, AF_ADJUST_UP }, /* ⅻ */

    /* Latin Extended-C */
    { 0x2C64, AF_IGNORE_CAPITAL_BOTTOM } , /* Ɽ */
    { 0x2C67, AF_IGNORE_CAPITAL_BOTTOM } , /* Ⱨ */
    { 0x2C68, AF_IGNORE_SMALL_BOTTOM } , /* ⱨ */
    { 0x2C69, AF_IGNORE_CAPITAL_BOTTOM } , /* Ⱪ */
    { 0x2C6A, AF_IGNORE_SMALL_BOTTOM } , /* ⱪ */
    { 0x2C6B, AF_IGNORE_CAPITAL_BOTTOM } , /* Ⱬ */
    { 0x2C6C, AF_IGNORE_SMALL_BOTTOM } , /* ⱬ */
    { 0x2C6E, AF_IGNORE_CAPITAL_BOTTOM } , /* Ɱ */

    { 0x2C7C, AF_ADJUST_UP }, /* ⱼ */
    { 0x2C7E, AF_IGNORE_CAPITAL_BOTTOM } , /* Ȿ */
    { 0x2C7F, AF_IGNORE_CAPITAL_BOTTOM } , /* Ɀ */

    /* Coptic */
    { 0x2CC2, AF_ADJUST_UP }, /* Ⳃ */
    { 0x2CC3, AF_ADJUST_UP }, /* ⳃ */

    /* Supplemental Punctuation */
    { 0x2E18, AF_ADJUST_UP }, /* ⸘ */

    { 0x2E2E, AF_ADJUST_UP | AF_ADJUST_NO_HEIGHT_CHECK }, /* ⸮ */

    /* Cyrillic Extended-B */
    { 0xA640, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꙁ */
    { 0xA641, AF_IGNORE_SMALL_BOTTOM } , /* ꙁ */
    { 0xA642, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꙃ */
    { 0xA643, AF_IGNORE_SMALL_BOTTOM } , /* ꙃ */

    { 0xA680, AF_IGNORE_CAPITAL_TOP } , /* Ꚁ */
    { 0xA681, AF_IGNORE_SMALL_TOP } , /* ꚁ */
    { 0xA688, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꚉ */
    { 0xA689, AF_IGNORE_SMALL_BOTTOM } , /* ꚉ */
    { 0xA68A, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꚋ */
    { 0xA68B, AF_IGNORE_SMALL_BOTTOM } , /* ꚋ */
    { 0xA68E, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꚏ */
    { 0xA68F, AF_IGNORE_SMALL_BOTTOM } , /* ꚏ */

    { 0xA690, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꚑ */
    { 0xA691, AF_IGNORE_SMALL_BOTTOM } , /* ꚑ */
    { 0xA696, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꚗ */
    { 0xA697, AF_IGNORE_SMALL_BOTTOM } , /* ꚗ */

    /* Latin Extended-D */
    { 0xA726, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꜧ */
    { 0xA727, AF_IGNORE_SMALL_BOTTOM } , /* ꜧ */

    { 0xA756, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꝗ */
    { 0xA758, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꝙ */

    { 0xA771, AF_IGNORE_SMALL_BOTTOM } , /* ꝱ */
    { 0xA772, AF_IGNORE_SMALL_BOTTOM } , /* ꝲ */
    { 0xA773, AF_IGNORE_SMALL_BOTTOM } , /* ꝳ */
    { 0xA774, AF_IGNORE_SMALL_BOTTOM } , /* ꝴ */
    { 0xA776, AF_IGNORE_SMALL_BOTTOM } , /* ꝶ */

    { 0xA790, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꞑ */
    { 0xA791, AF_IGNORE_SMALL_BOTTOM } , /* ꞑ */
    { 0xA794, AF_IGNORE_SMALL_BOTTOM } , /* ꞔ */
    { 0xA795, AF_IGNORE_SMALL_BOTTOM } , /* ꞕ */

    { 0xA7C0, AF_IGNORE_CAPITAL_TOP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ꟁ */
    { 0xA7C1, AF_IGNORE_SMALL_TOP | AF_IGNORE_SMALL_BOTTOM }, /* ꟁ */
    { 0xA7C4, AF_IGNORE_CAPITAL_BOTTOM } , /* Ꞔ */
    { 0xA7C5, AF_IGNORE_CAPITAL_BOTTOM } , /* Ʂ */
    { 0xA7C6, AF_IGNORE_CAPITAL_BOTTOM } , /* Ᶎ */
    { 0xA7CC, AF_IGNORE_CAPITAL_TOP | AF_IGNORE_CAPITAL_BOTTOM }, /* Ꟍ */
    { 0xA7CD, AF_IGNORE_SMALL_TOP | AF_IGNORE_SMALL_BOTTOM }, /* ꟍ */

    /* Latin Extended-E */
    { 0xAB3C, AF_IGNORE_SMALL_BOTTOM } , /* ꬼ */

    { 0xAB46, AF_IGNORE_SMALL_BOTTOM } , /* ꭆ */

    { 0xAB5C, AF_IGNORE_SMALL_BOTTOM } , /* ꭜ */

    { 0xAB66, AF_IGNORE_SMALL_BOTTOM } , /* ꭦ */
    { 0xAB67, AF_IGNORE_SMALL_BOTTOM } , /* ꭧ */
  };


  FT_LOCAL_DEF( FT_UInt32 )
  af_adjustment_database_lookup( FT_UInt32  codepoint )
  {
    /* Binary search for database entry */
    FT_Offset  low  = 0;
    FT_Offset  high = AF_ADJUSTMENT_DATABASE_LENGTH - 1;


    while ( high >= low )
    {
      FT_Offset  mid           = ( low + high ) / 2;
      FT_UInt32  mid_codepoint = adjustment_database[mid].codepoint;


      if ( mid_codepoint < codepoint )
        low = mid + 1;
      else if ( mid_codepoint > codepoint )
        high = mid - 1;
      else
        return adjustment_database[mid].flags;
    }

    return 0;
  }


#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

  static FT_Error
  add_substitute( FT_Int     glyph_idx,
                  size_t     value,
                  FT_UInt32  codepoint,
                  FT_Hash    reverse_map,
                  FT_Hash    subst_map,
                  FT_Memory  memory )
  {
    FT_Error  error;

    FT_Int  first_substitute = (FT_Int)( value & 0xFFFF );

    FT_UInt  used = reverse_map->used;


    /*
      OpenType features like 'unic' map lowercase letter glyphs to uppercase
      forms (and vice versa), which could lead to the use of wrong entries
      in the adjustment database.  For this reason we don't overwrite,
      prioritizing cmap entries.

      XXX Note, however, that this cannot cover all cases since there might
      be contradictory entries for glyphs not in the cmap.  A possible
      solution might be to specially mark pairs of related lowercase and
      uppercase characters in the adjustment database that have diacritics
      on different vertical sides (for example, U+0122 'Ģ' and U+0123 'ģ').
      The auto-hinter could then perform a topological analysis to do the
      right thing.
    */
    error = ft_hash_num_insert_no_overwrite( first_substitute, codepoint,
                                             reverse_map, memory );
    if ( error )
      return error;

    if ( reverse_map->used > used )
    {
      size_t*  subst = ft_hash_num_lookup( first_substitute, subst_map );


      if ( subst )
      {
        error = add_substitute( first_substitute, *subst, codepoint,
                                reverse_map, subst_map, memory );
        if ( error )
          return error;
      }
    }

    /* The remaining substitutes. */
    if ( value & 0xFFFF0000U )
    {
      FT_UInt  num_substitutes = value >> 16;

      FT_UInt  i;


      for ( i = 1; i <= num_substitutes; i++ )
      {
        FT_Int   idx        = glyph_idx + (FT_Int)( i << 16 );
        size_t*  substitute = ft_hash_num_lookup( idx, subst_map );


        used = reverse_map->used;

        error = ft_hash_num_insert_no_overwrite( *substitute,
                                                 codepoint,
                                                 reverse_map,
                                                 memory );
        if ( error )
          return error;

        if ( reverse_map->used > used )
        {
          size_t*  subst = ft_hash_num_lookup( *substitute, subst_map );


          if ( subst )
          {
            error = add_substitute( *substitute, *subst, codepoint,
                                    reverse_map, subst_map, memory );
            if ( error )
              return error;
          }
        }
      }
    }

    return FT_Err_Ok;
  }

#endif /* FT_CONFIG_OPTION_USE_HARFBUZZ */


  /* Construct a 'reverse cmap' (i.e., a mapping from glyph indices to   */
  /* character codes) for all glyphs that an input code point could turn */
  /* into.                                                               */
  /*                                                                     */
  /* If HarfBuzz support is not available, this is the direct inversion  */
  /* of the cmap table, otherwise the mapping gets extended with data    */
  /* from the 'GSUB' table.                                              */
  FT_LOCAL_DEF( FT_Error )
  af_reverse_character_map_new( FT_Hash         *map,
                                AF_StyleMetrics  metrics )
  {
    FT_Error  error;

    AF_FaceGlobals  globals = metrics->globals;
    FT_Face         face    = globals->face;
    FT_Memory       memory  = face->memory;

    FT_CharMap  old_charmap;

    FT_UInt32  codepoint;
    FT_Offset  i;


    FT_TRACE4(( "af_reverse_character_map_new:"
                " building reverse character map (style `%s')\n",
                af_style_names[metrics->style_class->style] ));

    /* Search for a unicode charmap.           */
    /* If there isn't one, create a blank map. */

    /* Back up `face->charmap` because `find_unicode_charmap` sets it. */
    old_charmap = face->charmap;

    if ( ( error = find_unicode_charmap( face ) ) )
      goto Exit;

    *map = NULL;
    if ( FT_QNEW( *map ) )
      goto Exit;

    error = ft_hash_num_init( *map, memory );
    if ( error )
      goto Exit;

    /* Initialize reverse cmap with data directly from the cmap table. */
    for ( i = 0; i < AF_ADJUSTMENT_DATABASE_LENGTH; i++ )
    {
      FT_Int  cmap_glyph;


      /*
        We cannot restrict `codepoint` to character ranges; we have no
        control what data the script-specific portion of the GSUB table
        actually holds.

        An example is `arial.ttf` version 7.00; in this font, there are
        lookups for Cyrillic (lookup 43), Greek (lookup 44), and Latin
        (lookup 45) that map capital letter glyphs to small capital glyphs.
        It is tempting to expect that script-specific versions of the 'c2sc'
        feature only use script-specific lookups.  However, this is not the
        case in this font: the feature uses all three lookups regardless of
        the script.

        The auto-hinter, while assigning glyphs to styles, uses the first
        coverage result it encounters for a particular glyph.  For example,
        if the coverage for Cyrillic is tested before Latin (as is currently
        the case), glyphs without a cmap entry that are covered in 'c2sc'
        are treated as Cyrillic.

        If we now look at glyph 3498, which is a small-caps version of the
        Latin character 'A grave' (U+00C0, glyph 172), we can see that it is
        registered as belonging to a Cyrillic style due to the algorithm
        just described.  As a result, checking only for characters from the
        Latin range would miss this glyph; we thus have to test all
        character codes in the database.
      */
      codepoint = adjustment_database[i].codepoint;

      cmap_glyph = (FT_Int)FT_Get_Char_Index( face, codepoint );
      if ( cmap_glyph == 0 )
        continue;

      error = ft_hash_num_insert( cmap_glyph, codepoint, *map, memory );
      if ( error )
        goto Exit;
    }

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

    if ( ft_hb_enabled( globals ) )
    {
      hb_font_t  *hb_font;
      hb_face_t  *hb_face;

      hb_set_t    *gsub_lookups;
      hb_script_t  script;

      unsigned int  script_count   = 1;
      hb_tag_t      script_tags[2] = { HB_TAG_NONE, HB_TAG_NONE };

      FT_Hash  subst_map = NULL;

      hb_codepoint_t  idx;
      FT_UInt         hash_idx;
      FT_Int          glyph_idx;
      size_t          value;


      /* No need to check whether HarfBuzz has allocation issues; */
      /* it continues to work in such cases and simply returns    */
      /* 'empty' objects that do nothing.                         */

      hb_font = globals->hb_font;
      hb_face = hb( font_get_face )( hb_font );

      gsub_lookups = hb( set_create )();

      script = af_hb_scripts[metrics->style_class->script];

      hb( ot_tags_from_script_and_language )( script, NULL,
                                              &script_count, script_tags,
                                              NULL, NULL );

      /* Compute set of all script-specific GSUB lookups. */
      hb( ot_layout_collect_lookups )( hb_face,
                                       HB_OT_TAG_GSUB,
                                       script_tags, NULL, NULL,
                                       gsub_lookups );

#ifdef FT_DEBUG_LEVEL_TRACE
      {
        FT_Bool  have_idx = FALSE;


        FT_TRACE4(( "  GSUB lookups to check:\n" ));

        FT_TRACE4(( "  " ));
        idx = HB_SET_VALUE_INVALID;
        while ( hb( set_next )( gsub_lookups, &idx ) )
          if ( globals->gsub_lookups_single_alternate[idx] )
          {
            have_idx = TRUE;
            FT_TRACE4(( "  %u", idx ));
          }
        if ( !have_idx )
          FT_TRACE4(( "  (none)" ));
        FT_TRACE4(( "\n" ));

        FT_TRACE4(( "\n" ));
      }
#endif

      if ( FT_QNEW( subst_map ) )
        goto Exit_HarfBuzz;

      error = ft_hash_num_init( subst_map, memory );
      if ( error )
        goto Exit_HarfBuzz;

      idx = HB_SET_VALUE_INVALID;
      while ( hb( set_next )( gsub_lookups, &idx ) )
      {
        FT_UInt32  offset = globals->gsub_lookups_single_alternate[idx];


        /* Put all substitutions into a single hash table.  Note that   */
        /* the hash values usually contain more than a single character */
        /* code; this can happen if different 'SingleSubst' subtables   */
        /* map a given glyph index to different substitutions, or if    */
        /* 'AlternateSubst' subtable entries are present.               */
        if ( offset )
          af_map_lookup( globals, subst_map, offset );
      }

      /*
        Now iterate over the collected substitution data in `subst_map`
        (using recursion to resolve one-to-many mappings) and insert the
        data into the reverse cmap.

        As an example, suppose we have the following cmap and substitution
        data:

          cmap: X -> a
                Y -> b
                Z -> c

          substitutions: a -> b
                         b -> c, d
                         d -> e

        The reverse map now becomes as follows.

          a -> X
          b -> Y
          c -> Z (via cmap, ignoring mapping from 'b')
          d -> Y (via 'b')
          e -> Y (via 'b' and 'd')
      */

      hash_idx = 0;
      while ( ft_hash_num_iterator( &hash_idx,
                                    &glyph_idx,
                                    &value,
                                    subst_map ) )
      {
        size_t*  val;


        /* Ignore keys that do not point to the first substitute. */
        if ( (FT_UInt)glyph_idx & 0xFFFF0000U )
          continue;

        /* Ignore glyph indices that are not related to accents. */
        val = ft_hash_num_lookup( glyph_idx, *map );
        if ( !val )
          continue;

        codepoint = *val;

        error = add_substitute( glyph_idx, value, codepoint,
                                *map, subst_map, memory );
        if ( error )
          break;
      }

    Exit_HarfBuzz:
      hb( set_destroy )( gsub_lookups );

      ft_hash_num_free( subst_map, memory );
      FT_FREE( subst_map );

      if ( error )
        goto Exit;
    }

#endif /* FT_CONFIG_OPTION_USE_HARFBUZZ */

    FT_TRACE4(( "    reverse character map built successfully"
                " with %u entries\n", ( *map )->used ));

#ifdef FT_DEBUG_LEVEL_TRACE

    {
      FT_UInt  cnt;


      FT_TRACE7(( "       gidx   code    flags\n" ));
               /* "      XXXXX  0xXXXX  XXXXXXXXXXX..." */
      FT_TRACE7(( "     ------------------------------\n" ));

      for ( cnt = 0; cnt < globals->glyph_count; cnt++ )
      {
        size_t*    val;
        FT_UInt32  adj_type;

        const char*  flag_names[] =
        {
          "up",          /* AF_ADJUST_UP    */
          "down",        /* AF_ADJUST_DOWN  */
          "double up",   /* AF_ADJUST_UP2   */
          "double down", /* AF_ADJUST_DOWN2 */

          "top tilde",          /* AF_ADJUST_TILDE_TOP     */
          "bottom tilde",       /* AF_ADJUST_TILDE_BOTTOM  */
          "below-top tilde",    /* AF_ADJUST_TILDE_TOP2    */
          "above-bottom tilde", /* AF_ADJUST_TILDE_BOTTOM2 */

          "ignore capital top",    /* AF_IGNORE_CAPITAL_TOP    */
          "ignore capital bottom", /* AF_IGNORE_CAPITAL_BOTTOM */
          "ignore small top",      /* AF_IGNORE_SMALL_TOP      */
          "ignore small bottom",   /* AF_IGNORE_SMALL_BOTTOM   */
        };
        size_t  flag_names_size = sizeof ( flag_names ) / sizeof ( char* );

        char  flag_str[256];
        int   need_comma;

        size_t  j;


        val = ft_hash_num_lookup( (FT_Int)cnt, *map );
        if ( !val )
          continue;
        codepoint = *val;

        adj_type = af_adjustment_database_lookup( codepoint );
        if ( !adj_type )
          continue;

        flag_str[0] = '\0';
        need_comma  = 0;

        for ( j = 0; j < flag_names_size; j++ )
        {
          if ( adj_type & (1 << j ) )
          {
            if ( !need_comma )
              need_comma = 1;
            else
              strcat( flag_str, ", " );
            strcat( flag_str, flag_names[j] );
          }
        }

        FT_TRACE7(( "      %5u  0x%04X  %s\n", cnt, codepoint, flag_str ));
      }
    }

#endif /* FT_DEBUG_LEVEL_TRACE */


  Exit:
    face->charmap = old_charmap;

    if ( error )
    {
      FT_TRACE4(( "    error while building reverse character map."
                  " Using blank map.\n" ));

      if ( *map )
        ft_hash_num_free( *map, memory );

      FT_FREE( *map );
      *map = NULL;
      return error;
    }

    return FT_Err_Ok;
  }


  FT_LOCAL_DEF( FT_Error )
  af_reverse_character_map_done( FT_Hash    map,
                                 FT_Memory  memory )
  {
    if ( map )
      ft_hash_num_free( map, memory );
    FT_FREE( map );

    return FT_Err_Ok;
  }


/* END */
