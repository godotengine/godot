/****************************************************************************
 *
 * afscript.h
 *
 *   Auto-fitter scripts (specification only).
 *
 * Copyright (C) 2013-2026 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /* The following part can be included multiple times. */
  /* Define `SCRIPT' as needed.                         */


  /* Add new scripts here.  The first and second arguments are the    */
  /* script name in lowercase and uppercase, respectively, followed   */
  /* by a description string.  Then comes the corresponding HarfBuzz  */
  /* script name tag, followed by a string of standard characters (to */
  /* derive the standard width and height of stems).                  */
  /*                                                                  */
  /* Note that fallback scripts only have a default style, thus we    */
  /* use `HB_SCRIPT_INVALID' as the HarfBuzz script name tag for      */
  /* them.                                                            */

  SCRIPT( adlm, ADLM,
          "Adlam",
          HB_SCRIPT_ADLAM,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x9E\xA4\x8C \xF0\x9E\xA4\xAE" ) /* 𞤌 𞤮 */

  SCRIPT( arab, ARAB,
          "Arabic",
          HB_SCRIPT_ARABIC,
          HINTING_BOTTOM_TO_TOP,
          "\xD9\x84 \xD8\xAD \xD9\x80" ) /* ل ح ـ */

  SCRIPT( armn, ARMN,
          "Armenian",
          HB_SCRIPT_ARMENIAN,
          HINTING_BOTTOM_TO_TOP,
          "\xD5\xBD \xD5\x8D" ) /* ս Ս */

  SCRIPT( avst, AVST,
          "Avestan",
          HB_SCRIPT_AVESTAN,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\xAC\x9A" ) /* 𐬚 */

  SCRIPT( bamu, BAMU,
          "Bamum",
          HB_SCRIPT_BAMUM,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\x9B\x81 \xEA\x9B\xAF" ) /* ꛁ ꛯ */

  /* there are no simple forms for letters; we thus use two digit shapes */
  SCRIPT( beng, BENG,
          "Bengali",
          HB_SCRIPT_BENGALI,
          HINTING_TOP_TO_BOTTOM,
          "\xE0\xA7\xA6 \xE0\xA7\xAA" ) /* ০ ৪ */

  SCRIPT( buhd, BUHD,
          "Buhid",
          HB_SCRIPT_BUHID,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x9D\x8B \xE1\x9D\x8F" ) /* ᝋ ᝏ */

  SCRIPT( cakm, CAKM,
          "Chakma",
          HB_SCRIPT_CHAKMA,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x91\x84\xA4 \xF0\x91\x84\x89 \xF0\x91\x84\x9B" ) /* 𑄤 𑄉 𑄛 */

  SCRIPT( cans, CANS,
          "Canadian Syllabics",
          HB_SCRIPT_CANADIAN_SYLLABICS,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x91\x8C \xE1\x93\x9A" ) /* ᑌ ᓚ */

  SCRIPT( cari, CARI,
          "Carian",
          HB_SCRIPT_CARIAN,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\x8A\xAB \xF0\x90\x8B\x89" ) /* 𐊫 𐋉 */

  SCRIPT( cher, CHER,
          "Cherokee",
          HB_SCRIPT_CHEROKEE,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x8E\xA4 \xE1\x8F\x85 \xEA\xAE\x95" ) /* Ꭴ Ꮕ ꮕ */

  SCRIPT( copt, COPT,
          "Coptic",
          HB_SCRIPT_COPTIC,
          HINTING_BOTTOM_TO_TOP,
          "\xE2\xB2\x9E \xE2\xB2\x9F" ) /* Ⲟ ⲟ */

  SCRIPT( cprt, CPRT,
          "Cypriot",
          HB_SCRIPT_CYPRIOT,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\xA0\x85 \xF0\x90\xA0\xA3" ) /* 𐠅 𐠣 */

  SCRIPT( cyrl, CYRL,
          "Cyrillic",
          HB_SCRIPT_CYRILLIC,
          HINTING_BOTTOM_TO_TOP,
          "\xD0\xBE \xD0\x9E" ) /* о О */

  SCRIPT( deva, DEVA,
          "Devanagari",
          HB_SCRIPT_DEVANAGARI,
          HINTING_TOP_TO_BOTTOM,
          "\xE0\xA4\xA0 \xE0\xA4\xB5 \xE0\xA4\x9F" ) /* ठ व ट */

  SCRIPT( dsrt, DSRT,
          "Deseret",
          HB_SCRIPT_DESERET,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\x90\x84 \xF0\x90\x90\xAC" ) /* 𐐄 𐐬 */

  SCRIPT( ethi, ETHI,
          "Ethiopic",
          HB_SCRIPT_ETHIOPIC,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x8B\x90" ) /* ዐ */

  SCRIPT( geor, GEOR,
          "Georgian (Mkhedruli)",
          HB_SCRIPT_GEORGIAN,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x83\x98 \xE1\x83\x94 \xE1\x83\x90 \xE1\xB2\xBF" ) /* ი ე ა Ი */

  SCRIPT( geok, GEOK,
          "Georgian (Khutsuri)",
          HB_SCRIPT_INVALID,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x82\xB6 \xE1\x82\xB1 \xE2\xB4\x99" ) /* Ⴖ Ⴑ ⴙ */

  SCRIPT( glag, GLAG,
          "Glagolitic",
          HB_SCRIPT_GLAGOLITIC,
          HINTING_BOTTOM_TO_TOP,
          "\xE2\xB0\x95 \xE2\xB1\x85" ) /* Ⱅ ⱅ */

  SCRIPT( goth, GOTH,
          "Gothic",
          HB_SCRIPT_GOTHIC,
          HINTING_TOP_TO_BOTTOM,
          "\xF0\x90\x8C\xB4 \xF0\x90\x8C\xBE \xF0\x90\x8D\x83" ) /* 𐌴 𐌾 𐍃 */

  SCRIPT( grek, GREK,
          "Greek",
          HB_SCRIPT_GREEK,
          HINTING_BOTTOM_TO_TOP,
          "\xCE\xBF \xCE\x9F" ) /* ο Ο */

  SCRIPT( gujr, GUJR,
          "Gujarati",
          HB_SCRIPT_GUJARATI,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xAA\x9F \xE0\xAB\xA6" ) /* ટ ૦ */

  SCRIPT( guru, GURU,
          "Gurmukhi",
          HB_SCRIPT_GURMUKHI,
          HINTING_TOP_TO_BOTTOM,
          "\xE0\xA8\xA0 \xE0\xA8\xB0 \xE0\xA9\xA6" ) /* ਠ ਰ ੦ */

  SCRIPT( hebr, HEBR,
          "Hebrew",
          HB_SCRIPT_HEBREW,
          HINTING_BOTTOM_TO_TOP,
          "\xD7\x9D" ) /* ם */

  SCRIPT( kali, KALI,
          "Kayah Li",
          HB_SCRIPT_KAYAH_LI,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\xA4\x8D \xEA\xA4\x80" ) /* ꤍ ꤀ */

  /* only digit zero has a simple shape in the Khmer script */
  SCRIPT( khmr, KHMR,
          "Khmer",
          HB_SCRIPT_KHMER,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x9F\xA0" ) /* ០ */

  SCRIPT( khms, KHMS,
          "Khmer Symbols",
          HB_SCRIPT_INVALID,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\xA7\xA1 \xE1\xA7\xAA" ) /* ᧡ ᧪ */

  SCRIPT( knda, KNDA,
          "Kannada",
          HB_SCRIPT_KANNADA,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xB3\xA6 \xE0\xB2\xAC" ) /* ೦ ಬ */

  /* only digit zero has a simple shape in the Lao script */
  SCRIPT( lao, LAO,
          "Lao",
          HB_SCRIPT_LAO,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xBB\x90" ) /* ໐ */

  SCRIPT( latn, LATN,
          "Latin",
          HB_SCRIPT_LATIN,
          HINTING_BOTTOM_TO_TOP,
          "o O 0" )

  SCRIPT( latb, LATB,
          "Latin Subscript Fallback",
          HB_SCRIPT_INVALID,
          HINTING_BOTTOM_TO_TOP,
          "\xE2\x82\x92 \xE2\x82\x80" ) /* ₒ ₀ */

  SCRIPT( latp, LATP,
          "Latin Superscript Fallback",
          HB_SCRIPT_INVALID,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\xB5\x92 \xE1\xB4\xBC \xE2\x81\xB0" ) /* ᵒ ᴼ ⁰ */

  SCRIPT( lisu, LISU,
          "Lisu",
          HB_SCRIPT_LISU,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\x93\xB3" ) /* ꓳ */

  SCRIPT( mlym, MLYM,
          "Malayalam",
          HB_SCRIPT_MALAYALAM,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xB4\xA0 \xE0\xB4\xB1" ) /* ഠ റ */

  SCRIPT( medf, MEDF,
          "Medefaidrin",
          HB_SCRIPT_MEDEFAIDRIN,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x96\xB9\xA1 \xF0\x96\xB9\x9B \xF0\x96\xB9\xAF" ) /* 𖹡 𖹛 𖹯 */

  SCRIPT( mong, MONG,
          "Mongolian",
          HB_SCRIPT_MONGOLIAN,
          HINTING_TOP_TO_BOTTOM,
          "\xE1\xA1\x82 \xE1\xA0\xAA" ) /* ᡂ ᠪ */

  SCRIPT( mymr, MYMR,
          "Myanmar",
          HB_SCRIPT_MYANMAR,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\x80\x9D \xE1\x80\x84 \xE1\x80\x82" ) /* ဝ င ဂ */

  SCRIPT( nkoo, NKOO,
          "N'Ko",
          HB_SCRIPT_NKO,
          HINTING_BOTTOM_TO_TOP,
          "\xDF\x8B \xDF\x80" ) /* ߋ ߀ */

  SCRIPT( none, NONE,
          "no script",
          HB_SCRIPT_INVALID,
          HINTING_BOTTOM_TO_TOP,
          "" )

  SCRIPT( olck, OLCK,
          "Ol Chiki",
          HB_SCRIPT_OL_CHIKI,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\xB1\x9B" ) /* ᱛ */

  SCRIPT( orkh, ORKH,
          "Old Turkic",
          HB_SCRIPT_OLD_TURKIC,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\xB0\x97" ) /* 𐰗 */

  SCRIPT( osge, OSGE,
          "Osage",
          HB_SCRIPT_OSAGE,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\x93\x82 \xF0\x90\x93\xAA" ) /* 𐓂 𐓪 */

  SCRIPT( osma, OSMA,
          "Osmanya",
          HB_SCRIPT_OSMANYA,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\x92\x86 \xF0\x90\x92\xA0" ) /* 𐒆 𐒠 */

  SCRIPT( rohg, ROHG,
          "Hanifi Rohingya",
          HB_SCRIPT_HANIFI_ROHINGYA,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\xB4\xB0" ) /* 𐴰 */

  SCRIPT( saur, SAUR,
          "Saurashtra",
          HB_SCRIPT_SAURASHTRA,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\xA2\x9D \xEA\xA3\x90" ) /* ꢝ ꣐ */

  SCRIPT( shaw, SHAW,
          "Shavian",
          HB_SCRIPT_SHAVIAN,
          HINTING_BOTTOM_TO_TOP,
          "\xF0\x90\x91\xB4" ) /* 𐑴 */

  SCRIPT( sinh, SINH,
          "Sinhala",
          HB_SCRIPT_SINHALA,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xB6\xA7" ) /* ට */

  /* only digit zero has a simple (round) shape in the Sundanese script */
  SCRIPT( sund, SUND,
          "Sundanese",
          HB_SCRIPT_SUNDANESE,
          HINTING_BOTTOM_TO_TOP,
          "\xE1\xAE\xB0" ) /* ᮰ */

  /* only digit zero has a simple (round) shape in the Tamil script */
  SCRIPT( taml, TAML,
          "Tamil",
          HB_SCRIPT_TAMIL,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xAF\xA6" ) /* ௦ */

  SCRIPT( tavt, TAVT,
          "Tai Viet",
          HB_SCRIPT_TAI_VIET,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\xAA\x92 \xEA\xAA\xAB" ) /* ꪒ ꪫ */

  /* there are no simple forms for letters; we thus use two digit shapes */
  SCRIPT( telu, TELU,
          "Telugu",
          HB_SCRIPT_TELUGU,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xB1\xA6 \xE0\xB1\xA7" ) /* ౦ ౧ */

  SCRIPT( tfng, TFNG,
          "Tifinagh",
          HB_SCRIPT_TIFINAGH,
          HINTING_BOTTOM_TO_TOP,
          "\xE2\xB5\x94" ) /* ⵔ */

  SCRIPT( thai, THAI,
          "Thai",
          HB_SCRIPT_THAI,
          HINTING_BOTTOM_TO_TOP,
          "\xE0\xB8\xB2 \xE0\xB9\x85 \xE0\xB9\x90" ) /* า ๅ ๐ */

  SCRIPT( vaii, VAII,
          "Vai",
          HB_SCRIPT_VAI,
          HINTING_BOTTOM_TO_TOP,
          "\xEA\x98\x93 \xEA\x96\x9C \xEA\x96\xB4" ) /* ꘓ ꖜ ꖴ */

#ifdef AF_CONFIG_OPTION_INDIC

  SCRIPT( limb, LIMB,
          "Limbu",
          HB_SCRIPT_LIMBU,
          HINTING_BOTTOM_TO_TOP,
          "o" ) /* XXX */

  SCRIPT( orya, ORYA,
          "Oriya",
          HB_SCRIPT_ORIYA,
          HINTING_BOTTOM_TO_TOP,
          "o" ) /* XXX */

  SCRIPT( sylo, SYLO,
          "Syloti Nagri",
          HB_SCRIPT_SYLOTI_NAGRI,
          HINTING_BOTTOM_TO_TOP,
          "o" ) /* XXX */

  SCRIPT( tibt, TIBT,
          "Tibetan",
          HB_SCRIPT_TIBETAN,
          HINTING_BOTTOM_TO_TOP,
          "o" ) /* XXX */

#endif /* AF_CONFIG_OPTION_INDIC */

#ifdef AF_CONFIG_OPTION_CJK

  SCRIPT( hani, HANI,
          "CJKV ideographs",
          HB_SCRIPT_HAN,
          HINTING_BOTTOM_TO_TOP,
          "\xE7\x94\xB0 \xE5\x9B\x97" ) /* 田 囗 */

#endif /* AF_CONFIG_OPTION_CJK */


/* END */
