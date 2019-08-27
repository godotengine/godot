/****************************************************************************
 *
 * afstyles.h
 *
 *   Auto-fitter styles (specification only).
 *
 * Copyright (C) 2013-2019 by
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
  /* Define `STYLE' as needed.                          */


  /* Add new styles here.  The first and second arguments are the  */
  /* style name in lowercase and uppercase, respectively, followed */
  /* by a description string.  The next arguments are the          */
  /* corresponding writing system, script, blue stringset, and     */
  /* coverage.                                                     */
  /*                                                               */
  /* Note that styles using `AF_COVERAGE_DEFAULT' should always    */
  /* come after styles with other coverages.  Also note that       */
  /* fallback scripts only use `AF_COVERAGE_DEFAULT' for its       */
  /* style.                                                        */
  /*                                                               */
  /* Example:                                                      */
  /*                                                               */
  /*   STYLE( cyrl_dflt, CYRL_DFLT,                                */
  /*          "Cyrillic default style",                            */
  /*          AF_WRITING_SYSTEM_LATIN,                             */
  /*          AF_SCRIPT_CYRL,                                      */
  /*          AF_BLUE_STRINGSET_CYRL,                              */
  /*          AF_COVERAGE_DEFAULT )                                */

#undef  STYLE_LATIN
#define STYLE_LATIN( s, S, f, F, ds, df, C ) \
          STYLE( s ## _ ## f, S ## _ ## F,   \
                 ds " " df " style",         \
                 AF_WRITING_SYSTEM_LATIN,    \
                 AF_SCRIPT_ ## S,            \
                 AF_BLUE_STRINGSET_ ## S,    \
                 AF_COVERAGE_ ## C )

#undef  META_STYLE_LATIN
#define META_STYLE_LATIN( s, S, ds )                     \
          STYLE_LATIN( s, S, c2cp, C2CP, ds,             \
                       "petite capitals from capitals", \
                       PETITE_CAPITALS_FROM_CAPITALS )   \
          STYLE_LATIN( s, S, c2sc, C2SC, ds,             \
                       "small capitals from capitals",  \
                       SMALL_CAPITALS_FROM_CAPITALS )    \
          STYLE_LATIN( s, S, ordn, ORDN, ds,             \
                       "ordinals",                       \
                       ORDINALS )                        \
          STYLE_LATIN( s, S, pcap, PCAP, ds,             \
                       "petite capitals",                \
                       PETITE_CAPITALS )                 \
          STYLE_LATIN( s, S, sinf, SINF, ds,             \
                       "scientific inferiors",           \
                       SCIENTIFIC_INFERIORS )            \
          STYLE_LATIN( s, S, smcp, SMCP, ds,             \
                       "small capitals",                 \
                       SMALL_CAPITALS )                  \
          STYLE_LATIN( s, S, subs, SUBS, ds,             \
                       "subscript",                      \
                       SUBSCRIPT )                       \
          STYLE_LATIN( s, S, sups, SUPS, ds,             \
                       "superscript",                    \
                       SUPERSCRIPT )                     \
          STYLE_LATIN( s, S, titl, TITL, ds,             \
                       "titling",                        \
                       TITLING )                         \
          STYLE_LATIN( s, S, dflt, DFLT, ds,             \
                       "default",                        \
                       DEFAULT )


  STYLE( adlm_dflt, ADLM_DFLT,
         "Adlam default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_ADLM,
         AF_BLUE_STRINGSET_ADLM,
         AF_COVERAGE_DEFAULT )

  STYLE( arab_dflt, ARAB_DFLT,
         "Arabic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_ARAB,
         AF_BLUE_STRINGSET_ARAB,
         AF_COVERAGE_DEFAULT )

  STYLE( armn_dflt, ARMN_DFLT,
         "Armenian default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_ARMN,
         AF_BLUE_STRINGSET_ARMN,
         AF_COVERAGE_DEFAULT )

  STYLE( avst_dflt, AVST_DFLT,
         "Avestan default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_AVST,
         AF_BLUE_STRINGSET_AVST,
         AF_COVERAGE_DEFAULT )

  STYLE( bamu_dflt, BAMU_DFLT,
         "Bamum default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_BAMU,
         AF_BLUE_STRINGSET_BAMU,
         AF_COVERAGE_DEFAULT )

  STYLE( beng_dflt, BENG_DFLT,
         "Bengali default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_BENG,
         AF_BLUE_STRINGSET_BENG,
         AF_COVERAGE_DEFAULT )

  STYLE( buhd_dflt, BUHD_DFLT,
         "Buhid default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_BUHD,
         AF_BLUE_STRINGSET_BUHD,
         AF_COVERAGE_DEFAULT )

  STYLE( cakm_dflt, CAKM_DFLT,
         "Chakma default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_CAKM,
         AF_BLUE_STRINGSET_CAKM,
         AF_COVERAGE_DEFAULT )

  STYLE( cans_dflt, CANS_DFLT,
         "Canadian Syllabics default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_CANS,
         AF_BLUE_STRINGSET_CANS,
         AF_COVERAGE_DEFAULT )

  STYLE( cari_dflt, CARI_DFLT,
         "Carian default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_CARI,
         AF_BLUE_STRINGSET_CARI,
         AF_COVERAGE_DEFAULT )

  STYLE( cher_dflt, CHER_DFLT,
         "Cherokee default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_CHER,
         AF_BLUE_STRINGSET_CHER,
         AF_COVERAGE_DEFAULT )

  STYLE( copt_dflt, COPT_DFLT,
         "Coptic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_COPT,
         AF_BLUE_STRINGSET_COPT,
         AF_COVERAGE_DEFAULT )

  STYLE( cprt_dflt, CPRT_DFLT,
         "Cypriot default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_CPRT,
         AF_BLUE_STRINGSET_CPRT,
         AF_COVERAGE_DEFAULT )

  META_STYLE_LATIN( cyrl, CYRL, "Cyrillic" )

  STYLE( deva_dflt, DEVA_DFLT,
         "Devanagari default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_DEVA,
         AF_BLUE_STRINGSET_DEVA,
         AF_COVERAGE_DEFAULT )

  STYLE( dsrt_dflt, DSRT_DFLT,
         "Deseret default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_DSRT,
         AF_BLUE_STRINGSET_DSRT,
         AF_COVERAGE_DEFAULT )

  STYLE( ethi_dflt, ETHI_DFLT,
         "Ethiopic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_ETHI,
         AF_BLUE_STRINGSET_ETHI,
         AF_COVERAGE_DEFAULT )

  STYLE( geor_dflt, GEOR_DFLT,
         "Georgian (Mkhedruli) default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GEOR,
         AF_BLUE_STRINGSET_GEOR,
         AF_COVERAGE_DEFAULT )

  STYLE( geok_dflt, GEOK_DFLT,
         "Georgian (Khutsuri) default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GEOK,
         AF_BLUE_STRINGSET_GEOK,
         AF_COVERAGE_DEFAULT )

  STYLE( glag_dflt, GLAG_DFLT,
         "Glagolitic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GLAG,
         AF_BLUE_STRINGSET_GLAG,
         AF_COVERAGE_DEFAULT )

  STYLE( goth_dflt, GOTH_DFLT,
         "Gothic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GOTH,
         AF_BLUE_STRINGSET_GOTH,
         AF_COVERAGE_DEFAULT )

  META_STYLE_LATIN( grek, GREK, "Greek" )

  STYLE( gujr_dflt, GUJR_DFLT,
         "Gujarati default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GUJR,
         AF_BLUE_STRINGSET_GUJR,
         AF_COVERAGE_DEFAULT )

  STYLE( guru_dflt, GURU_DFLT,
         "Gurmukhi default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_GURU,
         AF_BLUE_STRINGSET_GURU,
         AF_COVERAGE_DEFAULT )

  STYLE( hebr_dflt, HEBR_DFLT,
         "Hebrew default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_HEBR,
         AF_BLUE_STRINGSET_HEBR,
         AF_COVERAGE_DEFAULT )

  STYLE( kali_dflt, KALI_DFLT,
         "Kayah Li default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_KALI,
         AF_BLUE_STRINGSET_KALI,
         AF_COVERAGE_DEFAULT )

  STYLE( khmr_dflt, KHMR_DFLT,
         "Khmer default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_KHMR,
         AF_BLUE_STRINGSET_KHMR,
         AF_COVERAGE_DEFAULT )

  STYLE( khms_dflt, KHMS_DFLT,
         "Khmer Symbols default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_KHMS,
         AF_BLUE_STRINGSET_KHMS,
         AF_COVERAGE_DEFAULT )

  STYLE( knda_dflt, KNDA_DFLT,
         "Kannada default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_KNDA,
         AF_BLUE_STRINGSET_KNDA,
         AF_COVERAGE_DEFAULT )

  STYLE( lao_dflt, LAO_DFLT,
         "Lao default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_LAO,
         AF_BLUE_STRINGSET_LAO,
         AF_COVERAGE_DEFAULT )

  META_STYLE_LATIN( latn, LATN, "Latin" )

  STYLE( latb_dflt, LATB_DFLT,
         "Latin subscript fallback default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_LATB,
         AF_BLUE_STRINGSET_LATB,
         AF_COVERAGE_DEFAULT )

  STYLE( latp_dflt, LATP_DFLT,
         "Latin superscript fallback default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_LATP,
         AF_BLUE_STRINGSET_LATP,
         AF_COVERAGE_DEFAULT )

#ifdef FT_OPTION_AUTOFIT2
  STYLE( ltn2_dflt, LTN2_DFLT,
         "Latin 2 default style",
         AF_WRITING_SYSTEM_LATIN2,
         AF_SCRIPT_LATN,
         AF_BLUE_STRINGSET_LATN,
         AF_COVERAGE_DEFAULT )
#endif

  STYLE( lisu_dflt, LISU_DFLT,
         "Lisu default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_LISU,
         AF_BLUE_STRINGSET_LISU,
         AF_COVERAGE_DEFAULT )

  STYLE( mlym_dflt, MLYM_DFLT,
         "Malayalam default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_MLYM,
         AF_BLUE_STRINGSET_MLYM,
         AF_COVERAGE_DEFAULT )

  STYLE( mong_dflt, MONG_DFLT,
         "Mongolian default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_MONG,
         AF_BLUE_STRINGSET_MONG,
         AF_COVERAGE_DEFAULT )

  STYLE( mymr_dflt, MYMR_DFLT,
         "Myanmar default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_MYMR,
         AF_BLUE_STRINGSET_MYMR,
         AF_COVERAGE_DEFAULT )

  STYLE( nkoo_dflt, NKOO_DFLT,
         "N'Ko default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_NKOO,
         AF_BLUE_STRINGSET_NKOO,
         AF_COVERAGE_DEFAULT )

  STYLE( none_dflt, NONE_DFLT,
         "no style",
         AF_WRITING_SYSTEM_DUMMY,
         AF_SCRIPT_NONE,
         AF_BLUE_STRINGSET_NONE,
         AF_COVERAGE_DEFAULT )

  STYLE( olck_dflt, OLCK_DFLT,
         "Ol Chiki default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_OLCK,
         AF_BLUE_STRINGSET_OLCK,
         AF_COVERAGE_DEFAULT )

  STYLE( orkh_dflt, ORKH_DFLT,
         "Old Turkic default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_ORKH,
         AF_BLUE_STRINGSET_ORKH,
         AF_COVERAGE_DEFAULT )

  STYLE( osge_dflt, OSGE_DFLT,
         "Osage default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_OSGE,
         AF_BLUE_STRINGSET_OSGE,
         AF_COVERAGE_DEFAULT )

  STYLE( osma_dflt, OSMA_DFLT,
         "Osmanya default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_OSMA,
         AF_BLUE_STRINGSET_OSMA,
         AF_COVERAGE_DEFAULT )

  STYLE( saur_dflt, SAUR_DFLT,
         "Saurashtra default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_SAUR,
         AF_BLUE_STRINGSET_SAUR,
         AF_COVERAGE_DEFAULT )

  STYLE( shaw_dflt, SHAW_DFLT,
         "Shavian default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_SHAW,
         AF_BLUE_STRINGSET_SHAW,
         AF_COVERAGE_DEFAULT )

  STYLE( sinh_dflt, SINH_DFLT,
         "Sinhala default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_SINH,
         AF_BLUE_STRINGSET_SINH,
         AF_COVERAGE_DEFAULT )

  STYLE( sund_dflt, SUND_DFLT,
         "Sundanese default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_SUND,
         AF_BLUE_STRINGSET_SUND,
         AF_COVERAGE_DEFAULT )

  STYLE( taml_dflt, TAML_DFLT,
         "Tamil default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_TAML,
         AF_BLUE_STRINGSET_TAML,
         AF_COVERAGE_DEFAULT )

  STYLE( tavt_dflt, TAVT_DFLT,
         "Tai Viet default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_TAVT,
         AF_BLUE_STRINGSET_TAVT,
         AF_COVERAGE_DEFAULT )

  STYLE( telu_dflt, TELU_DFLT,
         "Telugu default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_TELU,
         AF_BLUE_STRINGSET_TELU,
         AF_COVERAGE_DEFAULT )

  STYLE( tfng_dflt, TFNG_DFLT,
         "Tifinagh default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_TFNG,
         AF_BLUE_STRINGSET_TFNG,
         AF_COVERAGE_DEFAULT )

  STYLE( thai_dflt, THAI_DFLT,
         "Thai default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_THAI,
         AF_BLUE_STRINGSET_THAI,
         AF_COVERAGE_DEFAULT )

  STYLE( vaii_dflt, VAII_DFLT,
         "Vai default style",
         AF_WRITING_SYSTEM_LATIN,
         AF_SCRIPT_VAII,
         AF_BLUE_STRINGSET_VAII,
         AF_COVERAGE_DEFAULT )

#ifdef AF_CONFIG_OPTION_INDIC

  /* no blue stringset support for the Indic writing system yet */
#undef  STYLE_DEFAULT_INDIC
#define STYLE_DEFAULT_INDIC( s, S, d )    \
          STYLE( s ## _dflt, S ## _DFLT,  \
                 d " default style",      \
                 AF_WRITING_SYSTEM_INDIC, \
                 AF_SCRIPT_ ## S,         \
                 (AF_Blue_Stringset)0,    \
                 AF_COVERAGE_DEFAULT )

  STYLE_DEFAULT_INDIC( limb, LIMB, "Limbu" )
  STYLE_DEFAULT_INDIC( orya, ORYA, "Oriya" )
  STYLE_DEFAULT_INDIC( sylo, SYLO, "Syloti Nagri" )
  STYLE_DEFAULT_INDIC( tibt, TIBT, "Tibetan" )

#endif /* AF_CONFIG_OPTION_INDIC */

#ifdef AF_CONFIG_OPTION_CJK

  STYLE( hani_dflt, HANI_DFLT,
         "CJKV ideographs default style",
         AF_WRITING_SYSTEM_CJK,
         AF_SCRIPT_HANI,
         AF_BLUE_STRINGSET_HANI,
         AF_COVERAGE_DEFAULT )

#endif /* AF_CONFIG_OPTION_CJK */


/* END */
