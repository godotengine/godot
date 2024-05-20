/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2022 University of Cambridge

This module is auto-generated from Unicode data files. DO NOT EDIT MANUALLY!
Instead, modify the maint/GenerateUcpHeader.py script and run it to generate
a new version of this code.

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

#ifndef PCRE2_UCP_H_IDEMPOTENT_GUARD
#define PCRE2_UCP_H_IDEMPOTENT_GUARD

/* This file contains definitions of the Unicode property values that are
returned by the UCD access macros and used throughout PCRE2.

IMPORTANT: The specific values of the first two enums (general and particular
character categories) are assumed by the table called catposstab in the file
pcre2_auto_possess.c. They are unlikely to change, but should be checked after
an update. */

/* These are the general character categories. */

enum {
  ucp_C,
  ucp_L,
  ucp_M,
  ucp_N,
  ucp_P,
  ucp_S,
  ucp_Z,
};

/* These are the particular character categories. */

enum {
  ucp_Cc,    /* Control */
  ucp_Cf,    /* Format */
  ucp_Cn,    /* Unassigned */
  ucp_Co,    /* Private use */
  ucp_Cs,    /* Surrogate */
  ucp_Ll,    /* Lower case letter */
  ucp_Lm,    /* Modifier letter */
  ucp_Lo,    /* Other letter */
  ucp_Lt,    /* Title case letter */
  ucp_Lu,    /* Upper case letter */
  ucp_Mc,    /* Spacing mark */
  ucp_Me,    /* Enclosing mark */
  ucp_Mn,    /* Non-spacing mark */
  ucp_Nd,    /* Decimal number */
  ucp_Nl,    /* Letter number */
  ucp_No,    /* Other number */
  ucp_Pc,    /* Connector punctuation */
  ucp_Pd,    /* Dash punctuation */
  ucp_Pe,    /* Close punctuation */
  ucp_Pf,    /* Final punctuation */
  ucp_Pi,    /* Initial punctuation */
  ucp_Po,    /* Other punctuation */
  ucp_Ps,    /* Open punctuation */
  ucp_Sc,    /* Currency symbol */
  ucp_Sk,    /* Modifier symbol */
  ucp_Sm,    /* Mathematical symbol */
  ucp_So,    /* Other symbol */
  ucp_Zl,    /* Line separator */
  ucp_Zp,    /* Paragraph separator */
  ucp_Zs,    /* Space separator */
};

/* These are Boolean properties. */

enum {
  ucp_ASCII,
  ucp_ASCII_Hex_Digit,
  ucp_Alphabetic,
  ucp_Bidi_Control,
  ucp_Bidi_Mirrored,
  ucp_Case_Ignorable,
  ucp_Cased,
  ucp_Changes_When_Casefolded,
  ucp_Changes_When_Casemapped,
  ucp_Changes_When_Lowercased,
  ucp_Changes_When_Titlecased,
  ucp_Changes_When_Uppercased,
  ucp_Dash,
  ucp_Default_Ignorable_Code_Point,
  ucp_Deprecated,
  ucp_Diacritic,
  ucp_Emoji,
  ucp_Emoji_Component,
  ucp_Emoji_Modifier,
  ucp_Emoji_Modifier_Base,
  ucp_Emoji_Presentation,
  ucp_Extended_Pictographic,
  ucp_Extender,
  ucp_Grapheme_Base,
  ucp_Grapheme_Extend,
  ucp_Grapheme_Link,
  ucp_Hex_Digit,
  ucp_IDS_Binary_Operator,
  ucp_IDS_Trinary_Operator,
  ucp_ID_Continue,
  ucp_ID_Start,
  ucp_Ideographic,
  ucp_Join_Control,
  ucp_Logical_Order_Exception,
  ucp_Lowercase,
  ucp_Math,
  ucp_Noncharacter_Code_Point,
  ucp_Pattern_Syntax,
  ucp_Pattern_White_Space,
  ucp_Prepended_Concatenation_Mark,
  ucp_Quotation_Mark,
  ucp_Radical,
  ucp_Regional_Indicator,
  ucp_Sentence_Terminal,
  ucp_Soft_Dotted,
  ucp_Terminal_Punctuation,
  ucp_Unified_Ideograph,
  ucp_Uppercase,
  ucp_Variation_Selector,
  ucp_White_Space,
  ucp_XID_Continue,
  ucp_XID_Start,
  /* This must be last */
  ucp_Bprop_Count
};

/* Size of entries in ucd_boolprop_sets[] */

#define ucd_boolprop_sets_item_size 2

/* These are the bidi class values. */

enum {
  ucp_bidiAL,   /* Arabic_Letter */
  ucp_bidiAN,   /* Arabic_Number */
  ucp_bidiB,    /* Paragraph_Separator */
  ucp_bidiBN,   /* Boundary_Neutral */
  ucp_bidiCS,   /* Common_Separator */
  ucp_bidiEN,   /* European_Number */
  ucp_bidiES,   /* European_Separator */
  ucp_bidiET,   /* European_Terminator */
  ucp_bidiFSI,  /* First_Strong_Isolate */
  ucp_bidiL,    /* Left_To_Right */
  ucp_bidiLRE,  /* Left_To_Right_Embedding */
  ucp_bidiLRI,  /* Left_To_Right_Isolate */
  ucp_bidiLRO,  /* Left_To_Right_Override */
  ucp_bidiNSM,  /* Nonspacing_Mark */
  ucp_bidiON,   /* Other_Neutral */
  ucp_bidiPDF,  /* Pop_Directional_Format */
  ucp_bidiPDI,  /* Pop_Directional_Isolate */
  ucp_bidiR,    /* Right_To_Left */
  ucp_bidiRLE,  /* Right_To_Left_Embedding */
  ucp_bidiRLI,  /* Right_To_Left_Isolate */
  ucp_bidiRLO,  /* Right_To_Left_Override */
  ucp_bidiS,    /* Segment_Separator */
  ucp_bidiWS,   /* White_Space */
};

/* These are grapheme break properties. The Extended Pictographic property
comes from the emoji-data.txt file. */

enum {
  ucp_gbCR,                    /*  0 */
  ucp_gbLF,                    /*  1 */
  ucp_gbControl,               /*  2 */
  ucp_gbExtend,                /*  3 */
  ucp_gbPrepend,               /*  4 */
  ucp_gbSpacingMark,           /*  5 */
  ucp_gbL,                     /*  6 Hangul syllable type L */
  ucp_gbV,                     /*  7 Hangul syllable type V */
  ucp_gbT,                     /*  8 Hangul syllable type T */
  ucp_gbLV,                    /*  9 Hangul syllable type LV */
  ucp_gbLVT,                   /* 10 Hangul syllable type LVT */
  ucp_gbRegional_Indicator,    /* 11 */
  ucp_gbOther,                 /* 12 */
  ucp_gbZWJ,                   /* 13 */
  ucp_gbExtended_Pictographic, /* 14 */
};

/* These are the script identifications. */

enum {
  /* Scripts which has characters in other scripts. */
  ucp_Latin,
  ucp_Greek,
  ucp_Cyrillic,
  ucp_Arabic,
  ucp_Syriac,
  ucp_Thaana,
  ucp_Devanagari,
  ucp_Bengali,
  ucp_Gurmukhi,
  ucp_Gujarati,
  ucp_Oriya,
  ucp_Tamil,
  ucp_Telugu,
  ucp_Kannada,
  ucp_Malayalam,
  ucp_Sinhala,
  ucp_Myanmar,
  ucp_Georgian,
  ucp_Hangul,
  ucp_Mongolian,
  ucp_Hiragana,
  ucp_Katakana,
  ucp_Bopomofo,
  ucp_Han,
  ucp_Yi,
  ucp_Tagalog,
  ucp_Hanunoo,
  ucp_Buhid,
  ucp_Tagbanwa,
  ucp_Limbu,
  ucp_Tai_Le,
  ucp_Linear_B,
  ucp_Cypriot,
  ucp_Buginese,
  ucp_Coptic,
  ucp_Glagolitic,
  ucp_Syloti_Nagri,
  ucp_Phags_Pa,
  ucp_Nko,
  ucp_Kayah_Li,
  ucp_Javanese,
  ucp_Kaithi,
  ucp_Mandaic,
  ucp_Chakma,
  ucp_Sharada,
  ucp_Takri,
  ucp_Duployan,
  ucp_Grantha,
  ucp_Khojki,
  ucp_Linear_A,
  ucp_Mahajani,
  ucp_Manichaean,
  ucp_Modi,
  ucp_Old_Permic,
  ucp_Psalter_Pahlavi,
  ucp_Khudawadi,
  ucp_Tirhuta,
  ucp_Multani,
  ucp_Adlam,
  ucp_Masaram_Gondi,
  ucp_Dogra,
  ucp_Gunjala_Gondi,
  ucp_Hanifi_Rohingya,
  ucp_Sogdian,
  ucp_Nandinagari,
  ucp_Yezidi,
  ucp_Cypro_Minoan,
  ucp_Old_Uyghur,

  /* Scripts which has no characters in other scripts. */
  ucp_Unknown,
  ucp_Common,
  ucp_Armenian,
  ucp_Hebrew,
  ucp_Thai,
  ucp_Lao,
  ucp_Tibetan,
  ucp_Ethiopic,
  ucp_Cherokee,
  ucp_Canadian_Aboriginal,
  ucp_Ogham,
  ucp_Runic,
  ucp_Khmer,
  ucp_Old_Italic,
  ucp_Gothic,
  ucp_Deseret,
  ucp_Inherited,
  ucp_Ugaritic,
  ucp_Shavian,
  ucp_Osmanya,
  ucp_Braille,
  ucp_New_Tai_Lue,
  ucp_Tifinagh,
  ucp_Old_Persian,
  ucp_Kharoshthi,
  ucp_Balinese,
  ucp_Cuneiform,
  ucp_Phoenician,
  ucp_Sundanese,
  ucp_Lepcha,
  ucp_Ol_Chiki,
  ucp_Vai,
  ucp_Saurashtra,
  ucp_Rejang,
  ucp_Lycian,
  ucp_Carian,
  ucp_Lydian,
  ucp_Cham,
  ucp_Tai_Tham,
  ucp_Tai_Viet,
  ucp_Avestan,
  ucp_Egyptian_Hieroglyphs,
  ucp_Samaritan,
  ucp_Lisu,
  ucp_Bamum,
  ucp_Meetei_Mayek,
  ucp_Imperial_Aramaic,
  ucp_Old_South_Arabian,
  ucp_Inscriptional_Parthian,
  ucp_Inscriptional_Pahlavi,
  ucp_Old_Turkic,
  ucp_Batak,
  ucp_Brahmi,
  ucp_Meroitic_Cursive,
  ucp_Meroitic_Hieroglyphs,
  ucp_Miao,
  ucp_Sora_Sompeng,
  ucp_Caucasian_Albanian,
  ucp_Bassa_Vah,
  ucp_Elbasan,
  ucp_Pahawh_Hmong,
  ucp_Mende_Kikakui,
  ucp_Mro,
  ucp_Old_North_Arabian,
  ucp_Nabataean,
  ucp_Palmyrene,
  ucp_Pau_Cin_Hau,
  ucp_Siddham,
  ucp_Warang_Citi,
  ucp_Ahom,
  ucp_Anatolian_Hieroglyphs,
  ucp_Hatran,
  ucp_Old_Hungarian,
  ucp_SignWriting,
  ucp_Bhaiksuki,
  ucp_Marchen,
  ucp_Newa,
  ucp_Osage,
  ucp_Tangut,
  ucp_Nushu,
  ucp_Soyombo,
  ucp_Zanabazar_Square,
  ucp_Makasar,
  ucp_Medefaidrin,
  ucp_Old_Sogdian,
  ucp_Elymaic,
  ucp_Nyiakeng_Puachue_Hmong,
  ucp_Wancho,
  ucp_Chorasmian,
  ucp_Dives_Akuru,
  ucp_Khitan_Small_Script,
  ucp_Tangsa,
  ucp_Toto,
  ucp_Vithkuqi,
  ucp_Kawi,
  ucp_Nag_Mundari,

  /* This must be last */
  ucp_Script_Count
};

/* Size of entries in ucd_script_sets[] */

#define ucd_script_sets_item_size 3

#endif  /* PCRE2_UCP_H_IDEMPOTENT_GUARD */

/* End of pcre2_ucp.h */
