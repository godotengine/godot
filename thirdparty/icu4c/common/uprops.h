// © 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
*******************************************************************************
*
*   Copyright (C) 2002-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  uprops.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2002feb24
*   created by: Markus W. Scherer
*
*   Constants for mostly non-core Unicode character properties
*   stored in uprops.icu.
*/

#ifndef __UPROPS_H__
#define __UPROPS_H__

#include "unicode/utypes.h"
#include "unicode/uset.h"
#include "uset_imp.h"
#include "udataswp.h"

/* indexes[] entries */
enum {
    UPROPS_PROPS32_INDEX,
    UPROPS_EXCEPTIONS_INDEX,
    UPROPS_EXCEPTIONS_TOP_INDEX,

    UPROPS_ADDITIONAL_TRIE_INDEX,
    UPROPS_ADDITIONAL_VECTORS_INDEX,
    UPROPS_ADDITIONAL_VECTORS_COLUMNS_INDEX,

    UPROPS_SCRIPT_EXTENSIONS_INDEX,

    UPROPS_RESERVED_INDEX_7,
    UPROPS_RESERVED_INDEX_8,

    /* size of the data file (number of 32-bit units after the header) */
    UPROPS_DATA_TOP_INDEX,

    /* maximum values for code values in vector word 0 */
    UPROPS_MAX_VALUES_INDEX=10,
    /* maximum values for code values in vector word 2 */
    UPROPS_MAX_VALUES_2_INDEX,

    UPROPS_INDEX_COUNT=16
};

/* definitions for the main properties words */
enum {
    /* general category shift==0                                0 (5 bits) */
    /* reserved                                                 5 (1 bit) */
    UPROPS_NUMERIC_TYPE_VALUE_SHIFT=6                       /*  6 (10 bits) */
};

#define GET_CATEGORY(props) ((props)&0x1f)
#define CAT_MASK(props) U_MASK(GET_CATEGORY(props))

#define GET_NUMERIC_TYPE_VALUE(props) ((props)>>UPROPS_NUMERIC_TYPE_VALUE_SHIFT)

/* constants for the storage form of numeric types and values */
enum {
    /** No numeric value. */
    UPROPS_NTV_NONE=0,
    /** Decimal digits: nv=0..9 */
    UPROPS_NTV_DECIMAL_START=1,
    /** Other digits: nv=0..9 */
    UPROPS_NTV_DIGIT_START=11,
    /** Small integers: nv=0..154 */
    UPROPS_NTV_NUMERIC_START=21,
    /** Fractions: ((ntv>>4)-12) / ((ntv&0xf)+1) = -1..17 / 1..16 */
    UPROPS_NTV_FRACTION_START=0xb0,
    /**
     * Large integers:
     * ((ntv>>5)-14) * 10^((ntv&0x1f)+2) = (1..9)*(10^2..10^33)
     * (only one significant decimal digit)
     */
    UPROPS_NTV_LARGE_START=0x1e0,
    /**
     * Sexagesimal numbers:
     * ((ntv>>2)-0xbf) * 60^((ntv&3)+1) = (1..9)*(60^1..60^4)
     */
    UPROPS_NTV_BASE60_START=0x300,
    /**
     * Fraction-20 values:
     * frac20 = ntv-0x324 = 0..0x17 -> 1|3|5|7 / 20|40|80|160|320|640
     * numerator: num = 2*(frac20&3)+1
     * denominator: den = 20<<(frac20>>2)
     */
    UPROPS_NTV_FRACTION20_START=UPROPS_NTV_BASE60_START+36,  // 0x300+9*4=0x324
    /**
     * Fraction-32 values:
     * frac32 = ntv-0x34c = 0..15 -> 1|3|5|7 / 32|64|128|256
     * numerator: num = 2*(frac32&3)+1
     * denominator: den = 32<<(frac32>>2)
     */
    UPROPS_NTV_FRACTION32_START=UPROPS_NTV_FRACTION20_START+24,  // 0x324+6*4=0x34c
    /** No numeric value (yet). */
    UPROPS_NTV_RESERVED_START=UPROPS_NTV_FRACTION32_START+16,  // 0x34c+4*4=0x35c

    UPROPS_NTV_MAX_SMALL_INT=UPROPS_NTV_FRACTION_START-UPROPS_NTV_NUMERIC_START-1
};

#define UPROPS_NTV_GET_TYPE(ntv) \
    ((ntv==UPROPS_NTV_NONE) ? U_NT_NONE : \
    (ntv<UPROPS_NTV_DIGIT_START) ?  U_NT_DECIMAL : \
    (ntv<UPROPS_NTV_NUMERIC_START) ? U_NT_DIGIT : \
    U_NT_NUMERIC)

/* number of properties vector words */
#define UPROPS_VECTOR_WORDS     3

/*
 * Properties in vector word 0
 * Bits
 * 31..24   DerivedAge version major/minor one nibble each
 * 23..22   3..1: Bits 21..20 & 7..0 = Script_Extensions index
 *             3: Script value from Script_Extensions
 *             2: Script=Inherited
 *             1: Script=Common
 *             0: Script=bits 21..20 & 7..0
 * 21..20   Bits 9..8 of the UScriptCode, or index to Script_Extensions
 * 19..17   East Asian Width
 * 16.. 8   UBlockCode
 *  7.. 0   UScriptCode, or index to Script_Extensions
 */

/* derived age: one nibble each for major and minor version numbers */
#define UPROPS_AGE_MASK         0xff000000
#define UPROPS_AGE_SHIFT        24

/* Script_Extensions: mask includes Script */
#define UPROPS_SCRIPT_X_MASK    0x00f000ff
#define UPROPS_SCRIPT_X_SHIFT   22

// The UScriptCode or Script_Extensions index is split across two bit fields.
// (Starting with Unicode 13/ICU 66/2019 due to more varied Script_Extensions.)
// Shift the high bits right by 12 to assemble the full value.
#define UPROPS_SCRIPT_HIGH_MASK    0x00300000
#define UPROPS_SCRIPT_HIGH_SHIFT   12
#define UPROPS_MAX_SCRIPT          0x3ff

#define UPROPS_EA_MASK          0x000e0000
#define UPROPS_EA_SHIFT         17

#define UPROPS_BLOCK_MASK       0x0001ff00
#define UPROPS_BLOCK_SHIFT      8

#define UPROPS_SCRIPT_LOW_MASK  0x000000ff

/* UPROPS_SCRIPT_X_WITH_COMMON must be the lowest value that involves Script_Extensions. */
#define UPROPS_SCRIPT_X_WITH_COMMON     0x400000
#define UPROPS_SCRIPT_X_WITH_INHERITED  0x800000
#define UPROPS_SCRIPT_X_WITH_OTHER      0xc00000

#ifdef __cplusplus

namespace {

inline uint32_t uprops_mergeScriptCodeOrIndex(uint32_t scriptX) {
    return
        ((scriptX & UPROPS_SCRIPT_HIGH_MASK) >> UPROPS_SCRIPT_HIGH_SHIFT) |
        (scriptX & UPROPS_SCRIPT_LOW_MASK);
}

}  // namespace

#endif  // __cplusplus

/*
 * Properties in vector word 1
 * Each bit encodes one binary property.
 * The following constants represent the bit number, use 1<<UPROPS_XYZ.
 * UPROPS_BINARY_1_TOP<=32!
 *
 * Keep this list of property enums in sync with
 * propListNames[] in icu/source/tools/genprops/props2.c!
 *
 * ICU 2.6/uprops format version 3.2 stores full properties instead of "Other_".
 */
enum {
    UPROPS_WHITE_SPACE,
    UPROPS_DASH,
    UPROPS_HYPHEN,
    UPROPS_QUOTATION_MARK,
    UPROPS_TERMINAL_PUNCTUATION,
    UPROPS_MATH,
    UPROPS_HEX_DIGIT,
    UPROPS_ASCII_HEX_DIGIT,
    UPROPS_ALPHABETIC,
    UPROPS_IDEOGRAPHIC,
    UPROPS_DIACRITIC,
    UPROPS_EXTENDER,
    UPROPS_NONCHARACTER_CODE_POINT,
    UPROPS_GRAPHEME_EXTEND,
    UPROPS_GRAPHEME_LINK,
    UPROPS_IDS_BINARY_OPERATOR,
    UPROPS_IDS_TRINARY_OPERATOR,
    UPROPS_RADICAL,
    UPROPS_UNIFIED_IDEOGRAPH,
    UPROPS_DEFAULT_IGNORABLE_CODE_POINT,
    UPROPS_DEPRECATED,
    UPROPS_LOGICAL_ORDER_EXCEPTION,
    UPROPS_XID_START,
    UPROPS_XID_CONTINUE,
    UPROPS_ID_START,                            /* ICU 2.6, uprops format version 3.2 */
    UPROPS_ID_CONTINUE,
    UPROPS_GRAPHEME_BASE,
    UPROPS_S_TERM,                              /* new in ICU 3.0 and Unicode 4.0.1 */
    UPROPS_VARIATION_SELECTOR,
    UPROPS_PATTERN_SYNTAX,                      /* new in ICU 3.4 and Unicode 4.1 */
    UPROPS_PATTERN_WHITE_SPACE,
    UPROPS_PREPENDED_CONCATENATION_MARK,        // new in ICU 60 and Unicode 10
    UPROPS_BINARY_1_TOP                         /* ==32 - full! */
};

/*
 * Properties in vector word 2
 * Bits
 * 31..26   unused since ICU 70 added uemoji.icu;
 *          in ICU 57..69 stored emoji properties
 * 25..20   Line Break
 * 19..15   Sentence Break
 * 14..10   Word Break
 *  9.. 5   Grapheme Cluster Break
 *  4.. 0   Decomposition Type
 */
enum {
    UPROPS_2_UNUSED_WAS_EXTENDED_PICTOGRAPHIC=26,  // ICU 62..69
    UPROPS_2_UNUSED_WAS_EMOJI_COMPONENT,  // ICU 60..69
    UPROPS_2_UNUSED_WAS_EMOJI,  // ICU 57..69
    UPROPS_2_UNUSED_WAS_EMOJI_PRESENTATION,  // ICU 57..69
    UPROPS_2_UNUSED_WAS_EMOJI_MODIFIER,  // ICU 57..69
    UPROPS_2_UNUSED_WAS_EMOJI_MODIFIER_BASE  // ICU 57..69
};

#define UPROPS_LB_MASK          0x03f00000
#define UPROPS_LB_SHIFT         20

#define UPROPS_SB_MASK          0x000f8000
#define UPROPS_SB_SHIFT         15

#define UPROPS_WB_MASK          0x00007c00
#define UPROPS_WB_SHIFT         10

#define UPROPS_GCB_MASK         0x000003e0
#define UPROPS_GCB_SHIFT        5

#define UPROPS_DT_MASK          0x0000001f

/**
 * Gets the main properties value for a code point.
 * Implemented in uchar.c for uprops.cpp.
 */
U_CFUNC uint32_t
u_getMainProperties(UChar32 c);

/**
 * Get a properties vector word for a code point.
 * Implemented in uchar.c for uprops.cpp.
 * @return 0 if no data or illegal argument
 */
U_CFUNC uint32_t
u_getUnicodeProperties(UChar32 c, int32_t column);

/**
 * Get the the maximum values for some enum/int properties.
 * Use the same column numbers as for u_getUnicodeProperties().
 * The returned value will contain maximum values stored in the same bit fields
 * as where the enum values are stored in the u_getUnicodeProperties()
 * return values for the same columns.
 *
 * Valid columns are those for properties words that contain enumerated values.
 * (ICU 2.6: columns 0 and 2)
 * For other column numbers, this function will return 0.
 *
 * @internal
 */
U_CFUNC int32_t
uprv_getMaxValues(int32_t column);

/**
 * Checks if c is alphabetic, or a decimal digit; implements UCHAR_POSIX_ALNUM.
 * @internal
 */
U_CFUNC UBool
u_isalnumPOSIX(UChar32 c);

/**
 * Checks if c is in
 * [^\p{space}\p{gc=Control}\p{gc=Surrogate}\p{gc=Unassigned}]
 * with space=\p{Whitespace} and Control=Cc.
 * Implements UCHAR_POSIX_GRAPH.
 * @internal
 */
U_CFUNC UBool
u_isgraphPOSIX(UChar32 c);

/**
 * Checks if c is in \p{graph}\p{blank} - \p{cntrl}.
 * Implements UCHAR_POSIX_PRINT.
 * @internal
 */
U_CFUNC UBool
u_isprintPOSIX(UChar32 c);

/** Some code points. @internal */
enum {
    TAB     =0x0009,
    LF      =0x000a,
    FF      =0x000c,
    CR      =0x000d,
    NBSP    =0x00a0,
    CGJ     =0x034f,
    FIGURESP=0x2007,
    HAIRSP  =0x200a,
    ZWNJ    =0x200c,
    ZWJ     =0x200d,
    RLM     =0x200f,
    NNBSP   =0x202f,
    ZWNBSP  =0xfeff
};

/**
 * Get the maximum length of a (regular/1.0/extended) character name.
 * @return 0 if no character names available.
 */
U_CAPI int32_t U_EXPORT2
uprv_getMaxCharNameLength(void);

/**
 * Fills set with characters that are used in Unicode character names.
 * Includes all characters that are used in regular/Unicode 1.0/extended names.
 * Just empties the set if no character names are available.
 * @param sa USetAdder to receive characters.
 */
U_CAPI void U_EXPORT2
uprv_getCharNameCharacters(const USetAdder *sa);

/**
 * Constants for which data and implementation files provide which properties.
 * Used by UnicodeSet for service-specific property enumeration.
 * @internal
 */
enum UPropertySource {
    /** No source, not a supported property. */
    UPROPS_SRC_NONE,
    /** From uchar.c/uprops.icu main trie */
    UPROPS_SRC_CHAR,
    /** From uchar.c/uprops.icu properties vectors trie */
    UPROPS_SRC_PROPSVEC,
    /** From unames.c/unames.icu */
    UPROPS_SRC_NAMES,
    /** From ucase.c/ucase.icu */
    UPROPS_SRC_CASE,
    /** From ubidi_props.c/ubidi.icu */
    UPROPS_SRC_BIDI,
    /** From uchar.c/uprops.icu main trie as well as properties vectors trie */
    UPROPS_SRC_CHAR_AND_PROPSVEC,
    /** From ucase.c/ucase.icu as well as unorm.cpp/unorm.icu */
    UPROPS_SRC_CASE_AND_NORM,
    /** From normalizer2impl.cpp/nfc.nrm */
    UPROPS_SRC_NFC,
    /** From normalizer2impl.cpp/nfkc.nrm */
    UPROPS_SRC_NFKC,
    /** From normalizer2impl.cpp/nfkc_cf.nrm */
    UPROPS_SRC_NFKC_CF,
    /** From normalizer2impl.cpp/nfc.nrm canonical iterator data */
    UPROPS_SRC_NFC_CANON_ITER,
    // Text layout properties.
    UPROPS_SRC_INPC,
    UPROPS_SRC_INSC,
    UPROPS_SRC_VO,
    UPROPS_SRC_EMOJI,
    UPROPS_SRC_IDSU,
    UPROPS_SRC_ID_COMPAT_MATH,
    /** One more than the highest UPropertySource (UPROPS_SRC_) constant. */
    UPROPS_SRC_COUNT
};
typedef enum UPropertySource UPropertySource;

/**
 * @see UPropertySource
 * @internal
 */
U_CFUNC UPropertySource U_EXPORT2
uprops_getSource(UProperty which);

/**
 * Enumerate uprops.icu's main data trie and add the
 * start of each range of same properties to the set.
 * @internal
 */
U_CFUNC void U_EXPORT2
uchar_addPropertyStarts(const USetAdder *sa, UErrorCode *pErrorCode);

/**
 * Enumerate uprops.icu's properties vectors trie and add the
 * start of each range of same properties to the set.
 * @internal
 */
U_CFUNC void U_EXPORT2
upropsvec_addPropertyStarts(const USetAdder *sa, UErrorCode *pErrorCode);

U_CFUNC void U_EXPORT2
uprops_addPropertyStarts(UPropertySource src, const USetAdder *sa, UErrorCode *pErrorCode);

/**
 * Return a set of characters for property enumeration.
 * For each two consecutive characters (start, limit) in the set,
 * all of the properties for start..limit-1 are all the same.
 *
 * @param sa USetAdder to receive result. Existing contents are lost.
 * @internal
 */
/*U_CFUNC void U_EXPORT2
uprv_getInclusions(const USetAdder *sa, UErrorCode *pErrorCode);
*/

/**
 * Swap the ICU Unicode character names file. See uchar.c.
 * @internal
 */
U_CAPI int32_t U_EXPORT2
uchar_swapNames(const UDataSwapper *ds,
                const void *inData, int32_t length, void *outData,
                UErrorCode *pErrorCode);

#ifdef __cplusplus

U_NAMESPACE_BEGIN

class UnicodeSet;

class CharacterProperties {
public:
    CharacterProperties() = delete;
    static const UnicodeSet *getInclusionsForProperty(UProperty prop, UErrorCode &errorCode);
    static const UnicodeSet *getBinaryPropertySet(UProperty property, UErrorCode &errorCode);
};

// implemented in uniset_props.cpp
U_CFUNC UnicodeSet *
uniset_getUnicode32Instance(UErrorCode &errorCode);

U_NAMESPACE_END

#endif

#endif
