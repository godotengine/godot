/*
*******************************************************************************
* Copyright (c) 1996-2010, International Business Machines Corporation
*               and others. All Rights Reserved.
*******************************************************************************
* File unorm.h
*
* Created by: Vladimir Weinstein 12052000
*
* Modification history :
*
* Date        Name        Description
* 02/01/01    synwee      Added normalization quickcheck enum and method.
*/
#ifndef UNORM_H
#define UNORM_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_NORMALIZATION

#include "unicode/uiter.h"
#include "unicode/unorm2.h"

/**
 * \file
 * \brief C API: Unicode Normalization 
 *
 * <h2>Unicode normalization API</h2>
 *
 * Note: This API has been replaced by the unorm2.h API and is only available
 * for backward compatibility. The functions here simply delegate to the
 * unorm2.h functions, for example unorm2_getInstance() and unorm2_normalize().
 * There is one exception: The new API does not provide a replacement for unorm_compare().
 *
 * <code>unorm_normalize</code> transforms Unicode text into an equivalent composed or
 * decomposed form, allowing for easier sorting and searching of text.
 * <code>unorm_normalize</code> supports the standard normalization forms described in
 * <a href="http://www.unicode.org/unicode/reports/tr15/" target="unicode">
 * Unicode Standard Annex #15: Unicode Normalization Forms</a>.
 *
 * Characters with accents or other adornments can be encoded in
 * several different ways in Unicode.  For example, take the character A-acute.
 * In Unicode, this can be encoded as a single character (the
 * "composed" form):
 *
 * \code
 *      00C1    LATIN CAPITAL LETTER A WITH ACUTE
 * \endcode
 *
 * or as two separate characters (the "decomposed" form):
 *
 * \code
 *      0041    LATIN CAPITAL LETTER A
 *      0301    COMBINING ACUTE ACCENT
 * \endcode
 *
 * To a user of your program, however, both of these sequences should be
 * treated as the same "user-level" character "A with acute accent".  When you are searching or
 * comparing text, you must ensure that these two sequences are treated 
 * equivalently.  In addition, you must handle characters with more than one
 * accent.  Sometimes the order of a character's combining accents is
 * significant, while in other cases accent sequences in different orders are
 * really equivalent.
 *
 * Similarly, the string "ffi" can be encoded as three separate letters:
 *
 * \code
 *      0066    LATIN SMALL LETTER F
 *      0066    LATIN SMALL LETTER F
 *      0069    LATIN SMALL LETTER I
 * \endcode
 *
 * or as the single character
 *
 * \code
 *      FB03    LATIN SMALL LIGATURE FFI
 * \endcode
 *
 * The ffi ligature is not a distinct semantic character, and strictly speaking
 * it shouldn't be in Unicode at all, but it was included for compatibility
 * with existing character sets that already provided it.  The Unicode standard
 * identifies such characters by giving them "compatibility" decompositions
 * into the corresponding semantic characters.  When sorting and searching, you
 * will often want to use these mappings.
 *
 * <code>unorm_normalize</code> helps solve these problems by transforming text into the
 * canonical composed and decomposed forms as shown in the first example above.  
 * In addition, you can have it perform compatibility decompositions so that 
 * you can treat compatibility characters the same as their equivalents.
 * Finally, <code>unorm_normalize</code> rearranges accents into the proper canonical
 * order, so that you do not have to worry about accent rearrangement on your
 * own.
 *
 * Form FCD, "Fast C or D", is also designed for collation.
 * It allows to work on strings that are not necessarily normalized
 * with an algorithm (like in collation) that works under "canonical closure", i.e., it treats precomposed
 * characters and their decomposed equivalents the same.
 *
 * It is not a normalization form because it does not provide for uniqueness of representation. Multiple strings
 * may be canonically equivalent (their NFDs are identical) and may all conform to FCD without being identical
 * themselves.
 *
 * The form is defined such that the "raw decomposition", the recursive canonical decomposition of each character,
 * results in a string that is canonically ordered. This means that precomposed characters are allowed for as long
 * as their decompositions do not need canonical reordering.
 *
 * Its advantage for a process like collation is that all NFD and most NFC texts - and many unnormalized texts -
 * already conform to FCD and do not need to be normalized (NFD) for such a process. The FCD quick check will
 * return UNORM_YES for most strings in practice.
 *
 * unorm_normalize(UNORM_FCD) may be implemented with UNORM_NFD.
 *
 * For more details on FCD see the collation design document:
 * http://source.icu-project.org/repos/icu/icuhtml/trunk/design/collation/ICU_collation_design.htm
 *
 * ICU collation performs either NFD or FCD normalization automatically if normalization
 * is turned on for the collator object.
 * Beyond collation and string search, normalized strings may be useful for string equivalence comparisons,
 * transliteration/transcription, unique representations, etc.
 *
 * The W3C generally recommends to exchange texts in NFC.
 * Note also that most legacy character encodings use only precomposed forms and often do not
 * encode any combining marks by themselves. For conversion to such character encodings the
 * Unicode text needs to be normalized to NFC.
 * For more usage examples, see the Unicode Standard Annex.
 */

/**
 * Constants for normalization modes.
 * @stable ICU 2.0
 */
typedef enum {
  /** No decomposition/composition. @stable ICU 2.0 */
  UNORM_NONE = 1, 
  /** Canonical decomposition. @stable ICU 2.0 */
  UNORM_NFD = 2,
  /** Compatibility decomposition. @stable ICU 2.0 */
  UNORM_NFKD = 3,
  /** Canonical decomposition followed by canonical composition. @stable ICU 2.0 */
  UNORM_NFC = 4,
  /** Default normalization. @stable ICU 2.0 */
  UNORM_DEFAULT = UNORM_NFC, 
  /** Compatibility decomposition followed by canonical composition. @stable ICU 2.0 */
  UNORM_NFKC =5,
  /** "Fast C or D" form. @stable ICU 2.0 */
  UNORM_FCD = 6,

  /** One more than the highest normalization mode constant. @stable ICU 2.0 */
  UNORM_MODE_COUNT
} UNormalizationMode;

/**
 * Constants for options flags for normalization.
 * Use 0 for default options,
 * including normalization according to the Unicode version
 * that is currently supported by ICU (see u_getUnicodeVersion).
 * @stable ICU 2.6
 */
enum {
    /**
     * Options bit set value to select Unicode 3.2 normalization
     * (except NormalizationCorrections).
     * At most one Unicode version can be selected at a time.
     * @stable ICU 2.6
     */
    UNORM_UNICODE_3_2=0x20
};

/**
 * Lowest-order bit number of unorm_compare() options bits corresponding to
 * normalization options bits.
 *
 * The options parameter for unorm_compare() uses most bits for
 * itself and for various comparison and folding flags.
 * The most significant bits, however, are shifted down and passed on
 * to the normalization implementation.
 * (That is, from unorm_compare(..., options, ...),
 * options>>UNORM_COMPARE_NORM_OPTIONS_SHIFT will be passed on to the
 * internal normalization functions.)
 *
 * @see unorm_compare
 * @stable ICU 2.6
 */
#define UNORM_COMPARE_NORM_OPTIONS_SHIFT 20

/**
 * Normalize a string.
 * The string will be normalized according the specified normalization mode
 * and options.
 * The source and result buffers must not be the same, nor overlap.
 *
 * @param source The string to normalize.
 * @param sourceLength The length of source, or -1 if NUL-terminated.
 * @param mode The normalization mode; one of UNORM_NONE, 
 *             UNORM_NFD, UNORM_NFC, UNORM_NFKC, UNORM_NFKD, UNORM_DEFAULT.
 * @param options The normalization options, ORed together (0 for no options).
 * @param result A pointer to a buffer to receive the result string.
 *               The result string is NUL-terminated if possible.
 * @param resultLength The maximum size of result.
 * @param status A pointer to a UErrorCode to receive any errors.
 * @return The total buffer size needed; if greater than resultLength,
 *         the output was truncated, and the error code is set to U_BUFFER_OVERFLOW_ERROR.
 * @stable ICU 2.0
 */
U_STABLE int32_t U_EXPORT2 
unorm_normalize(const UChar *source, int32_t sourceLength,
                UNormalizationMode mode, int32_t options,
                UChar *result, int32_t resultLength,
                UErrorCode *status);

/**
 * Performing quick check on a string, to quickly determine if the string is 
 * in a particular normalization format.
 * Three types of result can be returned UNORM_YES, UNORM_NO or
 * UNORM_MAYBE. Result UNORM_YES indicates that the argument
 * string is in the desired normalized format, UNORM_NO determines that
 * argument string is not in the desired normalized format. A 
 * UNORM_MAYBE result indicates that a more thorough check is required, 
 * the user may have to put the string in its normalized form and compare the 
 * results.
 *
 * @param source       string for determining if it is in a normalized format
 * @param sourcelength length of source to test, or -1 if NUL-terminated
 * @param mode         which normalization form to test for
 * @param status       a pointer to a UErrorCode to receive any errors
 * @return UNORM_YES, UNORM_NO or UNORM_MAYBE
 *
 * @see unorm_isNormalized
 * @stable ICU 2.0
 */
U_STABLE UNormalizationCheckResult U_EXPORT2
unorm_quickCheck(const UChar *source, int32_t sourcelength,
                 UNormalizationMode mode,
                 UErrorCode *status);

/**
 * Performing quick check on a string; same as unorm_quickCheck but
 * takes an extra options parameter like most normalization functions.
 *
 * @param src        String that is to be tested if it is in a normalization format.
 * @param srcLength  Length of source to test, or -1 if NUL-terminated.
 * @param mode       Which normalization form to test for.
 * @param options    The normalization options, ORed together (0 for no options).
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return UNORM_YES, UNORM_NO or UNORM_MAYBE
 *
 * @see unorm_quickCheck
 * @see unorm_isNormalized
 * @stable ICU 2.6
 */
U_STABLE UNormalizationCheckResult U_EXPORT2
unorm_quickCheckWithOptions(const UChar *src, int32_t srcLength, 
                            UNormalizationMode mode, int32_t options,
                            UErrorCode *pErrorCode);

/**
 * Test if a string is in a given normalization form.
 * This is semantically equivalent to source.equals(normalize(source, mode)) .
 *
 * Unlike unorm_quickCheck(), this function returns a definitive result,
 * never a "maybe".
 * For NFD, NFKD, and FCD, both functions work exactly the same.
 * For NFC and NFKC where quickCheck may return "maybe", this function will
 * perform further tests to arrive at a TRUE/FALSE result.
 *
 * @param src        String that is to be tested if it is in a normalization format.
 * @param srcLength  Length of source to test, or -1 if NUL-terminated.
 * @param mode       Which normalization form to test for.
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return Boolean value indicating whether the source string is in the
 *         "mode" normalization form.
 *
 * @see unorm_quickCheck
 * @stable ICU 2.2
 */
U_STABLE UBool U_EXPORT2
unorm_isNormalized(const UChar *src, int32_t srcLength,
                   UNormalizationMode mode,
                   UErrorCode *pErrorCode);

/**
 * Test if a string is in a given normalization form; same as unorm_isNormalized but
 * takes an extra options parameter like most normalization functions.
 *
 * @param src        String that is to be tested if it is in a normalization format.
 * @param srcLength  Length of source to test, or -1 if NUL-terminated.
 * @param mode       Which normalization form to test for.
 * @param options    The normalization options, ORed together (0 for no options).
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return Boolean value indicating whether the source string is in the
 *         "mode/options" normalization form.
 *
 * @see unorm_quickCheck
 * @see unorm_isNormalized
 * @stable ICU 2.6
 */
U_STABLE UBool U_EXPORT2
unorm_isNormalizedWithOptions(const UChar *src, int32_t srcLength,
                              UNormalizationMode mode, int32_t options,
                              UErrorCode *pErrorCode);

/**
 * Iterative normalization forward.
 * This function (together with unorm_previous) is somewhat
 * similar to the C++ Normalizer class (see its non-static functions).
 *
 * Iterative normalization is useful when only a small portion of a longer
 * string/text needs to be processed.
 *
 * For example, the likelihood may be high that processing the first 10% of some
 * text will be sufficient to find certain data.
 * Another example: When one wants to concatenate two normalized strings and get a
 * normalized result, it is much more efficient to normalize just a small part of
 * the result around the concatenation place instead of re-normalizing everything.
 *
 * The input text is an instance of the C character iteration API UCharIterator.
 * It may wrap around a simple string, a CharacterIterator, a Replaceable, or any
 * other kind of text object.
 *
 * If a buffer overflow occurs, then the caller needs to reset the iterator to the
 * old index and call the function again with a larger buffer - if the caller cares
 * for the actual output.
 * Regardless of the output buffer, the iterator will always be moved to the next
 * normalization boundary.
 *
 * This function (like unorm_previous) serves two purposes:
 *
 * 1) To find the next boundary so that the normalization of the part of the text
 * from the current position to that boundary does not affect and is not affected
 * by the part of the text beyond that boundary.
 *
 * 2) To normalize the text up to the boundary.
 *
 * The second step is optional, per the doNormalize parameter.
 * It is omitted for operations like string concatenation, where the two adjacent
 * string ends need to be normalized together.
 * In such a case, the output buffer will just contain a copy of the text up to the
 * boundary.
 *
 * pNeededToNormalize is an output-only parameter. Its output value is only defined
 * if normalization was requested (doNormalize) and successful (especially, no
 * buffer overflow).
 * It is useful for operations like a normalizing transliterator, where one would
 * not want to replace a piece of text if it is not modified.
 *
 * If doNormalize==TRUE and pNeededToNormalize!=NULL then *pNeeded... is set TRUE
 * if the normalization was necessary.
 *
 * If doNormalize==FALSE then *pNeededToNormalize will be set to FALSE.
 *
 * If the buffer overflows, then *pNeededToNormalize will be undefined;
 * essentially, whenever U_FAILURE is true (like in buffer overflows), this result
 * will be undefined.
 *
 * @param src The input text in the form of a C character iterator.
 * @param dest The output buffer; can be NULL if destCapacity==0 for pure preflighting.
 * @param destCapacity The number of UChars that fit into dest.
 * @param mode The normalization mode.
 * @param options The normalization options, ORed together (0 for no options).
 * @param doNormalize Indicates if the source text up to the next boundary
 *                    is to be normalized (TRUE) or just copied (FALSE).
 * @param pNeededToNormalize Output flag indicating if the normalization resulted in
 *                           different text from the input.
 *                           Not defined if an error occurs including buffer overflow.
 *                           Always FALSE if !doNormalize.
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return Length of output (number of UChars) when successful or buffer overflow.
 *
 * @see unorm_previous
 * @see unorm_normalize
 *
 * @stable ICU 2.1
 */
U_STABLE int32_t U_EXPORT2
unorm_next(UCharIterator *src,
           UChar *dest, int32_t destCapacity,
           UNormalizationMode mode, int32_t options,
           UBool doNormalize, UBool *pNeededToNormalize,
           UErrorCode *pErrorCode);

/**
 * Iterative normalization backward.
 * This function (together with unorm_next) is somewhat
 * similar to the C++ Normalizer class (see its non-static functions).
 * For all details see unorm_next.
 *
 * @param src The input text in the form of a C character iterator.
 * @param dest The output buffer; can be NULL if destCapacity==0 for pure preflighting.
 * @param destCapacity The number of UChars that fit into dest.
 * @param mode The normalization mode.
 * @param options The normalization options, ORed together (0 for no options).
 * @param doNormalize Indicates if the source text up to the next boundary
 *                    is to be normalized (TRUE) or just copied (FALSE).
 * @param pNeededToNormalize Output flag indicating if the normalization resulted in
 *                           different text from the input.
 *                           Not defined if an error occurs including buffer overflow.
 *                           Always FALSE if !doNormalize.
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return Length of output (number of UChars) when successful or buffer overflow.
 *
 * @see unorm_next
 * @see unorm_normalize
 *
 * @stable ICU 2.1
 */
U_STABLE int32_t U_EXPORT2
unorm_previous(UCharIterator *src,
               UChar *dest, int32_t destCapacity,
               UNormalizationMode mode, int32_t options,
               UBool doNormalize, UBool *pNeededToNormalize,
               UErrorCode *pErrorCode);

/**
 * Concatenate normalized strings, making sure that the result is normalized as well.
 *
 * If both the left and the right strings are in
 * the normalization form according to "mode/options",
 * then the result will be
 *
 * \code
 *     dest=normalize(left+right, mode, options)
 * \endcode
 *
 * With the input strings already being normalized,
 * this function will use unorm_next() and unorm_previous()
 * to find the adjacent end pieces of the input strings.
 * Only the concatenation of these end pieces will be normalized and
 * then concatenated with the remaining parts of the input strings.
 *
 * It is allowed to have dest==left to avoid copying the entire left string.
 *
 * @param left Left source string, may be same as dest.
 * @param leftLength Length of left source string, or -1 if NUL-terminated.
 * @param right Right source string. Must not be the same as dest, nor overlap.
 * @param rightLength Length of right source string, or -1 if NUL-terminated.
 * @param dest The output buffer; can be NULL if destCapacity==0 for pure preflighting.
 * @param destCapacity The number of UChars that fit into dest.
 * @param mode The normalization mode.
 * @param options The normalization options, ORed together (0 for no options).
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return Length of output (number of UChars) when successful or buffer overflow.
 *
 * @see unorm_normalize
 * @see unorm_next
 * @see unorm_previous
 *
 * @stable ICU 2.1
 */
U_STABLE int32_t U_EXPORT2
unorm_concatenate(const UChar *left, int32_t leftLength,
                  const UChar *right, int32_t rightLength,
                  UChar *dest, int32_t destCapacity,
                  UNormalizationMode mode, int32_t options,
                  UErrorCode *pErrorCode);

/**
 * Option bit for unorm_compare:
 * Both input strings are assumed to fulfill FCD conditions.
 * @stable ICU 2.2
 */
#define UNORM_INPUT_IS_FCD          0x20000

/**
 * Option bit for unorm_compare:
 * Perform case-insensitive comparison.
 * @stable ICU 2.2
 */
#define U_COMPARE_IGNORE_CASE       0x10000

#ifndef U_COMPARE_CODE_POINT_ORDER
/* see also unistr.h and ustring.h */
/**
 * Option bit for u_strCaseCompare, u_strcasecmp, unorm_compare, etc:
 * Compare strings in code point order instead of code unit order.
 * @stable ICU 2.2
 */
#define U_COMPARE_CODE_POINT_ORDER  0x8000
#endif

/**
 * Compare two strings for canonical equivalence.
 * Further options include case-insensitive comparison and
 * code point order (as opposed to code unit order).
 *
 * Canonical equivalence between two strings is defined as their normalized
 * forms (NFD or NFC) being identical.
 * This function compares strings incrementally instead of normalizing
 * (and optionally case-folding) both strings entirely,
 * improving performance significantly.
 *
 * Bulk normalization is only necessary if the strings do not fulfill the FCD
 * conditions. Only in this case, and only if the strings are relatively long,
 * is memory allocated temporarily.
 * For FCD strings and short non-FCD strings there is no memory allocation.
 *
 * Semantically, this is equivalent to
 *   strcmp[CodePointOrder](NFD(foldCase(NFD(s1))), NFD(foldCase(NFD(s2))))
 * where code point order and foldCase are all optional.
 *
 * UAX 21 2.5 Caseless Matching specifies that for a canonical caseless match
 * the case folding must be performed first, then the normalization.
 *
 * @param s1 First source string.
 * @param length1 Length of first source string, or -1 if NUL-terminated.
 *
 * @param s2 Second source string.
 * @param length2 Length of second source string, or -1 if NUL-terminated.
 *
 * @param options A bit set of options:
 *   - U_FOLD_CASE_DEFAULT or 0 is used for default options:
 *     Case-sensitive comparison in code unit order, and the input strings
 *     are quick-checked for FCD.
 *
 *   - UNORM_INPUT_IS_FCD
 *     Set if the caller knows that both s1 and s2 fulfill the FCD conditions.
 *     If not set, the function will quickCheck for FCD
 *     and normalize if necessary.
 *
 *   - U_COMPARE_CODE_POINT_ORDER
 *     Set to choose code point order instead of code unit order
 *     (see u_strCompare for details).
 *
 *   - U_COMPARE_IGNORE_CASE
 *     Set to compare strings case-insensitively using case folding,
 *     instead of case-sensitively.
 *     If set, then the following case folding options are used.
 *
 *   - Options as used with case-insensitive comparisons, currently:
 *
 *   - U_FOLD_CASE_EXCLUDE_SPECIAL_I
 *    (see u_strCaseCompare for details)
 *
 *   - regular normalization options shifted left by UNORM_COMPARE_NORM_OPTIONS_SHIFT
 *
 * @param pErrorCode ICU error code in/out parameter.
 *                   Must fulfill U_SUCCESS before the function call.
 * @return <0 or 0 or >0 as usual for string comparisons
 *
 * @see unorm_normalize
 * @see UNORM_FCD
 * @see u_strCompare
 * @see u_strCaseCompare
 *
 * @stable ICU 2.2
 */
U_STABLE int32_t U_EXPORT2
unorm_compare(const UChar *s1, int32_t length1,
              const UChar *s2, int32_t length2,
              uint32_t options,
              UErrorCode *pErrorCode);

#endif /* #if !UCONFIG_NO_NORMALIZATION */

#endif
