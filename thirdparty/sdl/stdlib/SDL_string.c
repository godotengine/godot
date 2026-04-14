/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// This file contains portable string manipulation functions for SDL

#include "SDL_vacopy.h"

#ifdef SDL_PLATFORM_VITA
#include <psp2/kernel/clib.h>
#endif

#include "SDL_sysstdlib.h"

#include "SDL_casefolding.h"

#if defined(__SIZEOF_WCHAR_T__)
#define SDL_SIZEOF_WCHAR_T __SIZEOF_WCHAR_T__
#elif defined(SDL_PLATFORM_WINDOWS)
#define SDL_SIZEOF_WCHAR_T 2
#else  // assume everything else is UTF-32 (add more tests if compiler-assert fails below!)
#define SDL_SIZEOF_WCHAR_T 4
#endif
SDL_COMPILE_TIME_ASSERT(sizeof_wchar_t, sizeof(wchar_t) == SDL_SIZEOF_WCHAR_T);


char *SDL_UCS4ToUTF8(Uint32 codepoint, char *dst)
{
    if (!dst) {
        return NULL;  // I guess...?
    } else if (codepoint > 0x10FFFF) {  // Outside the range of Unicode codepoints (also, larger than can be encoded in 4 bytes of UTF-8!).
        codepoint = SDL_INVALID_UNICODE_CODEPOINT;
    } else if ((codepoint >= 0xD800) && (codepoint <= 0xDFFF)) {  // UTF-16 surrogate values are illegal in UTF-8.
        codepoint = SDL_INVALID_UNICODE_CODEPOINT;
    }

    Uint8 *p = (Uint8 *)dst;
    if (codepoint <= 0x7F) {
        *p = (Uint8)codepoint;
        ++dst;
    } else if (codepoint <= 0x7FF) {
        p[0] = 0xC0 | (Uint8)((codepoint >> 6) & 0x1F);
        p[1] = 0x80 | (Uint8)(codepoint & 0x3F);
        dst += 2;
    } else if (codepoint <= 0xFFFF) {
        p[0] = 0xE0 | (Uint8)((codepoint >> 12) & 0x0F);
        p[1] = 0x80 | (Uint8)((codepoint >> 6) & 0x3F);
        p[2] = 0x80 | (Uint8)(codepoint & 0x3F);
        dst += 3;
    } else {
        SDL_assert(codepoint <= 0x10FFFF);
        p[0] = 0xF0 | (Uint8)((codepoint >> 18) & 0x07);
        p[1] = 0x80 | (Uint8)((codepoint >> 12) & 0x3F);
        p[2] = 0x80 | (Uint8)((codepoint >> 6) & 0x3F);
        p[3] = 0x80 | (Uint8)(codepoint & 0x3F);
        dst += 4;
    }

    return dst;
}


// this expects `from` and `to` to be UTF-32 encoding!
int SDL_CaseFoldUnicode(Uint32 from, Uint32 *to)
{
    // !!! FIXME: since the hashtable is static, maybe we should binary
    // !!! FIXME: search it instead of walking the whole bucket.

    if (from < 128) {   // low-ASCII, easy!
        if ((from >= 'A') && (from <= 'Z')) {
            *to = 'a' + (from - 'A');
            return 1;
        }
    } else if (from <= 0xFFFF) {  // the Basic Multilingual Plane.
        const Uint8 hash = ((from ^ (from >> 8)) & 0xFF);
        const Uint16 from16 = (Uint16) from;

        // see if it maps to a single char (most common)...
        {
            const CaseFoldHashBucket1_16 *bucket = &case_fold_hash1_16[hash];
            const int count = (int) bucket->count;
            for (int i = 0; i < count; i++) {
                const CaseFoldMapping1_16 *mapping = &bucket->list[i];
                if (mapping->from == from16) {
                    *to = mapping->to0;
                    return 1;
                }
            }
        }

        // see if it folds down to two chars...
        {
            const CaseFoldHashBucket2_16 *bucket = &case_fold_hash2_16[hash & 15];
            const int count = (int) bucket->count;
            for (int i = 0; i < count; i++) {
                const CaseFoldMapping2_16 *mapping = &bucket->list[i];
                if (mapping->from == from16) {
                    to[0] = mapping->to0;
                    to[1] = mapping->to1;
                    return 2;
                }
            }
        }

        // okay, maybe it's _three_ characters!
        {
            const CaseFoldHashBucket3_16 *bucket = &case_fold_hash3_16[hash & 3];
            const int count = (int) bucket->count;
            for (int i = 0; i < count; i++) {
                const CaseFoldMapping3_16 *mapping = &bucket->list[i];
                if (mapping->from == from16) {
                    to[0] = mapping->to0;
                    to[1] = mapping->to1;
                    to[2] = mapping->to2;
                    return 3;
                }
            }
        }

    } else {  // codepoint that doesn't fit in 16 bits.
        const Uint8 hash = ((from ^ (from >> 8)) & 0xFF);
        const CaseFoldHashBucket1_32 *bucket = &case_fold_hash1_32[hash & 15];
        const int count = (int) bucket->count;
        for (int i = 0; i < count; i++) {
            const CaseFoldMapping1_32 *mapping = &bucket->list[i];
            if (mapping->from == from) {
                *to = mapping->to0;
                return 1;
            }
        }
    }

    // Not found...there's no folding needed for this codepoint.
    *to = from;
    return 1;
}

#define UNICODE_STRCASECMP(bits, slen1, slen2, update_slen1, update_slen2) \
    Uint32 folded1[3], folded2[3]; \
    int head1 = 0, tail1 = 0, head2 = 0, tail2 = 0; \
    while (true) { \
        Uint32 cp1, cp2; \
        if (head1 != tail1) { \
            cp1 = folded1[tail1++]; \
        } else { \
            const Uint##bits *str1start = (const Uint##bits *) str1; \
            head1 = SDL_CaseFoldUnicode(StepUTF##bits(&str1, slen1), folded1); \
            update_slen1; \
            cp1 = folded1[0]; \
            tail1 = 1; \
        } \
        if (head2 != tail2) { \
            cp2 = folded2[tail2++]; \
        } else { \
            const Uint##bits *str2start = (const Uint##bits *) str2; \
            head2 = SDL_CaseFoldUnicode(StepUTF##bits(&str2, slen2), folded2); \
            update_slen2; \
            cp2 = folded2[0]; \
            tail2 = 1; \
        } \
        if (cp1 < cp2) { \
            return -1; \
        } else if (cp1 > cp2) { \
            return 1; \
        } else if (cp1 == 0) { \
            break;  /* complete match. */ \
        } \
    } \
    return 0


static Uint32 StepUTF8(const char **_str, const size_t slen)
{
    /*
     * From rfc3629, the UTF-8 spec:
     *  https://www.ietf.org/rfc/rfc3629.txt
     *
     *   Char. number range  |        UTF-8 octet sequence
     *      (hexadecimal)    |              (binary)
     *   --------------------+---------------------------------------------
     *   0000 0000-0000 007F | 0xxxxxxx
     *   0000 0080-0000 07FF | 110xxxxx 10xxxxxx
     *   0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx
     *   0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
     */

    const Uint8 *str = (const Uint8 *) *_str;
    const Uint32 octet = (Uint32) (slen ? *str : 0);

    if (octet == 0) {  // null terminator, end of string.
        return 0;  // don't advance `*_str`.
    } else if ((octet & 0x80) == 0) {  // 0xxxxxxx: one byte codepoint.
        (*_str)++;
        return octet;
    } else if (((octet & 0xE0) == 0xC0) && (slen >= 2)) {  // 110xxxxx 10xxxxxx: two byte codepoint.
        const Uint8 str1 = str[1];
        if ((str1 & 0xC0) == 0x80) {  // If trailing bytes aren't 10xxxxxx, sequence is bogus.
            const Uint32 result = ((octet & 0x1F) << 6) | (str1 & 0x3F);
            if (result >= 0x0080) {  // rfc3629 says you can't use overlong sequences for smaller values.
                *_str += 2;
                return result;
            }
        }
    } else if (((octet & 0xF0) == 0xE0) && (slen >= 3)) {  // 1110xxxx 10xxxxxx 10xxxxxx: three byte codepoint.
        const Uint8 str1 = str[1];
        const Uint8 str2 = str[2];
        if (((str1 & 0xC0) == 0x80) && ((str2 & 0xC0) == 0x80)) {  // If trailing bytes aren't 10xxxxxx, sequence is bogus.
            const Uint32 octet2 = ((Uint32) (str1 & 0x3F)) << 6;
            const Uint32 octet3 = ((Uint32) (str2 & 0x3F));
            const Uint32 result = ((octet & 0x0F) << 12) | octet2 | octet3;
            if (result >= 0x800) {  // rfc3629 says you can't use overlong sequences for smaller values.
                if ((result < 0xD800) || (result > 0xDFFF)) {  // UTF-16 surrogate values are illegal in UTF-8.
                    *_str += 3;
                    return result;
                }
            }
        }
    } else if (((octet & 0xF8) == 0xF0) && (slen >= 4)) {  // 11110xxxx 10xxxxxx 10xxxxxx 10xxxxxx: four byte codepoint.
        const Uint8 str1 = str[1];
        const Uint8 str2 = str[2];
        const Uint8 str3 = str[3];
        if (((str1 & 0xC0) == 0x80) && ((str2 & 0xC0) == 0x80) && ((str3 & 0xC0) == 0x80)) {  // If trailing bytes aren't 10xxxxxx, sequence is bogus.
            const Uint32 octet2 = ((Uint32) (str1 & 0x1F)) << 12;
            const Uint32 octet3 = ((Uint32) (str2 & 0x3F)) << 6;
            const Uint32 octet4 = ((Uint32) (str3 & 0x3F));
            const Uint32 result = ((octet & 0x07) << 18) | octet2 | octet3 | octet4;
            if (result >= 0x10000) {  // rfc3629 says you can't use overlong sequences for smaller values.
                *_str += 4;
                return result;
            }
        }
    }

    // bogus byte, skip ahead, return a REPLACEMENT CHARACTER.
    (*_str)++;
    return SDL_INVALID_UNICODE_CODEPOINT;
}

Uint32 SDL_StepUTF8(const char **pstr, size_t *pslen)
{
    if (!pslen) {
        return StepUTF8(pstr, 4);  // 4 == max codepoint size.
    }
    const char *origstr = *pstr;
    const Uint32 result = StepUTF8(pstr, *pslen);
    *pslen -= (size_t) (*pstr - origstr);
    return result;
}

Uint32 SDL_StepBackUTF8(const char *start, const char **pstr)
{
    if (!pstr || *pstr <= start) {
        return 0;
    }

    // Step back over the previous UTF-8 character
    const char *str = *pstr;
    do {
        if (str == start) {
            break;
        }
        --str;
    } while ((*str & 0xC0) == 0x80);

    size_t length = (*pstr - str);
    *pstr = str;
    return StepUTF8(&str, length);
}

#if (SDL_SIZEOF_WCHAR_T == 2)
static Uint32 StepUTF16(const Uint16 **_str, const size_t slen)
{
    const Uint16 *str = *_str;
    Uint32 cp = (Uint32) *(str++);
    if (cp == 0) {
        return 0;  // don't advance string pointer.
    } else if ((cp >= 0xDC00) && (cp <= 0xDFFF)) {
        cp = SDL_INVALID_UNICODE_CODEPOINT;  // Orphaned second half of surrogate pair
    } else if ((cp >= 0xD800) && (cp <= 0xDBFF)) {  // start of surrogate pair!
        const Uint32 pair = (Uint32) *str;
        if ((pair == 0) || ((pair < 0xDC00) || (pair > 0xDFFF))) {
            cp = SDL_INVALID_UNICODE_CODEPOINT;
        } else {
            str++;  // eat the other surrogate.
            cp = 0x10000 + (((cp - 0xD800) << 10) | (pair - 0xDC00));
        }
    }

    *_str = str;
    return (cp > 0x10FFFF) ? SDL_INVALID_UNICODE_CODEPOINT : cp;
}
#elif (SDL_SIZEOF_WCHAR_T == 4)
static Uint32 StepUTF32(const Uint32 **_str, const size_t slen)
{
    if (!slen) {
        return 0;
    }

    const Uint32 *str = *_str;
    const Uint32 cp = *str;
    if (cp == 0) {
        return 0;  // don't advance string pointer.
    }

    (*_str)++;
    return (cp > 0x10FFFF) ? SDL_INVALID_UNICODE_CODEPOINT : cp;
}
#endif

#define UTF8_IsLeadByte(c)     ((c) >= 0xC0 && (c) <= 0xF4)
#define UTF8_IsTrailingByte(c) ((c) >= 0x80 && (c) <= 0xBF)

static size_t UTF8_GetTrailingBytes(unsigned char c)
{
    if (c >= 0xC0 && c <= 0xDF) {
        return 1;
    } else if (c >= 0xE0 && c <= 0xEF) {
        return 2;
    } else if (c >= 0xF0 && c <= 0xF4) {
        return 3;
    }

    return 0;
}

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOL) || !defined(HAVE_STRTOUL) || !defined(HAVE_STRTOLL) || !defined(HAVE_STRTOULL) || !defined(HAVE_STRTOD)
/**
 * Parses an unsigned long long and returns the unsigned value and sign bit.
 *
 * Positive values are clamped to ULLONG_MAX.
 * The result `value == 0 && negative` indicates negative overflow
 * and might need to be handled differently depending on whether a
 * signed or unsigned integer is being parsed.
 */
static size_t SDL_ScanUnsignedLongLongInternal(const char *text, int count, int radix, unsigned long long *valuep, bool *negativep)
{
    const unsigned long long ullong_max = ~0ULL;

    const char *text_start = text;
    const char *number_start = text_start;
    unsigned long long value = 0;
    bool negative = false;
    bool overflow = false;

    if (radix == 0 || (radix >= 2 && radix <= 36)) {
        while (SDL_isspace(*text)) {
            ++text;
        }
        if (*text == '-' || *text == '+') {
            negative = *text == '-';
            ++text;
        }
        if ((radix == 0 || radix == 16) && *text == '0' && (text[1] == 'x' || text[1] == 'X')) {
            text += 2;
            radix = 16;
        } else if (radix == 0 && *text == '0' && (text[1] >= '0' && text[1] <= '9')) {
            ++text;
            radix = 8;
        } else if (radix == 0) {
            radix = 10;
        }
        number_start = text;
        do {
            unsigned long long digit;
            if (*text >= '0' && *text <= '9') {
                digit = *text - '0';
            } else if (radix > 10) {
                if (*text >= 'A' && *text < 'A' + (radix - 10)) {
                    digit = 10 + (*text - 'A');
                } else if (*text >= 'a' && *text < 'a' + (radix - 10)) {
                    digit = 10 + (*text - 'a');
                } else {
                    break;
                }
            } else {
                break;
            }
            if (value != 0 && radix > ullong_max / value) {
                overflow = true;
            } else {
                value *= radix;
                if (digit > ullong_max - value) {
                    overflow = true;
                } else {
                    value += digit;
                }
            }
            ++text;
        } while (count == 0 || (text - text_start) != count);
    }
    if (text == number_start) {
        if (radix == 16 && text > text_start && (*(text - 1) == 'x' || *(text - 1) == 'X')) {
            // the string was "0x"; consume the '0' but not the 'x'
            --text;
        } else {
            // no number was parsed, and thus no characters were consumed
            text = text_start;
        }
    }
    if (overflow) {
        if (negative) {
            value = 0;
        } else {
            value = ullong_max;
        }
    } else if (value == 0) {
        negative = false;
    }
    *valuep = value;
    *negativep = negative;
    return text - text_start;
}
#endif

#ifndef HAVE_WCSTOL
// SDL_ScanUnsignedLongLongInternalW assumes that wchar_t can be converted to int without truncating bits
SDL_COMPILE_TIME_ASSERT(wchar_t_int, sizeof(wchar_t) <= sizeof(int));

/**
 * Parses an unsigned long long and returns the unsigned value and sign bit.
 *
 * Positive values are clamped to ULLONG_MAX.
 * The result `value == 0 && negative` indicates negative overflow
 * and might need to be handled differently depending on whether a
 * signed or unsigned integer is being parsed.
 */
static size_t SDL_ScanUnsignedLongLongInternalW(const wchar_t *text, int count, int radix, unsigned long long *valuep, bool *negativep)
{
    const unsigned long long ullong_max = ~0ULL;

    const wchar_t *text_start = text;
    const wchar_t *number_start = text_start;
    unsigned long long value = 0;
    bool negative = false;
    bool overflow = false;

    if (radix == 0 || (radix >= 2 && radix <= 36)) {
        while (SDL_isspace(*text)) {
            ++text;
        }
        if (*text == '-' || *text == '+') {
            negative = *text == '-';
            ++text;
        }
        if ((radix == 0 || radix == 16) && *text == '0' && (text[1] == 'x' || text[1] == 'X')) {
            text += 2;
            radix = 16;
        } else if (radix == 0 && *text == '0' && (text[1] >= '0' && text[1] <= '9')) {
            ++text;
            radix = 8;
        } else if (radix == 0) {
            radix = 10;
        }
        number_start = text;
        do {
            unsigned long long digit;
            if (*text >= '0' && *text <= '9') {
                digit = *text - '0';
            } else if (radix > 10) {
                if (*text >= 'A' && *text < 'A' + (radix - 10)) {
                    digit = 10 + (*text - 'A');
                } else if (*text >= 'a' && *text < 'a' + (radix - 10)) {
                    digit = 10 + (*text - 'a');
                } else {
                    break;
                }
            } else {
                break;
            }
            if (value != 0 && radix > ullong_max / value) {
                overflow = true;
            } else {
                value *= radix;
                if (digit > ullong_max - value) {
                    overflow = true;
                } else {
                    value += digit;
                }
            }
            ++text;
        } while (count == 0 || (text - text_start) != count);
    }
    if (text == number_start) {
        if (radix == 16 && text > text_start && (*(text - 1) == 'x' || *(text - 1) == 'X')) {
            // the string was "0x"; consume the '0' but not the 'x'
            --text;
        } else {
            // no number was parsed, and thus no characters were consumed
            text = text_start;
        }
    }
    if (overflow) {
        if (negative) {
            value = 0;
        } else {
            value = ullong_max;
        }
    } else if (value == 0) {
        negative = false;
    }
    *valuep = value;
    *negativep = negative;
    return text - text_start;
}
#endif

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOL)
static size_t SDL_ScanLong(const char *text, int count, int radix, long *valuep)
{
    const unsigned long long_max = (~0UL) >> 1;
    unsigned long long value;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternal(text, count, radix, &value, &negative);
    if (negative) {
        const unsigned long abs_long_min = long_max + 1;
        if (value == 0 || value > abs_long_min) {
            value = 0ULL - abs_long_min;
        } else {
            value = 0ULL - value;
        }
    } else if (value > long_max) {
        value = long_max;
    }
    *valuep = (long)value;
    return len;
}
#endif

#ifndef HAVE_WCSTOL
static size_t SDL_ScanLongW(const wchar_t *text, int count, int radix, long *valuep)
{
    const unsigned long long_max = (~0UL) >> 1;
    unsigned long long value;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternalW(text, count, radix, &value, &negative);
    if (negative) {
        const unsigned long abs_long_min = long_max + 1;
        if (value == 0 || value > abs_long_min) {
            value = 0ULL - abs_long_min;
        } else {
            value = 0ULL - value;
        }
    } else if (value > long_max) {
        value = long_max;
    }
    *valuep = (long)value;
    return len;
}
#endif

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOUL)
static size_t SDL_ScanUnsignedLong(const char *text, int count, int radix, unsigned long *valuep)
{
    const unsigned long ulong_max = ~0UL;
    unsigned long long value;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternal(text, count, radix, &value, &negative);
    if (negative) {
        if (value == 0 || value > ulong_max) {
            value = ulong_max;
        } else if (value == ulong_max) {
            value = 1;
        } else {
            value = 0ULL - value;
        }
    } else if (value > ulong_max) {
        value = ulong_max;
    }
    *valuep = (unsigned long)value;
    return len;
}
#endif

#ifndef HAVE_VSSCANF
static size_t SDL_ScanUintPtrT(const char *text, uintptr_t *valuep)
{
    const uintptr_t uintptr_max = ~(uintptr_t)0;
    unsigned long long value;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternal(text, 0, 16, &value, &negative);
    if (negative) {
        if (value == 0 || value > uintptr_max) {
            value = uintptr_max;
        } else if (value == uintptr_max) {
            value = 1;
        } else {
            value = 0ULL - value;
        }
    } else if (value > uintptr_max) {
        value = uintptr_max;
    }
    *valuep = (uintptr_t)value;
    return len;
}
#endif

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOLL)
static size_t SDL_ScanLongLong(const char *text, int count, int radix, long long *valuep)
{
    const unsigned long long llong_max = (~0ULL) >> 1;
    unsigned long long value;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternal(text, count, radix, &value, &negative);
    if (negative) {
        const unsigned long long abs_llong_min = llong_max + 1;
        if (value == 0 || value > abs_llong_min) {
            value = 0ULL - abs_llong_min;
        } else {
            value = 0ULL - value;
        }
    } else if (value > llong_max) {
        value = llong_max;
    }
    *valuep = value;
    return len;
}
#endif

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOULL) || !defined(HAVE_STRTOD)
static size_t SDL_ScanUnsignedLongLong(const char *text, int count, int radix, unsigned long long *valuep)
{
    const unsigned long long ullong_max = ~0ULL;
    bool negative;
    size_t len = SDL_ScanUnsignedLongLongInternal(text, count, radix, valuep, &negative);
    if (negative) {
        if (*valuep == 0) {
            *valuep = ullong_max;
        } else {
            *valuep = 0ULL - *valuep;
        }
    }
    return len;
}
#endif

#if !defined(HAVE_VSSCANF) || !defined(HAVE_STRTOD)
static size_t SDL_ScanFloat(const char *text, double *valuep)
{
    const char *text_start = text;
    const char *number_start = text_start;
    double value = 0.0;
    bool negative = false;

    while (SDL_isspace(*text)) {
        ++text;
    }
    if (*text == '-' || *text == '+') {
        negative = *text == '-';
        ++text;
    }
    number_start = text;
    if (SDL_isdigit(*text)) {
        value += SDL_strtoull(text, (char **)(&text), 10);
        if (*text == '.') {
            double denom = 10;
            ++text;
            while (SDL_isdigit(*text)) {
                value += (double)(*text - '0') / denom;
                denom *= 10;
                ++text;
            }
        }
    }
    if (text == number_start) {
        // no number was parsed, and thus no characters were consumed
        text = text_start;
    } else if (negative) {
        value = -value;
    }
    *valuep = value;
    return text - text_start;
}
#endif

int SDL_memcmp(const void *s1, const void *s2, size_t len)
{
#ifdef SDL_PLATFORM_VITA
    /*
      Using memcmp on NULL is UB per POSIX / C99 7.21.1/2.
      But, both linux and bsd allow that, with an exception:
      zero length strings are always identical, so NULLs are never dereferenced.
      sceClibMemcmp on PSVita doesn't allow that, so we check ourselves.
    */
    if (len == 0) {
        return 0;
    }
    return sceClibMemcmp(s1, s2, len);
#elif defined(HAVE_MEMCMP)
    return memcmp(s1, s2, len);
#else
    char *s1p = (char *)s1;
    char *s2p = (char *)s2;
    while (len--) {
        if (*s1p != *s2p) {
            return *s1p - *s2p;
        }
        ++s1p;
        ++s2p;
    }
    return 0;
#endif // HAVE_MEMCMP
}

size_t SDL_strlen(const char *string)
{
#ifdef HAVE_STRLEN
    return strlen(string);
#else
    size_t len = 0;
    while (*string++) {
        ++len;
    }
    return len;
#endif // HAVE_STRLEN
}

size_t SDL_strnlen(const char *string, size_t maxlen)
{
#ifdef HAVE_STRNLEN
    return strnlen(string, maxlen);
#else
    size_t len = 0;
    while (len < maxlen && *string++) {
        ++len;
    }
    return len;
#endif // HAVE_STRNLEN
}

size_t SDL_wcslen(const wchar_t *string)
{
#ifdef HAVE_WCSLEN
    return wcslen(string);
#else
    size_t len = 0;
    while (*string++) {
        ++len;
    }
    return len;
#endif // HAVE_WCSLEN
}

size_t SDL_wcsnlen(const wchar_t *string, size_t maxlen)
{
#ifdef HAVE_WCSNLEN
    return wcsnlen(string, maxlen);
#else
    size_t len = 0;
    while (len < maxlen && *string++) {
        ++len;
    }
    return len;
#endif // HAVE_WCSNLEN
}

size_t SDL_wcslcpy(SDL_OUT_Z_CAP(maxlen) wchar_t *dst, const wchar_t *src, size_t maxlen)
{
#ifdef HAVE_WCSLCPY
    return wcslcpy(dst, src, maxlen);
#else
    size_t srclen = SDL_wcslen(src);
    if (maxlen > 0) {
        size_t len = SDL_min(srclen, maxlen - 1);
        SDL_memcpy(dst, src, len * sizeof(wchar_t));
        dst[len] = '\0';
    }
    return srclen;
#endif // HAVE_WCSLCPY
}

size_t SDL_wcslcat(SDL_INOUT_Z_CAP(maxlen) wchar_t *dst, const wchar_t *src, size_t maxlen)
{
#ifdef HAVE_WCSLCAT
    return wcslcat(dst, src, maxlen);
#else
    size_t dstlen = SDL_wcslen(dst);
    size_t srclen = SDL_wcslen(src);
    if (dstlen < maxlen) {
        SDL_wcslcpy(dst + dstlen, src, maxlen - dstlen);
    }
    return dstlen + srclen;
#endif // HAVE_WCSLCAT
}

wchar_t *SDL_wcsdup(const wchar_t *string)
{
    size_t len = ((SDL_wcslen(string) + 1) * sizeof(wchar_t));
    wchar_t *newstr = (wchar_t *)SDL_malloc(len);
    if (newstr) {
        SDL_memcpy(newstr, string, len);
    }
    return newstr;
}

wchar_t *SDL_wcsnstr(const wchar_t *haystack, const wchar_t *needle, size_t maxlen)
{
    size_t length = SDL_wcslen(needle);
    if (length == 0) {
        return (wchar_t *)haystack;
    }
    while (maxlen >= length && *haystack) {
        if (maxlen >= length && SDL_wcsncmp(haystack, needle, length) == 0) {
            return (wchar_t *)haystack;
        }
        ++haystack;
        --maxlen;
    }
    return NULL;
}

wchar_t *SDL_wcsstr(const wchar_t *haystack, const wchar_t *needle)
{
#ifdef HAVE_WCSSTR
    return SDL_const_cast(wchar_t *, wcsstr(haystack, needle));
#else
    return SDL_wcsnstr(haystack, needle, SDL_wcslen(haystack));
#endif // HAVE_WCSSTR
}

int SDL_wcscmp(const wchar_t *str1, const wchar_t *str2)
{
#ifdef HAVE_WCSCMP
    return wcscmp(str1, str2);
#else
    while (*str1 && *str2) {
        if (*str1 != *str2) {
            break;
        }
        ++str1;
        ++str2;
    }
    return *str1 - *str2;
#endif // HAVE_WCSCMP
}

int SDL_wcsncmp(const wchar_t *str1, const wchar_t *str2, size_t maxlen)
{
#ifdef HAVE_WCSNCMP
    return wcsncmp(str1, str2, maxlen);
#else
    while (*str1 && *str2 && maxlen) {
        if (*str1 != *str2) {
            break;
        }
        ++str1;
        ++str2;
        --maxlen;
    }
    if (!maxlen) {
        return 0;
    }
    return *str1 - *str2;

#endif // HAVE_WCSNCMP
}

int SDL_wcscasecmp(const wchar_t *wstr1, const wchar_t *wstr2)
{
#if (SDL_SIZEOF_WCHAR_T == 2)
    const Uint16 *str1 = (const Uint16 *) wstr1;
    const Uint16 *str2 = (const Uint16 *) wstr2;
    UNICODE_STRCASECMP(16, 2, 2, (void) str1start, (void) str2start);  // always NULL-terminated, no need to adjust lengths.
#elif (SDL_SIZEOF_WCHAR_T == 4)
    const Uint32 *str1 = (const Uint32 *) wstr1;
    const Uint32 *str2 = (const Uint32 *) wstr2;
    UNICODE_STRCASECMP(32, 1, 1, (void) str1start, (void) str2start);  // always NULL-terminated, no need to adjust lengths.
#else
    #error Unexpected wchar_t size
    return -1;
#endif
}

int SDL_wcsncasecmp(const wchar_t *wstr1, const wchar_t *wstr2, size_t maxlen)
{
    size_t slen1 = maxlen;
    size_t slen2 = maxlen;

#if (SDL_SIZEOF_WCHAR_T == 2)
    const Uint16 *str1 = (const Uint16 *) wstr1;
    const Uint16 *str2 = (const Uint16 *) wstr2;
    UNICODE_STRCASECMP(16, slen1, slen2, slen1 -= (size_t) (str1 - str1start), slen2 -= (size_t) (str2 - str2start));
#elif (SDL_SIZEOF_WCHAR_T == 4)
    const Uint32 *str1 = (const Uint32 *) wstr1;
    const Uint32 *str2 = (const Uint32 *) wstr2;
    UNICODE_STRCASECMP(32, slen1, slen2, slen1 -= (size_t) (str1 - str1start), slen2 -= (size_t) (str2 - str2start));
#else
    #error Unexpected wchar_t size
    return -1;
#endif
}

long SDL_wcstol(const wchar_t *string, wchar_t **endp, int base)
{
#ifdef HAVE_WCSTOL
    return wcstol(string, endp, base);
#else
    long value = 0;
    size_t len = SDL_ScanLongW(string, 0, base, &value);
    if (endp) {
        *endp = (wchar_t *)string + len;
    }
    return value;
#endif // HAVE_WCSTOL
}

size_t SDL_strlcpy(SDL_OUT_Z_CAP(maxlen) char *dst, const char *src, size_t maxlen)
{
#ifdef HAVE_STRLCPY
    return strlcpy(dst, src, maxlen);
#else
    size_t srclen = SDL_strlen(src);
    if (maxlen > 0) {
        size_t len = SDL_min(srclen, maxlen - 1);
        SDL_memcpy(dst, src, len);
        dst[len] = '\0';
    }
    return srclen;
#endif // HAVE_STRLCPY
}

size_t SDL_utf8strlcpy(SDL_OUT_Z_CAP(dst_bytes) char *dst, const char *src, size_t dst_bytes)
{
    size_t bytes = 0;

	if (dst_bytes > 0) {
		size_t src_bytes = SDL_strlen(src);
		size_t i = 0;
		size_t trailing_bytes = 0;

		bytes = SDL_min(src_bytes, dst_bytes - 1);
		if (bytes) {
			unsigned char c = (unsigned char)src[bytes - 1];
			if (UTF8_IsLeadByte(c)) {
				--bytes;
			} else if (UTF8_IsTrailingByte(c)) {
				for (i = bytes - 1; i != 0; --i) {
					c = (unsigned char)src[i];
					trailing_bytes = UTF8_GetTrailingBytes(c);
					if (trailing_bytes) {
						if ((bytes - i) != (trailing_bytes + 1)) {
							bytes = i;
						}

						break;
					}
				}
			}
			SDL_memcpy(dst, src, bytes);
		}
		dst[bytes] = '\0';
	}

    return bytes;
}

size_t SDL_utf8strlen(const char *str)
{
    size_t result = 0;
    while (SDL_StepUTF8(&str, NULL)) {
        result++;
    }
    return result;
}

size_t SDL_utf8strnlen(const char *str, size_t bytes)
{
    size_t result = 0;
    while (SDL_StepUTF8(&str, &bytes)) {
        result++;
    }
    return result;
}

size_t SDL_strlcat(SDL_INOUT_Z_CAP(maxlen) char *dst, const char *src, size_t maxlen)
{
#ifdef HAVE_STRLCAT
    return strlcat(dst, src, maxlen);
#else
    size_t dstlen = SDL_strlen(dst);
    size_t srclen = SDL_strlen(src);
    if (dstlen < maxlen) {
        SDL_strlcpy(dst + dstlen, src, maxlen - dstlen);
    }
    return dstlen + srclen;
#endif // HAVE_STRLCAT
}

char *SDL_strdup(const char *string)
{
    size_t len = SDL_strlen(string) + 1;
    char *newstr = (char *)SDL_malloc(len);
    if (newstr) {
        SDL_memcpy(newstr, string, len);
    }
    return newstr;
}

char *SDL_strndup(const char *string, size_t maxlen)
{
    size_t len = SDL_strnlen(string, maxlen);
    char *newstr = (char *)SDL_malloc(len + 1);
    if (newstr) {
        SDL_memcpy(newstr, string, len);
        newstr[len] = '\0';
    }
    return newstr;
}

char *SDL_strrev(char *string)
{
#ifdef HAVE__STRREV
    return _strrev(string);
#else
    size_t len = SDL_strlen(string);
    char *a = &string[0];
    char *b = &string[len - 1];
    len /= 2;
    while (len--) {
        const char c = *a; // NOLINT(clang-analyzer-core.uninitialized.Assign)
        *a++ = *b;
        *b-- = c;
    }
    return string;
#endif // HAVE__STRREV
}

char *SDL_strupr(char *string)
{
    char *bufp = string;
    while (*bufp) {
        *bufp = (char)SDL_toupper((unsigned char)*bufp);
        ++bufp;
    }
    return string;
}

char *SDL_strlwr(char *string)
{
    char *bufp = string;
    while (*bufp) {
        *bufp = (char)SDL_tolower((unsigned char)*bufp);
        ++bufp;
    }
    return string;
}

char *SDL_strchr(const char *string, int c)
{
#ifdef HAVE_STRCHR
    return SDL_const_cast(char *, strchr(string, c));
#elif defined(HAVE_INDEX)
    return SDL_const_cast(char *, index(string, c));
#else
    while (*string) {
        if (*string == c) {
            return (char *)string;
        }
        ++string;
    }
    if (c == '\0') {
        return (char *)string;
    }
    return NULL;
#endif // HAVE_STRCHR
}

char *SDL_strrchr(const char *string, int c)
{
#ifdef HAVE_STRRCHR
    return SDL_const_cast(char *, strrchr(string, c));
#elif defined(HAVE_RINDEX)
    return SDL_const_cast(char *, rindex(string, c));
#else
    const char *bufp = string + SDL_strlen(string);
    while (bufp >= string) {
        if (*bufp == c) {
            return (char *)bufp;
        }
        --bufp;
    }
    return NULL;
#endif // HAVE_STRRCHR
}

char *SDL_strnstr(const char *haystack, const char *needle, size_t maxlen)
{
#ifdef HAVE_STRNSTR
    return SDL_const_cast(char *, strnstr(haystack, needle, maxlen));
#else
    size_t length = SDL_strlen(needle);
    if (length == 0) {
        return (char *)haystack;
    }
    while (maxlen >= length && *haystack) {
        if (SDL_strncmp(haystack, needle, length) == 0) {
            return (char *)haystack;
        }
        ++haystack;
        --maxlen;
    }
    return NULL;
#endif // HAVE_STRSTR
}

char *SDL_strstr(const char *haystack, const char *needle)
{
#ifdef HAVE_STRSTR
    return SDL_const_cast(char *, strstr(haystack, needle));
#else
    return SDL_strnstr(haystack, needle, SDL_strlen(haystack));
#endif // HAVE_STRSTR
}

char *SDL_strcasestr(const char *haystack, const char *needle)
{
    const size_t length = SDL_strlen(needle);
    do {
        if (SDL_strncasecmp(haystack, needle, length) == 0) {
            return (char *)haystack;
        }
    } while (SDL_StepUTF8(&haystack, NULL));  // move ahead by a full codepoint at a time, regardless of bytes.

    return NULL;
}

#if !defined(HAVE__LTOA) || !defined(HAVE__I64TOA) || \
    !defined(HAVE__ULTOA) || !defined(HAVE__UI64TOA)
static const char ntoa_table[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
};
#endif // ntoa() conversion table

char *SDL_itoa(int value, char *string, int radix)
{
#ifdef HAVE_ITOA
    return itoa(value, string, radix);
#else
    return SDL_ltoa((long)value, string, radix);
#endif // HAVE_ITOA
}

char *SDL_uitoa(unsigned int value, char *string, int radix)
{
#ifdef HAVE__UITOA
    return _uitoa(value, string, radix);
#else
    return SDL_ultoa((unsigned long)value, string, radix);
#endif // HAVE__UITOA
}

char *SDL_ltoa(long value, char *string, int radix)
{
#ifdef HAVE__LTOA
    return _ltoa(value, string, radix);
#else
    char *bufp = string;

    if (value < 0) {
        *bufp++ = '-';
        SDL_ultoa(-value, bufp, radix);
    } else {
        SDL_ultoa(value, bufp, radix);
    }

    return string;
#endif // HAVE__LTOA
}

char *SDL_ultoa(unsigned long value, char *string, int radix)
{
#ifdef HAVE__ULTOA
    return _ultoa(value, string, radix);
#else
    char *bufp = string;

    if (value) {
        while (value > 0) {
            *bufp++ = ntoa_table[value % radix];
            value /= radix;
        }
    } else {
        *bufp++ = '0';
    }
    *bufp = '\0';

    // The numbers went into the string backwards. :)
    SDL_strrev(string);

    return string;
#endif // HAVE__ULTOA
}

char *SDL_lltoa(long long value, char *string, int radix)
{
#ifdef HAVE__I64TOA
    return _i64toa(value, string, radix);
#else
    char *bufp = string;

    if (value < 0) {
        *bufp++ = '-';
        SDL_ulltoa(-value, bufp, radix);
    } else {
        SDL_ulltoa(value, bufp, radix);
    }

    return string;
#endif // HAVE__I64TOA
}

char *SDL_ulltoa(unsigned long long value, char *string, int radix)
{
#ifdef HAVE__UI64TOA
    return _ui64toa(value, string, radix);
#else
    char *bufp = string;

    if (value) {
        while (value > 0) {
            *bufp++ = ntoa_table[value % radix];
            value /= radix;
        }
    } else {
        *bufp++ = '0';
    }
    *bufp = '\0';

    // The numbers went into the string backwards. :)
    SDL_strrev(string);

    return string;
#endif // HAVE__UI64TOA
}

int SDL_atoi(const char *string)
{
#ifdef HAVE_ATOI
    return atoi(string);
#else
    return SDL_strtol(string, NULL, 10);
#endif // HAVE_ATOI
}

double SDL_atof(const char *string)
{
#ifdef HAVE_ATOF
    return atof(string);
#else
    return SDL_strtod(string, NULL);
#endif // HAVE_ATOF
}

long SDL_strtol(const char *string, char **endp, int base)
{
#ifdef HAVE_STRTOL
    return strtol(string, endp, base);
#else
    long value = 0;
    size_t len = SDL_ScanLong(string, 0, base, &value);
    if (endp) {
        *endp = (char *)string + len;
    }
    return value;
#endif // HAVE_STRTOL
}

unsigned long SDL_strtoul(const char *string, char **endp, int base)
{
#ifdef HAVE_STRTOUL
    return strtoul(string, endp, base);
#else
    unsigned long value = 0;
    size_t len = SDL_ScanUnsignedLong(string, 0, base, &value);
    if (endp) {
        *endp = (char *)string + len;
    }
    return value;
#endif // HAVE_STRTOUL
}

long long SDL_strtoll(const char *string, char **endp, int base)
{
#ifdef HAVE_STRTOLL
    return strtoll(string, endp, base);
#else
    long long value = 0;
    size_t len = SDL_ScanLongLong(string, 0, base, &value);
    if (endp) {
        *endp = (char *)string + len;
    }
    return value;
#endif // HAVE_STRTOLL
}

unsigned long long SDL_strtoull(const char *string, char **endp, int base)
{
#ifdef HAVE_STRTOULL
    return strtoull(string, endp, base);
#else
    unsigned long long value = 0;
    size_t len = SDL_ScanUnsignedLongLong(string, 0, base, &value);
    if (endp) {
        *endp = (char *)string + len;
    }
    return value;
#endif // HAVE_STRTOULL
}

double SDL_strtod(const char *string, char **endp)
{
#ifdef HAVE_STRTOD
    return strtod(string, endp);
#else
    double value;
    size_t len = SDL_ScanFloat(string, &value);
    if (endp) {
        *endp = (char *)string + len;
    }
    return value;
#endif // HAVE_STRTOD
}

int SDL_strcmp(const char *str1, const char *str2)
{
#ifdef HAVE_STRCMP
    return strcmp(str1, str2);
#else
    int result;

    while (1) {
        result = ((unsigned char)*str1 - (unsigned char)*str2);
        if (result != 0 || (*str1 == '\0' /* && *str2 == '\0'*/)) {
            break;
        }
        ++str1;
        ++str2;
    }
    return result;
#endif // HAVE_STRCMP
}

int SDL_strncmp(const char *str1, const char *str2, size_t maxlen)
{
#ifdef HAVE_STRNCMP
    return strncmp(str1, str2, maxlen);
#else
    int result = 0;

    while (maxlen) {
        result = (int)(unsigned char)*str1 - (unsigned char)*str2;
        if (result != 0 || *str1 == '\0' /* && *str2 == '\0'*/) {
            break;
        }
        ++str1;
        ++str2;
        --maxlen;
    }
    return result;
#endif // HAVE_STRNCMP
}

int SDL_strcasecmp(const char *str1, const char *str2)
{
    UNICODE_STRCASECMP(8, 4, 4, (void) str1start, (void) str2start);  // always NULL-terminated, no need to adjust lengths.
}

int SDL_strncasecmp(const char *str1, const char *str2, size_t maxlen)
{
    size_t slen1 = maxlen;
    size_t slen2 = maxlen;
    UNICODE_STRCASECMP(8, slen1, slen2, slen1 -= (size_t) (str1 - ((const char *) str1start)), slen2 -= (size_t) (str2 - ((const char *) str2start)));
}

int SDL_sscanf(const char *text, SDL_SCANF_FORMAT_STRING const char *fmt, ...)
{
    int rc;
    va_list ap;
    va_start(ap, fmt);
    rc = SDL_vsscanf(text, fmt, ap);
    va_end(ap);
    return rc;
}

#ifdef HAVE_VSSCANF
int SDL_vsscanf(const char *text, const char *fmt, va_list ap)
{
    return vsscanf(text, fmt, ap);
}
#else
static bool CharacterMatchesSet(char c, const char *set, size_t set_len)
{
    bool invert = false;
    bool result = false;

    if (*set == '^') {
        invert = true;
        ++set;
        --set_len;
    }
    while (set_len > 0 && !result) {
        if (set_len >= 3 && set[1] == '-') {
            char low_char = SDL_min(set[0], set[2]);
            char high_char = SDL_max(set[0], set[2]);
            if (c >= low_char && c <= high_char) {
                result = true;
            }
            set += 3;
            set_len -= 3;
        } else {
            if (c == *set) {
                result = true;
            }
            ++set;
            --set_len;
        }
    }
    if (invert) {
        result = !result;
    }
    return result;
}

// NOLINTNEXTLINE(readability-non-const-parameter)
int SDL_vsscanf(const char *text, SDL_SCANF_FORMAT_STRING const char *fmt, va_list ap)
{
    const char *start = text;
    int result = 0;

    if (!text || !*text) {
        return -1;
    }

    while (*fmt) {
        if (*fmt == ' ') {
            while (SDL_isspace((unsigned char)*text)) {
                ++text;
            }
            ++fmt;
            continue;
        }
        if (*fmt == '%') {
            bool done = false;
            long count = 0;
            int radix = 10;
            enum
            {
                DO_SHORT,
                DO_INT,
                DO_LONG,
                DO_LONGLONG,
                DO_SIZE_T
            } inttype = DO_INT;
            size_t advance;
            bool suppress = false;

            ++fmt;
            if (*fmt == '%') {
                if (*text == '%') {
                    ++text;
                    ++fmt;
                    continue;
                }
                break;
            }
            if (*fmt == '*') {
                suppress = true;
                ++fmt;
            }
            fmt += SDL_ScanLong(fmt, 0, 10, &count);

            if (*fmt == 'c') {
                if (!count) {
                    count = 1;
                }
                if (suppress) {
                    while (count--) {
                        ++text;
                    }
                } else {
                    char *valuep = va_arg(ap, char *);
                    while (count--) {
                        *valuep++ = *text++;
                    }
                    ++result;
                }
                continue;
            }

            while (SDL_isspace((unsigned char)*text)) {
                ++text;
            }

            // FIXME: implement more of the format specifiers
            while (!done) {
                switch (*fmt) {
                case '*':
                    suppress = true;
                    break;
                case 'h':
                    if (inttype == DO_INT) {
                        inttype = DO_SHORT;
                    } else if (inttype > DO_SHORT) {
                        ++inttype;
                    }
                    break;
                case 'l':
                    if (inttype < DO_LONGLONG) {
                        ++inttype;
                    }
                    break;
                case 'I':
                    if (SDL_strncmp(fmt, "I64", 3) == 0) {
                        fmt += 2;
                        inttype = DO_LONGLONG;
                    }
                    break;
                case 'z':
                    inttype = DO_SIZE_T;
                    break;
                case 'i':
                {
                    int index = 0;
                    if (text[index] == '-') {
                        ++index;
                    }
                    if (text[index] == '0') {
                        if (SDL_tolower((unsigned char)text[index + 1]) == 'x') {
                            radix = 16;
                        } else {
                            radix = 8;
                        }
                    }
                }
                    SDL_FALLTHROUGH;
                case 'd':
                    if (inttype == DO_LONGLONG) {
                        long long value = 0;
                        advance = SDL_ScanLongLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            Sint64 *valuep = va_arg(ap, Sint64 *);
                            *valuep = value;
                            ++result;
                        }
                    } else if (inttype == DO_SIZE_T) {
                        long long value = 0;
                        advance = SDL_ScanLongLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            size_t *valuep = va_arg(ap, size_t *);
                            *valuep = (size_t)value;
                            ++result;
                        }
                    } else {
                        long value = 0;
                        advance = SDL_ScanLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            switch (inttype) {
                            case DO_SHORT:
                            {
                                short *valuep = va_arg(ap, short *);
                                *valuep = (short)value;
                            } break;
                            case DO_INT:
                            {
                                int *valuep = va_arg(ap, int *);
                                *valuep = (int)value;
                            } break;
                            case DO_LONG:
                            {
                                long *valuep = va_arg(ap, long *);
                                *valuep = value;
                            } break;
                            case DO_LONGLONG:
                            case DO_SIZE_T:
                                // Handled above
                                break;
                            }
                            ++result;
                        }
                    }
                    done = true;
                    break;
                case 'o':
                    if (radix == 10) {
                        radix = 8;
                    }
                    SDL_FALLTHROUGH;
                case 'x':
                case 'X':
                    if (radix == 10) {
                        radix = 16;
                    }
                    SDL_FALLTHROUGH;
                case 'u':
                    if (inttype == DO_LONGLONG) {
                        unsigned long long value = 0;
                        advance = SDL_ScanUnsignedLongLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            Uint64 *valuep = va_arg(ap, Uint64 *);
                            *valuep = value;
                            ++result;
                        }
                    } else if (inttype == DO_SIZE_T) {
                        unsigned long long value = 0;
                        advance = SDL_ScanUnsignedLongLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            size_t *valuep = va_arg(ap, size_t *);
                            *valuep = (size_t)value;
                            ++result;
                        }
                    } else {
                        unsigned long value = 0;
                        advance = SDL_ScanUnsignedLong(text, count, radix, &value);
                        text += advance;
                        if (advance && !suppress) {
                            switch (inttype) {
                            case DO_SHORT:
                            {
                                short *valuep = va_arg(ap, short *);
                                *valuep = (short)value;
                            } break;
                            case DO_INT:
                            {
                                int *valuep = va_arg(ap, int *);
                                *valuep = (int)value;
                            } break;
                            case DO_LONG:
                            {
                                long *valuep = va_arg(ap, long *);
                                *valuep = value;
                            } break;
                            case DO_LONGLONG:
                            case DO_SIZE_T:
                                // Handled above
                                break;
                            }
                            ++result;
                        }
                    }
                    done = true;
                    break;
                case 'p':
                {
                    uintptr_t value = 0;
                    advance = SDL_ScanUintPtrT(text, &value);
                    text += advance;
                    if (advance && !suppress) {
                        void **valuep = va_arg(ap, void **);
                        *valuep = (void *)value;
                        ++result;
                    }
                }
                    done = true;
                    break;
                case 'f':
                {
                    double value = 0.0;
                    advance = SDL_ScanFloat(text, &value);
                    text += advance;
                    if (advance && !suppress) {
                        float *valuep = va_arg(ap, float *);
                        *valuep = (float)value;
                        ++result;
                    }
                }
                    done = true;
                    break;
                case 's':
                    if (suppress) {
                        while (!SDL_isspace((unsigned char)*text)) {
                            ++text;
                            if (count) {
                                if (--count == 0) {
                                    break;
                                }
                            }
                        }
                    } else {
                        char *valuep = va_arg(ap, char *);
                        while (!SDL_isspace((unsigned char)*text)) {
                            *valuep++ = *text++;
                            if (count) {
                                if (--count == 0) {
                                    break;
                                }
                            }
                        }
                        *valuep = '\0';
                        ++result;
                    }
                    done = true;
                    break;
                case 'n':
                    switch (inttype) {
                    case DO_SHORT:
                    {
                        short *valuep = va_arg(ap, short *);
                        *valuep = (short)(text - start);
                    } break;
                    case DO_INT:
                    {
                        int *valuep = va_arg(ap, int *);
                        *valuep = (int)(text - start);
                    } break;
                    case DO_LONG:
                    {
                        long *valuep = va_arg(ap, long *);
                        *valuep = (long)(text - start);
                    } break;
                    case DO_LONGLONG:
                    {
                        long long *valuep = va_arg(ap, long long *);
                        *valuep = (long long)(text - start);
                    } break;
                    case DO_SIZE_T:
                    {
                        size_t *valuep = va_arg(ap, size_t *);
                        *valuep = (size_t)(text - start);
                    } break;
                    }
                    done = true;
                    break;
                case '[':
                {
                    const char *set = fmt + 1;
                    while (*fmt && *fmt != ']') {
                        ++fmt;
                    }
                    if (*fmt) {
                        size_t set_len = (fmt - set);
                        if (suppress) {
                            while (CharacterMatchesSet(*text, set, set_len)) {
                                ++text;
                                if (count) {
                                    if (--count == 0) {
                                        break;
                                    }
                                }
                            }
                        } else {
                            bool had_match = false;
                            char *valuep = va_arg(ap, char *);
                            while (CharacterMatchesSet(*text, set, set_len)) {
                                had_match = true;
                                *valuep++ = *text++;
                                if (count) {
                                    if (--count == 0) {
                                        break;
                                    }
                                }
                            }
                            *valuep = '\0';
                            if (had_match) {
                                ++result;
                            }
                        }
                    }
                }
                    done = true;
                    break;
                default:
                    done = true;
                    break;
                }
                ++fmt;
            }
            continue;
        }
        if (*text == *fmt) {
            ++text;
            ++fmt;
            continue;
        }
        // Text didn't match format specifier
        break;
    }

    return result;
}
#endif // HAVE_VSSCANF

int SDL_snprintf(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    int result;

    va_start(ap, fmt);
    result = SDL_vsnprintf(text, maxlen, fmt, ap);
    va_end(ap);

    return result;
}

int SDL_swprintf(SDL_OUT_Z_CAP(maxlen) wchar_t *text, size_t maxlen, SDL_PRINTF_FORMAT_STRING const wchar_t *fmt, ...)
{
    va_list ap;
    int result;

    va_start(ap, fmt);
    result = SDL_vswprintf(text, maxlen, fmt, ap);
    va_end(ap);

    return result;
}

#if defined(HAVE_LIBC) && defined(__WATCOMC__)
// _vsnprintf() doesn't ensure nul termination
int SDL_vsnprintf(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, const char *fmt, va_list ap)
{
    int result;
    if (!fmt) {
        fmt = "";
    }
    result = _vsnprintf(text, maxlen, fmt, ap);
    if (maxlen > 0) {
        text[maxlen - 1] = '\0';
    }
    if (result < 0) {
        result = (int)maxlen;
    }
    return result;
}
#elif defined(HAVE_VSNPRINTF)
int SDL_vsnprintf(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, const char *fmt, va_list ap)
{
    if (!fmt) {
        fmt = "";
    }
    return vsnprintf(text, maxlen, fmt, ap);
}
#else
#define TEXT_AND_LEN_ARGS (length < maxlen) ? &text[length] : NULL, (length < maxlen) ? (maxlen - length) : 0

// FIXME: implement more of the format specifiers
typedef enum
{
    SDL_CASE_NOCHANGE,
    SDL_CASE_LOWER,
    SDL_CASE_UPPER
} SDL_letter_case;

typedef struct
{
    bool left_justify;
    bool force_sign;
    bool force_type; // for now: used only by float printer, ignored otherwise.
    bool pad_zeroes;
    SDL_letter_case force_case;
    int width;
    int radix;
    int precision;
} SDL_FormatInfo;

static size_t SDL_PrintString(char *text, size_t maxlen, SDL_FormatInfo *info, const char *string)
{
    const char fill = (info && info->pad_zeroes) ? '0' : ' ';
    size_t width = 0;
    size_t filllen = 0;
    size_t length = 0;
    size_t slen, sz;

    if (!string) {
        string = "(null)";
    }

    sz = SDL_strlen(string);
    if (info && info->width > 0 && (size_t)info->width > sz) {
        width = info->width - sz;
        if (info->precision >= 0 && (size_t)info->precision < sz) {
            width += sz - (size_t)info->precision;
        }

        filllen = SDL_min(width, maxlen);
        if (!info->left_justify) {
            SDL_memset(text, fill, filllen);
            text += filllen;
            maxlen -= filllen;
            length += width;
            filllen = 0;
        }
    }

    SDL_strlcpy(text, string, maxlen);
    length += sz;

    if (filllen > 0) {
        SDL_memset(text + sz, fill, filllen);
        length += width;
    }

    if (info) {
        if (info->precision >= 0 && (size_t)info->precision < sz) {
            slen = (size_t)info->precision;
            if (slen < maxlen) {
                text[slen] = '\0';
            }
            length -= (sz - slen);
        }
        if (maxlen > 1) {
            if (info->force_case == SDL_CASE_LOWER) {
                SDL_strlwr(text);
            } else if (info->force_case == SDL_CASE_UPPER) {
                SDL_strupr(text);
            }
        }
    }
    return length;
}

static size_t SDL_PrintStringW(char *text, size_t maxlen, SDL_FormatInfo *info, const wchar_t *wide_string)
{
    size_t length = 0;
    if (wide_string) {
        char *string = SDL_iconv_string("UTF-8", "WCHAR_T", (char *)(wide_string), (SDL_wcslen(wide_string) + 1) * sizeof(*wide_string));
        length = SDL_PrintString(TEXT_AND_LEN_ARGS, info, string);
        SDL_free(string);
    } else {
        length = SDL_PrintString(TEXT_AND_LEN_ARGS, info, NULL);
    }
    return length;
}

static void SDL_IntPrecisionAdjust(char *num, size_t maxlen, SDL_FormatInfo *info)
{ // left-pad num with zeroes.
    size_t sz, pad, have_sign;

    if (!info) {
        return;
    }

    have_sign = 0;
    if (*num == '-' || *num == '+') {
        have_sign = 1;
        ++num;
        --maxlen;
    }
    sz = SDL_strlen(num);
    if (info->precision > 0 && sz < (size_t)info->precision) {
        pad = (size_t)info->precision - sz;
        if (pad + sz + 1 <= maxlen) { // otherwise ignore the precision
            SDL_memmove(num + pad, num, sz + 1);
            SDL_memset(num, '0', pad);
        }
    }
    info->precision = -1; // so that SDL_PrintString() doesn't make a mess.

    if (info->pad_zeroes && info->width > 0 && (size_t)info->width > sz + have_sign) {
        /* handle here: spaces are added before the sign
           but zeroes must be placed _after_ the sign. */
        // sz hasn't changed: we ignore pad_zeroes if a precision is given.
        pad = (size_t)info->width - sz - have_sign;
        if (pad + sz + 1 <= maxlen) {
            SDL_memmove(num + pad, num, sz + 1);
            SDL_memset(num, '0', pad);
        }
        info->width = 0; // so that SDL_PrintString() doesn't make a mess.
    }
}

static size_t SDL_PrintLong(char *text, size_t maxlen, SDL_FormatInfo *info, long value)
{
    char num[130], *p = num;

    if (info->force_sign && value >= 0L) {
        *p++ = '+';
    }

    SDL_ltoa(value, p, info ? info->radix : 10);
    SDL_IntPrecisionAdjust(num, sizeof(num), info);
    return SDL_PrintString(text, maxlen, info, num);
}

static size_t SDL_PrintUnsignedLong(char *text, size_t maxlen, SDL_FormatInfo *info, unsigned long value)
{
    char num[130];

    SDL_ultoa(value, num, info ? info->radix : 10);
    SDL_IntPrecisionAdjust(num, sizeof(num), info);
    return SDL_PrintString(text, maxlen, info, num);
}

static size_t SDL_PrintLongLong(char *text, size_t maxlen, SDL_FormatInfo *info, long long value)
{
    char num[130], *p = num;

    if (info->force_sign && value >= (Sint64)0) {
        *p++ = '+';
    }

    SDL_lltoa(value, p, info ? info->radix : 10);
    SDL_IntPrecisionAdjust(num, sizeof(num), info);
    return SDL_PrintString(text, maxlen, info, num);
}

static size_t SDL_PrintUnsignedLongLong(char *text, size_t maxlen, SDL_FormatInfo *info, unsigned long long value)
{
    char num[130];

    SDL_ulltoa(value, num, info ? info->radix : 10);
    SDL_IntPrecisionAdjust(num, sizeof(num), info);
    return SDL_PrintString(text, maxlen, info, num);
}

static size_t SDL_PrintFloat(char *text, size_t maxlen, SDL_FormatInfo *info, double arg, bool g)
{
    char num[327];
    size_t length = 0;
    size_t integer_length;
    int precision = info->precision;

    // This isn't especially accurate, but hey, it's easy. :)
    unsigned long long value;

    if (arg < 0.0 || (arg == 0.0 && 1.0 / arg < 0.0)) { // additional check for signed zero
        num[length++] = '-';
        arg = -arg;
    } else if (info->force_sign) {
        num[length++] = '+';
    }
    value = (unsigned long long)arg;
    integer_length = SDL_PrintUnsignedLongLong(&num[length], sizeof(num) - length, NULL, value);
    length += integer_length;
    arg -= value;
    if (precision < 0) {
        precision = 6;
    }
    if (g) {
        // The precision includes the integer portion
        precision -= SDL_min((int)integer_length, precision);
    }
    if (info->force_type || precision > 0) {
        const char decimal_separator = '.';
        double integer_value;

        SDL_assert(length < sizeof(num));
        num[length++] = decimal_separator;
        while (precision > 1) {
            arg *= 10.0;
            arg = SDL_modf(arg, &integer_value);
            SDL_assert(length < sizeof(num));
            num[length++] = '0' + (char)integer_value;
            --precision;
        }
        if (precision == 1) {
            arg *= 10.0;
            integer_value = SDL_round(arg);
            if (integer_value == 10.0) {
                // Carry the one...
                size_t i;

                for (i = length; i--; ) {
                    if (num[i] == decimal_separator) {
                        continue;
                    }
                    if (num[i] == '9') {
                        num[i] = '0';
                        if (i == 0 || num[i - 1] == '-' || num[i - 1] == '+') {
                            SDL_memmove(&num[i+1], &num[i], length - i);
                            num[i] = '1';
                            ++length;
                            break;
                        }
                    } else {
                        ++num[i];
                        break;
                    }
                }
                SDL_assert(length < sizeof(num));
                num[length++] = '0';
            } else {
                SDL_assert(length < sizeof(num));
                num[length++] = '0' + (char)integer_value;
            }
        }

        if (g) {
            // Trim trailing zeroes and decimal separator
            size_t i;

            for (i = length - 1; num[i] != decimal_separator; --i) {
                if (num[i] == '0') {
                    --length;
                } else {
                    break;
                }
            }
            if (num[i] == decimal_separator) {
                --length;
            }
        }
    }
    num[length] = '\0';

    info->precision = -1;
    length = SDL_PrintString(text, maxlen, info, num);

    if (info->width > 0 && (size_t)info->width > length) {
        const char fill = info->pad_zeroes ? '0' : ' ';
        size_t width = info->width - length;
        size_t filllen, movelen;

        filllen = SDL_min(width, maxlen);
        movelen = SDL_min(length, (maxlen - filllen));
        SDL_memmove(&text[filllen], text, movelen);
        SDL_memset(text, fill, filllen);
        length += width;
    }
    return length;
}

static size_t SDL_PrintPointer(char *text, size_t maxlen, SDL_FormatInfo *info, const void *value)
{
    char num[130];
    size_t length;

    if (!value) {
        return SDL_PrintString(text, maxlen, info, NULL);
    }

    SDL_ulltoa((unsigned long long)(uintptr_t)value, num, 16);
    length = SDL_PrintString(text, maxlen, info, "0x");
    return length + SDL_PrintString(TEXT_AND_LEN_ARGS, info, num);
}

// NOLINTNEXTLINE(readability-non-const-parameter)
int SDL_vsnprintf(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, va_list ap)
{
    size_t length = 0;

    if (!text) {
        maxlen = 0;
    }
    if (!fmt) {
        fmt = "";
    }
    while (*fmt) {
        if (*fmt == '%') {
            bool done = false;
            bool check_flag;
            SDL_FormatInfo info;
            enum
            {
                DO_INT,
                DO_LONG,
                DO_LONGLONG,
                DO_SIZE_T
            } inttype = DO_INT;

            SDL_zero(info);
            info.radix = 10;
            info.precision = -1;

            check_flag = true;
            while (check_flag) {
                ++fmt;
                switch (*fmt) {
                case '-':
                    info.left_justify = true;
                    break;
                case '+':
                    info.force_sign = true;
                    break;
                case '#':
                    info.force_type = true;
                    break;
                case '0':
                    info.pad_zeroes = true;
                    break;
                default:
                    check_flag = false;
                    break;
                }
            }

            if (*fmt >= '0' && *fmt <= '9') {
                info.width = SDL_strtol(fmt, (char **)&fmt, 0);
            } else if (*fmt == '*') {
                ++fmt;
                info.width = va_arg(ap, int);
            }

            if (*fmt == '.') {
                ++fmt;
                if (*fmt >= '0' && *fmt <= '9') {
                    info.precision = SDL_strtol(fmt, (char **)&fmt, 0);
                } else if (*fmt == '*') {
                    ++fmt;
                    info.precision = va_arg(ap, int);
                } else {
                    info.precision = 0;
                }
                if (info.precision < 0) {
                    info.precision = 0;
                }
            }

            while (!done) {
                switch (*fmt) {
                case '%':
                    if (length < maxlen) {
                        text[length] = '%';
                    }
                    ++length;
                    done = true;
                    break;
                case 'c':
                    // char is promoted to int when passed through (...)
                    if (length < maxlen) {
                        text[length] = (char)va_arg(ap, int);
                    }
                    ++length;
                    done = true;
                    break;
                case 'h':
                    // short is promoted to int when passed through (...)
                    break;
                case 'l':
                    if (inttype < DO_LONGLONG) {
                        ++inttype;
                    }
                    break;
                case 'I':
                    if (SDL_strncmp(fmt, "I64", 3) == 0) {
                        fmt += 2;
                        inttype = DO_LONGLONG;
                    }
                    break;
                case 'z':
                    inttype = DO_SIZE_T;
                    break;
                case 'i':
                case 'd':
                    if (info.precision >= 0) {
                        info.pad_zeroes = false;
                    }
                    switch (inttype) {
                    case DO_INT:
                        length += SDL_PrintLong(TEXT_AND_LEN_ARGS, &info,
                                                (long)va_arg(ap, int));
                        break;
                    case DO_LONG:
                        length += SDL_PrintLong(TEXT_AND_LEN_ARGS, &info,
                                                va_arg(ap, long));
                        break;
                    case DO_LONGLONG:
                        length += SDL_PrintLongLong(TEXT_AND_LEN_ARGS, &info,
                                                    va_arg(ap, long long));
                        break;
                    case DO_SIZE_T:
                        length += SDL_PrintLongLong(TEXT_AND_LEN_ARGS, &info,
                                                    va_arg(ap, size_t));
                        break;
                    }
                    done = true;
                    break;
                case 'p':
                    info.force_case = SDL_CASE_LOWER;
                    length += SDL_PrintPointer(TEXT_AND_LEN_ARGS, &info, va_arg(ap, void *));
                    done = true;
                    break;
                case 'x':
                    info.force_case = SDL_CASE_LOWER;
                    SDL_FALLTHROUGH;
                case 'X':
                    if (info.force_case == SDL_CASE_NOCHANGE) {
                        info.force_case = SDL_CASE_UPPER;
                    }
                    if (info.radix == 10) {
                        info.radix = 16;
                    }
                    SDL_FALLTHROUGH;
                case 'o':
                    if (info.radix == 10) {
                        info.radix = 8;
                    }
                    SDL_FALLTHROUGH;
                case 'u':
                    info.force_sign = false;
                    if (info.precision >= 0) {
                        info.pad_zeroes = false;
                    }
                    switch (inttype) {
                    case DO_INT:
                        length += SDL_PrintUnsignedLong(TEXT_AND_LEN_ARGS, &info,
                                                        va_arg(ap, unsigned int));
                        break;
                    case DO_LONG:
                        length += SDL_PrintUnsignedLong(TEXT_AND_LEN_ARGS, &info,
                                                        va_arg(ap, unsigned long));
                        break;
                    case DO_LONGLONG:
                        length += SDL_PrintUnsignedLongLong(TEXT_AND_LEN_ARGS, &info,
                                                            va_arg(ap, unsigned long long));
                        break;
                    case DO_SIZE_T:
                        length += SDL_PrintUnsignedLongLong(TEXT_AND_LEN_ARGS, &info,
                                                            va_arg(ap, size_t));
                        break;
                    }
                    done = true;
                    break;
                case 'f':
                    length += SDL_PrintFloat(TEXT_AND_LEN_ARGS, &info, va_arg(ap, double), false);
                    done = true;
                    break;
                case 'g':
                    length += SDL_PrintFloat(TEXT_AND_LEN_ARGS, &info, va_arg(ap, double), true);
                    done = true;
                    break;
                case 'S':
                    info.pad_zeroes = false;
                    length += SDL_PrintStringW(TEXT_AND_LEN_ARGS, &info, va_arg(ap, wchar_t *));
                    done = true;
                    break;
                case 's':
                    info.pad_zeroes = false;
                    if (inttype > DO_INT) {
                        length += SDL_PrintStringW(TEXT_AND_LEN_ARGS, &info, va_arg(ap, wchar_t *));
                    } else {
                        length += SDL_PrintString(TEXT_AND_LEN_ARGS, &info, va_arg(ap, char *));
                    }
                    done = true;
                    break;
                default:
                    done = true;
                    break;
                }
                ++fmt;
            }
        } else {
            if (length < maxlen) {
                text[length] = *fmt;
            }
            ++fmt;
            ++length;
        }
    }
    if (length < maxlen) {
        text[length] = '\0';
    } else if (maxlen > 0) {
        text[maxlen - 1] = '\0';
    }
    return (int)length;
}

#undef TEXT_AND_LEN_ARGS
#endif // HAVE_VSNPRINTF

int SDL_vswprintf(SDL_OUT_Z_CAP(maxlen) wchar_t *text, size_t maxlen, const wchar_t *fmt, va_list ap)
{
    char *fmt_utf8 = NULL;
    if (fmt) {
        fmt_utf8 = SDL_iconv_string("UTF-8", "WCHAR_T", (const char *)fmt, (SDL_wcslen(fmt) + 1) * sizeof(wchar_t));
        if (!fmt_utf8) {
            return -1;
        }
    }

    char tinybuf[64];  // for really small strings, calculate it once.

    // generate the text to find the final text length
    va_list aq;
    va_copy(aq, ap);
    const int utf8len = SDL_vsnprintf(tinybuf, sizeof (tinybuf), fmt_utf8, aq);
    va_end(aq);

    if (utf8len < 0) {
        SDL_free(fmt_utf8);
        return -1;
    }

    bool isstack = false;
    char *smallbuf = NULL;
    char *utf8buf;
    int result;

    if (utf8len < sizeof (tinybuf)) {   // whole thing fit in the stack buffer, just use that copy.
        utf8buf = tinybuf;
    } else {  // didn't fit in the stack buffer, allocate the needed space and run it again.
        utf8buf = smallbuf = SDL_small_alloc(char, utf8len + 1, &isstack);
        if (!smallbuf) {
            SDL_free(fmt_utf8);
            return -1;  // oh well.
        }
        const int utf8len2 = SDL_vsnprintf(smallbuf, utf8len + 1, fmt_utf8, ap);
        if (utf8len2 > utf8len) {
            SDL_free(fmt_utf8);
            return SDL_SetError("Formatted output changed between two runs");  // race condition on the parameters, and we no longer have room...yikes.
        }
    }

    SDL_free(fmt_utf8);

    wchar_t *wbuf = (wchar_t *)SDL_iconv_string("WCHAR_T", "UTF-8", utf8buf, utf8len + 1);
    if (wbuf) {
        if (text) {
            SDL_wcslcpy(text, wbuf, maxlen);
        }
        result = (int)SDL_wcslen(wbuf);
        SDL_free(wbuf);
    } else {
        result = -1;
    }

    if (smallbuf != NULL) {
        SDL_small_free(smallbuf, isstack);
    }

    return result;
}

int SDL_asprintf(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    int result;

    va_start(ap, fmt);
    result = SDL_vasprintf(strp, fmt, ap);
    va_end(ap);

    return result;
}

int SDL_vasprintf(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, va_list ap)
{
    int result;
    int size = 100; // Guess we need no more than 100 bytes
    char *p, *np;
    va_list aq;

    *strp = NULL;

    p = (char *)SDL_malloc(size);
    if (!p) {
        return -1;
    }

    while (1) {
        // Try to print in the allocated space
        va_copy(aq, ap);
        result = SDL_vsnprintf(p, size, fmt, aq);
        va_end(aq);

        // Check error code
        if (result < 0) {
            SDL_free(p);
            return result;
        }

        // If that worked, return the string
        if (result < size) {
            *strp = p;
            return result;
        }

        // Else try again with more space
        size = result + 1; // Precisely what is needed

        np = (char *)SDL_realloc(p, size);
        if (!np) {
            SDL_free(p);
            return -1;
        } else {
            p = np;
        }
    }
}

char * SDL_strpbrk(const char *str, const char *breakset)
{
#ifdef HAVE_STRPBRK
    return strpbrk(str, breakset);
#else

    for (; *str; str++) {
        const char *b;

        for (b = breakset; *b; b++) {
            if (*str == *b) {
                return (char *) str;
            }
        }
    }
    return NULL;
#endif
}
