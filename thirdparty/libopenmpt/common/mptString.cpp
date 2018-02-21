/*
 * mptString.cpp
 * -------------
 * Purpose: Small string-related utilities, number and message formatting.
 * Notes  : Currently none.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "stdafx.h"
#include "mptString.h"

#include "Endianness.h"

#if defined(MPT_CHARSET_CODECVTUTF8)
#include <codecvt>
#endif
#if defined(MPT_CHARSET_INTERNAL) || defined(MPT_CHARSET_WIN32)
#include <cstdlib>
#endif
#include <locale>
#include <string>
#include <stdexcept>
#include <vector>

#if defined(MODPLUG_TRACKER)
#include <cwctype>
#endif // MODPLUG_TRACKER

#if defined(MODPLUG_TRACKER)
#include <wctype.h>
#endif // MODPLUG_TRACKER

#if MPT_OS_WINDOWS
#include <windows.h>
#endif

#if defined(MPT_CHARSET_ICONV)
#include <errno.h>
#include <iconv.h>
#endif


OPENMPT_NAMESPACE_BEGIN



/*



Quick guide to the OpenMPT string type jungle
=============================================



This quick guide is only meant as a hint. There may be valid reasons to not
honor the recommendations found here. Staying consistent with surrounding and/or
related code sections may also be important.



List of string types
--------------------

 *  std::string (OpenMPT, libopenmpt)
    C++ string of unspecifed 8bit encoding. Try to always document the
    encoding if not clear from context. Do not use unless there is an obvious
    reason to do so.

 *  std::wstring (OpenMPT)
    UTF16 (on windows) or UTF32 (otherwise). Do not use unless there is an
    obvious reason to do so.

 *  char* (OpenMPT, libopenmpt)
    C string of unspecified encoding. Use only for static literals or in
    performance critical inner loops where full control and avoidance of memory
    allocations is required.

 *  wchar_t* (OpenMPT)
    C wide string. Use only if Unicode is required for static literals or in
    performance critical inner loops where full control and avoidance of memory
    allocation is required.

 *  CString (OpenMPT)
    MFC string type, either encoded in locale/CP_ACP (if !UNICODE) or UTF16 (if
    UNICODE). Specify literals with _T(""). Use in MFC GUI code.

 *  CStringA (OpenMPT)
    MFC ANSI string type. The encoding is always CP_ACP. Do not use unless there
    is an obvious reason to do so.

 *  CStringW (OpenMPT)
    MFC Unicode string type. Use in MFC GUI code when explicit Unicode support
    is required.

 *  mpt::PathString (OpenMPT, libopenmpt)
    String type representing paths and filenames. Always use for these in order
    to avoid potentially lossy conversions. Use MPT_PATHSTRING("") macro for
    literals.

 *  mpt::ustring (OpenMPT, libopenmpt)
    The default unicode string type. Can be encoded in UTF8 or UTF16 or UTF32,
    depending on MPT_USTRING_MODE_* and sizeof(wchar_t). Literals can written as
    MPT_USTRING(""). Use as your default string type if no other string type is
    a measurably better fit.

 *  MPT_UTF8 (OpenMPT, libopenmpt)
    Macro that generates a mpt::ustring from string literals containing
    non-ascii characters. In order to keep the source code in ascii encoding,
    always express non-ascii characters using explicit \x23 escaping. Note that
    depending on the underlying type of mpt::ustring, MPT_UTF8 *requires* a
    runtime conversion. Only use for string literals containing non-ascii
    characters (use MPT_USTRING otherwise).

 *  MPT_ULITERAL / MPT_UCHAR / MPT_UCHAR_TYPE (OpenMPT, libopenmpt)
    Macros which generate string literals, char literals and the char literal
    type respectively. These are especially useful in constexpr contexts or
    global data where MPT_USTRING is either unusable or requires a global
    contructor to run. Do NOT use as a performance optimization in place of
    MPT_USTRING however, because MPT_USTRING can be converted to C++11/14 user
    defined literals eventually, while MPT_ULITERAL cannot because of constexpr
    requirements.

 *  mpt::RawPathString (OpenMPT, libopenmpt)
    Internal representation of mpt::PathString. Only use for parsing path
    fragments.

 *  mpt::u8string (OpenMPT, libopenmpt)
    Internal representation of mpt::ustring. Do not use directly. Ever.

 *  std::basic_string<char> (OpenMPT)
    Same as std::string. Do not use std::basic_string in the templated form.

 *  std::basic_string<wchar_t> (OpenMPT)
    Same as std::wstring. Do not use std::basic_string in the templated form.

The following string types are available in order to avoid the need to overload
functions on a huge variety of string types. Use only ever as function argument
types.
Note that the locale charset is not available on all libopenmpt builds (in which
case the option is ignored or a sensible fallback is used; these types are
always available).
All these types publicly inherit from mpt::ustring and do not contain any
additional state. This means that they work the same way as mpt::ustring does
and do support type-slicing for both, read and write accesses.
These types only add conversion constructors for all string types that have a
defined encoding and for all 8bit string types using the specified encoding
heuristic.

 *  AnyUnicodeString (OpenMPT, libopenmpt)
    Is constructible from any Unicode string.

 *  AnyString (OpenMPT, libopenmpt)
    Tries to do the smartest auto-magic we can do.

 *  AnyLocaleString (OpenMPT, libopenmpt)
    char-based strings are assumed to be in locale encoding.

 *  AnyStringUTF8orLocale (OpenMPT, libopenmpt)
    char-based strings are tried in UTF8 first, if this fails, locale is used.

 *  AnyStringUTF8 (OpenMPT, libopenmpt)
    char-based strings are assumed to be in UTF8.



Encoding of 8bit strings
------------------------

8bit strings have an unspecified encoding. When the string is contained within a
CSoundFile object, the encoding is most likely CSoundFile::GetCharsetInternal(),
otherwise, try to gather the most probable encoding from surrounding or related
code sections.



Decision tree to help deciding which string type to use
-------------------------------------------------------

if in libopenmpt
 if in libopenmpt c++ interface
  T = std::string, the encoding is utf8
 elif in libopenmpt c interface
  T = char*, the encoding is utf8
 elif performance critical inner loop
  T = char*, document the encoding if not clear from context 
 elif string literal containing non-ascii characters
  T = MPT_UTF8
 elif path or file
  if parsing path fragments
   T = mpt::RawPathString
       template your function on the concrete underlying string type
       (std::string and std::wstring) or use preprocessor MPT_OS_WINDOWS
  else
   T = mpt::PathString
  fi
 else
  T = mpt::ustring
 fi
else
 if performance critical inner loop
  if needs unicode support
   T = MPT_UCHAR_TYPE* / MPT_ULITERAL
  else
   T = char*, document the encoding if not clear from context 
  fi
 elif string literal containing non-ascii characters
  T = MPT_UTF8
 elif path or file
  if parsing path fragments
   T = mpt::RawPathString
       template your function on the concrete underlying string type
       (std::string and std::wstring) or use preprocessor MPT_OS_WINDOWS
  else
   T = mpt::PathString
  fi
 elif mfc/gui code
  if directly interface with wide winapi
   T = CStringW
  elif needs unicode support
   T = CStringW
  else
   T = CString
  fi
 else
  if directly interfacing with wide winapi
   T = std::wstring
  else
   if constexpr context or global data
    T = MPT_UCHAR_TYPE* / MPT_ULITERAL
   else
    T = mpt::ustring
	 fi
  fi
 fi
fi

This boils down to: Prefer mpt::PathString and mpt::ustring, and only use any
other string type if there is an obvious reason to do so.



Character set conversions
-------------------------

Character set conversions in OpenMPT are always fuzzy.

Behaviour in case of an invalid source encoding and behaviour in case of an
unrepresentable destination encoding can be any of the following:
 *  The character is replaced by some replacement character ('?' or L'\ufffd' in
    most cases).
 *  The character is replaced by a similar character (either semantically
    similiar or visually similar).
 *  The character is transcribed with some ASCII text.
 *  The character is discarded.
 *  Conversion stops at this very character.

Additionally. conversion may stop or continue on \0 characters in the middle of
the string.

Behaviour can vary from one conversion tuple to any other.

If you need to ensure lossless conversion, do a roundtrip conversion and check
for equality.



Unicode handling
----------------

OpenMPT is generally not aware of and does not handle different Unicode
normalization forms.
You should be aware of the following possibilities:
 *  Conversion between UTF8, UTF16, UTF32 may or may not change between NFC and
    NFD.
 *  Conversion from any non-Unicode 8bit encoding can result in both, NFC or NFD
    forms.
 *  Conversion to any non-Unicode 8bit encoding may or may not involve
    conversion to NFC, NFD, NFKC or NFKD during the conversion. This in
    particular means that conversion of decomposed german umlauts to ISO8859-1
    may fail.
 *  Changing the normalization form of path strings may render the file
    inaccessible.

Unicode BOM may or may not be preserved and/or discarded during conversion.

Invalid Unicode code points may be treated as invalid or as valid characters
when converting between different Unicode encodings.



Interfacing with WinAPI
-----------------------

When in MFC code, use CString or CStringW as appropriate.
When in non MFC code, either use std::wstring when directly interfacing with the
Unicode API, or use the TCHAR helper functions: ToTcharBuf, FromTcharBuf,
ToTcharStr, FromTcharStr.



*/



namespace mpt { namespace String {



/*
default 1:1 mapping
static const uint32 CharsetTableISO8859_1[256] = {
	0x0000,0x0001,0x0002,0x0003,0x0004,0x0005,0x0006,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x000e,0x000f,
	0x0010,0x0011,0x0012,0x0013,0x0014,0x0015,0x0016,0x0017,0x0018,0x0019,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f,
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x007f,
	0x0080,0x0081,0x0082,0x0083,0x0084,0x0085,0x0086,0x0087,0x0088,0x0089,0x008a,0x008b,0x008c,0x008d,0x008e,0x008f,
	0x0090,0x0091,0x0092,0x0093,0x0094,0x0095,0x0096,0x0097,0x0098,0x0099,0x009a,0x009b,0x009c,0x009d,0x009e,0x009f,
	0x00a0,0x00a1,0x00a2,0x00a3,0x00a4,0x00a5,0x00a6,0x00a7,0x00a8,0x00a9,0x00aa,0x00ab,0x00ac,0x00ad,0x00ae,0x00af,
	0x00b0,0x00b1,0x00b2,0x00b3,0x00b4,0x00b5,0x00b6,0x00b7,0x00b8,0x00b9,0x00ba,0x00bb,0x00bc,0x00bd,0x00be,0x00bf,
	0x00c0,0x00c1,0x00c2,0x00c3,0x00c4,0x00c5,0x00c6,0x00c7,0x00c8,0x00c9,0x00ca,0x00cb,0x00cc,0x00cd,0x00ce,0x00cf,
	0x00d0,0x00d1,0x00d2,0x00d3,0x00d4,0x00d5,0x00d6,0x00d7,0x00d8,0x00d9,0x00da,0x00db,0x00dc,0x00dd,0x00de,0x00df,
	0x00e0,0x00e1,0x00e2,0x00e3,0x00e4,0x00e5,0x00e6,0x00e7,0x00e8,0x00e9,0x00ea,0x00eb,0x00ec,0x00ed,0x00ee,0x00ef,
	0x00f0,0x00f1,0x00f2,0x00f3,0x00f4,0x00f5,0x00f6,0x00f7,0x00f8,0x00f9,0x00fa,0x00fb,0x00fc,0x00fd,0x00fe,0x00ff
};
*/

#if defined(MPT_CHARSET_CODECVTUTF8) || defined(MPT_CHARSET_INTERNAL) || defined(MPT_CHARSET_WIN32)

static const uint32 CharsetTableISO8859_15[256] = {
	0x0000,0x0001,0x0002,0x0003,0x0004,0x0005,0x0006,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x000e,0x000f,
	0x0010,0x0011,0x0012,0x0013,0x0014,0x0015,0x0016,0x0017,0x0018,0x0019,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f,
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x007f,
	0x0080,0x0081,0x0082,0x0083,0x0084,0x0085,0x0086,0x0087,0x0088,0x0089,0x008a,0x008b,0x008c,0x008d,0x008e,0x008f,
	0x0090,0x0091,0x0092,0x0093,0x0094,0x0095,0x0096,0x0097,0x0098,0x0099,0x009a,0x009b,0x009c,0x009d,0x009e,0x009f,
	0x00a0,0x00a1,0x00a2,0x00a3,0x20ac,0x00a5,0x0160,0x00a7,0x0161,0x00a9,0x00aa,0x00ab,0x00ac,0x00ad,0x00ae,0x00af,
	0x00b0,0x00b1,0x00b2,0x00b3,0x017d,0x00b5,0x00b6,0x00b7,0x017e,0x00b9,0x00ba,0x00bb,0x0152,0x0153,0x0178,0x00bf,
	0x00c0,0x00c1,0x00c2,0x00c3,0x00c4,0x00c5,0x00c6,0x00c7,0x00c8,0x00c9,0x00ca,0x00cb,0x00cc,0x00cd,0x00ce,0x00cf,
	0x00d0,0x00d1,0x00d2,0x00d3,0x00d4,0x00d5,0x00d6,0x00d7,0x00d8,0x00d9,0x00da,0x00db,0x00dc,0x00dd,0x00de,0x00df,
	0x00e0,0x00e1,0x00e2,0x00e3,0x00e4,0x00e5,0x00e6,0x00e7,0x00e8,0x00e9,0x00ea,0x00eb,0x00ec,0x00ed,0x00ee,0x00ef,
	0x00f0,0x00f1,0x00f2,0x00f3,0x00f4,0x00f5,0x00f6,0x00f7,0x00f8,0x00f9,0x00fa,0x00fb,0x00fc,0x00fd,0x00fe,0x00ff
};

static const uint32 CharsetTableWindows1252[256] = {
	0x0000,0x0001,0x0002,0x0003,0x0004,0x0005,0x0006,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x000e,0x000f,
	0x0010,0x0011,0x0012,0x0013,0x0014,0x0015,0x0016,0x0017,0x0018,0x0019,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f,
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x007f,
	0x20ac,0x0081,0x201a,0x0192,0x201e,0x2026,0x2020,0x2021,0x02c6,0x2030,0x0160,0x2039,0x0152,0x008d,0x017d,0x008f,
	0x0090,0x2018,0x2019,0x201c,0x201d,0x2022,0x2013,0x2014,0x02dc,0x2122,0x0161,0x203a,0x0153,0x009d,0x017e,0x0178,
	0x00a0,0x00a1,0x00a2,0x00a3,0x00a4,0x00a5,0x00a6,0x00a7,0x00a8,0x00a9,0x00aa,0x00ab,0x00ac,0x00ad,0x00ae,0x00af,
	0x00b0,0x00b1,0x00b2,0x00b3,0x00b4,0x00b5,0x00b6,0x00b7,0x00b8,0x00b9,0x00ba,0x00bb,0x00bc,0x00bd,0x00be,0x00bf,
	0x00c0,0x00c1,0x00c2,0x00c3,0x00c4,0x00c5,0x00c6,0x00c7,0x00c8,0x00c9,0x00ca,0x00cb,0x00cc,0x00cd,0x00ce,0x00cf,
	0x00d0,0x00d1,0x00d2,0x00d3,0x00d4,0x00d5,0x00d6,0x00d7,0x00d8,0x00d9,0x00da,0x00db,0x00dc,0x00dd,0x00de,0x00df,
	0x00e0,0x00e1,0x00e2,0x00e3,0x00e4,0x00e5,0x00e6,0x00e7,0x00e8,0x00e9,0x00ea,0x00eb,0x00ec,0x00ed,0x00ee,0x00ef,
	0x00f0,0x00f1,0x00f2,0x00f3,0x00f4,0x00f5,0x00f6,0x00f7,0x00f8,0x00f9,0x00fa,0x00fb,0x00fc,0x00fd,0x00fe,0x00ff
};

static const uint32 CharsetTableCP437[256] = {
	0x0000,0x0001,0x0002,0x0003,0x0004,0x0005,0x0006,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x000e,0x000f,
	0x0010,0x0011,0x0012,0x0013,0x0014,0x0015,0x0016,0x0017,0x0018,0x0019,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f,
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x2302,
	0x00c7,0x00fc,0x00e9,0x00e2,0x00e4,0x00e0,0x00e5,0x00e7,0x00ea,0x00eb,0x00e8,0x00ef,0x00ee,0x00ec,0x00c4,0x00c5,
	0x00c9,0x00e6,0x00c6,0x00f4,0x00f6,0x00f2,0x00fb,0x00f9,0x00ff,0x00d6,0x00dc,0x00a2,0x00a3,0x00a5,0x20a7,0x0192,
	0x00e1,0x00ed,0x00f3,0x00fa,0x00f1,0x00d1,0x00aa,0x00ba,0x00bf,0x2310,0x00ac,0x00bd,0x00bc,0x00a1,0x00ab,0x00bb,
	0x2591,0x2592,0x2593,0x2502,0x2524,0x2561,0x2562,0x2556,0x2555,0x2563,0x2551,0x2557,0x255d,0x255c,0x255b,0x2510,
	0x2514,0x2534,0x252c,0x251c,0x2500,0x253c,0x255e,0x255f,0x255a,0x2554,0x2569,0x2566,0x2560,0x2550,0x256c,0x2567,
	0x2568,0x2564,0x2565,0x2559,0x2558,0x2552,0x2553,0x256b,0x256a,0x2518,0x250c,0x2588,0x2584,0x258c,0x2590,0x2580,
	0x03b1,0x00df,0x0393,0x03c0,0x03a3,0x03c3,0x00b5,0x03c4,0x03a6,0x0398,0x03a9,0x03b4,0x221e,0x03c6,0x03b5,0x2229,
	0x2261,0x00b1,0x2265,0x2264,0x2320,0x2321,0x00f7,0x2248,0x00b0,0x2219,0x00b7,0x221a,0x207f,0x00b2,0x25a0,0x00a0
};

#endif // MPT_CHARSET_CODECVTUTF8 || MPT_CHARSET_INTERNAL || MPT_CHARSET_WIN32


#define C(x) (static_cast<uint8>((x)))

// AMS1 actually only supports ASCII plus the modified control characters and no high chars at all.
// Just default to CP437 for those to keep things simple.
static const uint32 CharsetTableCP437AMS[256] = {
	C(' '),0x0001,0x0002,0x0003,0x00e4,0x0005,0x00e5,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x00c4,0x00c5, // differs from CP437
	0x0010,0x0011,0x0012,0x0013,0x00f6,0x0015,0x0016,0x0017,0x0018,0x00d6,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f, // differs from CP437
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x2302,
	0x00c7,0x00fc,0x00e9,0x00e2,0x00e4,0x00e0,0x00e5,0x00e7,0x00ea,0x00eb,0x00e8,0x00ef,0x00ee,0x00ec,0x00c4,0x00c5,
	0x00c9,0x00e6,0x00c6,0x00f4,0x00f6,0x00f2,0x00fb,0x00f9,0x00ff,0x00d6,0x00dc,0x00a2,0x00a3,0x00a5,0x20a7,0x0192,
	0x00e1,0x00ed,0x00f3,0x00fa,0x00f1,0x00d1,0x00aa,0x00ba,0x00bf,0x2310,0x00ac,0x00bd,0x00bc,0x00a1,0x00ab,0x00bb,
	0x2591,0x2592,0x2593,0x2502,0x2524,0x2561,0x2562,0x2556,0x2555,0x2563,0x2551,0x2557,0x255d,0x255c,0x255b,0x2510,
	0x2514,0x2534,0x252c,0x251c,0x2500,0x253c,0x255e,0x255f,0x255a,0x2554,0x2569,0x2566,0x2560,0x2550,0x256c,0x2567,
	0x2568,0x2564,0x2565,0x2559,0x2558,0x2552,0x2553,0x256b,0x256a,0x2518,0x250c,0x2588,0x2584,0x258c,0x2590,0x2580,
	0x03b1,0x00df,0x0393,0x03c0,0x03a3,0x03c3,0x00b5,0x03c4,0x03a6,0x0398,0x03a9,0x03b4,0x221e,0x03c6,0x03b5,0x2229,
	0x2261,0x00b1,0x2265,0x2264,0x2320,0x2321,0x00f7,0x2248,0x00b0,0x2219,0x00b7,0x221a,0x207f,0x00b2,0x25a0,0x00a0
};

// AMS2: Looking at Velvet Studio's bitmap font (TPIC32.PCX), these appear to be the only supported non-ASCII chars.
static const uint32 CharsetTableCP437AMS2[256] = {
	C(' '),0x00a9,0x221a,0x00b7,C('0'),C('1'),C('2'),C('3'),C('4'),C('5'),C('6'),C('7'),C('8'),C('9'),C('A'),C('B'), // differs from CP437
	C('C'),C('D'),C('E'),C('F'),C(' '),0x00a7,C(' '),C(' '),C(' '),C(' '),C(' '),C(' '),C(' '),C(' '),C(' '),C(' '), // differs from CP437
	0x0020,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
	0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
	0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
	0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
	0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
	0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0x2302,
	0x00c7,0x00fc,0x00e9,0x00e2,0x00e4,0x00e0,0x00e5,0x00e7,0x00ea,0x00eb,0x00e8,0x00ef,0x00ee,0x00ec,0x00c4,0x00c5,
	0x00c9,0x00e6,0x00c6,0x00f4,0x00f6,0x00f2,0x00fb,0x00f9,0x00ff,0x00d6,0x00dc,0x00a2,0x00a3,0x00a5,0x20a7,0x0192,
	0x00e1,0x00ed,0x00f3,0x00fa,0x00f1,0x00d1,0x00aa,0x00ba,0x00bf,0x2310,0x00ac,0x00bd,0x00bc,0x00a1,0x00ab,0x00bb,
	0x2591,0x2592,0x2593,0x2502,0x2524,0x2561,0x2562,0x2556,0x2555,0x2563,0x2551,0x2557,0x255d,0x255c,0x255b,0x2510,
	0x2514,0x2534,0x252c,0x251c,0x2500,0x253c,0x255e,0x255f,0x255a,0x2554,0x2569,0x2566,0x2560,0x2550,0x256c,0x2567,
	0x2568,0x2564,0x2565,0x2559,0x2558,0x2552,0x2553,0x256b,0x256a,0x2518,0x250c,0x2588,0x2584,0x258c,0x2590,0x2580,
	0x03b1,0x00df,0x0393,0x03c0,0x03a3,0x03c3,0x00b5,0x03c4,0x03a6,0x0398,0x03a9,0x03b4,0x221e,0x03c6,0x03b5,0x2229,
	0x2261,0x00b1,0x2265,0x2264,0x2320,0x2321,0x00f7,0x2248,0x00b0,0x2219,0x00b7,0x221a,0x207f,0x00b2,0x25a0,0x00a0
};

#undef C

#if MPT_COMPILER_MSVC
#pragma warning(disable:4428) // universal-character-name encountered in source
#endif

static std::wstring From8bit(const std::string &str, const uint32 (&table)[256], wchar_t replacement = L'\uFFFD')
{
	std::wstring res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint32 c = static_cast<uint32>(static_cast<uint8>(str[i]));
		if(c < mpt::size(table))
		{
			res.push_back(static_cast<wchar_t>(static_cast<uint32>(table[c])));
		} else
		{
			res.push_back(replacement);
		}
	}
	return res;
}

static std::string To8bit(const std::wstring &str, const uint32 (&table)[256], char replacement = '?')
{
	std::string res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint32 c = str[i];
		bool found = false;
		// Try non-control characters first.
		// In cases where there are actual characters mirrored in this range (like in AMS/AMS2 character sets),
		// characters in the common range are preferred this way.
		for(std::size_t x = 0x20; x < mpt::size(table); ++x)
		{
			if(c == table[x])
			{
				res.push_back(static_cast<char>(static_cast<uint8>(x)));
				found = true;
				break;
			}
		}
		if(!found)
		{
			// try control characters
			for(std::size_t x = 0x00; x < mpt::size(table) && x < 0x20; ++x)
			{
				if(c == table[x])
				{
					res.push_back(static_cast<char>(static_cast<uint8>(x)));
					found = true;
					break;
				}
			}
		}
		if(!found)
		{
			res.push_back(replacement);
		}
	}
	return res;
}

#if defined(MPT_CHARSET_CODECVTUTF8) || defined(MPT_CHARSET_INTERNAL) || defined(MPT_CHARSET_WIN32)

static std::wstring FromAscii(const std::string &str, wchar_t replacement = L'\uFFFD')
{
	std::wstring res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint8 c = str[i];
		if(c <= 0x7f)
		{
			res.push_back(static_cast<wchar_t>(static_cast<uint32>(c)));
		} else
		{
			res.push_back(replacement);
		}
	}
	return res;
}

static std::string ToAscii(const std::wstring &str, char replacement = '?')
{
	std::string res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint32 c = str[i];
		if(c <= 0x7f)
		{
			res.push_back(static_cast<char>(static_cast<uint8>(c)));
		} else
		{
			res.push_back(replacement);
		}
	}
	return res;
}

static std::wstring FromISO_8859_1(const std::string &str, wchar_t replacement = L'\uFFFD')
{
	MPT_UNREFERENCED_PARAMETER(replacement);
	std::wstring res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint8 c = str[i];
		res.push_back(static_cast<wchar_t>(static_cast<uint32>(c)));
	}
	return res;
}

static std::string ToISO_8859_1(const std::wstring &str, char replacement = '?')
{
	std::string res;
	res.reserve(str.length());
	for(std::size_t i = 0; i < str.length(); ++i)
	{
		uint32 c = str[i];
		if(c <= 0xff)
		{
			res.push_back(static_cast<char>(static_cast<uint8>(c)));
		} else
		{
			res.push_back(replacement);
		}
	}
	return res;
}

#if defined(MPT_ENABLE_CHARSET_LOCALE)

// Note:
//
//  std::codecvt::out in LLVM libc++ does not advance in and out pointers when
// running into a non-convertible cahracter. This can happen when no locale is
// set on FreeBSD or MacOSX. This behaviour violates the C++ standard.
//
//  We apply the following (albeit costly, even on other platforms) work-around:
//  If the conversion errors out and does not advance the pointers at all, we
// retry the conversion with a space character prepended to the string. If it
// still does error our, we retry the whole conversion character by character.
//  This is costly even on other platforms in one single case: The first
// character is an invalid Unicode code point or otherwise not convertible. Any
// following non-convertible characters are not a problem.

static std::wstring LocaleDecode(const std::string &str, const std::locale & locale, wchar_t replacement = L'\uFFFD', int retry = 0, bool * progress = nullptr)
{
	if(str.empty())
	{
		return std::wstring();
	}
	std::vector<wchar_t> out;
	typedef std::codecvt<wchar_t, char, std::mbstate_t> codecvt_type;
	std::mbstate_t state = std::mbstate_t();
	const codecvt_type & facet = std::use_facet<codecvt_type>(locale);
	codecvt_type::result result = codecvt_type::partial;
	const char * in_begin = str.data();
	const char * in_end = in_begin + str.size();
	out.resize((in_end - in_begin) * (facet.max_length() + 1));
	wchar_t * out_begin = &(out[0]);
	wchar_t * out_end = &(out[0]) + out.size();
	const char * in_next = nullptr;
	wchar_t * out_next = nullptr;
	do
	{
		if(retry == 2)
		{
			for(;;)
			{
				in_next = nullptr;
				out_next = nullptr;
				result = facet.in(state, in_begin, in_begin + 1, in_next, out_begin, out_end, out_next);
				if(result == codecvt_type::partial && in_next == in_begin + 1)
				{
					in_begin = in_next;
					out_begin = out_next;
					continue;
				} else
				{
					break;
				}
			}
		} else
		{
			in_next = nullptr;
			out_next = nullptr;
			result = facet.in(state, in_begin, in_end, in_next, out_begin, out_end, out_next);
		}
		if(result == codecvt_type::partial || (result == codecvt_type::error && out_next == out_end))
		{
			out.resize(out.size() * 2);
			in_begin = in_next;
			out_begin = &(out[0]) + (out_next - out_begin);
			out_end = &(out[0]) + out.size();
			continue;
		}
		if(retry == 0)
		{
			if(result == codecvt_type::error && in_next == in_begin && out_next == out_begin)
			{
				bool made_progress = true;
				LocaleDecode(std::string(" ") + str, locale, replacement, 1, &made_progress);
				if(!made_progress)
				{
					return LocaleDecode(str, locale, replacement, 2);
				}
			}
		} else if(retry == 1)
		{
			if(result == codecvt_type::error && in_next == in_begin && out_next == out_begin)
			{
				*progress = false;
			} else
			{
				*progress = true;
			}
			return std::wstring();
		}
		if(result == codecvt_type::error)
		{
			++in_next;
			*out_next = replacement;
			++out_next;
		}
		in_begin = in_next;
		out_begin = out_next;
	} while((result == codecvt_type::error && in_next < in_end && out_next < out_end) || (retry == 2 && in_next < in_end));
	return std::wstring(&(out[0]), out_next);
}

static std::string LocaleEncode(const std::wstring &str, const std::locale & locale, char replacement = '?', int retry = 0, bool * progress = nullptr)
{
	if(str.empty())
	{
		return std::string();
	}
	std::vector<char> out;
	typedef std::codecvt<wchar_t, char, std::mbstate_t> codecvt_type;
	std::mbstate_t state = std::mbstate_t();
	const codecvt_type & facet = std::use_facet<codecvt_type>(locale);
	codecvt_type::result result = codecvt_type::partial;
	const wchar_t * in_begin = str.data();
	const wchar_t * in_end = in_begin + str.size();
	out.resize((in_end - in_begin) * (facet.max_length() + 1));
	char * out_begin = &(out[0]);
	char * out_end = &(out[0]) + out.size();
	const wchar_t * in_next = nullptr;
	char * out_next = nullptr;
	do
	{
		if(retry == 2)
		{
			for(;;)
			{
				in_next = nullptr;
				out_next = nullptr;
				result = facet.out(state, in_begin, in_begin + 1, in_next, out_begin, out_end, out_next);
				if(result == codecvt_type::partial && in_next == in_begin + 1)
				{
					in_begin = in_next;
					out_begin = out_next;
					continue;
				} else
				{
					break;
				}
			}
		} else
		{
			in_next = nullptr;
			out_next = nullptr;
			result = facet.out(state, in_begin, in_end, in_next, out_begin, out_end, out_next);
		}
		if(result == codecvt_type::partial || (result == codecvt_type::error && out_next == out_end))
		{
			out.resize(out.size() * 2);
			in_begin = in_next;
			out_begin = &(out[0]) + (out_next - out_begin);
			out_end = &(out[0]) + out.size();
			continue;
		}
		if(retry == 0)
		{
			if(result == codecvt_type::error && in_next == in_begin && out_next == out_begin)
			{
				bool made_progress = true;
				LocaleEncode(std::wstring(L" ") + str, locale, replacement, 1, &made_progress);
				if(!made_progress)
				{
					return LocaleEncode(str, locale, replacement, 2);
				}
			}
		} else if(retry == 1)
		{
			if(result == codecvt_type::error && in_next == in_begin && out_next == out_begin)
			{
				*progress = false;
			} else
			{
				*progress = true;
			}
			return std::string();
		}
		if(result == codecvt_type::error)
		{
			++in_next;
			*out_next = replacement;
			++out_next;
		}
		in_begin = in_next;
		out_begin = out_next;
	} while((result == codecvt_type::error && in_next < in_end && out_next < out_end) || (retry == 2 && in_next < in_end));
	return std::string(&(out[0]), out_next);
}

static std::wstring FromLocale(const std::string &str, wchar_t replacement = L'\uFFFD')
{
	try
	{
		std::locale locale(""); // user locale
		return String::LocaleDecode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	try
	{
		std::locale locale; // current c++ locale
		return String::LocaleDecode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	try
	{
		std::locale locale = std::locale::classic(); // "C" locale
		return String::LocaleDecode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	MPT_ASSERT_NOTREACHED();
	return String::FromAscii(str, replacement); // fallback
}

static std::string ToLocale(const std::wstring &str, char replacement = '?')
{
	try
	{
		std::locale locale(""); // user locale
		return String::LocaleEncode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	try
	{
		std::locale locale; // current c++ locale
		return String::LocaleEncode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	try
	{
		std::locale locale = std::locale::classic(); // "C" locale
		return String::LocaleEncode(str, locale, replacement);
	} catch(...)
	{
		// nothing
	}
	MPT_ASSERT_NOTREACHED();
	return String::ToAscii(str, replacement); // fallback
}

#endif

#endif // MPT_CHARSET_CODECVTUTF8 || MPT_CHARSET_INTERNAL || MPT_CHARSET_WIN32

#if defined(MPT_CHARSET_CODECVTUTF8)

static std::wstring FromUTF8(const std::string &str, wchar_t replacement = L'\uFFFD')
{
	MPT_UNREFERENCED_PARAMETER(replacement);
	std::wstring_convert<std::codecvt_utf8<wchar_t> > conv;
	return conv.from_bytes(str);
}

static std::string ToUTF8(const std::wstring &str, char replacement = '?')
{
	MPT_UNREFERENCED_PARAMETER(replacement);
	std::wstring_convert<std::codecvt_utf8<wchar_t> > conv;
	return conv.to_bytes(str);
}

#endif // MPT_CHARSET_CODECVTUTF8

#if defined(MPT_CHARSET_INTERNAL) || defined(MPT_CHARSET_WIN32)

static std::wstring FromUTF8(const std::string &str, wchar_t replacement = L'\uFFFD')
{
	const std::string &in = str;

	std::wstring out;

	// state:
	std::size_t charsleft = 0;
	uint32 ucs4 = 0;

	for ( uint8 c : in ) {

		if ( charsleft == 0 ) {

			if ( ( c & 0x80 ) == 0x00 ) {
				out.push_back( (wchar_t)c );
			} else if ( ( c & 0xE0 ) == 0xC0 ) {
				ucs4 = c & 0x1F;
				charsleft = 1;
			} else if ( ( c & 0xF0 ) == 0xE0 ) {
				ucs4 = c & 0x0F;
				charsleft = 2;
			} else if ( ( c & 0xF8 ) == 0xF0 ) {
				ucs4 = c & 0x07;
				charsleft = 3;
			} else {
				out.push_back( replacement );
				ucs4 = 0;
				charsleft = 0;
			}

		} else {

			if ( ( c & 0xC0 ) != 0x80 ) {
				out.push_back( replacement );
				ucs4 = 0;
				charsleft = 0;
			}
			ucs4 <<= 6;
			ucs4 |= c & 0x3F;
			charsleft--;

			if ( charsleft == 0 ) {
				MPT_CONSTANT_IF ( sizeof( wchar_t ) == 2 ) {
					if ( ucs4 > 0x1fffff ) {
						out.push_back( replacement );
						ucs4 = 0;
						charsleft = 0;
					}
					if ( ucs4 <= 0xffff ) {
						out.push_back( (uint16)ucs4 );
					} else {
						uint32 surrogate = ucs4 - 0x10000;
						uint16 hi_sur = static_cast<uint16>( ( 0x36 << 10 ) | ( (surrogate>>10) & ((1<<10)-1) ) );
						uint16 lo_sur = static_cast<uint16>( ( 0x37 << 10 ) | ( (surrogate>> 0) & ((1<<10)-1) ) );
						out.push_back( hi_sur );
						out.push_back( lo_sur );
					}
				} else {
					out.push_back( static_cast<wchar_t>( ucs4 ) );
				}
				ucs4 = 0;
			}

		}

	}

	if ( charsleft != 0 ) {
		out.push_back( replacement );
		ucs4 = 0;
		charsleft = 0;
	}

	return out;

}

static std::string ToUTF8(const std::wstring &str, char replacement = '?')
{
	const std::wstring &in = str;

	std::string out;

	for ( std::size_t i=0; i<in.length(); i++ ) {

		wchar_t wc = in[i];

		uint32 ucs4 = 0;
		MPT_CONSTANT_IF ( sizeof( wchar_t ) == 2 ) {
			uint16 c = static_cast<uint16>( wc );
			if ( i + 1 < in.length() ) {
				// check for surrogate pair
				uint16 hi_sur = in[i+0];
				uint16 lo_sur = in[i+1];
				if ( hi_sur >> 10 == 0x36 && lo_sur >> 10 == 0x37 ) {
					// surrogate pair
					++i;
					hi_sur &= (1<<10)-1;
					lo_sur &= (1<<10)-1;
					ucs4 = ( static_cast<uint32>(hi_sur) << 10 ) | ( static_cast<uint32>(lo_sur) << 0 );
				} else {
					// no surrogate pair
					ucs4 = static_cast<uint32>( c );
				}
			} else {
				// no surrogate possible
				ucs4 = static_cast<uint32>( c );
			}
		} else {
			ucs4 = static_cast<uint32>( wc );
		}
		
		if ( ucs4 > 0x1fffff ) {
			out.push_back( replacement );
			continue;
		}

		uint8 utf8[6];
		std::size_t numchars = 0;
		for ( numchars = 0; numchars < 6; numchars++ ) {
			utf8[numchars] = ucs4 & 0x3F;
			ucs4 >>= 6;
			if ( ucs4 == 0 ) {
				break;
			}
		}
		numchars++;

		if ( numchars == 1 ) {
			out.push_back( utf8[0] );
			continue;
		}

		if ( numchars == 2 && utf8[numchars-1] == 0x01 ) {
			// generate shortest form
			out.push_back( utf8[0] | 0x40 );
			continue;
		}

		std::size_t charsleft = numchars;
		while ( charsleft > 0 ) {
			if ( charsleft == numchars ) {
				out.push_back( utf8[ charsleft - 1 ] | ( ((1<<numchars)-1) << (8-numchars) ) );
			} else {
				out.push_back( utf8[ charsleft - 1 ] | 0x80 );
			}
			charsleft--;
		}

	}

	return out;

}

#endif // MPT_CHARSET_INTERNAL || MPT_CHARSET_WIN32

#if defined(MPT_CHARSET_WIN32)

static bool TestCodePage(UINT cp)
{
	return IsValidCodePage(cp) ? true : false;
}

static bool HasCharset(Charset charset)
{
	bool result = false;
	switch(charset)
	{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
		case CharsetLocale:      result = true; break;
#endif
		case CharsetUTF8:        result = TestCodePage(CP_UTF8); break;
		case CharsetASCII:       result = TestCodePage(20127);   break;
		case CharsetISO8859_1:   result = TestCodePage(28591);   break;
		case CharsetISO8859_15:  result = TestCodePage(28605);   break;
		case CharsetCP437:       result = TestCodePage(437);     break;
		case CharsetWindows1252: result = TestCodePage(1252);    break;
		case CharsetCP437AMS:    result = false; break;
		case CharsetCP437AMS2:   result = false; break;
	}
	return result;
}

#endif // MPT_CHARSET_WIN32

#if defined(MPT_CHARSET_WIN32)

static UINT CharsetToCodepage(Charset charset)
{
	switch(charset)
	{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
		case CharsetLocale:      return CP_ACP;  break;
#endif
		case CharsetUTF8:        return CP_UTF8; break;
		case CharsetASCII:       return 20127;   break;
		case CharsetISO8859_1:   return 28591;   break;
		case CharsetISO8859_15:  return 28605;   break;
		case CharsetCP437:       return 437;     break;
		case CharsetCP437AMS:    return 437;     break; // fallback, should not happen
		case CharsetCP437AMS2:   return 437;     break; // fallback, should not happen
		case CharsetWindows1252: return 1252;    break;
	}
	return 0;
}

#endif // MPT_CHARSET_WIN32

#if defined(MPT_CHARSET_ICONV)

static const char * CharsetToString(Charset charset)
{
	switch(charset)
	{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
		case CharsetLocale:      return "";            break; // "char" breaks with glibc when no locale is set
#endif
		case CharsetUTF8:        return "UTF-8";       break;
		case CharsetASCII:       return "ASCII";       break;
		case CharsetISO8859_1:   return "ISO-8859-1";  break;
		case CharsetISO8859_15:  return "ISO-8859-15"; break;
		case CharsetCP437:       return "CP437";       break;
		case CharsetCP437AMS:    return "CP437";       break; // fallback, should not happen
		case CharsetCP437AMS2:   return "CP437";       break; // fallback, should not happen
		case CharsetWindows1252: return "CP1252";      break;
	}
	return 0;
}

static const char * CharsetToStringTranslit(Charset charset)
{
	switch(charset)
	{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
		case CharsetLocale:      return "//TRANSLIT";            break; // "char" breaks with glibc when no locale is set
#endif
		case CharsetUTF8:        return "UTF-8//TRANSLIT";       break;
		case CharsetASCII:       return "ASCII//TRANSLIT";       break;
		case CharsetISO8859_1:   return "ISO-8859-1//TRANSLIT";  break;
		case CharsetISO8859_15:  return "ISO-8859-15//TRANSLIT"; break;
		case CharsetCP437:       return "CP437//TRANSLIT";       break;
		case CharsetCP437AMS:    return "CP437//TRANSLIT";       break; // fallback, should not happen
		case CharsetCP437AMS2:   return "CP437//TRANSLIT";       break; // fallback, should not happen
		case CharsetWindows1252: return "CP1252//TRANSLIT";      break;
	}
	return 0;
}

static const char * Charset_wchar_t()
{
	#if !defined(MPT_ICONV_NO_WCHAR)
		return "wchar_t";
	#else // MPT_ICONV_NO_WCHAR
		// iconv on OSX does not handle wchar_t if no locale is set
		STATIC_ASSERT(sizeof(wchar_t) == 2 || sizeof(wchar_t) == 4);
		if(sizeof(wchar_t) == 2)
		{
			// "UTF-16" generates BOM
			MPT_MAYBE_CONSTANT_IF(mpt::endian_is_little())
			{
				return "UTF-16LE";
			}
			MPT_MAYBE_CONSTANT_IF(mpt::endian_is_big())
			{
				return "UTF-16BE";
			}
		} else if(sizeof(wchar_t) == 4)
		{
			// "UTF-32" generates BOM
			MPT_MAYBE_CONSTANT_IF(mpt::endian_is_little())
			{
				return "UTF-32LE";
			}
			MPT_MAYBE_CONSTANT_IF(mpt::endian_is_big())
			{
				return "UTF-32BE";
			}
		}
		return "";
	#endif // !MPT_ICONV_NO_WCHAR | MPT_ICONV_NO_WCHAR
}

#endif // MPT_CHARSET_ICONV


#if !defined(MPT_CHARSET_ICONV)
template<typename Tdststring>
Tdststring EncodeImplFallback(Charset charset, const std::wstring &src);
#endif // !MPT_CHARSET_ICONV

// templated on 8bit strings because of type-safe variants
template<typename Tdststring>
Tdststring EncodeImpl(Charset charset, const std::wstring &src)
{
	STATIC_ASSERT(sizeof(typename Tdststring::value_type) == sizeof(char));
	if(charset == CharsetCP437AMS || charset == CharsetCP437AMS2)
	{
		std::string out;
		if(charset == CharsetCP437AMS ) out = String::To8bit(src, CharsetTableCP437AMS );
		if(charset == CharsetCP437AMS2) out = String::To8bit(src, CharsetTableCP437AMS2);
		return Tdststring(out.begin(), out.end());
	}
#if defined(MPT_ENABLE_CHARSET_LOCALE)
	#if defined(MPT_LOCALE_ASSUME_CHARSET)
		if(charset == CharsetLocale)
		{
			charset = MPT_LOCALE_ASSUME_CHARSET;
		}
	#endif
#endif
	#if defined(MPT_CHARSET_WIN32)
		if(!HasCharset(charset))
		{
			return EncodeImplFallback<Tdststring>(charset, src);
		}
		const UINT codepage = CharsetToCodepage(charset);
		int required_size = WideCharToMultiByte(codepage, 0, src.c_str(), -1, nullptr, 0, nullptr, nullptr);
		if(required_size <= 0)
		{
			return Tdststring();
		}
		std::vector<CHAR> encoded_string(required_size);
		WideCharToMultiByte(codepage, 0, src.c_str(), -1, encoded_string.data(), required_size, nullptr, nullptr);
		return reinterpret_cast<const typename Tdststring::value_type*>(encoded_string.data());
	#elif defined(MPT_CHARSET_ICONV)
		iconv_t conv = iconv_t();
		conv = iconv_open(CharsetToStringTranslit(charset), Charset_wchar_t());
		if(!conv)
		{
			conv = iconv_open(CharsetToString(charset), Charset_wchar_t());
			if(!conv)
			{
				throw std::runtime_error("iconv conversion not working");
			}
		}
		std::vector<wchar_t> wide_string(src.c_str(), src.c_str() + src.length() + 1);
		std::vector<char> encoded_string(wide_string.size() * 8); // large enough
		char * inbuf = reinterpret_cast<char*>(wide_string.data());
		size_t inbytesleft = wide_string.size() * sizeof(wchar_t);
		char * outbuf = encoded_string.data();
		size_t outbytesleft = encoded_string.size();
		while(iconv(conv, &inbuf, &inbytesleft, &outbuf, &outbytesleft) == static_cast<size_t>(-1))
		{
			if(errno == EILSEQ || errno == EILSEQ)
			{
				inbuf += sizeof(wchar_t);
				inbytesleft -= sizeof(wchar_t);
				outbuf[0] = '?';
				outbuf++;
				outbytesleft--;
				iconv(conv, NULL, NULL, NULL, NULL); // reset state
			} else
			{
				iconv_close(conv);
				conv = iconv_t();
				return Tdststring();
			}
		}
		iconv_close(conv);
		conv = iconv_t();
		return reinterpret_cast<const typename Tdststring::value_type*>(encoded_string.data());
	#else
		return EncodeImplFallback<Tdststring>(charset, src);
	#endif
}


#if !defined(MPT_CHARSET_ICONV)
template<typename Tdststring>
Tdststring EncodeImplFallback(Charset charset, const std::wstring &src)
{
		std::string out;
		switch(charset)
		{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
			case CharsetLocale:      out = String::ToLocale(src); break;
#endif
			case CharsetUTF8:        out = String::ToUTF8(src); break;
			case CharsetASCII:       out = String::ToAscii(src); break;
			case CharsetISO8859_1:   out = String::ToISO_8859_1(src); break;
			case CharsetISO8859_15:  out = String::To8bit(src, CharsetTableISO8859_15); break;
			case CharsetCP437:       out = String::To8bit(src, CharsetTableCP437); break;
			case CharsetCP437AMS:    out = String::To8bit(src, CharsetTableCP437AMS); break;
			case CharsetCP437AMS2:   out = String::To8bit(src, CharsetTableCP437AMS2); break;
			case CharsetWindows1252: out = String::To8bit(src, CharsetTableWindows1252); break;
		}
		return Tdststring(out.begin(), out.end());
}
#endif // !MPT_CHARSET_ICONV


#if !defined(MPT_CHARSET_ICONV)
template<typename Tsrcstring>
std::wstring DecodeImplFallback(Charset charset, const Tsrcstring &src);
#endif // !MPT_CHARSET_ICONV

// templated on 8bit strings because of type-safe variants
template<typename Tsrcstring>
std::wstring DecodeImpl(Charset charset, const Tsrcstring &src)
{
	STATIC_ASSERT(sizeof(typename Tsrcstring::value_type) == sizeof(char));
	if(charset == CharsetCP437AMS || charset == CharsetCP437AMS2)
	{
		std::string in(src.begin(), src.end());
		std::wstring out;
		if(charset == CharsetCP437AMS ) out = String::From8bit(in, CharsetTableCP437AMS );
		if(charset == CharsetCP437AMS2) out = String::From8bit(in, CharsetTableCP437AMS2);
		return out;
	}
#if defined(MPT_ENABLE_CHARSET_LOCALE)
	#if defined(MPT_LOCALE_ASSUME_CHARSET)
		if(charset == CharsetLocale)
		{
			charset = MPT_LOCALE_ASSUME_CHARSET;
		}
	#endif
#endif
	#if defined(MPT_CHARSET_WIN32)
		if(!HasCharset(charset))
		{
			return DecodeImplFallback<Tsrcstring>(charset, src);
		}
		const UINT codepage = CharsetToCodepage(charset);
		int required_size = MultiByteToWideChar(codepage, 0, reinterpret_cast<const char*>(src.c_str()), -1, nullptr, 0);
		if(required_size <= 0)
		{
			return std::wstring();
		}
		std::vector<WCHAR> decoded_string(required_size);
		MultiByteToWideChar(codepage, 0, reinterpret_cast<const char*>(src.c_str()), -1, decoded_string.data(), required_size);
		return decoded_string.data();
	#elif defined(MPT_CHARSET_ICONV)
		iconv_t conv = iconv_t();
		conv = iconv_open(Charset_wchar_t(), CharsetToString(charset));
		if(!conv)
		{
			throw std::runtime_error("iconv conversion not working");
		}
		std::vector<char> encoded_string(reinterpret_cast<const char*>(src.c_str()), reinterpret_cast<const char*>(src.c_str()) + src.length() + 1);
		std::vector<wchar_t> wide_string(encoded_string.size() * 8); // large enough
		char * inbuf = encoded_string.data();
		size_t inbytesleft = encoded_string.size();
		char * outbuf = reinterpret_cast<char*>(wide_string.data());
		size_t outbytesleft = wide_string.size() * sizeof(wchar_t);
		while(iconv(conv, &inbuf, &inbytesleft, &outbuf, &outbytesleft) == static_cast<size_t>(-1))
		{
			if(errno == EILSEQ || errno == EILSEQ)
			{
				inbuf++;
				inbytesleft--;
				for(std::size_t i = 0; i < sizeof(wchar_t); ++i)
				{
					outbuf[i] = 0;
				}
				#if defined(MPT_PLATFORM_LITTLE_ENDIAN)
					outbuf[1] = uint8(0xff); outbuf[0] = uint8(0xfd);
				#elif defined(MPT_PLATFORM_BIG_ENDIAN)
					outbuf[sizeof(wchar_t)-1 - 1] = uint8(0xff); outbuf[sizeof(wchar_t)-1 - 0] = uint8(0xfd);
				#else
					MPT_MAYBE_CONSTANT_IF(mpt::endian_is_little())
					{
						outbuf[1] = uint8(0xff); outbuf[0] = uint8(0xfd);
					}
					MPT_MAYBE_CONSTANT_IF(mpt::endian_is_big())
					{
						outbuf[sizeof(wchar_t)-1 - 1] = uint8(0xff); outbuf[sizeof(wchar_t)-1 - 0] = uint8(0xfd);
					}
				#endif
				outbuf += sizeof(wchar_t);
				outbytesleft -= sizeof(wchar_t);
				iconv(conv, NULL, NULL, NULL, NULL); // reset state
			} else
			{
				iconv_close(conv);
				conv = iconv_t();
				return std::wstring();
			}
		}
		iconv_close(conv);
		conv = iconv_t();
		return wide_string.data();
	#else
		return DecodeImplFallback<Tsrcstring>(charset, src);
	#endif
}

#if !defined(MPT_CHARSET_ICONV)
template<typename Tsrcstring>
std::wstring DecodeImplFallback(Charset charset, const Tsrcstring &src)
{
		std::string in(src.begin(), src.end());
		std::wstring out;
		switch(charset)
		{
#if defined(MPT_ENABLE_CHARSET_LOCALE)
			case CharsetLocale:      out = String::FromLocale(in); break;
#endif
			case CharsetUTF8:        out = String::FromUTF8(in); break;
			case CharsetASCII:       out = String::FromAscii(in); break;
			case CharsetISO8859_1:   out = String::FromISO_8859_1(in); break;
			case CharsetISO8859_15:  out = String::From8bit(in, CharsetTableISO8859_15); break;
			case CharsetCP437:       out = String::From8bit(in, CharsetTableCP437); break;
			case CharsetCP437AMS:    out = String::From8bit(in, CharsetTableCP437AMS); break;
			case CharsetCP437AMS2:   out = String::From8bit(in, CharsetTableCP437AMS2); break;
			case CharsetWindows1252: out = String::From8bit(in, CharsetTableWindows1252); break;
		}
		return out;
}
#endif // !MPT_CHARSET_ICONV


// templated on 8bit strings because of type-safe variants
template<typename Tdststring, typename Tsrcstring>
Tdststring ConvertImpl(Charset to, Charset from, const Tsrcstring &src)
{
	STATIC_ASSERT(sizeof(typename Tdststring::value_type) == sizeof(char));
	STATIC_ASSERT(sizeof(typename Tsrcstring::value_type) == sizeof(char));
	if(to == from)
	{
		const typename Tsrcstring::value_type * src_beg = src.data();
		const typename Tsrcstring::value_type * src_end = src_beg + src.size();
		return Tdststring(reinterpret_cast<const typename Tdststring::value_type *>(src_beg), reinterpret_cast<const typename Tdststring::value_type *>(src_end));
	}
	#if defined(MPT_CHARSET_ICONV)
		if(to == CharsetCP437AMS || to == CharsetCP437AMS2 || from == CharsetCP437AMS || from == CharsetCP437AMS2)
		{
			return EncodeImpl<Tdststring>(to, DecodeImpl(from, src));
		}
		iconv_t conv = iconv_t();
		conv = iconv_open(CharsetToStringTranslit(to), CharsetToString(from));
		if(!conv)
		{
			conv = iconv_open(CharsetToString(to), CharsetToString(from));
			if(!conv)
			{
				throw std::runtime_error("iconv conversion not working");
			}
		}
		std::vector<char> src_string(reinterpret_cast<const char*>(src.c_str()), reinterpret_cast<const char*>(src.c_str()) + src.length() + 1);
		std::vector<char> dst_string(src_string.size() * 8); // large enough
		char * inbuf = src_string.data();
		size_t inbytesleft = src_string.size();
		char * outbuf = dst_string.data();
		size_t outbytesleft = dst_string.size();
		while(iconv(conv, &inbuf, &inbytesleft, &outbuf, &outbytesleft) == static_cast<size_t>(-1))
		{
			if(errno == EILSEQ || errno == EILSEQ)
			{
				inbuf++;
				inbytesleft--;
				outbuf[0] = '?';
				outbuf++;
				outbytesleft--;
				iconv(conv, NULL, NULL, NULL, NULL); // reset state
			} else
			{
				iconv_close(conv);
				conv = iconv_t();
				return Tdststring();
			}
		}
		iconv_close(conv);
		conv = iconv_t();
		return reinterpret_cast<const typename Tdststring::value_type*>(dst_string.data());
	#else
		return EncodeImpl<Tdststring>(to, DecodeImpl(from, src));
	#endif
}


} // namespace String


bool IsUTF8(const std::string &str)
{
	return (str == String::EncodeImpl<std::string>(mpt::CharsetUTF8, String::DecodeImpl<std::string>(mpt::CharsetUTF8, str)));
}


#if MPT_WSTRING_CONVERT
std::wstring ToWide(Charset from, const std::string &str)
{
	return String::DecodeImpl(from, str);
}
#endif

#if MPT_WSTRING_CONVERT
std::string ToCharset(Charset to, const std::wstring &str)
{
	return String::EncodeImpl<std::string>(to, str);
}
#endif
std::string ToCharset(Charset to, Charset from, const std::string &str)
{
	return String::ConvertImpl<std::string>(to, from, str);
}


#if defined(_MFC_VER)

CString ToCString(const std::wstring &str)
{
	#ifdef UNICODE
		return str.c_str();
	#else
		return ToCharset(CharsetLocale, str).c_str();
	#endif
}
CString ToCString(Charset from, const std::string &str)
{
	#ifdef UNICODE
		return ToWide(from, str).c_str();
	#else
		return ToCharset(CharsetLocale, from, str).c_str();
	#endif
}
std::wstring ToWide(const CString &str)
{
	#ifdef UNICODE
		return str.GetString();
	#else
		return ToWide(CharsetLocale, str.GetString());
	#endif
}
std::string ToCharset(Charset to, const CString &str)
{
	#ifdef UNICODE
		return ToCharset(to, str.GetString());
	#else
		return ToCharset(to, CharsetLocale, str.GetString());
	#endif
}

#ifdef UNICODE
// inline
#else // !UNICODE
CStringW ToCStringW(const CString &str)
{
	return ToWide(str).c_str();
}
CStringW ToCStringW(const std::wstring &str)
{
	return str.c_str();
}
CStringW ToCStringW(Charset from, const std::string &str)
{
	return ToWide(from, str).c_str();
}
CStringW ToCStringW(const CStringW &str)
{
	return str;
}
std::wstring ToWide(const CStringW &str)
{
	return str.GetString();
}
std::string ToCharset(Charset to, const CStringW &str)
{
	return ToCharset(to, str.GetString());
}
CString ToCString(const CStringW &str)
{
	return ToCharset(CharsetLocale, str).c_str();
}
#endif // UNICODE

#endif // MFC


#if MPT_USTRING_MODE_WIDE
// inline
#else // !MPT_USTRING_MODE_WIDE
#if MPT_WSTRING_CONVERT
mpt::ustring ToUnicode(const std::wstring &str)
{
	return String::EncodeImpl<mpt::ustring>(mpt::CharsetUTF8, str);
}
#endif
mpt::ustring ToUnicode(Charset from, const std::string &str)
{
	return String::ConvertImpl<mpt::ustring>(mpt::CharsetUTF8, from, str);
}
#if defined(_MFC_VER)
mpt::ustring ToUnicode(const CString &str)
{
	#ifdef UNICODE
		return String::EncodeImpl<mpt::ustring>(mpt::CharsetUTF8, str.GetString());
	#else // !UNICODE
		return String::ConvertImpl<mpt::ustring, std::string>(mpt::CharsetUTF8, mpt::CharsetLocale, str.GetString());
	#endif // UNICODE
}
#ifndef UNICODE
mpt::ustring ToUnicode(const CStringW &str)
{
	return String::EncodeImpl<mpt::ustring>(mpt::CharsetUTF8, str.GetString());
}
#endif // !UNICODE
#endif // MFC
#endif // MPT_USTRING_MODE_WIDE

#if MPT_USTRING_MODE_WIDE
// nothing, std::wstring overloads will catch all stuff
#else // !MPT_USTRING_MODE_WIDE
#if MPT_WSTRING_CONVERT
std::wstring ToWide(const mpt::ustring &str)
{
	return String::DecodeImpl<mpt::ustring>(mpt::CharsetUTF8, str);
}
#endif
std::string ToCharset(Charset to, const mpt::ustring &str)
{
	return String::ConvertImpl<std::string, mpt::ustring>(to, mpt::CharsetUTF8, str);
}
#if defined(_MFC_VER)
CString ToCString(const mpt::ustring &str)
{
	#ifdef UNICODE
		return String::DecodeImpl<mpt::ustring>(mpt::CharsetUTF8, str).c_str();
	#else // !UNICODE
		return String::ConvertImpl<std::string, mpt::ustring>(mpt::CharsetLocale, mpt::CharsetUTF8, str).c_str();
	#endif // UNICODE
}
#endif // MFC
#endif // MPT_USTRING_MODE_WIDE





char ToLowerCaseAscii(char c)
{
	if('A' <= c && c <= 'Z')
	{
		c += 'a' - 'A';
	}
	return c;
}

char ToUpperCaseAscii(char c)
{
	if('a' <= c && c <= 'z')
	{
		c -= 'a' - 'A';
	}
	return c;
}

std::string ToLowerCaseAscii(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), static_cast<char(*)(char)>(&mpt::ToLowerCaseAscii));
	return s;
}

std::string ToUpperCaseAscii(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), static_cast<char(*)(char)>(&mpt::ToUpperCaseAscii));
	return s;
}

int CompareNoCaseAscii(const char *a, const char *b, std::size_t n)
{
	while(n--)
	{
		unsigned char ac = static_cast<unsigned char>(mpt::ToLowerCaseAscii(*a));
		unsigned char bc = static_cast<unsigned char>(mpt::ToLowerCaseAscii(*b));
		if(ac != bc)
		{
			return ac < bc ? -1 : 1;
		} else if(!ac && !bc)
		{
			return 0;
		}
		++a;
		++b;
	}
	return 0;
}

int CompareNoCaseAscii(const std::string &a, const std::string &b)
{
	for(std::size_t i = 0; i < std::min(a.length(), b.length()); ++i)
	{
		unsigned char ac = static_cast<unsigned char>(mpt::ToLowerCaseAscii(a[i]));
		unsigned char bc = static_cast<unsigned char>(mpt::ToLowerCaseAscii(b[i]));
		if(ac != bc)
		{
			return ac < bc ? -1 : 1;
		} else if(!ac && !bc)
		{
			return 0;
		}
	}
	if(a.length() == b.length())
	{
		return 0;
	}
	return a.length() < b.length() ? -1 : 1;
}

#if defined(MODPLUG_TRACKER)

mpt::ustring ToLowerCase(const mpt::ustring &s)
{
	#if defined(_MFC_VER)
		#if defined(UNICODE)
			CString tmp = mpt::ToCString(s);
			tmp.MakeLower();
			return mpt::ToUnicode(tmp);
		#else // !UNICODE
			CStringW tmp = mpt::ToCStringW(s);
			tmp.MakeLower();
			return mpt::ToUnicode(tmp);
		#endif // UNICODE
	#else // !_MFC_VER
		std::wstring ws = mpt::ToWide(s);
		std::transform(ws.begin(), ws.end(), ws.begin(), &std::towlower);
		return mpt::ToUnicode(ws);
	#endif // _MFC_VER
}

mpt::ustring ToUpperCase(const mpt::ustring &s)
{
	#if defined(_MFC_VER)
		#if defined(UNICODE)
			CString tmp = mpt::ToCString(s);
			tmp.MakeUpper();
			return mpt::ToUnicode(tmp);
		#else // !UNICODE
			CStringW tmp = mpt::ToCStringW(s);
			tmp.MakeUpper();
			return mpt::ToUnicode(tmp);
		#endif // UNICODE
	#else // !_MFC_VER
		std::wstring ws = mpt::ToWide(s);
		std::transform(ws.begin(), ws.end(), ws.begin(), &std::towlower);
		return mpt::ToUnicode(ws);
	#endif // _MFC_VER
}

#endif // MODPLUG_TRACKER



} // namespace mpt



OPENMPT_NAMESPACE_END
