//===-- llvm/ADT/StringExtras.h - Useful string functions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful when dealing with strings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGEXTRAS_H
#define LLVM_ADT_STRINGEXTRAS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <iterator>

namespace llvm {
template<typename T> class SmallVectorImpl;

/// hexdigit - Return the hexadecimal character for the
/// given number \p X (which should be less than 16).
static inline char hexdigit(unsigned X, bool LowerCase = false) {
  const char HexChar = LowerCase ? 'a' : 'A';
  return X < 10 ? '0' + X : HexChar + X - 10;
}

/// Construct a string ref from a boolean.
static inline StringRef toStringRef(bool B) {
  return StringRef(B ? "true" : "false");
}

/// Interpret the given character \p C as a hexadecimal digit and return its
/// value.
///
/// If \p C is not a valid hex digit, -1U is returned.
static inline unsigned hexDigitValue(char C) {
  if (C >= '0' && C <= '9') return C-'0';
  if (C >= 'a' && C <= 'f') return C-'a'+10U;
  if (C >= 'A' && C <= 'F') return C-'A'+10U;
  return -1U;
}

/// utohex_buffer - Emit the specified number into the buffer specified by
/// BufferEnd, returning a pointer to the start of the string.  This can be used
/// like this: (note that the buffer must be large enough to handle any number):
///    char Buffer[40];
///    printf("0x%s", utohex_buffer(X, Buffer+40));
///
/// This should only be used with unsigned types.
///
template<typename IntTy>
static inline char *utohex_buffer(IntTy X, char *BufferEnd, bool LowerCase = false) {
  char *BufPtr = BufferEnd;
  *--BufPtr = 0;      // Null terminate buffer.
  if (X == 0) {
#pragma prefast(suppress: 22102, "buffer for BufferEnd always has room for one char")
    *--BufPtr = '0';  // Handle special case.
    return BufPtr;
  }

  while (X) {
    unsigned char Mod = static_cast<unsigned char>(X) & 15;
#pragma prefast(suppress: 22103, "X runs down to zero before end of buffer depending on type - cannot express for template in SAL")
    *--BufPtr = hexdigit(Mod, LowerCase);
    X >>= 4;
  }
  return BufPtr;
}

static inline std::string utohexstr(uint64_t X, bool LowerCase = false) {
  char Buffer[17];
  return utohex_buffer(X, Buffer+17, LowerCase);
}

static inline std::string utostr_32(uint32_t X, bool isNeg = false) {
  char Buffer[11];
  char *BufPtr = Buffer+11;

  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
#pragma prefast(suppress: 22102, "X runs down to zero before end of buffer")
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return std::string(BufPtr, Buffer+11);
}

static inline std::string utostr(uint64_t X, bool isNeg = false) {
  char Buffer[21];
  char *BufPtr = Buffer+21;

  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
#pragma prefast(suppress: 22102, "X runs down to zero before end of buffer")
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...
  return std::string(BufPtr, Buffer+21);
}


static inline std::string itostr(int64_t X) {
  if (X < 0)
    return utostr(static_cast<uint64_t>(-X), true);
  else
    return utostr(static_cast<uint64_t>(X));
}

/// StrInStrNoCase - Portable version of strcasestr.  Locates the first
/// occurrence of string 's1' in string 's2', ignoring case.  Returns
/// the offset of s2 in s1 or npos if s2 cannot be found.
StringRef::size_type StrInStrNoCase(StringRef s1, StringRef s2);

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The function returns a pair containing the extracted token and the
/// remaining tail string.
std::pair<StringRef, StringRef> getToken(StringRef Source,
                                         StringRef Delimiters = " \t\n\v\f\r");

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void SplitString(StringRef Source,
                 SmallVectorImpl<StringRef> &OutFragments,
                 StringRef Delimiters = " \t\n\v\f\r");

/// HashString - Hash function for strings.
///
/// This is the Bernstein hash function.
//
// FIXME: Investigate whether a modified bernstein hash function performs
// better: http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx
//   X*33+c -> X*33^c
static inline unsigned HashString(StringRef Str, unsigned Result = 0) {
  for (StringRef::size_type i = 0, e = Str.size(); i != e; ++i)
    Result = Result * 33 + (unsigned char)Str[i];
  return Result;
}

/// Returns the English suffix for an ordinal integer (-st, -nd, -rd, -th).
static inline StringRef getOrdinalSuffix(unsigned Val) {
  // It is critically important that we do this perfectly for
  // user-written sequences with over 100 elements.
  switch (Val % 100) {
  case 11:
  case 12:
  case 13:
    return "th";
  default:
    switch (Val % 10) {
      case 1: return "st";
      case 2: return "nd";
      case 3: return "rd";
      default: return "th";
    }
  }
}

template <typename IteratorT>
inline std::string join_impl(IteratorT Begin, IteratorT End,
                             StringRef Separator, std::input_iterator_tag) {
  std::string S;
  if (Begin == End)
    return S;

  S += (*Begin);
  while (++Begin != End) {
    S += Separator;
    S += (*Begin);
  }
  return S;
}

template <typename IteratorT>
inline std::string join_impl(IteratorT Begin, IteratorT End,
                             StringRef Separator, std::forward_iterator_tag) {
  std::string S;
  if (Begin == End)
    return S;

  size_t Len = (std::distance(Begin, End) - 1) * Separator.size();
  for (IteratorT I = Begin; I != End; ++I)
    Len += (*Begin).size();
  S.reserve(Len);
  S += (*Begin);
  while (++Begin != End) {
    S += Separator;
    S += (*Begin);
  }
  return S;
}

/// Joins the strings in the range [Begin, End), adding Separator between
/// the elements.
template <typename IteratorT>
inline std::string join(IteratorT Begin, IteratorT End, StringRef Separator) {
  typedef typename std::iterator_traits<IteratorT>::iterator_category tag;
  return join_impl(Begin, End, Separator, tag());
}

} // End llvm namespace

#endif
