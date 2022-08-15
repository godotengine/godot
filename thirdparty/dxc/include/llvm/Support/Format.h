//===- Format.h - Efficient printf-style formatting for streams -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the format() function, which can be used with other
// LLVM subsystems to provide printf-style formatting.  This gives all the power
// and risk of printf.  This can be used like this (with raw_ostreams as an
// example):
//
//    OS << "mynumber: " << format("%4.5f", 1234.412) << '\n';
//
// Or if you prefer:
//
//  OS << format("mynumber: %4.5f\n", 1234.412);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMAT_H
#define LLVM_SUPPORT_FORMAT_H

#include "dxc/Support/WinAdapter.h" // HLSL Change
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <cstdio>
#include <tuple>

namespace llvm {

/// This is a helper class used for handling formatted output.  It is the
/// abstract base class of a templated derived class.
class format_object_base {
protected:
  const char *Fmt;
  ~format_object_base() = default; // Disallow polymorphic deletion.
  format_object_base(const format_object_base &) = default;
  virtual void home(); // Out of line virtual method.

  /// Call snprintf() for this object, on the given buffer and size.
  virtual int snprint(_Out_ char *Buffer, unsigned BufferSize) const = 0;   // HLSL Change - SAL

public:
  format_object_base(const char *fmt) : Fmt(fmt) {}

  /// Format the object into the specified buffer.  On success, this returns
  /// the length of the formatted string.  If the buffer is too small, this
  /// returns a length to retry with, which will be larger than BufferSize.
  unsigned print(_Out_ char *Buffer, unsigned BufferSize) const {   // HLSL Change - SAL
    assert(BufferSize && "Invalid buffer size!");

    // Print the string, leaving room for the terminating null.
    int N = snprint(Buffer, BufferSize);

    // VC++ and old GlibC return negative on overflow, just double the size.
    if (N < 0)
      return BufferSize * 2;

    // Other implementations yield number of bytes needed, not including the
    // final '\0'.
    if (unsigned(N) >= BufferSize)
      return N + 1;

    // Otherwise N is the length of output (not including the final '\0').
    return N;
  }
};

/// These are templated helper classes used by the format function that
/// capture the object to be formated and the format string. When actually
/// printed, this synthesizes the string into a temporary buffer provided and
/// returns whether or not it is big enough.

template <typename... Ts>
class format_object final : public format_object_base {
  std::tuple<Ts...> Vals;

  template <std::size_t... Is>
  int snprint_tuple(char *Buffer, unsigned BufferSize,
                    index_sequence<Is...>) const {
#ifdef _MSC_VER
    // Use _TRUNCATE as the buffer size; truncation will still return -1 as
    // a result, thereby triggering the 'double on VC++' behavior in
    // caller, for example llvm::format_object_base::print(char * Buffer, unsigned int BufferSize)
    return _snprintf_s(Buffer, BufferSize, _TRUNCATE, Fmt, std::get<Is>(Vals)...);
#else
    return snprintf(Buffer, BufferSize, Fmt, std::get<Is>(Vals)...);
#endif
  }

public:
  format_object(const char *fmt, const Ts &... vals)
      : format_object_base(fmt), Vals(vals...) {}

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprint_tuple(Buffer, BufferSize, index_sequence_for<Ts...>());
  }
};

/// These are helper functions used to produce formatted output.  They use
/// template type deduction to construct the appropriate instance of the
/// format_object class to simplify their construction.
///
/// This is typically used like:
/// \code
///   OS << format("%0.4f", myfloat) << '\n';
/// \endcode

template <typename... Ts>
inline format_object<Ts...> format(const char *Fmt, const Ts &... Vals) {
  return format_object<Ts...>(Fmt, Vals...);
}

/// This is a helper class used for left_justify() and right_justify().
class FormattedString {
  StringRef Str;
  unsigned Width;
  bool RightJustify;
  friend class raw_ostream;
public:
    FormattedString(StringRef S, unsigned W, bool R)
      : Str(S), Width(W), RightJustify(R) { }
};

/// left_justify - append spaces after string so total output is
/// \p Width characters.  If \p Str is larger that \p Width, full string
/// is written with no padding.
inline FormattedString left_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, false);
}

/// right_justify - add spaces before string so total output is
/// \p Width characters.  If \p Str is larger that \p Width, full string
/// is written with no padding.
inline FormattedString right_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, true);
}

/// This is a helper class used for format_hex() and format_decimal().
class FormattedNumber {
  uint64_t HexValue;
  int64_t DecValue;
  unsigned Width;
  bool Hex;
  bool Upper;
  bool HexPrefix;
  friend class raw_ostream;
public:
  FormattedNumber(uint64_t HV, int64_t DV, unsigned W, bool H, bool U,
                  bool Prefix)
      : HexValue(HV), DecValue(DV), Width(W), Hex(H), Upper(U),
        HexPrefix(Prefix) {}
};

/// format_hex - Output \p N as a fixed width hexadecimal. If number will not
/// fit in width, full number is still printed.  Examples:
///   OS << format_hex(255, 4)              => 0xff
///   OS << format_hex(255, 4, true)        => 0xFF
///   OS << format_hex(255, 6)              => 0x00ff
///   OS << format_hex(255, 2)              => 0xff
inline FormattedNumber format_hex(uint64_t N, unsigned Width,
                                  bool Upper = false) {
  assert(Width <= 18 && "hex width must be <= 18");
  return FormattedNumber(N, 0, Width, true, Upper, true);
}

/// format_hex_no_prefix - Output \p N as a fixed width hexadecimal. Does not
/// prepend '0x' to the outputted string.  If number will not fit in width,
/// full number is still printed.  Examples:
///   OS << format_hex_no_prefix(255, 4)              => ff
///   OS << format_hex_no_prefix(255, 4, true)        => FF
///   OS << format_hex_no_prefix(255, 6)              => 00ff
///   OS << format_hex_no_prefix(255, 2)              => ff
inline FormattedNumber format_hex_no_prefix(uint64_t N, unsigned Width,
                                            bool Upper = false) {
  assert(Width <= 18 && "hex width must be <= 18");
  return FormattedNumber(N, 0, Width, true, Upper, false);
}

/// format_decimal - Output \p N as a right justified, fixed-width decimal. If 
/// number will not fit in width, full number is still printed.  Examples:
///   OS << format_decimal(0, 5)     => "    0"
///   OS << format_decimal(255, 5)   => "  255"
///   OS << format_decimal(-1, 3)    => " -1"
///   OS << format_decimal(12345, 3) => "12345"
inline FormattedNumber format_decimal(int64_t N, unsigned Width) {
  return FormattedNumber(0, N, Width, false, false, false);
}


} // end namespace llvm

#endif
