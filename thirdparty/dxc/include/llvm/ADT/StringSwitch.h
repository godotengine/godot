//===--- StringSwitch.h - Switch-on-literal-string Construct --------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements the StringSwitch template, which mimics a switch()
//  statement whose cases are string literals.
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_ADT_STRINGSWITCH_H
#define LLVM_ADT_STRINGSWITCH_H

#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstring>

namespace llvm {

/// \brief A switch()-like statement whose cases are string literals.
///
/// The StringSwitch class is a simple form of a switch() statement that
/// determines whether the given string matches one of the given string
/// literals. The template type parameter \p T is the type of the value that
/// will be returned from the string-switch expression. For example,
/// the following code switches on the name of a color in \c argv[i]:
///
/// \code
/// Color color = StringSwitch<Color>(argv[i])
///   .Case("red", Red)
///   .Case("orange", Orange)
///   .Case("yellow", Yellow)
///   .Case("green", Green)
///   .Case("blue", Blue)
///   .Case("indigo", Indigo)
///   .Cases("violet", "purple", Violet)
///   .Default(UnknownColor);
/// \endcode
template<typename T, typename R = T>
class StringSwitch {
  /// \brief The string we are matching.
  StringRef Str;

  /// \brief The pointer to the result of this switch statement, once known,
  /// null before that.
  const T *Result;

public:
  explicit StringSwitch(StringRef S)
  : Str(S), Result(nullptr) { }

  template<unsigned N>
  StringSwitch& Case(const char (&S)[N], const T& Value) {
    if (!Result && N-1 == Str.size() &&
        (std::memcmp(S, Str.data(), N-1) == 0)) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N>
  StringSwitch& EndsWith(const char (&S)[N], const T &Value) {
    if (!Result && Str.size() >= N-1 &&
        std::memcmp(S, Str.data() + Str.size() + 1 - N, N-1) == 0) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N>
  StringSwitch& StartsWith(const char (&S)[N], const T &Value) {
    if (!Result && Str.size() >= N-1 &&
        std::memcmp(S, Str.data(), N-1) == 0) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N0, unsigned N1>
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const T& Value) {
    return Case(S0, Value).Case(S1, Value);
  }

  template<unsigned N0, unsigned N1, unsigned N2>
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const T& Value) {
    return Case(S0, Value).Case(S1, Value).Case(S2, Value);
  }

  template<unsigned N0, unsigned N1, unsigned N2, unsigned N3>
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const char (&S3)[N3],
                      const T& Value) {
    return Case(S0, Value).Case(S1, Value).Case(S2, Value).Case(S3, Value);
  }

  template<unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4>
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const char (&S3)[N3],
                      const char (&S4)[N4], const T& Value) {
    return Case(S0, Value).Case(S1, Value).Case(S2, Value).Case(S3, Value)
      .Case(S4, Value);
  }

  R Default(const T& Value) const {
    if (Result)
      return *Result;

    return Value;
  }

  operator R() const {
    assert(Result && "Fell off the end of a string-switch");
    return *Result;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGSWITCH_H
