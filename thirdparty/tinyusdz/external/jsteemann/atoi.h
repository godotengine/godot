// Based on https://github.com/jsteemann/atoi
// Modified to return error codes
#ifndef JSTEEMANN_ATOI_H
#define JSTEEMANN_ATOI_H 1

#pragma once

#include <limits>
#include <type_traits>

// some macros to help with branch prediction
#if defined(__GNUC__) || defined(__GNUG__)
#define ATOI_LIKELY(v) __builtin_expect(!!(v), 1)
#define ATOI_UNLIKELY(v) __builtin_expect(!!(v), 0)
#else
#define ATOI_LIKELY(v) v
#define ATOI_UNLIKELY(v) v
#endif

#define JSTEEMANN_NOEXCEPT noexcept

namespace jsteemann {

enum ErrCode : int {
  SUCCESS,
  INVALID_INPUT = -1,
  INVALID_NEGATIVE_SIGN = -2,
  VALUE_OVERFLOW = -3,
  VALUE_UNDERFLOW= -4
};
//
// errcode
// 0 : success
// -1 : invalid input
// -2 : negative sign(`-`) detected for positive atoi
// -3 : overflow
// -4 : underflow
//

// low-level worker function to convert the string value between p
// (inclusive) and e (exclusive) into a negative number value of type T,
// without validation of the input string - use this only for trusted input!
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'.
// there is no validation of the input string, and overflow or underflow
// of the result value will not be detected.
// this function will not modify errno.
template<typename T>
inline T atoi_negative_unchecked(char const* p, char const* e) JSTEEMANN_NOEXCEPT {
  T result = 0;
  while (p != e) {
    result = (result << 1) + (result << 3) - (*(p++) - '0');
  }
  return result;
}

// low-level worker function to convert the string value between p
// (inclusive) and e (exclusive) into a positive number value of type T,
// without validation of the input string - use this only for trusted input!
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'.
// there is no validation of the input string, and overflow or underflow
// of the result value will not be detected.
// this function will not modify errno.
template<typename T>
inline T atoi_positive_unchecked(char const* p, char const* e) JSTEEMANN_NOEXCEPT {
  T result = 0;
  while (p != e) {
    result = (result << 1) + (result << 3) + *(p++) - '0';
  }

  return result;
}

// function to convert the string value between p
// (inclusive) and e (exclusive) into a number value of type T, without
// validation of the input string - use this only for trusted input!
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'. an
// optional '+' or '-' sign is allowed too.
// there is no validation of the input string, and overflow or underflow
// of the result value will not be detected.
// this function will not modify errno.
template<typename T>
inline T atoi_unchecked(char const* p, char const* e) JSTEEMANN_NOEXCEPT {
  if (ATOI_UNLIKELY(p == e)) {
    return T();
  }

  if (*p == '-') {
    if (!std::is_signed<T>::value) {
      return T();
    }
    return atoi_negative_unchecked<T>(++p, e);
  }
  if (ATOI_UNLIKELY(*p == '+')) {
    ++p;
  }

  return atoi_positive_unchecked<T>(p, e);
}

// low-level worker function to convert the string value between p
// (inclusive) and e (exclusive) into a negative number value of type T
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'.
// if any other character is found, the output parameter "errcode" will
// be set to -1(invalid input). if the parsed value is less than what type T can
// store without truncation, "errcode" will be set to -4(underflow).
// this function will not modify errno.
template<typename T>
inline T atoi_negative(char const* p, char const* e, int& errcode) JSTEEMANN_NOEXCEPT {
  if (ATOI_UNLIKELY(p == e)) {
    errcode = -1;
    return T();
  }

  constexpr T cutoff = (std::numeric_limits<T>::min)() / 10;
  constexpr char cutlim = -((std::numeric_limits<T>::min)() % 10);
  T result = 0;

  do {
    char c = *p;

    if ((c == '\0') || (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r') || (c == '\v') || (c == '\f')) {
      errcode = 0;
      return result;
    }

    // we expect only '0' to '9'. everything else is unexpected
    if (ATOI_UNLIKELY(c < '0' || c > '9')) {
      errcode = -1;
      return result;
    }

    c -= '0';
    // we expect the bulk of values to not hit the bounds restrictions
    if (ATOI_UNLIKELY(result < cutoff || (result == cutoff && c > cutlim))) {
      errcode = -4;
      return result;
    }
    result *= 10;
    result -= c;
  } while (++p < e);

  errcode = 0;
  return result;
}

// low-level worker function to convert the string value between p
// (inclusive) and e (exclusive) into a positive number value of type T
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'.
// if any other character is found, the output parameter "errcode" will
// be set to -1. if the parsed value is greater than what type T can
// store without truncation, "errcode" will be set to -3(overflow).
// this function will not modify errno.
template<typename T>
inline T atoi_positive(char const* p, char const* e, int& errcode) JSTEEMANN_NOEXCEPT {
  if (ATOI_UNLIKELY(p == e)) {
    errcode = -1;
    return T();
  }

  constexpr T cutoff = (std::numeric_limits<T>::max)() / 10;
  constexpr char cutlim = (std::numeric_limits<T>::max)() % 10;
  T result = 0;

  do {
    char c = *p;

    if ((c == '\0') || (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r') || (c == '\v') || (c == '\f')) {
      errcode = 0;
      return result;
    }

    // we expect only '0' to '9'. everything else is unexpected
    if (ATOI_UNLIKELY(c < '0' || c > '9')) {
      errcode = -1;
      return result;
    }

    c -= '0';
    // we expect the bulk of values to not hit the bounds restrictions
    if (ATOI_UNLIKELY(result > cutoff || (result == cutoff && c > cutlim))) {
      errcode = -3;
      return result;
    }
    result *= 10;
    result += c;
  } while (++p < e);

  errcode = 0;
  return result;
}

// function to convert the string value between p
// (inclusive) and e (exclusive) into a number value of type T
//
// the input string will always be interpreted as a base-10 number.
// expects the input string to contain only the digits '0' to '9'. an
// optional '+' or '-' sign is allowed too.
// if any other character is found, the output parameter "errcode" will
// be set to -1. if the parsed value is less or greater than what
// type T can store without truncation, "errcode" will be set to
// -3(oveerflow)
// this function will not modify errno.
template<typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type atoi(char const* p, char const* e, int& errcode) JSTEEMANN_NOEXCEPT {
  if (ATOI_UNLIKELY(p == e)) {
    errcode = -1;
    return T();
  }

  if (*p == '-') {
    return atoi_negative<T>(++p, e, errcode);
  }
  if (ATOI_UNLIKELY(*p == '+')) {
    ++p;
  }

  return atoi_positive<T>(p, e, errcode);
}

template<typename T>
inline typename std::enable_if<std::is_unsigned<T>::value, T>::type atoi(char const* p, char const* e, int &errcode) JSTEEMANN_NOEXCEPT {
  if (ATOI_UNLIKELY(p == e)) {
    errcode = -1;
    return T();
  }

  if (*p == '-') {
    errcode = -2;
    return T();
  }
  if (ATOI_UNLIKELY(*p == '+')) {
    ++p;
  }

  return atoi_positive<T>(p, e, errcode);
}

}

#endif
