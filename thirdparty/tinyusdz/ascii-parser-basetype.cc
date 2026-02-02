// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2022, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment, Inc.
//
// Ascii Basic type parser
//

#include <cstdio>
#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <atomic>
//#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#if defined(__wasi__)
#else
#include <mutex>
#include <thread>
#endif
#include <vector>

#include "ascii-parser.hh"
#include "str-util.hh"
#include "path-util.hh"
#include "tiny-format.hh"

//
#if !defined(TINYUSDZ_DISABLE_MODULE_USDA_READER)

//

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// external

#include "external/fast_float/include/fast_float/fast_float.h"
#include "external/jsteemann/atoi.h"
//#include "external/simple_match/include/simple_match/simple_match.hpp"
#include "nonstd/expected.hpp"

//

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//

// Tentative
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include "io-util.hh"
#include "pprinter.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "value-pprint.hh"
#include "value-types.hh"

#include "common-macros.inc"

namespace tinyusdz {

namespace ascii {

constexpr auto kAscii = "[ASCII]";

namespace {

// parseInt
// 0 = success
// -1 = bad input
// -2 = overflow
// -3 = underflow
int parseInt(const std::string &s, int *out_result) {
  size_t n = s.size();
  const char *c = s.c_str();

  if ((c == nullptr) || (*c) == '\0') {
    return -1;
  }

  size_t idx = 0;
  bool negative = c[0] == '-';
  if ((c[0] == '+') | (c[0] == '-')) {
    idx = 1;
    if (n == 1) {
      // sign char only
      return -1;
    }
  }

  int64_t result = 0;

  // allow zero-padded digits(e.g. "003")
  while (idx < n) {
    if ((c[idx] >= '0') && (c[idx] <= '9')) {
      int digit = int(c[idx] - '0');
      result = result * 10 + digit;
    } else {
      // bad input
      return -1;
    }

    if (negative) {
      if ((-result) < (std::numeric_limits<int>::min)()) {
        return -3;
      }
    } else {
      if (result > (std::numeric_limits<int>::max)()) {
        return -2;
      }
    }

    idx++;
  }

  if (negative) {
    (*out_result) = -int(result);
  } else {
    (*out_result) = int(result);
  }

  return 0;  // OK
}

nonstd::expected<float, std::string> ParseFloat(const std::string &s) {

  // Parse with fast_float
  float result;
  auto ans = fast_float::from_chars(s.data(), s.data() + s.size(), result);
  if (ans.ec != std::errc()) {
    // Current `fast_float` implementation does not report detailed parsing err.
    return nonstd::make_unexpected("Parse failed.");
  }

  return result;
}

nonstd::expected<double, std::string> ParseDouble(const std::string &s) {

  // Parse with fast_float
  double result;
  auto ans = fast_float::from_chars(s.data(), s.data() + s.size(), result);
  if (ans.ec != std::errc()) {
    // Current `fast_float` implementation does not report detailed parsing err.
    return nonstd::make_unexpected("Parse failed.");
  }

  return result;
}

}  // namespace

//
// -- Parse Basic Type
//
bool AsciiParser::ParseMatrix(value::matrix2f *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<float, 2>> content;
  if (!SepBy1TupleType<float, 2>(',', &content)) {
    return false;
  }

  if (content.size() != 2) {
    PushError("# of rows in matrix2f must be 2, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 2; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
  }

  return true;
}

bool AsciiParser::ParseMatrix(value::matrix3f *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<float, 3>> content;
  if (!SepBy1TupleType<float, 3>(',', &content)) {
    return false;
  }

  if (content.size() != 3) {
    PushError("# of rows in matrix3f must be 3, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 3; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
    result->m[i][2] = content[i][2];
  }

  return true;
}

bool AsciiParser::ParseMatrix(value::matrix4f *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<float, 4>> content;
  if (!SepBy1TupleType<float, 4>(',', &content)) {
    return false;
  }

  if (content.size() != 4) {
    PushError("# of rows in matrix4f must be 4, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 4; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
    result->m[i][2] = content[i][2];
    result->m[i][3] = content[i][3];
  }

  return true;
}

bool AsciiParser::ParseMatrix(value::matrix2d *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<double, 2>> content;
  if (!SepBy1TupleType<double, 2>(',', &content)) {
    return false;
  }

  if (content.size() != 2) {
    PushError("# of rows in matrix2d must be 2, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 2; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
  }

  return true;
}

bool AsciiParser::ParseMatrix(value::matrix3d *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<double, 3>> content;
  if (!SepBy1TupleType<double, 3>(',', &content)) {
    return false;
  }

  if (content.size() != 3) {
    PushError("# of rows in matrix3d must be 3, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 3; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
    result->m[i][2] = content[i][2];
  }

  return true;
}

bool AsciiParser::ParseMatrix(value::matrix4d *result) {

  if (!Expect('(')) {
    return false;
  }

  std::vector<std::array<double, 4>> content;
  if (!SepBy1TupleType<double, 4>(',', &content)) {
    return false;
  }

  if (content.size() != 4) {
    PushError("# of rows in matrix4d must be 4, but got " +
              std::to_string(content.size()) + "\n");
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  for (size_t i = 0; i < 4; i++) {
    result->m[i][0] = content[i][0];
    result->m[i][1] = content[i][1];
    result->m[i][2] = content[i][2];
    result->m[i][3] = content[i][3];
  }

  return true;
}

bool AsciiParser::ReadBasicType(Path *value) {
  if (value) {
    std::string str;
    if (!ReadPathIdentifier(&str)) {
      return false;
    }

    (*value) = pathutil::FromString(str);

    return true;
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<Path> *value) {

  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  Path v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix2d *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix2d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix2d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix3d *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix4d *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix4d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix4d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix2f *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix2f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix2f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix3f *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::matrix4f *value) {
  if (value) {
    return ParseMatrix(value);
  } else {
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::matrix4f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::matrix4f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

///
/// Parse the array of tuple. some may be None(e.g. `float3`: [(0, 1, 2),
/// None, (2, 3, 4), ...] )
///
template <typename T, size_t N>
bool AsciiParser::ParseTupleArray(
    std::vector<nonstd::optional<std::array<T, N>>> *result) {

  if (!Expect('[')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  // Empty array?
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == ']') {
      result->clear();
      return true;
    }

    Rewind(1);
  }

  if (!SepBy1TupleType<T, N>(',', result)) {
    return false;
  }

  if (!Expect(']')) {
    return false;
  }

  return true;
}

// instanciations
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int32_t, 2>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int32_t, 3>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int32_t, 4>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint32_t, 2>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint32_t, 3>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint32_t, 4>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int64_t, 2>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int64_t, 3>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<int64_t, 4>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint64_t, 2>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint64_t, 3>>> *result);
template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<uint64_t, 4>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<float, 2>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<float, 3>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<float, 4>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<double, 2>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<double, 3>>> *result);
//template bool AsciiParser::ParseTupleArray(std::vector<nonstd::optional<std::array<double, 4>>> *result);

///
/// Parse the array of tuple(e.g. `float3`: [(0, 1, 2), (2, 3, 4), ...] )
///
template <typename T, size_t N>
bool AsciiParser::ParseTupleArray(std::vector<std::array<T, N>> *result) {
  (void)result;

  if (!Expect('[')) {
    return false;
  }

  if (!SepBy1TupleType<T, N>(',', result)) {
    return false;
  }

  if (!Expect(']')) {
    return false;
  }

  return true;
}

bool AsciiParser::ReadBasicType(Identifier *value) {
  return ReadIdentifier(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<Identifier> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  Identifier v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::token *value) {
  // Try triple-quotated string first.
  {
    value::StringData sdata;
    if (MaybeTripleQuotedString(&sdata)) {
      // TODO: preserve quotation info.
      (*value) = value::token(sdata.value);
      return true;
    }
  }

  std::string s;
  if (!ReadStringLiteral(&s)) {
    PUSH_ERROR_AND_RETURN_TAG(kAscii, "Failed to parse string literal.");
    return false;
  }

  (*value) = value::token(s);
  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::token> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::token v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(std::string *value) {
  if (!value) {
    return false;
  }

  // May be triple-quoted string
  {
    value::StringData sdata;
    if (MaybeTripleQuotedString(&sdata)) {
      (*value) = sdata.value;
      return true;

    } else if (MaybeString(&sdata)) {
      (*value) = sdata.value;
      return true;
    }
  }

  // Just in case
  return ReadStringLiteral(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<std::string> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  std::string v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::StringData *value) {
  if (!value) {
    return false;
  }

  // May be triple-quoted string
  {
    value::StringData sdata;
    if (MaybeTripleQuotedString(&sdata)) {
      (*value) = sdata;
      return true;

    } else if (MaybeString(&sdata)) {
      (*value) = sdata;
      return true;
    }
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::StringData> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::StringData v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(PathIdentifier *value) {
  return ReadPathIdentifier(value);
}

bool AsciiParser::ReadBasicType(bool *value) {
  // 'true', 'false', '0' or '1'
  {
    std::string tok;

    auto loc = CurrLoc();
    bool ok = ReadIdentifier(&tok);

    if (ok) {
      if (tok == "true") {
        (*value) = true;
        return true;
      } else if (tok == "false") {
        (*value) = false;
        return true;
      }
    }

    // revert
    SeekTo(loc);
  }

  char sc;
  if (!Char1(&sc)) {
    return false;
  }
  _curr_cursor.col++;

  // sign or [0-9]
  if (sc == '0') {
    (*value) = false;
    return true;
  } else if (sc == '1') {
    (*value) = true;
    return true;
  } else {
    PushError("'0' or '1' expected.\n");
    return false;
  }
}

bool AsciiParser::ReadBasicType(nonstd::optional<bool> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  bool v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(int *value) {
  std::stringstream ss;

  // pxrUSD allow floating-point value to `int` type.
  // so first try fp parsing.
  auto loc = CurrLoc();
  std::string fp_str;
  if (LexFloat(&fp_str)) {
    auto flt = ParseDouble(fp_str);
    if (!flt) {
      PUSH_ERROR_AND_RETURN("Failed to parse floating value.");
    } else {
      (*value) = int(flt.value());
      return true;
    }
  }

  // revert
  SeekTo(loc);

  // head character
  bool has_sign = false;
  // bool negative = false;
  {
    char sc;
    if (!Char1(&sc)) {
      return false;
    }
    _curr_cursor.col++;

    // sign or [0-9]
    if (sc == '+') {
      // negative = false;
      has_sign = true;
    } else if (sc == '-') {
      // negative = true;
      has_sign = true;
    } else if ((sc >= '0') && (sc <= '9')) {
      // ok
    } else {
      PushError("Sign or 0-9 expected, but got '" + std::to_string(sc) +
                "'.\n");
      return false;
    }

    ss << sc;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if ((c >= '0') && (c <= '9')) {
      ss << c;
    } else {
      _sr->seek_from_current(-1);
      break;
    }
  }

  if (has_sign && (ss.str().size() == 1)) {
    // sign only
    PushError("Integer value expected but got sign character only.\n");
    return false;
  }

  if ((ss.str().size() > 1) && (ss.str()[0] == '0')) {
    PushError("Zero padded integer value is not allowed.\n");
    return false;
  }

  // std::cout << "ReadInt token: " << ss.str() << "\n";

  int int_value;
  int err = parseInt(ss.str(), &int_value);
  if (err != 0) {
    if (err == -1) {
      PushError("Invalid integer input: `" + ss.str() + "`\n");
      return false;
    } else if (err == -2) {
      PushError("Integer overflows: `" + ss.str() + "`\n");
      return false;
    } else if (err == -3) {
      PushError("Integer underflows: `" + ss.str() + "`\n");
      return false;
    } else {
      PushError("Unknown parseInt error.\n");
      return false;
    }
  }

  (*value) = int_value;

  return true;
}

bool AsciiParser::ReadBasicType(value::int2 *value) {
  return ParseBasicTypeTuple<int32_t, 2>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::int2> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::int2 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::int3 *value) {
  return ParseBasicTypeTuple<int32_t, 3>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::int3> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::int3 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::int4 *value) {
  return ParseBasicTypeTuple<int32_t, 4>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::int4> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::int4 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::uint2 *value) {
  return ParseBasicTypeTuple<uint32_t, 2>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::uint2> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::uint2 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::uint3 *value) {
  return ParseBasicTypeTuple<uint32_t, 3>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::uint3> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::uint3 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::uint4 *value) {
  return ParseBasicTypeTuple<uint32_t, 4>(value);
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::uint4> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::uint4 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}


bool AsciiParser::ReadBasicType(uint32_t *value) {
  std::stringstream ss;
  constexpr uint32_t kMaxDigits = 100;

  // head character
  bool has_sign = false;
  bool negative = false;
  {
    char sc;
    if (!Char1(&sc)) {
      return false;
    }
    _curr_cursor.col++;

    // sign or [0-9]
    if (sc == '+') {
      negative = false;
      has_sign = true;
    } else if (sc == '-') {
      negative = true;
      has_sign = true;
    } else if ((sc >= '0') && (sc <= '9')) {
      // ok
    } else {
      PushError("Sign or 0-9 expected, but got '" + std::to_string(sc) +
                "'.\n");
      return false;
    }

    ss << sc;
  }

  if (negative) {
    PushError("Unsigned value expected but got '-' sign.");
    return false;
  }

  uint32_t digits=0;
  while (!Eof()) {

    if (digits > kMaxDigits) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if ((c >= '0') && (c <= '9')) {
      ss << c;
      digits++;
    } else {
      _sr->seek_from_current(-1);
      break;
    }
  }

  std::string str = ss.str();

  if (has_sign && (str.size() == 1)) {
    // sign only
    PushError("Integer value expected but got sign character only.\n");
    return false;
  }

  if ((str.size() > 1) && (str[0] == '0')) {
    PushError("Zero padded integer value is not allowed.\n");
    return false;
  }

  // std::cout << "ReadInt token: " << ss.str() << "\n";

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  try {
    (*value) = uint32_t(std::stoull(str));
  } catch (const std::invalid_argument &e) {
    (void)e;
    PushError("Not an 64bit unsigned integer literal.\n");
    return false;
  } catch (const std::out_of_range &e) {
    (void)e;
    PushError("64bit unsigned integer value out of range.\n");
    return false;
  }
  return true;
#else
  // use jsteemann/atoi
  int retcode = 0;
  const char* start = str.c_str();
  const char* end = str.c_str() + str.size();
  auto result = jsteemann::atoi<uint32_t>(
      start, end, retcode);
  DCOUT("sz = " << str.size());
  DCOUT("ss = " << str << ", retcode = " << retcode
                << ", result = " << result);
  if (retcode == jsteemann::SUCCESS) {
    (*value) = result;
    return true;
  } else if (retcode == jsteemann::INVALID_INPUT) {
    PushError("Not an 32bit unsigned integer literal.\n");
    return false;
  } else if (retcode == jsteemann::INVALID_NEGATIVE_SIGN) {
    PushError("Negative sign `-` specified for uint32 integer.\n");
    return false;
  } else if (retcode == jsteemann::VALUE_OVERFLOW) {
    PushError("Integer value overflows.\n");
    return false;
  }

  PushError("Invalid integer literal\n");
  return false;
#endif
}

bool AsciiParser::ReadBasicType(int64_t *value) {
  std::stringstream ss;

  // head character
  bool has_sign = false;
  bool negative = false;
  {
    char sc;
    if (!Char1(&sc)) {
      return false;
    }
    _curr_cursor.col++;

    // sign or [0-9]
    if (sc == '+') {
      negative = false;
      has_sign = true;
    } else if (sc == '-') {
      negative = true;
      has_sign = true;
    } else if ((sc >= '0') && (sc <= '9')) {
      // ok
    } else {
      PushError("Sign or 0-9 expected, but got '" + std::to_string(sc) +
                "'.\n");
      return false;
    }

    ss << sc;
  }

  if (negative) {
    PushError("Unsigned value expected but got '-' sign.");
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if ((c >= '0') && (c <= '9')) {
      ss << c;
    } else {
      _sr->seek_from_current(-1);
      break;
    }
  }

  std::string str = ss.str();

  if (has_sign && (str.size() == 1)) {
    // sign only
    PushError("Integer value expected but got sign character only.\n");
    return false;
  }

  if ((str.size() > 1) && (str[0] == '0')) {
    PushError("Zero padded integer value is not allowed.\n");
    return false;
  }

  // std::cout << "ReadInt token: " << ss.str() << "\n";

  // TODO(syoyo): Use ryu parse.
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  try {
    (*value) = std::stoull(str);
  } catch (const std::invalid_argument &e) {
    (void)e;
    PushError("Not an 64bit unsigned integer literal.\n");
    return false;
  } catch (const std::out_of_range &e) {
    (void)e;
    PushError("64bit unsigned integer value out of range.\n");
    return false;
  }

  return true;
#else
  // use jsteemann/atoi
  int retcode;
  const char* start = str.c_str();
  const char* end = str.c_str() + str.size();
  auto result = jsteemann::atoi<int64_t>(
      start, end, retcode);
  if (retcode == jsteemann::SUCCESS) {
    (*value) = result;
    return true;
  } else if (retcode == jsteemann::INVALID_INPUT) {
    PushError("Not an 32bit unsigned integer literal.\n");
    return false;
  } else if (retcode == jsteemann::INVALID_NEGATIVE_SIGN) {
    PushError("Negative sign `-` specified for uint32 integer.\n");
    return false;
  } else if (retcode == jsteemann::VALUE_OVERFLOW) {
    PushError("Integer value overflows.\n");
    return false;
  }

  PushError("Invalid integer literal\n");
  return false;
#endif

  // std::cout << "read int ok\n";
}

bool AsciiParser::ReadBasicType(uint64_t *value) {
  std::stringstream ss;

  // head character
  bool has_sign = false;
  bool negative = false;
  {
    char sc;
    if (!Char1(&sc)) {
      return false;
    }
    _curr_cursor.col++;

    // sign or [0-9]
    if (sc == '+') {
      negative = false;
      has_sign = true;
    } else if (sc == '-') {
      negative = true;
      has_sign = true;
    } else if ((sc >= '0') && (sc <= '9')) {
      // ok
    } else {
      PushError("Sign or 0-9 expected, but got '" + std::to_string(sc) +
                "'.\n");
      return false;
    }

    ss << sc;
  }

  if (negative) {
    PushError("Unsigned value expected but got '-' sign.");
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if ((c >= '0') && (c <= '9')) {
      ss << c;
    } else {
      _sr->seek_from_current(-1);
      break;
    }
  }

  std::string str = ss.str();

  if (has_sign && (str.size() == 1)) {
    // sign only
    PushError("Integer value expected but got sign character only.\n");
    return false;
  }

  if ((str.size() > 1) && (str[0] == '0')) {
    PushError("Zero padded integer value is not allowed.\n");
    return false;
  }

  // std::cout << "ReadInt token: " << ss.str() << "\n";

  // TODO(syoyo): Use ryu parse.
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  try {
    (*value) = std::stoull(str);
  } catch (const std::invalid_argument &e) {
    (void)e;
    PushError("Not an 64bit unsigned integer literal.\n");
    return false;
  } catch (const std::out_of_range &e) {
    (void)e;
    PushError("64bit unsigned integer value out of range.\n");
    return false;
  }

  return true;
#else
  // use jsteemann/atoi
  int retcode;
  const char* start = str.c_str();
  const char* end = str.c_str() + str.size();
  auto result = jsteemann::atoi<uint64_t>(
      start, end, retcode);
  if (retcode == jsteemann::SUCCESS) {
    (*value) = result;
    return true;
  } else if (retcode == jsteemann::INVALID_INPUT) {
    PushError("Not an 32bit unsigned integer literal.\n");
    return false;
  } else if (retcode == jsteemann::INVALID_NEGATIVE_SIGN) {
    PushError("Negative sign `-` specified for uint32 integer.\n");
    return false;
  } else if (retcode == jsteemann::VALUE_OVERFLOW) {
    PushError("Integer value overflows.\n");
    return false;
  }

  PushError("Invalid integer literal\n");
  return false;
#endif

  // std::cout << "read int ok\n";
}

bool AsciiParser::ReadBasicType(value::float2 *value) {
  return ParseBasicTypeTuple(value);
}

bool AsciiParser::ReadBasicType(value::float3 *value) {
  return ParseBasicTypeTuple(value);
}

bool AsciiParser::ReadBasicType(value::point3f *value) {
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::normal3f *value) {
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::vector3h *value) {
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = value::float_to_half_full(v[0]);
    value->y = value::float_to_half_full(v[1]);
    value->z = value::float_to_half_full(v[2]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::vector3f *value) {
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::vector3d *value) {
  value::double3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::float4 *value) {
  return ParseBasicTypeTuple(value);
}

bool AsciiParser::ReadBasicType(value::double2 *value) {
  return ParseBasicTypeTuple(value);
}

bool AsciiParser::ReadBasicType(value::double3 *value) {
  return ParseBasicTypeTuple(value);
}

bool AsciiParser::ReadBasicType(value::point3h *value) {
  // parse as float3
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = value::float_to_half_full(v[0]);
    value->y = value::float_to_half_full(v[1]);
    value->z = value::float_to_half_full(v[2]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::point3d *value) {
  value::double3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::color3h *value) {
  // parse as float3
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = value::float_to_half_full(v[0]);
    value->g = value::float_to_half_full(v[1]);
    value->b = value::float_to_half_full(v[2]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::color3f *value) {
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = v[0];
    value->g = v[1];
    value->b = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::color3d *value) {
  value::double3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = v[0];
    value->g = v[1];
    value->b = v[2];
    return true;
  }
  return false;
}

#if 0
template <>
bool AsciiParser::ReadBasicType(value::point4h *value) {
  // parse as float4
  value::float4 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = value::float_to_half_full(v[0]);
    value->y = value::float_to_half_full(v[1]);
    value->z = value::float_to_half_full(v[2]);
    value->w = value::float_to_half_full(v[3]);
    return true;
  }
  return false;
}
#endif

bool AsciiParser::ReadBasicType(value::color4h *value) {
  // parse as float4
  value::float4 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = value::float_to_half_full(v[0]);
    value->g = value::float_to_half_full(v[1]);
    value->b = value::float_to_half_full(v[2]);
    value->a = value::float_to_half_full(v[3]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::color4f *value) {
  value::float4 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = v[0];
    value->g = v[1];
    value->b = v[2];
    value->a = v[3];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::color4d *value) {
  value::double4 v;
  if (ParseBasicTypeTuple(&v)) {
    value->r = v[0];
    value->g = v[1];
    value->b = v[2];
    value->a = v[3];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::normal3h *value) {
  // parse as float3
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = value::float_to_half_full(v[0]);
    value->y = value::float_to_half_full(v[1]);
    value->z = value::float_to_half_full(v[2]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::normal3d *value) {
  value::double3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->x = v[0];
    value->y = v[1];
    value->z = v[2];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(value::double4 *value) {
  return ParseBasicTypeTuple(value);
}

///
/// Parses 1 or more occurences of value with basic type 'T', separated by
/// `sep`
///
template <typename T>
bool AsciiParser::SepBy1BasicType(const char sep,
                                  std::vector<nonstd::optional<T>> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  {
    nonstd::optional<T> value;
    if (!ReadBasicType(&value)) {
      PushError("Not starting with the value of requested type.\n");
      return false;
    }

    result->push_back(value);
  }

  while (!Eof()) {
    // sep
    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    nonstd::optional<T> value;
    if (!ReadBasicType(&value)) {
      break;
    }

    result->push_back(value);
  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

///
/// Parses 1 or more occurences of value with basic type 'T', separated by
/// `sep`
///
template <typename T>
bool AsciiParser::SepBy1BasicType(const char sep, std::vector<T> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  {
    T value;
    if (!ReadBasicType(&value)) {
      PushError("Not starting with the value of requested type.\n");
      return false;
    }

    result->push_back(value);
  }

  while (!Eof()) {
    // sep
    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    T value;
    if (!ReadBasicType(&value)) {
      break;
    }

    result->push_back(value);
  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

///
/// Parses 1 or more occurences of value with basic type 'T', separated by
/// `sep`.
/// Allow `sep` character in the last item of the array.
///
template <typename T>
bool AsciiParser::SepBy1BasicType(const char sep, const char end_symbol, std::vector<T> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  {
    T value;
    if (!ReadBasicType(&value)) {
      PushError("Not starting with the value of requested type.\n");
      return false;
    }

    result->push_back(value);
  }

  while (!Eof()) {
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    // sep
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == sep) {
      // Look next token
      if (!SkipCommentAndWhitespaceAndNewline()) {
        return false;
      }

      char nc;
      if (!LookChar1(&nc)) {
        return false;
      }

      if (nc == end_symbol) {
        // end
        break;
      }
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    T value;
    if (!ReadBasicType(&value)) {
      break;
    }

    result->push_back(value);


  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

///
/// Parses 1 or more occurences of value with tuple type 'T', separated by
/// `sep`
///
template <typename T, size_t N>
bool AsciiParser::SepBy1TupleType(
    const char sep, std::vector<nonstd::optional<std::array<T, N>>> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  if (MaybeNone()) {
    result->push_back(nonstd::nullopt);
  } else {
    std::array<T, N> value;
    if (!ParseBasicTypeTuple<T, N>(&value)) {
      PushError("Not starting with the tuple value of requested type.\n");
      return false;
    }

    result->push_back(value);
  }

  while (!Eof()) {
    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    if (MaybeNone()) {
      result->push_back(nonstd::nullopt);
    } else {
      std::array<T, N> value;
      if (!ParseBasicTypeTuple<T, N>(&value)) {
        break;
      }
      result->push_back(value);
    }
  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

///
/// Parses 1 or more occurences of value with tuple type 'T', separated by
/// `sep`
///
template <typename T, size_t N>
bool AsciiParser::SepBy1TupleType(const char sep,
                                  std::vector<std::array<T, N>> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  {
    std::array<T, N> value;
    if (!ParseBasicTypeTuple<T, N>(&value)) {
      PushError("Not starting with the tuple value of requested type.\n");
      return false;
    }

    result->push_back(value);
  }

  while (!Eof()) {
    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    std::array<T, N> value;
    if (!ParseBasicTypeTuple<T, N>(&value)) {
      break;
    }

    result->push_back(value);
  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

///
/// Parse '[', Sep1By(','), ']'
///
template <typename T>
bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<T>> *result) {
  if (!Expect('[')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  // Empty array?
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == ']') {
      result->clear();
      return true;
    }

    Rewind(1);
  }

  if (!SepBy1BasicType<T>(',', ']', result)) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  if (!Expect(']')) {
    return false;
  }

  return true;
}

///
/// Parse '[', Sep1By(','), ']'
///
template <typename T>
bool AsciiParser::ParseBasicTypeArray(std::vector<T> *result) {
  if (!Expect('[')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  // Empty array?
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == ']') {
      result->clear();
      return true;
    }

    Rewind(1);
  }

  if (!SepBy1BasicType<T>(',', ']', result)) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  if (!Expect(']')) {
    return false;
  }
  return true;
}

///
/// Parses 1 or more occurences of asset references, separated by
/// `sep`
/// TODO: Parse LayerOffset: e.g. `(offset = 10; scale = 2)`
///
template <>
bool AsciiParser::SepBy1BasicType(const char sep,
                                  const char end_symbol,
                                  std::vector<Reference> *result) {
  result->clear();

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  {
    Reference ref;
    bool triple_deliminated{false};

    if (!ParseReference(&ref, &triple_deliminated)) {
      PushError("Failed to parse Reference.\n");
      return false;
    }

    (void)triple_deliminated;

    result->push_back(ref);
  }

  while (!Eof()) {
    // sep
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == sep) {
      // Look next token
      if (!SkipCommentAndWhitespaceAndNewline()) {
        return false;
      }

      char nc;
      if (!LookChar1(&nc)) {
        return false;
      }

      if (nc == end_symbol) {
        // end
        break;
      }
    }

    if (c != sep) {
      // end
      _sr->seek_from_current(-1);  // unwind single char
      break;
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    Reference ref;
    bool triple_deliminated{false};
    if (!ParseReference(&ref, &triple_deliminated)) {
      break;
    }

    (void)triple_deliminated;
    result->push_back(ref);
  }

  if (result->empty()) {
    PushError("Empty array.\n");
    return false;
  }

  return true;
}

bool AsciiParser::ParsePurpose(Purpose *result) {
  if (!result) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  std::string str;
  if (!ReadIdentifier(&str)) {
    return false;
  }

  if (str == "\"default\"") {
    (*result) = Purpose::Default;
  } else if (str == "\"render\"") {
    (*result) = Purpose::Render;
  } else if (str == "\"proxy\"") {
    (*result) = Purpose::Proxy;
  } else if (str == "\"guide\"") {
    (*result) = Purpose::Guide;
  } else {
    PUSH_ERROR_AND_RETURN_TAG(kAscii, "Invalid purpose value: " + str + "\n");
  }

  return true;
}

template <typename T, size_t N>
bool AsciiParser::ParseBasicTypeTuple(std::array<T, N> *result) {
  if (!Expect('(')) {
    return false;
  }

  std::vector<T> values;
  if (!SepBy1BasicType<T>(',', &values)) {
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  if (values.size() != N) {
    std::string msg = "The number of tuple elements must be " +
                      std::to_string(N) + ", but got " +
                      std::to_string(values.size()) + "\n";
    PushError(msg);
    return false;
  }

  for (size_t i = 0; i < N; i++) {
    (*result)[i] = values[i];
  }

  return true;
}

template <typename T, size_t N>
bool AsciiParser::ParseBasicTypeTuple(
    nonstd::optional<std::array<T, N>> *result) {
  if (MaybeNone()) {
    (*result) = nonstd::nullopt;
    return true;
  }

  if (!Expect('(')) {
    return false;
  }

  std::vector<T> values;
  if (!SepBy1BasicType<T>(',', &values)) {
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  if (values.size() != N) {
    PUSH_ERROR_AND_RETURN("The number of tuple elements must be " +
                          std::to_string(N) + ", but got " +
                          std::to_string(values.size()));
  }

  std::array<T, N> ret;
  for (size_t i = 0; i < N; i++) {
    ret[i] = values[i];
  }

  (*result) = ret;

  return true;
}

///
/// Parse array of asset references
/// Allow non-list version
///
template<>
bool AsciiParser::ParseBasicTypeArray(std::vector<Reference> *result) {
  if (!SkipWhitespace()) {
    return false;
  }

  char c;
  if (!Char1(&c)) {
    return false;
  }

  if (c != '[') {
    Rewind(1);

    // Guess non-list version
    Reference ref;
    bool triple_deliminated{false};
    if (!ParseReference(&ref, &triple_deliminated)) {
      return false;
    }

    (void)triple_deliminated;
    result->clear();
    result->push_back(ref);

  } else {

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    // Empty array?
    {
      char ce;
      if (!Char1(&ce)) {
        return false;
      }

      if (ce == ']') {
        result->clear();
        return true;
      }

      Rewind(1);
    }


    if (!SepBy1BasicType<Reference>(',', ']', result)) {
      return false;
    }
    DCOUT("parsed ref array");

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    if (!Expect(']')) {
      return false;
    }

  }

  return true;
}

///
/// Parse array of asset payload
/// Allow non-list version
///
template<>
bool AsciiParser::ParseBasicTypeArray(std::vector<Payload> *result) {
  if (!SkipWhitespace()) {
    return false;
  }

  char c;
  if (!Char1(&c)) {
    return false;
  }

  if (c != '[') {
    Rewind(1);

    DCOUT("Guess non-list version");
    // Guess non-list version
    Payload pl;
    bool triple_deliminated{false};
    if (!ParsePayload(&pl, &triple_deliminated)) {
      return false;
    }

    (void)triple_deliminated;
    result->clear();
    result->push_back(pl);

  } else {

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    // Empty array?
    {
      char ce;
      if (!Char1(&ce)) {
        return false;
      }

      if (ce == ']') {
        result->clear();
        return true;
      }

      Rewind(1);
    }

    if (!SepBy1BasicType(',', ']', result)) {
      return false;
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    if (!Expect(']')) {
      return false;
    }
  }

  return true;
}


///
/// Parse PathVector
///
template<>
bool AsciiParser::ParseBasicTypeArray(std::vector<Path> *result) {
  if (!SkipWhitespace()) {
    return false;
  }

  if (!Expect('[')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  // Empty array?
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == ']') {
      result->clear();
      return true;
    }

    Rewind(1);
  }

  if (!SepBy1BasicType(',', ']', result)) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  if (!Expect(']')) {
    return false;
  }

  return true;
}

template <typename T>
bool AsciiParser::MaybeNonFinite(T *out) {
  auto loc = CurrLoc();

  // "-inf", "inf" or "nan"
  std::vector<char> buf(4);
  if (!CharN(3, &buf)) {
    return false;
  }
  SeekTo(loc);

  if ((buf[0] == 'i') && (buf[1] == 'n') && (buf[2] == 'f')) {
    (*out) = std::numeric_limits<T>::infinity();
    return true;
  }

  if ((buf[0] == 'n') && (buf[1] == 'a') && (buf[2] == 'n')) {
    (*out) = std::numeric_limits<T>::quiet_NaN();
    return true;
  }

  bool ok = CharN(4, &buf);
  SeekTo(loc);

  if (ok) {
    if ((buf[0] == '-') && (buf[1] == 'i') && (buf[2] == 'n') &&
        (buf[3] == 'f')) {
      (*out) = -std::numeric_limits<T>::infinity();
      return true;
    }

    // NOTE: support "-nan"?
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord2h *value) {
  // parse as float2
  value::float2 v;
  if (ParseBasicTypeTuple(&v)) {
    value->s = value::float_to_half_full(v[0]);
    value->t = value::float_to_half_full(v[1]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord2h> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord2h v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord2f *value) {
  value::float2 v;
  if (ParseBasicTypeTuple(&v)) {
    value->s = v[0];
    value->t = v[1];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord2f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord2f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord2d *value) {
  value::double2 v;
  if (ParseBasicTypeTuple(&v)) {
    value->s = v[0];
    value->t = v[1];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord2d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord2d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord3h *value) {
  // parse as float3
  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->s = value::float_to_half_full(v[0]);
    value->t = value::float_to_half_full(v[1]);
    value->r = value::float_to_half_full(v[2]);
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord3h> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord3h v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord3f *value) {

  value::float3 v;
  if (ParseBasicTypeTuple(&v)) {
    value->s = v[0];
    value->t = v[1];
    value->r = v[2];
    return true;
  }

  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::texcoord3d *value) {
  if (!Expect('(')) {
    return false;
  }

  std::vector<double> values;
  if (!SepBy1BasicType<double>(',', &values)) {
    return false;
  }

  if (!Expect(')')) {
    return false;
  }

  if (values.size() != 3) {
    std::string msg = "The number of tuple elements must be " +
                      std::to_string(3) + ", but got " +
                      std::to_string(values.size()) + "\n";
    PUSH_ERROR_AND_RETURN(msg);
  }

  value->s = values[0];
  value->t = values[1];
  value->r = values[2];

  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::texcoord3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::texcoord3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::float2> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::float2 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::float3> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::float3 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::float4> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::float4 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::double2> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::double2 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::double3> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::double3 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::double4> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::double4 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::point3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::point3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::point3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::point3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::normal3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::normal3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::normal3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::normal3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::vector3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::vector3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::vector3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::vector3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::color3f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::color3f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::color4f> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::color4f v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::color3d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::color3d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::color4d> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::color4d v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<int> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  int v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<uint32_t> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  uint32_t v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<int64_t> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  int64_t v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<uint64_t> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  uint64_t v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(float *value) {
  // -inf, inf, nan
  {
    float v;
    if (MaybeNonFinite(&v)) {
      (*value) = v;
      return true;
    }
  }

  std::string value_str;
  if (!LexFloat(&value_str)) {
    PUSH_ERROR_AND_RETURN("Failed to lex floating value literal.");
  }

  auto flt = ParseFloat(value_str);
  if (flt) {
    (*value) = flt.value();
  } else {
    PUSH_ERROR_AND_RETURN("Failed to parse floating value.");
  }

  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<float> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  float v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(double *value) {
  // -inf, inf, nan
  {
    double v;
    if (MaybeNonFinite(&v)) {
      (*value) = v;
      return true;
    }
  }

  std::string value_str;
  if (!LexFloat(&value_str)) {
    PUSH_ERROR_AND_RETURN("Failed to lex floating value literal.");
  }

  auto flt = ParseDouble(value_str);
  if (!flt) {
    PUSH_ERROR_AND_RETURN("Failed to parse floating value.");
  } else {
    (*value) = flt.value();
  }

  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<double> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  double v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::half *value) {
  // Parse as float
  float v;
  if (!ReadBasicType(&v)) {
    return false;
  }

  (*value) = value::float_to_half_full(v);
  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::half> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::half v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::half2 *value) {
  // Parse as float
  value::float2 v;
  if (!ReadBasicType(&v)) {
    return false;
  }

  (*value)[0] = value::float_to_half_full(v[0]);
  (*value)[1] = value::float_to_half_full(v[1]);
  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::half2> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::half2 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::half3 *value) {
  // Parse as float
  value::float3 v;
  if (!ReadBasicType(&v)) {
    return false;
  }

  (*value)[0] = value::float_to_half_full(v[0]);
  (*value)[1] = value::float_to_half_full(v[1]);
  (*value)[2] = value::float_to_half_full(v[2]);
  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::half3> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::half3 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::half4 *value) {
  // Parse as float
  value::float4 v;
  if (!ReadBasicType(&v)) {
    return false;
  }

  (*value)[0] = value::float_to_half_full(v[0]);
  (*value)[1] = value::float_to_half_full(v[1]);
  (*value)[2] = value::float_to_half_full(v[2]);
  (*value)[3] = value::float_to_half_full(v[3]);
  return true;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::half4> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::half4 v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::quath *value) {
  value::half4 v;
  if (ReadBasicType(&v)) {
    value->real = v[0];
    value->imag[0] = v[1];
    value->imag[1] = v[2];
    value->imag[2] = v[3];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::quath> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::quath v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::quatf *value) {
  value::float4 v;
  if (ReadBasicType(&v)) {
    value->real = v[0];
    value->imag[0] = v[1];
    value->imag[1] = v[2];
    value->imag[2] = v[3];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::quatf> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::quatf v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::quatd *value) {
  value::double4 v;
  if (ReadBasicType(&v)) {
    value->real = v[0];
    value->imag[0] = v[1];
    value->imag[1] = v[2];
    value->imag[2] = v[3];
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::quatd> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::quatd v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(value::AssetPath *value) {
  bool triple_deliminated;
  if (ParseAssetIdentifier(value, &triple_deliminated)) {
    return true;
  }
  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<value::AssetPath> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  value::AssetPath v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(Reference *value) {
  bool triple_deliminated;
  if (ParseReference(value, &triple_deliminated)) {
    return true;
  }
  (void)triple_deliminated;

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<Reference> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  Reference v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

bool AsciiParser::ReadBasicType(Payload *value) {
  bool triple_deliminated;
  if (ParsePayload(value, &triple_deliminated)) {
    return true;
  }
  (void)triple_deliminated;

  return false;
}

bool AsciiParser::ReadBasicType(nonstd::optional<Payload> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  Payload v;
  if (ReadBasicType(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

// 1D array
template <typename T>
bool AsciiParser::ReadBasicType(std::vector<T> *value) {
  return ParseBasicTypeArray(value);
}

template <typename T>
bool AsciiParser::ReadBasicType(nonstd::optional<std::vector<T>> *value) {
  if (MaybeNone()) {
    (*value) = nonstd::nullopt;
    return true;
  }

  std::vector<T> v;
  if (ParseBasicTypeArray(&v)) {
    (*value) = v;
    return true;
  }

  return false;
}

// -- end basic

//
// Explicit template instanciations
//

#if 0
//template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<bool>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<int32_t>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::int2>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<uint32_t>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<int64_t>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<uint64_t>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::half>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::half2>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::half3>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::half4>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<float>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::float2>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::float3>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::float4>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<double>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::double2>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::double3>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::double4>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord2h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord2f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord2d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord3h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord3f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::texcoord3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::point3h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::point3f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::point3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::normal3h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::normal3f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::normal3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::vector3h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::vector3f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::vector3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color3h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color3f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color4h>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color4f>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::color4d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::matrix2d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::matrix3d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::matrix4d>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::token>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::StringData>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<std::string>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<Reference>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<Path>> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<nonstd::optional<value::AssetPath>> *result);
#endif

template bool AsciiParser::ParseBasicTypeArray(std::vector<bool> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<int32_t> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::int2> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::int3> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::int4> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<uint32_t> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::uint2> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::uint3> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::uint4> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<int64_t> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<uint64_t> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half2> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half3> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half4> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<float> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float2> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float3> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float4> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<double> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double2> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double3> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double4> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4h> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix2f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix3f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix4f> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix2d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix3d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix4d> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quath> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quatf> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quatd> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::token> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::StringData> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<std::string> *result);
//template bool AsciiParser::ParseBasicTypeArray(std::vector<Reference> *result);
//template bool AsciiParser::ParseBasicTypeArray(std::vector<Path> *result);
template bool AsciiParser::ParseBasicTypeArray(std::vector<value::AssetPath> *result);


}  // namespace ascii
}  // namespace tinyusdz

#else  // TINYUSDZ_DISABLE_MODULE_USDA_READER

#endif  // TINYUSDZ_DISABLE_MODULE_USDA_READER
