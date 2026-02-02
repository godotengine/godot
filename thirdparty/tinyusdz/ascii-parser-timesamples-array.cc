// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// To deal with too many sections in generated .obj error(happens in MinGW and MSVC)
// Split ParseTimeSamples to two .cc files.
//
// TODO
// - [x] Rewrite code with less C++ template code.

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
#include "tiny-format.hh"

//
#if !defined(TINYUSDZ_DISABLE_MODULE_USDA_READER)

//

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// external

//#include "external/fast_float/include/fast_float/fast_float.h"
//#include "external/jsteemann/atoi.h"
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

#include "common-macros.inc"
#include "io-util.hh"
#include "pprinter.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "value-pprint.hh"
#include "value-types.hh"

namespace tinyusdz {

namespace ascii {

extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<bool> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<int32_t> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<uint32_t> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<int64_t> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<uint64_t> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half2> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half3> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::half4> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<float> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float2> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float3> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::float4> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<double> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double2> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double3> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::double4> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord2d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::texcoord3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::point3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::normal3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::vector3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4h> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::color4d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix2f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix3f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix4f> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix2d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix3d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::matrix4d> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quath> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quatf> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::quatd> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::token> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::StringData> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<std::string> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<Reference> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<Path> *result);
extern  template bool AsciiParser::ParseBasicTypeArray(std::vector<value::AssetPath> *result);

//
// -- impl ParseTimeSampleData
//

bool AsciiParser::ParseTimeSampleValueOfArrayType(const uint32_t type_id, value::Value *result) {

  if (!result) {
    return false;
  }

  if (MaybeNone()) {
    (*result) = value::ValueBlock();
    return true;
  }

  value::Value val;

#define PARSE_TYPE(__tyid, __type)                       \
  if (__tyid == value::TypeTraits<__type>::type_id()) {             \
    std::vector<__type> typed_val; \
    if (!ParseBasicTypeArray(&typed_val)) {                             \
      PUSH_ERROR_AND_RETURN("Failed to parse value with requested type `" + value::GetTypeName(__tyid) + "[]`"); \
    }                                                                  \
    val = value::Value(typed_val); \
  } else

  // NOTE: `string` does not support multi-line string.
  PARSE_TYPE(type_id, value::AssetPath)
  PARSE_TYPE(type_id, value::token)
  PARSE_TYPE(type_id, std::string)
  PARSE_TYPE(type_id, float)
  PARSE_TYPE(type_id, int32_t)
  PARSE_TYPE(type_id, uint32_t)
  PARSE_TYPE(type_id, int64_t)
  PARSE_TYPE(type_id, uint64_t)
  PARSE_TYPE(type_id, value::half)
  PARSE_TYPE(type_id, value::half2)
  PARSE_TYPE(type_id, value::half3)
  PARSE_TYPE(type_id, value::half4)
  PARSE_TYPE(type_id, float)
  PARSE_TYPE(type_id, value::float2)
  PARSE_TYPE(type_id, value::float3)
  PARSE_TYPE(type_id, value::float4)
  PARSE_TYPE(type_id, double)
  PARSE_TYPE(type_id, value::double2)
  PARSE_TYPE(type_id, value::double3)
  PARSE_TYPE(type_id, value::double4)
  PARSE_TYPE(type_id, value::quath)
  PARSE_TYPE(type_id, value::quatf)
  PARSE_TYPE(type_id, value::quatd)
  PARSE_TYPE(type_id, value::color3f)
  PARSE_TYPE(type_id, value::color4f)
  PARSE_TYPE(type_id, value::color3d)
  PARSE_TYPE(type_id, value::color4d)
  PARSE_TYPE(type_id, value::vector3f)
  PARSE_TYPE(type_id, value::normal3f)
  PARSE_TYPE(type_id, value::point3f)
  PARSE_TYPE(type_id, value::texcoord2f)
  PARSE_TYPE(type_id, value::texcoord3f)
  PARSE_TYPE(type_id, value::matrix2f)
  PARSE_TYPE(type_id, value::matrix3f)
  PARSE_TYPE(type_id, value::matrix4f)
  PARSE_TYPE(type_id, value::matrix2d)
  PARSE_TYPE(type_id, value::matrix3d)
  PARSE_TYPE(type_id, value::matrix4d) {
    PUSH_ERROR_AND_RETURN(" : TODO: timeSamples type " + value::GetTypeName(type_id));
  }

#undef PARSE_TYPE

  (*result) = val;

  return true;

}

// `type_name` does not contain "[]"
bool AsciiParser::ParseTimeSampleValueOfArrayType(const std::string &type_name, value::Value *result) {
  nonstd::optional<uint32_t> type_id = value::TryGetTypeId(type_name);
  if (!type_id) {
    PUSH_ERROR_AND_RETURN("Unsupported/invalid type name: " + type_name);
  }

  return ParseTimeSampleValueOfArrayType(type_id.value(), result);
}

bool AsciiParser::ParseTimeSamplesOfArray(const std::string &type_name,
                                   value::TimeSamples *ts_out) {

  value::TimeSamples ts;

  if (!Expect('{')) {
    return false;
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '}') {
      break;
    }

    Rewind(1);

    double timeVal;
    // -inf, inf and nan are handled.
    if (!ReadBasicType(&timeVal)) {
      PushError("Parse time value failed.");
      return false;
    }

    if (!SkipWhitespace()) {
      return false;
    }

    if (!Expect(':')) {
      return false;
    }

    if (!SkipWhitespace()) {
      return false;
    }

    value::Value value;
    if (!ParseTimeSampleValueOfArrayType(type_name, &value)) { // could be None(ValueBlock)
      return false;
    }

    // The last element may have separator ','
    {
      // Semicolon ';' is not allowed as a separator for timeSamples array
      // values.
      if (!SkipWhitespace()) {
        return false;
      }

      char sep{};
      if (!Char1(&sep)) {
        return false;
      }

      DCOUT("sep = " << sep);
      if (sep == '}') {
        // End of item
        ts.add_sample(timeVal, value);
        break;
      } else if (sep == ',') {
        // ok
      } else {
        Rewind(1);

        // Look ahead Newline + '}'
        auto loc = CurrLoc();

        if (SkipWhitespaceAndNewline()) {
          char nc;
          if (!Char1(&nc)) {
            return false;
          }

          if (nc == '}') {
            // End of item
            ts.add_sample(timeVal, value);
            break;
          }
        }

        // Rewind and continue parsing.
        SeekTo(loc);
      }
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    ts.add_sample(timeVal, value);
  }

  DCOUT("Parse TimeSamples success. # of items = " << ts.size());

  if (ts_out) {
    (*ts_out) = std::move(ts);
  }

  return true;
}

}  // namespace ascii
}  // namespace tinyusdz

#else  // TINYUSDZ_DISABLE_MODULE_USDA_READER

#endif  // TINYUSDZ_DISABLE_MODULE_USDA_READER
