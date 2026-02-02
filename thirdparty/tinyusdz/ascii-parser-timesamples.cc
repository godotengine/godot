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

#include "ascii-parser.hh"
#include "str-util.hh"
#include "tiny-format.hh"
//
#include "io-util.hh"
#include "pprinter.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "value-pprint.hh"
#include "value-types.hh"
//
#include "common-macros.inc"

namespace tinyusdz {

namespace ascii {

bool AsciiParser::ParseTimeSampleValue(const uint32_t type_id, value::Value *result) {

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
    __type typed_val; \
    if (!ReadBasicType(&typed_val)) {                             \
      PUSH_ERROR_AND_RETURN("Failed to parse value with requested type `" + value::GetTypeName(__tyid) + "`"); \
    }                                                                  \
    val = value::Value(typed_val); \
  } else

  // NOTE: `string` does not support multi-line string.
  PARSE_TYPE(type_id, value::AssetPath)
  PARSE_TYPE(type_id, value::token)
  PARSE_TYPE(type_id, std::string)
  PARSE_TYPE(type_id, float)
  PARSE_TYPE(type_id, int32_t)
  PARSE_TYPE(type_id, value::int2)
  PARSE_TYPE(type_id, value::int3)
  PARSE_TYPE(type_id, value::int4)
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

bool AsciiParser::ParseTimeSampleValue(const std::string &type_name, value::Value *result) {

  nonstd::optional<uint32_t> type_id = value::TryGetTypeId(type_name);

  if (!type_id) {
    PUSH_ERROR_AND_RETURN("Unsupported/invalid timeSamples type " + type_name);
  }

  return ParseTimeSampleValue(type_id.value(), result);
}


bool AsciiParser::ParseTimeSamples(const std::string &type_name,
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
    if (!ParseTimeSampleValue(type_name, &value)) { // could be None(ValueBlock)
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
