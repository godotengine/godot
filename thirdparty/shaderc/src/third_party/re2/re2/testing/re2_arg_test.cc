// Copyright 2005 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This tests to make sure numbers are parsed from strings
// correctly.
// Todo: Expand the test to validate strings parsed to the other types
// supported by RE2::Arg class

#include <stdint.h>
#include <string.h>

#include "util/test.h"
#include "re2/re2.h"

namespace re2 {

struct SuccessTable {
  const char * value_string;
  int64_t value;
  bool success[6];
};

// Test boundary cases for different integral sizes.
// Specifically I want to make sure that values outside the boundries
// of an integral type will fail and that negative numbers will fail
// for unsigned types. The following table contains the boundaries for
// the various integral types and has entries for whether or not each
// type can contain the given value.
const SuccessTable kSuccessTable[] = {
// string       integer value     i16    u16    i32    u32    i64    u64
// 0 to 2^7-1
{ "0",          0,              { true,  true,  true,  true,  true,  true  }},
{ "127",        127,            { true,  true,  true,  true,  true,  true  }},

// -1 to -2^7
{ "-1",         -1,             { true,  false, true,  false, true,  false }},
{ "-128",       -128,           { true,  false, true,  false, true,  false }},

// 2^7 to 2^8-1
{ "128",        128,            { true,  true,  true,  true,  true,  true  }},
{ "255",        255,            { true,  true,  true,  true,  true,  true  }},

// 2^8 to 2^15-1
{ "256",        256,            { true,  true,  true,  true,  true,  true  }},
{ "32767",      32767,          { true,  true,  true,  true,  true,  true  }},

// -2^7-1 to -2^15
{ "-129",       -129,           { true,  false, true,  false, true,  false }},
{ "-32768",     -32768,         { true,  false, true,  false, true,  false }},

// 2^15 to 2^16-1
{ "32768",      32768,          { false, true,  true,  true,  true,  true  }},
{ "65535",      65535,          { false, true,  true,  true,  true,  true  }},

// 2^16 to 2^31-1
{ "65536",      65536,          { false, false, true,  true,  true,  true  }},
{ "2147483647", 2147483647,     { false, false, true,  true,  true,  true  }},

// -2^15-1 to -2^31
{ "-32769",     -32769,         { false, false, true,  false, true,  false }},
{ "-2147483648", static_cast<int64_t>(0xFFFFFFFF80000000LL),
  { false, false, true,  false, true,  false }},

// 2^31 to 2^32-1
{ "2147483648", 2147483648U,    { false, false, false, true,  true,  true  }},
{ "4294967295", 4294967295U,    { false, false, false, true,  true,  true  }},

// 2^32 to 2^63-1
{ "4294967296", 4294967296LL,   { false, false, false, false, true,  true  }},
{ "9223372036854775807",
  9223372036854775807LL,        { false, false, false, false, true,  true  }},

// -2^31-1 to -2^63
{ "-2147483649", -2147483649LL, { false, false, false, false, true,  false }},
{ "-9223372036854775808", static_cast<int64_t>(0x8000000000000000LL),
  { false, false, false, false, true,  false }},

// 2^63 to 2^64-1
{ "9223372036854775808", static_cast<int64_t>(9223372036854775808ULL),
  { false, false, false, false, false, true  }},
{ "18446744073709551615", static_cast<int64_t>(18446744073709551615ULL),
  { false, false, false, false, false, true  }},

// >= 2^64
{ "18446744073709551616", 0,    { false, false, false, false, false, false }},
};

const int kNumStrings = arraysize(kSuccessTable);

// It's ugly to use a macro, but we apparently can't use the EXPECT_EQ
// macro outside of a TEST block and this seems to be the only way to
// avoid code duplication.  I can also pull off a couple nice tricks
// using concatenation for the type I'm checking against.
#define PARSE_FOR_TYPE(type, column) {                                   \
  type r;                                                                \
  for (int i = 0; i < kNumStrings; ++i) {                                \
    RE2::Arg arg(&r);                                                    \
    const char* const p = kSuccessTable[i].value_string;                 \
    bool retval = arg.Parse(p, strlen(p));                               \
    bool success = kSuccessTable[i].success[column];                     \
    EXPECT_EQ(retval, success)                                           \
        << "Parsing '" << p << "' for type " #type " should return "     \
        << success;                                                      \
    if (success) {                                                       \
      EXPECT_EQ(r, (type)kSuccessTable[i].value);                        \
    }                                                                    \
  }                                                                      \
}

TEST(RE2ArgTest, Int16Test) {
  PARSE_FOR_TYPE(int16_t, 0);
}

TEST(RE2ArgTest, Uint16Test) {
  PARSE_FOR_TYPE(uint16_t, 1);
}

TEST(RE2ArgTest, Int32Test) {
  PARSE_FOR_TYPE(int32_t, 2);
}

TEST(RE2ArgTest, Uint32Test) {
  PARSE_FOR_TYPE(uint32_t, 3);
}

TEST(RE2ArgTest, Int64Test) {
  PARSE_FOR_TYPE(int64_t, 4);
}

TEST(RE2ArgTest, Uint64Test) {
  PARSE_FOR_TYPE(uint64_t, 5);
}

}  // namespace re2
