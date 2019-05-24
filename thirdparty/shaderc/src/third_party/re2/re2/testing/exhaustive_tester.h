// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_TESTING_EXHAUSTIVE_TESTER_H_
#define RE2_TESTING_EXHAUSTIVE_TESTER_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "util/util.h"
#include "re2/testing/regexp_generator.h"
#include "re2/testing/string_generator.h"

namespace re2 {

// Doing this simplifies the logic below.
#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(NDEBUG)
// We are in a debug build.
const bool RE2_DEBUG_MODE = true;
#elif __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || __has_feature(thread_sanitizer)
// Not a debug build, but still under sanitizers.
const bool RE2_DEBUG_MODE = true;
#else
const bool RE2_DEBUG_MODE = false;
#endif

// Exhaustive regular expression test: generate all regexps within parameters,
// then generate all strings of a given length over a given alphabet,
// then check that NFA, DFA, and PCRE agree about whether each regexp matches
// each possible string, and if so, where the match is.
//
// Can also be used in a "random" mode that generates a given number
// of random regexp and strings, allowing testing of larger expressions
// and inputs.
class ExhaustiveTester : public RegexpGenerator {
 public:
  ExhaustiveTester(int maxatoms,
                   int maxops,
                   const std::vector<std::string>& alphabet,
                   const std::vector<std::string>& ops,
                   int maxstrlen,
                   const std::vector<std::string>& stralphabet,
                   const std::string& wrapper,
                   const std::string& topwrapper)
    : RegexpGenerator(maxatoms, maxops, alphabet, ops),
      strgen_(maxstrlen, stralphabet),
      wrapper_(wrapper),
      topwrapper_(topwrapper),
      regexps_(0), tests_(0), failures_(0),
      randomstrings_(0), stringseed_(0), stringcount_(0)  { }

  int regexps()  { return regexps_; }
  int tests()    { return tests_; }
  int failures() { return failures_; }

  // Needed for RegexpGenerator interface.
  void HandleRegexp(const std::string& regexp);

  // Causes testing to generate random input strings.
  void RandomStrings(int32_t seed, int32_t count) {
    randomstrings_ = true;
    stringseed_ = seed;
    stringcount_ = count;
  }

 private:
  StringGenerator strgen_;
  std::string wrapper_;      // Regexp wrapper - either empty or has one %s.
  std::string topwrapper_;   // Regexp top-level wrapper.
  int regexps_;   // Number of HandleRegexp calls
  int tests_;     // Number of regexp tests.
  int failures_;  // Number of tests failed.

  bool randomstrings_;  // Whether to use random strings
  int32_t stringseed_;  // If so, the seed.
  int stringcount_;     // If so, how many to generate.

  ExhaustiveTester(const ExhaustiveTester&) = delete;
  ExhaustiveTester& operator=(const ExhaustiveTester&) = delete;
};

// Runs an exhaustive test on the given parameters.
void ExhaustiveTest(int maxatoms, int maxops,
                    const std::vector<std::string>& alphabet,
                    const std::vector<std::string>& ops,
                    int maxstrlen,
                    const std::vector<std::string>& stralphabet,
                    const std::string& wrapper,
                    const std::string& topwrapper);

// Runs an exhaustive test using the given parameters and
// the basic egrep operators.
void EgrepTest(int maxatoms, int maxops, const std::string& alphabet,
               int maxstrlen, const std::string& stralphabet,
               const std::string& wrapper);

}  // namespace re2

#endif  // RE2_TESTING_EXHAUSTIVE_TESTER_H_
