// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Exhaustive testing of regular expression matching.

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "util/test.h"
#include "util/utf.h"
#include "re2/testing/exhaustive_tester.h"

namespace re2 {

// Test simple character classes by themselves.
TEST(CharacterClasses, Exhaustive) {
  std::vector<std::string> atoms = Split(" ",
    "[a] [b] [ab] [^bc] [b-d] [^b-d] []a] [-a] [a-] [^-a] [a-b-c] a b .");
  ExhaustiveTest(2, 1, atoms, RegexpGenerator::EgrepOps(),
                 5, Explode("ab"), "", "");
}

// Test simple character classes inside a___b (for example, a[a]b).
TEST(CharacterClasses, ExhaustiveAB) {
  std::vector<std::string> atoms = Split(" ",
    "[a] [b] [ab] [^bc] [b-d] [^b-d] []a] [-a] [a-] [^-a] [a-b-c] a b .");
  ExhaustiveTest(2, 1, atoms, RegexpGenerator::EgrepOps(),
                 5, Explode("ab"), "a%sb", "");
}

// Returns UTF8 for Rune r
static std::string UTF8(Rune r) {
  char buf[UTFmax+1];
  buf[runetochar(buf, &r)] = 0;
  return std::string(buf);
}

// Returns a vector of "interesting" UTF8 characters.
// Unicode is now too big to just return all of them,
// so UTF8Characters return a set likely to be good test cases.
static const std::vector<std::string>& InterestingUTF8() {
  static bool init;
  static std::vector<std::string> v;

  if (init)
    return v;

  init = true;
  // All the Latin1 equivalents are interesting.
  for (int i = 1; i < 256; i++)
    v.push_back(UTF8(i));

  // After that, the codes near bit boundaries are
  // interesting, because they span byte sequence lengths.
  for (int j = 0; j < 8; j++)
    v.push_back(UTF8(256 + j));
  for (int i = 512; i < Runemax; i <<= 1)
    for (int j = -8; j < 8; j++)
      v.push_back(UTF8(i + j));

  // The codes near Runemax, including Runemax itself, are interesting.
  for (int j = -8; j <= 0; j++)
    v.push_back(UTF8(Runemax + j));

  return v;
}

// Test interesting UTF-8 characters against character classes.
TEST(InterestingUTF8, SingleOps) {
  std::vector<std::string> atoms = Split(" ",
    ". ^ $ \\a \\f \\n \\r \\t \\v \\d \\D \\s \\S \\w \\W \\b \\B "
    "[[:alnum:]] [[:alpha:]] [[:blank:]] [[:cntrl:]] [[:digit:]] "
    "[[:graph:]] [[:lower:]] [[:print:]] [[:punct:]] [[:space:]] "
    "[[:upper:]] [[:xdigit:]] [\\s\\S] [\\d\\D] [^\\w\\W] [^\\d\\D]");
  std::vector<std::string> ops;  // no ops
  ExhaustiveTest(1, 0, atoms, ops,
                 1, InterestingUTF8(), "", "");
}

// Test interesting UTF-8 characters against character classes,
// but wrap everything inside AB.
TEST(InterestingUTF8, AB) {
  std::vector<std::string> atoms = Split(" ",
    ". ^ $ \\a \\f \\n \\r \\t \\v \\d \\D \\s \\S \\w \\W \\b \\B "
    "[[:alnum:]] [[:alpha:]] [[:blank:]] [[:cntrl:]] [[:digit:]] "
    "[[:graph:]] [[:lower:]] [[:print:]] [[:punct:]] [[:space:]] "
    "[[:upper:]] [[:xdigit:]] [\\s\\S] [\\d\\D] [^\\w\\W] [^\\d\\D]");
  std::vector<std::string> ops;  // no ops
  std::vector<std::string> alpha = InterestingUTF8();
  for (size_t i = 0; i < alpha.size(); i++)
    alpha[i] = "a" + alpha[i] + "b";
  ExhaustiveTest(1, 0, atoms, ops,
                 1, alpha, "a%sb", "");
}

}  // namespace re2

