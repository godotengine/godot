// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Random testing of regular expression matching.

#include <stdio.h>
#include <string>
#include <vector>

#include "util/test.h"
#include "re2/testing/exhaustive_tester.h"

DEFINE_int32(regexpseed, 404, "Random regexp seed.");
DEFINE_int32(regexpcount, 100, "How many random regexps to generate.");
DEFINE_int32(stringseed, 200, "Random string seed.");
DEFINE_int32(stringcount, 100, "How many random strings to generate.");

namespace re2 {

// Runs a random test on the given parameters.
// (Always uses the same random seeds for reproducibility.
// Can give different seeds on command line.)
static void RandomTest(int maxatoms, int maxops,
                       const std::vector<std::string>& alphabet,
                       const std::vector<std::string>& ops,
                       int maxstrlen,
                       const std::vector<std::string>& stralphabet,
                       const std::string& wrapper) {
  // Limit to smaller test cases in debug mode,
  // because everything is so much slower.
  if (RE2_DEBUG_MODE) {
    maxatoms--;
    maxops--;
    maxstrlen /= 2;
  }

  ExhaustiveTester t(maxatoms, maxops, alphabet, ops,
                     maxstrlen, stralphabet, wrapper, "");
  t.RandomStrings(FLAGS_stringseed, FLAGS_stringcount);
  t.GenerateRandom(FLAGS_regexpseed, FLAGS_regexpcount);
  printf("%d regexps, %d tests, %d failures [%d/%d str]\n",
         t.regexps(), t.tests(), t.failures(), maxstrlen, (int)stralphabet.size());
  EXPECT_EQ(0, t.failures());
}

// Tests random small regexps involving literals and egrep operators.
TEST(Random, SmallEgrepLiterals) {
  RandomTest(5, 5, Explode("abc."), RegexpGenerator::EgrepOps(),
             15, Explode("abc"),
             "");
}

// Tests random bigger regexps involving literals and egrep operators.
TEST(Random, BigEgrepLiterals) {
  RandomTest(10, 10, Explode("abc."), RegexpGenerator::EgrepOps(),
             15, Explode("abc"),
             "");
}

// Tests random small regexps involving literals, capturing parens,
// and egrep operators.
TEST(Random, SmallEgrepCaptures) {
  RandomTest(5, 5, Split(" ", "a (b) ."), RegexpGenerator::EgrepOps(),
             15, Explode("abc"),
             "");
}

// Tests random bigger regexps involving literals, capturing parens,
// and egrep operators.
TEST(Random, BigEgrepCaptures) {
  RandomTest(10, 10, Split(" ", "a (b) ."), RegexpGenerator::EgrepOps(),
             15, Explode("abc"),
             "");
}

// Tests random large complicated expressions, using all the possible
// operators, some literals, some parenthesized literals, and predefined
// character classes like \d.  (Adding larger character classes would
// make for too many possibilities.)
TEST(Random, Complicated) {
  std::vector<std::string> ops = Split(" ",
    "%s%s %s|%s %s* %s*? %s+ %s+? %s? %s?? "
    "%s{0} %s{0,} %s{1} %s{1,} %s{0,1} %s{0,2} %s{1,2} "
    "%s{2} %s{2,} %s{3,4} %s{4,5}");

  // Use (?:\b) and (?:\B) instead of \b and \B,
  // because PCRE rejects \b* but accepts (?:\b)*.
  // Ditto ^ and $.
  std::vector<std::string> atoms = Split(" ",
    ". (?:^) (?:$) \\a \\f \\n \\r \\t \\v "
    "\\d \\D \\s \\S \\w \\W (?:\\b) (?:\\B) "
    "a (a) b c - \\\\");
  std::vector<std::string> alphabet = Explode("abc123\001\002\003\t\r\n\v\f\a");
  RandomTest(10, 10, atoms, ops, 20, alphabet, "");
}

}  // namespace re2

