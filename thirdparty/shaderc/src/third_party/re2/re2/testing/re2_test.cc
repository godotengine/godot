// -*- coding: utf-8 -*-
// Copyright 2002-2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Test extractions for PartialMatch/Consume

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <map>
#include <string>
#include <utility>
#if !defined(_MSC_VER) && !defined(__CYGWIN__) && !defined(__MINGW32__)
#include <sys/mman.h>
#include <unistd.h>  /* for sysconf */
#endif

#include "util/test.h"
#include "util/logging.h"
#include "util/strutil.h"
#include "re2/re2.h"
#include "re2/regexp.h"

namespace re2 {

TEST(RE2, HexTests) {
#define ASSERT_HEX(type, value)                                         \
  do {                                                                  \
    type v;                                                             \
    ASSERT_TRUE(                                                        \
        RE2::FullMatch(#value, "([0-9a-fA-F]+)[uUlL]*", RE2::Hex(&v))); \
    ASSERT_EQ(v, 0x##value);                                            \
    ASSERT_TRUE(RE2::FullMatch("0x" #value, "([0-9a-fA-FxX]+)[uUlL]*",  \
                               RE2::CRadix(&v)));                       \
    ASSERT_EQ(v, 0x##value);                                            \
  } while (0)

  ASSERT_HEX(short,              2bad);
  ASSERT_HEX(unsigned short,     2badU);
  ASSERT_HEX(int,                dead);
  ASSERT_HEX(unsigned int,       deadU);
  ASSERT_HEX(long,               7eadbeefL);
  ASSERT_HEX(unsigned long,      deadbeefUL);
  ASSERT_HEX(long long,          12345678deadbeefLL);
  ASSERT_HEX(unsigned long long, cafebabedeadbeefULL);

#undef ASSERT_HEX
}

TEST(RE2, OctalTests) {
#define ASSERT_OCTAL(type, value)                                           \
  do {                                                                      \
    type v;                                                                 \
    ASSERT_TRUE(RE2::FullMatch(#value, "([0-7]+)[uUlL]*", RE2::Octal(&v))); \
    ASSERT_EQ(v, 0##value);                                                 \
    ASSERT_TRUE(RE2::FullMatch("0" #value, "([0-9a-fA-FxX]+)[uUlL]*",       \
                               RE2::CRadix(&v)));                           \
    ASSERT_EQ(v, 0##value);                                                 \
  } while (0)

  ASSERT_OCTAL(short,              77777);
  ASSERT_OCTAL(unsigned short,     177777U);
  ASSERT_OCTAL(int,                17777777777);
  ASSERT_OCTAL(unsigned int,       37777777777U);
  ASSERT_OCTAL(long,               17777777777L);
  ASSERT_OCTAL(unsigned long,      37777777777UL);
  ASSERT_OCTAL(long long,          777777777777777777777LL);
  ASSERT_OCTAL(unsigned long long, 1777777777777777777777ULL);

#undef ASSERT_OCTAL
}

TEST(RE2, DecimalTests) {
#define ASSERT_DECIMAL(type, value)                                            \
  do {                                                                         \
    type v;                                                                    \
    ASSERT_TRUE(RE2::FullMatch(#value, "(-?[0-9]+)[uUlL]*", &v));              \
    ASSERT_EQ(v, value);                                                       \
    ASSERT_TRUE(                                                               \
        RE2::FullMatch(#value, "(-?[0-9a-fA-FxX]+)[uUlL]*", RE2::CRadix(&v))); \
    ASSERT_EQ(v, value);                                                       \
  } while (0)

  ASSERT_DECIMAL(short,              -1);
  ASSERT_DECIMAL(unsigned short,     9999);
  ASSERT_DECIMAL(int,                -1000);
  ASSERT_DECIMAL(unsigned int,       12345U);
  ASSERT_DECIMAL(long,               -10000000L);
  ASSERT_DECIMAL(unsigned long,      3083324652U);
  ASSERT_DECIMAL(long long,          -100000000000000LL);
  ASSERT_DECIMAL(unsigned long long, 1234567890987654321ULL);

#undef ASSERT_DECIMAL
}

TEST(RE2, Replace) {
  struct ReplaceTest {
    const char *regexp;
    const char *rewrite;
    const char *original;
    const char *single;
    const char *global;
    int        greplace_count;
  };
  static const ReplaceTest tests[] = {
    { "(qu|[b-df-hj-np-tv-z]*)([a-z]+)",
      "\\2\\1ay",
      "the quick brown fox jumps over the lazy dogs.",
      "ethay quick brown fox jumps over the lazy dogs.",
      "ethay ickquay ownbray oxfay umpsjay overay ethay azylay ogsday.",
      9 },
    { "\\w+",
      "\\0-NOSPAM",
      "abcd.efghi@google.com",
      "abcd-NOSPAM.efghi@google.com",
      "abcd-NOSPAM.efghi-NOSPAM@google-NOSPAM.com-NOSPAM",
      4 },
    { "^",
      "(START)",
      "foo",
      "(START)foo",
      "(START)foo",
      1 },
    { "^",
      "(START)",
      "",
      "(START)",
      "(START)",
      1 },
    { "$",
      "(END)",
      "",
      "(END)",
      "(END)",
      1 },
    { "b",
      "bb",
      "ababababab",
      "abbabababab",
      "abbabbabbabbabb",
      5 },
    { "b",
      "bb",
      "bbbbbb",
      "bbbbbbb",
      "bbbbbbbbbbbb",
      6 },
    { "b+",
      "bb",
      "bbbbbb",
      "bb",
      "bb",
      1 },
    { "b*",
      "bb",
      "bbbbbb",
      "bb",
      "bb",
      1 },
    { "b*",
      "bb",
      "aaaaa",
      "bbaaaaa",
      "bbabbabbabbabbabb",
      6 },
    // Check newline handling
    { "a.*a",
      "(\\0)",
      "aba\naba",
      "(aba)\naba",
      "(aba)\n(aba)",
      2 },
    { "", NULL, NULL, NULL, NULL, 0 }
  };

  for (const ReplaceTest* t = tests; t->original != NULL; t++) {
    std::string one(t->original);
    ASSERT_TRUE(RE2::Replace(&one, t->regexp, t->rewrite));
    ASSERT_EQ(one, t->single);
    std::string all(t->original);
    ASSERT_EQ(RE2::GlobalReplace(&all, t->regexp, t->rewrite), t->greplace_count)
      << "Got: " << all;
    ASSERT_EQ(all, t->global);
  }
}

static void TestCheckRewriteString(const char* regexp, const char* rewrite,
                              bool expect_ok) {
  std::string error;
  RE2 exp(regexp);
  bool actual_ok = exp.CheckRewriteString(rewrite, &error);
  EXPECT_EQ(expect_ok, actual_ok) << " for " << rewrite << " error: " << error;
}

TEST(CheckRewriteString, all) {
  TestCheckRewriteString("abc", "foo", true);
  TestCheckRewriteString("abc", "foo\\", false);
  TestCheckRewriteString("abc", "foo\\0bar", true);

  TestCheckRewriteString("a(b)c", "foo", true);
  TestCheckRewriteString("a(b)c", "foo\\0bar", true);
  TestCheckRewriteString("a(b)c", "foo\\1bar", true);
  TestCheckRewriteString("a(b)c", "foo\\2bar", false);
  TestCheckRewriteString("a(b)c", "f\\\\2o\\1o", true);

  TestCheckRewriteString("a(b)(c)", "foo\\12", true);
  TestCheckRewriteString("a(b)(c)", "f\\2o\\1o", true);
  TestCheckRewriteString("a(b)(c)", "f\\oo\\1", false);
}

TEST(RE2, Extract) {
  std::string s;

  ASSERT_TRUE(RE2::Extract("boris@kremvax.ru", "(.*)@([^.]*)", "\\2!\\1", &s));
  ASSERT_EQ(s, "kremvax!boris");

  ASSERT_TRUE(RE2::Extract("foo", ".*", "'\\0'", &s));
  ASSERT_EQ(s, "'foo'");
  // check that false match doesn't overwrite
  ASSERT_FALSE(RE2::Extract("baz", "bar", "'\\0'", &s));
  ASSERT_EQ(s, "'foo'");
}

TEST(RE2, Consume) {
  RE2 r("\\s*(\\w+)");    // matches a word, possibly proceeded by whitespace
  std::string word;

  std::string s("   aaa b!@#$@#$cccc");
  StringPiece input(s);

  ASSERT_TRUE(RE2::Consume(&input, r, &word));
  ASSERT_EQ(word, "aaa") << " input: " << input;
  ASSERT_TRUE(RE2::Consume(&input, r, &word));
  ASSERT_EQ(word, "b") << " input: " << input;
  ASSERT_FALSE(RE2::Consume(&input, r, &word)) << " input: " << input;
}

TEST(RE2, ConsumeN) {
  const std::string s(" one two three 4");
  StringPiece input(s);

  RE2::Arg argv[2];
  const RE2::Arg* const args[2] = { &argv[0], &argv[1] };

  // 0 arg
  EXPECT_TRUE(RE2::ConsumeN(&input, "\\s*(\\w+)", args, 0));  // Skips "one".

  // 1 arg
  std::string word;
  argv[0] = &word;
  EXPECT_TRUE(RE2::ConsumeN(&input, "\\s*(\\w+)", args, 1));
  EXPECT_EQ("two", word);

  // Multi-args
  int n;
  argv[1] = &n;
  EXPECT_TRUE(RE2::ConsumeN(&input, "\\s*(\\w+)\\s*(\\d+)", args, 2));
  EXPECT_EQ("three", word);
  EXPECT_EQ(4, n);
}

TEST(RE2, FindAndConsume) {
  RE2 r("(\\w+)");      // matches a word
  std::string word;

  std::string s("   aaa b!@#$@#$cccc");
  StringPiece input(s);

  ASSERT_TRUE(RE2::FindAndConsume(&input, r, &word));
  ASSERT_EQ(word, "aaa");
  ASSERT_TRUE(RE2::FindAndConsume(&input, r, &word));
  ASSERT_EQ(word, "b");
  ASSERT_TRUE(RE2::FindAndConsume(&input, r, &word));
  ASSERT_EQ(word, "cccc");
  ASSERT_FALSE(RE2::FindAndConsume(&input, r, &word));

  // Check that FindAndConsume works without any submatches.
  // Earlier version used uninitialized data for
  // length to consume.
  input = "aaa";
  ASSERT_TRUE(RE2::FindAndConsume(&input, "aaa"));
  ASSERT_EQ(input, "");
}

TEST(RE2, FindAndConsumeN) {
  const std::string s(" one two three 4");
  StringPiece input(s);

  RE2::Arg argv[2];
  const RE2::Arg* const args[2] = { &argv[0], &argv[1] };

  // 0 arg
  EXPECT_TRUE(RE2::FindAndConsumeN(&input, "(\\w+)", args, 0));  // Skips "one".

  // 1 arg
  std::string word;
  argv[0] = &word;
  EXPECT_TRUE(RE2::FindAndConsumeN(&input, "(\\w+)", args, 1));
  EXPECT_EQ("two", word);

  // Multi-args
  int n;
  argv[1] = &n;
  EXPECT_TRUE(RE2::FindAndConsumeN(&input, "(\\w+)\\s*(\\d+)", args, 2));
  EXPECT_EQ("three", word);
  EXPECT_EQ(4, n);
}

TEST(RE2, MatchNumberPeculiarity) {
  RE2 r("(foo)|(bar)|(baz)");
  std::string word1;
  std::string word2;
  std::string word3;

  ASSERT_TRUE(RE2::PartialMatch("foo", r, &word1, &word2, &word3));
  ASSERT_EQ(word1, "foo");
  ASSERT_EQ(word2, "");
  ASSERT_EQ(word3, "");
  ASSERT_TRUE(RE2::PartialMatch("bar", r, &word1, &word2, &word3));
  ASSERT_EQ(word1, "");
  ASSERT_EQ(word2, "bar");
  ASSERT_EQ(word3, "");
  ASSERT_TRUE(RE2::PartialMatch("baz", r, &word1, &word2, &word3));
  ASSERT_EQ(word1, "");
  ASSERT_EQ(word2, "");
  ASSERT_EQ(word3, "baz");
  ASSERT_FALSE(RE2::PartialMatch("f", r, &word1, &word2, &word3));

  std::string a;
  ASSERT_TRUE(RE2::FullMatch("hello", "(foo)|hello", &a));
  ASSERT_EQ(a, "");
}

TEST(RE2, Match) {
  RE2 re("((\\w+):([0-9]+))");   // extracts host and port
  StringPiece group[4];

  // No match.
  StringPiece s = "zyzzyva";
  ASSERT_FALSE(
      re.Match(s, 0, s.size(), RE2::UNANCHORED, group, arraysize(group)));

  // Matches and extracts.
  s = "a chrisr:9000 here";
  ASSERT_TRUE(
      re.Match(s, 0, s.size(), RE2::UNANCHORED, group, arraysize(group)));
  ASSERT_EQ(group[0], "chrisr:9000");
  ASSERT_EQ(group[1], "chrisr:9000");
  ASSERT_EQ(group[2], "chrisr");
  ASSERT_EQ(group[3], "9000");

  std::string all, host;
  int port;
  ASSERT_TRUE(RE2::PartialMatch("a chrisr:9000 here", re, &all, &host, &port));
  ASSERT_EQ(all, "chrisr:9000");
  ASSERT_EQ(host, "chrisr");
  ASSERT_EQ(port, 9000);
}

static void TestRecursion(int size, const char* pattern) {
  // Fill up a string repeating the pattern given
  std::string domain;
  domain.resize(size);
  size_t patlen = strlen(pattern);
  for (int i = 0; i < size; i++) {
    domain[i] = pattern[i % patlen];
  }
  // Just make sure it doesn't crash due to too much recursion.
  RE2 re("([a-zA-Z0-9]|-)+(\\.([a-zA-Z0-9]|-)+)*(\\.)?", RE2::Quiet);
  RE2::FullMatch(domain, re);
}

// A meta-quoted string, interpreted as a pattern, should always match
// the original unquoted string.
static void TestQuoteMeta(const std::string& unquoted,
                          const RE2::Options& options = RE2::DefaultOptions) {
  std::string quoted = RE2::QuoteMeta(unquoted);
  RE2 re(quoted, options);
  EXPECT_TRUE(RE2::FullMatch(unquoted, re))
      << "Unquoted='" << unquoted << "', quoted='" << quoted << "'.";
}

// A meta-quoted string, interpreted as a pattern, should always match
// the original unquoted string.
static void NegativeTestQuoteMeta(
    const std::string& unquoted, const std::string& should_not_match,
    const RE2::Options& options = RE2::DefaultOptions) {
  std::string quoted = RE2::QuoteMeta(unquoted);
  RE2 re(quoted, options);
  EXPECT_FALSE(RE2::FullMatch(should_not_match, re))
      << "Unquoted='" << unquoted << "', quoted='" << quoted << "'.";
}

// Tests that quoted meta characters match their original strings,
// and that a few things that shouldn't match indeed do not.
TEST(QuoteMeta, Simple) {
  TestQuoteMeta("foo");
  TestQuoteMeta("foo.bar");
  TestQuoteMeta("foo\\.bar");
  TestQuoteMeta("[1-9]");
  TestQuoteMeta("1.5-2.0?");
  TestQuoteMeta("\\d");
  TestQuoteMeta("Who doesn't like ice cream?");
  TestQuoteMeta("((a|b)c?d*e+[f-h]i)");
  TestQuoteMeta("((?!)xxx).*yyy");
  TestQuoteMeta("([");
}
TEST(QuoteMeta, SimpleNegative) {
  NegativeTestQuoteMeta("foo", "bar");
  NegativeTestQuoteMeta("...", "bar");
  NegativeTestQuoteMeta("\\.", ".");
  NegativeTestQuoteMeta("\\.", "..");
  NegativeTestQuoteMeta("(a)", "a");
  NegativeTestQuoteMeta("(a|b)", "a");
  NegativeTestQuoteMeta("(a|b)", "(a)");
  NegativeTestQuoteMeta("(a|b)", "a|b");
  NegativeTestQuoteMeta("[0-9]", "0");
  NegativeTestQuoteMeta("[0-9]", "0-9");
  NegativeTestQuoteMeta("[0-9]", "[9]");
  NegativeTestQuoteMeta("((?!)xxx)", "xxx");
}

TEST(QuoteMeta, Latin1) {
  TestQuoteMeta("3\xb2 = 9", RE2::Latin1);
}

TEST(QuoteMeta, UTF8) {
  TestQuoteMeta("Plácido Domingo");
  TestQuoteMeta("xyz");  // No fancy utf8.
  TestQuoteMeta("\xc2\xb0");  // 2-byte utf8 -- a degree symbol.
  TestQuoteMeta("27\xc2\xb0 degrees");  // As a middle character.
  TestQuoteMeta("\xe2\x80\xb3");  // 3-byte utf8 -- a double prime.
  TestQuoteMeta("\xf0\x9d\x85\x9f");  // 4-byte utf8 -- a music note.
  TestQuoteMeta("27\xc2\xb0");  // Interpreted as Latin-1, this should
                                // still work.
  NegativeTestQuoteMeta("27\xc2\xb0",
                        "27\\\xc2\\\xb0");  // 2-byte utf8 -- a degree symbol.
}

TEST(QuoteMeta, HasNull) {
  std::string has_null;

  // string with one null character
  has_null += '\0';
  TestQuoteMeta(has_null);
  NegativeTestQuoteMeta(has_null, "");

  // Don't want null-followed-by-'1' to be interpreted as '\01'.
  has_null += '1';
  TestQuoteMeta(has_null);
  NegativeTestQuoteMeta(has_null, "\1");
}

TEST(ProgramSize, BigProgram) {
  RE2 re_simple("simple regexp");
  RE2 re_medium("medium.*regexp");
  RE2 re_complex("complex.{1,128}regexp");

  ASSERT_GT(re_simple.ProgramSize(), 0);
  ASSERT_GT(re_medium.ProgramSize(), re_simple.ProgramSize());
  ASSERT_GT(re_complex.ProgramSize(), re_medium.ProgramSize());

  ASSERT_GT(re_simple.ReverseProgramSize(), 0);
  ASSERT_GT(re_medium.ReverseProgramSize(), re_simple.ReverseProgramSize());
  ASSERT_GT(re_complex.ReverseProgramSize(), re_medium.ReverseProgramSize());
}

TEST(ProgramFanout, BigProgram) {
  RE2 re1("(?:(?:(?:(?:(?:.)?){1})*)+)");
  RE2 re10("(?:(?:(?:(?:(?:.)?){10})*)+)");
  RE2 re100("(?:(?:(?:(?:(?:.)?){100})*)+)");
  RE2 re1000("(?:(?:(?:(?:(?:.)?){1000})*)+)");

  std::map<int, int> histogram;

  // 3 is the largest non-empty bucket and has 1 element.
  ASSERT_EQ(3, re1.ProgramFanout(&histogram));
  ASSERT_EQ(1, histogram[3]);

  // 7 is the largest non-empty bucket and has 10 elements.
  ASSERT_EQ(7, re10.ProgramFanout(&histogram));
  ASSERT_EQ(10, histogram[7]);

  // 10 is the largest non-empty bucket and has 100 elements.
  ASSERT_EQ(10, re100.ProgramFanout(&histogram));
  ASSERT_EQ(100, histogram[10]);

  // 13 is the largest non-empty bucket and has 1000 elements.
  ASSERT_EQ(13, re1000.ProgramFanout(&histogram));
  ASSERT_EQ(1000, histogram[13]);

  // 2 is the largest non-empty bucket and has 3 elements.
  // This differs from the others due to how reverse `.' works.
  ASSERT_EQ(2, re1.ReverseProgramFanout(&histogram));
  ASSERT_EQ(3, histogram[2]);

  // 5 is the largest non-empty bucket and has 10 elements.
  ASSERT_EQ(5, re10.ReverseProgramFanout(&histogram));
  ASSERT_EQ(10, histogram[5]);

  // 9 is the largest non-empty bucket and has 100 elements.
  ASSERT_EQ(9, re100.ReverseProgramFanout(&histogram));
  ASSERT_EQ(100, histogram[9]);

  // 12 is the largest non-empty bucket and has 1000 elements.
  ASSERT_EQ(12, re1000.ReverseProgramFanout(&histogram));
  ASSERT_EQ(1000, histogram[12]);
}

// Issue 956519: handling empty character sets was
// causing NULL dereference.  This tests a few empty character sets.
// (The way to get an empty character set is to negate a full one.)
TEST(EmptyCharset, Fuzz) {
  static const char *empties[] = {
    "[^\\S\\s]",
    "[^\\S[:space:]]",
    "[^\\D\\d]",
    "[^\\D[:digit:]]"
  };
  for (int i = 0; i < arraysize(empties); i++)
    ASSERT_FALSE(RE2(empties[i]).Match("abc", 0, 3, RE2::UNANCHORED, NULL, 0));
}

// Bitstate assumes that kInstFail instructions in
// alternations or capture groups have been "compiled away".
TEST(EmptyCharset, BitstateAssumptions) {
  // Captures trigger use of Bitstate.
  static const char *nop_empties[] = {
    "((((()))))" "[^\\S\\s]?",
    "((((()))))" "([^\\S\\s])?",
    "((((()))))" "([^\\S\\s]|[^\\S\\s])?",
    "((((()))))" "(([^\\S\\s]|[^\\S\\s])|)"
  };
  StringPiece group[6];
  for (int i = 0; i < arraysize(nop_empties); i++)
    ASSERT_TRUE(RE2(nop_empties[i]).Match("", 0, 0, RE2::UNANCHORED, group, 6));
}

// Test that named groups work correctly.
TEST(Capture, NamedGroups) {
  {
    RE2 re("(hello world)");
    ASSERT_EQ(re.NumberOfCapturingGroups(), 1);
    const std::map<std::string, int>& m = re.NamedCapturingGroups();
    ASSERT_EQ(m.size(), 0);
  }

  {
    RE2 re("(?P<A>expr(?P<B>expr)(?P<C>expr))((expr)(?P<D>expr))");
    ASSERT_EQ(re.NumberOfCapturingGroups(), 6);
    const std::map<std::string, int>& m = re.NamedCapturingGroups();
    ASSERT_EQ(m.size(), 4);
    ASSERT_EQ(m.find("A")->second, 1);
    ASSERT_EQ(m.find("B")->second, 2);
    ASSERT_EQ(m.find("C")->second, 3);
    ASSERT_EQ(m.find("D")->second, 6);  // $4 and $5 are anonymous
  }
}

TEST(RE2, CapturedGroupTest) {
  RE2 re("directions from (?P<S>.*) to (?P<D>.*)");
  int num_groups = re.NumberOfCapturingGroups();
  EXPECT_EQ(2, num_groups);
  std::string args[4];
  RE2::Arg arg0(&args[0]);
  RE2::Arg arg1(&args[1]);
  RE2::Arg arg2(&args[2]);
  RE2::Arg arg3(&args[3]);

  const RE2::Arg* const matches[4] = {&arg0, &arg1, &arg2, &arg3};
  EXPECT_TRUE(RE2::FullMatchN("directions from mountain view to san jose",
                              re, matches, num_groups));
  const std::map<std::string, int>& named_groups = re.NamedCapturingGroups();
  EXPECT_TRUE(named_groups.find("S") != named_groups.end());
  EXPECT_TRUE(named_groups.find("D") != named_groups.end());

  // The named group index is 1-based.
  int source_group_index = named_groups.find("S")->second;
  int destination_group_index = named_groups.find("D")->second;
  EXPECT_EQ(1, source_group_index);
  EXPECT_EQ(2, destination_group_index);

  // The args is zero-based.
  EXPECT_EQ("mountain view", args[source_group_index - 1]);
  EXPECT_EQ("san jose", args[destination_group_index - 1]);
}

TEST(RE2, FullMatchWithNoArgs) {
  ASSERT_TRUE(RE2::FullMatch("h", "h"));
  ASSERT_TRUE(RE2::FullMatch("hello", "hello"));
  ASSERT_TRUE(RE2::FullMatch("hello", "h.*o"));
  ASSERT_FALSE(RE2::FullMatch("othello", "h.*o"));  // Must be anchored at front
  ASSERT_FALSE(RE2::FullMatch("hello!", "h.*o"));   // Must be anchored at end
}

TEST(RE2, PartialMatch) {
  ASSERT_TRUE(RE2::PartialMatch("x", "x"));
  ASSERT_TRUE(RE2::PartialMatch("hello", "h.*o"));
  ASSERT_TRUE(RE2::PartialMatch("othello", "h.*o"));
  ASSERT_TRUE(RE2::PartialMatch("hello!", "h.*o"));
  ASSERT_TRUE(RE2::PartialMatch("x", "((((((((((((((((((((x))))))))))))))))))))"));
}

TEST(RE2, PartialMatchN) {
  RE2::Arg argv[2];
  const RE2::Arg* const args[2] = { &argv[0], &argv[1] };

  // 0 arg
  EXPECT_TRUE(RE2::PartialMatchN("hello", "e.*o", args, 0));
  EXPECT_FALSE(RE2::PartialMatchN("othello", "a.*o", args, 0));

  // 1 arg
  int i;
  argv[0] = &i;
  EXPECT_TRUE(RE2::PartialMatchN("1001 nights", "(\\d+)", args, 1));
  EXPECT_EQ(1001, i);
  EXPECT_FALSE(RE2::PartialMatchN("three", "(\\d+)", args, 1));

  // Multi-arg
  std::string s;
  argv[1] = &s;
  EXPECT_TRUE(RE2::PartialMatchN("answer: 42:life", "(\\d+):(\\w+)", args, 2));
  EXPECT_EQ(42, i);
  EXPECT_EQ("life", s);
  EXPECT_FALSE(RE2::PartialMatchN("hi1", "(\\w+)(1)", args, 2));
}

TEST(RE2, FullMatchZeroArg) {
  // Zero-arg
  ASSERT_TRUE(RE2::FullMatch("1001", "\\d+"));
}

TEST(RE2, FullMatchOneArg) {
  int i;

  // Single-arg
  ASSERT_TRUE(RE2::FullMatch("1001", "(\\d+)",   &i));
  ASSERT_EQ(i, 1001);
  ASSERT_TRUE(RE2::FullMatch("-123", "(-?\\d+)", &i));
  ASSERT_EQ(i, -123);
  ASSERT_FALSE(RE2::FullMatch("10", "()\\d+", &i));
  ASSERT_FALSE(
      RE2::FullMatch("1234567890123456789012345678901234567890", "(\\d+)", &i));
}

TEST(RE2, FullMatchIntegerArg) {
  int i;

  // Digits surrounding integer-arg
  ASSERT_TRUE(RE2::FullMatch("1234", "1(\\d*)4", &i));
  ASSERT_EQ(i, 23);
  ASSERT_TRUE(RE2::FullMatch("1234", "(\\d)\\d+", &i));
  ASSERT_EQ(i, 1);
  ASSERT_TRUE(RE2::FullMatch("-1234", "(-\\d)\\d+", &i));
  ASSERT_EQ(i, -1);
  ASSERT_TRUE(RE2::PartialMatch("1234", "(\\d)", &i));
  ASSERT_EQ(i, 1);
  ASSERT_TRUE(RE2::PartialMatch("-1234", "(-\\d)", &i));
  ASSERT_EQ(i, -1);
}

TEST(RE2, FullMatchStringArg) {
  std::string s;
  // String-arg
  ASSERT_TRUE(RE2::FullMatch("hello", "h(.*)o", &s));
  ASSERT_EQ(s, std::string("ell"));
}

TEST(RE2, FullMatchStringPieceArg) {
  int i;
  // StringPiece-arg
  StringPiece sp;
  ASSERT_TRUE(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &sp, &i));
  ASSERT_EQ(sp.size(), 4);
  ASSERT_TRUE(memcmp(sp.data(), "ruby", 4) == 0);
  ASSERT_EQ(i, 1234);
}

TEST(RE2, FullMatchMultiArg) {
  int i;
  std::string s;
  // Multi-arg
  ASSERT_TRUE(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &s, &i));
  ASSERT_EQ(s, std::string("ruby"));
  ASSERT_EQ(i, 1234);
}

TEST(RE2, FullMatchN) {
  RE2::Arg argv[2];
  const RE2::Arg* const args[2] = { &argv[0], &argv[1] };

  // 0 arg
  EXPECT_TRUE(RE2::FullMatchN("hello", "h.*o", args, 0));
  EXPECT_FALSE(RE2::FullMatchN("othello", "h.*o", args, 0));

  // 1 arg
  int i;
  argv[0] = &i;
  EXPECT_TRUE(RE2::FullMatchN("1001", "(\\d+)", args, 1));
  EXPECT_EQ(1001, i);
  EXPECT_FALSE(RE2::FullMatchN("three", "(\\d+)", args, 1));

  // Multi-arg
  std::string s;
  argv[1] = &s;
  EXPECT_TRUE(RE2::FullMatchN("42:life", "(\\d+):(\\w+)", args, 2));
  EXPECT_EQ(42, i);
  EXPECT_EQ("life", s);
  EXPECT_FALSE(RE2::FullMatchN("hi1", "(\\w+)(1)", args, 2));
}

TEST(RE2, FullMatchIgnoredArg) {
  int i;
  std::string s;

  // Old-school NULL should be ignored.
  ASSERT_TRUE(
      RE2::FullMatch("ruby:1234", "(\\w+)(:)(\\d+)", &s, (void*)NULL, &i));
  ASSERT_EQ(s, std::string("ruby"));
  ASSERT_EQ(i, 1234);

  // C++11 nullptr should also be ignored.
  ASSERT_TRUE(RE2::FullMatch("rubz:1235", "(\\w+)(:)(\\d+)", &s, nullptr, &i));
  ASSERT_EQ(s, std::string("rubz"));
  ASSERT_EQ(i, 1235);
}

TEST(RE2, FullMatchTypedNullArg) {
  std::string s;

  // Ignore non-void* NULL arg
  ASSERT_TRUE(RE2::FullMatch("hello", "he(.*)lo", (char*)NULL));
  ASSERT_TRUE(RE2::FullMatch("hello", "h(.*)o", (std::string*)NULL));
  ASSERT_TRUE(RE2::FullMatch("hello", "h(.*)o", (StringPiece*)NULL));
  ASSERT_TRUE(RE2::FullMatch("1234", "(.*)", (int*)NULL));
  ASSERT_TRUE(RE2::FullMatch("1234567890123456", "(.*)", (long long*)NULL));
  ASSERT_TRUE(RE2::FullMatch("123.4567890123456", "(.*)", (double*)NULL));
  ASSERT_TRUE(RE2::FullMatch("123.4567890123456", "(.*)", (float*)NULL));

  // Fail on non-void* NULL arg if the match doesn't parse for the given type.
  ASSERT_FALSE(RE2::FullMatch("hello", "h(.*)lo", &s, (char*)NULL));
  ASSERT_FALSE(RE2::FullMatch("hello", "(.*)", (int*)NULL));
  ASSERT_FALSE(RE2::FullMatch("1234567890123456", "(.*)", (int*)NULL));
  ASSERT_FALSE(RE2::FullMatch("hello", "(.*)", (double*)NULL));
  ASSERT_FALSE(RE2::FullMatch("hello", "(.*)", (float*)NULL));
}

// Check that numeric parsing code does not read past the end of
// the number being parsed.
// This implementation requires mmap(2) et al. and thus cannot
// be used unless they are available.
TEST(RE2, NULTerminated) {
#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
  char *v;
  int x;
  long pagesize = sysconf(_SC_PAGE_SIZE);

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
  v = static_cast<char*>(mmap(NULL, 2*pagesize, PROT_READ|PROT_WRITE,
                              MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
  ASSERT_TRUE(v != reinterpret_cast<char*>(-1));
  LOG(INFO) << "Memory at " << (void*)v;
  ASSERT_EQ(munmap(v + pagesize, pagesize), 0) << " error " << errno;
  v[pagesize - 1] = '1';

  x = 0;
  ASSERT_TRUE(RE2::FullMatch(StringPiece(v + pagesize - 1, 1), "(.*)", &x));
  ASSERT_EQ(x, 1);
#endif
}

TEST(RE2, FullMatchTypeTests) {
  // Type tests
  std::string zeros(1000, '0');
  {
    char c;
    ASSERT_TRUE(RE2::FullMatch("Hello", "(H)ello", &c));
    ASSERT_EQ(c, 'H');
  }
  {
    unsigned char c;
    ASSERT_TRUE(RE2::FullMatch("Hello", "(H)ello", &c));
    ASSERT_EQ(c, static_cast<unsigned char>('H'));
  }
  {
    int16_t v;
    ASSERT_TRUE(RE2::FullMatch("100",     "(-?\\d+)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100",    "(-?\\d+)", &v)); ASSERT_EQ(v, -100);
    ASSERT_TRUE(RE2::FullMatch("32767",   "(-?\\d+)", &v)); ASSERT_EQ(v, 32767);
    ASSERT_TRUE(RE2::FullMatch("-32768",  "(-?\\d+)", &v)); ASSERT_EQ(v, -32768);
    ASSERT_FALSE(RE2::FullMatch("-32769", "(-?\\d+)", &v));
    ASSERT_FALSE(RE2::FullMatch("32768",  "(-?\\d+)", &v));
  }
  {
    uint16_t v;
    ASSERT_TRUE(RE2::FullMatch("100",    "(\\d+)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("32767",  "(\\d+)", &v)); ASSERT_EQ(v, 32767);
    ASSERT_TRUE(RE2::FullMatch("65535",  "(\\d+)", &v)); ASSERT_EQ(v, 65535);
    ASSERT_FALSE(RE2::FullMatch("65536", "(\\d+)", &v));
  }
  {
    int32_t v;
    static const int32_t max = INT32_C(0x7fffffff);
    static const int32_t min = -max - 1;
    ASSERT_TRUE(RE2::FullMatch("100",          "(-?\\d+)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100",         "(-?\\d+)", &v)); ASSERT_EQ(v, -100);
    ASSERT_TRUE(RE2::FullMatch("2147483647",   "(-?\\d+)", &v)); ASSERT_EQ(v, max);
    ASSERT_TRUE(RE2::FullMatch("-2147483648",  "(-?\\d+)", &v)); ASSERT_EQ(v, min);
    ASSERT_FALSE(RE2::FullMatch("-2147483649", "(-?\\d+)", &v));
    ASSERT_FALSE(RE2::FullMatch("2147483648",  "(-?\\d+)", &v));

    ASSERT_TRUE(RE2::FullMatch(zeros + "2147483647", "(-?\\d+)", &v));
    ASSERT_EQ(v, max);
    ASSERT_TRUE(RE2::FullMatch("-" + zeros + "2147483648", "(-?\\d+)", &v));
    ASSERT_EQ(v, min);

    ASSERT_FALSE(RE2::FullMatch("-" + zeros + "2147483649", "(-?\\d+)", &v));
    ASSERT_TRUE(RE2::FullMatch("0x7fffffff", "(.*)", RE2::CRadix(&v)));
    ASSERT_EQ(v, max);
    ASSERT_FALSE(RE2::FullMatch("000x7fffffff", "(.*)", RE2::CRadix(&v)));
  }
  {
    uint32_t v;
    static const uint32_t max = UINT32_C(0xffffffff);
    ASSERT_TRUE(RE2::FullMatch("100",         "(\\d+)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("4294967295",  "(\\d+)", &v)); ASSERT_EQ(v, max);
    ASSERT_FALSE(RE2::FullMatch("4294967296", "(\\d+)", &v));
    ASSERT_FALSE(RE2::FullMatch("-1",         "(\\d+)", &v));

    ASSERT_TRUE(RE2::FullMatch(zeros + "4294967295", "(\\d+)", &v)); ASSERT_EQ(v, max);
  }
  {
    int64_t v;
    static const int64_t max = INT64_C(0x7fffffffffffffff);
    static const int64_t min = -max - 1;
    std::string str;

    ASSERT_TRUE(RE2::FullMatch("100",  "(-?\\d+)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100", "(-?\\d+)", &v)); ASSERT_EQ(v, -100);

    str = std::to_string(max);
    ASSERT_TRUE(RE2::FullMatch(str,    "(-?\\d+)", &v)); ASSERT_EQ(v, max);

    str = std::to_string(min);
    ASSERT_TRUE(RE2::FullMatch(str,    "(-?\\d+)", &v)); ASSERT_EQ(v, min);

    str = std::to_string(max);
    ASSERT_NE(str.back(), '9');
    str.back()++;
    ASSERT_FALSE(RE2::FullMatch(str,   "(-?\\d+)", &v));

    str = std::to_string(min);
    ASSERT_NE(str.back(), '9');
    str.back()++;
    ASSERT_FALSE(RE2::FullMatch(str,   "(-?\\d+)", &v));
  }
  {
    uint64_t v;
    int64_t v2;
    static const uint64_t max = UINT64_C(0xffffffffffffffff);
    std::string str;

    ASSERT_TRUE(RE2::FullMatch("100",  "(-?\\d+)", &v));  ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100", "(-?\\d+)", &v2)); ASSERT_EQ(v2, -100);

    str = std::to_string(max);
    ASSERT_TRUE(RE2::FullMatch(str,    "(-?\\d+)", &v)); ASSERT_EQ(v, max);

    ASSERT_NE(str.back(), '9');
    str.back()++;
    ASSERT_FALSE(RE2::FullMatch(str,   "(-?\\d+)", &v));
  }
}

TEST(RE2, FloatingPointFullMatchTypes) {
  std::string zeros(1000, '0');
  {
    float v;
    ASSERT_TRUE(RE2::FullMatch("100",   "(.*)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100.", "(.*)", &v)); ASSERT_EQ(v, -100);
    ASSERT_TRUE(RE2::FullMatch("1e23",  "(.*)", &v)); ASSERT_EQ(v, float(1e23));
    ASSERT_TRUE(RE2::FullMatch(" 100",  "(.*)", &v)); ASSERT_EQ(v, 100);

    ASSERT_TRUE(RE2::FullMatch(zeros + "1e23",  "(.*)", &v));
    ASSERT_EQ(v, float(1e23));

    // 6700000000081920.1 is an edge case.
    // 6700000000081920 is exactly halfway between
    // two float32s, so the .1 should make it round up.
    // However, the .1 is outside the precision possible with
    // a float64: the nearest float64 is 6700000000081920.
    // So if the code uses strtod and then converts to float32,
    // round-to-even will make it round down instead of up.
    // To pass the test, the parser must call strtof directly.
    // This test case is carefully chosen to use only a 17-digit
    // number, since C does not guarantee to get the correctly
    // rounded answer for strtod and strtof unless the input is
    // short.
    //
    // This is known to fail on Cygwin and MinGW due to a broken
    // implementation of strtof(3). And apparently MSVC too. Sigh.
#if !defined(_MSC_VER) && !defined(__CYGWIN__) && !defined(__MINGW32__)
    ASSERT_TRUE(RE2::FullMatch("0.1", "(.*)", &v));
    ASSERT_EQ(v, 0.1f) << StringPrintf("%.8g != %.8g", v, 0.1f);
    ASSERT_TRUE(RE2::FullMatch("6700000000081920.1", "(.*)", &v));
    ASSERT_EQ(v, 6700000000081920.1f)
      << StringPrintf("%.8g != %.8g", v, 6700000000081920.1f);
#endif
  }
  {
    double v;
    ASSERT_TRUE(RE2::FullMatch("100",   "(.*)", &v)); ASSERT_EQ(v, 100);
    ASSERT_TRUE(RE2::FullMatch("-100.", "(.*)", &v)); ASSERT_EQ(v, -100);
    ASSERT_TRUE(RE2::FullMatch("1e23",  "(.*)", &v)); ASSERT_EQ(v, 1e23);
    ASSERT_TRUE(RE2::FullMatch(zeros + "1e23", "(.*)", &v));
    ASSERT_EQ(v, double(1e23));

    ASSERT_TRUE(RE2::FullMatch("0.1", "(.*)", &v));
    ASSERT_EQ(v, 0.1) << StringPrintf("%.17g != %.17g", v, 0.1);
    ASSERT_TRUE(RE2::FullMatch("1.00000005960464485", "(.*)", &v));
    ASSERT_EQ(v, 1.0000000596046448)
      << StringPrintf("%.17g != %.17g", v, 1.0000000596046448);
  }
}

TEST(RE2, FullMatchAnchored) {
  int i;
  // Check that matching is fully anchored
  ASSERT_FALSE(RE2::FullMatch("x1001", "(\\d+)",  &i));
  ASSERT_FALSE(RE2::FullMatch("1001x", "(\\d+)",  &i));
  ASSERT_TRUE(RE2::FullMatch("x1001",  "x(\\d+)", &i)); ASSERT_EQ(i, 1001);
  ASSERT_TRUE(RE2::FullMatch("1001x",  "(\\d+)x", &i)); ASSERT_EQ(i, 1001);
}

TEST(RE2, FullMatchBraces) {
  // Braces
  ASSERT_TRUE(RE2::FullMatch("0abcd",  "[0-9a-f+.-]{5,}"));
  ASSERT_TRUE(RE2::FullMatch("0abcde", "[0-9a-f+.-]{5,}"));
  ASSERT_FALSE(RE2::FullMatch("0abc",  "[0-9a-f+.-]{5,}"));
}

TEST(RE2, Complicated) {
  // Complicated RE2
  ASSERT_TRUE(RE2::FullMatch("foo", "foo|bar|[A-Z]"));
  ASSERT_TRUE(RE2::FullMatch("bar", "foo|bar|[A-Z]"));
  ASSERT_TRUE(RE2::FullMatch("X",   "foo|bar|[A-Z]"));
  ASSERT_FALSE(RE2::FullMatch("XY", "foo|bar|[A-Z]"));
}

TEST(RE2, FullMatchEnd) {
  // Check full-match handling (needs '$' tacked on internally)
  ASSERT_TRUE(RE2::FullMatch("fo", "fo|foo"));
  ASSERT_TRUE(RE2::FullMatch("foo", "fo|foo"));
  ASSERT_TRUE(RE2::FullMatch("fo", "fo|foo$"));
  ASSERT_TRUE(RE2::FullMatch("foo", "fo|foo$"));
  ASSERT_TRUE(RE2::FullMatch("foo", "foo$"));
  ASSERT_FALSE(RE2::FullMatch("foo$bar", "foo\\$"));
  ASSERT_FALSE(RE2::FullMatch("fox", "fo|bar"));

  // Uncomment the following if we change the handling of '$' to
  // prevent it from matching a trailing newline
  if (false) {
    // Check that we don't get bitten by pcre's special handling of a
    // '\n' at the end of the string matching '$'
    ASSERT_FALSE(RE2::PartialMatch("foo\n", "foo$"));
  }
}

TEST(RE2, FullMatchArgCount) {
  // Number of args
  int a[16];
  ASSERT_TRUE(RE2::FullMatch("", ""));

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("1", "(\\d){1}", &a[0]));
  ASSERT_EQ(a[0], 1);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("12", "(\\d)(\\d)", &a[0], &a[1]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("123", "(\\d)(\\d)(\\d)", &a[0], &a[1], &a[2]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("1234", "(\\d)(\\d)(\\d)(\\d)", &a[0], &a[1],
                             &a[2], &a[3]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);
  ASSERT_EQ(a[3], 4);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("12345", "(\\d)(\\d)(\\d)(\\d)(\\d)", &a[0], &a[1],
                             &a[2], &a[3], &a[4]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);
  ASSERT_EQ(a[3], 4);
  ASSERT_EQ(a[4], 5);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("123456", "(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)", &a[0],
                             &a[1], &a[2], &a[3], &a[4], &a[5]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);
  ASSERT_EQ(a[3], 4);
  ASSERT_EQ(a[4], 5);
  ASSERT_EQ(a[5], 6);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("1234567", "(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)",
                             &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);
  ASSERT_EQ(a[3], 4);
  ASSERT_EQ(a[4], 5);
  ASSERT_EQ(a[5], 6);
  ASSERT_EQ(a[6], 7);

  memset(a, 0, sizeof(0));
  ASSERT_TRUE(RE2::FullMatch("1234567890123456",
                             "(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)"
                             "(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)(\\d)",
                             &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
                             &a[7], &a[8], &a[9], &a[10], &a[11], &a[12],
                             &a[13], &a[14], &a[15]));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
  ASSERT_EQ(a[2], 3);
  ASSERT_EQ(a[3], 4);
  ASSERT_EQ(a[4], 5);
  ASSERT_EQ(a[5], 6);
  ASSERT_EQ(a[6], 7);
  ASSERT_EQ(a[7], 8);
  ASSERT_EQ(a[8], 9);
  ASSERT_EQ(a[9], 0);
  ASSERT_EQ(a[10], 1);
  ASSERT_EQ(a[11], 2);
  ASSERT_EQ(a[12], 3);
  ASSERT_EQ(a[13], 4);
  ASSERT_EQ(a[14], 5);
  ASSERT_EQ(a[15], 6);
}

TEST(RE2, Accessors) {
  // Check the pattern() accessor
  {
    const std::string kPattern = "http://([^/]+)/.*";
    const RE2 re(kPattern);
    ASSERT_EQ(kPattern, re.pattern());
  }

  // Check RE2 error field.
  {
    RE2 re("foo");
    ASSERT_TRUE(re.error().empty());  // Must have no error
    ASSERT_TRUE(re.ok());
    ASSERT_EQ(re.error_code(), RE2::NoError);
  }
}

TEST(RE2, UTF8) {
  // Check UTF-8 handling
  // Three Japanese characters (nihongo)
  const char utf8_string[] = {
       (char)0xe6, (char)0x97, (char)0xa5, // 65e5
       (char)0xe6, (char)0x9c, (char)0xac, // 627c
       (char)0xe8, (char)0xaa, (char)0x9e, // 8a9e
       0
  };
  const char utf8_pattern[] = {
       '.',
       (char)0xe6, (char)0x9c, (char)0xac, // 627c
       '.',
       0
  };

  // Both should match in either mode, bytes or UTF-8
  RE2 re_test1(".........", RE2::Latin1);
  ASSERT_TRUE(RE2::FullMatch(utf8_string, re_test1));
  RE2 re_test2("...");
  ASSERT_TRUE(RE2::FullMatch(utf8_string, re_test2));

  // Check that '.' matches one byte or UTF-8 character
  // according to the mode.
  std::string s;
  RE2 re_test3("(.)", RE2::Latin1);
  ASSERT_TRUE(RE2::PartialMatch(utf8_string, re_test3, &s));
  ASSERT_EQ(s, std::string("\xe6"));
  RE2 re_test4("(.)");
  ASSERT_TRUE(RE2::PartialMatch(utf8_string, re_test4, &s));
  ASSERT_EQ(s, std::string("\xe6\x97\xa5"));

  // Check that string matches itself in either mode
  RE2 re_test5(utf8_string, RE2::Latin1);
  ASSERT_TRUE(RE2::FullMatch(utf8_string, re_test5));
  RE2 re_test6(utf8_string);
  ASSERT_TRUE(RE2::FullMatch(utf8_string, re_test6));

  // Check that pattern matches string only in UTF8 mode
  RE2 re_test7(utf8_pattern, RE2::Latin1);
  ASSERT_FALSE(RE2::FullMatch(utf8_string, re_test7));
  RE2 re_test8(utf8_pattern);
  ASSERT_TRUE(RE2::FullMatch(utf8_string, re_test8));
}

TEST(RE2, UngreedyUTF8) {
  // Check that ungreedy, UTF8 regular expressions don't match when they
  // oughtn't -- see bug 82246.
  {
    // This code always worked.
    const char* pattern = "\\w+X";
    const std::string target = "a aX";
    RE2 match_sentence(pattern, RE2::Latin1);
    RE2 match_sentence_re(pattern);

    ASSERT_FALSE(RE2::FullMatch(target, match_sentence));
    ASSERT_FALSE(RE2::FullMatch(target, match_sentence_re));
  }
  {
    const char* pattern = "(?U)\\w+X";
    const std::string target = "a aX";
    RE2 match_sentence(pattern, RE2::Latin1);
    ASSERT_EQ(match_sentence.error(), "");
    RE2 match_sentence_re(pattern);

    ASSERT_FALSE(RE2::FullMatch(target, match_sentence));
    ASSERT_FALSE(RE2::FullMatch(target, match_sentence_re));
  }
}

TEST(RE2, Rejects) {
  {
    RE2 re("a\\1", RE2::Quiet);
    ASSERT_FALSE(re.ok()); }
  {
    RE2 re("a[x", RE2::Quiet);
    ASSERT_FALSE(re.ok());
  }
  {
    RE2 re("a[z-a]", RE2::Quiet);
    ASSERT_FALSE(re.ok());
  }
  {
    RE2 re("a[[:foobar:]]", RE2::Quiet);
    ASSERT_FALSE(re.ok());
  }
  {
    RE2 re("a(b", RE2::Quiet);
    ASSERT_FALSE(re.ok());
  }
  {
    RE2 re("a\\", RE2::Quiet);
    ASSERT_FALSE(re.ok());
  }
}

TEST(RE2, NoCrash) {
  // Test that using a bad regexp doesn't crash.
  {
    RE2 re("a\\", RE2::Quiet);
    ASSERT_FALSE(re.ok());
    ASSERT_FALSE(RE2::PartialMatch("a\\b", re));
  }

  // Test that using an enormous regexp doesn't crash
  {
    RE2 re("(((.{100}){100}){100}){100}", RE2::Quiet);
    ASSERT_FALSE(re.ok());
    ASSERT_FALSE(RE2::PartialMatch("aaa", re));
  }

  // Test that a crazy regexp still compiles and runs.
  {
    RE2 re(".{512}x", RE2::Quiet);
    ASSERT_TRUE(re.ok());
    std::string s;
    s.append(515, 'c');
    s.append("x");
    ASSERT_TRUE(RE2::PartialMatch(s, re));
  }
}

TEST(RE2, Recursion) {
  // Test that recursion is stopped.
  // This test is PCRE-legacy -- there's no recursion in RE2.
  int bytes = 15 * 1024;  // enough to crash PCRE
  TestRecursion(bytes, ".");
  TestRecursion(bytes, "a");
  TestRecursion(bytes, "a.");
  TestRecursion(bytes, "ab.");
  TestRecursion(bytes, "abc.");
}

TEST(RE2, BigCountedRepetition) {
  // Test that counted repetition works, given tons of memory.
  RE2::Options opt;
  opt.set_max_mem(256<<20);

  RE2 re(".{512}x", opt);
  ASSERT_TRUE(re.ok());
  std::string s;
  s.append(515, 'c');
  s.append("x");
  ASSERT_TRUE(RE2::PartialMatch(s, re));
}

TEST(RE2, DeepRecursion) {
  // Test for deep stack recursion.  This would fail with a
  // segmentation violation due to stack overflow before pcre was
  // patched.
  // Again, a PCRE legacy test.  RE2 doesn't recurse.
  std::string comment("x*");
  std::string a(131072, 'a');
  comment += a;
  comment += "*x";
  RE2 re("((?:\\s|xx.*\n|x[*](?:\n|.)*?[*]x)*)");
  ASSERT_TRUE(RE2::FullMatch(comment, re));
}

// Suggested by Josh Hyman.  Failed when SearchOnePass was
// not implementing case-folding.
TEST(CaseInsensitive, MatchAndConsume) {
  std::string result;
  std::string text = "A fish named *Wanda*";
  StringPiece sp(text);

  EXPECT_TRUE(RE2::PartialMatch(sp, "(?i)([wand]{5})", &result));
  EXPECT_TRUE(RE2::FindAndConsume(&sp, "(?i)([wand]{5})", &result));
}

// RE2 should permit implicit conversions from string, StringPiece, const char*,
// and C string literals.
TEST(RE2, ImplicitConversions) {
  std::string re_string(".");
  StringPiece re_stringpiece(".");
  const char* re_cstring = ".";
  EXPECT_TRUE(RE2::PartialMatch("e", re_string));
  EXPECT_TRUE(RE2::PartialMatch("e", re_stringpiece));
  EXPECT_TRUE(RE2::PartialMatch("e", re_cstring));
  EXPECT_TRUE(RE2::PartialMatch("e", "."));
}

// Bugs introduced by 8622304
TEST(RE2, CL8622304) {
  // reported by ingow
  std::string dir;
  EXPECT_TRUE(RE2::FullMatch("D", "([^\\\\])"));  // ok
  EXPECT_TRUE(RE2::FullMatch("D", "([^\\\\])", &dir));  // fails

  // reported by jacobsa
  std::string key, val;
  EXPECT_TRUE(RE2::PartialMatch("bar:1,0x2F,030,4,5;baz:true;fooby:false,true",
              "(\\w+)(?::((?:[^;\\\\]|\\\\.)*))?;?",
              &key,
              &val));
  EXPECT_EQ(key, "bar");
  EXPECT_EQ(val, "1,0x2F,030,4,5");
}


// Check that RE2 returns correct regexp pieces on error.
// In particular, make sure it returns whole runes
// and that it always reports invalid UTF-8.
// Also check that Perl error flag piece is big enough.
static struct ErrorTest {
  const char *regexp;
  const char *error;
} error_tests[] = {
  { "ab\\αcd", "\\α" },
  { "ef\\x☺01", "\\x☺0" },
  { "gh\\x1☺01", "\\x1☺" },
  { "ij\\x1", "\\x1" },
  { "kl\\x", "\\x" },
  { "uv\\x{0000☺}", "\\x{0000☺" },
  { "wx\\p{ABC", "\\p{ABC" },
  { "yz(?smiUX:abc)", "(?smiUX" },   // used to return (?s but the error is X
  { "aa(?sm☺i", "(?sm☺" },
  { "bb[abc", "[abc" },

  { "mn\\x1\377", "" },  // no argument string returned for invalid UTF-8
  { "op\377qr", "" },
  { "st\\x{00000\377", "" },
  { "zz\\p{\377}", "" },
  { "zz\\x{00\377}", "" },
  { "zz(?P<name\377>abc)", "" },
};
TEST(RE2, ErrorArgs) {
  for (int i = 0; i < arraysize(error_tests); i++) {
    RE2 re(error_tests[i].regexp, RE2::Quiet);
    EXPECT_FALSE(re.ok());
    EXPECT_EQ(re.error_arg(), error_tests[i].error) << re.error();
  }
}

// Check that "never match \n" mode never matches \n.
static struct NeverTest {
  const char* regexp;
  const char* text;
  const char* match;
} never_tests[] = {
  { "(.*)", "abc\ndef\nghi\n", "abc" },
  { "(?s)(abc.*def)", "abc\ndef\n", NULL },
  { "(abc(.|\n)*def)", "abc\ndef\n", NULL },
  { "(abc[^x]*def)", "abc\ndef\n", NULL },
  { "(abc[^x]*def)", "abczzzdef\ndef\n", "abczzzdef" },
};
TEST(RE2, NeverNewline) {
  RE2::Options opt;
  opt.set_never_nl(true);
  for (int i = 0; i < arraysize(never_tests); i++) {
    const NeverTest& t = never_tests[i];
    RE2 re(t.regexp, opt);
    if (t.match == NULL) {
      EXPECT_FALSE(re.PartialMatch(t.text, re));
    } else {
      StringPiece m;
      EXPECT_TRUE(re.PartialMatch(t.text, re, &m));
      EXPECT_EQ(m, t.match);
    }
  }
}

// Check that dot_nl option works.
TEST(RE2, DotNL) {
  RE2::Options opt;
  opt.set_dot_nl(true);
  EXPECT_TRUE(RE2::PartialMatch("\n", RE2(".", opt)));
  EXPECT_FALSE(RE2::PartialMatch("\n", RE2("(?-s).", opt)));
  opt.set_never_nl(true);
  EXPECT_FALSE(RE2::PartialMatch("\n", RE2(".", opt)));
}

// Check that there are no capturing groups in "never capture" mode.
TEST(RE2, NeverCapture) {
  RE2::Options opt;
  opt.set_never_capture(true);
  RE2 re("(r)(e)", opt);
  EXPECT_EQ(0, re.NumberOfCapturingGroups());
}

// Bitstate bug was looking at submatch[0] even if nsubmatch == 0.
// Triggered by a failed DFA search falling back to Bitstate when
// using Match with a NULL submatch set.  Bitstate tried to read
// the submatch[0] entry even if nsubmatch was 0.
TEST(RE2, BitstateCaptureBug) {
  RE2::Options opt;
  opt.set_max_mem(20000);
  RE2 re("(_________$)", opt);
  StringPiece s = "xxxxxxxxxxxxxxxxxxxxxxxxxx_________x";
  EXPECT_FALSE(re.Match(s, 0, s.size(), RE2::UNANCHORED, NULL, 0));
}

// C++ version of bug 609710.
TEST(RE2, UnicodeClasses) {
  const std::string str = "ABCDEFGHI譚永鋒";
  std::string a, b, c;

  EXPECT_TRUE(RE2::FullMatch("A", "\\p{L}"));
  EXPECT_TRUE(RE2::FullMatch("A", "\\p{Lu}"));
  EXPECT_FALSE(RE2::FullMatch("A", "\\p{Ll}"));
  EXPECT_FALSE(RE2::FullMatch("A", "\\P{L}"));
  EXPECT_FALSE(RE2::FullMatch("A", "\\P{Lu}"));
  EXPECT_TRUE(RE2::FullMatch("A", "\\P{Ll}"));

  EXPECT_TRUE(RE2::FullMatch("譚", "\\p{L}"));
  EXPECT_FALSE(RE2::FullMatch("譚", "\\p{Lu}"));
  EXPECT_FALSE(RE2::FullMatch("譚", "\\p{Ll}"));
  EXPECT_FALSE(RE2::FullMatch("譚", "\\P{L}"));
  EXPECT_TRUE(RE2::FullMatch("譚", "\\P{Lu}"));
  EXPECT_TRUE(RE2::FullMatch("譚", "\\P{Ll}"));

  EXPECT_TRUE(RE2::FullMatch("永", "\\p{L}"));
  EXPECT_FALSE(RE2::FullMatch("永", "\\p{Lu}"));
  EXPECT_FALSE(RE2::FullMatch("永", "\\p{Ll}"));
  EXPECT_FALSE(RE2::FullMatch("永", "\\P{L}"));
  EXPECT_TRUE(RE2::FullMatch("永", "\\P{Lu}"));
  EXPECT_TRUE(RE2::FullMatch("永", "\\P{Ll}"));

  EXPECT_TRUE(RE2::FullMatch("鋒", "\\p{L}"));
  EXPECT_FALSE(RE2::FullMatch("鋒", "\\p{Lu}"));
  EXPECT_FALSE(RE2::FullMatch("鋒", "\\p{Ll}"));
  EXPECT_FALSE(RE2::FullMatch("鋒", "\\P{L}"));
  EXPECT_TRUE(RE2::FullMatch("鋒", "\\P{Lu}"));
  EXPECT_TRUE(RE2::FullMatch("鋒", "\\P{Ll}"));

  EXPECT_TRUE(RE2::PartialMatch(str, "(.).*?(.).*?(.)", &a, &b, &c));
  EXPECT_EQ("A", a);
  EXPECT_EQ("B", b);
  EXPECT_EQ("C", c);

  EXPECT_TRUE(RE2::PartialMatch(str, "(.).*?([\\p{L}]).*?(.)", &a, &b, &c));
  EXPECT_EQ("A", a);
  EXPECT_EQ("B", b);
  EXPECT_EQ("C", c);

  EXPECT_FALSE(RE2::PartialMatch(str, "\\P{L}"));

  EXPECT_TRUE(RE2::PartialMatch(str, "(.).*?([\\p{Lu}]).*?(.)", &a, &b, &c));
  EXPECT_EQ("A", a);
  EXPECT_EQ("B", b);
  EXPECT_EQ("C", c);

  EXPECT_FALSE(RE2::PartialMatch(str, "[^\\p{Lu}\\p{Lo}]"));

  EXPECT_TRUE(RE2::PartialMatch(str, ".*(.).*?([\\p{Lu}\\p{Lo}]).*?(.)", &a, &b, &c));
  EXPECT_EQ("譚", a);
  EXPECT_EQ("永", b);
  EXPECT_EQ("鋒", c);
}

TEST(RE2, LazyRE2) {
  // Test with and without options.
  static LazyRE2 a = {"a"};
  static LazyRE2 b = {"b", RE2::Latin1};

  EXPECT_EQ("a", a->pattern());
  EXPECT_EQ(RE2::Options::EncodingUTF8, a->options().encoding());

  EXPECT_EQ("b", b->pattern());
  EXPECT_EQ(RE2::Options::EncodingLatin1, b->options().encoding());
}

// Bug reported by saito. 2009/02/17
TEST(RE2, NullVsEmptyString) {
  RE2 re(".*");
  EXPECT_TRUE(re.ok());

  StringPiece null;
  EXPECT_TRUE(RE2::FullMatch(null, re));

  StringPiece empty("");
  EXPECT_TRUE(RE2::FullMatch(empty, re));
}

// Similar to the previous test, check that the null string and the empty
// string both match, but also that the null string can only provide null
// submatches whereas the empty string can also provide empty submatches.
TEST(RE2, NullVsEmptyStringSubmatches) {
  RE2 re("()|(foo)");
  EXPECT_TRUE(re.ok());

  // matches[0] is overall match, [1] is (), [2] is (foo), [3] is nonexistent.
  StringPiece matches[4];

  for (int i = 0; i < arraysize(matches); i++)
    matches[i] = "bar";

  StringPiece null;
  EXPECT_TRUE(re.Match(null, 0, null.size(), RE2::UNANCHORED,
                       matches, arraysize(matches)));
  for (int i = 0; i < arraysize(matches); i++) {
    EXPECT_TRUE(matches[i] == StringPiece());
    EXPECT_TRUE(matches[i].data() == NULL);  // always null
    EXPECT_TRUE(matches[i] == "");
  }

  for (int i = 0; i < arraysize(matches); i++)
    matches[i] = "bar";

  StringPiece empty("");
  EXPECT_TRUE(re.Match(empty, 0, empty.size(), RE2::UNANCHORED,
                       matches, arraysize(matches)));
  EXPECT_TRUE(matches[0] == StringPiece());
  EXPECT_TRUE(matches[0].data() != NULL);  // empty, not null
  EXPECT_TRUE(matches[0] == "");
  EXPECT_TRUE(matches[1] == StringPiece());
  EXPECT_TRUE(matches[1].data() != NULL);  // empty, not null
  EXPECT_TRUE(matches[1] == "");
  EXPECT_TRUE(matches[2] == StringPiece());
  EXPECT_TRUE(matches[2].data() == NULL);
  EXPECT_TRUE(matches[2] == "");
  EXPECT_TRUE(matches[3] == StringPiece());
  EXPECT_TRUE(matches[3].data() == NULL);
  EXPECT_TRUE(matches[3] == "");
}

// Issue 1816809
TEST(RE2, Bug1816809) {
  RE2 re("(((((llx((-3)|(4)))(;(llx((-3)|(4))))*))))");
  StringPiece piece("llx-3;llx4");
  std::string x;
  EXPECT_TRUE(RE2::Consume(&piece, re, &x));
}

// Issue 3061120
TEST(RE2, Bug3061120) {
  RE2 re("(?i)\\W");
  EXPECT_FALSE(RE2::PartialMatch("x", re));  // always worked
  EXPECT_FALSE(RE2::PartialMatch("k", re));  // broke because of kelvin
  EXPECT_FALSE(RE2::PartialMatch("s", re));  // broke because of latin long s
}

TEST(RE2, CapturingGroupNames) {
  // Opening parentheses annotated with group IDs:
  //      12    3        45   6         7
  RE2 re("((abc)(?P<G2>)|((e+)(?P<G2>.*)(?P<G1>u+)))");
  EXPECT_TRUE(re.ok());
  const std::map<int, std::string>& have = re.CapturingGroupNames();
  std::map<int, std::string> want;
  want[3] = "G2";
  want[6] = "G2";
  want[7] = "G1";
  EXPECT_EQ(want, have);
}

TEST(RE2, RegexpToStringLossOfAnchor) {
  EXPECT_EQ(RE2("^[a-c]at", RE2::POSIX).Regexp()->ToString(), "^[a-c]at");
  EXPECT_EQ(RE2("^[a-c]at").Regexp()->ToString(), "(?-m:^)[a-c]at");
  EXPECT_EQ(RE2("ca[t-z]$", RE2::POSIX).Regexp()->ToString(), "ca[t-z]$");
  EXPECT_EQ(RE2("ca[t-z]$").Regexp()->ToString(), "ca[t-z](?-m:$)");
}

// Issue 10131674
TEST(RE2, Bug10131674) {
  // Some of these escapes describe values that do not fit in a byte.
  RE2 re("\\140\\440\\174\\271\\150\\656\\106\\201\\004\\332", RE2::Latin1);
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(RE2::FullMatch("hello world", re));
}

TEST(RE2, Bug18391750) {
  // Stray write past end of match_ in nfa.cc, caught by fuzzing + address sanitizer.
  const char t[] = {
      (char)0x28, (char)0x28, (char)0xfc, (char)0xfc, (char)0x08, (char)0x08,
      (char)0x26, (char)0x26, (char)0x28, (char)0xc2, (char)0x9b, (char)0xc5,
      (char)0xc5, (char)0xd4, (char)0x8f, (char)0x8f, (char)0x69, (char)0x69,
      (char)0xe7, (char)0x29, (char)0x7b, (char)0x37, (char)0x31, (char)0x31,
      (char)0x7d, (char)0xae, (char)0x7c, (char)0x7c, (char)0xf3, (char)0x29,
      (char)0xae, (char)0xae, (char)0x2e, (char)0x2a, (char)0x29, (char)0x00,
  };
  RE2::Options opt;
  opt.set_encoding(RE2::Options::EncodingLatin1);
  opt.set_longest_match(true);
  opt.set_dot_nl(true);
  opt.set_case_sensitive(false);
  RE2 re(t, opt);
  ASSERT_TRUE(re.ok());
  RE2::PartialMatch(t, re);
}

TEST(RE2, Bug18458852) {
  // Bug in parser accepting invalid (too large) rune,
  // causing compiler to fail in DCHECK in UTF-8
  // character class code.
  const char b[] = {
      (char)0x28, (char)0x05, (char)0x05, (char)0x41, (char)0x41, (char)0x28,
      (char)0x24, (char)0x5b, (char)0x5e, (char)0xf5, (char)0x87, (char)0x87,
      (char)0x90, (char)0x29, (char)0x5d, (char)0x29, (char)0x29, (char)0x00,
  };
  RE2 re(b);
  ASSERT_FALSE(re.ok());
}

TEST(RE2, Bug18523943) {
  // Bug in BitState: case kFailInst failed the match entirely.

  RE2::Options opt;
  const char a[] = {
      (char)0x29, (char)0x29, (char)0x24, (char)0x00,
  };
  const char b[] = {
      (char)0x28, (char)0x0a, (char)0x2a, (char)0x2a, (char)0x29, (char)0x00,
  };
  opt.set_log_errors(false);
  opt.set_encoding(RE2::Options::EncodingLatin1);
  opt.set_posix_syntax(true);
  opt.set_longest_match(true);
  opt.set_literal(false);
  opt.set_never_nl(true);

  RE2 re((const char*)b, opt);
  ASSERT_TRUE(re.ok());
  std::string s1;
  ASSERT_TRUE(RE2::PartialMatch((const char*)a, re, &s1));
}

TEST(RE2, Bug21371806) {
  // Bug in parser accepting Unicode groups in Latin-1 mode,
  // causing compiler to fail in DCHECK in prog.cc.

  RE2::Options opt;
  opt.set_encoding(RE2::Options::EncodingLatin1);

  RE2 re("g\\p{Zl}]", opt);
  ASSERT_TRUE(re.ok());
}

TEST(RE2, Bug26356109) {
  // Bug in parser caused by factoring of common prefixes in alternations.

  // In the past, this was factored to "a\\C*?[bc]". Thus, the automaton would
  // consume "ab" and then stop (when unanchored) whereas it should consume all
  // of "abc" as per first-match semantics.
  RE2 re("a\\C*?c|a\\C*?b");
  ASSERT_TRUE(re.ok());

  std::string s = "abc";
  StringPiece m;

  ASSERT_TRUE(re.Match(s, 0, s.size(), RE2::UNANCHORED, &m, 1));
  ASSERT_EQ(m, s) << " (UNANCHORED) got m='" << m << "', want '" << s << "'";

  ASSERT_TRUE(re.Match(s, 0, s.size(), RE2::ANCHOR_BOTH, &m, 1));
  ASSERT_EQ(m, s) << " (ANCHOR_BOTH) got m='" << m << "', want '" << s << "'";
}

TEST(RE2, Issue104) {
  // RE2::GlobalReplace always advanced by one byte when the empty string was
  // matched, which would clobber any rune that is longer than one byte.

  std::string s = "bc";
  ASSERT_EQ(3, RE2::GlobalReplace(&s, "a*", "d"));
  ASSERT_EQ("dbdcd", s);

  s = "ąć";
  ASSERT_EQ(3, RE2::GlobalReplace(&s, "Ć*", "Ĉ"));
  ASSERT_EQ("ĈąĈćĈ", s);

  s = "人类";
  ASSERT_EQ(3, RE2::GlobalReplace(&s, "大*", "小"));
  ASSERT_EQ("小人小类小", s);
}

}  // namespace re2
