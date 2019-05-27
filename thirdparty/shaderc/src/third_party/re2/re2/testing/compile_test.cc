// Copyright 2007 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test prog.cc, compile.cc

#include <string>

#include "util/test.h"
#include "util/logging.h"
#include "re2/regexp.h"
#include "re2/prog.h"

namespace re2 {

// Simple input/output tests checking that
// the regexp compiles to the expected code.
// These are just to sanity check the basic implementation.
// The real confidence tests happen by testing the NFA/DFA
// that run the compiled code.

struct Test {
  const char* regexp;
  const char* code;
};

static Test tests[] = {
  { "a",
    "3. byte [61-61] 0 -> 4\n"
    "4. match! 0\n" },
  { "ab",
    "3. byte [61-61] 0 -> 4\n"
    "4. byte [62-62] 0 -> 5\n"
    "5. match! 0\n" },
  { "a|c",
    "3+ byte [61-61] 0 -> 5\n"
    "4. byte [63-63] 0 -> 5\n"
    "5. match! 0\n" },
  { "a|b",
    "3. byte [61-62] 0 -> 4\n"
    "4. match! 0\n" },
  { "[ab]",
    "3. byte [61-62] 0 -> 4\n"
    "4. match! 0\n" },
  { "a+",
    "3. byte [61-61] 0 -> 4\n"
    "4+ nop -> 3\n"
    "5. match! 0\n" },
  { "a+?",
    "3. byte [61-61] 0 -> 4\n"
    "4+ match! 0\n"
    "5. nop -> 3\n" },
  { "a*",
    "3+ byte [61-61] 1 -> 3\n"
    "4. match! 0\n" },
  { "a*?",
    "3+ match! 0\n"
    "4. byte [61-61] 0 -> 3\n" },
  { "a?",
    "3+ byte [61-61] 1 -> 5\n"
    "4. nop -> 5\n"
    "5. match! 0\n" },
  { "a??",
    "3+ nop -> 5\n"
    "4. byte [61-61] 0 -> 5\n"
    "5. match! 0\n" },
  { "a{4}",
    "3. byte [61-61] 0 -> 4\n"
    "4. byte [61-61] 0 -> 5\n"
    "5. byte [61-61] 0 -> 6\n"
    "6. byte [61-61] 0 -> 7\n"
    "7. match! 0\n" },
  { "(a)",
    "3. capture 2 -> 4\n"
    "4. byte [61-61] 0 -> 5\n"
    "5. capture 3 -> 6\n"
    "6. match! 0\n" },
  { "(?:a)",
    "3. byte [61-61] 0 -> 4\n"
    "4. match! 0\n" },
  { "",
    "3. match! 0\n" },
  { ".",
    "3+ byte [00-09] 0 -> 5\n"
    "4. byte [0b-ff] 0 -> 5\n"
    "5. match! 0\n" },
  { "[^ab]",
    "3+ byte [00-09] 0 -> 6\n"
    "4+ byte [0b-60] 0 -> 6\n"
    "5. byte [63-ff] 0 -> 6\n"
    "6. match! 0\n" },
  { "[Aa]",
    "3. byte/i [61-61] 0 -> 4\n"
    "4. match! 0\n" },
  { "\\C+",
    "3. byte [00-ff] 0 -> 4\n"
    "4+ altmatch -> 5 | 6\n"
    "5+ nop -> 3\n"
    "6. match! 0\n" },
  { "\\C*",
    "3+ altmatch -> 4 | 5\n"
    "4+ byte [00-ff] 1 -> 3\n"
    "5. match! 0\n" },
  { "\\C?",
    "3+ byte [00-ff] 1 -> 5\n"
    "4. nop -> 5\n"
    "5. match! 0\n" },
  // Issue 20992936
  { "[[-`]",
    "3. byte [5b-60] 0 -> 4\n"
    "4. match! 0\n" },
};

TEST(TestRegexpCompileToProg, Simple) {
  int failed = 0;
  for (int i = 0; i < arraysize(tests); i++) {
    const re2::Test& t = tests[i];
    Regexp* re = Regexp::Parse(t.regexp, Regexp::PerlX|Regexp::Latin1, NULL);
    if (re == NULL) {
      LOG(ERROR) << "Cannot parse: " << t.regexp;
      failed++;
      continue;
    }
    Prog* prog = re->CompileToProg(0);
    if (prog == NULL) {
      LOG(ERROR) << "Cannot compile: " << t.regexp;
      re->Decref();
      failed++;
      continue;
    }
    ASSERT_TRUE(re->CompileToProg(1) == NULL);
    std::string s = prog->Dump();
    if (s != t.code) {
      LOG(ERROR) << "Incorrect compiled code for: " << t.regexp;
      LOG(ERROR) << "Want:\n" << t.code;
      LOG(ERROR) << "Got:\n" << s;
      failed++;
    }
    delete prog;
    re->Decref();
  }
  EXPECT_EQ(failed, 0);
}

static void DumpByteMap(StringPiece pattern, Regexp::ParseFlags flags,
                        std::string* bytemap) {
  Regexp* re = Regexp::Parse(pattern, flags, NULL);
  EXPECT_TRUE(re != NULL);

  Prog* prog = re->CompileToProg(0);
  EXPECT_TRUE(prog != NULL);
  *bytemap = prog->DumpByteMap();
  delete prog;

  re->Decref();
}

TEST(TestCompile, Latin1Ranges) {
  // The distinct byte ranges involved in the Latin-1 dot ([^\n]).

  std::string bytemap;

  DumpByteMap(".", Regexp::PerlX|Regexp::Latin1, &bytemap);
  EXPECT_EQ("[00-09] -> 0\n"
            "[0a-0a] -> 1\n"
            "[0b-ff] -> 0\n",
            bytemap);
}

TEST(TestCompile, OtherByteMapTests) {
  std::string bytemap;

  // Test that "absent" ranges are mapped to the same byte class.
  DumpByteMap("[0-9A-Fa-f]+", Regexp::PerlX|Regexp::Latin1, &bytemap);
  EXPECT_EQ("[00-2f] -> 0\n"
            "[30-39] -> 1\n"
            "[3a-40] -> 0\n"
            "[41-46] -> 1\n"
            "[47-60] -> 0\n"
            "[61-66] -> 1\n"
            "[67-ff] -> 0\n",
            bytemap);

  // Test the byte classes for \b.
  DumpByteMap("\\b", Regexp::LikePerl|Regexp::Latin1, &bytemap);
  EXPECT_EQ("[00-2f] -> 0\n"
            "[30-39] -> 1\n"
            "[3a-40] -> 0\n"
            "[41-5a] -> 1\n"
            "[5b-5e] -> 0\n"
            "[5f-5f] -> 1\n"
            "[60-60] -> 0\n"
            "[61-7a] -> 1\n"
            "[7b-ff] -> 0\n",
            bytemap);

  // Bug in the ASCII case-folding optimization created too many byte classes.
  DumpByteMap("[^_]", Regexp::LikePerl|Regexp::Latin1, &bytemap);
  EXPECT_EQ("[00-5e] -> 0\n"
            "[5f-5f] -> 1\n"
            "[60-ff] -> 0\n",
            bytemap);
}

TEST(TestCompile, UTF8Ranges) {
  // The distinct byte ranges involved in the UTF-8 dot ([^\n]).
  // Once, erroneously split between 0x3f and 0x40 because it is
  // a 6-bit boundary.

  std::string bytemap;

  DumpByteMap(".", Regexp::PerlX, &bytemap);
  EXPECT_EQ("[00-09] -> 0\n"
            "[0a-0a] -> 1\n"
            "[0b-7f] -> 0\n"
            "[80-8f] -> 2\n"
            "[90-9f] -> 3\n"
            "[a0-bf] -> 4\n"
            "[c0-c1] -> 1\n"
            "[c2-df] -> 5\n"
            "[e0-e0] -> 6\n"
            "[e1-ef] -> 7\n"
            "[f0-f0] -> 8\n"
            "[f1-f3] -> 9\n"
            "[f4-f4] -> 10\n"
            "[f5-ff] -> 1\n",
            bytemap);
}

TEST(TestCompile, InsufficientMemory) {
  Regexp* re = Regexp::Parse(
      "^(?P<name1>[^\\s]+)\\s+(?P<name2>[^\\s]+)\\s+(?P<name3>.+)$",
      Regexp::LikePerl, NULL);
  EXPECT_TRUE(re != NULL);
  Prog* prog = re->CompileToProg(920);
  // If the memory budget has been exhausted, compilation should fail
  // and return NULL instead of trying to do anything with NoMatch().
  EXPECT_TRUE(prog == NULL);
  re->Decref();
}

static void Dump(StringPiece pattern, Regexp::ParseFlags flags,
                 std::string* forward, std::string* reverse) {
  Regexp* re = Regexp::Parse(pattern, flags, NULL);
  EXPECT_TRUE(re != NULL);

  if (forward != NULL) {
    Prog* prog = re->CompileToProg(0);
    EXPECT_TRUE(prog != NULL);
    *forward = prog->Dump();
    delete prog;
  }

  if (reverse != NULL) {
    Prog* prog = re->CompileToReverseProg(0);
    EXPECT_TRUE(prog != NULL);
    *reverse = prog->Dump();
    delete prog;
  }

  re->Decref();
}

TEST(TestCompile, Bug26705922) {
  // Bug in the compiler caused inefficient bytecode to be generated for Unicode
  // groups: common suffixes were cached, but common prefixes were not factored.

  std::string forward, reverse;

  Dump("[\\x{10000}\\x{10010}]", Regexp::LikePerl, &forward, &reverse);
  EXPECT_EQ("3. byte [f0-f0] 0 -> 4\n"
            "4. byte [90-90] 0 -> 5\n"
            "5. byte [80-80] 0 -> 6\n"
            "6+ byte [80-80] 0 -> 8\n"
            "7. byte [90-90] 0 -> 8\n"
            "8. match! 0\n",
            forward);
  EXPECT_EQ("3+ byte [80-80] 0 -> 5\n"
            "4. byte [90-90] 0 -> 5\n"
            "5. byte [80-80] 0 -> 6\n"
            "6. byte [90-90] 0 -> 7\n"
            "7. byte [f0-f0] 0 -> 8\n"
            "8. match! 0\n",
            reverse);

  Dump("[\\x{8000}-\\x{10FFF}]", Regexp::LikePerl, &forward, &reverse);
  EXPECT_EQ("3+ byte [e8-ef] 0 -> 5\n"
            "4. byte [f0-f0] 0 -> 8\n"
            "5. byte [80-bf] 0 -> 6\n"
            "6. byte [80-bf] 0 -> 7\n"
            "7. match! 0\n"
            "8. byte [90-90] 0 -> 5\n",
            forward);
  EXPECT_EQ("3. byte [80-bf] 0 -> 4\n"
            "4. byte [80-bf] 0 -> 5\n"
            "5+ byte [e8-ef] 0 -> 7\n"
            "6. byte [90-90] 0 -> 8\n"
            "7. match! 0\n"
            "8. byte [f0-f0] 0 -> 7\n",
            reverse);

  Dump("[\\x{80}-\\x{10FFFF}]", Regexp::LikePerl, NULL, &reverse);
  EXPECT_EQ("3. byte [80-bf] 0 -> 4\n"
            "4+ byte [c2-df] 0 -> 7\n"
            "5+ byte [a0-bf] 1 -> 8\n"
            "6. byte [80-bf] 0 -> 9\n"
            "7. match! 0\n"
            "8. byte [e0-e0] 0 -> 7\n"
            "9+ byte [e1-ef] 0 -> 7\n"
            "10+ byte [90-bf] 1 -> 13\n"
            "11+ byte [80-bf] 1 -> 14\n"
            "12. byte [80-8f] 0 -> 15\n"
            "13. byte [f0-f0] 0 -> 7\n"
            "14. byte [f1-f3] 0 -> 7\n"
            "15. byte [f4-f4] 0 -> 7\n",
            reverse);
}

TEST(TestCompile, Bug35237384) {
  // Bug in the compiler caused inefficient bytecode to be generated for
  // nested nullable subexpressions.

  std::string forward;

  Dump("a**{3,}", Regexp::Latin1|Regexp::NeverCapture, &forward, NULL);
  EXPECT_EQ("3+ byte [61-61] 1 -> 3\n"
            "4. nop -> 5\n"
            "5+ byte [61-61] 1 -> 5\n"
            "6. nop -> 7\n"
            "7+ byte [61-61] 1 -> 7\n"
            "8. match! 0\n",
            forward);

  Dump("(a*|b*)*{3,}", Regexp::Latin1|Regexp::NeverCapture, &forward, NULL);
  EXPECT_EQ("3+ nop -> 6\n"
            "4+ nop -> 8\n"
            "5. nop -> 21\n"
            "6+ byte [61-61] 1 -> 6\n"
            "7. nop -> 3\n"
            "8+ byte [62-62] 1 -> 8\n"
            "9. nop -> 3\n"
            "10+ byte [61-61] 1 -> 10\n"
            "11. nop -> 21\n"
            "12+ byte [62-62] 1 -> 12\n"
            "13. nop -> 21\n"
            "14+ byte [61-61] 1 -> 14\n"
            "15. nop -> 18\n"
            "16+ byte [62-62] 1 -> 16\n"
            "17. nop -> 18\n"
            "18+ nop -> 14\n"
            "19+ nop -> 16\n"
            "20. match! 0\n"
            "21+ nop -> 10\n"
            "22+ nop -> 12\n"
            "23. nop -> 18\n",
      forward);

  Dump("((|S.+)+|(|S.+)+|){2}", Regexp::Latin1|Regexp::NeverCapture, &forward, NULL);
  EXPECT_EQ("3+ nop -> 36\n"
            "4+ nop -> 31\n"
            "5. nop -> 33\n"
            "6+ byte [00-09] 0 -> 8\n"
            "7. byte [0b-ff] 0 -> 8\n"
            "8+ nop -> 6\n"
            "9+ nop -> 29\n"
            "10. nop -> 28\n"
            "11+ byte [00-09] 0 -> 13\n"
            "12. byte [0b-ff] 0 -> 13\n"
            "13+ nop -> 11\n"
            "14+ nop -> 26\n"
            "15. nop -> 28\n"
            "16+ byte [00-09] 0 -> 18\n"
            "17. byte [0b-ff] 0 -> 18\n"
            "18+ nop -> 16\n"
            "19+ nop -> 36\n"
            "20. nop -> 33\n"
            "21+ byte [00-09] 0 -> 23\n"
            "22. byte [0b-ff] 0 -> 23\n"
            "23+ nop -> 21\n"
            "24+ nop -> 31\n"
            "25. nop -> 33\n"
            "26+ nop -> 28\n"
            "27. byte [53-53] 0 -> 11\n"
            "28. match! 0\n"
            "29+ nop -> 28\n"
            "30. byte [53-53] 0 -> 6\n"
            "31+ nop -> 33\n"
            "32. byte [53-53] 0 -> 21\n"
            "33+ nop -> 29\n"
            "34+ nop -> 26\n"
            "35. nop -> 28\n"
            "36+ nop -> 33\n"
            "37. byte [53-53] 0 -> 16\n",
      forward);
}

}  // namespace re2
